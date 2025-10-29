#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import numpy as np
import cvxpy as cp
from typing import Optional, Dict, Any
import os
import time
from ament_index_python.packages import get_package_share_directory
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

# DeePC Optimization class
class KernelDeePCOptimization:
    """
    Class for your kernel DeePC optimization.
    """

    def __init__(self, config: Dict[str, Any], logger = None):
        """
        Args:
            config: dictionary with user-defined configuration, e.g.:
                {
                  "m": int, # input dimension
                  "p": int, # output dimension
                  "T_ini": int, # length of past horizon
                  "N": int, # prediction horizon
                  "R": np.ndarray, # (mN x mN) input cost matrix
                  "Q": np.ndarray, # (pN x pN) output tracking cost matrix
                  "K": np.ndarray, # (Hc x Hc) kernel Gram matrix
                  "X": np.ndarray, # (Hx x Hc) Hankel data matrix (Up; Yp; Uf)
                  "Hy_future": np.ndarray, # (pN x Hc) future outputs Hankel matrix (Yf)
                  "gamma": float, # regularization for gram matrix
                  "rbf_scale": float, # RBF kernel scale
                  "lambda_g": float, # regularization weight for g
                  "lambda_k": float, # regularization weight for kernel matrix
        """
        self.cfg = config
        self.logger = logger

        # Extract dimensions
        self.m, self.p = self.cfg["m"], self.cfg["p"]

        # Horizons
        self.T_ini = self.cfg["T_ini"]
        self.N = self.cfg["N"]

        # Extract cost matrices
        self.R = sparse.csr_matrix(self.cfg["R"])
        self.Q = sparse.csr_matrix(self.cfg["Q"])

        # Extract Kernel and Hankel matrices
        self.K = self.cfg["K"]
        self.Hc = self.K.shape[0]
        self.gamma = self.cfg["gamma"]
        self.Kg = sparse.csr_matrix(self.K + self.gamma*np.eye(self.Hc))
        self.X = self.cfg["X"]
        self.Hy_future = sparse.csr_matrix(self.cfg["Hy_future"])
        self.rbf_scale = self.cfg["rbf_scale"]

        # Regularization weights
        self.lambda_g = self.cfg["lambda_g"]
        self.lambda_k = self.cfg["lambda_k"]

        # Decision variables
        self.u = cp.Variable(self.m * self.N)

        # Parameters that change online:
        self.q_param = cp.Parameter(self.m * self.N)

        # split X into the two parts used by eq. (36)
        d_ini = self.m*self.T_ini + self.p*self.T_ini
        self.X1 = self.X[:d_ini, :] # columns x1_i
        self.X2 = sparse.csr_matrix(self.X[d_ini:, :]) # columns x2_i
        self.X1_col_norm2 = np.sum(self.X1**2, axis=0) # precompute squared norms of columns of X1

        # -------- Matrices for QP without g --------
        # A = Hy_future^T Q Hy_future + lambda_g I + lambda_k Kg^T Kg
        A_dense = (self.Hy_future.T @ self.Q @ self.Hy_future + self.lambda_g * sparse.eye(self.Hc) +self.lambda_k * self.Kg.T @ self.Kg).toarray()
        self.cf = cho_factor(A_dense, lower=True, check_finite=False)
        
        E = self.lambda_k * self.Kg.T
        # Build M = lambda_k I - E^T A^{-1} E
        E_dense = E.toarray() # (Hc x Hc)
        AinvE = cho_solve(self.cf, E_dense, check_finite=False)  # solves A X = E  -> X = A^{-1}E
        M_dense = self.lambda_k * np.eye(self.Hc) - (E_dense.T @ AinvE)

        M_min_eig = np.linalg.eigvalsh(M_dense).min()
        self.logger.info(f"[DeePCOptimization] eigmin(M) = {M_min_eig:.3e}")

        # P = R + X2 M X2^T Hessian
        XM = (self.X2 @ M_dense)  # (mN x Hc)
        P_dense = self.R.toarray() + XM @ self.X2.T.toarray()  # (mN x mN)
        P_dense = (P_dense + P_dense.T) / 2.0  # ensure symmetry

        w = np.linalg.eigvalsh(P_dense)
        lam_min = w.min()
        self.logger.info(f"[DeePCOptimization] eigmin(P) before bump = {lam_min:.3e}")

        self.P = sparse.csc_matrix(P_dense)
        self.M_dense = M_dense

        self.E = E.tocsc()
        self.X2 = self.X2.tocsc()
        #---------------------------

        # Input constraints
        self.constraints = []
        u_max_per_joint = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
        u_max = np.tile(u_max_per_joint, self.N)
        u_min = -u_max
        self.constraints += [ self.u >= u_min, self.u <= u_max]

        # objective
        obj = 0.5 * cp.quad_form(self.u, self.P) + self.q_param @ self.u

        # Build problem
        self.problem = cp.Problem(cp.Minimize(obj), self.constraints)

        # Shapes sanity
        self.logger.info(f"shapes: X1={self.X1.shape}, X2={self.X2.shape}, HyF={self.Hy_future.shape}, Kg={self.Kg.shape}")

        # Conditioning of A
        condA = np.linalg.cond(A_dense)
        self.logger.info(f"[DeePCOptimization] cond(A) = {condA:.3e}")


    def build_k_rbf(self, u_ini: np.ndarray, y_ini: np.ndarray) -> np.ndarray:
        """DPP-safe RBF term: exp(-rbf_scale * ||x1 - [u_ini;y_ini]||^2) for all columns."""
        xy_ini = np.hstack([u_ini, y_ini]) # (d_ini,)
        term_xy = float(xy_ini @ xy_ini) # scalar
        term_ax = self.X1.T @ xy_ini # (Hc,)
        d2 = self.X1_col_norm2 + term_xy - 2.0*term_ax # (Hc,)
        return np.exp(-self.rbf_scale * d2).reshape(-1) # (Hc,)
    
    def A_solve(self, B):
        """Solve A X = B for B shape (Hc,) or (Hc, k). Returns ndarray."""
        B = np.atleast_2d(B)
        if B.shape[0] != self.Hc:  # accept (k, Hc) -> transpose
            B = B.T
        X = cho_solve(self.cf, B, check_finite=False)
        return X if X.ndim > 1 else X.reshape(-1,1)


    def update(self, u_ini: np.ndarray, y_ini: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Refresh Parameters from latest (u_ini, y_ini), rebuild objective and solve.

        Args:
            u_ini: array of shape (m*T_ini,)
            y_ini: array of shape (p*T_ini,)

        Returns:
            dict with keys like {"u_opt": ..., "g_opt": ..., "status": ...}
            or None if the problem failed.
        """

        k_rbf = self.build_k_rbf(u_ini=u_ini, y_ini=y_ini) # (Hc,)

        # q = X2 @ (M @ k_rbf)
        q = self.X2 @ (self.M_dense @ k_rbf) # (mN,)
        self.q_param.value = np.asarray(q).reshape(-1)

        start_time = time.perf_counter()

        # Solve
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            self.logger.error(f"[DeePCOptimization] Solve error: {e}")
            return None
        
        solve_time = (time.perf_counter() - start_time) * 1000  # ms
        self.logger.info(f"[DeePCOptimization] Solve time: {solve_time:.2f} ms")

        status = self.problem.status
        if status not in ("optimal", "optimal_inaccurate"):
            self.logger.warning(f"[DeePCOptimization] Problem status: {status}")
            return None

        out = {
            "status": status,
            "u_opt": np.array(self.u.value).reshape(-1),
        }

        u_opt = np.array(self.u.value).reshape(-1)
        self.logger.info(f"[DeePCOptimization] u_opt = {np.round(u_opt, 6)}")

        # if self.g is not None and self.g.value is not None:
        #     out["g_opt"] = np.array(self.g.value).reshape(-1)
        return out


# ROS2 optimization_node
class OptimizationNode(Node):
    """
    ROS 2 node that listens for u_ini and y_ini, runs the kernel DeePC optimization,
    and publishes the first control move.
    """

    def __init__(self):
        super().__init__("optimization_node")

        # Parameters
        self.declare_parameter("m", 7) # inputs
        self.declare_parameter("p", 7) # outputs 
        self.declare_parameter("T_ini", 40) # past horizon
        self.declare_parameter("N", 8) # prediction horizon
        self.declare_parameter("topic_init", "/deepc/init")  # combined message [u_ini ; y_ini]
        self.declare_parameter("publish_topic_u", "/deepc/u_opt")

        m = self.get_parameter("m").value
        p = self.get_parameter("p").value
        T_ini = self.get_parameter("T_ini").value
        N = self.get_parameter("N").value

        # TODO: Tune cost matrices
        R = np.eye(m * N) * 1e-1
        Q = np.eye(p * N) * 1e2

        # Load Kernel gram matrix and Hankel matrices
        share_dir = get_package_share_directory('nonlinear_deepc_controller')
        bundle_path = os.path.join(share_dir, 'data', 'kernel_deepc_bundle.npz')
        data = np.load(bundle_path)
        K = data["K"]
        X = data["X"]
        Hy_future = data["Hy_future"]
        gamma = float(data["gamma"])
        rbf_scale = float(data["rbf_scale"])

        # regularization lambdas
        lambda_g = 1e7
        lambda_k = 1e3


        cfg = dict(
            m=m, p=p, T_ini=T_ini, N=N,
            R=R, Q=Q,
            K = K, X = X, Hy_future = Hy_future,
            gamma = gamma,
            rbf_scale = rbf_scale,
            lambda_g = lambda_g,
            lambda_k = lambda_k,
        )
        self.optimizer = KernelDeePCOptimization(cfg, logger=self.get_logger())

        # Subscriber for combined init vector [u_ini ; y_ini]
        topic_init = self.get_parameter("topic_init").value
        self.sub_init = self.create_subscription(Float64MultiArray, topic_init, self.cb_init, 10)

        # Publisher for the first control move u0 (or full sequence if you prefer)
        self.pub_u = self.create_publisher(Float64MultiArray, self.get_parameter("publish_topic_u").value, 10)

        self.get_logger().info("optimization_node up. Waiting for /deepc/init (Float64MultiArray: [u_ini ; y_ini]).")

    def cb_init(self, msg: Float64MultiArray):
        """
        Callback function, expects data shaped as a flat vector [u_ini ; y_ini]
        where u_ini has length m*T_ini and y_ini has length p*T_ini.
        """
        m, p, T_ini, N = self.optimizer.m, self.optimizer.p, self.optimizer.cfg["T_ini"], self.optimizer.N
        arr = np.array(msg.data, dtype=float).reshape(-1)
        expected = m * T_ini + p * T_ini
        if arr.size != expected:
            self.get_logger().error(f"/deepc/init length {arr.size} != expected {expected}.")
            return

        u_ini = arr[: m * T_ini]
        y_ini = arr[m * T_ini : m * T_ini + p * T_ini]

        # Run the optimization
        result = self.optimizer.update(u_ini=u_ini, y_ini=y_ini)
        if result is None:
            self.get_logger().warn("Optimization failed or infeasible.")
            return

        u_opt = result["u_opt"]
        # Publish u0 (first m entries) so the low-level controller can apply it
        u0 = u_opt[:m]
        out = Float64MultiArray(data=u0.tolist())
        self.pub_u.publish(out)
        self.get_logger().info(f"Optimization status: {result['status']}. Published u0 with shape {u0.shape}.")


def main():
    rclpy.init()
    node = OptimizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
