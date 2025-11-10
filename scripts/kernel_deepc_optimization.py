#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn as TCR

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
        self.u_ref = cp.Parameter(self.m * self.N)

        # TODO: Add reference to output? Maybe regulating  it to 0 causes high u?

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
        # self.logger.info(f"[DeePCOptimization] eigmin(M) = {M_min_eig:.3e}")

        # P = R + X2 M X2^T Hessian
        XM = (self.X2 @ M_dense)  # (mN x Hc)
        P_dense = self.R.toarray() + XM @ self.X2.T.toarray()  # (mN x mN)
        P_dense = (P_dense + P_dense.T) / 2.0  # ensure symmetry

        w = np.linalg.eigvalsh(P_dense)
        lam_min = w.min()
        # self.logger.info(f"[DeePCOptimization] eigmin(P) before bump = {lam_min:.3e}")

        self.P = sparse.csc_matrix(P_dense)
        self.M_dense = M_dense

        self.E = E.tocsc()
        self.X2 = self.X2.tocsc()
        #---------------------------

        # Input constraints
        self.constraints = []
        u_max_per_joint = np.array([1, 1, 1, 1, 1, 1, 1], dtype=float) # TODO: tune?
        u_max = np.tile(u_max_per_joint, self.N)
        u_min = -u_max
        self.constraints += [ self.u >= u_min, self.u <= u_max]

        # Trust region on input # TODO: tune or make delta adaptive? (30% of u_ref))
        self.delta = 0.3
        for j in range(self.m):
            sl = slice(j, self.m*self.N, self.m)
            self.constraints += [
                self.u[sl] <= self.u_ref[sl] + self.delta,
                self.u[sl] >= self.u_ref[sl] - self.delta,
            ]

        # objective
        obj = 0.5 * cp.quad_form(self.u - self.u_ref, self.P) + self.q_param @ self.u

        # Build problem
        self.problem = cp.Problem(cp.Minimize(obj), self.constraints)

        # Shapes sanity
        # self.logger.info(f"shapes: X1={self.X1.shape}, X2={self.X2.shape}, HyF={self.Hy_future.shape}, Kg={self.Kg.shape}")

        # Conditioning of A
        condA = np.linalg.cond(A_dense)
        # self.logger.info(f"[DeePCOptimization] cond(A) = {condA:.3e}")


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


    def update(self, u_ini: np.ndarray, y_ini: np.ndarray, u_ref: Optional[np.ndarray] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Refresh Parameters from latest (u_ini, y_ini), rebuild objective and solve.

        Args:
            u_ini: array of shape (m*T_ini,)
            y_ini: array of shape (p*T_ini,)
            u_ref: reference input trajectory of shape (m*N,)

        Returns:
            dict with keys like {"u_opt": ..., "g_opt": ..., "status": ...}
            or None if the problem failed.
        """

        k_rbf = self.build_k_rbf(u_ini=u_ini, y_ini=y_ini) # (Hc,)

        # q = X2 @ (M @ k_rbf)
        q = self.X2 @ (self.M_dense @ k_rbf) # (mN,)
        self.q_param.value = np.asarray(q).reshape(-1)

        if u_ref is None:
            self.u_ref.value = np.zeros(self.m * self.N)
        else:
            self.u_ref.value = u_ref.reshape(-1)

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


# ROS2 optimization Node
class OptimizationNode(LifecycleNode):
    """
    Lifecycle version:
      - Heavy init (loading data, building optimizer) in on_configure()
      - Start I/O (pub/sub) in on_activate()
      - Clean up in on_deactivate()/on_cleanup()/on_shutdown()
    """

    def __init__(self):
        super().__init__("optimization_node")

        # Parameters
        self.declare_parameter("m", 7) # inputs
        self.declare_parameter("p", 7) # outputs 
        self.declare_parameter("T_ini", 40) # past horizon
        self.declare_parameter("N", 8) # prediction horizon
        self.declare_parameter("topic_init", "/deepc/init")  # combined message [u_ini ; y_ini; u_ref]
        self.declare_parameter("publish_topic_u", "/deepc/u_opt")

        # Placeholders created in states
        self.optimizer: Optional[KernelDeePCOptimization] = None
        self.sub_init = None
        self.pub_u = None

        self._qos = QoSProfile(depth = 10)

    # --- Lifecycle state handlers ---

    def on_configure(self, state) -> TCR:
        """
        Build all heavy objects here. If this returns SUCCESS,
        launch will then request ACTIVATE.
        """
        try:
            # Read params
            m = int(self.get_parameter("m").value)
            p = int(self.get_parameter("p").value)
            T_ini = int(self.get_parameter("T_ini").value)
            N = int(self.get_parameter("N").value)

            topic_init = self.get_parameter("topic_init").value
            publish_topic_u = self.get_parameter("publish_topic_u").value

            self._topic_init = topic_init
            self._publish_topic_u = publish_topic_u

            # Cost matrices TODO: tune
            R = np.eye(m * N) * 1e1
            Q = np.eye(p * N) * 1e1

            # Load data from share
            share_dir = get_package_share_directory("nonlinear_deepc_controller")
            bundle_path = os.path.join(share_dir, "data", "kernel_deepc_bundle.npz")
            data = np.load(bundle_path)
            K = data["K"]
            X = data["X"]
            Hy_future = data["Hy_future"]
            gamma = float(data["gamma"])
            rbf_scale = float(data["rbf_scale"])

            # TODO: tune
            lambda_g = 1e5
            lambda_k = 1e1

            cfg = dict(
                m=m, p=p, T_ini=T_ini, N=N,
                R=R, Q=Q,
                K=K, X=X, Hy_future=Hy_future,
                gamma=gamma,
                rbf_scale=rbf_scale,
                lambda_g=lambda_g,
                lambda_k=lambda_k,
            )
            # Heavy construction here
            self.optimizer = KernelDeePCOptimization(cfg, logger=self.get_logger())

            self.get_logger().info("Configured optimizer (INACTIVE).")
            return TCR.SUCCESS

        except Exception as e:
            self.get_logger().error(f"on_configure() failed: {e}")
            return TCR.FAILURE

    def on_activate(self, state) -> TCR:
        """
        Create publishers/subscribers/timers here so the node only starts I/O when ACTIVE.
        """
        try:
            # Publisher for u_opt
            self.pub_u = self.create_publisher(
                Float64MultiArray, self._publish_topic_u, self._qos
            )
            # Subscriber for [u_ini; y_ini]
            self.sub_init = self.create_subscription(
                Float64MultiArray, self._topic_init, self.cb_init, self._qos
            )

            self.get_logger().info("optimization Node is ACTIVE. Waiting for init data")
            return TCR.SUCCESS

        except Exception as e:
            self.get_logger().error(f"on_activate() failed: {e}")
            return TCR.FAILURE

    def on_deactivate(self, state) -> TCR:
        """
        Stop I/O but keep configuration/optimizer in memory.
        """
        try:
            # Destroy I/O entities 
            if self.sub_init is not None:
                self.destroy_subscription(self.sub_init)
                self.sub_init = None
            if self.pub_u is not None:
                self.destroy_publisher(self.pub_u)
                self.pub_u = None

            self.get_logger().info("optimization Node DEACTIVATED.")
            return TCR.SUCCESS
        except Exception as e:
            self.get_logger().error(f"on_deactivate() failed: {e}")
            return TCR.FAILURE

    def on_cleanup(self, state) -> TCR:
        """
        Node goes back to UNCONFIGURED.
        """
        try:
            self.optimizer = None
            self.get_logger().info("optimization Node CLEANED UP.")
            return TCR.SUCCESS
        except Exception as e:
            self.get_logger().error(f"on_cleanup() failed: {e}")
            return TCR.FAILURE

    def on_shutdown(self, state) -> TCR:
        self.get_logger().info("optimization Node SHUTDOWN.")
        return TCR.SUCCESS

    # --- Subscriber callback -------

    def cb_init(self, msg: Float64MultiArray):
        if self.optimizer is None:
            self.get_logger().warn("Received init but optimizer not ready.")
            return

        m = self.optimizer.m
        p = self.optimizer.p
        T_ini = self.optimizer.cfg["T_ini"]
        N = self.optimizer.cfg["N"]

        arr = np.array(msg.data, dtype=float).reshape(-1)
        len_uini = m * T_ini
        len_yini = p * T_ini
        len_uref = m * N
        expected = len_uini + len_yini + len_uref
        if arr.size != expected:
            self.get_logger().error(f"/deepc/init length {arr.size} != expected {expected}.")
            return

        u_ini = arr[:len_uini]
        y_ini = arr[len_uini:len_uini + len_yini]
        u_ref = arr[len_uini + len_yini:]

        if u_ref.size == m:
            u_ref = np.tile(u_ref, self.optimizer.N)

        result = self.optimizer.update(u_ini=u_ini, y_ini=y_ini, u_ref=u_ref)
        if result is None:
            self.get_logger().warn("Optimization failed or infeasible.")
            return

        u_opt = result["u_opt"]
        u0 = u_opt[:m]
        out = Float64MultiArray(data=u0.tolist())
        if self.pub_u is not None:
            self.pub_u.publish(out)
        self.get_logger().info(
            f"Optimization status: {result['status']}. Published u0 with shape {u0.shape}."
        )


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
