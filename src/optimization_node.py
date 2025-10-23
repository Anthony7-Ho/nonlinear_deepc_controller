#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import numpy as np
import cvxpy as cp
from typing import Optional, Dict, Any
import os

# DeePC Optimization class
class KernelDeePCOptimization:
    """
    Class for your kernel DeePC optimization.
    """

    def __init__(self, config: Dict[str, Any]):
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

        # Extract dimensions
        self.m, self.p = self.cfg["m"], self.cfg["p"]

        # Horizons
        self.T_ini = self.cfg["T_ini"]
        self.N = self.cfg["N"]

        # Extract cost matrices
        self.R = self.cfg["R"]
        self.Q = self.cfg["Q"]

        # Extract Kernel and Hankel matrices
        self.K = self.cfg["K"]
        self.Hc = self.K.shape[0]
        self.X = self.cfg["X"]
        self.Hy_future = self.cfg["Hy_future"]
        self.gamma = self.cfg["gamma"]
        self.rbf_scale = self.cfg["rbf_scale"]

        # Regularization weights
        self.lambda_g = self.cfg["lambda_g"]
        self.lambda_k = self.cfg["lambda_k"]

        # Decision variables
        self.u = cp.Variable(self.m * self.N)
        self.g = cp.Variable(self.Hc)  # size Hc

        # Parameters that change online:
        self.u_ini = cp.Parameter(self.m * self.T_ini)
        self.y_ini = cp.Parameter(self.p * self.T_ini)

        # Input constraints
        self.constraints = []
        u_max_per_joint = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
        u_max = np.tile(u_max_per_joint, self.N)
        u_min = -u_max
        self.constraints += [ self.u >= u_min, self.u <= u_max]

        # Cost
        # split X into the two parts used by eq. (36)
        d_ini = self.m*self.T_ini + self.p*self.T_ini
        d_u = self.m*self.N
        self.X1 = self.X[:, :d_ini] # rows x1_i
        self.X2 = self.X[:, d_ini:] # rows x2_i
        self.X1_row_norm2 = np.sum(self.X1**2, axis=1) # precompute squared norms of rows of X1

        # build k(u_ini, y_ini, u) for the mixed kernel
        self.k_vec = self.build_k_vector()
        Kernel_cost = (self.K + self.gamma*np.eye(self.Hc)) @ self.g - self.k_vec

        cost = (
            cp.quad_form(self.Hy_future * self.g, self.Q) +
            cp.quad_form(self.u, self.R) +
            self.lambda_g * cp.sum_squares(self.g) +
            self.lambda_k * cp.sum_squares(Kernel_cost)
        )

        # Build problem
        self.problem = cp.Problem(cp.Minimize(cost), self.constraints)


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
        # Update parameters
        self.u_ini.value = u_ini
        self.y_ini.value = y_ini

        # Solve
        try:
            self.problem.solve(solver=cp.MOSEK, warm_start=True, verbose=False)
        except Exception as e:
            print(f"[DeePCOptimization] Solve error: {e}")
            return None

        status = self.problem.status
        if status not in ("optimal", "optimal_inaccurate"):
            print(f"[DeePCOptimization] Problem status: {status}")
            return None

        out = {
            "status": status,
            "u_opt": np.array(self.u.value).reshape(-1),
        }
        if self.g is not None and self.g.value is not None:
            out["g_opt"] = np.array(self.g.value).reshape(-1)
        return out
    
    def build_k_vector(self):
        """
        Returns a CVXPY expression of shape (Hc,) implementing
        k(u_ini, y_ini, u) from eq. (36):
            k = exp( - ||x1_i - [u_ini;y_ini]||^2 / (2*sigma^2) ) + x2_i^T u
        The first term depends only on parameters (u_ini, y_ini);
        the second is affine in the decision variable u.
        """
        # concatenate parameters col(u_ini; y_ini)
        xy_ini = cp.hstack([self.u_ini, self.y_ini]) # shape (d_ini,)

        # compute squared distances 
        term_xy = cp.sum_squares(xy_ini) * np.ones(self.Hc)    # broadcast scalar
        term_ax = self.X1 @ xy_ini                         # shape (Hc,)
        d2 = self.X1_row_norm2 + term_xy - 2.0*term_ax     # elementwise

        # RBF part
        rbf = cp.exp(-d2 / (2.0 * (self.rbf_scale**2))) # shape (Hc,)

        # linear part
        lin_u = self.X2 @ self.u # shape (Hc,)

        # mixed kernel vector
        return rbf + lin_u


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
        R = np.eye(m * N) * 1e-3
        Q = np.eye(p * N) * 1.0

        # Load Kernel gram matrix and Hankel matrices
        base_dir = os.path.dirname(__file__)
        bundle_path = os.path.join(base_dir, "../data_processing/kernel_deepc_bundle.npz")
        data = np.load(bundle_path)
        K = data["K"]
        X_sub = data["X_sub"]
        Hy_future_sub = data["Hy_future_sub"]
        best_lambda = float(data["best_lambda"])
        best_gamma_rbf = float(data["best_gamma_rbf"])

        # regularization lambdas
        lambda_g = 1e3
        lambda_k = 1e4


        cfg = dict(
            m=m, p=p, T_ini=T_ini, N=N,
            R=R, Q=Q,
            K = K, X = X_sub, Hy_future = Hy_future_sub,
            gamma = best_lambda,
            rbf_scale = best_gamma_rbf,
            lambda_g = lambda_g,
            lambda_k = lambda_k,
        )
        self.optimizer = KernelDeePCOptimization(cfg)

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
