// src/kernel_bundle.cpp
#include "nonlinear_deepc_controller/kernel_bundle.hpp"

#include <fstream>
#include <vector>
#include <cmath>

#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>

namespace nonlinear_deepc_controller {

using json = nlohmann::json;

bool KernelBundle::loadBinMatrix(const std::string& path, int rows, int cols, Eigen::MatrixXd& M, const rclcpp::Logger& logger) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    RCLCPP_ERROR(logger, "KernelBundle: cannot open %s", path.c_str());
    return false;
  }

  std::vector<double> buf(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(double)));

  if (!f.good()) {
    RCLCPP_ERROR(logger, "KernelBundle: error reading %s", path.c_str());
    return false;
  }

  M.resize(rows, cols);
  // Stored row-major in Python
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      M(r, c) = buf[static_cast<size_t>(r * cols + c)];
    }
  }
  return true;
}

bool KernelBundle::load(const std::string& dir, const rclcpp::Logger& logger) {
  RCLCPP_INFO(logger, "KernelBundle: loading all joints from %s", dir.c_str());

  bool first_joint = true;

  for (int j = 0; j < kNumJoints; ++j) {
    const std::string joint_dir = dir + "/joint_" + std::to_string(j);
    const std::string meta_path = joint_dir + "/meta.json";
    const std::string Kg_path = joint_dir + "/Kg.bin";
    const std::string X_path = joint_dir + "/X.bin";
    const std::string Hy_path = joint_dir + "/Hy_future.bin";

    std::ifstream mf(meta_path);
    if (!mf.is_open()) {
      RCLCPP_ERROR(logger, "KernelBundle: cannot open %s (joint %d)", meta_path.c_str(), j);
      return false;
    }

    json meta;
    try {
      mf >> meta;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(logger, "KernelBundle: error parsing %s: %s", meta_path.c_str(), e.what());
      return false;
    }

    const int Kg_rows = meta["Kg"]["rows"];
    const int Kg_cols = meta["Kg"]["cols"];
    const int X_rows = meta["X"]["rows"];
    const int X_cols = meta["X"]["cols"];
    const int Hy_rows = meta["Hy_future"]["rows"];
    const int Hy_cols = meta["Hy_future"]["cols"];

    const double rbf_scale_meta = meta["rbf_scale"];
    const double lambda_g_meta = meta["lambda_g"];
    const double lambda_k_meta = meta["lambda_k"];
    const int T_past_meta = meta["T_past"];
    const int T_future_meta = meta["T_future"];

    if (first_joint) {
      Hc_ = Kg_cols;
      d_full_ = X_rows;
      N_pred_ = Hy_rows;

      rbf_scale_ = rbf_scale_meta;
      lambda_g_ = lambda_g_meta;
      lambda_k_ = lambda_k_meta;

      // sanity check: d_full = 2*T_past + T_future
      if (d_full_ != 2 * T_past_meta + T_future_meta) {
        RCLCPP_WARN(logger,
                    "KernelBundle: d_full (%d) != 2*T_past(%d)+T_future(%d) from meta.json",
                    d_full_, T_past_meta, T_future_meta);
      }
    } else {
      if (Kg_cols != Hc_ || X_rows != d_full_ || Hy_rows != N_pred_) {
        RCLCPP_ERROR(logger, "KernelBundle: dimension mismatch in joint %d", j);
        return false;
      }
      // Hyperparams: allow slight mismatch but warn
      if (std::fabs(rbf_scale_meta - rbf_scale_) > 1e-7 ||
          std::fabs(lambda_g_meta - lambda_g_)  > 1e-5 ||
          std::fabs(lambda_k_meta - lambda_k_)  > 1e-2) {
        RCLCPP_WARN(logger, "KernelBundle: hyperparams differ in joint %d; using joint_0 values", j);
      }
    }

    // Load matrices
    if (!loadBinMatrix(Kg_path, Kg_rows, Kg_cols, Kg_list_[j], logger)) return false;
    if (!loadBinMatrix(X_path, X_rows, X_cols, X_list_[j], logger)) return false;
    if (!loadBinMatrix(Hy_path, Hy_rows, Hy_cols, Hy_future_list_[j], logger)) return false;

    // Precompute column norms ||x_i||^2 for each column of X (X is d_full x Hc)
    X_col_norm2_list_[j].resize(Hc_);
    for (int c = 0; c < Hc_; ++c) {
      X_col_norm2_list_[j](c) = X_list_[j].col(c).squaredNorm();
    }

    // Build and factorize A_j = lambda_g I + lambda_k * Kg^T Kg  (size Hc x Hc)
    Eigen::MatrixXd A_j = lambda_g_ * Eigen::MatrixXd::Identity(Hc_, Hc_) + lambda_k_ * (Kg_list_[j].transpose() * Kg_list_[j]);


    // Debug: min eigenvalue (A_j should be SPD; min_eig should be > 0)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A_j);
    if (es.info() == Eigen::Success) {
      const double min_eig = es.eigenvalues().minCoeff();

      std::cout << "[KernelBundle] joint " << j
                << " min eigenvalue = "
                << std::setprecision(17) << min_eig
                << std::endl;
     } else {
      RCLCPP_WARN(logger, "KernelBundle: eigenvalue computation failed for joint %d", j);
     }

    A_chol_list_[j].compute(A_j);
    if (A_chol_list_[j].info() != Eigen::Success) {
      RCLCPP_ERROR(logger, "KernelBundle: Cholesky factorization failed for joint %d", j);
      return false;
    }

    first_joint = false;
  }

  loaded_ = true;
  RCLCPP_INFO(logger, "KernelBundle: loaded OK. Hc=%d, d_full=%d, N_pred=%d", Hc_, d_full_, N_pred_);
  return true;
}

}  // namespace nonlinear_deepc_controller
