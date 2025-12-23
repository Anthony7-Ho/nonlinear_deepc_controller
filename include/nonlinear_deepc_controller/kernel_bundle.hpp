// include/nonlinear_deepc_controller/kernel_bundle.hpp
#pragma once

#include <array>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <rclcpp/logger.hpp>

namespace nonlinear_deepc_controller {

/**
 * Holds the per-joint kernel DeePC bundle and all precomputations needed
 * for fast closed-form prediction.
 *
 * Expected directory structure:
 *   <dir>/joint_0/meta.json
 *   <dir>/joint_0/Kg.bin
 *   <dir>/joint_0/X.bin
 *   <dir>/joint_0/Hy_future.bin
 *   ...
 *   <dir>/joint_6/...
 */
class KernelBundle {
public:
  static constexpr int kNumJoints = 7;

  bool load(const std::string& dir, const rclcpp::Logger& logger);

  bool loaded() const { return loaded_; }

  // Shared dimensions/hyperparams
  int Hc() const { return Hc_; } // number of columns
  int d_full() const { return d_full_; } // regressor length
  int N_pred() const { return N_pred_; } // future horizon length (rows of Hy_future)

  double rbf_scale() const { return rbf_scale_; }
  double lambda_g() const { return lambda_g_; }
  double lambda_k() const { return lambda_k_; }

  // Per-joint accessors
  const Eigen::MatrixXd& Kg(int j) const { return Kg_list_[j]; }
  const Eigen::MatrixXd& X(int j) const { return X_list_[j]; }
  const Eigen::MatrixXd& Hy_future(int j) const { return Hy_future_list_[j]; }
  const Eigen::VectorXd& X_col_norm2(int j) const { return X_col_norm2_list_[j]; }
  const Eigen::LLT<Eigen::MatrixXd>& cholA(int j) const { return A_chol_list_[j]; }

private:
  bool loadBinMatrix(const std::string& path, int rows, int cols, Eigen::MatrixXd& M, const rclcpp::Logger& logger);

  bool loaded_{false};

  int Hc_{0};
  int d_full_{0};
  int N_pred_{0};

  double rbf_scale_{0.0};
  double lambda_g_{0.0};
  double lambda_k_{0.0};

  std::array<Eigen::MatrixXd, kNumJoints> Kg_list_;
  std::array<Eigen::MatrixXd, kNumJoints> X_list_;
  std::array<Eigen::MatrixXd, kNumJoints> Hy_future_list_;

  std::array<Eigen::VectorXd, kNumJoints> X_col_norm2_list_;
  std::array<Eigen::LLT<Eigen::MatrixXd>, kNumJoints> A_chol_list_;
};

}  // namespace nonlinear_deepc_controller
