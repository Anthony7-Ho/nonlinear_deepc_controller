// src/friction_predictor.cpp
#include "nonlinear_deepc_controller/friction_predictor.hpp"
#include <cmath>

namespace nonlinear_deepc_controller {

FrictionPredictor::Vector7d
FrictionPredictor::predict(const std::deque<Vector7d>& u_hist, const std::deque<Vector7d>& y_hist, const Vector7d& last_tau_imp) const {
  Vector7d result = Vector7d::Zero();

  if (!bundle_ || !bundle_->loaded()) return result;
  if (T_ini_ <= 0) return result;

  if (static_cast<int>(u_hist.size()) < T_ini_ || static_cast<int>(y_hist.size()) < T_ini_) {
    return result;
  }

  const int Hc = bundle_->Hc();
  const int dfull = bundle_->d_full();
  const int Npred = bundle_->N_pred();

  (void)dfull;

  for (int joint = 0; joint < KernelBundle::kNumJoints; ++joint) {
    Eigen::VectorXd u_ini(T_ini_);
    Eigen::VectorXd y_ini(T_ini_);

    // Copy histories in order (oldest -> newest)
    for (int k = 0; k < T_ini_; ++k) {
      u_ini(k) = static_cast<double>(u_hist[static_cast<size_t>(k)](joint));
      y_ini(k) = static_cast<double>(y_hist[static_cast<size_t>(k)](joint));
    }

    // Future input: tile last impedance torque
    Eigen::VectorXd u_future(Npred);
    const double u_val = static_cast<double>(last_tau_imp(joint));
    u_future.setConstant(u_val);

    // x_ini = [u_ini; y_ini; u_future]
    Eigen::VectorXd x_ini(2 * T_ini_ + Npred);
    x_ini.segment(0, T_ini_) = u_ini;
    x_ini.segment(T_ini_, T_ini_) = y_ini;
    x_ini.segment(2*T_ini_, Npred) = u_future;

    // RBF distances:
    // ||x_i - x_ini||^2 = ||x_i||^2 + ||x_ini||^2 - 2 x_i^T x_ini
    const double term_xy = x_ini.squaredNorm();
    const Eigen::VectorXd term_ax = bundle_->X(joint).transpose() * x_ini;
    Eigen::VectorXd d2 = bundle_->X_col_norm2(joint) + term_xy * Eigen::VectorXd::Ones(Hc) - 2.0 * term_ax;

    Eigen::VectorXd k_vec(Hc);
    const double gamma = bundle_->rbf_scale();
    for (int i = 0; i < Hc; ++i) {
      k_vec(i) = std::exp(-gamma * d2(i));
    }

    // rhs = lambda_k * Kg^T * k
    const Eigen::VectorXd rhs = bundle_->lambda_k() * (bundle_->Kg(joint).transpose() * k_vec);

    // g = A^{-1} rhs
    const Eigen::VectorXd g = bundle_->cholA(joint).solve(rhs);

    // y_future = Hy_future * g
    const Eigen::VectorXd y_future = bundle_->Hy_future(joint) * g;

    // First predicted step
    result(joint) = y_future(0);
  }

  return result;
}

}  // namespace nonlinear_deepc_controller
