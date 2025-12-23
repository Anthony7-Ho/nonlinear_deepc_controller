// include/nonlinear_deepc_controller/friction_predictor.hpp
#pragma once

#include <deque>
#include <Eigen/Dense>
#include "nonlinear_deepc_controller/kernel_bundle.hpp"

namespace nonlinear_deepc_controller {

/**
 * Closed-form kernel DeePC predictor for friction torque.
 *
 * For each joint j:
 *  x_ini = [u_ini; y_ini; u_future]
 *  k_i = exp(-rbf_scale * ||x_i - x_ini||^2)
 *  g = A^{-1} (lambda_k * Kg^T k)
 *  y_future = Hy_future * g
 *  prediction = y_future(0)
 */
class FrictionPredictor {
public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  void configure(int T_ini) { T_ini_ = T_ini; }
  void setBundle(const KernelBundle* bundle) { bundle_ = bundle; }

  Vector7d predict(const std::deque<Vector7d>& u_hist, const std::deque<Vector7d>& y_hist,const Vector7d& last_tau_imp) const;

private:
  int T_ini_{0};
  const KernelBundle* bundle_{nullptr};
};

}  // namespace nonlinear_deepc_controller
