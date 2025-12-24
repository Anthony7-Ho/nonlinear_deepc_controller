#pragma once

#include <deque>
#include <Eigen/Dense>

namespace nonlinear_deepc_controller {

class HistoryBuffer {
/**
 * Manages histories of applied torques (u) and measured external torques (y)
 * for the Kernel Cartesian Impedance Controller.
 */
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  explicit HistoryBuffer(int T_ini = 20) : T_ini_(T_ini) {}

  void setHorizon(int T_ini) { T_ini_ = T_ini; }

  void reset() {
    u_hist_.clear();
    y_hist_.clear();
    first_update_ = true;
  }

  // warmup step: adds zero input and measured tau_ext
  bool warmupStep(const Vector7d& tau_ext) {
    if (static_cast<int>(u_hist_.size()) >= T_ini_) u_hist_.pop_front();
    if (static_cast<int>(y_hist_.size()) >= T_ini_) y_hist_.pop_front();
    u_hist_.push_back(Vector7d::Zero());
    y_hist_.push_back(tau_ext);
    return (static_cast<int>(u_hist_.size()) >= T_ini_) &&
           (static_cast<int>(y_hist_.size()) >= T_ini_);
  }

  void pushU(const Vector7d& u_curr) {
    if (static_cast<int>(u_hist_.size()) >= T_ini_) {
      u_hist_.pop_front();
    }
    u_hist_.push_back(u_curr);
  }

  void pushY(const Vector7d& y_next) {
    if (static_cast<int>(y_hist_.size()) >= T_ini_) {
      y_hist_.pop_front();
    }
    y_hist_.push_back(y_next);
  }

  bool firstUpdate() const { return first_update_; }
  void markUpdated() { first_update_ = false; }

  const std::deque<Vector7d>& u() const { return u_hist_; }
  const std::deque<Vector7d>& y() const { return y_hist_; }

 private:
  int T_ini_{20};
  std::deque<Vector7d> u_hist_;
  std::deque<Vector7d> y_hist_;
  bool first_update_{true};
};

}  // namespace nonlinear_deepc_controller
