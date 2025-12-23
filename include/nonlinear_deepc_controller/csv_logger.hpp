// include/nonlinear_deepc_controller/csv_logger.hpp
#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <Eigen/Dense>

namespace nonlinear_deepc_controller {

/**
 * Minimal CSV logger for:
 *  time_s,
 *  tau_ext_0..6,
 *  friction_pred_0..6,
 *  tau_residual_0..6,
 *  x,y,z
 *
 * Push samples at control rate, decimate internally, and write on deactivate.
 */
class CsvLogger {
public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using Vector3d = Eigen::Matrix<double, 3, 1>;

  void configure(int decimation, bool stop_at_end, double post_window) {
    decimation_ = std::max(1, decimation);
    stop_at_end_ = stop_at_end;
    post_window_ = std::max(0.0, post_window);
  }

  void reset() {
    log_counter_ = 0;
    active_ = true;
    t_.clear();
    tau_ext_.clear();
    friction_pred_.clear();
    tau_resid_.clear();
    ee_pos_.clear();
  }

  // Gate logging based on elapsed time
  void updateActiveWindow(double elapsed_time, double t_end) {
    if (stop_at_end_ && elapsed_time > t_end + post_window_) {
      active_ = false;
    }
  }

  bool active() const { return active_; }

  // Push one sample
  void push(double time_s, const Vector7d& tau_ext, const Vector7d& friction_pred, const Vector3d& ee_pos) {
    if (!active_) return;

    if (++log_counter_ < decimation_) return;
    log_counter_ = 0;

    t_.push_back(time_s);
    tau_ext_.push_back(tau_ext);
    friction_pred_.push_back(friction_pred);
    tau_resid_.push_back(tau_ext - friction_pred);
    ee_pos_.push_back(ee_pos);
  }

  void write(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("CsvLogger: cannot open log file: " + path);

    const size_t N = std::min({t_.size(), tau_ext_.size(), friction_pred_.size(), tau_resid_.size(), ee_pos_.size()});

    // Header
    out << "time_s";
    for (int j = 0; j < 7; ++j) out << ",tau_ext_" << j;
    for (int j = 0; j < 7; ++j) out << ",friction_pred_" << j;
    for (int j = 0; j < 7; ++j) out << ",tau_residual_" << j;
    out << ",x,y,z\n";

    out << std::fixed << std::setprecision(6);

    for (size_t i = 0; i < N; ++i) {
      out << t_[i];
      for (int j = 0; j < 7; ++j) out << "," << tau_ext_[i](j);
      for (int j = 0; j < 7; ++j) out << "," << friction_pred_[i](j);
      for (int j = 0; j < 7; ++j) out << "," << tau_resid_[i](j);
      
      // ee_pos columns
      out << "," << ee_pos_[i](0)
          << "," << ee_pos_[i](1)
          << "," << ee_pos_[i](2);

      out << "\n";
    }
  }

private:
  int decimation_{1};
  int log_counter_{0};
  bool stop_at_end_{true};
  double post_window_{0.0};
  bool active_{true};

  std::vector<double> t_;
  std::vector<Vector7d> tau_ext_;
  std::vector<Vector7d> friction_pred_;
  std::vector<Vector7d> tau_resid_;
  std::vector<Vector3d> ee_pos_;
};

}  // namespace nonlinear_deepc_controller
