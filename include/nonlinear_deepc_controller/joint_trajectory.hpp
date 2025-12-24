#pragma once

#include <string>
#include <vector>
#include <Eigen/Eigen>

namespace nonlinear_deepc_controller {

/**
 * Joint trajectory loaded from CSV file.
 * CSV file should have columns:
 * time_s, q_0..q_6, dq_0..dq_6
 */

class JointTrajectory {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  // load CSV with columns:
  bool loadFromCsv(const std::string& path);

  // interpolate q/dq at time t
  Vector7d interpQ(double t) const;
  Vector7d interpDq(double t) const;

  bool empty() const { return t_grid_.empty(); }
  const std::vector<double>& tGrid() const { return t_grid_; }
  double tEnd() const { return t_grid_.empty() ? 0.0 : t_grid_.back(); }
  bool hasDq() const { return !dq_traj_.empty(); }

 private:
  Vector7d interp(const std::vector<Vector7d>& data, double t) const;

  std::vector<double> t_grid_;
  std::vector<Vector7d> q_traj_;
  std::vector<Vector7d> dq_traj_;
};

}  // namespace nonlinear_deepc_controller
