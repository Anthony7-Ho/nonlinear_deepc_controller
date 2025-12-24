#include "nonlinear_deepc_controller/joint_trajectory.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>

namespace nonlinear_deepc_controller {

bool JointTrajectory::loadFromCsv(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) return false;

  std::string line;
  if (!std::getline(f, line)) return false;

  // Parse header
  std::vector<std::string> cols;
  {
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) cols.push_back(c);
  }

  auto col_index = [&](const std::string& name)->int{
    for (size_t i = 0; i < cols.size(); ++i) {
      if (cols[i] == name) return static_cast<int>(i);
    }
    return -1;
  };

  int t_idx = col_index("time_s");
  if (t_idx < 0) t_idx = col_index("t");

  int q_idx[7];
  for (int i = 0; i < 7; ++i) q_idx[i] = col_index("q_" + std::to_string(i));

  int dq_idx[7];
  bool have_dq = true;
  for (int i = 0; i < 7; ++i) {
    dq_idx[i] = col_index("dq_" + std::to_string(i));
    if (dq_idx[i] < 0) have_dq = false;
  }

  if (t_idx < 0) return false;
  for (int i = 0; i < 7; ++i) if (q_idx[i] < 0) return false;

  t_grid_.clear();
  q_traj_.clear();
  dq_traj_.clear();

  while (std::getline(f, line)) {
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string tok;
    std::vector<double> vals;

    while (std::getline(ss, tok, ',')) {
      if (tok.empty() || tok == " ") { vals.push_back(0.0); continue; }
      vals.push_back(std::stod(tok));
    }

    if (static_cast<int>(vals.size()) <= t_idx) continue;

    double t = vals[t_idx];

    Vector7d q;
    for (int i = 0; i < 7; ++i) q(i) = vals[q_idx[i]];

    t_grid_.push_back(t);
    q_traj_.push_back(q);

    if (have_dq) {
      Vector7d dq;
      for (int i = 0; i < 7; ++i) dq(i) = vals[dq_idx[i]];
      dq_traj_.push_back(dq);
    }
  }

  return !t_grid_.empty();
}

JointTrajectory::Vector7d
JointTrajectory::interp(const std::vector<Vector7d>& data, double t) const {
  if (t <= t_grid_.front()) return data.front();
  if (t >= t_grid_.back())  return data.back();

  auto it = std::upper_bound(t_grid_.begin(), t_grid_.end(), t);
  size_t k = static_cast<size_t>(std::distance(t_grid_.begin(), it) - 1);
  double t0 = t_grid_[k], t1 = t_grid_[k + 1];
  double a = (t - t0) / (t1 - t0);
  return (1.0 - a) * data[k] + a * data[k + 1];
}

JointTrajectory::Vector7d JointTrajectory::interpQ(double t) const {
  if (t_grid_.empty()) return Vector7d::Zero();
  return interp(q_traj_, t);
}

JointTrajectory::Vector7d JointTrajectory::interpDq(double t) const {
  if (t_grid_.empty() || dq_traj_.empty()) return Vector7d::Zero();
  return interp(dq_traj_, t);
}

}  // namespace nonlinear_deepc_controller
