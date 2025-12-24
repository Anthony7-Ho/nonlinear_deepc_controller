#include "nonlinear_deepc_controller/cartesian_trajectory.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>

namespace nonlinear_deepc_controller {

bool CartesianTrajectory::loadFromCsv(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) return false;

  std::string line;
  if (!std::getline(f, line)) return false;

  std::vector<std::string> cols;
  {
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) cols.push_back(c);
  }
  auto col_index = [&](const std::string& name) -> int {
    for (size_t i = 0; i < cols.size(); ++i) {
      if (cols[i] == name) return static_cast<int>(i);
    }
    return -1;
  };

  int t_idx = col_index("time_s");
  int x_idx = col_index("x");
  int y_idx = col_index("y");
  int z_idx = col_index("z");
  int qx_idx = col_index("qx");
  int qy_idx = col_index("qy");
  int qz_idx = col_index("qz");
  int qw_idx = col_index("qw");

  if (t_idx < 0 || x_idx < 0 || y_idx < 0 || z_idx < 0 || qx_idx < 0 || qy_idx < 0 || qz_idx < 0 || qw_idx < 0) {
    return false;
  }

  t_grid_.clear();
  cart_pos_traj_.clear();
  cart_quat_traj_.clear();

  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string tok;
    std::vector<double> vals;
    while (std::getline(ss, tok, ',')) {
      if (tok.empty() || tok == " ") {
        vals.push_back(0.0);
      } else {
        vals.push_back(std::stod(tok));
      }
    }
    if (static_cast<int>(vals.size()) <= qw_idx) continue;

    double t = vals[t_idx];
    double px = vals[x_idx];
    double py = vals[y_idx];
    double pz = vals[z_idx];
    double qx = vals[qx_idx];
    double qy = vals[qy_idx];
    double qz = vals[qz_idx];
    double qw = vals[qw_idx];

    t_grid_.push_back(t);
    cart_pos_traj_.emplace_back(px, py, pz);
    cart_quat_traj_.emplace_back(qw, qx, qy, qz);  // Eigen expects (w,x,y,z)
  }

  return !t_grid_.empty();
}

CartesianTrajectory::Vector3d CartesianTrajectory::interpPos(double t) const {
  if (cart_pos_traj_.empty()) return Vector3d::Zero();
  if (t <= t_grid_.front()) return cart_pos_traj_.front();
  if (t >= t_grid_.back())  return cart_pos_traj_.back();

  auto it = std::upper_bound(t_grid_.begin(), t_grid_.end(), t);
  size_t k = static_cast<size_t>(std::distance(t_grid_.begin(), it) - 1);
  double t0 = t_grid_[k], t1 = t_grid_[k + 1];
  double a = (t - t0) / (t1 - t0);
  return (1.0 - a) * cart_pos_traj_[k] + a * cart_pos_traj_[k + 1];
}

Eigen::Quaterniond CartesianTrajectory::interpQuat(double t) const {
  if (cart_quat_traj_.empty()) return Eigen::Quaterniond::Identity();
  if (t <= t_grid_.front()) return cart_quat_traj_.front();
  if (t >= t_grid_.back())  return cart_quat_traj_.back();

  auto it = std::upper_bound(t_grid_.begin(), t_grid_.end(), t);
  size_t k = static_cast<size_t>(std::distance(t_grid_.begin(), it) - 1);
  double t0 = t_grid_[k], t1 = t_grid_[k + 1];
  double a = (t - t0) / (t1 - t0);
  return cart_quat_traj_[k].slerp(a, cart_quat_traj_[k + 1]);
}

}  // namespace nonlinear_deepc_controller
