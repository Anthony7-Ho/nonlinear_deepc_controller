#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace nonlinear_deepc_controller {

class CartesianTrajectory {
/**
 * Loads and stores a Cartesian trajectory from a CSV file.
 * The CSV file is expected to have the following columns:
 * time_s,x,y,z,qx,qy,qz,qw
 */
 public:
  using Vector3d = Eigen::Matrix<double, 3, 1>;

  bool loadFromCsv(const std::string& path);

  Vector3d interpPos(double t) const;
  Eigen::Quaterniond interpQuat(double t) const;

  const std::vector<double>& tGrid() const { return t_grid_; }
  bool empty() const { return t_grid_.empty(); }

 private:
  std::vector<double> t_grid_;
  std::vector<Vector3d> cart_pos_traj_;
  std::vector<Eigen::Quaterniond> cart_quat_traj_;
};

}  // namespace nonlinear_deepc_controller
