#pragma once

#include <string>
#include <vector>
#include <array>
#include <atomic>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <franka_msgs/msg/franka_robot_state.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace nonlinear_deepc_controller {

/**
 * Joint impedance controller that follows a CSV trajectory and logs
 * tau_cmd and tau_ext (rows = taus, columns = timesteps).y
 */
class JointImpedanceController final : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  // Parameters
  std::string arm_id_;
  std::string csv_path_;
  std::string log_path_;
  std::string robot_state_topic_;
  Vector7d k_gains_{Vector7d::Zero()};
  Vector7d d_gains_{Vector7d::Zero()};

  // Constants
  const int num_joints = 7;

  // State
  Vector7d q_{Vector7d::Zero()};
  Vector7d dq_{Vector7d::Zero()};
  Vector7d dq_filtered_{Vector7d::Zero()};
  Vector7d initial_q_{Vector7d::Zero()};
  double elapsed_time_{0.0};

  // Trajectory (from CSV)
  std::vector<double> t_grid_;
  std::vector<Vector7d> q_traj_;
  std::vector<Vector7d> dq_traj_;

  // logging (controller-rate)
  std::vector<Vector7d> tau_cmd_hist_;
  std::vector<Vector7d> tau_ext_hist_;
  std::vector<Vector7d> dq_state_hist_;
  std::vector<Vector7d> q_state_hist_;
  std::vector<double> t_hist_; // controller-time stamps
  int  log_decimation_{1}; // log every Nth update
  int  log_counter_{0};
  bool stop_logging_at_end_{true};
  double post_log_window_{0.0};
  bool logging_active_{true};
  

  // RobotState subscriber
  rclcpp::Subscription<franka_msgs::msg::FrankaRobotState>::SharedPtr state_sub_;
  std::atomic<bool> have_tau_ext_{false};
  std::array<double, 7> tau_ext_last_{{0, 0, 0, 0, 0, 0, 0}};
  std::atomic<bool> have_dq_state_{false};
  std::array<double, 7> dq_state_last_{{0, 0, 0, 0, 0, 0, 0}};
  std::atomic<bool> have_q_state_{false};
  std::array<double, 7> q_state_last_{{0, 0, 0, 0, 0, 0, 0}};

  // Helpers
  void updateJointStates();
  bool loadCsvTrajectory(const std::string& path);
  void writeLogCsv(const std::string& path) const;
  Vector7d interp(const std::vector<Vector7d>& data, double t) const;
};

}  // namespace nonlinear_deepc_controller

