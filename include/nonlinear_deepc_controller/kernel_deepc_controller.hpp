#pragma once

#include <string>
#include <vector>
#include <array>
#include <deque>
#include <atomic>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <franka_msgs/msg/franka_robot_state.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace nonlinear_deepc_controller {

/**
 * KernelDeePC-aware joint controller with warmup:
 *  - Warmup: apply zero torque and fill u_ini (zeros) + y_ini (measured tau_ext) for T_ini samples.
 *  - Publish [u_ini; y_ini], then wait for friction_prediction.
 *  - On friction_prediction: blend impedance with friction_prediction; maintain rolling histories and publish new [u_ini; y_ini] on each new friction_prediction.
 *  - Log tau_ext to CSV.
 */
class KernelDeePCController final : public controller_interface::ControllerInterface {
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
  std::string deepc_init_topic_; // publishes [u_ini; y_ini]
  std::string deepc_friction_prediction_topic_; // subscribes to friction_prediction
  int T_ini_{40}; // length of past horizon
  Vector7d k_gains_{Vector7d::Zero()};
  Vector7d d_gains_{Vector7d::Zero()};

  // decay parameters
  double alpha_max_{1.0}; // initial alpha when fresh friction_prediction arrives
  double alpha_decay_seconds_{0.05};  // decay time constant

  // Constants
  const int num_joints = 7;

  // States
  enum class Phase { WARMUP, WAIT_FOR_FRICTION_PREDICTION, TRACKING };
  Phase phase_{Phase::WARMUP};

  Vector7d q_{Vector7d::Zero()};
  Vector7d dq_{Vector7d::Zero()};
  Vector7d dq_filtered_{Vector7d::Zero()};
  Vector7d initial_q_{Vector7d::Zero()};
  double elapsed_time_{0.0};

  // Trajectory (from CSV)
  std::vector<double> t_grid_;
  std::vector<Vector7d> q_traj_;
  std::vector<Vector7d> dq_traj_;

  // DeePC history buffers (fixed size T_ini)
  std::deque<Vector7d> u_hist_; // applied torque history
  std::deque<Vector7d> y_hist_; // measured tau_ext history
  Vector7d prev_tau_applied_{Vector7d::Zero()}; // u_{k-1}
  Vector7d last_tau_imp_{Vector7d::Zero()};

  // DeePC IO
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr init_pub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr friction_prediction_sub_;
  std::atomic<bool> have_friction_prediction_{false};
  Vector7d latest_friction_prediction_{Vector7d::Zero()};
  double time_since_friction_prediction_{1e9};   // seconds since last friction_prediction (large at start)

  // RobotState subscriber
  rclcpp::Subscription<franka_msgs::msg::FrankaRobotState>::SharedPtr state_sub_;
  std::atomic<bool> have_tau_ext_{false};
  std::array<double, 7> tau_ext_last_{{0, 0, 0, 0, 0, 0, 0}};

  // Logging
  std::vector<Vector7d> tau_ext_hist_;
  std::vector<double> t_hist_; // controller-time stamps
  int  log_decimation_{1}; // log every Nth update
  int  log_counter_{0};
  bool stop_logging_at_end_{true};
  double post_log_window_{0.0};
  bool logging_active_{true};

  // Helpers
  void updateJointStates();
  bool loadCsvTrajectory(const std::string& path);
  void writeLogCsv(const std::string& path) const;
  Vector7d interp(const std::vector<Vector7d>& data, double t) const;

  // warmup & histories
  void resetHistories(); // clear histories; set prev_tau_applied_ = 0
  bool warmupStep(const Vector7d& tau_ext); // push one warmup sample; return true when full
  void pushHistories(const Vector7d& u_prev, // push u_{k-1}, y_k during tracking
                     const Vector7d& y_curr);
  void publishInit(const char* reason); // publish [u_ini; y_ini; u_ref]

  void friction_predictionCallback(const std_msgs::msg::Float64MultiArray& msg);
};

}  // namespace nonlinear_deepc_controller
