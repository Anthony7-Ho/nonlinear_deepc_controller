#include "nonlinear_deepc_controller/joint_impedance_controller.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

namespace nonlinear_deepc_controller {

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

CallbackReturn JointImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::vector<double>>("k_gains", {24,24,24,24,10,6,2});
    auto_declare<std::vector<double>>("d_gains", {2,2,2,1,1,1,0.5});
    auto_declare<std::string>("csv_path", std::string(std::getenv("HOME")) + "/trajectory_validation.csv"); //TODO: change path that controller has to follow
    auto_declare<std::string>("log_path",
      std::string(std::getenv("HOME")) + "/franka_ros2_ws/src/nonlinear_deepc_controller/data_processing/testtest.csv"); //TODO: change path where it has to store u and tau_ext
    auto_declare<std::string>("robot_state_topic", "/franka_robot_state_broadcaster/robot_state");
    auto_declare<int>("log_decimation", 1);
    auto_declare<bool>("stop_logging_at_end", true);
    auto_declare<double>("post_log_window", 0.0);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "on_init error: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  csv_path_ = get_node()->get_parameter("csv_path").as_string();
  log_path_ = get_node()->get_parameter("log_path").as_string();
  robot_state_topic_ = get_node()->get_parameter("robot_state_topic").as_string();

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();

  if (k_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (k_gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains should be of size %d but is of size %zu",
                 num_joints, k_gains.size());
    return CallbackReturn::FAILURE;
  }
  if (d_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (d_gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains should be of size %d but is of size %zu",
                 num_joints, d_gains.size());
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < num_joints; ++i) {
    k_gains_(i) = k_gains.at(i);
    d_gains_(i) = d_gains.at(i);
  }
  dq_filtered_.setZero();

  if (!traj_.loadFromCsv(csv_path_)) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to load trajectory CSV: %s", csv_path_.c_str());
    return CallbackReturn::FAILURE;
  }

  logger_.configure(
    get_node()->get_parameter("log_decimation").as_int(),
    get_node()->get_parameter("stop_logging_at_end").as_bool(),
    get_node()->get_parameter("post_log_window").as_double());
  logger_.setHeader(Logger::joint_header());

  // RobotState subscriber
  state_sub_ = get_node()->create_subscription<franka_msgs::msg::FrankaRobotState>(
      robot_state_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
      [this](const franka_msgs::msg::FrankaRobotState& msg) {
        if (msg.tau_ext_hat_filtered.effort.size() >= static_cast<size_t>(num_joints)) {
          for (int i = 0; i < num_joints; ++i) {
            tau_ext_last_[i] = msg.tau_ext_hat_filtered.effort[i];
          }
          have_tau_ext_.store(true, std::memory_order_release);
          //RCLCPP_INFO(get_node()->get_logger(), "Received tau_ext update");
        }
        if (msg.measured_joint_state.velocity.size() >= static_cast<size_t>(num_joints)) {
          for (int i = 0; i < num_joints; ++i) {
            dq_state_last_[i] = msg.measured_joint_state.velocity[i];
          }
          have_dq_state_.store(true, std::memory_order_release);
          //RCLCPP_INFO(get_node()->get_logger(), "Received dq_state update");
        }
        if (msg.measured_joint_state.position.size() >= static_cast<size_t>(num_joints)) {
          for (int i = 0; i < num_joints; ++i) {
            q_state_last_[i] = msg.measured_joint_state.position[i];
          }
          have_q_state_.store(true, std::memory_order_release);
          //RCLCPP_INFO(get_node()->get_logger(), "Received q_state update");
        }
        else {
          //RCLCPP_WARN(get_node()->get_logger(), "Received incomplete messages");
        }
      });

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  dq_filtered_.setZero();
  initial_q_ = q_;
  elapsed_time_ = 0.0;

  logger_.reset();

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) {
  try {
    logger_.write(log_path_);
    RCLCPP_INFO(get_node()->get_logger(), "Wrote torque log to %s", log_path_.c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed writing torque log: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

void JointImpedanceController::updateJointStates() {
  for (int i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i)  = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

// ======================= Main control loop ========================
controller_interface::return_type
JointImpedanceController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  elapsed_time_ += period.seconds();
  updateJointStates();

  // Desired from CSV (interpolated)
  Vector7d q_des  = traj_.interpQ(elapsed_time_);
  Vector7d dq_des = traj_.interpDq(elapsed_time_);

  // PD impedance
  constexpr double kAlpha = 0.99;
  dq_filtered_ = (1.0 - kAlpha) * dq_filtered_ + kAlpha * dq_;
  Vector7d tau_cmd = k_gains_.cwiseProduct(q_des - q_) + d_gains_.cwiseProduct(dq_des - dq_filtered_);

  // Send efforts
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_cmd(i));
  }

  // External torques (latest)
  Vector7d tau_ext = Vector7d::Zero();
  if (have_tau_ext_.load(std::memory_order_acquire)) {
    for (int i = 0; i < num_joints; ++i) tau_ext(i) = tau_ext_last_[i];
  }

  // dq from robot_state (latest)
  Vector7d dq_state = Vector7d::Zero();
  if (have_dq_state_.load(std::memory_order_acquire)) {
    for (int i = 0; i < num_joints; ++i) dq_state(i) = dq_state_last_[i];
  }

  // q from robot_state (latest)
  Vector7d q_state = Vector7d::Zero();
  if (have_q_state_.load(std::memory_order_acquire)) {
    for (int i = 0; i < num_joints; ++i) q_state(i) = q_state_last_[i];
  }

  static bool first_debug = true;
  if (first_debug) {
    first_debug = false;
    RCLCPP_INFO(
        get_node()->get_logger(),
        "debug: tau_ext[0]=%f dq_state[0]=%f q_state[0]=%f",
        tau_ext(0), dq_state(0), q_state(0));
  }

  // logging
  const double t_end = traj_.empty() ? 0.0 : traj_.tGrid().back();
  logger_.updateActiveWindow(elapsed_time_, t_end);
  
  bool have_all_state =
    have_tau_ext_.load(std::memory_order_acquire) &&
    have_dq_state_.load(std::memory_order_acquire) &&
    have_q_state_.load(std::memory_order_acquire);
  
  if (logger_.active() && have_all_state) {
    auto row = logger_.makeRow();
    row.add(elapsed_time_)
       .addEigenVector(tau_cmd)
       .addEigenVector(tau_ext)
       .addEigenVector(q_state)
       .addEigenVector(dq_state);
    row.checkSizeOrThrow();
    logger_.pushRow(row.vec());
  }

  return controller_interface::return_type::OK;
}

}  // namespace nonlinear_deepc_controller

PLUGINLIB_EXPORT_CLASS(nonlinear_deepc_controller::JointImpedanceController,
                       controller_interface::ControllerInterface)


