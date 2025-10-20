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
    auto_declare<std::string>("csv_path", std::string(std::getenv("HOME")) + "/trajectory_export.csv");
    auto_declare<std::string>("log_path",
      std::string(std::getenv("HOME")) + "/franka_ros2_ws/src/nonlinear_deepc_controller/data_processing/tau_log.csv");
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

  if (!loadCsvTrajectory(csv_path_)) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to load trajectory CSV: %s", csv_path_.c_str());
    return CallbackReturn::FAILURE;
  }

  // RobotState subscriber
  state_sub_ = get_node()->create_subscription<franka_msgs::msg::FrankaRobotState>(
      robot_state_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
      [this](const franka_msgs::msg::FrankaRobotState& msg) {
        if (msg.tau_ext_hat_filtered.effort.size() >= static_cast<size_t>(num_joints)) {
          for (int i = 0; i < num_joints; ++i) {
            tau_ext_last_[i] = msg.tau_ext_hat_filtered.effort[i];
          }
          have_tau_ext_.store(true, std::memory_order_release);
        }
      });

  // Logging params
  log_decimation_ = get_node()->get_parameter("log_decimation").as_int();
  if (log_decimation_ < 1) log_decimation_ = 1;
  stop_logging_at_end_ = get_node()->get_parameter("stop_logging_at_end").as_bool();
  post_log_window_ = get_node()->get_parameter("post_log_window").as_double();
  if (post_log_window_ < 0.0) post_log_window_ = 0.0;

  // Prepare histories
  logging_active_ = true;
  tau_cmd_hist_.clear();
  tau_ext_hist_.clear();
  t_hist_.clear();
  // Reserve to reduce reallocs
  tau_cmd_hist_.reserve(t_grid_.size() * 200);
  tau_ext_hist_.reserve(t_grid_.size() * 200);
  t_hist_.reserve(t_grid_.size() * 200);

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  dq_filtered_.setZero();
  initial_q_ = q_;
  elapsed_time_ = 0.0;

  logging_active_ = true;
  log_counter_ = 0;
  tau_cmd_hist_.clear();
  tau_ext_hist_.clear();
  t_hist_.clear();

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) {
  try {
    writeLogCsv(log_path_);
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

// Interpolate data at time t using linear interpolation
JointImpedanceController::Vector7d
JointImpedanceController::interp(const std::vector<Vector7d>& data, double t) const {
  if (t <= t_grid_.front()) return data.front();
  if (t >= t_grid_.back())  return data.back();

  auto it = std::upper_bound(t_grid_.begin(), t_grid_.end(), t);
  size_t k = static_cast<size_t>(std::distance(t_grid_.begin(), it) - 1);
  double t0 = t_grid_[k], t1 = t_grid_[k + 1];
  double a = (t - t0) / (t1 - t0);
  return (1.0 - a) * data[k] + a * data[k + 1];
}

controller_interface::return_type
JointImpedanceController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  elapsed_time_ += period.seconds();
  updateJointStates();

  // Desired from CSV (interpolated)
  Vector7d q_des = interp(q_traj_, elapsed_time_);
  Vector7d dq_des = dq_traj_.empty() ? Vector7d::Zero() : interp(dq_traj_, elapsed_time_);

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

  // Per-update logging with stop-at-end guard
  const double t_end = t_grid_.back();
  if (stop_logging_at_end_ && elapsed_time_ > t_end + post_log_window_) {
    logging_active_ = false;
  }

  if (logging_active_) {
    if (++log_counter_ >= log_decimation_) {
      tau_cmd_hist_.push_back(tau_cmd);
      tau_ext_hist_.push_back(tau_ext);
      t_hist_.push_back(elapsed_time_);
      log_counter_ = 0;
    }
  }

  return controller_interface::return_type::OK;
}

// Load trajectory from CSV file
bool JointImpedanceController::loadCsvTrajectory(const std::string& path) {
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

  t_grid_.clear(); q_traj_.clear(); dq_traj_.clear();

  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string tok; std::vector<double> vals;
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

void JointImpedanceController::writeLogCsv(const std::string& path) const {
  std::ofstream out(path);
  if (!out.is_open()) throw std::runtime_error("cannot open log file");

  // Use the shortest length across the three histories.
  const size_t N = std::min({t_hist_.size(), tau_cmd_hist_.size(), tau_ext_hist_.size()});

  // Header: label, then all timestamps at controller-rate (prefixed with t_)
  out << "label";
  out << std::fixed << std::setprecision(6);
  for (size_t c = 0; c < N; ++c) {
    out << ",t_" << t_hist_[c];
  }
  out << "\n";

  // Helper to write one joint row from a given source history
  auto write_row = [&](const std::string& label, int joint_idx, const std::vector<Vector7d>& src) {
    out << label;
    out << std::setprecision(10);
    for (size_t c = 0; c < N; ++c) {
      out << "," << src[c](joint_idx);
    }
    out << "\n";
  };

  // 7 rows: commanded torques
  for (int j = 0; j < num_joints; ++j) {
    write_row("tau_cmd_" + std::to_string(j), j, tau_cmd_hist_);
  }
  // 7 rows: external torques
  for (int j = 0; j < num_joints; ++j) {
    write_row("tau_ext_" + std::to_string(j), j, tau_ext_hist_);
  }
}

}  // namespace nonlinear_deepc_controller

PLUGINLIB_EXPORT_CLASS(nonlinear_deepc_controller::JointImpedanceController,
                       controller_interface::ControllerInterface)


