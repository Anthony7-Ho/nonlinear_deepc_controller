#include "nonlinear_deepc_controller/kernel_deepc_controller.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace nonlinear_deepc_controller {

controller_interface::InterfaceConfiguration
KernelDeePCController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
KernelDeePCController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

CallbackReturn KernelDeePCController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::vector<double>>("k_gains", {24,24,24,24,10,6,2});
    auto_declare<std::vector<double>>("d_gains", {2,2,2,1,1,1,0.5});
    auto_declare<std::string>("csv_path", std::string(std::getenv("HOME")) + "/trajectory_export_test.csv"); //TODO: change path
    auto_declare<std::string>("log_path",
      std::string(std::getenv("HOME")) + "/franka_ros2_ws/src/nonlinear_deepc_controller/performance_evaluation/tau_ext_test_deepc.csv"); //TODO: change path
    auto_declare<std::string>("robot_state_topic", "/franka_robot_state_broadcaster/robot_state");

    // DeePC topics + window
    auto_declare<std::string>("deepc_init_topic", "/deepc/init"); // pub [u_ini; y_ini]
    auto_declare<std::string>("deepc_friction_prediction_topic", "/deepc/friction_prediction"); // sub friction_prediction
    auto_declare<int>("T_ini", 40);

    // Decay parameters
    auto_declare<double>("alpha_max", 1.0);
    auto_declare<double>("alpha_decay_seconds", 0.01); // TODO: tune, decrease --> faster decay of alpha

    // Logging params
    auto_declare<int>("log_decimation", 1);
    auto_declare<bool>("stop_logging_at_end", true);
    auto_declare<double>("post_log_window", 0.0);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "on_init error: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn KernelDeePCController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  csv_path_ = get_node()->get_parameter("csv_path").as_string();
  log_path_ = get_node()->get_parameter("log_path").as_string();
  robot_state_topic_ = get_node()->get_parameter("robot_state_topic").as_string();

  deepc_init_topic_ = get_node()->get_parameter("deepc_init_topic").as_string();
  deepc_friction_prediction_topic_ = get_node()->get_parameter("deepc_friction_prediction_topic").as_string();
  T_ini_ = get_node()->get_parameter("T_ini").as_int();

  alpha_max_ = get_node()->get_parameter("alpha_max").as_double();
  alpha_decay_seconds_ = get_node()->get_parameter("alpha_decay_seconds").as_double();
  // if (alpha_decay_seconds_ <= 1e-6) alpha_decay_seconds_ = 1e-3;

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();

  if (k_gains.size() != static_cast<size_t>(num_joints) ||
      d_gains.size() != static_cast<size_t>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains/d_gains must have size %d.", num_joints);
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

  // DeePC pub/sub
  init_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
      deepc_init_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());

  friction_prediction_sub_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      deepc_friction_prediction_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
      std::bind(&KernelDeePCController::friction_predictionCallback, this, std::placeholders::_1));

  // Logging params
  log_decimation_ = get_node()->get_parameter("log_decimation").as_int();
  if (log_decimation_ < 1) log_decimation_ = 1;
  stop_logging_at_end_ = get_node()->get_parameter("stop_logging_at_end").as_bool();
  post_log_window_ = get_node()->get_parameter("post_log_window").as_double();
  if (post_log_window_ < 0.0) post_log_window_ = 0.0;

  // Prepare logs
  logging_active_ = true;
  tau_ext_hist_.clear();
  t_hist_.clear();
  tau_ext_hist_.reserve(t_grid_.size() * 200);
  t_hist_.reserve(t_grid_.size() * 200);

  // Histories + phase
  resetHistories();
  phase_ = Phase::WARMUP;
  have_friction_prediction_.store(false, std::memory_order_release);
  time_since_friction_prediction_ = 1e9;
  latest_friction_prediction_.setZero();

  return CallbackReturn::SUCCESS;
}

CallbackReturn KernelDeePCController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  dq_filtered_.setZero();
  initial_q_ = q_;
  elapsed_time_ = 0.0;

  logging_active_ = true;
  log_counter_ = 0;
  tau_ext_hist_.clear();
  t_hist_.clear();

  // start again in warmup
  resetHistories();
  phase_ = Phase::WARMUP;
  have_friction_prediction_.store(false, std::memory_order_release);
  time_since_friction_prediction_ = 1e9;
  latest_friction_prediction_.setZero();

  RCLCPP_INFO(get_node()->get_logger(), "KernelDeePCController activated. Warmup started.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn KernelDeePCController::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) {
  try {
    writeLogCsv(log_path_);
    RCLCPP_INFO(get_node()->get_logger(), "Wrote tau_ext log to %s", log_path_.c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed writing tau_ext log: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

void KernelDeePCController::updateJointStates() {
  for (int i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i)  = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

// interpolate trajectory at time t
KernelDeePCController::Vector7d
KernelDeePCController::interp(const std::vector<Vector7d>& data, double t) const {
  if (data.empty()) return Vector7d::Zero();
  if (t <= t_grid_.front()) return data.front();
  if (t >= t_grid_.back())  return data.back();

  auto it = std::upper_bound(t_grid_.begin(), t_grid_.end(), t);
  size_t k = static_cast<size_t>(std::distance(t_grid_.begin(), it) - 1);
  double t0 = t_grid_[k], t1 = t_grid_[k + 1];
  double a = (t - t0) / (t1 - t0);
  return (1.0 - a) * data[k] + a * data[k + 1];
}

controller_interface::return_type
KernelDeePCController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  updateJointStates();

  // Latest external torque
  Vector7d tau_ext = Vector7d::Zero();
  if (have_tau_ext_.load(std::memory_order_acquire)) {
    for (int i = 0; i < num_joints; ++i) tau_ext(i) = tau_ext_last_[i];
  }

  Vector7d tau_cmd = Vector7d::Zero();  // default: 0 torque
  constexpr double kAlphaFilt = 0.99;
  dq_filtered_ = (1.0 - kAlphaFilt) * dq_filtered_ + kAlphaFilt * dq_;

  switch (phase_) {
    case Phase::WARMUP: {
      // apply zero torque, collect (u=0, y=tau_ext) samples until buffers are full
      if (have_tau_ext_.load(std::memory_order_acquire)) {
        bool full = warmupStep(tau_ext);
        if (full) {
          // publish once, then wait for friction_prediction (still zero torque)
          publishInit("warmup_done");
          phase_ = Phase::WAIT_FOR_FRICTION_PREDICTION;
          elapsed_time_ = 0.0;
          RCLCPP_INFO(get_node()->get_logger(), "Warmup complete. Waiting for friction_prediction...");
        }
      }
      break;
    }

    case Phase::WAIT_FOR_FRICTION_PREDICTION: {
      // keep sending zero torques; histories are not updated here
      // (we freeze the window that we sent to the optimizer)
      if (have_friction_prediction_.load(std::memory_order_acquire)) {
        phase_ = Phase::TRACKING;
        time_since_friction_prediction_ = 0.0;
        elapsed_time_ = 0.0;
        RCLCPP_INFO(get_node()->get_logger(), "Received first friction_prediction. Entering TRACKING.");
      }
      break;
    }

    case Phase::TRACKING: {
      // blend impedance with friction_prediction
      elapsed_time_ += period.seconds();
      Vector7d q_des = interp(q_traj_, elapsed_time_);
      Vector7d dq_des = dq_traj_.empty() ? Vector7d::Zero() : interp(dq_traj_, elapsed_time_);
      Vector7d tau_imp = k_gains_.cwiseProduct(q_des - q_) + d_gains_.cwiseProduct(dq_des - dq_filtered_);
      last_tau_imp_ = tau_imp;

      time_since_friction_prediction_ += period.seconds();
      double alpha = alpha_max_ * std::exp(-time_since_friction_prediction_ / alpha_decay_seconds_);
      if (alpha < 0.0) alpha = 0.0;
      if (alpha > alpha_max_) alpha = alpha_max_;

      tau_cmd = tau_imp + alpha * latest_friction_prediction_;

      // update histories with one-step delay
      pushHistories(prev_tau_applied_, tau_ext);
      prev_tau_applied_ = tau_cmd;
      break;
    }
  }

  // Send efforts (tau_cmd is zero unless in TRACKING)
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_cmd(i));
  }

  // Logging: only during tracking phase
  const double t_end = t_grid_.empty() ? 0.0 : t_grid_.back();
  if (stop_logging_at_end_ && elapsed_time_ > t_end + post_log_window_) {
    logging_active_ = false;
  }
  if (logging_active_ && phase_ == Phase::TRACKING) {  // <-- gate by phase
    if (++log_counter_ >= log_decimation_) {
      tau_ext_hist_.push_back(tau_ext);
      t_hist_.push_back(elapsed_time_);
      log_counter_ = 0;
    }
  }


  return controller_interface::return_type::OK;
}

bool KernelDeePCController::loadCsvTrajectory(const std::string& path) {
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

void KernelDeePCController::writeLogCsv(const std::string& path) const {
  std::ofstream out(path);
  if (!out.is_open()) throw std::runtime_error("cannot open log file");

  // Use the shortest length across the two histories we keep for logging (t and tau_ext).
  const size_t N = std::min(t_hist_.size(), tau_ext_hist_.size());

  // Header: label, then timestamps at controller-rate (prefixed with t_)
  out << "label";
  out << std::fixed << std::setprecision(6);
  for (size_t c = 0; c < N; ++c) {
    out << ",t_" << t_hist_[c];
  }
  out << "\n";

  auto write_row = [&](const std::string& label, int joint_idx) {
    out << label;
    out << std::setprecision(10);
    for (size_t c = 0; c < N; ++c) {
      out << "," << tau_ext_hist_[c](joint_idx);
    }
    out << "\n";
  };

  // 7 rows: external torques only
  for (int j = 0; j < num_joints; ++j) {
    write_row("tau_ext_" + std::to_string(j), j);
  }
}

void KernelDeePCController::resetHistories() {
  u_hist_.clear();
  y_hist_.clear();
  prev_tau_applied_.setZero();
}

bool KernelDeePCController::warmupStep(const Vector7d& tau_ext) {
  // During warmup we apply zero torques, so u_prev is zero.
  if (static_cast<int>(u_hist_.size()) >= T_ini_) u_hist_.pop_front();
  if (static_cast<int>(y_hist_.size()) >= T_ini_) y_hist_.pop_front();
  u_hist_.push_back(Vector7d::Zero());
  y_hist_.push_back(tau_ext);
  return (static_cast<int>(u_hist_.size()) >= T_ini_)
      && (static_cast<int>(y_hist_.size()) >= T_ini_);
}

void KernelDeePCController::pushHistories(const Vector7d& u_prev, const Vector7d& y_curr) {
  if (static_cast<int>(u_hist_.size()) >= T_ini_) u_hist_.pop_front();
  if (static_cast<int>(y_hist_.size()) >= T_ini_) y_hist_.pop_front();
  u_hist_.push_back(u_prev);
  y_hist_.push_back(y_curr);
}

void KernelDeePCController::publishInit(const char* reason) {
  if (!init_pub_) return;
  if (static_cast<int>(u_hist_.size()) < T_ini_ || static_cast<int>(y_hist_.size()) < T_ini_) {
    RCLCPP_WARN(get_node()->get_logger(), "publishInit called but buffers not full.");
    return;
  }

  const int N = 8; //TODO: Should be same as prediction horizon in optimizer

  std_msgs::msg::Float64MultiArray msg;
  msg.data.reserve(7 * T_ini_ * 2 + 7 * N);

  // u_ini: oldest -> newest
  for (const auto& u : u_hist_) {
    for (int j = 0; j < num_joints; ++j) msg.data.push_back(u(j));
  }
  // y_ini
  for (const auto& y : y_hist_) {
    for (int j = 0; j < num_joints; ++j) msg.data.push_back(y(j));
  }

  // u_ref := current impedance torque, tiled over N
  for (int k = 0; k < N; ++k) {
    for (int j = 0; j < num_joints; ++j) {
      msg.data.push_back(last_tau_imp_(j));
    }
      
  }
  
  init_pub_->publish(msg);
  RCLCPP_INFO(get_node()->get_logger(), "Published [u_ini; y_ini; u_ref] (%s).", reason);
}

void KernelDeePCController::friction_predictionCallback(const std_msgs::msg::Float64MultiArray& msg) {
  if (msg.data.size() < static_cast<size_t>(num_joints)) {
    RCLCPP_WARN(get_node()->get_logger(), "friction_prediction too small: got %zu, need at least %d",
                msg.data.size(), num_joints);
    return;
  }
  for (int i = 0; i < num_joints; ++i) {
    latest_friction_prediction_(i) = msg.data[i];
  }
  have_friction_prediction_.store(true, std::memory_order_release);
  time_since_friction_prediction_ = 0.0;

  // If we were waiting, move to tracking
  if (phase_ == Phase::WAIT_FOR_FRICTION_PREDICTION) {
    phase_ = Phase::TRACKING;
    // Start a fresh log for the tracking phase
    tau_ext_hist_.clear();
    t_hist_.clear();
    log_counter_ = 0;
    RCLCPP_INFO(get_node()->get_logger(), "friction_prediction received. Transition to TRACKING.");
  }

  // On each new friction_prediction, publish the current [u_ini; y_ini; u_ref] for the solver
  if (static_cast<int>(u_hist_.size()) >= T_ini_ && static_cast<int>(y_hist_.size()) >= T_ini_) {
    publishInit("friction_prediction_callback");
  }
}

}  // namespace nonlinear_deepc_controller

PLUGINLIB_EXPORT_CLASS(nonlinear_deepc_controller::KernelDeePCController,
                       controller_interface::ControllerInterface)
