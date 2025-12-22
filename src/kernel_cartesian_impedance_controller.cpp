#include "nonlinear_deepc_controller/kernel_cartesian_impedance_controller.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include <nlohmann/json.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace nonlinear_deepc_controller {

using json = nlohmann::json;

// Pseudo-inverse helper
inline void pseudoInverse(const Eigen::MatrixXd& M_, Eigen::MatrixXd& M_pinv_, bool damped = true) {
  double lambda_ = damped ? 0.2 : 0.0;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto sing_vals_ = svd.singularValues();
  Eigen::MatrixXd S_ = Eigen::MatrixXd::Zero(M_.rows(), M_.cols());
  for (int i = 0; i < sing_vals_.size(); i++) {
    S_(i, i) = sing_vals_(i) / (sing_vals_(i) * sing_vals_(i) + lambda_ * lambda_);
  }
  M_pinv_ = svd.matrixV() * S_.transpose() * svd.matrixU().transpose();
}

controller_interface::InterfaceConfiguration
KernelCartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
KernelCartesianImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }

  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(franka_robot_model_name);
  }  

  return config;
}


CallbackReturn KernelCartesianImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::string>("csv_path", std::string(std::getenv("HOME")) + "/cartesian_test_y.csv"); // TODO: change
    auto_declare<std::string>("log_path", std::string(std::getenv("HOME")) + 
        "/franka_ros2_ws/src/nonlinear_deepc_controller/performance_evaluation/cartesian_log_kernel_y.csv"); // TODO: change
    auto_declare<std::string>("robot_state_topic", "/franka_robot_state_broadcaster/robot_state");

    // Kernel bundle dir (relative under package share)
    auto_declare<std::string>("kernel_bundle_dir", "data/kernel_deepc_bundle");

    auto_declare<int>("T_ini", 20); // length of past horizon (should match T_past used in data_processing)

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

CallbackReturn KernelCartesianImpedanceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  robot_name_ = arm_id_;
  csv_path_ = get_node()->get_parameter("csv_path").as_string();
  log_path_ = get_node()->get_parameter("log_path").as_string();
  robot_state_topic_ = get_node()->get_parameter("robot_state_topic").as_string();

  kernel_bundle_dir_param_ = get_node()->get_parameter("kernel_bundle_dir").as_string();
  T_ini_ = get_node()->get_parameter("T_ini").as_int();

  try {
    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        franka_semantic_components::FrankaRobotModel(
            robot_name_ + "/" + k_robot_model_interface_name,
            robot_name_ + "/" + k_robot_state_interface_name));
  } catch (const std::exception& e) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to create FrankaRobotModel: %s", e.what());
    return CallbackReturn::FAILURE;
  }

  if (!loadCartesianTrajectory(csv_path_)) {
    RCLCPP_FATAL(get_node()->get_logger(), "Failed to load Cartesian trajectory CSV: %s", csv_path_.c_str());
    return CallbackReturn::FAILURE;
  }


  // RobotState subscriber
  state_sub_ = get_node()->create_subscription<franka_msgs::msg::FrankaRobotState>(
      robot_state_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
      [this](const franka_msgs::msg::FrankaRobotState& msg) {
        if (msg.tau_ext_hat_filtered.effort.size() >= static_cast<size_t>(num_joints)) {
          // read friction torque
          for (int i = 0; i < num_joints; ++i) {
            tau_ext_last_[i] = msg.tau_ext_hat_filtered.effort[i];
          }
          have_tau_ext_.store(true, std::memory_order_release);
        }

        // end-effector position from PoseStamped
        const auto & p = msg.o_t_ee.pose.position;   // <- PoseStamped
        ee_pos_last_(0) = p.x;
        ee_pos_last_(1) = p.y;
        ee_pos_last_(2) = p.z;
        have_ee_pose_.store(true, std::memory_order_release);

        // External wrench for Cartesian impedance
        auto wrench = msg.o_f_ext_hat_k;
        auto arr = convertToStdArray(wrench);
        O_F_ext_hat_K_ = arr;
        arrayToMatrix(O_F_ext_hat_K_, O_F_ext_hat_K_M_);
      });

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
  time_since_friction_prediction_ = 0.0;

  // Resolve absolute bundle dir and load kernel model
  try {
    std::string pkg_share =
        ament_index_cpp::get_package_share_directory("nonlinear_deepc_controller");
    kernel_bundle_dir_ = pkg_share + "/" + kernel_bundle_dir_param_;
  } catch (const std::exception& e) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "Failed to get package share directory: %s", e.what());
    return CallbackReturn::FAILURE;
  }

  if (!loadKernelBundle(kernel_bundle_dir_)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "Failed to load kernel bundle from %s", kernel_bundle_dir_.c_str());
    return CallbackReturn::FAILURE;
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn KernelCartesianImpedanceController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

  std::array<double, 16> initial_pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_pose.data()));
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  position_d_target_ = position_d_;
  orientation_d_target_ = orientation_d_;

  dq_filtered_.setZero();
  initial_q_ = q_;
  elapsed_time_ = 0.0;

  logging_active_ = true;
  log_counter_ = 0;
  tau_ext_hist_.clear();
  friction_pred_hist_.clear();
  tau_residual_hist_.clear();
  ee_pos_hist_.clear();
  t_hist_.clear();

  resetHistories();
  phase_ = Phase::WARMUP;
  time_since_friction_prediction_ = 0.0;

  RCLCPP_INFO(get_node()->get_logger(), "KernelCartesianImpedanceController activated. Warmup started.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn KernelCartesianImpedanceController::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->release_interfaces();

  try {
    writeLogCsv(log_path_);
    RCLCPP_INFO(get_node()->get_logger(), "Wrote tau_ext log to %s", log_path_.c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed writing tau_ext log: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

void KernelCartesianImpedanceController::updateJointStates() {
  for (int i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i)  = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

// ============ Main control loop ===============
controller_interface::return_type
KernelCartesianImpedanceController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  updateJointStates();

  // Latest external torque
  Vector7d tau_ext = Vector7d::Zero();

  // cartesian end-effector position
  Vector3d ee_pos = Vector3d::Zero();

  if (have_tau_ext_.load(std::memory_order_acquire)) {
    for (int i = 0; i < num_joints; ++i) {
      tau_ext(i) = tau_ext_last_[i];
    }
  }

  if (have_ee_pose_.load(std::memory_order_acquire)) {
    ee_pos = ee_pos_last_;  // just copy the vector
  }

  Vector7d friction_pred = Vector7d::Zero();
  Vector7d tau_cmd = Vector7d::Zero();

  switch (phase_) {
    case Phase::WARMUP: {
      // apply zero torque, collect (u=0, y=tau_ext) samples until buffers are full
      if (have_tau_ext_.load(std::memory_order_acquire)) {
        bool full = warmupStep(tau_ext);
        if (full) {
          phase_ = Phase::TRACKING;
          elapsed_time_ = 0.0;
          time_since_friction_prediction_ = 0.0;
          RCLCPP_INFO(get_node()->get_logger(), "Warmup complete. Entering TRACKING.");
        }
      }
      break;
    }

    case Phase::TRACKING: {
      elapsed_time_ += period.seconds();
      time_since_friction_prediction_ += period.seconds();

      if (!first_update_) {
        pushYHistory(tau_ext);
      } else {
        first_update_ = false;
      }

      // Interpolate Cartesian reference
      Eigen::Vector3d pos_target = interpPos(cart_pos_traj_, elapsed_time_);
      Eigen::Quaterniond quat_target = interpQuat(cart_quat_traj_, elapsed_time_);
      position_d_target_ = pos_target;
      orientation_d_target_ = quat_target;
        
      // Get model quantities
      std::array<double, 49> mass = franka_robot_model_->getMassMatrix();
      std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
      std::array<double, 42> jacobian_array =
          franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
      std::array<double, 16> pose =
          franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
        
      Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
      Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
      Eigen::Map<Eigen::Matrix<double, 7, 7>> M(mass.data());
      Eigen::Affine3d transform(Eigen::Matrix4d::Map(pose.data()));
      Eigen::Vector3d position(transform.translation());
      Eigen::Quaterniond orientation(transform.rotation());
        
      updateJointStates(); // updates q_, dq_
        
      // Filtered targets and stiffness
      update_stiffness_and_references();
        
      // Cartesian error
      error_.head<3>() = position - position_d_;
        
      if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
        orientation.coeffs() << -orientation.coeffs();
      }
      Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
      error_.tail<3>() << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
      error_.tail<3>() = -transform.rotation() * error_.tail<3>();
      
      // Integrator on pose error
      I_error_ += Sm_ * dt_cart_ * integrator_weights_.cwiseProduct(error_);
      for (int i = 0; i < 6; ++i) {
        I_error_(i,0) =
            std::min(std::max(-max_I_(i,0), I_error_(i,0)), max_I_(i,0));
      }
      
      // Operational space inertia
      Lambda_ = (jacobian * M.inverse() * jacobian.transpose()).inverse();
      Theta_ = Lambda_;
      
      // D correction
      D_ = D_gain_ * K_.cwiseMax(0.0).cwiseSqrt()
                 * Lambda_.cwiseMax(0.0).diagonal().cwiseSqrt().asDiagonal();
      D_.topRightCorner(3,3).setZero();
      D_.bottomLeftCorner(3,3).setZero();
      
      // Task-space velocity
      Eigen::Matrix<double, 6, 1> v = jacobian * dq_;
      
      // Cartesian impedance force
      F_impedance_ = -1.0 * (D_ * v + K_ * error_ /* + I_error_ */);
      
      // External forces & contact control
      F_ext_ = 0.9 * F_ext_ + 0.1 * O_F_ext_hat_K_M_;
      I_F_error_ += dt_cart_ * Sf_ * (F_contact_des_ - F_ext_);
      F_cmd_ = Sf_ * (0.4 * (F_contact_des_ - F_ext_) +
                      0.9 * I_F_error_ +
                      0.9 * F_contact_des_);
      
      // Nullspace + torque
      Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_impedance(7), tau_d(7);
      
      pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv_);
      
      tau_nullspace =
          (Eigen::MatrixXd::Identity(7, 7) -
           jacobian.transpose() * jacobian_transpose_pinv_) *
          (nullspace_stiffness_ * (config_control_ ? 1.0 : 0.0) *
               (q_d_nullspace_ - q_) -
           (2.0 * std::sqrt(nullspace_stiffness_)) * dq_);
      
      tau_impedance = jacobian.transpose() * Sm_ * F_impedance_ +
          jacobian.transpose() * Sf_ * F_cmd_;
      
      // base cartesian torque before friction comp
      Eigen::Matrix<double,7,1> tau_cart = tau_impedance + tau_nullspace + coriolis;
      last_tau_imp_ = tau_cart;  // used in DeePC as u_future
      
      // Friction prediction
      auto t0 = std::chrono::high_resolution_clock::now();
      friction_pred = computeFrictionPrediction();
      auto t1 = std::chrono::high_resolution_clock::now();
      double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      
      RCLCPP_INFO_THROTTLE(
          get_node()->get_logger(),
          *get_node()->get_clock(),
          500,
          "computeFrictionPrediction() time = %.3f ms",
          dt_ms);

      /*
      // hard ball scheduling around origin
      const double r = position.norm();   // ||p||, ball centered at origin

      constexpr double r0 = 0.55;         // [m] tune this split radius
      constexpr double alpha_close = 1.4;
      constexpr double alpha_far   = 1.8;

      const double alpha = (r <= r0) ? alpha_close : alpha_far;

      Eigen::Matrix<double,7,1> tau_fb_filtered = saturateTorqueRate(tau_cart, tau_J_d_M_);

      // Save filtered output for next iteration
      tau_J_d_M_ = tau_fb_filtered;
      
      for (int j = 0; j < 7; ++j) {
        // Only add friction compensation if joint torque is actually overcoming friction
        //constexpr double dq_thresh = 0.02;
        if (std::abs(tau_fb_filtered(j)) > std::abs(friction_pred(j))) {
          tau_ff_des_(j) = alpha * friction_pred(j);
        }
      }

      Eigen::Matrix<double,7,1> tau_ff = saturateTorqueRateFF(tau_ff_des_, tau_ff_prev_);
      tau_ff_prev_ = tau_ff;

      Eigen::Matrix<double,7,1> tau_cmd = tau_fb_filtered + tau_ff;
      */

      
      const double r = position.norm();   // ||p||, ball centered at origin

      constexpr double r0 = 0.55;         // [m] tune this split radius
      constexpr double alpha_close = 1.5; //1.4
      constexpr double alpha_far = 2.5; //1.8

      const double alpha = (r <= r0) ? alpha_close : alpha_far;
      
      for (int j = 0; j < 7; ++j) {
        // Only add friction compensation if joint torque is actually overcoming friction
        //constexpr double dq_thresh = 0.02;
        if (std::abs(tau_cart(j)) > std::abs(friction_pred(j))) {
          tau_ff_des_(j) = alpha * friction_pred(j);
        }
      }

      // Remove nullspace component
      Eigen::Matrix<double, 7, 7> P_tau = jacobian.transpose() * jacobian_transpose_pinv_;
      Eigen::Matrix<double, 7, 1> tau_ff_task = P_tau * tau_ff_des_;

      Eigen::Matrix<double,7,1> tau_des = tau_cart + tau_ff_task;

      Eigen::Matrix<double,7,1> tau_cmd = saturateTorqueRateFF(tau_des, tau_J_d_M_);
      tau_J_d_M_ = tau_cmd;
      
      // Update histories
      pushUHistory(tau_cmd);
      prev_tau_applied_ = tau_cmd;

      // Send efforts
      for (int i = 0; i < num_joints; ++i) {
        command_interfaces_[i].set_value(tau_cmd(i));
      }

      break;
    }
  }

  // Logging: only during tracking phase
  const double t_end = t_grid_.empty() ? 0.0 : t_grid_.back();
  if (stop_logging_at_end_ && elapsed_time_ > t_end + post_log_window_) {
    logging_active_ = false;
  }
  if (logging_active_ && phase_ == Phase::TRACKING) {
    if (++log_counter_ >= log_decimation_) {
      tau_ext_hist_.push_back(tau_ext);
      friction_pred_hist_.push_back(friction_pred);

      Vector7d tau_resid = tau_ext - friction_pred;
      tau_residual_hist_.push_back(tau_resid);
      ee_pos_hist_.push_back(ee_pos);

      t_hist_.push_back(elapsed_time_);
      log_counter_ = 0;
    }
  }

  return controller_interface::return_type::OK;
}

// ========= Cartesian impedance helpers ============
void KernelCartesianImpedanceController::arrayToMatrix(const std::array<double, 7>& inputArray, Eigen::Matrix<double, 7, 1>& resultMatrix) {
  for (size_t i = 0; i < 7; ++i) {
    resultMatrix(i,0) = inputArray[i];
  }
}

void KernelCartesianImpedanceController::arrayToMatrix(const std::array<double, 6>& inputArray, Eigen::Matrix<double, 6, 1>& resultMatrix) {
  for (size_t i = 0; i < 6; ++i) {
    resultMatrix(i,0) = inputArray[i];
  }
}

std::array<double, 6>
KernelCartesianImpedanceController::convertToStdArray(const geometry_msgs::msg::WrenchStamped& wrench) {
  std::array<double, 6> result;
  result[0] = wrench.wrench.force.x;
  result[1] = wrench.wrench.force.y;
  result[2] = wrench.wrench.force.z;
  result[3] = wrench.wrench.torque.x;
  result[4] = wrench.wrench.torque.y;
  result[5] = wrench.wrench.torque.z;
  return result;
}

Eigen::Matrix<double, 7, 1>
KernelCartesianImpedanceController::saturateTorqueRate(const Eigen::Matrix<double, 7, 1>& tau_d_calculated, const Eigen::Matrix<double, 7, 1>& tau_J_d_in) {
  Eigen::Matrix<double, 7, 1> tau_d_saturated;
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d_in[i];
    tau_d_saturated[i] =
        tau_J_d_in[i] +
        std::max(std::min(difference, delta_tau_max_cart_), -delta_tau_max_cart_);
  }
  return tau_d_saturated;
}

Eigen::Matrix<double, 7, 1>
KernelCartesianImpedanceController::saturateTorqueRateFF(const Eigen::Matrix<double, 7, 1>& tau_d_calculated, const Eigen::Matrix<double, 7, 1>& tau_J_d_in) {
  Eigen::Matrix<double, 7, 1> tau_d_saturated;
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d_in[i];
    tau_d_saturated[i] =
        tau_J_d_in[i] +
        std::max(std::min(difference, delta_tau_max_cart_), -delta_tau_max_cart_ff_);
  }
  return tau_d_saturated;
}

void KernelCartesianImpedanceController::update_stiffness_and_references() {
  nullspace_stiffness_ =
      filter_params_ * nullspace_stiffness_target_ +
      (1.0 - filter_params_) * nullspace_stiffness_;

  position_d_ =
      filter_params_ * position_d_target_ +
      (1.0 - filter_params_) * position_d_;

  orientation_d_ =
      orientation_d_.slerp(filter_params_, orientation_d_target_);

  F_contact_des_ = 0.05 * F_contact_target_ + 0.95 * F_contact_des_;
}
// =====================================

// =================== Trajectory & logging helpers ========================
bool KernelCartesianImpedanceController::loadCartesianTrajectory(const std::string& path) {
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

  if (t_idx < 0 || x_idx < 0 || y_idx < 0 || z_idx < 0 ||
      qx_idx < 0 || qy_idx < 0 || qz_idx < 0 || qw_idx < 0) {
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

    double t  = vals[t_idx];
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

KernelCartesianImpedanceController::Vector3d
KernelCartesianImpedanceController::interpPos(const std::vector<Vector3d>& data, double t) const {
  if (data.empty()) return Vector3d::Zero();
  if (t <= t_grid_.front()) return data.front();
  if (t >= t_grid_.back())  return data.back();

  auto it = std::upper_bound(t_grid_.begin(), t_grid_.end(), t);
  size_t k = static_cast<size_t>(std::distance(t_grid_.begin(), it) - 1);
  double t0 = t_grid_[k], t1 = t_grid_[k + 1];
  double a = (t - t0) / (t1 - t0);
  return (1.0 - a) * data[k] + a * data[k + 1];
}

Eigen::Quaterniond
KernelCartesianImpedanceController::interpQuat(const std::vector<Eigen::Quaterniond>& data, double t) const {
  if (data.empty()) return Eigen::Quaterniond::Identity();
  if (t <= t_grid_.front()) return data.front();
  if (t >= t_grid_.back())  return data.back();

  auto it = std::upper_bound(t_grid_.begin(), t_grid_.end(), t);
  size_t k = static_cast<size_t>(std::distance(t_grid_.begin(), it) - 1);
  double t0 = t_grid_[k], t1 = t_grid_[k + 1];
  double a = (t - t0) / (t1 - t0);
  return data[k].slerp(a, data[k + 1]);
}


void KernelCartesianImpedanceController::writeLogCsv(const std::string& path) const {
  std::ofstream out(path);
  if (!out.is_open()) throw std::runtime_error("cannot open log file");

  const size_t N = std::min({t_hist_.size(), tau_ext_hist_.size(), friction_pred_hist_.size(), tau_residual_hist_.size(), ee_pos_hist_.size()});

  // Header
  out << "time_s";
  for (int j = 0; j < num_joints; ++j) {
    out << ",tau_ext_" << j;
  }
  for (int j = 0; j < num_joints; ++j) {
    out << ",friction_pred_" << j;
  }
  for (int j = 0; j < num_joints; ++j) {
    out << ",tau_residual_" << j;
  }
  out << ",x,y,z";
  out << "\n";

  out << std::fixed << std::setprecision(6);

  for (size_t i = 0; i < N; ++i) {
    // time column
    out << t_hist_[i];

    // tau_ext columns
    for (int j = 0; j < num_joints; ++j)
      out << "," << tau_ext_hist_[i](j);

    // friction_prediction columns
    if (i < friction_pred_hist_.size()) {
      for (int j = 0; j < num_joints; ++j) {
        out << "," << friction_pred_hist_[i](j);
      }
    } else {
      for (int j = 0; j < num_joints; ++j) {
        out << ",0.0";
      }
    }

    // tau_residual columns
    if (i < tau_residual_hist_.size()) {
      for (int j = 0; j < num_joints; ++j) {
        out << "," << tau_residual_hist_[i](j);
      }
    } else {
      for (int j = 0; j < num_joints; ++j) {
        out << ",0.0";
      }
    }

    // ee_pos columns
    if (i < ee_pos_hist_.size()) {
          out << "," << ee_pos_hist_[i](0)
              << "," << ee_pos_hist_[i](1)
              << "," << ee_pos_hist_[i](2);
    } else {
      out << ",0.0,0.0,0.0";
    }
    out << "\n";
  }

}
// ===============================================

// ====================== History management ========================
void KernelCartesianImpedanceController::resetHistories() {
  u_hist_.clear();
  y_hist_.clear();
  prev_tau_applied_.setZero();
  first_update_ = true;
}

bool KernelCartesianImpedanceController::warmupStep(const Vector7d& tau_ext) {
  if (static_cast<int>(u_hist_.size()) >= T_ini_) u_hist_.pop_front();
  if (static_cast<int>(y_hist_.size()) >= T_ini_) y_hist_.pop_front();
  u_hist_.push_back(Vector7d::Zero());
  y_hist_.push_back(tau_ext);
  return (static_cast<int>(u_hist_.size()) >= T_ini_) && (static_cast<int>(y_hist_.size()) >= T_ini_);
}

void KernelCartesianImpedanceController::pushUHistory(const Vector7d& u_curr) {
  if (static_cast<int>(u_hist_.size()) >= T_ini_) {
    u_hist_.pop_front();
  }
  u_hist_.push_back(u_curr);
}

void KernelCartesianImpedanceController::pushYHistory(const Vector7d& y_next) {
  if (static_cast<int>(y_hist_.size()) >= T_ini_) {
    y_hist_.pop_front();
  }
  y_hist_.push_back(y_next);
}
// =========================================================

// ===================== Kernel bundle loading ========================
bool KernelCartesianImpedanceController::loadBinMatrix(const std::string& path, int rows, int cols, Eigen::MatrixXd& M) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Cannot open %s", path.c_str());
    return false;
  }
  std::vector<double> buf(rows * cols);
  f.read(reinterpret_cast<char*>(buf.data()),
         static_cast<std::streamsize>(buf.size() * sizeof(double)));
  if (!f.good()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Error reading %s", path.c_str());
    return false;
  }
  M.resize(rows, cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      M(r, c) = static_cast<double>(buf[r * cols + c]);
    }
  }
  return true;
}

bool KernelCartesianImpedanceController::loadKernelBundle(const std::string& dir) {
  RCLCPP_INFO(get_node()->get_logger(), "Loading kernel bundles for all joints from %s", dir.c_str());

  bool first_joint = true;

  for (int j = 0; j < num_joints; ++j) {
    std::string joint_dir = dir + "/joint_" + std::to_string(j);
    std::string meta_path = joint_dir + "/meta.json";
    std::string Kg_path = joint_dir + "/Kg.bin";
    std::string X_path = joint_dir + "/X.bin";
    std::string Hy_path = joint_dir + "/Hy_future.bin";

    std::ifstream mf(meta_path);
    if (!mf.is_open()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Cannot open %s for joint %d", meta_path.c_str(), j);
      return false;
    }

    json meta;
    try {
      mf >> meta;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_node()->get_logger(), "Error parsing %s: %s", meta_path.c_str(), e.what());
      return false;
    }

    int Kg_rows = meta["Kg"]["rows"];
    int Kg_cols = meta["Kg"]["cols"];
    int X_rows = meta["X"]["rows"];
    int X_cols = meta["X"]["cols"];
    int Hy_rows = meta["Hy_future"]["rows"];
    int Hy_cols = meta["Hy_future"]["cols"];

    double rbf_scale_meta = meta["rbf_scale"];
    double lambda_g_meta = meta["lambda_g"];
    double lambda_k_meta = meta["lambda_k"];
    int T_past_meta = meta["T_past"];
    int T_future_meta = meta["T_future"];

    if (first_joint) {
      Hc_ = Kg_cols;
      d_full_ = X_rows;
      N_pred_ = Hy_rows;
      rbf_scale_ = rbf_scale_meta;
      lambda_g_ = lambda_g_meta;
      lambda_k_ = lambda_k_meta;

      // sanity check d_full_ == 2*T_past + T_future
      if (d_full_ != 2 * T_past_meta + T_future_meta) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "d_full (%d) != 2*T_past(%d)+T_future(%d) from meta.json",
                    d_full_, T_past_meta, T_future_meta);
      }
    } else {
      if (Kg_cols != Hc_ || X_rows != d_full_ || Hy_rows != N_pred_) {
        RCLCPP_ERROR(get_node()->get_logger(), "Dimension mismatch in joint %d kernel bundle.", j);
        return false;
      }
      if (std::fabs(rbf_scale_meta - rbf_scale_) > 1e-7 ||
          std::fabs(lambda_g_meta - lambda_g_)  > 1e-5 ||
          std::fabs(lambda_k_meta - lambda_k_)  > 1e-2) {
        RCLCPP_WARN(get_node()->get_logger(), "Hyperparameters differ in joint %d meta.json; using joint_0 values.", j);
      }
    }

    // Load matrices for this joint
    if (!loadBinMatrix(Kg_path, Kg_rows, Kg_cols, Kg_list_[j])) return false;
    if (!loadBinMatrix(X_path, X_rows, X_cols, X_list_[j]))  return false;
    if (!loadBinMatrix(Hy_path, Hy_rows, Hy_cols, Hy_future_list_[j])) return false;

    // Precompute column norms ||x_i||^2
    X_col_norm2_list_[j].resize(Hc_);
    for (int c = 0; c < Hc_; ++c) {
      X_col_norm2_list_[j](c) = X_list_[j].col(c).squaredNorm();
    }

    // Build A_j = lambda_g I + lambda_k * Kg^T Kg
    Eigen::MatrixXd A_j = lambda_g_ * Eigen::MatrixXd::Identity(Kg_cols, Kg_cols) +
                          lambda_k_ * (Kg_list_[j].transpose() * Kg_list_[j]);

    // Debug: min eigenvalue
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A_j);
    double min_eig = es.eigenvalues().minCoeff();
    RCLCPP_ERROR(get_node()->get_logger(), "Joint %d: min eigenvalue of A_j = %f", j, min_eig);

    A_chol_list_[j].compute(A_j);
    if (A_chol_list_[j].info() != Eigen::Success) {
      RCLCPP_ERROR(get_node()->get_logger(), "Cholesky factorization of A for joint %d failed.", j);
      return false;
    }

    first_joint = false;
  }

  kernel_loaded_ = true;
  RCLCPP_INFO(get_node()->get_logger(),
              "Kernel bundles loaded for all %d joints. Hc=%d, d_full=%d, N_pred=%d, T_ini=%d",
              num_joints, Hc_, d_full_, N_pred_, T_ini_);

  return true;
}
// =========================================================


// =================== Friction prediction =======================
// Closed-form friction prediction (per joint, pure RBF)
KernelCartesianImpedanceController::Vector7d
KernelCartesianImpedanceController::computeFrictionPrediction() {
  Vector7d result = Vector7d::Zero();

  if (!kernel_loaded_) return result;
  if (static_cast<int>(u_hist_.size()) < T_ini_ ||
      static_cast<int>(y_hist_.size()) < T_ini_) {
    return result;
  }

  // For each joint, build its own x_ini and do closed-form prediction
  for (int joint = 0; joint < num_joints; ++joint) {
    // u_ini, y_ini (length T_ini_) for this joint
    Eigen::VectorXd u_ini(T_ini_);
    Eigen::VectorXd y_ini(T_ini_);

    int idx = 0;
    for (const auto& u_vec : u_hist_) {
      if (idx >= T_ini_) break;
      u_ini(idx) = static_cast<double>(u_vec(joint));
      ++idx;
    }

    idx = 0;
    for (const auto& y_vec : y_hist_) {
      if (idx >= T_ini_) break;
      y_ini(idx) = static_cast<double>(y_vec(joint));
      ++idx;
    }

    // Future input: current impedance torque for this joint, tiled N_pred_ times
    Eigen::VectorXd u_future(N_pred_);
    double u_val = static_cast<double>(last_tau_imp_(joint));
    for (int k = 0; k < N_pred_; ++k) {
      u_future(k) = u_val;
    }

    // x_ini = [u_ini; y_ini; u_future]  (length d_full_)
    Eigen::VectorXd x_ini(d_full_);
    x_ini.segment(0, T_ini_) = u_ini;
    x_ini.segment(T_ini_, T_ini_) = y_ini;
    x_ini.segment(2*T_ini_, N_pred_) = u_future;

    // Compute RBF k-vector: k_i = exp(-rbf_scale * ||x_i - x_ini||^2)
    double term_xy = x_ini.squaredNorm();
    Eigen::VectorXd term_ax = X_list_[joint].transpose() * x_ini;
    Eigen::VectorXd d2 = X_col_norm2_list_[joint]
                         + term_xy * Eigen::VectorXd::Ones(Hc_)
                         - 2.0 * term_ax;

    Eigen::VectorXd k_vec(Hc_);
    for (int i = 0; i < Hc_; ++i) {
      k_vec(i) = std::exp(-rbf_scale_ * d2(i));
    }

    // rhs = lambda_k * Kg^T k_vec
    Eigen::VectorXd rhs = lambda_k_ * (Kg_list_[joint].transpose() * k_vec);

    // Solve A_j g_j = rhs with LLT factor
    Eigen::VectorXd g = A_chol_list_[joint].solve(rhs);

    // Predict future outputs y for this joint
    Eigen::VectorXd y_future = Hy_future_list_[joint] * g;

    // First step = friction prediction (receding horizon)
    result(joint) = y_future(0);
  }
  return result;
}
// =========================================================


}  // namespace nonlinear_deepc_controller

PLUGINLIB_EXPORT_CLASS(nonlinear_deepc_controller::KernelCartesianImpedanceController,
                       controller_interface::ControllerInterface)
