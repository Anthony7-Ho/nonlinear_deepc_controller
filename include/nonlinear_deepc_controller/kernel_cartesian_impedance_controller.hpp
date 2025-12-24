#pragma once

#include <string>
#include <vector>
#include <array>
#include <deque>
#include <atomic>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <franka_msgs/msg/franka_robot_state.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <franka/robot_state.h>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include "nonlinear_deepc_controller/kernel_bundle.hpp"
#include "nonlinear_deepc_controller/friction_predictor.hpp"
#include "nonlinear_deepc_controller/logger.hpp"
#include "nonlinear_deepc_controller/cartesian_trajectory.hpp"
#include "nonlinear_deepc_controller/history_buffer.hpp"


using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace nonlinear_deepc_controller {
/**
 * Friction-aware joint controller with warmup:
 *  - Warmup: apply zero torque and fill u_ini (zeros) + y_ini (measured tau_ext) for T_ini samples.
 *  - At current input, compute friction prediction using the Kernel predictor closed-form solution.
 *  - Add the predicted friction as feedforward compensation to impedance control.
 *  - Maintain rolling histories.
 *  - Log tau_ext to CSV.
 */
class KernelCartesianImpedanceController final : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>; // For per-joint vectors
  using Vector3d = Eigen::Matrix<double, 3, 1>; // For 3D vectors (cartesion position)

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

  // Kernel bundle directory
  std::string kernel_bundle_dir_param_;
  std::string kernel_bundle_dir_;

  KernelBundle kernel_bundle_;
  FrictionPredictor predictor_;
  Logger logger_;
  CartesianTrajectory traj_;
  HistoryBuffer hist_{20};

  int T_ini_{20}; // length of past horizon (should match T_past used in data_processing)
  Vector7d k_gains_{Vector7d::Zero()};
  Vector7d d_gains_{Vector7d::Zero()};
  Vector7d last_pred_{Vector7d::Zero()};

  // Hyperparams
  double alpha_close{1.5};
  double alpha_far{2.5};
  double r0{0.55};
  double beta{0.5};

  // Constants
  const int num_joints = 7;
  int N_pred_{10}; // prediction horizon N (should match T_future used in data_processing)

  // States
  enum class Phase { WARMUP, TRACKING };
  Phase phase_{Phase::WARMUP};

  Vector7d q_{Vector7d::Zero()};
  Vector7d dq_{Vector7d::Zero()};
  Vector7d dq_filtered_{Vector7d::Zero()};
  Vector7d initial_q_{Vector7d::Zero()};
  double elapsed_time_{0.0};
  Vector7d prev_tau_applied_{Vector7d::Zero()}; // u_{k-1}
  Vector7d last_tau_imp_{Vector7d::Zero()};

  // RobotState subscriber
  rclcpp::Subscription<franka_msgs::msg::FrankaRobotState>::SharedPtr state_sub_;
  std::atomic<bool> have_tau_ext_{false};
  std::array<double, 7> tau_ext_last_{{0, 0, 0, 0, 0, 0, 0}}; // last received tau_ext
  std::array<double, 16> o_t_ee_last_{{0.0}}; // last end-effector pose
  std::atomic<bool> have_ee_pose_{false};
  Vector3d ee_pos_last_{Vector3d::Zero()};

  double time_since_friction_prediction_{0.0};

  // Robot model names
  const std::string state_interface_name_{"robot_state"};
  const std::string k_robot_state_interface_name{"robot_state"};
  const std::string k_robot_model_interface_name{"robot_model"};
  std::string robot_name_{"fr3"};
  
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

  // Cartesian impedance params
  std::array<double, 7> q_subscribed{};
  std::array<double, 7> tau_J_d_{0,0,0,0,0,0,0};
  std::array<double, 6> O_F_ext_hat_K_{0,0,0,0,0,0};
  Eigen::Matrix<double, 7, 1> q_subscribed_M_;
  Eigen::Matrix<double, 7, 1> tau_J_d_M_{Eigen::Matrix<double,7,1>::Zero()};
  Eigen::Matrix<double, 7, 1> tau_ff_des_ = Eigen::Matrix<double,7,1>::Zero();
  Eigen::Matrix<double, 7, 1> tau_ff_prev_{Eigen::Matrix<double,7,1>::Zero()};
  Eigen::Matrix<double, 6, 1> O_F_ext_hat_K_M_{Eigen::Matrix<double,6,1>::Zero()};
  Eigen::Matrix<double, 7, 1> q_cart_{Eigen::Matrix<double,7,1>::Zero()};
  Eigen::Matrix<double, 7, 1> dq_cart_{Eigen::Matrix<double,7,1>::Zero()};
  Eigen::MatrixXd jacobian_transpose_pinv_;
  Eigen::Matrix<double,7,1> dq_task_;
  
  const std::string state_interface_name_cart_{"robot_state"}; 
  
  // from your header:
  const double delta_tau_max_cart_{15.0};
  const double delta_tau_max_cart_ff_{15.0};
  const double dt_cart_{0.001};
  
  // Impedance control matrices
  Eigen::Matrix<double, 6, 6> Lambda_ =
      Eigen::Matrix<double,6,6>::Identity(); // operational space mass
  Eigen::Matrix<double, 6, 6> Sm_ =
      Eigen::Matrix<double,6,6>::Identity(); // position+rotation selection
  Eigen::Matrix<double, 6, 6> Sf_ =
      Eigen::Matrix<double,6,6>::Zero(); // force selection
  
  Eigen::Matrix<double, 6, 6> K_ = (Eigen::MatrixXd(6,6) <<
    3000,   0,   0,   0,   0,   0,
      0, 3000,   0,   0,   0,   0,
      0,   0, 3000,   0,   0,   0,
      0,   0,   0, 100,   0,   0,
      0,   0,   0,   0, 100,   0,
      0,   0,   0,   0,   0,  100).finished();
  
  Eigen::Matrix<double, 6, 6> D_ = (Eigen::MatrixXd(6,6) <<
     35,   0,   0,   0,   0,   0,
      0,  35,   0,   0,   0,   0,
      0,   0,  35,   0,   0,   0,
      0,   0,   0,  25,   0,   0,
      0,   0,   0,   0,  25,   0,
      0,   0,   0,   0,   0,   6).finished();
  
  Eigen::Matrix<double, 6, 6> Theta_ =
      Eigen::Matrix<double,6,6>::Identity();
  Eigen::Matrix<double, 6, 6> T_ = (Eigen::MatrixXd(6,6) <<
    1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 0, 2.5, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 2.5).finished();
  
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  Eigen::Matrix<double, 6, 6> cartesian_inertia_target_;
  
  Eigen::Vector3d position_d_target_{0.5, 0.0, 0.5};
  Eigen::Vector3d rotation_d_target_{M_PI, 0.0, 0.0};
  Eigen::Quaterniond orientation_d_target_;
  
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  
  Eigen::Matrix<double, 6, 1> F_impedance_;
  Eigen::Matrix<double, 6, 1> F_contact_des_ =
      Eigen::Matrix<double,6,1>::Zero();
  Eigen::Matrix<double, 6, 1> F_contact_target_ =
      Eigen::Matrix<double,6,1>::Zero();
  Eigen::Matrix<double, 6, 1> F_ext_ =
      Eigen::Matrix<double,6,1>::Zero();
  Eigen::Matrix<double, 6, 1> F_cmd_ =
      Eigen::Matrix<double,6,1>::Zero();
  
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;
  
  Eigen::Matrix<double, 6, 1> error_;
  double nullspace_stiffness_{0.001};
  double nullspace_stiffness_target_{0.001};
  double D_gain_{2.05};
  
  // Logging counters
  int outcounter_{0};
  const int update_frequency_{2}; // frequency for prints
  
  // Integrators:
  Eigen::Matrix<double, 6, 1> I_error_ =
      Eigen::Matrix<double,6,1>::Zero();
  Eigen::Matrix<double, 6, 1> I_F_error_ =
      Eigen::Matrix<double,6,1>::Zero();
  Eigen::Matrix<double, 6, 1> integrator_weights_ =
    (Eigen::MatrixXd(6,1) << 75.0, 75.0, 75.0, 75.0, 75.0, 4.0).finished();
  Eigen::Matrix<double, 6, 1> max_I_ =
    (Eigen::MatrixXd(6,1) << 30.0, 30.0, 30.0, 50.0, 50.0, 2.0).finished();
  
  std::mutex position_and_orientation_d_target_mutex_;
  
  bool config_control_{false};
  bool do_logging_{false};
  
  double filter_params_{0.001};
  int mode_cart_{1};

  // Helpers
  void updateJointStates();
  void update_stiffness_and_references();
  void arrayToMatrix(const std::array<double, 6>& inputArray, Eigen::Matrix<double, 6, 1>& resultMatrix);
  void arrayToMatrix(const std::array<double, 7>& inputArray, Eigen::Matrix<double, 7, 1>& resultMatrix);
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d);
  Eigen::Matrix<double, 7, 1> saturateTorqueRateFF(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d);
  std::array<double, 6> convertToStdArray(const geometry_msgs::msg::WrenchStamped& wrench);

  // warmup & histories
  void resetHistories();
  bool warmupStep(const Vector7d& tau_ext);
  void pushUHistory(const Vector7d& u_curr);
  void pushYHistory(const Vector7d& y_next);
};
}  // namespace nonlinear_deepc_controller
