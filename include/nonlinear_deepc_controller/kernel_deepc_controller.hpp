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

  // Kernel bundle directory
  std::string kernel_bundle_dir_param_;
  std::string kernel_bundle_dir_;

  int T_ini_{20}; // length of past horizon (should match T_past used in data_processing)
  Vector7d k_gains_{Vector7d::Zero()};
  Vector7d d_gains_{Vector7d::Zero()};
  double ff_gain_{-1.0};

  // decay parameters (for blending friction compensation)
  double alpha_max_{1.0};
  double alpha_decay_seconds_{0.05};

  // Constants
  const int num_joints = 7;
  int N_pred_{10}; // prediction horizon N (should match T_future used in data_processing)
  bool first_update_{true};


  // States
  enum class Phase { WARMUP, TRACKING };
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

  // Histories (fixed size T_ini)
  std::deque<Vector7d> u_hist_; // applied torque history
  std::deque<Vector7d> y_hist_; // measured tau_ext history
  Vector7d prev_tau_applied_{Vector7d::Zero()}; // u_{k-1}
  Vector7d last_tau_imp_{Vector7d::Zero()};
  Vector7d last_tau_cmd_{Vector7d::Zero()};

  // RobotState subscriber
  rclcpp::Subscription<franka_msgs::msg::FrankaRobotState>::SharedPtr state_sub_;
  std::atomic<bool> have_tau_ext_{false};
  std::array<double, 7> tau_ext_last_{{0, 0, 0, 0, 0, 0, 0}};

  // Logging
  std::vector<Vector7d> tau_ext_hist_;
  std::vector<Vector7d> friction_pred_hist_;
  std::vector<Vector7d> tau_residual_hist_;
  std::vector<Vector7d> q_des_hist_;
  std::vector<Vector7d> q_curr_hist_;
  std::vector<double> t_hist_;

  int  log_decimation_{1};
  int  log_counter_{0};
  bool stop_logging_at_end_{true};
  double post_log_window_{0.0};
  bool logging_active_{true};

  double time_since_friction_prediction_{0.0};

  // Kernel DeePC closed-form model (per joint)
  bool kernel_loaded_{false};

  // Shared dimensions
  int Hc_{100}; // dimesnion of Kg (Hc x Hc, number of cluster centers from data_processing)
  int d_full_{50}; // length of stacked regressor [u_ini; y_ini; u_future] -Y (T_past + T_past + T_future)

  // For each joint j:
  // Kg_list_[j]: (Hc_ x Hc_) -> Kg = K + gamma_reg*I
  // X_list_[j]: (d_full_ x Hc_) = [Hu_past; Hy_past; Hu_future]
  // Hy_future_list_[j]: (N_pred_ x Hc_)
  // X_col_norm2_list_[j]: precomputed ||x_i||^2 for each column of X_list_[j]
  // A_chol_list_[j]: LLT of A_j = lambda_g I + lambda_k Kg^T Kg
  std::array<Eigen::MatrixXd, 7> Kg_list_;
  std::array<Eigen::MatrixXd, 7> X_list_;
  std::array<Eigen::MatrixXd, 7> Hy_future_list_;

  std::array<Eigen::VectorXd, 7> X_col_norm2_list_;
  std::array<Eigen::LLT<Eigen::MatrixXd>, 7> A_chol_list_;

  // Hyperparameters (shared across joints, loaded from meta.json
  double rbf_scale_{0.0}; // RBF gamma (best_gamma_rbf)
  double lambda_g_{0.0}; // regularization for g
  double lambda_k_{0.0}; // regularization for kernel

  // Helpers
  void updateJointStates();
  bool loadCsvTrajectory(const std::string& path);
  void writeLogCsv(const std::string& path) const;
  Vector7d interp(const std::vector<Vector7d>& data, double t) const;

  // warmup & histories
  void resetHistories();
  bool warmupStep(const Vector7d& tau_ext);
  void pushUHistory(const Vector7d& u_curr);
  void pushYHistory(const Vector7d& y_next);

  // Kernel bundle IO (loads all 7 joint_* folders under kernel_bundle_dir_)
  bool loadKernelBundle(const std::string& dir);
  bool loadBinMatrix(const std::string& path, int rows, int cols, Eigen::MatrixXd& M);

  // Closed-form friction prediction
  Vector7d computeFrictionPrediction();
};
}  // namespace nonlinear_deepc_controller
