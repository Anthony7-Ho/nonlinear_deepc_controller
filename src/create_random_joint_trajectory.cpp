#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/collision_detection/collision_common.h>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <franka_msgs/msg/franka_robot_state.hpp>

#include <random>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <cstdlib>

// Constants
static const std::string GROUP_NAME = "fr3_arm";
static const std::string BASE_FRAME = "fr3_link0";
static const std::string EEF_LINK = "fr3_hand_tcp";
static const int N_WAYPOINTS = 30; // Increased number of waypoints for better joint exploration
static const double DWELL_SEC = 0.5; // Dwell time at each waypoint slightly shorter

// Box in BASE_FRAME where random poses are sampled:
static const double X_MIN = 0.30, X_MAX = 0.70;
static const double Y_MIN = -0.40, Y_MAX = 0.40;
static const double Z_MIN = 0.25, Z_MAX = 0.70;

// Velocity and acceleration scaling for MoveIt
static const double VEL_MIN = 0.05;
static const double VEL_MAX = 0.5; // Much wider velocity range
static const double ACC_SCALE = 0.80;

static const std::string CSV_OUT = std::string(std::getenv("HOME")) + "/trajectory_joint_train.csv";

// Convert builtin Duration + rclcpp::Duration -> builtin Duration
inline builtin_interfaces::msg::Duration add_duration(
    const builtin_interfaces::msg::Duration &a,
    const rclcpp::Duration &b)
{
  const int64_t ns = rclcpp::Duration(a).nanoseconds() + b.nanoseconds();
  builtin_interfaces::msg::Duration out;
  out.sec = static_cast<int32_t>(ns / 1000000000LL);
  out.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  return out;
}

// Append source (src) trajectory into destination (dst), shifting src by t_offset = last(dst).time_from_start
static void append_trajectory(moveit_msgs::msg::RobotTrajectory &dst,
                              const moveit_msgs::msg::RobotTrajectory &src,
                              bool skip_first_src_point = true)
{
  const auto &src_jt = src.joint_trajectory;
  if (src_jt.points.empty()) return;

  if (dst.joint_trajectory.joint_names.empty()) dst.joint_trajectory.joint_names = src_jt.joint_names;

  // 1 ms gap to guarantee strictly increasing times at the seam
  constexpr double EPS_GAP = 1e-3;

  rclcpp::Duration t_offset(0, 0);
  if (!dst.joint_trajectory.points.empty()){
    t_offset = rclcpp::Duration(dst.joint_trajectory.points.back().time_from_start) + rclcpp::Duration::from_seconds(EPS_GAP);
  }

  const bool skip = skip_first_src_point && !dst.joint_trajectory.points.empty();
  const std::size_t start_idx = skip ? 1 : 0;

  for (std::size_t i = start_idx; i < src_jt.points.size(); ++i) {
    auto pt = src_jt.points[i];
    if (t_offset > rclcpp::Duration(0, 0)) {
      pt.time_from_start = add_duration(pt.time_from_start, t_offset);
    }
    dst.joint_trajectory.points.push_back(std::move(pt));
  }
}

// Append a dwell segment after a motion (keep same q, zero dq)
static void append_dwell(moveit_msgs::msg::RobotTrajectory &traj, double dwell_sec)
{
  if (traj.joint_trajectory.points.empty()) return;

  const auto &last = traj.joint_trajectory.points.back();
  const size_t nq = last.positions.size();

  moveit_msgs::msg::RobotTrajectory hold;
  hold.joint_trajectory.joint_names = traj.joint_trajectory.joint_names;

  trajectory_msgs::msg::JointTrajectoryPoint p;
  p.positions = last.positions;
  p.velocities.assign(nq, 0.0);
  p.accelerations.assign(nq, 0.0);

  // dwell
  p.time_from_start.sec = static_cast<int32_t>(std::floor(dwell_sec));
  p.time_from_start.nanosec = static_cast<uint32_t>((dwell_sec - std::floor(dwell_sec)) * 1e9);

  hold.joint_trajectory.points.push_back(p);

  // Don't skip the first source point when appending a dwell
  append_trajectory(traj, hold, /*skip_first_src_point=*/false);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("create_random_joint_trajectory");
  auto logger = node->get_logger();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() {
    executor.spin();
  });

  // Publisher for RViz display
  auto disp_pub = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
      "/display_planned_path", rclcpp::QoS(1));

  // MoveGroup interface
  moveit::planning_interface::MoveGroupInterface move_group(node, GROUP_NAME);
  auto robot_model = move_group.getRobotModel();
  planning_scene::PlanningScene planning_scene(robot_model);
  move_group.setPoseReferenceFrame(BASE_FRAME);
  move_group.setPlanningPipelineId("ompl");
  move_group.setPlannerId("RRTConnectkConfigDefault");
  move_group.setPlanningTime(5.0);
  move_group.setNumPlanningAttempts(3);
  move_group.setMaxAccelerationScalingFactor(ACC_SCALE);

  RCLCPP_INFO(logger, "Planning with group='%s', base='%s' in JOINT SPACE",
              GROUP_NAME.c_str(), BASE_FRAME.c_str());

  move_group.setStartStateToCurrentState();

  // Random generator for per-segment velocity scaling
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> vel_dist(VEL_MIN, VEL_MAX);

  moveit_msgs::msg::RobotTrajectory combined;
  int planned_and_executed = 0;

  // Track the start state for each next segment
  auto start_state_ptr = move_group.getCurrentState(1.0);
  moveit::core::RobotState last_state = start_state_ptr ? *start_state_ptr : moveit::core::RobotState(robot_model);
  if (!start_state_ptr) last_state.setToDefaultValues();
  const moveit::core::JointModelGroup* jmg = robot_model->getJointModelGroup(GROUP_NAME);

  // Plan all segments first, then execute once
  for (size_t i = 0; i < N_WAYPOINTS; ++i) {
    const double vel_scale = vel_dist(gen);
    move_group.setMaxVelocityScalingFactor(vel_scale);
    RCLCPP_INFO(logger, "Segment %zu: velocity scaling = %.3f", i, vel_scale);

    move_group.setStartStateToCurrentState();

    const std::size_t MAX_SAMPLING_RETRIES = 5000;
    bool found_valid_target = false;
    moveit::core::RobotState target_state = last_state;

    // Sample random joint states and check self collision
    for (std::size_t tries = 0; tries < MAX_SAMPLING_RETRIES; ++tries) {
      target_state.setToRandomPositions(jmg);
      target_state.update(); // Update to calculate forward kinematics

      const Eigen::Isometry3d& eef_transform = target_state.getGlobalLinkTransform(EEF_LINK);
      double x = eef_transform.translation().x();
      double y = eef_transform.translation().y();
      double z = eef_transform.translation().z();

      if (x < X_MIN || x > X_MAX || y < Y_MIN || y > Y_MAX || z < Z_MIN || z > Z_MAX) {
        continue;
      }
      
      collision_detection::CollisionRequest req;
      collision_detection::CollisionResult res;
      planning_scene.checkSelfCollision(req, res, target_state);

      if (!res.collision) {
        found_valid_target = true;
        break;
      }
    }

    if (!found_valid_target) {
      RCLCPP_WARN(logger, "Segment %zu: Failed to sample valid joint state after %zu retries; skipping segment.", i, MAX_SAMPLING_RETRIES);
      move_group.clearPoseTargets();
      continue;
    }

    // Set the randomly found joint state as the target
    move_group.setJointValueTarget(target_state);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool ok = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (!ok || plan.trajectory_.joint_trajectory.points.empty()) {
      RCLCPP_WARN(logger, "Segment %zu: planning failed.", i);
      continue;
    }

    // motion ends with zero velocity/acceleration
    auto &pts = plan.trajectory_.joint_trajectory.points;
    if (!pts.empty()) {
      auto &last = pts.back();
      std::fill(last.velocities.begin(), last.velocities.end(), 0.0);
      std::fill(last.accelerations.begin(), last.accelerations.end(), 0.0);
    }

    // Add a dwell
    append_dwell(plan.trajectory_, DWELL_SEC);

    // Visualize
    moveit_msgs::msg::DisplayTrajectory disp_msg;
    disp_msg.model_id = robot_model->getName();
    moveit::core::robotStateToRobotStateMsg(last_state, disp_msg.trajectory_start);
    disp_msg.trajectory.push_back(plan.trajectory_);
    disp_pub->publish(disp_msg);

    // Execute each segment
    const bool exec_ok = (move_group.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!exec_ok)
      RCLCPP_ERROR(logger, "Segment %zu: execution failed!", i);
    else
      RCLCPP_INFO(logger, "Segment %zu: execution succeeded.", i);

    const auto &pts_after = plan.trajectory_.joint_trajectory.points;
    if (!pts_after.empty()) {
      const auto &q_end = pts_after.back().positions;
      last_state.setJointGroupPositions(jmg, q_end);
      last_state.update();
    }

    // Stitch into combined trajectory
    append_trajectory(combined, plan.trajectory_, true);
    ++planned_and_executed;

    move_group.clearPoseTargets();
  }

  if (combined.joint_trajectory.points.empty()) {
    RCLCPP_ERROR(logger, "No successful segments executed. Try adjusting bounds or planning settings.");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(logger, "Combined executed trajectory has %zu points from %d segments.", combined.joint_trajectory.points.size(), planned_and_executed);

  // Publish combined trajectory to RViz
  moveit_msgs::msg::DisplayTrajectory disp_msg;
  disp_msg.model_id = robot_model->getName();

  if (auto current_ptr = move_group.getCurrentState(0.5)) {
    moveit::core::robotStateToRobotStateMsg(*current_ptr, disp_msg.trajectory_start);
  }

  disp_msg.trajectory.push_back(combined);
  disp_pub->publish(disp_msg);
  RCLCPP_INFO(logger, "Published combined path to /display_planned_path");

  // Save trajectory to CSV
  try {
    const auto &jt = combined.joint_trajectory;
    const size_t nq = jt.joint_names.size();

    std::ofstream ofs(CSV_OUT);
    ofs << std::fixed << std::setprecision(9);

    ofs << "time_s";
    for (size_t j = 0; j < nq; ++j) ofs << ",q_" << j;
    for (size_t j = 0; j < nq; ++j) ofs << ",dq_" << j;
    ofs << "\n";

    for (const auto &pt : jt.points) {
      const double t = static_cast<double>(pt.time_from_start.sec) + static_cast<double>(pt.time_from_start.nanosec) * 1e-9;

      ofs << std::setprecision(9) << t << std::setprecision(9);

      if (!pt.positions.empty()){
        for (auto v : pt.positions) {
          ofs << "," << v;
        }
      }
      else {
        for (size_t j = 0; j < nq; ++j) {
          ofs << "," << std::numeric_limits<double>::quiet_NaN();
        }
      }

      if (!pt.velocities.empty()) {
        for (auto v : pt.velocities) {
          ofs << "," << v;
        }
      }
      else {
        for (size_t j = 0; j < nq; ++j) {
          ofs << "," << std::numeric_limits<double>::quiet_NaN();
        }
      }
      ofs << "\n";
    }
    RCLCPP_INFO(logger, "Saved trajectory to %s", CSV_OUT.c_str());
  } catch (const std::exception &e) {
    RCLCPP_ERROR(logger, "Failed to save CSV: %s", e.what());
  }

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}