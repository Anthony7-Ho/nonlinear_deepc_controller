#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/collision_detection/collision_common.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <builtin_interfaces/msg/duration.hpp>

#include <random>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <cstdlib>

// Constants
static const std::string GROUP_NAME = "fr3_arm";
static const std::string EEF_LINK = "fr3_hand_tcp";
static const std::string BASE_FRAME = "fr3_link0";
static const int N_WAYPOINTS = 12; // number of random waypoints to generate
static const double DWELL_SEC = 0.30; // seconds to dwell at each waypoint


// Box in BASE_FRAME where random poses are sampled:
static const double X_MIN = 0.30, X_MAX = 0.70;
static const double Y_MIN = -0.40, Y_MAX = 0.40;
static const double Z_MIN = 0.40, Z_MAX = 0.70;
// Velocity and acceleration scaling for MoveIt
static const double VEL_MIN = 0.05;
static const double VEL_MAX = 0.15;
static const double ACC_SCALE = 0.10;

static const std::string CSV_OUT =
    std::string(std::getenv("HOME")) + "/trajectory_export.csv";

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


// Generate a random pose within the defined box and with random orientation
static geometry_msgs::msg::PoseStamped random_pose()
{
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<double> dx(X_MIN, X_MAX);
  static std::uniform_real_distribution<double> dy(Y_MIN, Y_MAX);
  static std::uniform_real_distribution<double> dz(Z_MIN, Z_MAX);
  static std::uniform_real_distribution<double> u01(0.0, 1.0);
  static std::uniform_real_distribution<double> angle(-M_PI, M_PI);

  geometry_msgs::msg::PoseStamped p;
  p.header.frame_id = BASE_FRAME;

  // Random position in box
  p.pose.position.x = dx(gen);
  p.pose.position.y = dy(gen);
  p.pose.position.z = dz(gen);

  // Random orientation (uniform on SO(3) via random quaternion)
  double u1 = u01(gen);
  double u2 = u01(gen);
  double u3 = u01(gen);
  double sqrt1_minus_u1 = std::sqrt(1 - u1);
  double sqrt_u1 = std::sqrt(u1);

  p.pose.orientation.x = sqrt1_minus_u1 * std::sin(2 * M_PI * u2);
  p.pose.orientation.y = sqrt1_minus_u1 * std::cos(2 * M_PI * u2);
  p.pose.orientation.z = sqrt_u1 * std::sin(2 * M_PI * u3);
  p.pose.orientation.w = sqrt_u1 * std::cos(2 * M_PI * u3);

  return p;
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
  auto node = rclcpp::Node::make_shared("create_random_trajectory");
  auto logger = node->get_logger();

  // Publisher for RViz display
  auto disp_pub = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
      "/display_planned_path", rclcpp::QoS(1));

  // MoveGroup interface
  moveit::planning_interface::MoveGroupInterface move_group(node, GROUP_NAME);
  auto robot_model = move_group.getRobotModel();
  planning_scene::PlanningScene planning_scene(robot_model);
  move_group.setPoseReferenceFrame(BASE_FRAME);
  move_group.setEndEffectorLink(EEF_LINK);
  move_group.setPlanningPipelineId("ompl");
  move_group.setPlannerId("RRTConnectkConfigDefault");
  move_group.setPlanningTime(5.0);
  move_group.setNumPlanningAttempts(3);
  move_group.setMaxAccelerationScalingFactor(ACC_SCALE);

  RCLCPP_INFO(logger, "Planning with group='%s', eef='%s', base='%s'",
              GROUP_NAME.c_str(), EEF_LINK.c_str(), BASE_FRAME.c_str());

  move_group.setStartStateToCurrentState();

  // Random generator for per-segment velocity scaling
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> vel_dist(VEL_MIN, VEL_MAX);

  // Generate random target poses
  std::vector<geometry_msgs::msg::PoseStamped> waypoints;
  waypoints.reserve(N_WAYPOINTS);
  for (int i = 0; i < N_WAYPOINTS; ++i) {
    waypoints.push_back(random_pose());
  }

  RCLCPP_INFO(logger, "Generated %d random waypoints.", (int)waypoints.size());

  moveit_msgs::msg::RobotTrajectory combined;
  int planned_and_executed = 0;

  // Track the start state for each next segment
  auto start_state_ptr = move_group.getCurrentState(1.0);
  moveit::core::RobotState last_state = start_state_ptr ? *start_state_ptr : moveit::core::RobotState(move_group.getRobotModel());
  if (!start_state_ptr) last_state.setToDefaultValues();
  const moveit::core::JointModelGroup* jmg = move_group.getRobotModel()->getJointModelGroup(GROUP_NAME);

  // Plan all segments first, then execute once
  for (size_t i = 0; i < waypoints.size(); ++i) {
    const double vel_scale = vel_dist(gen);
    move_group.setMaxVelocityScalingFactor(vel_scale);
    RCLCPP_INFO(logger, "Segment %zu: velocity scaling = %.3f", i, vel_scale);

    move_group.setStartState(last_state);

    const std::size_t MAX_IK_RETRIES = 25;

    geometry_msgs::msg::PoseStamped target = waypoints[i]; // start from the pre-sampled pose
    bool ik_ok = false;
    // Try multiple times to get a valid IK + collision-free pose
    for (std::size_t tries = 0; tries < MAX_IK_RETRIES; ++tries) {
      moveit::core::RobotState probe = last_state;
      ik_ok = probe.setFromIK(jmg, target.pose, EEF_LINK, 0.05);
      if (!ik_ok) { target = random_pose(); continue; }

      collision_detection::CollisionRequest req;
      collision_detection::CollisionResult res;
      planning_scene.checkSelfCollision(req, res, probe);

      if (!res.collision) {
        waypoints[i] = target;
        break;
      }
      // colliding: resample
      target = random_pose();
    }

    if (!ik_ok) {
      RCLCPP_WARN(logger, "Segment %zu: IK failed after %zu retries; skipping segment.", i, MAX_IK_RETRIES);
      move_group.clearPoseTargets();
      continue;  // skip if all retries failed
    }

    move_group.setPoseTarget(waypoints[i].pose, EEF_LINK);

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

  // Execute the combined trajectory once at the end
  if (!combined.joint_trajectory.points.empty()) {
    RCLCPP_INFO(logger, "Executing combined trajectory with %zu points...", combined.joint_trajectory.points.size());

    // Wrap combined trajectory as a MoveGroup plan
    moveit::planning_interface::MoveGroupInterface::Plan big_plan;
    big_plan.trajectory_ = combined;

    // Visualize before execution
    moveit_msgs::msg::DisplayTrajectory disp_msg;
    disp_msg.model_id = move_group.getRobotModel()->getName();
    if (auto current_ptr = move_group.getCurrentState(0.5))
      moveit::core::robotStateToRobotStateMsg(*current_ptr, disp_msg.trajectory_start);
    disp_msg.trajectory.push_back(combined);
    disp_pub->publish(disp_msg);
    RCLCPP_INFO(logger, "Published combined trajectory preview to /display_planned_path");

    // Execute the entire path once
    const bool exec_ok = (move_group.execute(big_plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (exec_ok)
      RCLCPP_INFO(logger, "Execution complete!");
    else
      RCLCPP_ERROR(logger, "Execution failed!");
  }


  if (combined.joint_trajectory.points.empty()) {
    RCLCPP_ERROR(logger, "No successful segments executed. Try adjusting bounds or planning settings.");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(logger, "Combined executed trajectory has %zu points from %d segments.", combined.joint_trajectory.points.size(), planned_and_executed);

  // Publish combined trajectory to RViz
  moveit_msgs::msg::DisplayTrajectory disp_msg;
  disp_msg.model_id = move_group.getRobotModel()->getName();

  if (auto current_ptr = move_group.getCurrentState(0.5)) {
    moveit::core::robotStateToRobotStateMsg(*current_ptr, disp_msg.trajectory_start);
  }

  disp_msg.trajectory.push_back(combined);
  disp_pub->publish(disp_msg);
  RCLCPP_INFO(logger, "Published combined path to /display_planned_path");

  // Save to CSV
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

    ofs.close();
    RCLCPP_INFO(logger, "Wrote CSV: %s (%zu rows)", CSV_OUT.c_str(), jt.points.size());
  } catch (const std::exception &e) {
    RCLCPP_ERROR(logger, "CSV write failed: %s", e.what());
  }

  // Post-process CSV to remove duplicate time_s rows
  try {
    std::ifstream ifs(CSV_OUT);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to reopen CSV for deduplication");
    }

    std::vector<std::string> lines;
    lines.reserve(1000);
    std::string line;
    std::getline(ifs, line);  // header
    const std::string header = line;
    lines.push_back(header);

    std::unordered_set<std::string> seen_times;
    while (std::getline(ifs, line)) {
      if (line.empty()) continue;
      std::istringstream ss(line);
      std::string time_str;
      std::getline(ss, time_str, ',');
      if (seen_times.insert(time_str).second) {
        // first time seeing this timestamp
        lines.push_back(line);
      }
    }
    ifs.close();

    std::ofstream ofs(CSV_OUT, std::ios::trunc);
    for (const auto &l : lines) {
      ofs << l << "\n";
    }
    ofs.close();
    RCLCPP_INFO(logger, "Removed duplicate timestamps. Final rows: %zu", lines.size() - 1);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(logger, "CSV post-processing failed: %s", e.what());
  }


  rclcpp::shutdown();
  return 0;
}
