#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/collision_detection/collision_common.h>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct SharedSamplingConfig {
  std::string base_frame = "fr3_link0";
  std::string eef_link = "fr3_hand_tcp";
  double x_min = 0.30;
  double x_max = 0.70;
  double y_min = -0.40;
  double y_max = 0.40;
  double z_min = 0.40;
  double z_max = 0.70;
  double table_size_x = 2.0;
  double table_size_y = 2.0;
  double table_thickness = 0.08;
};

struct JointRandomConfig {
  std::string group_name = "fr3_arm";
  double dwell_sec = 0.5;
  double vel_min = 0.05;
  double vel_max = 0.50;
  double acc_scale = 0.80;
  std::size_t max_sampling_retries = 5000;
  int train_waypoints = 30;
  int validation_waypoints = 15;
  std::string output_prefix = "trajectory_joint";
};

template <typename T>
static void set_if_present(const nlohmann::json &j, const char *key, T &target)
{
  if (j.contains(key) && !j.at(key).is_null()) {
    target = j.at(key).get<T>();
  }
}

static bool load_config(const std::string &path,
                        SharedSamplingConfig &shared,
                        JointRandomConfig &joint,
                        const rclcpp::Logger &logger)
{
  try {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
      RCLCPP_WARN(logger, "Could not open config file: %s. Using built-in defaults.", path.c_str());
      return false;
    }

    nlohmann::json root;
    ifs >> root;

    if (root.contains("shared") && root["shared"].is_object()) {
      const auto &s = root["shared"];
      set_if_present(s, "base_frame", shared.base_frame);
      set_if_present(s, "eef_link", shared.eef_link);

      if (s.contains("sampling_box") && s["sampling_box"].is_object()) {
        const auto &b = s["sampling_box"];
        set_if_present(b, "x_min", shared.x_min);
        set_if_present(b, "x_max", shared.x_max);
        set_if_present(b, "y_min", shared.y_min);
        set_if_present(b, "y_max", shared.y_max);
        set_if_present(b, "z_min", shared.z_min);
        set_if_present(b, "z_max", shared.z_max);
      }

      if (s.contains("table") && s["table"].is_object()) {
        const auto &t = s["table"];
        set_if_present(t, "size_x", shared.table_size_x);
        set_if_present(t, "size_y", shared.table_size_y);
        set_if_present(t, "thickness", shared.table_thickness);
      }
    }

    if (root.contains("joint_random") && root["joint_random"].is_object()) {
      const auto &j = root["joint_random"];
      set_if_present(j, "group_name", joint.group_name);
      set_if_present(j, "dwell_sec", joint.dwell_sec);
      set_if_present(j, "vel_min", joint.vel_min);
      set_if_present(j, "vel_max", joint.vel_max);
      set_if_present(j, "acc_scale", joint.acc_scale);
      set_if_present(j, "max_sampling_retries", joint.max_sampling_retries);
      set_if_present(j, "train_waypoints", joint.train_waypoints);
      set_if_present(j, "validation_waypoints", joint.validation_waypoints);
      set_if_present(j, "output_prefix", joint.output_prefix);
    }

    RCLCPP_INFO(logger, "Loaded trajectory config from %s", path.c_str());
    return true;
  } catch (const std::exception &e) {
    RCLCPP_WARN(logger, "Failed to parse config file %s (%s). Using built-in defaults.", path.c_str(), e.what());
    return false;
  }
}

// Convert builtin Duration + rclcpp::Duration -> builtin Duration
static builtin_interfaces::msg::Duration add_duration(
    const builtin_interfaces::msg::Duration &a,
    const rclcpp::Duration &b)
{
  const int64_t ns = rclcpp::Duration(a).nanoseconds() + b.nanoseconds();
  builtin_interfaces::msg::Duration out;
  out.sec = static_cast<int32_t>(ns / 1000000000LL);
  out.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  return out;
}

static bool is_in_sampling_box(const Eigen::Vector3d &p, const SharedSamplingConfig &cfg)
{
  return p.x() >= cfg.x_min && p.x() <= cfg.x_max &&
         p.y() >= cfg.y_min && p.y() <= cfg.y_max &&
         p.z() >= cfg.z_min && p.z() <= cfg.z_max;
}

static bool is_state_in_sampling_box(const moveit::core::RobotState &state,
                                     const std::string &eef_link,
                                     const SharedSamplingConfig &cfg)
{
  const auto &eef_tf = state.getGlobalLinkTransform(eef_link);
  return is_in_sampling_box(eef_tf.translation(), cfg);
}

static bool trajectory_stays_in_sampling_box(
    const moveit_msgs::msg::RobotTrajectory &traj,
    const moveit::core::RobotModelConstPtr &robot_model,
    const moveit::core::JointModelGroup *jmg,
    const std::string &eef_link,
    const SharedSamplingConfig &cfg)
{
  if (!jmg) {
    return false;
  }

  moveit::core::RobotState probe(robot_model);
  const std::size_t dof = jmg->getVariableCount();
  for (const auto &pt : traj.joint_trajectory.points) {
    if (pt.positions.size() != dof) {
      return false;
    }

    probe.setJointGroupPositions(jmg, pt.positions);
    probe.update();
    if (!is_state_in_sampling_box(probe, eef_link, cfg)) {
      return false;
    }
  }

  return true;
}

// Append source (src) trajectory into destination (dst), shifting src by t_offset = last(dst).time_from_start
static void append_trajectory(moveit_msgs::msg::RobotTrajectory &dst,
                              const moveit_msgs::msg::RobotTrajectory &src,
                              bool skip_first_src_point = true)
{
  const auto &src_jt = src.joint_trajectory;
  if (src_jt.points.empty()) {
    return;
  }

  if (dst.joint_trajectory.joint_names.empty()) {
    dst.joint_trajectory.joint_names = src_jt.joint_names;
  }

  constexpr double EPS_GAP = 1e-3;

  rclcpp::Duration t_offset(0, 0);
  if (!dst.joint_trajectory.points.empty()) {
    t_offset = rclcpp::Duration(dst.joint_trajectory.points.back().time_from_start) +
               rclcpp::Duration::from_seconds(EPS_GAP);
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
  if (traj.joint_trajectory.points.empty()) {
    return;
  }

  const auto &last = traj.joint_trajectory.points.back();
  const size_t nq = last.positions.size();

  moveit_msgs::msg::RobotTrajectory hold;
  hold.joint_trajectory.joint_names = traj.joint_trajectory.joint_names;

  trajectory_msgs::msg::JointTrajectoryPoint p;
  p.positions = last.positions;
  p.velocities.assign(nq, 0.0);
  p.accelerations.assign(nq, 0.0);

  p.time_from_start.sec = static_cast<int32_t>(std::floor(dwell_sec));
  p.time_from_start.nanosec =
      static_cast<uint32_t>((dwell_sec - std::floor(dwell_sec)) * 1e9);

  hold.joint_trajectory.points.push_back(p);
  append_trajectory(traj, hold, false);
}

static std::string normalize_mode(std::string mode)
{
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (mode == "val") {
    return "validation";
  }
  if (mode == "train" || mode == "validation") {
    return mode;
  }
  return "train";
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("create_random_joint_trajectory");
  auto logger = node->get_logger();

  std::string default_config_path;
  try {
    default_config_path =
        ament_index_cpp::get_package_share_directory("nonlinear_deepc_controller") +
        "/config/trajectory_generation_config.json";
  } catch (const std::exception &) {
    default_config_path = "trajectory_generation_config.json";
  }

  const auto config_path =
      node->declare_parameter<std::string>("trajectory_config_path", default_config_path);
  const auto raw_mode = node->declare_parameter<std::string>("mode", "train");
  const auto mode = normalize_mode(raw_mode);

  std::string raw_mode_lc = raw_mode;
  std::transform(raw_mode_lc.begin(), raw_mode_lc.end(), raw_mode_lc.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const bool raw_mode_supported =
      (raw_mode_lc == "train" || raw_mode_lc == "validation" || raw_mode_lc == "val");
  if (!raw_mode_supported) {
    RCLCPP_WARN(logger, "Unsupported mode '%s'. Falling back to 'train'.", raw_mode.c_str());
  }

  SharedSamplingConfig shared_cfg;
  JointRandomConfig joint_cfg;
  load_config(config_path, shared_cfg, joint_cfg, logger);

  const int n_waypoints =
      (mode == "train") ? joint_cfg.train_waypoints : joint_cfg.validation_waypoints;
  const std::string mode_suffix = (mode == "train") ? "train" : "val";

  const char *home = std::getenv("HOME");
  const std::string out_dir = home ? std::string(home) : std::string("/tmp");
  const std::string csv_out = out_dir + "/" + joint_cfg.output_prefix + "_" + mode_suffix + ".csv";

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() {
    executor.spin();
  });

  auto disp_pub = node->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
      "/display_planned_path", rclcpp::QoS(1));

  moveit::planning_interface::MoveGroupInterface move_group(node, joint_cfg.group_name);
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  auto robot_model = move_group.getRobotModel();
  planning_scene::PlanningScene planning_scene(robot_model);

  move_group.setEndEffectorLink(shared_cfg.eef_link);
  move_group.setPoseReferenceFrame(shared_cfg.base_frame);
  move_group.setPlanningPipelineId("ompl");
  move_group.setPlannerId("RRTConnectkConfigDefault");
  move_group.setPlanningTime(5.0);
  move_group.setNumPlanningAttempts(3);
  move_group.setMaxAccelerationScalingFactor(joint_cfg.acc_scale);

  RCLCPP_INFO(logger,
              "Mode=%s, waypoints=%d, group=%s, base=%s, output=%s",
              mode.c_str(), n_waypoints, joint_cfg.group_name.c_str(),
              shared_cfg.base_frame.c_str(), csv_out.c_str());

  std::string active_eef_link = move_group.getEndEffectorLink();
  if (active_eef_link.empty()) {
    active_eef_link = shared_cfg.eef_link;
  }

  if (!robot_model->hasLinkModel(active_eef_link)) {
    RCLCPP_ERROR(logger, "End-effector link '%s' not found in robot model.",
                 active_eef_link.c_str());
    rclcpp::shutdown();
    return 1;
  }

  moveit_msgs::msg::CollisionObject table;
  table.header.frame_id = shared_cfg.base_frame;
  table.id = "safety_table";

  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = {
      shared_cfg.table_size_x,
      shared_cfg.table_size_y,
      shared_cfg.table_thickness,
  };

  geometry_msgs::msg::Pose table_pose;
  table_pose.orientation.w = 1.0;
  table_pose.position.x = 0.0;
  table_pose.position.y = 0.0;
  table_pose.position.z = -0.5 * shared_cfg.table_thickness;

  table.primitives.push_back(primitive);
  table.primitive_poses.push_back(table_pose);
  table.operation = moveit_msgs::msg::CollisionObject::ADD;

  if (!planning_scene_interface.applyCollisionObject(table)) {
    RCLCPP_WARN(logger, "Failed to add safety table collision object to planning scene.");
  } else {
    RCLCPP_INFO(logger,
                "Added safety table to planning scene (table top at z=0 in %s).",
                shared_cfg.base_frame.c_str());
  }

  move_group.setStartStateToCurrentState();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> vel_dist(joint_cfg.vel_min, joint_cfg.vel_max);

  moveit_msgs::msg::RobotTrajectory combined;
  int planned_and_executed = 0;

  auto start_state_ptr = move_group.getCurrentState(1.0);
  moveit::core::RobotState last_state =
      start_state_ptr ? *start_state_ptr : moveit::core::RobotState(robot_model);
  if (!start_state_ptr) {
    last_state.setToDefaultValues();
  }

  const moveit::core::JointModelGroup *jmg =
      robot_model->getJointModelGroup(joint_cfg.group_name);
  if (!jmg) {
    RCLCPP_ERROR(logger, "Joint model group '%s' not found.",
                 joint_cfg.group_name.c_str());
    rclcpp::shutdown();
    spin_thread.join();
    return 1;
  }

  const std::size_t max_segment_attempts =
      static_cast<std::size_t>(std::max(1, n_waypoints)) *
      std::max<std::size_t>(joint_cfg.max_sampling_retries, static_cast<std::size_t>(1));
  std::size_t segment_attempts = 0;

  while (planned_and_executed < n_waypoints) {
    if (segment_attempts >= max_segment_attempts) {
      RCLCPP_ERROR(
          logger,
          "Unable to reach requested waypoint count: completed %d/%d after %zu attempts. "
          "Check bounds/collision constraints or increase retry limits.",
          planned_and_executed, n_waypoints, segment_attempts);
      rclcpp::shutdown();
      spin_thread.join();
      return 1;
    }
    ++segment_attempts;

    const int segment_idx = planned_and_executed;
    const double vel_scale = vel_dist(gen);
    move_group.setMaxVelocityScalingFactor(vel_scale);
    RCLCPP_INFO(logger, "Segment %d (attempt %zu/%zu): velocity scaling = %.3f",
                segment_idx, segment_attempts, max_segment_attempts, vel_scale);

    move_group.setStartStateToCurrentState();

    bool found_valid_target = false;
    moveit::core::RobotState target_state = last_state;

    // Pre-compute 90%-clamped bounds for each active joint.
    const auto &joint_models = jmg->getActiveJointModels();
    std::vector<std::pair<double, double>> clamped_bounds;
    clamped_bounds.reserve(joint_models.size());
    for (const auto *jm : joint_models) {
      const auto &raw = jm->getVariableBounds();
      if (raw.empty()) {
        clamped_bounds.emplace_back(0.0, 0.0);
        continue;
      }
      const double lo        = raw[0].min_position_;
      const double hi        = raw[0].max_position_;
      const double mid       = 0.5 * (lo + hi);
      const double half_span = 0.5 * (hi - lo) * 0.90;
      clamped_bounds.emplace_back(mid - half_span, mid + half_span);
    }

    for (std::size_t tries = 0; tries < joint_cfg.max_sampling_retries; ++tries) {
      // Draw each joint uniformly within its 90%-clamped range.
      std::vector<double> q(joint_models.size());
      for (std::size_t ji = 0; ji < joint_models.size(); ++ji) {
        const double lo = clamped_bounds[ji].first;
        const double hi = clamped_bounds[ji].second;
        q[ji] = lo + (hi - lo) *
                std::uniform_real_distribution<double>(0.0, 1.0)(gen);
      }
      target_state.setJointGroupPositions(jmg, q);
      target_state.update();

      if (!is_state_in_sampling_box(target_state, active_eef_link, shared_cfg)) {
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
      RCLCPP_WARN(logger,
                  "Segment %d: failed to sample valid joint state after %zu retries; retrying with a new target.",
                  segment_idx, joint_cfg.max_sampling_retries);
      move_group.clearPoseTargets();
      continue;
    }

    move_group.setJointValueTarget(target_state);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool ok = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (!ok || plan.trajectory_.joint_trajectory.points.empty()) {
      RCLCPP_WARN(logger, "Segment %d: planning failed; retrying with a new target.", segment_idx);
      move_group.clearPoseTargets();
      continue;
    }

    if (!trajectory_stays_in_sampling_box(plan.trajectory_, robot_model, jmg,
                                          active_eef_link, shared_cfg)) {
      RCLCPP_WARN(logger,
                  "Segment %d: planned path leaves sampling box; retrying with a new target.", segment_idx);
      move_group.clearPoseTargets();
      continue;
    }

    auto &pts = plan.trajectory_.joint_trajectory.points;
    if (!pts.empty()) {
      auto &last = pts.back();
      std::fill(last.velocities.begin(), last.velocities.end(), 0.0);
      std::fill(last.accelerations.begin(), last.accelerations.end(), 0.0);
    }

    append_dwell(plan.trajectory_, joint_cfg.dwell_sec);

    moveit_msgs::msg::DisplayTrajectory disp_msg;
    disp_msg.model_id = robot_model->getName();
    moveit::core::robotStateToRobotStateMsg(last_state, disp_msg.trajectory_start);
    disp_msg.trajectory.push_back(plan.trajectory_);
    disp_pub->publish(disp_msg);

    const bool exec_ok =
        (move_group.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!exec_ok) {
      RCLCPP_ERROR(logger, "Segment %d: execution failed; retrying.", segment_idx);
      move_group.clearPoseTargets();
      continue;
    }
    RCLCPP_INFO(logger, "Segment %d: execution succeeded.", segment_idx);

    const auto &pts_after = plan.trajectory_.joint_trajectory.points;
    if (!pts_after.empty()) {
      const auto &q_end = pts_after.back().positions;
      last_state.setJointGroupPositions(jmg, q_end);
      last_state.update();
    }

    append_trajectory(combined, plan.trajectory_, true);
    ++planned_and_executed;

    move_group.clearPoseTargets();
  }

  if (combined.joint_trajectory.points.empty()) {
    RCLCPP_ERROR(logger, "No successful segments executed. Try adjusting bounds or planning settings.");
    rclcpp::shutdown();
    spin_thread.join();
    return 1;
  }

  RCLCPP_INFO(logger,
              "Combined executed trajectory has %zu points from %d segments.",
              combined.joint_trajectory.points.size(), planned_and_executed);

  moveit_msgs::msg::DisplayTrajectory disp_msg;
  disp_msg.model_id = robot_model->getName();

  if (auto current_ptr = move_group.getCurrentState(0.5)) {
    moveit::core::robotStateToRobotStateMsg(*current_ptr, disp_msg.trajectory_start);
  }

  disp_msg.trajectory.push_back(combined);
  disp_pub->publish(disp_msg);
  RCLCPP_INFO(logger, "Published combined path to /display_planned_path");

  try {
    const auto &jt = combined.joint_trajectory;
    const size_t nq = jt.joint_names.size();

    std::ofstream ofs(csv_out);
    ofs << std::fixed << std::setprecision(9);

    ofs << "time_s";
    for (size_t j = 0; j < nq; ++j) {
      ofs << ",q_" << j;
    }
    for (size_t j = 0; j < nq; ++j) {
      ofs << ",dq_" << j;
    }
    ofs << "\n";

    for (const auto &pt : jt.points) {
      const double t = static_cast<double>(pt.time_from_start.sec) +
                       static_cast<double>(pt.time_from_start.nanosec) * 1e-9;

      ofs << std::setprecision(9) << t << std::setprecision(9);

      if (!pt.positions.empty()) {
        for (auto v : pt.positions) {
          ofs << "," << v;
        }
      } else {
        for (size_t j = 0; j < nq; ++j) {
          ofs << "," << std::numeric_limits<double>::quiet_NaN();
        }
      }

      if (!pt.velocities.empty()) {
        for (auto v : pt.velocities) {
          ofs << "," << v;
        }
      } else {
        for (size_t j = 0; j < nq; ++j) {
          ofs << "," << std::numeric_limits<double>::quiet_NaN();
        }
      }
      ofs << "\n";
    }
    RCLCPP_INFO(logger, "Saved trajectory to %s", csv_out.c_str());
  } catch (const std::exception &e) {
    RCLCPP_ERROR(logger, "Failed to save CSV: %s", e.what());
  }

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
