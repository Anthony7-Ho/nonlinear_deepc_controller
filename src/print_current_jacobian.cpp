#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <iomanip>
#include <iostream>

static const std::string GROUP_NAME = "fr3_arm";
static const std::string EEF_LINK   = "fr3_hand_tcp";

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("jacobian_printer");

  // Parameters (still configurable if you want)
  std::string group_name =
      node->declare_parameter<std::string>("group_name", GROUP_NAME);
  std::string link_name =
      node->declare_parameter<std::string>("link_name", EEF_LINK);
  std::string joint_state_topic =
      node->declare_parameter<std::string>("joint_state_topic", "/joint_states");

  RCLCPP_INFO(node->get_logger(), "Creating MoveGroupInterface for '%s'", group_name.c_str());

  // This will use the same MoveIt config as your other node
  moveit::planning_interface::MoveGroupInterface move_group(node, group_name);
  auto robot_model = move_group.getRobotModel();
  if (!robot_model) {
    RCLCPP_FATAL(node->get_logger(), "Failed to get RobotModel from MoveGroupInterface.");
    rclcpp::shutdown();
    return 1;
  }

  const moveit::core::JointModelGroup* jmg =
      robot_model->getJointModelGroup(group_name);
  if (!jmg) {
    RCLCPP_FATAL(node->get_logger(), "JointModelGroup '%s' not found.", group_name.c_str());
    rclcpp::shutdown();
    return 1;
  }

  const moveit::core::LinkModel* link_model =
      robot_model->getLinkModel(link_name);
  if (!link_model) {
    RCLCPP_FATAL(node->get_logger(), "Link '%s' not found in robot model.", link_name.c_str());
    rclcpp::shutdown();
    return 1;
  }

  auto robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
  robot_state->setToDefaultValues();

  RCLCPP_INFO(node->get_logger(), "Subscribing to '%s' to get current joint state",
              joint_state_topic.c_str());

  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic, 10,
      [node, robot_state, jmg, link_model, group_name, link_name]
      (sensor_msgs::msg::JointState::ConstSharedPtr msg)
      {
        try {
          robot_state->setVariableValues(*msg);
        } catch (const std::exception& e) {
          RCLCPP_WARN(node->get_logger(), "Failed to set RobotState from JointState: %s",
                      e.what());
          return;
        }

        robot_state->updateLinkTransforms();

        Eigen::Vector3d reference_point_position(0.0, 0.0, 0.0);
        Eigen::MatrixXd J;
        robot_state->getJacobian(jmg, link_model, reference_point_position, J);

        if (J.rows() != 6) {
          RCLCPP_WARN(node->get_logger(), "Unexpected Jacobian rows: %ld", J.rows());
        }

        // Compute approximate Cartesian stiffness matrix
        Eigen::Matrix<double, 7, 1> k_joint;
        k_joint << 24.0, 24.0, 24.0, 24.0, 10.0, 6.0, 2.0;
        Eigen::Matrix<double, 7, 7> K_J = k_joint.asDiagonal();

        Eigen::MatrixXd J_pinv = J.completeOrthogonalDecomposition().pseudoInverse();

        Eigen::Matrix<double, 6, 6> K_C = J_pinv.transpose() * K_J * J_pinv;

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "Jacobian for group '" << group_name
                  << "' at link '" << link_name << "'\n";
        std::cout << "J (" << J.rows() << " x " << J.cols() << "):\n";
        for (int r = 0; r < J.rows(); ++r) {
          for (int c = 0; c < J.cols(); ++c) {
            std::cout << std::setw(12) << J(r, c) << " ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";

        std::cout << "Joint stiffness K_J (diag(k1..k7)):\n";
        std::cout << K_J << "\n\n";

        std::cout << "Approx. Cartesian stiffness K_C = J^+^T * K_J * J^+ (6x6):\n";
        std::cout << K_C << "\n\n";

        RCLCPP_INFO(node->get_logger(), "Jacobian and stiffness matrices printed. Shutting down.");
        rclcpp::shutdown();
      });

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
