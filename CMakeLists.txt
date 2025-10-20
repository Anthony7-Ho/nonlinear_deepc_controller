cmake_minimum_required(VERSION 3.8)
project(nonlinear_deepc_controller)

# ---- C++ + warnings ---------------------------------------------------------
#if(NOT CMAKE_CXX_STANDARD)
#  set(CMAKE_CXX_STANDARD 17)
#  set(CMAKE_CXX_STANDARD_REQUIRED ON)
#endif()
#if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  add_compile_options(-Wall -Wextra -Wpedantic)
#endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(franka_semantic_components REQUIRED)
find_package(franka_msgs REQUIRED) 

# Dependencies for executables
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIRS})

# ---- Controller library (ros2_control plugin) -------------------------------
# Files from the code I sent:
#   include/joint_impedance_controller/joint_impedance_controller.hpp
#   src/joint_impedance_controller.cpp
add_library(${PROJECT_NAME} SHARED
  src/joint_impedance_controller.cpp
)

# target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rclcpp_lifecycle
  controller_interface
  hardware_interface
  pluginlib
  franka_semantic_components
  franka_msgs
)

# Export plugin description so controller_manager can load it
pluginlib_export_plugin_description_file(
  controller_interface nonlinear_deepc_controller.xml
)

# executables
add_executable(create_random_trajectory src/create_random_trajectory.cpp)
ament_target_dependencies(create_random_trajectory
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  geometry_msgs
)

# Install
install(TARGETS
  ${PROJECT_NAME}
  create_random_trajectory
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# Install plugin XML (controller_plugins.xml) if it lives at package root
install(FILES nonlinear_deepc_controller.xml
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Exports 
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(
  rclcpp
  rclcpp_lifecycle
  controller_interface
  hardware_interface
  pluginlib
  franka_semantic_components
  franka_msgs
)

ament_package()

