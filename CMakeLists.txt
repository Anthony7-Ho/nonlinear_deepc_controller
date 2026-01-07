cmake_minimum_required(VERSION 3.8)
project(nonlinear_deepc_controller)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(franka_semantic_components REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)

# Dependencies for executables
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIRS})

# Controller library
add_library(${PROJECT_NAME} SHARED
  src/joint_impedance_controller.cpp
  src/kernel_cartesian_impedance_controller.cpp
  src/kernel_bundle.cpp
  src/friction_predictor.cpp
  src/cartesian_trajectory.cpp
  src/joint_trajectory.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rclcpp_lifecycle
  controller_interface
  hardware_interface
  pluginlib
  franka_semantic_components
  franka_msgs
)

ament_python_install_package(${PROJECT_NAME})

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
  franka_msgs
)

add_executable(mixed_cartesian_trajectory_generator src/mixed_cartesian_trajectory_generator.cpp)
ament_target_dependencies(mixed_cartesian_trajectory_generator
  rclcpp
)

# install(PROGRAMS
# scripts/kernel_deepc_optimization.py
# DESTINATION lib/${PROJECT_NAME}
# )

# Install
install(TARGETS
  ${PROJECT_NAME}
  create_random_trajectory
  mixed_cartesian_trajectory_generator
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# install data files to share/<pkg>/data
# install(FILES
#   data_processing/kernel_deepc_bundle.npz
#   DESTINATION share/${PROJECT_NAME}/data
# )
install(
  DIRECTORY data_processing/kernel_deepc_bundle/
  DESTINATION share/${PROJECT_NAME}/data/kernel_deepc_bundle
)


install(DIRECTORY include/
  DESTINATION include
)

# Install plugin XML (controller_plugins.xml)
install(FILES nonlinear_deepc_controller.xml
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY config
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

