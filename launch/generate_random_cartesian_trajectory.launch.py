import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import yaml


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None


def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots', 'fr3', 'fr3.urdf.xacro'
    )

    robot_description_config = Command(
        [
            FindExecutable(name='xacro'), ' ', franka_xacro_file,
            ' hand:=true',
            ' robot_ip:=', robot_ip,
            ' use_fake_hardware:=', use_fake_hardware,
            ' fake_sensor_commands:=', fake_sensor_commands,
            ' ros2_control:=true'
        ]
    )

    robot_description = {
        'robot_description': ParameterValue(robot_description_config, value_type=str)
    }

    franka_semantic_xacro_file = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'),
        'srdf', 'fr3_arm.srdf.xacro'
    )

    robot_description_semantic_config = Command(
        [FindExecutable(name='xacro'), ' ', franka_semantic_xacro_file, ' hand:=true']
    )

    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(
            robot_description_semantic_config, value_type=str
        )
    }

    kinematics_yaml = load_yaml('franka_fr3_moveit_config', 'config/kinematics.yaml')
    planning_pipelines_param = {'planning_pipelines': ['ompl']}

    ompl_planning_yaml = load_yaml('franka_fr3_moveit_config', 'config/ompl_planning.yaml') or {}
    ompl_param = {
        'ompl': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
                                'default_planner_request_adapters/ResolveConstraintFrames '
                                'default_planner_request_adapters/FixWorkspaceBounds '
                                'default_planner_request_adapters/FixStartStateBounds '
                                'default_planner_request_adapters/FixStartStateCollision '
                                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
            **ompl_planning_yaml,
        }
    }

    random_cartesian_generator_node = Node(
        package='nonlinear_deepc_controller',
        executable='create_random_trajectory',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            planning_pipelines_param,
            ompl_param,
        ],
    )

    robot_arg = DeclareLaunchArgument(
        robot_ip_parameter_name, description='Hostname or IP address of the robot.'
    )
    use_fake_hardware_arg = DeclareLaunchArgument(
        use_fake_hardware_parameter_name, default_value='false', description='Use fake hardware'
    )
    fake_sensor_commands_arg = DeclareLaunchArgument(
        fake_sensor_commands_parameter_name, default_value='false',
        description=f"Fake sensor commands. Only valid when '{use_fake_hardware_parameter_name}' is true"
    )

    return LaunchDescription([
        robot_arg,
        use_fake_hardware_arg,
        fake_sensor_commands_arg,
        random_cartesian_generator_node,
    ])
