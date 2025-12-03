import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    Shutdown,
    TimerAction,
)
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

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

    # Command-line args
    db_arg = DeclareLaunchArgument('db', default_value='False', description='Database flag')

    # planning_context
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

    # Planning pipeline params for move_group
    planning_pipelines_param = {'planning_pipelines': ['ompl']}

    # OMPL namespace with plugin + tuning
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
            **ompl_planning_yaml,  # merge yaml into the same 'ompl' namespace
        }
    }


    # Controllers for move_group execution
    moveit_simple_controllers_yaml = load_yaml(
        'franka_fr3_moveit_config', 'config/fr3_controllers.yaml'
    )
    moveit_controllers = {
        'moveit_simple_controller_manager': moveit_simple_controllers_yaml,
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }

    trajectory_execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }

    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    # move_group
    run_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            planning_pipelines_param,
            ompl_param,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
        ],
    )

    # RViz
    rviz_base = os.path.join(get_package_share_directory('franka_fr3_moveit_config'), 'rviz')
    rviz_full_config = os.path.join(rviz_base, 'moveit.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_full_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            planning_pipelines_param,
            ompl_param,
            kinematics_yaml,
        ],
    )

    # robot_state_publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    # ros2_control
    ros2_controllers_path = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'),
        'config', 'fr3_ros_controllers.yaml',
    )
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description, ros2_controllers_path],
        remappings=[('joint_states', 'franka/joint_states')],
        output={'stdout': 'screen', 'stderr': 'screen'},
        on_exit=Shutdown(),
    )

    # Load controllers
    load_controllers = []
    for controller in ['fr3_arm_controller', 'joint_state_broadcaster']:
        load_controllers += [
            ExecuteProcess(
                cmd=[f'ros2 run controller_manager spawner {controller}'],
                shell=True,
                output='screen',
            )
        ]

    # joint_state_publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'source_list': ['franka/joint_states', 'fr3_gripper/joint_states'], 'rate': 30}],
    )

    # optional real-hw state broadcaster
    franka_robot_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['franka_robot_state_broadcaster'],
        output='screen',
        condition=UnlessCondition(use_fake_hardware),
    )

    # Gripper
    gripper_launch_file = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution(
            [FindPackageShare('franka_gripper'), 'launch', 'gripper.launch.py'])]),
        launch_arguments={'robot_ip': robot_ip,
                          use_fake_hardware_parameter_name: use_fake_hardware}.items(),
    )

    # print current jacobian node
    print_current_jacobian_node = Node(
        package='nonlinear_deepc_controller',
        executable='print_current_jacobian',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            planning_pipelines_param,
            ompl_param,
        ],
    )

    delayed_print_jacobian = TimerAction(period=3.0, actions=[print_current_jacobian_node])

    # Launch args
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

    return LaunchDescription(
        [
            robot_arg,
            use_fake_hardware_arg,
            fake_sensor_commands_arg,
            db_arg,
            rviz_node,
            robot_state_publisher,
            run_move_group_node,
            ros2_control_node,
            joint_state_publisher,
            franka_robot_state_broadcaster,
            gripper_launch_file,
            delayed_print_jacobian,
        ] + load_controllers
    )
