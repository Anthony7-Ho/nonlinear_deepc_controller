from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    EmitEvent,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, LifecycleNode
from launch_ros.substitutions import FindPackageShare
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch.events import matches_action
from lifecycle_msgs.msg import Transition

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



def generate_launch_description():
    # Reuse the same arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip', default_value='192.168.1.200', description='Hostname or IP address of the robot.'
    )
    arm_id_arg = DeclareLaunchArgument(
        'arm_id', default_value='fr3',
        description='Arm id (fer, fr3, fp3). Must match your URDF/ros2_control names.'
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='false', description='Start RViz.'
    )
    use_fake_hw_arg = DeclareLaunchArgument(
        'use_fake_hardware', default_value='false', description='Use fake ros2_control hardware.'
    )
    fake_sensor_cmds_arg = DeclareLaunchArgument(
        'fake_sensor_commands', default_value='false',
        description="Only valid when 'use_fake_hardware' is true."
    )
    load_gripper_arg = DeclareLaunchArgument(
        'load_gripper', default_value='true',
        description='Load Franka gripper.'
    )

    robot_ip = LaunchConfiguration('robot_ip')
    arm_id = LaunchConfiguration('arm_id')
    load_gripper = LaunchConfiguration('load_gripper')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands')
    use_rviz = LaunchConfiguration('use_rviz')

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

    # Include Franka bringup launch file
    franka_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])
        ]),
        launch_arguments={
            'robot_ip': robot_ip,
            'arm_id': arm_id,
            'load_gripper': load_gripper,
            'use_fake_hardware': use_fake_hardware,
            'fake_sensor_commands': fake_sensor_commands,
            'use_rviz': use_rviz
        }.items()
    )

    # Parameters for the controller
    controller_param_file = PathJoinSubstitution([
        FindPackageShare('nonlinear_deepc_controller'),
        'config',
        'kernel_cartesian_impedance_controller.yaml'
    ])


    # Spawn controller
    # Instance name: joint_impedance_controller
    # Type: must match plugin class name in the XML:
    spawner_args = [
        'kernel_cartesian_impedance_controller',
        '--controller-type', 'nonlinear_deepc_controller/KernelCartesianImpedanceController',
        '--param-file', controller_param_file,
    ]

    # Spawn the controller spawner node
    spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=spawner_args,
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
        ],
    )


    return LaunchDescription([
        robot_ip_arg, arm_id_arg, use_rviz_arg, use_fake_hw_arg, fake_sensor_cmds_arg, load_gripper_arg,
        franka_bringup_launch,
        spawner,
    ])
