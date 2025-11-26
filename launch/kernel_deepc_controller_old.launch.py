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

    # Optimizer LifecycleNode
    optimizer = LifecycleNode(
        package='nonlinear_deepc_controller',
        executable='kernel_deepc_optimization.py',
        name='optimization_node',
        output='screen',
        namespace='',
    )

    # Ask optimizer to CONFIGURE once it's started
    request_configure = EmitEvent(event = ChangeState(
        lifecycle_node_matcher=matches_action(optimizer),
        transition_id=Transition.TRANSITION_CONFIGURE
    ))

    # Prepare an ACTIVATE event
    request_activate = EmitEvent(event = ChangeState(
        lifecycle_node_matcher=matches_action(optimizer),
        transition_id=Transition.TRANSITION_ACTIVATE
    ))

    # When optimizer reaches 'inactive' (configured), send ACTIVATE
    activate_when_configured = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=optimizer,
            goal_state='inactive',
            entities=[request_activate]
        )
    )

    # Spawn controller
    # Instance name: joint_impedance_controller
    # Type: must match plugin class name in the XML:
    spawner_args = [
        'kernel_deepc_controller',
        '--controller-type', 'nonlinear_deepc_controller/KernelDeePCController',
    ]

    # Spawn the controller spawner node
    spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=spawner_args,
        output='screen',
    )

    # Gate spawner start on optimizer reaching 'active'
    start_spawner_when_optimizer_active = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=optimizer,
            goal_state='active',
            entities=[spawner]
        )
    )

    return LaunchDescription([
        robot_ip_arg, arm_id_arg, use_rviz_arg, use_fake_hw_arg, fake_sensor_cmds_arg, load_gripper_arg,
        franka_bringup_launch,
        optimizer,
        request_configure,
        activate_when_configured,
        start_spawner_when_optimizer_active,
    ])
