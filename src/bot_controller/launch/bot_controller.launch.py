#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import UnlessCondition, IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── Declare launch arguments ──────────────────────────────────────
    diff_drive_controller_arg = DeclareLaunchArgument(
        'diff_drive_controller',
        default_value='False',
        description='Use differential drive controller'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation time'
    )

    diff_drive_controller = LaunchConfiguration('diff_drive_controller')
    use_sim_time = LaunchConfiguration('use_sim_time')

    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '180',
            '--switch-timeout', '60',
            '--service-call-timeout', '60',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    diff_drive_node = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'bot_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '180',
            '--switch-timeout', '60',
            '--service-call-timeout', '60',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(diff_drive_controller),
    )

    forward_position_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'forward_position_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '180',
            '--switch-timeout', '60',
            '--service-call-timeout', '60',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=UnlessCondition(diff_drive_controller),
    )


    forward_velocity_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'forward_velocity_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '180',
            '--switch-timeout', '60',
            '--service-call-timeout', '60',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=UnlessCondition(diff_drive_controller),
    )


    ackermann_twist_controller_node = Node(
        package='custom_ackermann_controller',
        executable='ackermann_twist_controller',
        name='ackermann_twist_controller',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=UnlessCondition(diff_drive_controller),
    )

    wheel_odometry_node = Node(
        package='custom_ackermann_controller',
        executable='wheel_odometry',
        name='wheel_odometry_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'wheelbase': 0.9},
            {'wheel_radius': 0.175},
            {'track_width': 0.67},
            {'odom_frame': 'odom'},
            {'base_frame': 'base_footprint'},
            {'left_wheel_joint': 'base_back_left_wheel_joint'},
            {'right_wheel_joint': 'base_back_right_wheel_joint'},
            {'left_steering_joint': 'base_front_left_steering_joint'},
            {'right_steering_joint': 'base_front_right_steering_joint'},
            {'publish_rate': 50.0},
        ],
        condition=UnlessCondition(diff_drive_controller),
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('custom_ackermann_controller'),
                'config',
                'robot_localization_minimal.yaml',
            ]),
            {'use_sim_time': use_sim_time},
        ],
        condition=UnlessCondition(diff_drive_controller),
    )

    chain_after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[forward_position_controller, diff_drive_node],
        )
    )

    chain_after_position = RegisterEventHandler(
        OnProcessExit(
            target_action=forward_position_controller,
            on_exit=[forward_velocity_controller],
        )
    )

    return LaunchDescription([
        diff_drive_controller_arg,
        use_sim_time_arg,

        joint_state_broadcaster,

        chain_after_jsb,
        chain_after_position,

        ackermann_twist_controller_node,
        wheel_odometry_node,
        ekf_node,
    ])