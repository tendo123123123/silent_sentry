#!/usr/bin/env python3
"""
The controller_manager runs NATIVELY on the robot's ROS_DOMAIN_ID,
not inside Gazebo. The EmconSystemInterface hardware plugin handles
gz-transport communication to the physics engine.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import UnlessCondition, IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    


    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation time'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace (empty for single-robot, e.g. "alpha" for fleet)'
    )

    bot_name_arg = DeclareLaunchArgument(
        'bot_name',
        default_value='autobot',
        description='Robot name for URDF xacro parameterization'
    )


    use_sim_time = LaunchConfiguration('use_sim_time')
    namespace = LaunchConfiguration('namespace')
    bot_name = LaunchConfiguration('bot_name')



    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([
            FindPackageShare('bot_description'), 'urdf', 'bot.urdf.xacro'
        ]),
        ' bot_name:=', bot_name
    ])

    controller_params_file = PathJoinSubstitution([
        FindPackageShare('bot_controller'), 'config', 'bot_controller.yaml'
    ])

    # Native ros2_control_node
    # This replaces the gz_ros2_control plugin that previously ran the
    # controller_manager inside the Gazebo process.  Now the
    # controller_manager lives on the robot's ROS_DOMAIN_ID and talks
    # to Gazebo through the EmconSystemInterface (gz-transport).
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robot_description_content},
            controller_params_file,
            {'use_sim_time': use_sim_time},
        ],
        output='screen',
    )

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
    )

    ackermann_twist_controller_node = Node(
        package='custom_ackermann_controller',
        executable='ackermann_twist_controller',
        name='ackermann_twist_controller',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
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
                'robot_localization.yaml',
            ]),
            {'use_sim_time': use_sim_time},
        ],
    )



    return LaunchDescription([
        use_sim_time_arg,
        namespace_arg,
        bot_name_arg,

        # Native controller manager
        ros2_control_node,

        joint_state_broadcaster,
        forward_position_controller,
        forward_velocity_controller,

        ackermann_twist_controller_node,
        wheel_odometry_node,
        ekf_node,
    ])