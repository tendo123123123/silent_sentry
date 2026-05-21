import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    # Package path
    pkg_share = FindPackageShare('custom_ackermann_controller')
    
    # Configuration file paths
    enhanced_wheel_config = PathJoinSubstitution([
        pkg_share, 'config', 'enhanced_wheel_odometry.yaml'
    ])
    
    enhanced_imu_config = PathJoinSubstitution([
        pkg_share, 'config', 'enhanced_imu_processor.yaml'
    ])
    
    enhanced_ekf_config = PathJoinSubstitution([
        pkg_share, 'config', 'enhanced_ekf.yaml'
    ])
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    enable_imu_processing_arg = DeclareLaunchArgument(
        'enable_imu_processing',
        default_value='true',
        description='Enable enhanced IMU processing'
    )
     
    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/scan/points',
        description='3D Laser scan topic'
    )
    
    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/imu',
        description='IMU data topic'
    )
    
    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_footprint',
        description='Robot base frame'
    )
    
    # Enhanced wheel odometry node
    enhanced_wheel_odometry_node = Node(
        package='custom_ackermann_controller',
        executable='enhanced_wheel_odometry',
        name='enhanced_wheel_odometry',
        parameters=[
            enhanced_wheel_config,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'base_frame': LaunchConfiguration('base_frame'),
            }
        ],
        output='screen',
        remappings=[
            ('/wheel_states', '/wheel_states'),
            ('/cmd_vel', '/cmd_vel'),
        ]
    )
    
    # Enhanced IMU processor node (conditional)
    enhanced_imu_processor_node = Node(
        package='custom_ackermann_controller',
        executable='enhanced_imu_processor',
        name='enhanced_imu_processor',
        parameters=[
            enhanced_imu_config,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'imu_topic': LaunchConfiguration('imu_topic'),
                'base_frame': LaunchConfiguration('base_frame'),
            }
        ],
        output='screen',
        condition=launch.conditions.IfCondition(LaunchConfiguration('enable_imu_processing')),
        remappings=[
            ('/imu', LaunchConfiguration('imu_topic')),
        ]
    )
    
    # Robot localization EKF node
    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            enhanced_ekf_config,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }
        ],
        remappings=[
            ('/odometry/filtered', '/odometry/filtered'),
        ]
    )
    
    
    
    return LaunchDescription([
        # Launch arguments
        use_sim_time_arg,
        enable_imu_processing_arg,
        scan_topic_arg,
        imu_topic_arg,
        base_frame_arg,
        
        # Core localization nodes
        enhanced_wheel_odometry_node,
        enhanced_imu_processor_node,
        robot_localization_node,
    ])

