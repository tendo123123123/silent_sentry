"""
Terramechanic Localization Launch File
======================================
Launches the complete desert-hardened localization stack:

  1. imu_filter_madgwick    — AHRS fusion (external imu_tools)
  2. terramechanic_odometry — Bekker-Wong wheel odometry
  3. local_dem_builder      — LiDAR rolling DEM
  4. factor_graph_fuser     — GTSAM iSAM2 dead-reckoning (50 Hz)
  5. trn_slam_node          — MCL DEM matching (3 Hz)
  6. odom_visualizer        — Matplotlib viz + benchmarking

TF Tree (REP-105):
    map ──(TRN SLAM)──> odom ──(FactorGraph 50Hz)──> base_footprint

Architecture notes:
    - factor_graph_fuser is the local dead-reckoning backend (odom→base).
    - trn_slam_node holds global correction authority (map→odom TF).
    - TRN match_quality feeds back into factor-graph covariance scaling.
    - This split prevents frame-tearing while allowing soft correction.
"""

import os

from launch import LaunchDescription
from launch.events import matches_action
from launch_ros.actions import Node, LifecycleNode
from launch.actions import DeclareLaunchArgument, LogInfo, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessStart
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
import lifecycle_msgs.msg
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(launch_dir)
    source_config_dir = os.path.join(package_root, 'config')
    source_package_init = os.path.join(
        package_root,
        'ugv_localization',
        '__init__.py',
    )
    use_source_tree = (
        os.path.isdir(source_config_dir)
        and os.path.isfile(source_package_init)
    )

    if use_source_tree:
        terramech_config = os.path.join(source_config_dir, 'terramechanic_odometry.yaml')
        fg_config = os.path.join(source_config_dir, 'factor_graph.yaml')
        trn_config = os.path.join(source_config_dir, 'trn_slam.yaml')
        existing_pythonpath = os.environ.get('PYTHONPATH', '')
        source_pythonpath = package_root
        if existing_pythonpath:
            source_pythonpath = package_root + os.pathsep + existing_pythonpath
        custom_pkg_env = {'PYTHONPATH': source_pythonpath}
        launch_mode_info = LogInfo(
            msg=(
                'terramechanic_localization: using source-tree code and configs from '
                f'{package_root}'
            )
        )
    else:
        pkg_share = FindPackageShare('ugv_localization')
        terramech_config = PathJoinSubstitution([
            pkg_share, 'config', 'terramechanic_odometry.yaml'
        ])
        fg_config = PathJoinSubstitution([
            pkg_share, 'config', 'factor_graph.yaml'
        ])
        trn_config = PathJoinSubstitution([
            pkg_share, 'config', 'trn_slam.yaml'
        ])
        custom_pkg_env = None
        launch_mode_info = LogInfo(
            msg='terramechanic_localization: using installed package share/configs'
        )

    # ---- Launch Arguments ----
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time (Gazebo)'
    )

    imu_raw_topic_arg = DeclareLaunchArgument(
        'imu_raw_topic', default_value='/imu',
        description='Raw IMU topic (sensor_msgs/Imu) — Gazebo bridge publishes on /imu'
    )

    imu_mag_topic_arg = DeclareLaunchArgument(
        'imu_mag_topic', default_value='/imu/mag',
        description='Magnetometer topic (sensor_msgs/MagneticField)'
    )

    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic', default_value='/scan/points',
        description='3D LiDAR PointCloud2 topic'
    )

    global_dem_path_arg = DeclareLaunchArgument(
        'global_dem_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('bot_navigation'),
            'maps',
            'synthetic_dem.bin',
        ]),
        description='Path to global DEM binary file'
    )

    model_name_arg = DeclareLaunchArgument(
        'model_name', default_value='alpha',
        description='Gazebo model name — used by odom visualizer to match ground truth TFMessage child_frame_id'
    )

    # ====================================================================
    # NODE 1: IMU Filter Madgwick (from imu_tools package)
    # ====================================================================
    # Subscribes: /imu/data_raw, /imu/mag
    # Publishes:  /imu/data_filtered (Imu with orientation quaternion)
    #
    # This is the gravity leakage fix: Madgwick produces a quaternion that
    # tracks the body's orientation relative to gravity. Downstream nodes
    # use this to mathematically subtract the gravity vector from the
    # accelerometer reading.
    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter_madgwick_node',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'use_mag': False},
        ],
        remappings=[
            ('imu/data_raw', LaunchConfiguration('imu_raw_topic')),
            ('imu/mag', LaunchConfiguration('imu_mag_topic')),
            ('imu/data', '/imu/data_filtered'),
        ],
        output='screen',
    )

    # ====================================================================
    # NODE 2: Terramechanic Wheel Odometry
    # ====================================================================
    # Subscribes: /joint_states, /imu/data_filtered
    # Publishes:  /terramechanic_odom (Odometry with slip-scaled covariance)
    #
    terramech_odom_node = Node(
        package='ugv_terramechanics',
        executable='terramechanic_odometry',
        name='terramechanic_odometry_node',
        parameters=[
            terramech_config,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        additional_env=custom_pkg_env,
        output='screen',
    )

    # ====================================================================
    # NODE 3: Local DEM Builder [C++ LIFECYCLE]
    # ====================================================================
    # Subscribes: /scan/points (PointCloud2)
    # Publishes:  /elevation_map/local (OccupancyGrid)
    #             /elevation_map/local_dem (typed LocalDEM)
    #
    local_dem_node = LifecycleNode(
        package='ugv_local_dem',
        executable='dem_builder_node',
        name='local_dem_builder',
        namespace='',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'lidar_topic': LaunchConfiguration('lidar_topic'),
            'base_frame': 'base_footprint',
            'odom_frame': 'odom',
            'grid_resolution': 1.0,
            'grid_size_x': 20.0,
            'grid_size_y': 20.0,
            'publish_rate': 2.0,
            'cloud_queue_size': 20,
            'deskew_scan_period': 0.1,          # 10 Hz LiDAR => 100 ms sweep
            'deskew_clockwise': False,
            'rolling_submap_distance': 50.0,    # captures dune-scale context while keeping rolling buffer bounded
            'submap_spatial_bin_size': 5.0,
            'uamc_drift_variance': 0.01,
            'ground_height_min': -0.5,
            'ground_height_max': 1.5,
            'obstacle_height_threshold': 0.5,
            'ransac_distance_threshold': 0.15,
            'ransac_iterations': 50,
            'min_points_per_cell': 2,
            'min_range': 0.5,
            'max_range': 30.0,
            'spawn_elevation': 7.2,
        }],
        output='screen',
    )

    # When local_dem_builder transitions to inactive (after configure), activate it
    dem_activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=local_dem_node,
            goal_state='inactive',
            entities=[
                LogInfo(msg="local_dem_builder configured to inactive. Activating..."),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(local_dem_node),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                )),
            ],
        )
    )

    # Configure local_dem_builder automatically on startup
    dem_configure_trigger = RegisterEventHandler(
        OnProcessStart(
            target_action=local_dem_node,
            on_start=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(local_dem_node),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
                ))
            ]
        )
    )

    # ====================================================================
    # NODE 4: Factor Graph Fuser (odom → base_footprint)
    # ====================================================================
    # Local dead-reckoning backend. Fuses:
    #   - wheel forward displacement from terramechanic_odom
    #   - SE(3) IMU preintegration from /imu/data_filtered
    #   - absolute IMU attitude priors from Madgwick
    # It subscribes to /trn/match_quality for covariance scaling feedback.
    # Global pose correction stays in the separate map→odom TF.
    #
    # Subscribes: /terramechanic_odom (vx, ω), /imu/data_filtered (yaw)
    # Publishes:  /odometry/filtered, odom→base_footprint TF
    #
    # Node-side readiness gating waits for first filtered IMU and first
    # terramechanic odom sample before publishing odom→base_footprint.
    # ====================================================================
    # NODE 4: Factor Graph Fuser (odom → base_footprint) [C++ LIFECYCLE]
    # ====================================================================
    # Local dead-reckoning backend. Fuses:
    #   - wheel forward displacement from terramechanic_odom
    #   - SE(3) IMU preintegration from /imu/data_filtered
    #   - absolute IMU attitude priors from Madgwick
    # It subscribes to /trn/match_quality for covariance scaling feedback.
    # Global pose correction stays in the separate map→odom TF.
    #
    # Subscribes: /terramechanic_odom (vx, ω), /imu/data_filtered (yaw)
    # Publishes:  /odometry/filtered, odom→base_footprint TF
    #
    fg_node = LifecycleNode(
        package='ugv_estimation',
        executable='fuser_node',
        name='factor_graph_fuser',
        namespace='',
        parameters=[
            fg_config,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        output='screen',
    )

    # When factor_graph_fuser transitions to inactive (after configure), activate it
    fg_activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=fg_node,
            goal_state='inactive',
            entities=[
                LogInfo(msg="factor_graph_fuser configured to inactive. Activating..."),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(fg_node),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                )),
            ],
        )
    )

    # Configure factor_graph_fuser automatically on startup
    fg_configure_trigger = RegisterEventHandler(
        OnProcessStart(
            target_action=fg_node,
            on_start=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(fg_node),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
                ))
            ]
        )
    )

    # ====================================================================
    # NODE 5: TRN SLAM (map → odom) [C++ LIFECYCLE]
    # ====================================================================
    # Subscribes: /elevation_map/local_dem, /odometry/filtered
    # Publishes:  map→odom TF, /trn/match_quality, /trn/entropy
    #
    trn_slam_node = LifecycleNode(
        package='ugv_trn',
        executable='trn_node',
        name='trn_slam_node',
        namespace='',
        parameters=[
            trn_config,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'global_dem_path': LaunchConfiguration('global_dem_path'),
            },
        ],
        output='screen',
    )

    # When trn_slam_node transitions to inactive (after configure), activate it
    trn_activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=trn_slam_node,
            goal_state='inactive',
            entities=[
                LogInfo(msg="trn_slam_node configured to inactive. Activating..."),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(trn_slam_node),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                )),
            ],
        )
    )

    # Configure trn_slam_node automatically on startup
    trn_configure_trigger = RegisterEventHandler(
        OnProcessStart(
            target_action=trn_slam_node,
            on_start=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(trn_slam_node),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
                ))
            ]
        )
    )

    # ====================================================================
    # NODE 6: Odometry Visualizer (graphical + benchmarking)
    # ====================================================================
    # Real-time matplotlib visualization of odometry vs ground truth
    # Displays: XY trajectory, position error, heading error, drift %
    odom_visualizer_node = Node(
        package='ugv_localization',
        executable='odom_visualizer',
        name='odom_visualizer_node',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'max_history': 3000,
            'update_rate_hz': 5.0,
            'ground_truth_topic': '/ground_truth/pose',
            'model_name': LaunchConfiguration('model_name'),
            'save_csv_on_exit': True,
            'csv_path': '/tmp/odom_comparison.csv',
        }],
        additional_env=custom_pkg_env,
        output='screen',
    )

    return LaunchDescription([
        # Launch arguments
        launch_mode_info,
        use_sim_time_arg,
        imu_raw_topic_arg,
        imu_mag_topic_arg,
        lidar_topic_arg,
        global_dem_path_arg,
        model_name_arg,

        # Nodes start immediately; each node blocks on its own runtime prerequisites.
        imu_filter_node,
        terramech_odom_node,
        odom_visualizer_node,

        # C++ Lifecycle Node configurations and triggers
        local_dem_node,
        dem_activate_event,
        dem_configure_trigger,

        fg_node,
        fg_activate_event,
        fg_configure_trigger,

        trn_slam_node,
        trn_activate_event,
        trn_configure_trigger,
    ])

