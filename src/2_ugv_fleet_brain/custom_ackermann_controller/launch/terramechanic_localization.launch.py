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
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(launch_dir)
    source_config_dir = os.path.join(package_root, 'config')
    source_package_init = os.path.join(
        package_root,
        'custom_ackermann_controller',
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
        pkg_share = FindPackageShare('custom_ackermann_controller')
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
            'synthetic_dem.tif',
        ]),
        description='Path to global DEM GeoTIFF file'
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
    # STARTUP DELAY — 1 second:
    #   Terramechanic node now gates internally on /imu/data_filtered.
    #   Short delay ensures Madgwick has started before terramechanic spins.
    terramech_odom_node = Node(
        package='custom_ackermann_controller',
        executable='terramechanic_odometry',
        name='terramechanic_odometry_node',
        parameters=[
            terramech_config,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        additional_env=custom_pkg_env,
        output='screen',
    )
    terramech_odom_node_delayed = TimerAction(
        period=1.0,
        actions=[terramech_odom_node],
    )

    # ====================================================================
    # NODE 3: Local DEM Builder
    # ====================================================================
    # Subscribes: /scan/points (PointCloud2)
    # Publishes:  /elevation_map/local (OccupancyGrid)
    #             /elevation_map/local_float (Float32MultiArray)
    #
    # STARTUP DELAY — 2 seconds:
    #   Wait for factor graph to initialize and publish odom→base_footprint TF.
    #   local_dem_builder deskews each sweep, filters self-hits, and accumulates
    #   a rolling odom-frame ground submap before rasterizing the latest DEM.
    local_dem_node = Node(
        package='custom_ackermann_controller',
        executable='local_dem_builder',
        name='local_dem_builder',
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
            'deskew_scan_period': 0.1,          # 10 Hz LiDAR => 100 ms sweep; keeps deskew timing matched to the hardware
            'deskew_clockwise': False,
            'rolling_submap_distance': 50.0,    # captures dune-scale context while keeping the rolling 3D buffer bounded in RAM
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
            # Z-height alignment: terrain elevation at spawn point.
            # Converts robot-relative LiDAR z to world-absolute z for
            # consistent matching against the global synthetic DEM.
            # Value: ~7.2m (DEM center=7.207m, the terrain under spawn).
            'spawn_elevation': 7.2,
        }],
        additional_env=custom_pkg_env,
        output='screen',
    )
    local_dem_node_delayed = TimerAction(
        period=2.0,
        actions=[local_dem_node],
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
    # STARTUP DELAY — 1 second:
    #   Factor graph now gates IMU preintegration on first /terramechanic_odom
    #   and resets to identity on first wheel tick, eliminating the 40 m drift.
    #   Short delay ensures terramechanic is publishing before we spin.
    fg_node = Node(
        package='custom_ackermann_controller',
        executable='factor_graph_fuser',
        name='factor_graph_fuser',
        parameters=[
            fg_config,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        additional_env=custom_pkg_env,
        output='screen',
    )
    fg_node_delayed = TimerAction(
        period=1.0,
        actions=[fg_node],
    )

    # ====================================================================
    # NODE 5: TRN SLAM (map → odom)
    # ====================================================================
    # Subscribes: /elevation_map/local_float, /odometry/filtered
    # Publishes:  map→odom TF, /trn/match_quality, /trn/entropy
    #
    # STARTUP DELAY — 3 seconds:
    #   Must wait for factor graph + local_dem_builder to be stable.
    #   TRN now has a DEM-bounds sanity gate on priors to prevent ROI crashes.
    trn_slam_node = Node(
        package='custom_ackermann_controller',
        executable='trn_slam_node',
        name='trn_slam_node',
        parameters=[
            trn_config,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'global_dem_path': LaunchConfiguration('global_dem_path'),
            },
        ],
        additional_env=custom_pkg_env,
        output='screen',
    )
    trn_slam_node_delayed = TimerAction(
        period=3.0,
        actions=[trn_slam_node],
    )

    # ====================================================================
    # NODE 6: Odometry Visualizer (graphical + benchmarking)
    # ====================================================================
    # Real-time matplotlib visualization of odometry vs ground truth
    # Displays: XY trajectory, position error, heading error, drift %
    odom_visualizer_node = Node(
        package='custom_ackermann_controller',
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

        # Nodes (order matters for startup, but they're all async)
        imu_filter_node,              # Must start first — downstream nodes need filtered IMU
        terramech_odom_node_delayed,  # +1s delay: waits for Madgwick convergence
        fg_node_delayed,              # +1s delay: waits for AHRS convergence + terrain settle
        local_dem_node_delayed,       # +2s delay: waits for factor graph to publish odom→base_footprint TF
        trn_slam_node_delayed,        # +3s delay: waits for factor graph + local DEM to be stable
        odom_visualizer_node,  # Graphical comparison + benchmarking — needs GT + filtered + raw odom
    ])

