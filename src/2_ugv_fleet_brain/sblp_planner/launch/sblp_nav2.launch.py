"""
SBLP-over-Nav2 patrol launch.
=============================
Starts the Lévy-patrol goal generator that dispatches waypoints to Nav2.

Prerequisites (launch these first):
  1. World       : ros2 launch base_station_bringup world_only.launch.xml
  2. Robot       : ros2 launch fleet_bringup robot_only.launch.xml
  3. Localization: ros2 launch ugv_localization terramechanic_localization.launch.py
  4. Navigation  : ros2 launch bot_navigation nav2_trn.launch.py

Then:
  ros2 launch sblp_planner sblp_nav2.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('sblp_planner'),
        'config', 'sblp_params.yaml')
    # Same a-priori traversability raster the Nav2 global static layer uses, so
    # SBLP and the planner agree on which terrain is lethal.
    terrain_map = os.path.join(
        get_package_share_directory('bot_navigation'),
        'maps', 'continuous_planner_map.pgm')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_terrain_gating = LaunchConfiguration('use_terrain_gating')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        # Default ON: with gating off, Lévy waypoints landed on lethal dune
        # slopes (~21% of the patrol region) and the Smac planner correctly
        # reported "no valid path found", stalling patrol for the full goal
        # timeout. Gating rejects those waypoints at selection time.
        DeclareLaunchArgument('use_terrain_gating', default_value='true',
                              description='Reject Lévy waypoints on lethal terrain '
                                          '(requires an aligned terrain costmap)'),
        Node(
            package='sblp_planner',
            executable='sblp_goal_generator',
            name='sblp_goal_generator',
            output='screen',
            parameters=[config, {
                'use_sim_time': use_sim_time,
                'use_terrain_gating': use_terrain_gating,
                'terrain_costmap_path': terrain_map,
                # cost = (slope/30)^2, so 0.8 -> 26.8 deg: deliberately stricter
                # than the planner's ~29.9 deg lethal cut so goals are not
                # placed right on the edge of impassable terrain.
                'terrain_cost_threshold': 0.8,
                'terrain_origin_x': -450.0,
                'terrain_origin_y': -150.0,
                'terrain_resolution': 1.0,
            }],
        ),
    ])
