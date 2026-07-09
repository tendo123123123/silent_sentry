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
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_terrain_gating = LaunchConfiguration('use_terrain_gating')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_terrain_gating', default_value='false',
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
            }],
        ),
    ])
