"""
SBLP Patrol Launch
==================
Starts the Spatially-Bounded Lévy Patrol planner on top of an already-running
simulation + localization stack.

Prerequisites (launch these first, in order):
  1. World      : ros2 launch base_station_bringup world_only.launch.xml
  2. Robot      : ros2 launch fleet_bringup robot_only.launch.xml
  3. Localization: ros2 launch ugv_localization terramechanic_localization.launch.py

SBLP publishes /cmd_vel_raw. That only drives the wheels once it reaches
/cmd_vel. Choose one of:
  * direct_cmd_vel:=true  -> remap /cmd_vel_raw to /cmd_vel (bypass EMCON; simplest)
  * direct_cmd_vel:=false -> keep /cmd_vel_raw and run the emcon_controller
                             (State 0 passes commands through to /cmd_vel)
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('sblp_planner'),
        'config', 'sblp_params.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    direct_cmd_vel = LaunchConfiguration('direct_cmd_vel')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use Gazebo simulation clock'),
        DeclareLaunchArgument('direct_cmd_vel', default_value='true',
                              description='Remap /cmd_vel_raw to /cmd_vel '
                                          '(bypass EMCON controller)'),

        # Variant A: publish straight to /cmd_vel (no EMCON in the loop).
        Node(
            package='sblp_planner',
            executable='sblp_node',
            name='sblp_planner',
            output='screen',
            parameters=[config, {'use_sim_time': use_sim_time}],
            remappings=[('/cmd_vel_raw', '/cmd_vel')],
            condition=IfCondition(direct_cmd_vel),
        ),

        # Variant B: publish /cmd_vel_raw for the EMCON controller to arbitrate.
        Node(
            package='sblp_planner',
            executable='sblp_node',
            name='sblp_planner',
            output='screen',
            parameters=[config, {'use_sim_time': use_sim_time}],
            condition=UnlessCondition(direct_cmd_vel),
        ),
    ])
