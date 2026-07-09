"""
Nav2 bring-up for the TRN-localized Silent Sentry stack.
========================================================
Runs the Nav2 planner/controller/behaviors WITHOUT AMCL and WITHOUT a static
map->odom publisher, because the Terrain-Referenced Navigation stack already
owns the map->odom transform. Nav2 here provides obstacle avoidance and
terrain-aware global planning only; localization comes from TRN.

The global costmap's static layer is fed the a-priori terrain traversability
map (slope-derived), so the Smac Hybrid-A* planner routes around lethal slopes
in addition to LiDAR-sensed obstacles.

Prerequisites (launch first): world, robot, and
  ros2 launch ugv_localization terramechanic_localization.launch.py

Usage:
  ros2 launch bot_navigation nav2_trn.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg = get_package_share_directory('bot_navigation')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    map_yaml = LaunchConfiguration('map')

    # Nav2 servers WITHOUT amcl (TRN owns map->odom).
    lifecycle_nodes = [
        'map_server',
        'controller_server',
        'smoother_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
        'velocity_smoother',
    ]

    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key='',
            param_rewrites={'use_sim_time': use_sim_time, 'autostart': autostart},
            convert_types=True),
        allow_substs=True)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(pkg, 'config', 'nav2_params.yaml')),
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(pkg, 'maps', 'terrain_costmap.yaml'),
            description='Terrain traversability map served as the global static layer'),

        Node(package='nav2_map_server', executable='map_server', name='map_server',
             output='screen',
             parameters=[configured_params, {'yaml_filename': map_yaml}],
             remappings=remappings),
        Node(package='nav2_planner', executable='planner_server', name='planner_server',
             output='screen', parameters=[configured_params], remappings=remappings),
        Node(package='nav2_controller', executable='controller_server',
             output='screen', parameters=[configured_params],
             remappings=remappings + [('cmd_vel', 'cmd_vel_nav'),
                                      ('/odom', '/odometry/filtered')]),
        Node(package='nav2_smoother', executable='smoother_server', name='smoother_server',
             output='screen', parameters=[configured_params], remappings=remappings),
        Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server',
             output='screen', parameters=[configured_params], remappings=remappings),
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
             output='screen', parameters=[configured_params], remappings=remappings),
        Node(package='nav2_waypoint_follower', executable='waypoint_follower',
             name='waypoint_follower', output='screen',
             parameters=[configured_params], remappings=remappings),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother',
             name='velocity_smoother', output='screen', parameters=[configured_params],
             remappings=remappings + [('cmd_vel', 'cmd_vel_nav'),
                                      ('cmd_vel_smoothed', 'cmd_vel')]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[{'use_sim_time': use_sim_time},
                         {'autostart': autostart},
                         {'node_names': lifecycle_nodes}]),
    ])
