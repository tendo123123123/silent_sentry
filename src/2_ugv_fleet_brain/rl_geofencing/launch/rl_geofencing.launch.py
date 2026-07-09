"""Launch the Base Station RL elastic geo-fencing brain."""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('rl_geofencing'),
        'config', 'rl_geofencing.yaml')

    return LaunchDescription([
        Node(
            package='rl_geofencing',
            executable='base_station_node',
            name='base_station_rl_node',
            output='screen',
            parameters=[config],
        ),
    ])
