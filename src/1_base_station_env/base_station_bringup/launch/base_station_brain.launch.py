#!/usr/bin/env python3
"""
Base Station RL Brain Launch File
=================================
Launches the RL Coverage Brain node configured with sector uncertainty grid tracking,
RL policy inference for fleet reallocation, and Zenoh micro-burst broadcasting.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('base_station_bringup')
    
    config_file = os.path.join(pkg_dir, 'config', 'rl_brain_params.yaml')

    return LaunchDescription([
        # Enable Zenoh RMW for Base Station micro-burst routing
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp'),

        DeclareLaunchArgument(
            'params_file',
            default_value=config_file,
            description='Full path to parameter file for RL Coverage Brain'
        ),

        Node(
            package='base_station_bringup',
            executable='rl_coverage_brain.py',
            name='rl_coverage_brain',
            output='screen',
            parameters=[LaunchConfiguration('params_file')]
        )
    ])
