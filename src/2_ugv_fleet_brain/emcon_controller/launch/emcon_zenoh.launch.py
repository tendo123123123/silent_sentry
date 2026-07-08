#!/usr/bin/env python3
"""
EMCON Zenoh Launch File
=======================
Launches the EMCON Controller node configured with Eclipse Zenoh RMW profile
to suppress standard DDS multicast discovery spam and enable micro-burst transport.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('emcon_controller')
    
    config_file = os.path.join(pkg_dir, 'config', 'emcon_params.yaml')
    zenoh_profile = os.path.join(pkg_dir, 'config', 'zenoh_emcon_profile.json5')

    return LaunchDescription([
        # Set Zenoh RMW implementation and config URI to eliminate RF discovery spam
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp'),
        SetEnvironmentVariable('ZENOH_CONFIG', zenoh_profile),
        SetEnvironmentVariable('RMW_ZENOH_CONFIG_URI', zenoh_profile),

        DeclareLaunchArgument(
            'params_file',
            default_value=config_file,
            description='Full path to parameter file for EMCON Controller'
        ),

        Node(
            package='emcon_controller',
            executable='emcon_node',
            name='emcon_controller',
            output='screen',
            parameters=[LaunchConfiguration('params_file')]
        )
    ])
