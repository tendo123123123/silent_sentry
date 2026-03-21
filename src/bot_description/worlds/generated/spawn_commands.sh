#!/usr/bin/env bash
set -e
ros2 run ros_gz_sim create -world thar_desert -name alpha -topic /alpha/robot_description -x -380 -y -70 -z 2.0 -Y 0.0
ros2 run ros_gz_sim create -world thar_desert -name bravo -topic /bravo/robot_description -x -360 -y 0 -z 2.0 -Y 0.5
ros2 run ros_gz_sim create -world thar_desert -name charlie -topic /charlie/robot_description -x -340 -y 70 -z 2.0 -Y 1.0
