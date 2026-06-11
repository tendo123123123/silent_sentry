#!/bin/bash
# ==============================================================================
# Record ROS 2 Bag for Offline Localization Testing
# ==============================================================================
# Run this script while your Gazebo simulation and base station are running.
# It records all the raw simulated sensor data and ground truth needed to replay 
# the entire localization stack (Odometry, IMU filter, TRN, Factor Graph) offline.
# 
# To replay later:
# ros2 bag play localization_dataset_01
# ==============================================================================

BAG_NAME="localization_dataset_$(date +%Y%m%d_%H%M%S)"

echo "Starting recording: ${BAG_NAME}"
echo "Press Ctrl+C to stop recording."

ros2 bag record -o ${BAG_NAME} \
    /imu \
    /scan/points \
    /joint_states \
    /tf \
    /tf_static \
    /ground_truth/pose \
    /clock \
    /cmd_vel

# Note on topics:
# /imu                  -> Raw IMU data from Gazebo (fed into Madgwick filter)
# /scan/points          -> Raw 3D LiDAR (fed into DEM builder / TRN)
# /joint_states         -> Wheel speeds (fed into terramechanic odometry)
# /tf, /tf_static       -> Required for all frame transforms
# /ground_truth/pose    -> Ground truth for the evaluation plots
# /clock                -> Required for use_sim_time=true offline replay
# /cmd_vel              -> Good to have for debugging steering inputs
