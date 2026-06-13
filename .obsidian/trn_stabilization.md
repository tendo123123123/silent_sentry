# TRN & Odometry Stabilization Fixes

## 1. TRN Math Engine Revert
- **Problem**: Sobel-gradient matching (`dz/dx`, `dz/dy`) was mathematically invariant to vertical drift, but structurally flawed on continuous slopes. Because an inclined plane has a uniform gradient, all particles received equal weights (`score = 1.0`), resulting in a completely random Yaw publication that instantly spiraled the robot.
- **Fix**: Reverted `evaluate_particle_likelihood` back to Mean Absolute Difference (MAD) with zero-mean Z-offset compensation. This algorithm provides the same drift invariance but accurately anchors to the absolute topographical shape.

## 2. Absolute Z Correction
- **Problem**: TRN was forcing Z to remain at `map_prior.z()`, completely ignoring any vertical movement correction. In `fuser_node.cpp`, Z-covariance was inflated to `1e6` to force the factor graph to ignore TRN's Z.
- **Fix**: 
  - Modified `evaluate_particle_likelihood` to calculate and return `z_offset`.
  - In `execute_match_cycle`, aggregated `z_offsets` using a weighted mean to output the true absolute Z pose of the robot on the global DEM.
  - Removed `trn_covariance(5,5) = 1e6` in `fuser_node.cpp` to finally allow the Factor Graph to trust the TRN's absolute Z measurement.

## 3. TRN Gating & Thresholding
- **Problem**: Lowering `min_peak_quality` to `0.22` allowed matches with over 75cm of MAD error into the graph, causing severe false-positive graph corruptions.
- **Fix**: Restored `min_peak_quality` to `0.40` in `trn_slam.yaml` to ensure only high-confidence submap matches trigger a loop closure.

## 4. Wheel Odometry Yaw Decoupling
- **Problem**: Increasing `wheel_radius_nominal` (from `0.175` to `0.190`) to fix speed under-prediction inadvertently scaled up the kinematic angular velocity (`omega_z`). When `omega_z` was integrated into the wheel factor path, it forced the Factor Graph to learn massive lateral (Y) biases to resolve the conflict with the IMU.
- **Fix**: Decoupled the wheel encoder yaw. In `fuser_node.cpp`, the wheel integration step now explicitly pulls the yaw rate directly from the IMU (`last_imu_omega_z_`). The wheel odometry `BetweenFactor` now only contributes pure longitudinal displacement (`ds`), eliminating lateral conflicts.

## 5. 3D Factor Graph Decoupling (Dune Fix)
- **Problem**: Wheel encoders only measure longitudinal (X) speed. However, `BetweenFactor<Pose3>` enforces constraints across all 6 DOF. On 3D terrain (like dunes), the 2D wheel factor asserted `Z=0` and `Pitch=0`, causing the graph to violently fight the IMU's actual vertical measurements and spiral out of control.
- **Fix**: In `se3_fuser_core.cpp` (`evaluate_slip_gate`), explicitly inflated the sigmas for Roll (0), Pitch (1), Yaw (2), Lateral Y (4), and Vertical Z (5) to `1e6`. This surgically restricts the wheel factor to ONLY constrain forward movement (X), fully liberating the IMU for pure 3D dead-reckoning.
