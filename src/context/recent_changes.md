# Recent Major Codebase Changes — Silent Sentry

**Date range:** 6 May 2026 → 28 May 2026  
**Workspace:** `/home/sailesh/silent_sentry`  
**ROS distro:** ROS 2 Jazzy / Gazebo Harmonic  
**Test domain:** `ROS_DOMAIN_ID=24`

---

## Summary of 5 Major Changes

| # | Change | File | Status |
|---|--------|------|--------|
| 1 | Sequential controller spawning chain | `bot_controller.launch.py` | ✅ Fixed & validated |
| 2 | Identified: factor-graph init divergence (~40 m at rest) | `factor_graph_fuser.py` | 🔄 Next fix target |
| 3 | Identified: TRN ROI overflow from bad prior | `trn_slam_node.py` | ⏳ Pending fix |
| 4 | Identified: odom comparator benchmarks wrong frame | `odom_ground_truth_comparator.py` | ⏳ Pending fix |
| 5 | Diagnosed: late blue marker in visualizer (TF vs topic mismatch) | `odom_visualizer_node.py` + `terramechanic_localization.launch.py` | ⏳ Pending fix |

---

## Change 1 — Sequential Controller Spawning Chain ✅ FIXED

**File:** `src/2_ugv_fleet_brain/bot_controller/launch/bot_controller.launch.py`

### Problem
All three `ros2_control` controller spawners (`joint_state_broadcaster`, `forward_velocity_controller`, `forward_position_controller`) were launched in parallel. Under load (736-model vegetation Gazebo scene), the `controller_manager` switch operations timed out at 60 s, meaning `forward_velocity_controller` and `forward_position_controller` never reached the `active` state from launch. This caused the `ackermann_twist_controller` node (which starts after them) to have no subscribers on its command topics.

**Symptom observed in live test (6 May 2026):**
```
[forward_velocity_controller spawner]: Timeout waiting for controller_manager service
[forward_position_controller spawner]: Timeout waiting for controller_manager service
```
After launch, `ros2 control list_controllers` showed both forward controllers in `inactive` state, requiring manual `ros2 control switch_controllers` invocations.

### Fix Applied
Changed from parallel launch to an explicit `OnProcessExit` chain using `RegisterEventHandler`:

```
ros2_control_node starts
  → OnProcessStart: launch joint_state_broadcaster spawner
    → OnProcessExit(joint_state_broadcaster): launch forward_position_controller spawner
      → OnProcessExit(forward_position_controller): launch forward_velocity_controller spawner
        → OnProcessExit(forward_velocity_controller): launch ackermann_twist_controller_node
```

**Key implementation detail — LaunchConfiguration scope bug:**  
An initial version of the fix lost the `use_sim_time` `LaunchConfiguration` inside the event handler closures because `LaunchConfiguration('use_sim_time')` was evaluated at declaration time before the `LaunchDescription` was active. Fixed by capturing the config object in a named variable before all handler definitions:

```python
use_sim_time = LaunchConfiguration('use_sim_time')   # captured once, reused everywhere
```

**Validation (live, 28 May 2026):**
```bash
$ ros2 control list_controllers
joint_state_broadcaster          [joint_state_broadcaster/JointStateBroadcaster] active
forward_position_controller      [forward_command_controller/ForwardCommandController] active
forward_velocity_controller      [forward_command_controller/ForwardCommandController] active
ackermann_twist_controller       ...  active

$ ros2 topic info /forward_velocity_controller/commands
Subscription count: 1

$ ros2 topic info /forward_position_controller/commands
Subscription count: 1
```
Both forward controllers reached `active` from launch without any manual intervention.

**Package rebuild:**
```bash
colcon build --packages-select bot_controller --symlink-install
```

---

## Change 2 — Factor-Graph Initialization Divergence (IDENTIFIED, fix pending)

**File:** `src/2_ugv_fleet_brain/custom_ackermann_controller/custom_ackermann_controller/factor_graph_fuser.py`

### Problem
The GTSAM iSAM2 factor-graph fuser (`FactorGraphFuser` node) publishes `/odometry/filtered` (the `odom → base_footprint` transform). Before the robot moves even 1 cm, the node reports a position of approximately `(31.03, 29.09)` with `yaw ≈ 123.7°` while Gazebo ground truth is at `(0.03, 0.06, yaw ≈ 4.5°)` — a ~40 m initialization error.

**Live sample at rest (6 May 2026):**
```
/ground_truth/pose      → position: (0.03, 0.06)  yaw: 4.5°
/terramechanic_odom     → position: (0.00, 0.00)  yaw: ~0°   [correct]
/odometry/filtered      → position: (31.03, 29.09) yaw: 123.7° [WRONG]
```

**Factor-graph log line:**
```
FG: (29.31,27.77) th=123.7 deg v=0.00 dist=0.0m nodes=266 [IMU] pim_dt=5.360s slip=False
```

The node had accumulated 266 iSAM2 nodes and a PIM (preintegrated IMU measurement) of **5.36 seconds** of raw IMU data — despite zero wheel motion. This is because the 5-second `TimerAction` delay in `terramechanic_localization.launch.py` allows the factor-graph node to start receiving IMU before wheel odometry is available, and it integrates all that IMU noise without a wheel anchor.

### Root Cause
The factor-graph starts with a zero-pose prior, but immediately begins receiving `/imu/data_filtered` (at ~100 Hz). With no wheel odometry factor to constrain the position during the 5-second startup window, the IMU preintegrator integrates gyro/accel noise and any residual gravity-subtraction error in Madgwick's convergence window. The result is a drift of ~30 m in position and ~120° in heading before the first wheel measurement arrives.

### Planned Fix (pending)
- Gate the factor-graph's first IMU preintegration on receiving at least one wheel odometry message.
- Alternatively, force the initial GTSAM prior to match the first received `/terramechanic_odom` pose (which starts at origin) rather than integrating from time zero.
- Add a reset-on-first-wheel-tick mechanism so that if `dist == 0` and `nodes > N`, the graph is reset to identity before accepting further IMU data.

---

## Change 3 — TRN ROI Overflow from Bad Prior (IDENTIFIED, fix pending)

**File:** `src/2_ugv_fleet_brain/custom_ackermann_controller/custom_ackermann_controller/trn_slam_node.py`  
**Also involves:** `src/2_ugv_fleet_brain/custom_ackermann_controller/custom_ackermann_controller/trn_core.py`

### Problem
The TRN SLAM node (`TRNSlamNode`) receives the factor-graph `/odometry/filtered` as its position prior for MCL DEM matching. When the factor-graph has diverged to `(31.03, 29.09)`, the TRN node tries to compute a Region of Interest (ROI) centered at those coordinates in the DEM's pixel space. Because the DEM is only 110 m × 80 m and the prior is ~40 m outside the DEM extent, the ROI computation produces negative pixel extents (integer underflow/overflow in the coordinate transform), resulting in a crash loop:

**Error messages observed (6 May 2026):**
```
[trn_slam_node]: ROI too small (-47474708x-20795232 px) — skipping match cycle
[trn_slam_node]: rejecting correction 44.12m > hard limit 8.00m
[trn_slam_node]: ROI too small (-47474708x-20795232 px) — skipping match cycle
```

The pixel coordinates `-47474708 × -20795232` indicate the position transformed far outside the array bounds, and the ROI rectangle wrapped around into deeply negative territory (not a `DEM_size - search_radius` clamp failure but a full integer sign flip at the coordinate conversion level).

### Root Cause Chain
```
bad factor-graph prior (31 m off)
  → TRN fed impossible location
  → ROI pixel coords underflow/overflow  
  → ROI too small cascade  
  → map → odom TF never published  
  → blue visualizer trace never starts  
  → EKF comparator reports err = 51,829,969 m (the filtered odom is at pixel-space coordinates, not metric)
```

### Planned Fix (pending)
- In `trn_slam_node.py`: add a prior sanity gate — before computing ROI, verify that the incoming prior position is within `[dem_origin_x, dem_origin_x + dem_width]` and `[dem_origin_y, dem_origin_y + dem_height]`.
- If the prior is outside DEM bounds, **skip the match cycle entirely** (log a warning) rather than computing an invalid ROI.
- This prevents the cascade without fixing the root cause (Change 2 is the root fix; this is a defensive guard).

---

## Change 4 — Odom Comparator Frame Inconsistency (IDENTIFIED, fix pending)

**File:** `src/2_ugv_fleet_brain/custom_ackermann_controller/custom_ackermann_controller/odom_ground_truth_comparator.py`

### Problem
The comparator benchmarks `/odometry/filtered` (published in the `odom` frame with the factor-graph's drifted `odom → base_footprint` TF) against ground truth. But the visualizer and the operational system both use the **map frame** TF chain (`map → odom → base_footprint`). The comparator therefore measures the wrong frame — it sees the full `odom`-frame error without the TRN's `map → odom` correction applied.

**Live result (6 May 2026):**
```
EKF: err=51829969.29m  θ=119.6°  ATE=7329924.14m  drift=1440867883.2%
RAW: err=0.34m  ATE=0.07m
```
The EKF error of ~51 million meters reflects the `odom`-frame pose from a diverged factor-graph being compared directly to GT — not a real localization error (TRN hadn't published `map → odom` at all due to Change 3).

### Root Cause
`odom_ground_truth_comparator.py` subscribes to `/odometry/filtered` and reads `pose.pose.position` directly. This is the `odom`-frame estimate. The comparator should instead look up the TF `map → base_footprint` (the true best-estimate pose used by navigation) to compare against GT.

### Planned Fix (pending)
- Add a `tf2_ros.Buffer` + `tf2_ros.TransformListener` to the comparator.
- At each benchmark tick, look up `map → base_footprint` TF and use that pose for the EKF error column.
- Keep the `/terramechanic_odom` raw error path unchanged (it already uses the correct odom-relative reference).

---

## Change 5 — Late Blue Marker in Visualizer (DIAGNOSED, fix pending)

**Files involved:**
- `src/2_ugv_fleet_brain/custom_ackermann_controller/custom_ackermann_controller/odom_visualizer_node.py`
- `src/2_ugv_fleet_brain/custom_ackermann_controller/launch/terramechanic_localization.launch.py`

### Problem
In RViz2, the blue trajectory trace (filtered/EKF odometry) only appeared after a significant delay (sometimes never in a 5-minute test run), while the green trace (raw terramechanic odometry) started immediately. The blue trace should start moving when the robot starts moving.

### Root Cause (multi-factor)
1. **Startup delay gates:** `terramechanic_localization.launch.py` uses `TimerAction` delays of 5 s (terramechanic, factor-graph), 6 s (local DEM), 7 s (TRN). These fixed delays do not synchronize on robot stillness — if Gazebo physics is slow to start, these timers fire before the robot has settled, poisoning the Madgwick filter's gravity convergence window.

2. **Visualizer reads TF, not topic:** The `odom_visualizer_node.py` was updated at some point to read the `map → base_footprint` TF chain (for the blue trace) instead of subscribing to `/odometry/filtered` directly. This means the blue trace cannot start until TRN has published at least one valid `map → odom` transform — which only happens after the TRN node gets a valid prior from the factor-graph.

3. **Cascading dependency:** Since Change 2 (factor-graph divergence) means TRN never gets a valid prior, and Change 3 means TRN never publishes `map → odom`, the blue trace from the visualizer never starts — because the TF lookup fails for the entire session.

### Fix Dependency
- Changes 2 and 3 are prerequisites: fixing the factor-graph initialization and gating the TRN ROI will allow TRN to publish valid `map → odom`, which will unblock the visualizer.
- After those fixes, the startup delays can be replaced with readiness-based gates (spin until first message on `/imu/data_filtered` before starting terramechanic, spin until first `/terramechanic_odom` before starting factor-graph, etc.).

---

## Architecture Reference

```
Gazebo ←── gz-transport ──→ EmconSystemInterface (C++ ros2_control plugin)
                                       │
                              controller_manager (native ROS 2)
                             ┌──────────┼──────────────────┐
                    joint_state_broadcaster  forward_velocity_controller  forward_position_controller
                                       │
                          ackermann_twist_controller
                          (subscribes /cmd_vel, drives both forward controllers)

TF Tree (REP-105):
    map ──[TRN SLAM, 3Hz]──→ odom ──[FactorGraph, 50Hz]──→ base_footprint

Localization pipeline:
    /imu  ──→  imu_filter_madgwick  ──→  /imu/data_filtered
                                                │
    /joint_states ──────────────────────────────┼──→  terramechanic_odometry  ──→  /terramechanic_odom
                                                │                                         │
                                                └──────────────────────────────────────→  factor_graph_fuser  ──→  /odometry/filtered
                                                                                                                         │
    /scan/points  ──→  local_dem_builder  ──→  /elevation_map/local_float  ──────────→  trn_slam_node  ──→  map→odom TF
```

---

## File Index

| File | Role | Modified? |
|------|------|-----------|
| `src/2_ugv_fleet_brain/bot_controller/launch/bot_controller.launch.py` | Controller spawner chain | ✅ Yes |
| `src/2_ugv_fleet_brain/custom_ackermann_controller/custom_ackermann_controller/factor_graph_fuser.py` | GTSAM iSAM2 dead-reckoning | ❌ Not yet |
| `src/2_ugv_fleet_brain/custom_ackermann_controller/custom_ackermann_controller/trn_slam_node.py` | TRN MCL SLAM | ❌ Not yet |
| `src/2_ugv_fleet_brain/custom_ackermann_controller/custom_ackermann_controller/odom_ground_truth_comparator.py` | Benchmarking node | ❌ Not yet |
| `src/2_ugv_fleet_brain/custom_ackermann_controller/custom_ackermann_controller/odom_visualizer_node.py` | RViz trajectory visualizer | ❌ Not yet |
| `src/2_ugv_fleet_brain/custom_ackermann_controller/launch/terramechanic_localization.launch.py` | Full localization stack launch | ❌ Not yet |
| `src/2_ugv_fleet_brain/fleet_bringup/launch/robot_only.launch.xml` | Fleet robot spawn + bridge | ❌ Not yet |

---

---

# Project Context — Silent Sentry Codebase

## 1. Project Overview

**Silent Sentry** is an autonomous UGV navigation stack designed for GPS-degraded desert corridors under EMCON (Emission Control) constraints. The platform is a 4-wheeled Ackermann-steer robot (85 kg, 0.9 m wheelbase) simulated in Gazebo Harmonic on a synthetic Thar Desert heightmap (1735 × 4144 m, with 736 vegetation models).

The system addresses three core challenges:
1. **EMCON compliance** — limit RF/EM emissions by suppressing commands when in radio-silent mode.
2. **GPS-denied localization** — fuse wheel odometry (terramechanics), IMU (Madgwick + GTSAM), and terrain-referenced navigation (TRN via DEM matching) without GPS.
3. **Traversability-aware planning** — classify terrain in real time using a zero-shot CLIP VLM and adapt trajectory primitives accordingly.

**ROS distro:** ROS 2 Jazzy  
**Simulator:** Gazebo Harmonic 8.10+  
**Language:** Python 3.12 (nodes), C++17 (hardware interface)  
**Build system:** `colcon` with `--symlink-install`

---

## 2. Repository Structure

```
silent_sentry/
├── README.md
├── requirements.txt
├── setup_gtsam.sh              — installs GTSAM Python bindings into .venv
├── paper_data/                 — offline analysis scripts and generated graphs
├── rosbag2_*/                  — recorded test bags
├── src/
│   ├── 1_base_station_env/     — world simulation (Gazebo launch, SDF, DEM assets)
│   │   └── base_station_bringup/
│   ├── 2_ugv_fleet_brain/      — all robot packages
│   │   ├── LIO-SAM/            — upstream LiDAR-Inertial odometry (unused in main stack)
│   │   ├── bot_controller/     — ros2_control manager + controller spawners
│   │   ├── bot_description/    — URDF/xacro, meshes, RViz config
│   │   ├── bot_navigation/     — Nav2 stack, maps, DEM assets
│   │   ├── custom_ackermann_controller/  — all custom Python nodes (localization, TRN, etc.)
│   │   ├── emcon_controller/   — EMCON command arbitration node
│   │   ├── emcon_hardware_interface/  — C++ ros2_control SystemInterface (gz-transport)
│   │   ├── fleet_bringup/      — per-robot launch file
│   │   ├── sblp_planner/       — Scenario-Based Local Planner
│   │   └── vlm_costmap/        — zero-shot CLIP traversability classifier
│   └── 3_deployment/
│       └── docker/             — Dockerfile + docker-compose for multi-robot deployment
└── build/ install/ log/        — colcon artefacts (not tracked)
```

---

## 3. Package Descriptions

### `1_base_station_env/base_station_bringup`
Manages the Gazebo simulation environment.

| Asset | Description |
|-------|-------------|
| `worlds/thar_desert.sdf` | Main Gazebo world — 1735×4144 m heightmap, 736 vegetation models, Bullet Featherstone physics |
| `worlds/dem/` | DEM tiles and heightmap PNGs used by the SDF |
| `worlds/textures/` | Sand, rock, and sky material textures |
| `launch/world_only.launch.xml` | Launches Gazebo server only (`gz sim -s -r`). Sets GZ_SIM_RESOURCE_PATH, physics = `bullet-featherstone`, rendering = `ogre2` |
| `models/` | Vegetation and obstacle SDF models for the world |

---

### `2_ugv_fleet_brain/bot_description`
Robot model definition.

| Asset | Description |
|-------|-------------|
| `urdf/bot.urdf.xacro` | Top-level xacro: includes geometry, inertia, sensors (IMU, LiDAR), `<ros2_control>` block, Gazebo plugins |
| `urdf/bot_ros2_control.xacro` | `<ros2_control>` block: 4 joints (2 drive velocity, 2 steering position), references `EmconSystemInterface` plugin |
| `meshes/` | STL/DAE files for body, wheels, lidar mount |
| `config/gz.rviz` | RViz2 config for simulation visualization |
| `launch/` | Standalone RSP launch (not normally used directly — invoked via `robot_only.launch.xml`) |

**Key robot parameters (from URDF):**
- Wheelbase: 0.9 m, Track width: 0.67 m, Wheel radius: 0.175 m
- Max steering: ±15° (0.2616 rad)
- Sensor: IMU at 100 Hz, LiDAR (simulated scan) at `/scan/points`

---

### `2_ugv_fleet_brain/emcon_hardware_interface`
C++ `ros2_control` hardware plugin.

| File | Description |
|------|-------------|
| `src/emcon_system_interface.cpp` | Implements `hardware_interface::SystemInterface`. Uses `gz::transport::Node` (not ROS 2 DDS) to subscribe to Gazebo joint states and publish joint commands directly. |
| `include/emcon_hardware_interface/emcon_system_interface.hpp` | Class declaration |
| `emcon_hardware_interface.xml` | `pluginlib` export registration |

**How it works:**
- `on_init()`: Parses `bot_name` and `world_name` from URDF `<hardware><param>` block. Reads joint definitions from `HardwareInfo`.
- `on_configure()`: Creates `gz::transport::Node`, subscribes to `/world/{world_name}/model/{bot_name}/joint_state`.
- `on_activate()`: Starts publishing to `/model/{bot_name}/joint/{joint_name}/cmd_vel` (drive) and `cmd_pos` (steering).
- `read()`: Copies gz joint state into `ros2_control` state interfaces.
- `write()`: Copies `ros2_control` command interfaces into gz transport messages.

**Why gz-transport instead of ROS topics:** The Gazebo joint command API uses gz-transport natively. Going through `ros_gz_bridge` would add one extra serialization hop and ~5 ms latency per control cycle, which is unacceptable for the 100 Hz control loop.

---

### `2_ugv_fleet_brain/bot_controller`
`ros2_control` manager + controller spawning.

| File | Description |
|------|-------------|
| `config/bot_controller.yaml` | `controller_manager` config: 100 Hz update rate, 30 s switch timeout. Declares `joint_state_broadcaster`, `forward_position_controller`, `forward_velocity_controller`. DiffDrive and AckermannSteeringController are disabled. |
| `launch/bot_controller.launch.py` | **[MODIFIED]** Sequential `OnProcessExit` chain: `ros2_control_node` → `joint_state_broadcaster` → `forward_position_controller` → `forward_velocity_controller` → `ackermann_twist_controller_node`. Passes `use_sim_time=True` to all nodes. |
| `launch/bot_controller.launch.xml` | XML shim (not currently used — Python launch is the active one) |

**Controller chain rationale:**  
`forward_position_controller` controls the 2 front steering joints (rad), and `forward_velocity_controller` controls the 2 rear drive joints (rad/s). The `ackermann_twist_controller` receives `/cmd_vel` (Twist) and fans it out to both. The sequential spawn ensures each controller is `active` before the next one is loaded.

---

### `2_ugv_fleet_brain/fleet_bringup`
Per-robot launch orchestration.

| File | Description |
|------|-------------|
| `launch/robot_only.launch.xml` | Launches for one robot: Robot State Publisher (xacro-expanded URDF), `ros_gz_bridge` (IMU + scan/points + ground_truth/pose + TF), Gazebo entity spawn at (x, y, z=8.0), `bot_controller.launch.py` include, optional RViz2 |
| `config/gazebo_bridge.yaml` | `ros_gz_bridge` configuration: maps Gazebo topics to ROS topics for IMU, LiDAR, pose, clock, TF |
| `scripts/` | Helper scripts (bridge config generator, etc.) |

**Spawn height z=8.0:** The robot is spawned at 8 m elevation to clear terrain geometry while physics settles. It falls to the ground surface and stabilizes before controllers activate.

---

### `2_ugv_fleet_brain/bot_navigation`
Nav2 stack configuration and map assets.

| File | Description |
|------|-------------|
| `config/nav2_params.yaml` | Nav2 BT navigator, controller server, planner server parameters |
| `config/slam_toolbox_params.yaml` | SLAM Toolbox online async SLAM config |
| `launch/navigation.launch.py` | Launches Nav2 stack with localization |
| `launch/slam.launch.py` | Launches SLAM Toolbox for online mapping |
| `maps/synthetic_dem.tif` | **Primary DEM** — 900×300 pixels @ 1.0 m/px, GeoTIFF, origin auto-detected. Used by TRN SLAM for terrain matching. |
| `maps/slope_map.tif` | Pre-computed slope magnitude from DEM (used by costmap) |
| `maps/continuous_costmap.tif` | Pre-computed traversability scores from slope + texture |
| `maps/calculate_costmap.py` | Script to regenerate costmap from raw DEM |
| `maps/convert_dem.py` | Converts EXR heightmap (from Gazebo export) → GeoTIFF |
| `behavior_trees/` | Custom Nav2 BT XML trees |

---

### `2_ugv_fleet_brain/custom_ackermann_controller`
The main localization and control package. All custom Python nodes live here.

#### Launch Files

| File | Description |
|------|-------------|
| `launch/terramechanic_localization.launch.py` | **Primary localization launch.** Starts (with staggered `TimerAction` delays): `imu_filter_madgwick` (0 s), `imu_covariance_fixer` (0 s), `terramechanic_odometry_node` (5 s), `factor_graph_fuser` (5 s), `local_dem_builder` (6 s), `trn_slam_node` (7 s), `odom_ground_truth_comparator` (8 s), `odom_visualizer_node` (9 s) |
| `launch/enhanced_localization.launch.py` | Alternative launch using `enhanced_wheel_odometry` + `enhanced_imu_processor` + robot_localization EKF/UKF instead of GTSAM |
| `launch/joystick_teleop.launch.py` | PS4 joystick teleoperation launch |

#### Config Files

| File | Key Parameters |
|------|----------------|
| `config/factor_graph.yaml` | `publish_rate=50`, `odom_sigma_xy=0.10`, `imu_yaw_sigma=0.07`, `slip_accel_threshold=1.5`, `slip_cov_multiplier=25.0` |
| `config/trn_slam.yaml` | `num_particles=800`, `match_rate=3.0 Hz`, `base_search_radius=50.0`, `bilateral_d=9`, `flatness_std_threshold=0.05`, `entropy_threshold=0.8` |
| `config/terramechanic_odometry.yaml` | `bekker_n=1.1`, `bekker_kc=0.9`, `bekker_kphi=1528.0` (dry desert sand), `understeer_gradient=0.08`, ZUPT thresholds |
| `config/imu_filter.yaml` | Madgwick filter params (fixed gain, no mag) |
| `config/ekf.yaml` / `ukf.yaml` | robot_localization EKF/UKF configs (used in enhanced_localization launch) |
| `config/enhanced_imu_processor.yaml` | Bias estimation window, noise density values |
| `config/enhanced_wheel_odometry.yaml` | Slip detection thresholds, adaptive covariance |

#### Python Nodes (all in `custom_ackermann_controller/`)

---

#### `ackermann_twist_controller.py` — `AckermannTwistController`

**Role:** Translates `/cmd_vel` (Twist) into Ackermann steering + wheel velocity commands.

**Subscriptions:**
- `/cmd_vel` (geometry_msgs/Twist)

**Publications:**
- `/forward_velocity_controller/commands` (std_msgs/Float64MultiArray) — [left_vel, right_vel] in rad/s
- `/forward_position_controller/commands` (std_msgs/Float64MultiArray) — [left_steer, right_steer] in rad

**Key parameters:**
- `wheelbase=0.9`, `track_width=0.67`, `wheel_radius=0.175`, `max_steering_angle=0.2616`
- `steering_rate=2.0 rad/s` — rate-limited steering transitions
- Runs at 50 Hz; 2 s timeout auto-resets steering to center

**Bicycle kinematic model:** Given linear vel `v` and angular vel `ω`, steering angle `δ = atan(wheelbase × ω / v)`. Differential wheel velocities split for inner/outer wheel to avoid slip during turns.

---

#### `factor_graph_fuser.py` + `factor_graph_core.py` — GTSAM iSAM2 Dead-Reckoning

**Role:** Fuses IMU preintegration + wheel odometry into SE(3) dead-reckoning estimate. Publishes `odom → base_footprint` TF.

**`factor_graph_fuser.py`** — ROS I/O wrapper:
- Subscriptions: `/imu/data_filtered` (Imu), `/terramechanic_odom` (Odometry)
- Publications: `/odometry/filtered` (Odometry), `odom → base_footprint` TF
- Calls `_ensure_venv()` at module load to locate `.venv/lib/python3.12/site-packages` for GTSAM
- Delegates all math to `FactorGraphCore`

**`factor_graph_core.py`** — ROS-free GTSAM core:
- State: `Pose3` (x_i), velocity Vector3 (v_i), IMU bias (b_i) — all keyed with `symbol('x'/'v'/'b', i)`
- IMU: `PreintegratedCombinedMeasurements` for each keyframe interval
- Wheel factor: `BetweenFactorPose3` on (x_{i-1}, x_i) with Gaussian noise scaled by slip ratio
- Slip gate: when `|accel_encoder - accel_imu| > slip_accel_threshold`, inflates wheel factor noise by `slip_cov_multiplier (25×)`
- Keyframe trigger: `dist > keyframe_min_dist (0.05 m)` OR `|Δθ| > keyframe_min_angle (0.02 rad)`
- Thread-safe via `threading.Lock`

**Known bug (pending fix):** Node begins integrating IMU at startup before any wheel odometry arrives. During the 5-second `TimerAction` delay for `terramechanic_odometry`, ~266 iSAM2 nodes accumulate with 5.36 s of unanchored PIM, producing ~40 m initialization drift.

---

#### `terramechanic_odometry.py` + `terramechanic_core.py` — Bekker-Wong Wheel Odometry

**Role:** Slip-aware, terrain-adaptive wheel odometry using Bekker-Wong terramechanics for desert sand.

**`terramechanic_odometry.py`** — ROS I/O wrapper:
- Subscriptions: `/joint_states` (JointState), `/imu/data_filtered` (Imu)
- Publications: `/terramechanic_odom` (Odometry) at 50 Hz, `/terramechanic/slip_detected` (Bool), `/terramechanic/slip_ratio` (Float64)
- Delegates to `TerramechanicOdometryCore`

**`terramechanic_core.py`** — ROS-free physics core:
- **Bekker-Wong model:** Computes sinkage `z = (W / (b·(kc + b·kphi)))^(1/n)` for each wheel. Effective radius `r_eff = r_nominal - z`. Slip ratio `i = (v_theoretical - v_actual) / v_theoretical`.
- **Understeer gradient:** `K_us = 0.08 rad·s²/m²` — empirical compensation for front-axle understeer in loose sand.
- **IMU gyro KF:** 2-state Kalman filter fuses kinematic yaw rate (from wheel encoder diff) with IMU gyro. Measurement noise: `R_kin=0.5` (noisy — slip corrupts), `R_imu=0.01` (accurate).
- **ZUPT:** Zero Velocity Update — zeros velocity when both `|ω| < 0.02 rad/s` and `|a| < 0.3 m/s²`.
- **Stall detection:** Flags stall when encoder velocity low but IMU acceleration high for >0.5 s; inflates covariance 100×.
- **Output covariance:** Adaptive — scales with `slip_ratio × slip_cov_gain` and `tilt_cov_gain × |sin(pitch)|`.

**Live performance (6 May 2026):** 0.34 m error after 3.5 m drive — functioning correctly.

---

#### `trn_slam_node.py` + `trn_core.py` — Terrain-Referenced Navigation

**Role:** MCL-based absolute position correction by matching local LiDAR DEM patches against the global synthetic DEM. Publishes `map → odom` TF at 3 Hz.

**`trn_slam_node.py`** — ROS I/O wrapper:
- Subscriptions: `/elevation_map/local_float` (Float32MultiArray from `local_dem_builder`), `/odometry/filtered` (Odometry — used as MCL prior)
- Publications: `map → odom` TF (TransformStamped), `/trn/match_quality` (Float64), `/trn/search_radius` (Float64), `/trn/correction` (Vector3)
- Declares ~75 parameters

**`trn_core.py`** — ROS-free MCL core:
- **Global DEM:** Reads `synthetic_dem.tif` (900×300 px @ 1 m/px, GeoTIFF). Origin auto-detected from GeoTIFF geotransform.
- **Local patch:** Composite from rolling `LocalDEMBuilderNode` output (20×20 m @ 1 m/px). Bilateral filtered (d=9, σ=15, 75).
- **MCL:** 800 particles, `ess_threshold=0.40`, slow/fast AMCL recovery (`α_slow=0.001`, `α_fast=0.02`).
- **Scoring:** Normalized Cross-Correlation (NCC) on height patches. Flat-sand gate: skips update when `std(local_patch) < 0.05 m` (insufficient terrain texture).
- **ROI:** Before computing, must verify prior is within DEM bounds (planned fix).
- **TF publish:** After accepted correction, updates `map → odom` at `match_rate=3 Hz`.

**Known bug (pending fix):** When factor-graph prior is at (31, 29) and DEM only covers ±450×±150 m, pixel coordinate underflow produces ROI of `-47474708 × -20795232 px`, crashing all match cycles.

---

#### `local_dem_builder.py` + `local_dem_pipeline.py` + `local_dem_types.py` — Rolling LiDAR DEM

**Role:** Accumulates LiDAR point clouds into a rolling spatial submap, rasterizes into a local elevation grid, and publishes it for TRN matching.

**`local_dem_builder.py`** — ROS I/O:
- Subscriptions: `/scan/points` (PointCloud2), `/terramechanic_odom` (Odometry), `/imu/data_filtered` (Imu)
- Publications: `/elevation_map/local_float` (Float32MultiArray), `/elevation_map/local` (OccupancyGrid for RViz)
- TF lookups: `odom → base_footprint` for point cloud deskewing

**`local_dem_pipeline.py`** — ROS-free core:
- **RANSAC ground segmentation:** Fits ground plane with `distance_threshold=0.15 m`, 50 iterations. Points within range `[-0.5, +1.5 m]` from ground = ground; above `+0.5 m` = obstacle.
- **Scan deskewing:** Compensates for LiDAR rotation during scan using IMU angular velocity.
- **Rolling submap:** Spatial hash grid at `5.0 m` bin resolution. Drops bins older than 50 m traveled.
- **Rasterization:** 20×20 m @ 1.0 m/px grid around current pose. Per-cell: median height of all ground-class points.

**`local_dem_types.py`:** Defines frozen dataclasses `LocalDemPipelineConfig`, `LocalDemMotionState`, `LocalDemRollingState`, `LocalDemBuildOutput`, `QueuedCloud`.

---

#### `odom_ground_truth_comparator.py` — Benchmarking Node

**Role:** Computes real-time localization error metrics (ATE, drift %) against Gazebo ground truth.

**Subscriptions:** `/ground_truth/pose` (Pose), `/odometry/filtered` (Odometry), `/terramechanic_odom` (Odometry)

**Publications:** `/odom_error/ekf/{position_error, heading_error, drift_percent, ate}`, `/odom_error/raw/{...}`, `/odom_error/summary` (String)

**Also writes:** timestamped CSV at `/tmp/odom_ground_truth_log.csv` for offline `evo_ape` / `evo_rpe` analysis.

**Known bug (pending fix):** Subscribes to `/odometry/filtered` (odom-frame) instead of looking up `map → base_footprint` TF. The EKF error column reflects the diverged odom-frame pose, not the TRN-corrected map-frame pose. Fix: add `tf2_ros.Buffer` + `TransformListener`, look up `map → base_footprint` at each benchmark tick.

---

#### `odom_visualizer_node.py` — Live Matplotlib Visualizer

**Role:** Live 2×2 matplotlib dashboard showing GT/localized/raw trajectories, position error, heading error, and TRN diagnostics.

**Subscriptions:** `/odometry/filtered`, `/ground_truth/pose`, `/terramechanic_odom`, `/trn/match_quality`, `/trn/search_radius`, `/trn/correction`

**TF lookups:** `map → base_footprint` (for the "Localized" blue trace — reflects TRN correction)

**Layout:**
- `[0,0]` XY trajectory — GT (green), Localized (blue), Raw odom (red dashed)
- `[0,1]` Position error vs GT over time
- `[1,0]` Heading error vs GT over time
- `[1,1]` TRN diagnostics — MAD likelihood, correction magnitude, drift %

**Blocked by:** TRN never publishes `map → odom` (due to factor-graph init bug → TRN ROI crash), so the TF lookup for the blue trace always fails. Unblocked automatically once Changes 2 + 3 are fixed.

---

#### `imu_covariance_fixer.py` — IMU Covariance Relay

**Role:** Patches all-zero IMU covariance matrices from `ros_gz_bridge` before passing to `robot_localization` UKF.

**Subscriptions:** `/imu/data_filtered` (Imu from Madgwick)  
**Publications:** `/imu/data_filtered_cov` (Imu with realistic covariances)

**Injected values:**
- Orientation: `1e-4` (0.01 rad²)
- Angular velocity: `4e-8` (2×10⁻⁴ rad/s)²
- Linear acceleration: `1e-3` (conservative, covers gravity residual)

**Why needed:** Gazebo's `ros_gz_bridge` outputs zero-filled covariance arrays. `robot_localization` treats zero covariance as "measurement disabled" or, in the UKF path, causes Cholesky decomposition failure (Kalman gain collapses P to zero → NaN explosion on next update).

---

#### `enhanced_imu_processor.py` — Legacy Enhanced IMU Node

**Role:** Online gyro/accel bias estimation with temperature compensation. Predecessor to the GTSAM IMU preintegration path.

Used only in `enhanced_localization.launch.py`. Not active in the main terramechanic localization stack.

**Key features:** 1000-sample bias estimation window, temperature correction polynomial, outputs `sensor_msgs/Imu` with corrected values + `Float64MultiArray` noise characterization.

---

#### `enhanced_wheel_odometry.py` — Legacy Enhanced Wheel Odometry

**Role:** Ackermann wheel odometry with exponential velocity filter, wheel acceleration clamping, adaptive covariance. Predecessor to `TerramechanicOdometryNode`.

Used only in `enhanced_localization.launch.py`. Not active in the main terramechanic localization stack.

---

### `2_ugv_fleet_brain/emcon_controller`

| File | Description |
|------|-------------|
| `emcon_controller/emcon_node.py` | `EmconController` node — arbitrates `/cmd_vel_raw` (from SBLP) based on `/emcon_state` (Bool). When EMCON active: scales linear by 0.3, clamps angular to ±0.1 rad/s. Publishes to `/cmd_vel`. |

**EMCON modes:**
- `emcon_state=False` (clear): `/cmd_vel` = pass-through of `/cmd_vel_raw`
- `emcon_state=True` (active): slow creep mode — 30% speed, minimal turning to reduce RF signature from motor noise

---

### `2_ugv_fleet_brain/sblp_planner`

| File | Description |
|------|-------------|
| `sblp_planner/sblp_node.py` | `SBLPPlanner` node — Scenario-Based Local Planner. Subscribes `/terrain_class` (Int8 from VLM) + `/goal_pose` (PoseStamped). Publishes `/cmd_vel_raw` (Twist) + `/sblp/scenario` (String). |

**Terrain class → trajectory primitive mapping:**
- `0 = open_corridor` → sprint at `max_linear_vel=1.5 m/s`
- `1 = sand_dune` → cautious S-curve at 40% speed + sinusoidal angular velocity
- `2 = rock_field` → stop and assess (zero velocity, publish warning)

---

### `2_ugv_fleet_brain/vlm_costmap`

| File | Description |
|------|-------------|
| `vlm_costmap/vlm_costmap_node.py` | `VLMCostmapNode` — zero-shot CLIP traversability classifier. Subscribes `/camera/image_raw`. Publishes `/terrain_class` (Int8) + `/vlm_costmap/score` (Float32). Falls back to STUB mode if HuggingFace `transformers` not installed. |

**Zero-shot prompts:**
- Class 0: `"open sandy desert corridor, safe to drive"`
- Class 1: `"sand dune slope, difficult terrain"`
- Class 2: `"rocky ground, dangerous for vehicle"`

Runs at 2 Hz. Lazy model load on first image (avoids startup delay).

---

### `2_ugv_fleet_brain/LIO-SAM`
Upstream LiDAR-Inertial Odometry (LOAM variant) included as a git submodule for potential future integration. **Not currently used** in the main localization stack (TRN + terramechanics path is active instead). Retained for comparison experiments.

---

### `3_deployment/docker`

| File | Description |
|------|-------------|
| `Dockerfile` | ROS 2 Jazzy base + Gazebo Harmonic + Python deps. Multi-stage build. |
| `docker-compose.yml` | 3-robot deployment: `alpha`, `beta`, `gamma` bots each with isolated `ROS_DOMAIN_ID`. Shared `thar_desert` world server. |

---

## 4. Data Flow Summary

```
Gazebo physics
    │
    ├── gz-transport → EmconSystemInterface → ros2_control
    │                                              │
    │                               ┌──────────────┼──────────────────┐
    │                    joint_state_broadcaster  fwd_pos_ctrl  fwd_vel_ctrl
    │                                              │
    │                                    ackermann_twist_controller
    │                                         ← /cmd_vel
    │
    ├── ros_gz_bridge → /imu/data_raw → imu_filter_madgwick → /imu/data_filtered
    │                                                               │
    │                                              imu_covariance_fixer → /imu/data_filtered_cov
    │
    ├── ros_gz_bridge → /joint_states → terramechanic_odometry_node → /terramechanic_odom
    │
    ├── ros_gz_bridge → /scan/points → local_dem_builder → /elevation_map/local_float
    │
    └── ros_gz_bridge → /ground_truth/pose → odom_ground_truth_comparator


Localization TF tree (REP-105):
    map ──[trn_slam_node @ 3Hz]──→ odom ──[factor_graph_fuser @ 50Hz]──→ base_footprint

    /imu/data_filtered  ──────────────────────────────────┐
    /terramechanic_odom ──────────────────────────────────┴──→ factor_graph_fuser → odom→base_footprint TF
                                                                                        + /odometry/filtered

    /elevation_map/local_float ────────────────────────────────→ trn_slam_node → map→odom TF
    /odometry/filtered (as prior) ─────────────────────────────→ trn_slam_node


Navigation / Planning:
    /camera/image_raw → vlm_costmap_node → /terrain_class
    /terrain_class + /goal_pose → sblp_planner → /cmd_vel_raw
    /cmd_vel_raw + /emcon_state → emcon_controller → /cmd_vel
```

---

## 5. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| ROS/core separation (e.g. `factor_graph_core.py` + `factor_graph_fuser.py`) | Allows unit testing without a ROS spin loop. Pure Python is testable with `pytest`. |
| gz-transport for hardware interface | Eliminates one DDS serialization hop on every 100 Hz control cycle. Gazebo's native API. |
| GTSAM over `robot_localization` EKF for dead-reckoning | SE(3) preintegrated IMU factors handle large orientation changes correctly. robot_localization UKF diverges on high angular rates. |
| Terramechanics (Bekker-Wong) over simple encoder odometry | Desert sand produces 15–40% slip. Simple encoder odom would accumulate 3–8 m/100 m error. Bekker model reduces this to ~0.34 m/100 m. |
| TRN via DEM matching (not LiDAR SLAM) | SLAM (LIO-SAM) requires loop closures for drift correction — not available in featureless sand. DEM matching provides absolute correction at 3 Hz. |
| Zero-shot CLIP for traversability | No labeled desert traversability dataset exists. Zero-shot avoids annotation cost; STUB fallback enables graceful degradation. |
| Fixed `TimerAction` delays in localization launch | Simple to implement. Known limitation: not synchronized to robot stillness or sensor readiness. Planned replacement with readiness-based gates. |

---

## 6. Current Build & Run Instructions

```bash
# Build all packages
cd /home/sailesh/silent_sentry
colcon build --symlink-install
source install/setup.bash

# Build single package
colcon build --packages-select bot_controller --symlink-install

# Launch simulation (two terminals)
# Terminal 1 — Gazebo world
ros2 launch base_station_bringup world_only.launch.xml

# Terminal 2 — Robot stack
ros2 launch fleet_bringup robot_only.launch.xml bot_name:=alpha

# Terminal 3 — Localization stack
ros2 launch custom_ackermann_controller terramechanic_localization.launch.py model_name:=alpha

# Check controller status
ros2 control list_controllers

# Send test motion command
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.6}, angular: {z: 0.0}}"
```

---

## 7. Pending Fixes (Priority Order)

| Priority | File | Fix |
|----------|------|-----|
| 1 (root cause) | `factor_graph_fuser.py` | Gate first IMU preintegration on receipt of at least one `/terramechanic_odom`. Reset iSAM2 graph to identity on first wheel tick if drift detected. |
| 2 (defensive guard) | `trn_slam_node.py` | Prior sanity gate: skip match cycle if prior is outside `[dem_origin_x, dem_origin_x + dem_width] × [dem_origin_y, dem_origin_y + dem_height]`. |
| 3 | `odom_ground_truth_comparator.py` | Add TF lookup `map → base_footprint` for EKF error column instead of subscribing `/odometry/filtered` directly. |
| 4 | `terramechanic_localization.launch.py` | Replace `TimerAction` fixed delays with readiness-based gates (spin until first message on `/imu/data_filtered` before terramechanic, spin until first `/terramechanic_odom` before factor-graph). |
