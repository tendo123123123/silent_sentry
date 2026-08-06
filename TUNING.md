# Silent Sentry — Navigation & Localization Tuning Guide

This document is the single reference for tuning the autonomous patrol stack:
SBLP patrol brain → Nav2 (global planner + local controller + costmaps) →
Ackermann controller → Gazebo, plus the TRN localization backend and the
DEM-prior obstacle detector that feed them.

It exists because the stack can settle into a **planning/execution deadlock**
(the "equilibrium"): the robot stops, the global planner reports
`no valid path found`, the controller reports `detected collision ahead` /
`Controller patience exceeded`, and recovery behaviours (clear costmap → wait →
backup) loop without progress. Every parameter that contributes is catalogued
below, in the order you should tune them.

---

## 0. The deadlock, decomposed

Observed log signature:

```
planner_server: GridBased plugin failed to plan from (17.46, 5.77) to (72.75, 29.05): "no valid path found"
controller_server: RegulatedPurePursuitController detected collision ahead!
controller_server: Controller patience exceeded
behavior_server: Collision Ahead - Exiting DriveOnHeading / backup failed
planner_server: Planner loop missed its desired rate of 20.0000 Hz. Current loop rate is 3.0303 Hz
```

Root causes, in priority order:

| # | Cause | Evidence | Subsystem |
|---|-------|----------|-----------|
| 1 | **Goal outside reachable window.** Lévy step (`l_max=60`) ≥ global costmap half-window (60 m). Goal lands at/over the edge. | `plan from (17,5) to (72,29)` = 59 m; repeated `no valid path found` | SBLP + global_costmap |
| 2 | **Global planner too slow.** Smac Hybrid-A* over 400×400 cells @0.3 with 72 angle bins. | `Planner loop ... 3.03 Hz` | planner_server |
| 3 | **Controller collision lock.** RPP projects forward, sees inflated/real obstacle on the only available path, refuses to move. | `detected collision ahead` × N → `patience exceeded` | controller_server + costmap inflation |
| 4 | **Futile recovery loop.** Goal is unreachable, so clear/wait/backup can never help; backup itself hits `Collision Ahead`. | `Running backup` → `backup failed` → `Running wait` | behavior_server + BT |
| 5 | **Costmap churn from localization jumps.** TRN `map→odom` corrections (3 Hz) shift obstacles in the map frame; local costmap marks/clears abruptly. | local path changes abruptly; ATE drift in monitor | trn_slam + costmaps |

Fix order: **1 → 2 → 3 → 4 → 5.** Do not tune the controller before the
planner can produce a path, and do not tune localization churn before the
geometry is reachable.

---

## 1. Reachability — SBLP step vs global costmap window (FIX FIRST)

The single most important invariant:

> **`SBLP l_max` must be comfortably smaller than the global costmap half-window.**
> Rule of thumb: `l_max ≤ 0.6 × (costmap_width / 2)`.

With `width=120` → half-window 60 m → `l_max` should be ≤ ~36 m. It is currently
60 m, so goals routinely land on the costmap boundary where no path exists.

Two levers (apply both):

| Parameter | File | Was | **Applied** | Why |
|-----------|------|-----|-------------|-----|
| `l_max` | `sblp_planner/sblp_goal_generator.py` (declare) / launch | `60.0` | **`35.0`** | Keep Lévy goals well inside the reachable window |
| `l_min` | same | `8.0` | **`6.0`** | Allow shorter hops so the robot isn't forced into long risky legs |
| `width` / `height` | `bot_navigation/config/nav2_params.yaml` → `global_costmap` | `120` | **`140`** | Half-window 70 m gives margin over `l_max=35` |
| `resolution` | same `global_costmap` | `0.3` | **`0.4`** | Fewer cells (see §2) so the bigger window stays real-time |

Also confirm the geofence is larger than a step so goals aren't clipped:
`geofence_polygon` = `[-100,-75, 100,-75, 100,75, -100,75]` (200×150 m) — fine.

Validation: `ros2 topic echo /sblp/current_waypoint` — every goal should be
within `l_max` of the robot, and `planner_server` should return a path within
~1 s instead of `no valid path found`.

---

## 2. Global planner speed — Smac Hybrid-A*

Target: planner returns in < 0.5 s so the 1 Hz `RateController` replan in the BT
is never starved. Current `3.03 Hz` (≈0.33 s) is borderline but degrades badly
on long spans; combined with the unreachable goal it wastes the full
`primitive_search_max_duration_ms`.

| Parameter | Was | **Applied** | Why |
|-----------|-----|-------------|-----|
| `resolution` (global_costmap) | `0.3` | **`0.4`** | 400×400 → 350×350 cells; ~25% fewer nodes |
| `angle_quantization_bins` | `72` | **`48`** | Search space scales with bins; 48 (7.5°) is ample for R=3.36 m |
| `primitive_search_max_duration_ms` | `1500` | **`800`** | Fail fast and let the BT replan instead of blocking |
| `analytic_expansion_ratio` | `3.5` | `3.5` | Keep (already reverted from the reversing tweak) |
| `tolerance` | `0.5` | `0.5` | Goal position tolerance; fine |
| `cost_penalty` | `2.0` | `2.0` | Keep; raise only if paths hug obstacles |
| `allow_unknown` | `true` | `true` | Rolling window has unknown fringe; must stay true |

If planning is still slow, drop `max_lookahead`/window before touching the
motion model. Do **not** switch off `REEDS_SHEPP` (reverse is wanted as a last
resort — `reverse_penalty=5.0` already makes it rare).

---

## 3. Controller — Regulated Pure Pursuit (collision lock)

`detected collision ahead` means RPP's forward collision check finds cost on the
path within its horizon and zeroes the command. On open terrain this fires when
inflation is too fat or a phantom/real obstacle sits on the only path.

| Parameter | File | Current | Suggested | Why |
|-----------|------|---------|-----------|-----|
| `use_collision_detection` | nav2_params `FollowPath` | `true` | `true` | Keep — genuine safety |
| `max_allowed_time_to_collision_up_to_carrot` | `FollowPath` | `1.5` | **`1.0` applied** | Shorter horizon → stops less eagerly on distant/edge cost |
| `desired_linear_vel` | `FollowPath` | `0.5` | `0.5` | Patrol cruise; fine |
| `lookahead_dist` | `FollowPath` | `1.5` | `1.5` | Fine for 0.9 m wheelbase |
| `min_lookahead_dist` | `FollowPath` | `0.8` | `0.8` | Keep |
| `max_lookahead_dist` | `FollowPath` | `3.5` | `3.5` | Keep ≤ local costmap half-size (5 m) |
| `use_regulated_linear_velocity_scaling` | `FollowPath` | `true` | `true` | Smooth curvature slow-down |
| `regulated_linear_scaling_min_speed` | `FollowPath` | `0.4` | `0.4` | Crawl floor so it never stalls on arcs |
| `use_cost_regulated_linear_velocity_scaling` | `FollowPath` | `false` | `false` | Keep OFF until obstacle marking is proven clean |
| `failure_tolerance` | controller_server | `0.3` | **`0.5` applied** | More slack before aborting to recovery |
| `controller_frequency` | controller_server | `10.0` | `10.0` | Achievable; keep |

### Costmap inflation (the usual collision-lock culprit)

Over-inflation closes the narrow gaps a car-like robot needs, so both the
planner (`no path`) and controller (`collision ahead`) fail.

| Parameter | Layer | Current | Suggested | Why |
|-----------|-------|---------|-----------|-----|
| `inflation_radius` | global inflation | `0.9` | **`0.6` applied** | 0.9 m around a 0.4 m half-width robot leaves little free space |
| `cost_scaling_factor` | global inflation | `3.0` | `3.0` | Keep the falloff steep |
| `inflation_radius` | local inflation | `0.5` | **`0.45` applied** | Just over robot half-width + margin |
| `footprint` | both | `[[-0.7,-0.4]...]` | keep | Matches 1.4×0.8 m chassis |

---

## 4. Recovery behaviours & Behavior Tree

Recoveries only help when the goal is *reachable*. Once §1 is fixed these fire
rarely. Tune so they don't waste time when they can't help.

| Parameter | File | Current | Suggested | Why |
|-----------|------|---------|-----------|-----|
| `number_of_retries` (NavigateRecovery) | `behavior_trees/ackermann_to_bt.xml` | `6` | **`3` applied** | Fail the goal sooner → SBLP picks a new reachable waypoint |
| `RateController hz` | same BT | `1.0` | `1.0` | Replan cadence; raise to 2.0 only if CPU allows |
| `Wait wait_duration` | same BT | `5.0` | `5.0` | Fine |
| `BackUp backup_dist` (recovery) | same BT | `0.30` | `0.30` | Kept as last-resort extraction; short & slow |
| `movement_time_allowance` (progress_checker) | nav2_params | `15.0` | **`10.0` applied** | Detect "stuck" faster |
| `required_movement_radius` | nav2_params | `0.5` | `0.5` | Keep |
| `goal_timeout_s` (SBLP) | sblp_goal_generator | `90.0` | **`45.0` applied** | Abandon a bad waypoint sooner; less time wedged |

---

## 5. Localization churn — TRN (feeds costmap stability)

TRN `map→odom` jumps move obstacles in the map frame, so the local costmap
marks/clears abruptly and the local path "changes abruptly." Covered in detail
by the terrain-aliasing tuning already applied; the relevant knobs:

| Parameter | File | Current | Suggested | Why |
|-----------|------|---------|-----------|-----|
| `base_search_radius` / `initial_search_radius` | `ugv_localization/config/trn_slam.yaml` | `15.0` | `15.0` | Restored from 5 m; keeps true peak in ROI |
| `entropy_threshold` | trn_slam.yaml | `1.2` | `1.2` → up to `1.6` | Reject featureless-dune aliasing (false confident locks) |
| `min_peak_quality` | trn_slam.yaml | `0.70` | `0.70` | Reject marginal matches |
| `motion_noise_xy_frac` | trn_slam.yaml | `0.15` | `0.15` → `0.20` | More spread to recover from a bad lock |
| `amcl_random_fraction` | trn_slam.yaml | `0.10` | `0.10` → `0.15` | Faster kidnap recovery |

Metric to watch (odom_visualizer): **ATE bounded (< ~2 m), not likelihood.**
High MAD likelihood + rising ATE = confident wrong lock → raise
`entropy_threshold`.

---

## 6. Obstacle detector (DEM-prior)

Phantom obstacles = costmap marks free ground → planner/controller both fail.

| Parameter | File | Current | Suggested | Why |
|-----------|------|---------|-----------|-----|
| `tau_prior` | `ugv_obstacle/config/obstacle.yaml` | `0.4` | `0.4` | Height over DEM to call obstacle |
| `tau_local` | obstacle.yaml | `0.4` | `0.4` | Local 3×3 jump threshold |
| `low_conf_relax` | obstacle.yaml | `0.2` | `0.2` | Stay cautious when TRN confidence low |
| `self_radius` | obstacle.yaml | `1.2` | **`1.5` applied** | Mast LiDAR sees own chassis past 1.2 m |
| `min_points_per_cell` | obstacle.yaml | `2` | **`3` applied** | Reject sparse LiDAR noise |
| `cell_size` | obstacle.yaml | `0.4` | `0.4` | Local-jump grid |

Note: the `classify()` path (feeds `/scan/obstacles` → costmap) now
auto-calibrates the z-datum vs the DEM, so a constant map/DEM offset no longer
produces phantoms. The `buildAndPublishGrid()` path (`/obstacle/grid`,
`/obstacle/costmap`) still uses the raw absolute diff — not consumed by the
costmap today, cosmetic only.

---

## 7. Ackermann / terramechanics (secondary)

Rarely the cause of the deadlock, but relevant to smoothness and to the wheel-
slip drift that feeds TRN.

| Parameter | File | Current | Note |
|-----------|------|---------|------|
| `max_steering_angle` | `ackermann_twist_controller.py` | `0.2616` (15°) | Ties to `minimum_turning_radius=3.36`; keep consistent |
| `understeer_gradient` | `terramechanic_odometry.yaml` | `0.08` | Sand understeer; raise if turns undershoot |
| `stall_detection_enabled` | terramechanic_odometry.yaml | `true` | De-weights encoders on collision slip |
| `stall_covariance_multiplier` | terramechanic_odometry.yaml | `100.0` | Inflation during stall |
| `sand_slip_coefficient` | terramechanic_odometry.yaml | `0.4` | Gravity slip on slopes |

Kinematic consistency check: `minimum_turning_radius (planner) == wheelbase /
tan(max_steering_angle)` → `0.9 / tan(0.2616) = 3.37 m` ≈ `3.36`. Keep these two
in lock-step; if you change one, change the other.

---

## 8. First-pass change set — **APPLIED**

These target the deadlock directly. All are in the tree now:

1. ✅ `sblp_goal_generator.py`: `l_max 60 → 35`, `l_min 8 → 6`.
2. ✅ `nav2_params.yaml` global_costmap: `width/height 120 → 140`, `resolution 0.3 → 0.4`.
3. ✅ `nav2_params.yaml` planner: `angle_quantization_bins 72 → 48`, `primitive_search_max_duration_ms 1500 → 800`.
4. ✅ `nav2_params.yaml` inflation: global `inflation_radius 0.9 → 0.6`, local `0.5 → 0.45`.
5. ✅ `nav2_params.yaml` controller: `max_allowed_time_to_collision_up_to_carrot 1.5 → 1.0`, `failure_tolerance 0.3 → 0.5`.
6. ✅ `ackermann_to_bt.xml`: `number_of_retries 6 → 3`.
7. ✅ `sblp_goal_generator.py`: `goal_timeout_s 90 → 45`.

Plus, from §4 and §6:

8. ✅ `nav2_params.yaml`: `movement_time_allowance 15 → 10`.
9. ✅ `obstacle.yaml`: `self_radius 1.2 → 1.5`, `min_points_per_cell 2 → 3`.

Rebuild `bot_navigation` + `sblp_planner` + `ugv_obstacle`, relaunch nav2_trn +
sblp_nav2.

### Deliberately NOT applied (diagnostic-gated)

These depend on what the run actually shows; applying them blind can make things
worse. Change only in response to the stated symptom:

| Parameter | Apply when |
|-----------|------------|
| `entropy_threshold` 1.2 → 1.6 (trn_slam.yaml) | MAD likelihood ~1.0 **and** ATE still climbing (false dune lock persists) |
| `entropy_threshold` 1.2 → 1.0 | Matcher aborts so often that dead-reckoning drift dominates (long gaps between corrections) |
| `motion_noise_xy_frac` 0.15 → 0.20 | Filter can't recover after a bad lock (particles too tight) |
| `amcl_random_fraction` 0.10 → 0.15 | Kidnap recovery sluggish |
| `RateController hz` 1.0 → 2.0 (BT) | Reroutes feel laggy **and** CPU headroom exists (planner comfortably meeting rate) |
| `understeer_gradient` 0.08 ↑ | Turns consistently undershoot commanded arc |

## 9. Validation checklist

- [ ] `planner_server` returns a path in < 1 s, no `no valid path found` on open ground.
- [ ] `ros2 topic echo /sblp/current_waypoint` — goals within `l_max` of robot.
- [ ] Robot maintains ~0.5 m/s, smooth arcs (R ≥ 3.36 m), no in-place freeze.
- [ ] `detected collision ahead` only near real obstacles, not open sand.
- [ ] Recovery (`Running backup/wait`) is rare, not a loop.
- [ ] odom_visualizer ATE bounded (< ~2 m), not climbing.
- [ ] Local costmap clean on open terrain (no phantom blobs).

## 10. Golden rules

1. Tune **geometry/reachability before speed before control before localization.**
2. Change **one lever at a time**, watch one metric, then the next.
3. `l_max < 0.6 × global half-window` — never break this.
4. Keep `minimum_turning_radius` and `max_steering_angle` consistent.
5. Prefer failing a goal fast (SBLP re-selects) over long recovery loops.
