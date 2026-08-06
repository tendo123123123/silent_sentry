#!/usr/bin/env python3
"""
SBLP Nav2 Goal Generator
========================
Runs the Spatially-Bounded Lévy Patrol as a *behavior layer on top of Nav2*.
Instead of publishing raw /cmd_vel (no obstacle awareness), it generates
heavy-tailed, geo-fenced, terrain-gated Lévy waypoints and dispatches each one
as a nav2_msgs/NavigateToPose goal. Nav2 then performs the actual obstacle
avoidance, terrain-aware global planning (Smac Hybrid-A* over the terrain +
LiDAR costmaps), and control. When a goal succeeds, aborts, or times out, the
next Lévy waypoint is generated — so patrol continues unpredictably and safely.

This keeps concerns cleanly separated:
  * SBLP  -> WHERE to go (unpredictable, sector-bounded, avoids lethal slopes)
  * Nav2  -> HOW to get there safely (path planning + obstacle avoidance)

Subscribes:
  /sblp/micro_burst  (std_msgs/String) — Base Station elastic geo-fence stretch
Publishes:
  /sblp/current_waypoint (geometry_msgs/PoseStamped) — active goal (RViz)
  /sblp/status           (std_msgs/String)           — telemetry
Action client:
  navigate_to_pose (nav2_msgs/action/NavigateToPose)
TF:
  map -> base_footprint (current pose for waypoint generation)
"""
import json
import math
import random

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

import tf2_ros
from nav2_msgs.action import NavigateToPose

from sblp_planner.sblp_core import SBLPConfig, SBLPCore, reshape_flat_polygon
from sblp_planner.sblp_terrain import load_terrain_costmap


class SBLPGoalGenerator(Node):
    def __init__(self):
        super().__init__('sblp_goal_generator')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('max_linear_vel', 1.5)
        self.declare_parameter('max_angular_vel', 0.6)
        self.declare_parameter('levy_beta', 1.8)
        self.declare_parameter('l_min', 6.0)
        # l_max MUST stay < 0.6 * global costmap half-window (140m -> 70m half
        # -> 42m cap) so Lévy goals land well inside the reachable rolling
        # window; 60m == half-window caused constant "no valid path found".
        self.declare_parameter('l_max', 35.0)
        self.declare_parameter('turn_sigma_rad', 1.4)
        self.declare_parameter('reorient_probability', 0.05)
        self.declare_parameter('max_rejection_attempts', 50)
        self.declare_parameter('seed', -1)
        self.declare_parameter('goal_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('goal_timeout_s', 45.0)
        self.declare_parameter('geofence_polygon',
                               [-100.0, -75.0, 100.0, -75.0, 100.0, 75.0, -100.0, 75.0])
        # Terrain gating (optional): reject waypoints on lethal-slope terrain.
        self.declare_parameter('use_terrain_gating', False)
        self.declare_parameter('terrain_costmap_path', '')
        self.declare_parameter('terrain_origin_x', -450.0)
        self.declare_parameter('terrain_origin_y', -150.0)
        self.declare_parameter('terrain_resolution', 1.0)
        self.declare_parameter('terrain_cost_threshold', 0.7)
        # True = match nav2_map_server's row order (image row 0 = max y), so
        # SBLP agrees with the planner's static layer. See sblp_terrain.py for
        # the unresolved TRN-vs-map_server orientation inconsistency.
        self.declare_parameter('terrain_flip_y', True)

        gp = self.get_parameter
        seed = gp('seed').value
        rng = random.Random(seed) if seed is not None and seed >= 0 else random.Random()
        self.goal_frame = gp('goal_frame').value
        self.base_frame = gp('base_frame').value
        self.goal_timeout_s = gp('goal_timeout_s').value

        config = SBLPConfig(
            max_linear_vel=gp('max_linear_vel').value,
            max_angular_vel=gp('max_angular_vel').value,
            levy_beta=gp('levy_beta').value,
            l_min=gp('l_min').value,
            l_max=gp('l_max').value,
            turn_sigma_rad=gp('turn_sigma_rad').value,
            reorient_probability=gp('reorient_probability').value,
            max_rejection_attempts=gp('max_rejection_attempts').value,
            terrain_cost_threshold=gp('terrain_cost_threshold').value,
            geofence_polygon=reshape_flat_polygon(list(gp('geofence_polygon').value)),
        )

        terrain_fn = None
        if gp('use_terrain_gating').value:
            path = gp('terrain_costmap_path').value
            costmap = load_terrain_costmap(
                path,
                origin_x=gp('terrain_origin_x').value,
                origin_y=gp('terrain_origin_y').value,
                resolution=gp('terrain_resolution').value,
                flip_y=gp('terrain_flip_y').value)
            terrain_fn = costmap.cost_at
            self.get_logger().info(
                f'SBLP: terrain gating ON (costmap: {path}, '
                f'flip_y={gp("terrain_flip_y").value}, '
                f'threshold={gp("terrain_cost_threshold").value})')

        self.core = SBLPCore(config, rng=rng, terrain_cost_fn=terrain_fn)

        # ── State ─────────────────────────────────────────────────────────
        self.busy = False
        self._goal_start = None
        self._goal_handle = None

        # ── TF for current map-frame pose ─────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── I/O ───────────────────────────────────────────────────────────
        self.create_subscription(String, '/sblp/micro_burst', self._micro_burst_cb, 10)
        self._wp_pub = self.create_publisher(PoseStamped, '/sblp/current_waypoint', 10)
        self._status_pub = self.create_publisher(String, '/sblp/status', 10)

        self._nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_timer(1.0, self._tick)
        self.get_logger().info('SBLP Nav2 goal generator ready (waiting for Nav2 action server)')

    # ── Current pose via TF ─────────────────────────────────────────────
    def _current_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.goal_frame, self.base_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    # ── Main loop ────────────────────────────────────────────────────────
    def _tick(self):
        # Enforce a per-goal timeout so a hard-to-reach waypoint never stalls patrol.
        if self.busy and self._goal_start is not None:
            age = (self.get_clock().now() - self._goal_start).nanoseconds / 1e9
            if age > self.goal_timeout_s:
                self.get_logger().warn(
                    f'SBLP: goal timed out after {age:.0f}s; canceling and reselecting.')
                if self._goal_handle is not None:
                    self._goal_handle.cancel_goal_async()
                self.busy = False
            return
        if self.busy:
            return

        pose = self._current_pose()
        if pose is None:
            self.get_logger().info('SBLP: waiting for map->base_footprint TF...',
                                   throttle_duration_sec=5.0)
            return
        if not self._nav.server_is_ready():
            self._nav.wait_for_server(timeout_sec=0.0)
            self.get_logger().info('SBLP: waiting for Nav2 navigate_to_pose server...',
                                   throttle_duration_sec=5.0)
            return

        x, y, yaw = pose
        wp = self.core.generate_waypoint(x, y, yaw)
        self._send_goal(wp)

    def _send_goal(self, wp):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.goal_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = wp.x
        goal.pose.pose.position.y = wp.y
        goal.pose.pose.orientation.z = math.sin(wp.yaw * 0.5)
        goal.pose.pose.orientation.w = math.cos(wp.yaw * 0.5)

        self._wp_pub.publish(goal.pose)
        self._status_pub.publish(String(data=json.dumps({
            'scenario': 'levy_patrol_nav2',
            'source': wp.source,
            'levy_beta': self.core.beta,
            'target': [round(wp.x, 2), round(wp.y, 2)],
            'step_len': round(wp.step_len, 2),
        })))

        self.busy = True
        self._goal_start = self.get_clock().now()
        self.get_logger().info(
            f'SBLP: dispatching Lévy goal ({wp.x:.1f}, {wp.y:.1f}) '
            f'[{wp.source}, l={wp.step_len:.1f}m] to Nav2')
        self._nav.send_goal_async(goal).add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('SBLP: Nav2 rejected goal; reselecting next tick.')
            self.busy = False
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        status = future.result().status
        # 4=SUCCEEDED, 5=CANCELED, 6=ABORTED (action_msgs/GoalStatus)
        label = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(status, str(status))
        self.get_logger().info(f'SBLP: goal finished [{label}]; generating next Lévy waypoint.')
        self.busy = False
        self._goal_handle = None

    def _micro_burst_cb(self, msg: String):
        try:
            applied = self.core.apply_micro_burst(msg.data)
            self.get_logger().warn(f'SBLP: elastic geo-fence micro-burst applied: {applied}')
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'SBLP: failed to parse micro-burst: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SBLPGoalGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
