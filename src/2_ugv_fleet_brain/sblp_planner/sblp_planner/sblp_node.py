#!/usr/bin/env python3
"""
SBLP Planner Node
=================
Spatially-Bounded Lévy Patrol (SBLP) ROS 2 wrapper. All planning mathematics
live in :mod:`sblp_planner.sblp_core`; this node handles ROS I/O only.

Guarantees mathematically unpredictable, geo-fenced sector coverage with zero
inter-agent communication by replacing deterministic waypoints with a
heavy-tailed Lévy distribution P(l) ~ l^(-beta), 1 < beta <= 3, respected by a
rejection sampler that keeps the agent inside its assigned polygon. Navigation
uses continuous-curvature (pure-pursuit) Ackermann steering.

Subscribes:
  /goal_pose          (geometry_msgs/PoseStamped) — optional external override
  /odometry/filtered  (nav_msgs/Odometry)         — robot pose estimate
  /terramechanic_odom (nav_msgs/Odometry)         — fallback pose estimate
  /sblp/micro_burst   (std_msgs/String)           — Base Station parameter stretch

Publishes:
  /cmd_vel_raw            (geometry_msgs/Twist)       — Ackermann velocity command
  /sblp/scenario          (std_msgs/String)           — active primitive name
  /sblp/status            (std_msgs/String)           — telemetry / Lévy metrics
  /sblp/current_waypoint  (geometry_msgs/PoseStamped) — active waypoint (RViz)
"""
import json
import math
import random

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry

from sblp_planner.sblp_core import SBLPConfig, SBLPCore, reshape_flat_polygon


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SBLPPlanner(Node):
    def __init__(self):
        super().__init__('sblp_planner')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('max_linear_vel', 1.5)
        self.declare_parameter('max_angular_vel', 0.6)
        self.declare_parameter('levy_beta', 1.8)
        self.declare_parameter('l_min', 3.0)
        self.declare_parameter('l_max', 35.0)
        self.declare_parameter('waypoint_tolerance', 1.5)
        self.declare_parameter('max_rejection_attempts', 50)
        self.declare_parameter('seed', -1)  # <0 → nondeterministic
        # Geo-fence polygon as a FLAT [x0,y0,x1,y1,...] double array. ROS 2
        # parameters do not support nested lists, so a flat array is required.
        self.declare_parameter(
            'geofence_polygon',
            [-60.0, -60.0, 60.0, -60.0, 60.0, 60.0, -60.0, 60.0])

        gp = self.get_parameter
        seed = gp('seed').value
        rng = random.Random(seed) if seed is not None and seed >= 0 else random.Random()

        config = SBLPConfig(
            max_linear_vel=gp('max_linear_vel').value,
            max_angular_vel=gp('max_angular_vel').value,
            levy_beta=gp('levy_beta').value,
            l_min=gp('l_min').value,
            l_max=gp('l_max').value,
            waypoint_tolerance=gp('waypoint_tolerance').value,
            max_rejection_attempts=gp('max_rejection_attempts').value,
            geofence_polygon=reshape_flat_polygon(list(gp('geofence_polygon').value)),
        )
        self.core = SBLPCore(config, rng=rng)

        # ── Robot state ───────────────────────────────────────────────────
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_received = False
        self.target = None          # (x, y, yaw)
        self.waypoint_source = 'none'

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_subscription(Odometry, '/terramechanic_odom', self._fallback_odom_cb, 10)
        self.create_subscription(String, '/sblp/micro_burst', self._micro_burst_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        self._scenario_pub = self.create_publisher(String, '/sblp/scenario', 10)
        self._status_pub = self.create_publisher(String, '/sblp/status', 10)
        self._waypoint_pub = self.create_publisher(PoseStamped, '/sblp/current_waypoint', 10)

        self.create_timer(0.1, self._plan_loop)  # 10 Hz
        self.get_logger().info(
            f'SBLP Planner ready (beta={self.core.beta}, '
            f'polygon vertices={len(self.core.polygon)})')

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _goal_cb(self, msg: PoseStamped):
        tx = msg.pose.position.x
        ty = msg.pose.position.y
        tyaw = _yaw_from_quat(msg.pose.orientation)
        self._set_target(tx, ty, tyaw, 'external_override')
        self.get_logger().info(f'SBLP: external override waypoint ({tx:.2f}, {ty:.2f})')

    def _odom_cb(self, msg: Odometry):
        self._update_pose(msg)

    def _fallback_odom_cb(self, msg: Odometry):
        if not self.odom_received:
            self._update_pose(msg)

    def _update_pose(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = _yaw_from_quat(msg.pose.pose.orientation)
        self.odom_received = True

    def _micro_burst_cb(self, msg: String):
        try:
            applied = self.core.apply_micro_burst(msg.data)
            self.get_logger().warn(f'SBLP: elastic geo-fence micro-burst applied: {applied}')
        except Exception as e:  # noqa: BLE001 — log and continue on malformed input
            self.get_logger().error(f'SBLP: failed to parse micro-burst: {e}')

    # ── Waypoint handling ───────────────────────────────────────────────
    def _set_target(self, x: float, y: float, yaw: float, source: str):
        self.target = (x, y, yaw)
        self.waypoint_source = source
        wp = PoseStamped()
        wp.header.stamp = self.get_clock().now().to_msg()
        wp.header.frame_id = 'odom'
        wp.pose.position.x = x
        wp.pose.position.y = y
        wp.pose.orientation.z = math.sin(yaw * 0.5)
        wp.pose.orientation.w = math.cos(yaw * 0.5)
        self._waypoint_pub.publish(wp)

    def _new_levy_target(self):
        wp = self.core.generate_waypoint(self.current_x, self.current_y, self.current_yaw)
        self._set_target(wp.x, wp.y, wp.yaw, wp.source)
        return wp

    # ── Planning loop ─────────────────────────────────────────────────────
    def _plan_loop(self):
        if not self.odom_received:
            return

        if self.target is None:
            self._new_levy_target()

        if self.core.reached(self.current_x, self.current_y, self.target):
            self._new_levy_target()

        speed, angular = self.core.pure_pursuit(
            self.current_x, self.current_y, self.current_yaw, self.target)
        self._publish_cmd(speed, angular, 'levy_patrol')

    def _publish_cmd(self, linear: float, angular: float, scenario: str):
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self._cmd_pub.publish(cmd)
        self._scenario_pub.publish(String(data=scenario))
        status = {
            'scenario': scenario,
            'waypoint_source': self.waypoint_source,
            'levy_beta': self.core.beta,
            'target': [round(self.target[0], 2), round(self.target[1], 2)] if self.target else None,
            'odom_ok': self.odom_received,
        }
        self._status_pub.publish(String(data=json.dumps(status)))


def main(args=None):
    rclpy.init(args=args)
    node = SBLPPlanner()
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
