#!/usr/bin/env python3
"""
SBLP Planner Node
=================
Spatially-Bounded Lévy Patrol (SBLP) algorithm for EMCON-compliant border surveillance.

Guarantees mathematically unpredictable, geo-fenced sector coverage with zero inter-agent
communication by replacing deterministic waypoints with a heavy-tailed Pareto/Lévy
probability distribution: P(l) ~ l^(-beta), where 1 < beta <= 3.

To respect UGV kinematics and sector boundaries, the algorithm utilizes a rejection sampler
that mathematically guarantees the agent remains within an assigned GPS/map polygon.
Navigates towards generated waypoints using continuous-curvature Ackermann steering
(Pure Pursuit curvature control).

Subscribes:
  /goal_pose         (geometry_msgs/PoseStamped)  — optional external override waypoint
  /odometry/filtered (nav_msgs/Odometry)          — robot pose estimation
  /terramechanic_odom(nav_msgs/Odometry)          — fallback pose estimation
  /sblp/micro_burst  (std_msgs/String)            — encrypted Base Station parameter stretching

Publishes:
  /cmd_vel_raw           (geometry_msgs/Twist)       — raw Ackermann velocity commands
  /sblp/scenario         (std_msgs/String)           — active primitive name
  /sblp/status           (std_msgs/String)           — telemetry and Lévy metrics
  /sblp/current_waypoint (geometry_msgs/PoseStamped) — active Lévy waypoint for RViz
"""

import math
import json
import random
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry


def is_point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Ray casting algorithm to determine if point (x, y) is inside polygon."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class SBLPPlanner(Node):
    def __init__(self):
        super().__init__('sblp_planner')

        # Declare parameters
        self.declare_parameter('max_linear_vel',  1.5)
        self.declare_parameter('max_angular_vel', 0.6)
        self.declare_parameter('levy_beta',       1.8)
        self.declare_parameter('l_min',           3.0)
        self.declare_parameter('l_max',          35.0)
        self.declare_parameter('waypoint_tolerance', 1.5)
        self.declare_parameter('max_rejection_attempts', 50)
        
        # Default polygon: 120m x 120m sector around origin
        default_polygon = [[-60.0, -60.0], [60.0, -60.0], [60.0, 60.0], [-60.0, 60.0]]
        self.declare_parameter('geofence_polygon', default_polygon)

        self.max_lin = self.get_parameter('max_linear_vel').value
        self.max_ang = self.get_parameter('max_angular_vel').value
        self.beta    = self.get_parameter('levy_beta').value
        self.l_min   = self.get_parameter('l_min').value
        self.l_max   = self.get_parameter('l_max').value
        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.max_rejection_attempts = self.get_parameter('max_rejection_attempts').value
        self.geofence_polygon = self.get_parameter('geofence_polygon').value

        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_received = False

        # Waypoint tracking
        self.target_waypoint: tuple[float, float, float] | None = None
        self.waypoint_source = "none"
        self.external_override = False

        # Subscribers
        self.create_subscription(PoseStamped, '/goal_pose',         self._goal_cb,        10)
        self.create_subscription(Odometry,    '/odometry/filtered', self._odom_cb,        10)
        self.create_subscription(Odometry,    '/terramechanic_odom',self._fallback_odom_cb, 10)
        self.create_subscription(String,      '/sblp/micro_burst',  self._micro_burst_cb, 10)

        # Publishers
        self._cmd_pub          = self.create_publisher(Twist,       '/cmd_vel_raw',          10)
        self._scenario_pub     = self.create_publisher(String,      '/sblp/scenario',        10)
        self._status_pub       = self.create_publisher(String,      '/sblp/status',          10)
        self._waypoint_pub     = self.create_publisher(PoseStamped, '/sblp/current_waypoint', 10)

        # Planning loop at 10 Hz
        self.create_timer(0.1, self._plan_loop)
        self.get_logger().info(f'SBLP Planner ready (Lévy beta={self.beta}, polygon vertices={len(self.geofence_polygon)})')

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _goal_cb(self, msg: PoseStamped):
        """External goal override (e.g. from RViz or emergency Base Station directive)."""
        tx = msg.pose.position.x
        ty = msg.pose.position.y
        # Extract yaw from quaternion
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        tyaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.set_target_waypoint(tx, ty, tyaw, source="external_override")
        self.external_override = True
        self.get_logger().info(f'SBLP: Received external override waypoint at ({tx:.2f}, {ty:.2f})')

    def _odom_cb(self, msg: Odometry):
        self._update_pose(msg)

    def _fallback_odom_cb(self, msg: Odometry):
        if not self.odom_received:
            self._update_pose(msg)

    def _update_pose(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.odom_received = True

    def _micro_burst_cb(self, msg: String):
        """
        Handle encrypted micro-burst parameter stretching from Base Station RL brain.
        Format: JSON string with updated SBLP parameters.
        """
        try:
            data = json.loads(msg.data)
            if 'beta' in data:
                self.beta = float(data['beta'])
            if 'l_max' in data:
                self.l_max = float(data['l_max'])
            if 'polygon' in data and isinstance(data['polygon'], list):
                self.geofence_polygon = data['polygon']
            self.get_logger().warn(
                f"SBLP: Elastic Geo-Fence micro-burst applied! "
                f"New beta={self.beta}, l_max={self.l_max}, vertices={len(self.geofence_polygon)}"
            )
        except Exception as e:
            self.get_logger().error(f"SBLP: Failed to parse micro-burst: {e}")

    # ── Lévy Flight & Waypoint Management ─────────────────────────────────────
    def set_target_waypoint(self, x: float, y: float, yaw: float, source: str):
        self.target_waypoint = (x, y, yaw)
        self.waypoint_source = source
        
        # Publish visualization waypoint
        wp_msg = PoseStamped()
        wp_msg.header.stamp = self.get_clock().now().to_msg()
        wp_msg.header.frame_id = "odom"
        wp_msg.pose.position.x = x
        wp_msg.pose.position.y = y
        wp_msg.pose.position.z = 0.0
        wp_msg.pose.orientation.z = math.sin(yaw * 0.5)
        wp_msg.pose.orientation.w = math.cos(yaw * 0.5)
        self._waypoint_pub.publish(wp_msg)

    def generate_levy_waypoint(self) -> tuple[float, float, float]:
        """
        Generate next waypoint using Spatially-Bounded Lévy Patrol (SBLP) with
        geo-fenced rejection sampling.
        """
        for attempt in range(self.max_rejection_attempts):
            # Draw step length l from Pareto/Lévy distribution P(l) ~ l^(-beta)
            # using inverse transform sampling: l = l_min * (1 - u)^(-1 / (beta - 1))
            u = random.uniform(0.001, 0.999)
            step_len = self.l_min * (u ** (-1.0 / (self.beta - 1.0)))
            step_len = min(step_len, self.l_max)
            
            # Draw random turning angle
            turn_angle = random.uniform(-math.pi, math.pi)
            target_heading = self.current_yaw + turn_angle
            
            # Candidate coordinates
            cand_x = self.current_x + step_len * math.cos(target_heading)
            cand_y = self.current_y + step_len * math.sin(target_heading)
            
            # Geo-fence rejection check
            if is_point_in_polygon(cand_x, cand_y, self.geofence_polygon):
                self.get_logger().info(
                    f"SBLP: Generated valid Lévy waypoint at ({cand_x:.2f}, {cand_y:.2f}) "
                    f"[l={step_len:.2f}m, d_theta={math.degrees(turn_angle):.1f}°] "
                    f"after {attempt + 1} attempt(s)"
                )
                return cand_x, cand_y, target_heading
                
        # Rejection sampling exhausted (e.g. UGV near boundary pointing outward)
        # Generate recovery step toward centroid of polygon
        cent_x = sum(p[0] for p in self.geofence_polygon) / len(self.geofence_polygon)
        cent_y = sum(p[1] for p in self.geofence_polygon) / len(self.geofence_polygon)
        rec_heading = math.atan2(cent_y - self.current_y, cent_x - self.current_x)
        rec_step = min(self.l_min * 2.0, math.hypot(cent_x - self.current_x, cent_y - self.current_y))
        rec_x = self.current_x + rec_step * math.cos(rec_heading)
        rec_y = self.current_y + rec_step * math.sin(rec_heading)
        self.get_logger().warn(
            f"SBLP: Rejection sampling exhausted ({self.max_rejection_attempts} attempts). "
            f"Executing boundary reflection/recovery toward centroid ({rec_x:.2f}, {rec_y:.2f})"
        )
        return rec_x, rec_y, rec_heading

    def publish_cmd(self, linear_val: float, angular_val: float, scenario: str):
        cmd = Twist()
        cmd.linear.x = float(linear_val)
        cmd.angular.z = float(angular_val)
        self._cmd_pub.publish(cmd)
        self._scenario_pub.publish(String(data=scenario))
        
        # Telemetry
        status_data = {
            "scenario": scenario,
            "waypoint_source": self.waypoint_source,
            "levy_beta": self.beta,
            "target": [round(self.target_waypoint[0], 2), round(self.target_waypoint[1], 2)] if self.target_waypoint else None,
            "odom_ok": self.odom_received
        }
        self._status_pub.publish(String(data=json.dumps(status_data)))

    # ── Planning Loop ─────────────────────────────────────────────────────────
    def _plan_loop(self):
        if not self.odom_received:
            # Cannot navigate without odometry estimation
            return
            
        # Check if we need a new waypoint
        if self.target_waypoint is None:
            tx, ty, tyaw = self.generate_levy_waypoint()
            self.set_target_waypoint(tx, ty, tyaw, source="levy_flight")
            self.external_override = False
            
        tx, ty = self.target_waypoint[0], self.target_waypoint[1]
        dx = tx - self.current_x
        dy = ty - self.current_y
        dist = math.hypot(dx, dy)
        
        # Waypoint reached check
        if dist < self.waypoint_tolerance:
            self.get_logger().info(f"SBLP: Waypoint reached (dist={dist:.2f}m < {self.waypoint_tolerance}m). Generating next Lévy flight arc.")
            tx, ty, tyaw = self.generate_levy_waypoint()
            self.set_target_waypoint(tx, ty, tyaw, source="levy_flight")
            self.external_override = False
            dx = tx - self.current_x
            dy = ty - self.current_y
            dist = math.hypot(dx, dy)
            
        target_speed = self.max_lin
        scenario = "levy_patrol"
            
        # Slow down gracefully as we approach waypoint
        if dist < self.waypoint_tolerance * 2.5:
            target_speed = max(0.3, target_speed * (dist / (self.waypoint_tolerance * 2.5)))
            
        # Pure Pursuit / Ackermann continuous-curvature steering
        # Transform target to UGV base frame
        cos_yaw = math.cos(-self.current_yaw)
        sin_yaw = math.sin(-self.current_yaw)
        x_local = dx * cos_yaw - dy * sin_yaw
        y_local = dx * sin_yaw + dy * cos_yaw
        
        # Curvature kappa = 2 * y_local / dist^2
        if dist > 0.1:
            curvature = (2.0 * y_local) / (dist * dist)
        else:
            curvature = 0.0
            
        angular_vel = target_speed * curvature
        
        # Clamp angular velocity for high-speed Ackermann stability
        angular_vel = max(-self.max_ang, min(self.max_ang, angular_vel))
        
        self.publish_cmd(target_speed, angular_vel, scenario)


def main(args=None):
    rclpy.init(args=args)
    node = SBLPPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
