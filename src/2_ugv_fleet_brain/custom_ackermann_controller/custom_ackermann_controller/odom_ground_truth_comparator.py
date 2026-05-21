#!/usr/bin/env python3
"""
Odometry Ground Truth Comparator
=================================
Benchmarking node for the terramechanics localization stack.

Subscribes to Gazebo's true model pose (via PosePublisher + ros_gz_bridge)
and compares it against the estimated odometry from different sources:

  1. /odometry/filtered   — EKF fused output (primary)
  2. /terramechanic_odom   — Raw Bekker-Wong wheel odometry

Publishes real-time error metrics for monitoring and tuning:
  - Absolute position error (m)
  - Heading error (deg)
  - Drift percentage (position error / distance traveled)
  - Rolling ATE (Absolute Trajectory Error) over a window

Also logs time-stamped CSV for offline analysis with external tools
(e.g. evo_ape, evo_rpe).

Subscriptions:
  - /ground_truth/pose     (geometry_msgs/Pose)     - Gazebo ground truth
  - /odometry/filtered     (nav_msgs/Odometry)       - EKF output
  - /terramechanic_odom    (nav_msgs/Odometry)       - Raw wheel odometry

Publications:
  - /odom_error/ekf/position_error   (Float64)  - Position error [m]
  - /odom_error/ekf/heading_error    (Float64)  - Heading error [deg]
  - /odom_error/ekf/drift_percent    (Float64)  - Drift as % of distance
  - /odom_error/ekf/ate              (Float64)  - Rolling ATE [m]
  - /odom_error/raw/position_error   (Float64)  - Raw odom position error [m]
  - /odom_error/raw/heading_error    (Float64)  - Raw odom heading error [deg]
  - /odom_error/summary              (String)   - Human-readable summary
"""

import rclpy
from rclpy.node import Node
import math
import numpy as np
import os
import time
from collections import deque

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
from geometry_msgs.msg import Pose
from tf_transformations import euler_from_quaternion


class OdomGroundTruthComparator(Node):
    def __init__(self):
        super().__init__('odom_ground_truth_comparator')

        # ---- Parameters ----
        self.declare_parameter('publish_rate', 2.0)           # Hz for summary
        self.declare_parameter('ate_window_size', 100)        # samples for rolling ATE
        self.declare_parameter('csv_log_enabled', True)
        self.declare_parameter('csv_log_path', '/tmp/odom_ground_truth_log.csv')
        self.declare_parameter('model_name', 'alpha')         # Gazebo model name
        self.declare_parameter('ground_truth_topic', '/ground_truth/pose')
        self.declare_parameter('ekf_odom_topic', '/odometry/filtered')
        self.declare_parameter('raw_odom_topic', '/terramechanic_odom')

        self.publish_rate = self._p('publish_rate')
        self.ate_window = self.get_parameter('ate_window_size').get_parameter_value().integer_value
        self.csv_enabled = self.get_parameter('csv_log_enabled').get_parameter_value().bool_value
        self.csv_path = self.get_parameter('csv_log_path').get_parameter_value().string_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        gt_topic = self.get_parameter('ground_truth_topic').get_parameter_value().string_value
        ekf_topic = self.get_parameter('ekf_odom_topic').get_parameter_value().string_value
        raw_topic = self.get_parameter('raw_odom_topic').get_parameter_value().string_value

        # ---- State ----
        # Ground truth
        self.gt_x = 0.0
        self.gt_y = 0.0
        self.gt_z = 0.0
        self.gt_yaw = 0.0
        self.gt_received = False
        self.gt_initial_x = None
        self.gt_initial_y = None
        self.gt_initial_yaw = None

        # EKF estimated
        self.ekf_x = 0.0
        self.ekf_y = 0.0
        self.ekf_yaw = 0.0
        self.ekf_received = False

        # Raw odom estimated
        self.raw_x = 0.0
        self.raw_y = 0.0
        self.raw_yaw = 0.0
        self.raw_received = False

        # Distance tracking (ground truth path length)
        self.gt_last_x = None
        self.gt_last_y = None
        self.gt_total_distance = 0.0

        # Rolling ATE buffers
        self.ekf_errors = deque(maxlen=self.ate_window)
        self.raw_errors = deque(maxlen=self.ate_window)

        # Trajectory history for CSV logging
        self.trajectory_log = []
        self.log_counter = 0

        # ---- Publishers: EKF error metrics ----
        self.ekf_pos_err_pub = self.create_publisher(
            Float64, '/odom_error/ekf/position_error', 10)
        self.ekf_head_err_pub = self.create_publisher(
            Float64, '/odom_error/ekf/heading_error', 10)
        self.ekf_drift_pub = self.create_publisher(
            Float64, '/odom_error/ekf/drift_percent', 10)
        self.ekf_ate_pub = self.create_publisher(
            Float64, '/odom_error/ekf/ate', 10)

        # ---- Publishers: Raw odom error metrics ----
        self.raw_pos_err_pub = self.create_publisher(
            Float64, '/odom_error/raw/position_error', 10)
        self.raw_head_err_pub = self.create_publisher(
            Float64, '/odom_error/raw/heading_error', 10)

        # ---- Publisher: Human-readable summary ----
        self.summary_pub = self.create_publisher(
            String, '/odom_error/summary', 10)

        # ---- Subscribers ----
        self.gt_sub = self.create_subscription(
            Pose, gt_topic, self.gt_callback, 10)
        self.ekf_sub = self.create_subscription(
            Odometry, ekf_topic, self.ekf_callback, 10)
        self.raw_sub = self.create_subscription(
            Odometry, raw_topic, self.raw_callback, 10)

        # ---- Timer for summary publishing ----
        self.summary_timer = self.create_timer(
            1.0 / self.publish_rate, self.publish_summary)

        # ---- CSV file setup ----
        if self.csv_enabled:
            self._init_csv()

        self.get_logger().info(
            f'Ground Truth Comparator initialized\n'
            f'  Model: {self.model_name}\n'
            f'  GT topic: {gt_topic}\n'
            f'  EKF topic: {ekf_topic}\n'
            f'  Raw topic: {raw_topic}\n'
            f'  CSV logging: {self.csv_path if self.csv_enabled else "disabled"}'
        )

    def _p(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    # =========================================================================
    # CSV Logging
    # =========================================================================
    def _init_csv(self):
        """Initialize CSV log file with header."""
        try:
            with open(self.csv_path, 'w') as f:
                f.write(
                    'timestamp,'
                    'gt_x,gt_y,gt_yaw,'
                    'ekf_x,ekf_y,ekf_yaw,'
                    'raw_x,raw_y,raw_yaw,'
                    'ekf_pos_err,ekf_head_err,'
                    'raw_pos_err,raw_head_err,'
                    'gt_distance,ekf_drift_pct\n'
                )
            self.get_logger().info(f'CSV log initialized: {self.csv_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to create CSV log: {e}')
            self.csv_enabled = False

    def _log_csv(self, ekf_pos_err, ekf_head_err, raw_pos_err, raw_head_err, drift_pct):
        """Append one row to the CSV log."""
        if not self.csv_enabled:
            return

        self.log_counter += 1
        # Log every 5th sample to avoid huge files (at 50Hz input → ~10Hz log)
        if self.log_counter % 5 != 0:
            return

        try:
            stamp = self.get_clock().now().seconds_nanoseconds()
            t = f'{stamp[0]}.{stamp[1]:09d}'

            with open(self.csv_path, 'a') as f:
                f.write(
                    f'{t},'
                    f'{self.gt_x:.4f},{self.gt_y:.4f},{self.gt_yaw:.4f},'
                    f'{self.ekf_x:.4f},{self.ekf_y:.4f},{self.ekf_yaw:.4f},'
                    f'{self.raw_x:.4f},{self.raw_y:.4f},{self.raw_yaw:.4f},'
                    f'{ekf_pos_err:.4f},{ekf_head_err:.4f},'
                    f'{raw_pos_err:.4f},{raw_head_err:.4f},'
                    f'{self.gt_total_distance:.4f},{drift_pct:.4f}\n'
                )
        except Exception:
            pass

    # =========================================================================
    # Callbacks
    # =========================================================================
    def gt_callback(self, msg: Pose):
        """
        Receive ground truth from /model/{name}/pose via ros_gz_bridge.
        gz.msgs.Pose → geometry_msgs/Pose: single-model topic, no name-matching needed.
        """
        self.gt_x = msg.position.x
        self.gt_y = msg.position.y
        self.gt_z = msg.position.z

        q = msg.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.gt_yaw = yaw

        # Set initial pose (for relative comparison with odom at (0,0))
        if self.gt_initial_x is None:
            self.gt_initial_x = self.gt_x
            self.gt_initial_y = self.gt_y
            self.gt_initial_yaw = self.gt_yaw
            self.gt_last_x = self.gt_x
            self.gt_last_y = self.gt_y
            self.get_logger().info(
                f'Ground truth initial pose captured: '
                f'({self.gt_x:.2f}, {self.gt_y:.2f}, yaw={math.degrees(self.gt_yaw):.1f}°)'
            )

        # Track total distance traveled
        dx = self.gt_x - self.gt_last_x
        dy = self.gt_y - self.gt_last_y
        step = math.sqrt(dx * dx + dy * dy)
        if step > 0.01:  # Ignore jitter below 1cm
            self.gt_total_distance += step
            self.gt_last_x = self.gt_x
            self.gt_last_y = self.gt_y

        if not self.gt_received:
            self.get_logger().info(
                f'GT data flowing: x={self.gt_x:.2f} y={self.gt_y:.2f} yaw={math.degrees(yaw):.1f}°'
            )
        self.gt_received = True

    def ekf_callback(self, msg: Odometry):
        """Receive EKF-filtered odometry."""
        self.ekf_x = msg.pose.pose.position.x
        self.ekf_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.ekf_yaw = yaw
        self.ekf_received = True

        # Compute and publish errors on every EKF update
        if self.gt_received:
            self._compute_and_publish_errors()

    def raw_callback(self, msg: Odometry):
        """Receive raw terramechanic odometry."""
        self.raw_x = msg.pose.pose.position.x
        self.raw_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.raw_yaw = yaw
        self.raw_received = True

    # =========================================================================
    # Error Computation
    # =========================================================================
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def _compute_and_publish_errors(self):
        """Compute all error metrics and publish."""
        if self.gt_initial_x is None:
            return

        # Ground truth pose relative to initial (to match odom starting at 0,0)
        gt_rel_x = self.gt_x - self.gt_initial_x
        gt_rel_y = self.gt_y - self.gt_initial_y
        gt_rel_yaw = self._normalize_angle(self.gt_yaw - self.gt_initial_yaw)

        # ---- EKF errors ----
        ekf_pos_err = math.sqrt(
            (self.ekf_x - gt_rel_x) ** 2 +
            (self.ekf_y - gt_rel_y) ** 2
        )
        ekf_head_err = abs(math.degrees(
            self._normalize_angle(self.ekf_yaw - gt_rel_yaw)
        ))

        # Drift percentage
        drift_pct = 0.0
        if self.gt_total_distance > 1.0:
            drift_pct = (ekf_pos_err / self.gt_total_distance) * 100.0

        # Rolling ATE (RMSE of position errors in window)
        self.ekf_errors.append(ekf_pos_err)
        ekf_ate = math.sqrt(
            sum(e ** 2 for e in self.ekf_errors) / len(self.ekf_errors)
        )

        # Publish EKF metrics
        self._pub_f64(self.ekf_pos_err_pub, ekf_pos_err)
        self._pub_f64(self.ekf_head_err_pub, ekf_head_err)
        self._pub_f64(self.ekf_drift_pub, drift_pct)
        self._pub_f64(self.ekf_ate_pub, ekf_ate)

        # ---- Raw odom errors ----
        raw_pos_err = 0.0
        raw_head_err = 0.0
        if self.raw_received:
            raw_pos_err = math.sqrt(
                (self.raw_x - gt_rel_x) ** 2 +
                (self.raw_y - gt_rel_y) ** 2
            )
            raw_head_err = abs(math.degrees(
                self._normalize_angle(self.raw_yaw - gt_rel_yaw)
            ))
            self.raw_errors.append(raw_pos_err)

            self._pub_f64(self.raw_pos_err_pub, raw_pos_err)
            self._pub_f64(self.raw_head_err_pub, raw_head_err)

        # ---- CSV logging ----
        self._log_csv(ekf_pos_err, ekf_head_err, raw_pos_err, raw_head_err, drift_pct)

    @staticmethod
    def _pub_f64(pub, value: float):
        msg = Float64()
        msg.data = value
        pub.publish(msg)

    # =========================================================================
    # Summary Publishing
    # =========================================================================
    def publish_summary(self):
        """Periodic human-readable summary for console monitoring."""
        if not self.gt_received:
            return

        if self.gt_initial_x is None:
            return

        gt_rel_x = self.gt_x - self.gt_initial_x
        gt_rel_y = self.gt_y - self.gt_initial_y
        gt_rel_yaw = self._normalize_angle(self.gt_yaw - self.gt_initial_yaw)

        ekf_pos_err = math.sqrt(
            (self.ekf_x - gt_rel_x) ** 2 +
            (self.ekf_y - gt_rel_y) ** 2
        ) if self.ekf_received else float('nan')

        ekf_head_err = abs(math.degrees(
            self._normalize_angle(self.ekf_yaw - gt_rel_yaw)
        )) if self.ekf_received else float('nan')

        raw_pos_err = math.sqrt(
            (self.raw_x - gt_rel_x) ** 2 +
            (self.raw_y - gt_rel_y) ** 2
        ) if self.raw_received else float('nan')

        drift_pct = (ekf_pos_err / self.gt_total_distance * 100.0
                     if self.gt_total_distance > 1.0 else 0.0)

        ekf_ate = 0.0
        if len(self.ekf_errors) > 0:
            ekf_ate = math.sqrt(
                sum(e ** 2 for e in self.ekf_errors) / len(self.ekf_errors)
            )

        raw_ate = 0.0
        if len(self.raw_errors) > 0:
            raw_ate = math.sqrt(
                sum(e ** 2 for e in self.raw_errors) / len(self.raw_errors)
            )

        summary = (
            f'GT=({gt_rel_x:.2f},{gt_rel_y:.2f},{math.degrees(gt_rel_yaw):.1f}°) '
            f'dist={self.gt_total_distance:.1f}m | '
            f'EKF: err={ekf_pos_err:.2f}m θ={ekf_head_err:.1f}° '
            f'ATE={ekf_ate:.2f}m drift={drift_pct:.1f}% | '
            f'RAW: err={raw_pos_err:.2f}m ATE={raw_ate:.2f}m'
        )

        summary_msg = String()
        summary_msg.data = summary
        self.summary_pub.publish(summary_msg)

        self.get_logger().info(f'OdomBench: {summary}')


def main(args=None):
    rclpy.init(args=args)
    node = OdomGroundTruthComparator()
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
