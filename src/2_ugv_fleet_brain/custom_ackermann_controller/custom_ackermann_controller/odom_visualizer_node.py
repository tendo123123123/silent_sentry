#!/usr/bin/env python3
"""
Odometry Comparison Visualizer — v3 (TF-aware, map-frame)
============================================================
Monitors the FULL localization stack by reading the map→base_footprint
TF chain rather than /odometry/filtered directly.

Previous version bug: it read /odometry/filtered which is the odom-frame
dead-reckoning pose. The TRN correction is expressed as the map→odom TF
and was completely ignored, making the "EKF Fused" line appear to drift
even when TRN was working correctly.

This version computes:
  - Localized pose = map → base_footprint  (TRN + dead-reckoning combined)
  - Raw odom pose  = /terramechanic_odom    (wheel odom only)
  - Ground truth   = Gazebo model pose
  All tracks aligned to start at (0,0) relative to their initial pose.

Layout (2×2):
  [0,0]  XY Trajectory  — GT (green), Localized (blue), Raw Odom (red dashed)
         + heading arrows (quiver) for current GT and Localized positions
  [0,1]  Position Error — Localized vs GT (blue), Raw Odom vs GT (red dashed)
  [1,0]  Heading Error  — Localized vs GT heading
    [1,1]  TRN Diagnostics — MAD likelihood, correction magnitude, drift %

Subscriptions:
  - /odometry/filtered          (nav_msgs/Odometry)      - odom-frame dead-reckoning
  - /ground_truth/pose          (geometry_msgs/Pose)     - Gazebo ground truth
  - /terramechanic_odom         (nav_msgs/Odometry)      - Raw wheel odometry
    - /trn/match_quality          (std_msgs/Float64)       - MCL MAD likelihood
  - /trn/search_radius          (std_msgs/Float64)       - search radius
  - /trn/correction             (geometry_msgs/Vector3)  - TRN correction vector

TF lookups (live):
  map → base_footprint   (full localized pose = TRN + dead-reckoning)
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math
from collections import deque

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose, Vector3
from tf_transformations import euler_from_quaternion

import tf2_ros

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class OdomVisualizerNode(Node):
    def __init__(self):
        super().__init__('odom_visualizer_node')

        # Parameters
        self.declare_parameter('max_history', 5000)
        self.declare_parameter('update_rate_hz', 2.0)
        self.declare_parameter('ground_truth_topic', '/ground_truth/pose')
        self.declare_parameter('model_name', 'alpha')
        self.declare_parameter('save_csv_on_exit', True)
        self.declare_parameter('csv_path', '/tmp/odom_comparison.csv')

        self.max_history = self.get_parameter('max_history').get_parameter_value().integer_value
        self.update_rate = self.get_parameter('update_rate_hz').get_parameter_value().double_value
        self.gt_topic    = self.get_parameter('ground_truth_topic').get_parameter_value().string_value
        self.model_name  = self.get_parameter('model_name').get_parameter_value().string_value
        self.save_csv    = self.get_parameter('save_csv_on_exit').get_parameter_value().bool_value
        self.csv_path    = self.get_parameter('csv_path').get_parameter_value().string_value

        # TF buffer for map→base_footprint lookup
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        def _dq():
            return deque(maxlen=self.max_history)

        # Ground truth (relative to initial)
        self.gt_x, self.gt_y, self.gt_yaw, self.gt_time = _dq(), _dq(), _dq(), _dq()

        # Localized: map→base_footprint (relative to initial TF reading)
        self.loc_x, self.loc_y, self.loc_yaw, self.loc_time = _dq(), _dq(), _dq(), _dq()

        # Raw wheel odom (relative to initial)
        self.raw_x, self.raw_y, self.raw_yaw, self.raw_time = _dq(), _dq(), _dq(), _dq()

        # Error history (synced with error_time)
        self.loc_pos_err  = _dq()
        self.raw_pos_err  = _dq()
        self.heading_err  = _dq()
        self.drift_hist   = _dq()
        self.error_time   = _dq()

        # TRN diagnostics
        self.trn_quality  = _dq()
        self.trn_corr_mag = _dq()
        self.trn_time     = _dq()
        self.last_trn_q   = 0.0
        self.last_trn_r   = 0.0
        self.last_trn_c   = 0.0

        # Cached odom-frame pose (used when map→base_footprint TF not yet available)
        self.odom_x = 0.0;  self.odom_y = 0.0;  self.odom_yaw = 0.0

        # GT alignment state
        self.gt_initial_x = self.gt_initial_y = self.gt_initial_yaw = None
        self.total_dist_gt = 0.0
        self.gt_received   = False

        # Localized alignment (captured on first successful TF lookup)
        self.loc_initial_x = self.loc_initial_y = self.loc_initial_yaw = None

        # Raw odom alignment
        self.raw_initial_x = self.raw_initial_y = self.raw_initial_yaw = None

        self.start_time = self.get_clock().now()

        # Publishers
        self.pos_err_pub  = self.create_publisher(Float64, '/odom_viz/position_error', 10)
        self.head_err_pub = self.create_publisher(Float64, '/odom_viz/heading_error',  10)
        self.drift_pub    = self.create_publisher(Float64, '/odom_viz/drift_percent',  10)
        self.ate_pub      = self.create_publisher(Float64, '/odom_viz/ate',            10)

        # Subscribers
        self.create_subscription(Odometry, '/odometry/filtered',  self._odom_cb,    10)
        self.create_subscription(Odometry, '/terramechanic_odom', self._raw_cb,     10)
        self.create_subscription(Pose,     self.gt_topic,         self._gt_cb,      10)
        self.create_subscription(Float64,  '/trn/match_quality',  self._trn_q_cb,   10)
        self.create_subscription(Float64,  '/trn/search_radius',  self._trn_r_cb,   10)
        self.create_subscription(Vector3,  '/trn/correction',     self._trn_cor_cb, 10)

        self.create_timer(0.1, self._compute_errors)

        self._setup_plot()

        self.get_logger().info(
            f'Odom Visualizer v3 (TF-aware, map frame) — '
            f'GT: {self.gt_topic} | refresh={self.update_rate}Hz')

    def _t(self):
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    # =========================================================================
    # Callbacks
    # =========================================================================
    def _odom_cb(self, msg: Odometry):
        """Cache odom-frame pose for TF fallback composition."""
        self.odom_x  = msg.pose.pose.position.x
        self.odom_y  = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw    = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.odom_yaw = yaw

    def _raw_cb(self, msg: Odometry):
        """Raw terramechanic wheel odometry — aligned to start at (0,0)."""
        t  = self._t()
        q  = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        rx = msg.pose.pose.position.x
        ry = msg.pose.pose.position.y

        if self.raw_initial_x is None:
            self.raw_initial_x   = rx
            self.raw_initial_y   = ry
            self.raw_initial_yaw = yaw

        dx   = rx - self.raw_initial_x
        dy   = ry - self.raw_initial_y
        ci   = math.cos(-self.raw_initial_yaw)
        si   = math.sin(-self.raw_initial_yaw)
        self.raw_x.append(dx * ci - dy * si)
        self.raw_y.append(dx * si + dy * ci)
        self.raw_yaw.append(_wrap(yaw - self.raw_initial_yaw))
        self.raw_time.append(t)

    def _gt_cb(self, msg: Pose):
        raw_x   = msg.position.x
        raw_y   = msg.position.y
        q       = msg.orientation
        _, _, raw_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if self.gt_initial_x is None:
            self.gt_initial_x   = raw_x
            self.gt_initial_y   = raw_y
            self.gt_initial_yaw = raw_yaw
            self.get_logger().info(
                f'GT initial pose captured: ({raw_x:.2f}, {raw_y:.2f}, '
                f'yaw={math.degrees(raw_yaw):.1f}°)')

        dx   = raw_x - self.gt_initial_x
        dy   = raw_y - self.gt_initial_y
        ci   = math.cos(-self.gt_initial_yaw)
        si   = math.sin(-self.gt_initial_yaw)
        ax   = dx * ci - dy * si
        ay   = dx * si + dy * ci
        ayaw = _wrap(raw_yaw - self.gt_initial_yaw)

        step = math.hypot(ax - (self.gt_x[-1] if self.gt_x else 0.0),
                          ay - (self.gt_y[-1] if self.gt_y else 0.0))
        if step > 0.01:
            self.total_dist_gt += step

        self.gt_x.append(ax);  self.gt_y.append(ay)
        self.gt_yaw.append(ayaw);  self.gt_time.append(self._t())
        self.gt_received = True

    def _trn_q_cb(self, msg: Float64):
        self.last_trn_q = msg.data
        self.trn_quality.append(msg.data)
        self.trn_time.append(self._t())

    def _trn_r_cb(self, msg: Float64):
        self.last_trn_r = msg.data

    def _trn_cor_cb(self, msg: Vector3):
        self.last_trn_c = math.hypot(msg.x, msg.y)
        self.trn_corr_mag.append(self.last_trn_c)

    # =========================================================================
    # Error Computation — uses map→base_footprint TF for localized pose
    # =========================================================================
    def _compute_errors(self):
        if not self.gt_received:
            return

        t = self._t()

        # --- Look up full localized pose via TF ---
        loc_x = loc_y = loc_yaw = None
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02))
            loc_x   = tf.transform.translation.x
            loc_y   = tf.transform.translation.y
            q       = tf.transform.rotation
            _, _, loc_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        except Exception:
            try:
                tf_mo = self.tf_buffer.lookup_transform(
                    'map', 'odom',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.02))
                mo_x   = tf_mo.transform.translation.x
                mo_y   = tf_mo.transform.translation.y
                q_mo   = tf_mo.transform.rotation
                _, _, mo_yaw = euler_from_quaternion(
                    [q_mo.x, q_mo.y, q_mo.z, q_mo.w])
                c, s   = math.cos(mo_yaw), math.sin(mo_yaw)
                loc_x  = mo_x + c * self.odom_x - s * self.odom_y
                loc_y  = mo_y + s * self.odom_x + c * self.odom_y
                loc_yaw = _wrap(mo_yaw + self.odom_yaw)
            except Exception:
                return

        if loc_x is None:
            return

        # Align localized to start at (0,0)
        if self.loc_initial_x is None:
            self.loc_initial_x   = loc_x
            self.loc_initial_y   = loc_y
            self.loc_initial_yaw = loc_yaw

        dx   = loc_x - self.loc_initial_x
        dy   = loc_y - self.loc_initial_y
        ci   = math.cos(-self.loc_initial_yaw)
        si   = math.sin(-self.loc_initial_yaw)
        alx  = dx * ci - dy * si
        aly  = dx * si + dy * ci
        alyaw = _wrap(loc_yaw - self.loc_initial_yaw)

        self.loc_x.append(alx);  self.loc_y.append(aly)
        self.loc_yaw.append(alyaw);  self.loc_time.append(t)

        if not self.gt_x:
            return
        gt_x   = self.gt_x[-1];  gt_y   = self.gt_y[-1]
        gt_yaw = self.gt_yaw[-1]

        pos_err   = math.hypot(alx - gt_x, aly - gt_y)
        head_err  = math.degrees(_wrap(alyaw - gt_yaw))
        drift_pct = (pos_err / self.total_dist_gt * 100.0
                     if self.total_dist_gt > 0.5 else 0.0)

        self.loc_pos_err.append(pos_err)
        self.heading_err.append(head_err)
        self.drift_hist.append(drift_pct)
        self.error_time.append(t)

        if self.raw_x:
            self.raw_pos_err.append(
                math.hypot(self.raw_x[-1] - gt_x, self.raw_y[-1] - gt_y))

        self._pub(self.pos_err_pub,  pos_err)
        self._pub(self.head_err_pub, head_err)
        self._pub(self.drift_pub,    drift_pct)
        if len(self.loc_pos_err) >= 10:
            self._pub(self.ate_pub,
                      float(np.mean(list(self.loc_pos_err)[-100:])))

    def _pub(self, pub, value):
        msg = Float64()
        msg.data = float(value)
        pub.publish(msg)

    # =========================================================================
    # Plot Setup
    # =========================================================================
    def _setup_plot(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(
            'Silent Sentry — TRN Localization Monitor (map frame)',
            fontsize=13)

        # ---- [0,0] XY Trajectory with heading arrows ----
        ax = self.ax_traj = self.axes[0, 0]
        ax.set_title('XY Trajectory (map frame, relative to start)')
        ax.set_xlabel('X (m)');  ax.set_ylabel('Y (m)')
        ax.set_aspect('equal');  ax.grid(True, alpha=0.3)
        self.line_gt,  = ax.plot([], [], 'g-',  lw=2,   label='Ground Truth')
        self.line_loc, = ax.plot([], [], 'b-',  lw=1.5, label='Localized (map→base)')
        self.line_raw, = ax.plot([], [], 'r--', lw=0.8, alpha=0.5,
                                 label='Raw Odom')
        self.qv_gt  = ax.quiver([], [], [], [], color='darkgreen',
                                scale=25, width=0.006, headwidth=4)
        self.qv_loc = ax.quiver([], [], [], [], color='royalblue',
                                scale=25, width=0.006, headwidth=4)
        ax.legend(loc='upper left', fontsize=8)
        self.stats_text = ax.text(
            0.02, 0.97, '', transform=ax.transAxes,
            fontsize=7.5, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='lightyellow', alpha=0.85))

        # ---- [0,1] Position Error ----
        ax = self.ax_pos = self.axes[0, 1]
        ax.set_title('Position Error — map frame (ATE)')
        ax.set_xlabel('Time (s)');  ax.set_ylabel('Error (m)')
        ax.grid(True, alpha=0.3)
        self.line_loc_err, = ax.plot([], [], 'b-',  lw=1.2,
                                     label='Localized (map→base)')
        self.line_raw_err, = ax.plot([], [], 'r--', lw=0.8, alpha=0.6,
                                     label='Raw Odom')
        ax.legend(loc='upper left', fontsize=8)

        # ---- [1,0] Heading Error ----
        ax = self.ax_head = self.axes[1, 0]
        ax.set_title('Heading Error (deg)')
        ax.set_xlabel('Time (s)');  ax.set_ylabel('Error (°)')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        self.line_head_err, = ax.plot([], [], 'm-', lw=1.2)

        # ---- [1,1] TRN Diagnostics ----
        ax = self.ax_trn = self.axes[1, 1]
        ax.set_title('TRN Diagnostics')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        self.line_trn_q,    = ax.plot([], [], 'b-',  lw=1.2, label='MAD likelihood')
        self.ax_trn_r = ax.twinx()
        self.ax_trn_r.set_ylabel('Correction |Δ| (m)', color='orange')
        self.line_trn_cor,  = self.ax_trn_r.plot([], [], color='orange',
                                                  lw=0.9, alpha=0.8,
                                                  label='|correction|')
        self.ax_drift_r = ax
        self.line_drift, = ax.plot([], [], 'c--', lw=0.8, alpha=0.6,
                                   label='Drift %/100')
        ax.legend(loc='upper left', fontsize=7)

        plt.tight_layout()

    # =========================================================================
    # Plot Update
    # =========================================================================
    def update_plot(self):
        # -- Trajectory --
        if len(self.gt_x) > 1:
            self.line_gt.set_data(list(self.gt_x),  list(self.gt_y))
        if len(self.loc_x) > 1:
            self.line_loc.set_data(list(self.loc_x), list(self.loc_y))
        if len(self.raw_x) > 1:
            self.line_raw.set_data(list(self.raw_x), list(self.raw_y))

        # Heading arrows at current positions
        if self.gt_x and self.gt_yaw:
            gx, gy, gyaw = self.gt_x[-1],  self.gt_y[-1],  self.gt_yaw[-1]
            self.qv_gt.set_offsets([[gx, gy]])
            self.qv_gt.set_UVC([math.cos(gyaw)], [math.sin(gyaw)])
        if self.loc_x and self.loc_yaw:
            lx, ly, lyaw = self.loc_x[-1], self.loc_y[-1], self.loc_yaw[-1]
            self.qv_loc.set_offsets([[lx, ly]])
            self.qv_loc.set_UVC([math.cos(lyaw)], [math.sin(lyaw)])

        # Stats text
        if self.gt_received and self.loc_pos_err:
            ate = float(np.mean(list(self.loc_pos_err)[-100:]))
            pe  = self.loc_pos_err[-1]
            he  = self.heading_err[-1] if self.heading_err else 0.0
            dr  = self.drift_hist[-1]  if self.drift_hist  else 0.0
            dist = self.total_dist_gt
            self.stats_text.set_text(
                f'ATE={ate:.2f}m  Err={pe:.2f}m\n'
                f'Δyaw={he:+.1f}°  Drift={dr:.1f}%\n'
                f'Dist={dist:.1f}m  Q={self.last_trn_q:.2f}  '
                f'R={self.last_trn_r:.1f}m  |Δ|={self.last_trn_c:.2f}m')

        # Auto-scale trajectory
        all_x = list(self.gt_x) + list(self.loc_x) + list(self.raw_x)
        all_y = list(self.gt_y) + list(self.loc_y) + list(self.raw_y)
        if len(all_x) > 2:
            mg = 3.0
            self.ax_traj.set_xlim(min(all_x) - mg, max(all_x) + mg)
            self.ax_traj.set_ylim(min(all_y) - mg, max(all_y) + mg)

        # -- Position Error --
        if len(self.error_time) > 1:
            et = list(self.error_time)
            self.line_loc_err.set_data(et, list(self.loc_pos_err))
            if len(self.raw_pos_err) == len(et):
                self.line_raw_err.set_data(et, list(self.raw_pos_err))
            self.ax_pos.set_xlim(0, max(et[-1], 1))
            mx = max(list(self.loc_pos_err)[-300:]) if self.loc_pos_err else 1.0
            self.ax_pos.set_ylim(0, max(mx * 1.2, 0.2))

        # -- Heading Error --
        if len(self.error_time) > 1 and self.heading_err:
            et = list(self.error_time)
            self.line_head_err.set_data(et[-len(self.heading_err):],
                                        list(self.heading_err))
            self.ax_head.set_xlim(0, max(et[-1], 1))
            rh = list(self.heading_err)[-300:]
            self.ax_head.set_ylim(min(rh) - 5, max(rh) + 5)

        # -- TRN Diagnostics --
        if self.trn_time:
            tt = list(self.trn_time)
            self.line_trn_q.set_data(tt[-len(self.trn_quality):],
                                     list(self.trn_quality))
            if self.trn_corr_mag:
                self.line_trn_cor.set_data(
                    tt[-len(self.trn_corr_mag):], list(self.trn_corr_mag))
                mx_c = max(list(self.trn_corr_mag)[-200:]) if self.trn_corr_mag else 1.0
                self.ax_trn_r.set_ylim(0, max(mx_c * 1.3, 0.5))
            if self.drift_hist:
                et = list(self.error_time)
                dh = [d / 100.0 for d in self.drift_hist]
                self.line_drift.set_data(et[-len(dh):], dh)
            self.ax_trn.set_xlim(0, max(tt[-1], 1))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save_csv_data(self):
        if not self.save_csv:
            return
        try:
            import csv
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'time_s', 'gt_x', 'gt_y', 'gt_yaw',
                    'loc_x', 'loc_y', 'loc_yaw',
                    'raw_x', 'raw_y', 'raw_yaw',
                    'loc_pos_err_m', 'heading_err_deg', 'drift_pct'
                ])
                n = min(len(self.error_time),
                        len(self.gt_x), len(self.loc_x),
                        len(self.loc_pos_err))
                for i in range(n):
                    writer.writerow([
                        f'{self.error_time[i]:.3f}',
                        f'{self.gt_x[i]:.4f}', f'{self.gt_y[i]:.4f}',
                        f'{self.gt_yaw[i]:.4f}',
                        f'{self.loc_x[i]:.4f}', f'{self.loc_y[i]:.4f}',
                        f'{self.loc_yaw[i]:.4f}',
                        f'{list(self.raw_x)[i] if i < len(self.raw_x) else 0.0:.4f}',
                        f'{list(self.raw_y)[i] if i < len(self.raw_y) else 0.0:.4f}',
                        f'{list(self.raw_yaw)[i] if i < len(self.raw_yaw) else 0.0:.4f}',
                        f'{self.loc_pos_err[i]:.4f}',
                        f'{list(self.heading_err)[i] if i < len(self.heading_err) else 0.0:.2f}',
                        f'{list(self.drift_hist)[i] if i < len(self.drift_hist) else 0.0:.4f}'
                    ])

            self.get_logger().info(f'Saved comparison data to {self.csv_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save CSV: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = OdomVisualizerNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1.0 / node.update_rate)
            node.update_plot()
    except KeyboardInterrupt:
        pass
    finally:
        node.save_csv_data()
        plt.close('all')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

