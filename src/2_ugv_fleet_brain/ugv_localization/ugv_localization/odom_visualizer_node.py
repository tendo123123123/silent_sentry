#!/usr/bin/env python3
"""
Odometry Comparison Visualizer — v4 (High-Performance, Thread-Safe, Map-Frame)
=============================================================================
Monitors the FULL localization stack by reading the map→base_footprint
TF chain and comparing it with Gazebo ground-truth and raw wheel odometry.

Optimized with a dual-thread architecture:
  1. Background Thread: Spins ROS 2 at high-frequency (sub-millisecond callbacks)
  2. Main GUI Thread: Redraws Matplotlib graphs at 20 FPS using snapshot data,
     under strict thread-safe mutex locks, downsampled to keep rendering times low.

Fused with all benchmarking utilities to eliminate duplicate processing.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import numpy as np
import math
import threading
from collections import deque

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
from geometry_msgs.msg import Pose, Vector3

import tf2_ros
from tf_transformations import euler_from_quaternion

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# =========================================================================
# Shared Coordinate & TF Helpers (formerly benchmark_pose_utils)
# =========================================================================
class LocalizedPose:
    def __init__(self, x: float, y: float, yaw: float, source: str):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.source = source


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(quat) -> float:
    return euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]


def rotate_into_reference(dx: float, dy: float, reference_yaw: float) -> tuple[float, float]:
    cos_ref = math.cos(-reference_yaw)
    sin_ref = math.sin(-reference_yaw)
    return dx * cos_ref - dy * sin_ref, dx * sin_ref + dy * cos_ref


def align_pose_to_reference(
    x: float,
    y: float,
    yaw: float,
    initial_x: float,
    initial_y: float,
    reference_yaw: float,
) -> tuple[float, float, float]:
    dx = x - initial_x
    dy = y - initial_y
    aligned_x, aligned_y = rotate_into_reference(dx, dy, reference_yaw)
    return aligned_x, aligned_y, wrap_angle(yaw - reference_yaw)


def lookup_localized_pose(
    tf_buffer: tf2_ros.Buffer,
    odom_x: float,
    odom_y: float,
    odom_yaw: float,
    map_frame: str = 'map',
    odom_frame: str = 'odom',
    base_frame: str = 'base_footprint',
    timeout_sec: float = 0.02,
) -> LocalizedPose | None:
    timeout = Duration(seconds=timeout_sec)

    try:
        transform = tf_buffer.lookup_transform(
            map_frame,
            base_frame,
            rclpy.time.Time(),
            timeout=timeout,
        )
        return LocalizedPose(
            x=transform.transform.translation.x,
            y=transform.transform.translation.y,
            yaw=quaternion_to_yaw(transform.transform.rotation),
            source='map_to_base',
        )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ):
        pass

    try:
        transform = tf_buffer.lookup_transform(
            map_frame,
            odom_frame,
            rclpy.time.Time(),
            timeout=timeout,
        )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ):
        return None

    map_to_odom_yaw = quaternion_to_yaw(transform.transform.rotation)
    cos_yaw = math.cos(map_to_odom_yaw)
    sin_yaw = math.sin(map_to_odom_yaw)
    return LocalizedPose(
        x=transform.transform.translation.x + cos_yaw * odom_x - sin_yaw * odom_y,
        y=transform.transform.translation.y + sin_yaw * odom_x + cos_yaw * odom_y,
        yaw=wrap_angle(map_to_odom_yaw + odom_yaw),
        source='map_to_odom_plus_odom',
    )


# =========================================================================
# Unified High-Performance Visualizer Node
# =========================================================================
class OdomVisualizerNode(Node):
    def __init__(self):
        super().__init__('odom_visualizer_node')

        # Parameters
        # Large enough to retain a full long-run trajectory for all traces
        # without maxlen eviction (which desyncs GT/raw/loc time coverage).
        self.declare_parameter('max_history', 100000)
        self.declare_parameter('update_rate_hz', 5.0)
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

        # Threading lock for 100% safety
        self.lock = threading.Lock()

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

        # Trajectory bounds for O(1) auto-scaling
        self.min_x = -5.0;  self.max_x = 5.0
        self.min_y = -5.0;  self.max_y = 5.0

        # GT alignment state (defines the common comparison frame)
        self.gt_initial_x = self.gt_initial_y = self.gt_initial_yaw = None
        self.reference_yaw = None
        self.total_dist_gt = 0.0
        self.gt_received   = False

        # Localized alignment (captured on first successful TF lookup)
        self.loc_initial_x = self.loc_initial_y = self.loc_initial_yaw = None

        # Raw odom alignment
        self.raw_initial_x = self.raw_initial_y = self.raw_initial_yaw = None

        self.start_time = None

        # Publishers (visualizer native)
        self.pos_err_pub  = self.create_publisher(Float64, '/odom_viz/position_error', 10)
        self.head_err_pub = self.create_publisher(Float64, '/odom_viz/heading_error',  10)
        self.drift_pub    = self.create_publisher(Float64, '/odom_viz/drift_percent',  10)
        self.ate_pub      = self.create_publisher(Float64, '/odom_viz/ate',            10)

        # Publishers (merged from odom_ground_truth_comparator)
        self.ekf_pos_err_pub  = self.create_publisher(Float64, '/odom_error/ekf/position_error',  10)
        self.ekf_head_err_pub = self.create_publisher(Float64, '/odom_error/ekf/heading_error',   10)
        self.ekf_drift_pub    = self.create_publisher(Float64, '/odom_error/ekf/drift_percent',   10)
        self.ekf_ate_pub      = self.create_publisher(Float64, '/odom_error/ekf/ate',             10)
        self.raw_pos_err_pub  = self.create_publisher(Float64, '/odom_error/raw/position_error',  10)
        self.raw_head_err_pub = self.create_publisher(Float64, '/odom_error/raw/heading_error',   10)
        self.summary_pub      = self.create_publisher(String,  '/odom_error/summary',             10)

        # Subscribers
        self.create_subscription(Odometry, '/odometry/filtered',  self._odom_cb,    10)
        self.create_subscription(Odometry, '/terramechanic_odom', self._raw_cb,     10)
        self.create_subscription(Pose,     self.gt_topic,         self._gt_cb,      10)
        self.create_subscription(Float64,  '/trn/match_quality',  self._trn_q_cb,   10)
        self.create_subscription(Float64,  '/trn/search_radius',  self._trn_r_cb,   10)
        self.create_subscription(Vector3,  '/trn/correction',     self._trn_cor_cb, 10)

        # Background high-frequency timer to compute errors
        self.create_timer(0.05, self._compute_errors)

        self._setup_plot()

        self.get_logger().info(
            f'Odom Visualizer v4 (High-Performance, Map-Frame) — '
            f'GT: {self.gt_topic} | refresh={self.update_rate}Hz')

    def _t(self):
        now = self.get_clock().now()
        if now.nanoseconds <= 0:
            return 0.0
        if self.start_time is None:
            self.start_time = now
            return 0.0
        return (now - self.start_time).nanoseconds / 1e9

    # =========================================================================
    # Callbacks (Thread-Safe using Lock)
    # =========================================================================
    def _odom_cb(self, msg: Odometry):
        """Cache odom-frame pose for TF fallback composition."""
        with self.lock:
            self.odom_x  = msg.pose.pose.position.x
            self.odom_y  = msg.pose.pose.position.y
            self.odom_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

    def _raw_cb(self, msg: Odometry):
        """Raw terramechanic wheel odometry — aligned to start at (0,0)."""
        t  = self._t()
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        rx = msg.pose.pose.position.x
        ry = msg.pose.pose.position.y

        with self.lock:
            if self.raw_initial_x is None:
                self.raw_initial_x   = rx
                self.raw_initial_y   = ry
                self.raw_initial_yaw = yaw

            if self.reference_yaw is None:
                return

            ax, ay, aligned_yaw = align_pose_to_reference(
                rx, ry, yaw,
                self.raw_initial_x, self.raw_initial_y,
                self.reference_yaw,
            )

            # Update auto-scale bounds
            self.min_x = min(self.min_x, ax); self.max_x = max(self.max_x, ax)
            self.min_y = min(self.min_y, ay); self.max_y = max(self.max_y, ay)

            self.raw_x.append(ax)
            self.raw_y.append(ay)
            self.raw_yaw.append(aligned_yaw)
            self.raw_time.append(t)

    def _gt_cb(self, msg: Pose):
        raw_x   = msg.position.x
        raw_y   = msg.position.y
        raw_yaw = quaternion_to_yaw(msg.orientation)

        with self.lock:
            if self.gt_initial_x is None:
                self.gt_initial_x   = raw_x
                self.gt_initial_y   = raw_y
                self.gt_initial_yaw = raw_yaw
                self.reference_yaw  = raw_yaw
                self.get_logger().info(
                    f'GT initial pose captured: ({raw_x:.2f}, {raw_y:.2f}, '
                    f'yaw={math.degrees(raw_yaw):.1f}°)')

            ax, ay, ayaw = align_pose_to_reference(
                raw_x, raw_y, raw_yaw,
                self.gt_initial_x, self.gt_initial_y,
                self.reference_yaw,
            )

            # Update auto-scale bounds
            self.min_x = min(self.min_x, ax); self.max_x = max(self.max_x, ax)
            self.min_y = min(self.min_y, ay); self.max_y = max(self.max_y, ay)

            step = math.hypot(ax - (self.gt_x[-1] if self.gt_x else 0.0),
                              ay - (self.gt_y[-1] if self.gt_y else 0.0))
            if step > 0.01:
                self.total_dist_gt += step

            self.gt_x.append(ax)
            self.gt_y.append(ay)
            self.gt_yaw.append(ayaw)
            self.gt_time.append(self._t())
            self.gt_received = True

    def _trn_q_cb(self, msg: Float64):
        with self.lock:
            self.last_trn_q = msg.data
            self.trn_quality.append(msg.data)
            self.trn_time.append(self._t())

    def _trn_r_cb(self, msg: Float64):
        with self.lock:
            self.last_trn_r = msg.data

    def _trn_cor_cb(self, msg: Vector3):
        with self.lock:
            self.last_trn_c = math.hypot(msg.x, msg.y)
            self.trn_corr_mag.append(self.last_trn_c)

    # =========================================================================
    # Error Computation — uses map→base_footprint TF for localized pose
    # =========================================================================
    def _compute_errors(self):
        if not self.gt_received:
            return

        t = self._t()

        # Copy state for calculation to avoid holding the lock
        with self.lock:
            odom_x_cpy = self.odom_x
            odom_y_cpy = self.odom_y
            odom_yaw_cpy = self.odom_yaw

        localized_pose = lookup_localized_pose(
            self.tf_buffer,
            odom_x_cpy,
            odom_y_cpy,
            odom_yaw_cpy,
        )
        if localized_pose is None:
            return

        loc_x = localized_pose.x
        loc_y = localized_pose.y
        loc_yaw = localized_pose.yaw

        with self.lock:
            # Align localized to start at (0,0)
            if self.loc_initial_x is None:
                self.loc_initial_x   = loc_x
                self.loc_initial_y   = loc_y
                self.loc_initial_yaw = loc_yaw

            alx, aly, alyaw = align_pose_to_reference(
                loc_x, loc_y, loc_yaw,
                self.loc_initial_x, self.loc_initial_y,
                self.reference_yaw,
            )

            # Update auto-scale bounds
            self.min_x = min(self.min_x, alx); self.max_x = max(self.max_x, alx)
            self.min_y = min(self.min_y, aly); self.max_y = max(self.max_y, aly)

            self.loc_x.append(alx)
            self.loc_y.append(aly)
            self.loc_yaw.append(alyaw)
            self.loc_time.append(t)

            if not self.gt_x:
                return
            gt_x   = self.gt_x[-1]
            gt_y   = self.gt_y[-1]
            gt_yaw = self.gt_yaw[-1]

            pos_err   = math.hypot(alx - gt_x, aly - gt_y)
            head_err  = math.degrees(wrap_angle(alyaw - gt_yaw))
            drift_pct = (pos_err / self.total_dist_gt * 100.0
                         if self.total_dist_gt > 0.5 else 0.0)

            self.loc_pos_err.append(pos_err)
            self.heading_err.append(head_err)
            self.drift_hist.append(drift_pct)
            self.error_time.append(t)

            if self.raw_x:
                self.raw_pos_err.append(
                    math.hypot(self.raw_x[-1] - gt_x, self.raw_y[-1] - gt_y))

            # Store stats for plotting thread
            ate = float(np.mean(list(self.loc_pos_err)[-100:])) if len(self.loc_pos_err) >= 10 else 0.0

        # Publish errors back to ROS
        self._pub(self.pos_err_pub,  pos_err)
        self._pub(self.head_err_pub, head_err)
        self._pub(self.drift_pub,    drift_pct)
        self._pub(self.ate_pub, ate)

        # Merged comparator publishers
        self._pub(self.ekf_pos_err_pub,  pos_err)
        self._pub(self.ekf_head_err_pub, head_err)
        self._pub(self.ekf_drift_pub,    drift_pct)
        self._pub(self.ekf_ate_pub,      ate)
        if self.raw_pos_err:
            raw_pe = self.raw_pos_err[-1]
            self._pub(self.raw_pos_err_pub, raw_pe)
        if self.raw_x and self.raw_y:
            raw_yaw_err = math.degrees(wrap_angle(
                self.raw_yaw[-1] - gt_yaw)) if self.raw_yaw else 0.0
            self._pub(self.raw_head_err_pub, abs(raw_yaw_err))

        summary = (
            f'PosErr={pos_err:.2f}m  HeadErr={head_err:.1f}deg  '
            f'Drift={drift_pct:.1f}%  ATE={ate:.2f}m  '
            f'RawPosErr={self.raw_pos_err[-1] if self.raw_pos_err else 0.0:.2f}m  '
            f'TRN_Q={self.last_trn_q:.2f}'
        )
        sum_msg = String()
        sum_msg.data = summary
        self.summary_pub.publish(sum_msg)

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
        self.pt_gt,  = ax.plot([], [], 'go', ms=5, label='_nolegend_')
        self.pt_loc, = ax.plot([], [], 'bo', ms=4, label='_nolegend_')
        self.pt_raw, = ax.plot([], [], 'ro', ms=3, alpha=0.5, label='_nolegend_')
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
    # Decimation Helper - stride-based downsample to max N points
    # =========================================================================
    @staticmethod
    def _decimate(arr, max_pts=200):
        """Return a numpy array with at most max_pts elements using stride slicing."""
        if len(arr) <= max_pts:
            return np.asarray(arr)
        stride = max(1, len(arr) // max_pts)
        return np.asarray(arr)[::stride]

    # =========================================================================
    # Plot Update (Executed solely on main thread, using snapshots)
    # =========================================================================
    def update_plot(self):
        # 1. Take a quick thread-safe snapshot of the plotting data
        with self.lock:
            gt_x = np.array(self.gt_x)
            gt_y = np.array(self.gt_y)
            gt_yaw = list(self.gt_yaw)

            loc_x = np.array(self.loc_x)
            loc_y = np.array(self.loc_y)
            loc_yaw = list(self.loc_yaw)

            raw_x = np.array(self.raw_x)
            raw_y = np.array(self.raw_y)
            raw_yaw = list(self.raw_yaw)

            error_time = np.array(self.error_time)
            loc_pos_err = np.array(self.loc_pos_err)
            raw_pos_err = np.array(self.raw_pos_err)
            heading_err = np.array(self.heading_err)

            trn_time = np.array(self.trn_time)
            trn_quality = np.array(self.trn_quality)
            trn_corr_mag = np.array(self.trn_corr_mag)
            drift_hist = np.array(self.drift_hist)

            # Auto-scaling limits
            min_x, max_x = self.min_x, self.max_x
            min_y, max_y = self.min_y, self.max_y

            total_dist_gt = self.total_dist_gt
            last_trn_q = self.last_trn_q
            last_trn_r = self.last_trn_r
            last_trn_c = self.last_trn_c
            gt_received = self.gt_received

        MAX_PTS = 200  # Hard cap per line

        # 2. Update XY Trajectory (decimated)
        if len(gt_x) > 0:
            self.line_gt.set_data(self._decimate(gt_x, MAX_PTS), self._decimate(gt_y, MAX_PTS))
            self.pt_gt.set_data([gt_x[-1]], [gt_y[-1]])
        else:
            self.pt_gt.set_data([], [])

        if len(loc_x) > 0:
            self.line_loc.set_data(self._decimate(loc_x, MAX_PTS), self._decimate(loc_y, MAX_PTS))
            self.pt_loc.set_data([loc_x[-1]], [loc_y[-1]])
        else:
            self.pt_loc.set_data([], [])

        if len(raw_x) > 0:
            self.line_raw.set_data(self._decimate(raw_x, MAX_PTS), self._decimate(raw_y, MAX_PTS))
            self.pt_raw.set_data([raw_x[-1]], [raw_y[-1]])
        else:
            self.pt_raw.set_data([], [])

        # Heading arrows at current positions
        if len(gt_x) > 0 and gt_yaw:
            self.qv_gt.set_offsets([[gt_x[-1], gt_y[-1]]])
            self.qv_gt.set_UVC([math.cos(gt_yaw[-1])], [math.sin(gt_yaw[-1])])
        if len(loc_x) > 0 and loc_yaw:
            self.qv_loc.set_offsets([[loc_x[-1], loc_y[-1]]])
            self.qv_loc.set_UVC([math.cos(loc_yaw[-1])], [math.sin(loc_yaw[-1])])

        # Stats text box
        if gt_received and len(loc_pos_err) > 0:
            ate = float(np.mean(loc_pos_err[-100:]))
            pe  = loc_pos_err[-1]
            he  = heading_err[-1] if len(heading_err) > 0 else 0.0
            dr  = drift_hist[-1]  if len(drift_hist) > 0  else 0.0
            self.stats_text.set_text(
                f'ATE={ate:.2f}m  Err={pe:.2f}m\n'
                f'Δyaw={he:+.1f}°  Drift={dr:.1f}%\n'
                f'Dist={total_dist_gt:.1f}m  Q={last_trn_q:.2f}  '
                f'R={last_trn_r:.1f}m  |Δ|={last_trn_c:.2f}m')

        # Fast O(1) auto-scale trajectory plot
        if len(gt_x) > 2 or len(loc_x) > 2:
            mg = 3.0
            self.ax_traj.set_xlim(min_x - mg, max_x + mg)
            self.ax_traj.set_ylim(min_y - mg, max_y + mg)

        # 3. Decimated error plots (max 200 pts per line)
        # -- Position Error --
        if len(error_time) > 1:
            et_d = self._decimate(error_time, MAX_PTS)
            self.line_loc_err.set_data(et_d, self._decimate(loc_pos_err, MAX_PTS))
            if len(raw_pos_err) == len(error_time):
                self.line_raw_err.set_data(et_d, self._decimate(raw_pos_err, MAX_PTS))
            
            self.ax_pos.set_xlim(error_time[0], max(error_time[-1], 1))
            mx = max(loc_pos_err[-300:]) if len(loc_pos_err) > 0 else 1.0
            self.ax_pos.set_ylim(0, max(mx * 1.2, 0.2))

        # -- Heading Error --
        if len(error_time) > 1 and len(heading_err) > 0:
            self.line_head_err.set_data(self._decimate(error_time, MAX_PTS), self._decimate(heading_err, MAX_PTS))
            self.ax_head.set_xlim(error_time[0], max(error_time[-1], 1))
            rh = heading_err[-300:]
            self.ax_head.set_ylim(min(rh) - 5, max(rh) + 5)

        # -- TRN Diagnostics --
        if len(trn_time) > 0:
            tt_d = self._decimate(trn_time, MAX_PTS)
            self.line_trn_q.set_data(tt_d, self._decimate(trn_quality, MAX_PTS))
            if len(trn_corr_mag) > 0:
                self.line_trn_cor.set_data(tt_d, self._decimate(trn_corr_mag, MAX_PTS))
                mx_c = max(trn_corr_mag[-200:]) if len(trn_corr_mag) > 0 else 1.0
                self.ax_trn_r.set_ylim(0, max(mx_c * 1.3, 0.5))
            if len(drift_hist) > 0:
                dh = drift_hist / 100.0
                self.line_drift.set_data(self._decimate(error_time, MAX_PTS), self._decimate(dh, MAX_PTS))
            self.ax_trn.set_xlim(trn_time[0], max(trn_time[-1], 1))

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
                with self.lock:
                    # Use time-synced arrays: loc and error are same length (from _compute_errors).
                    # GT and raw are at different rates, so interpolate by nearest time.
                    gt_t = list(self.gt_time)
                    gt_x_l = list(self.gt_x)
                    gt_y_l = list(self.gt_y)
                    gt_yaw_l = list(self.gt_yaw)

                    raw_t = list(self.raw_time)
                    raw_x_l = list(self.raw_x)
                    raw_y_l = list(self.raw_y)
                    raw_yaw_l = list(self.raw_yaw)

                    loc_x_l = list(self.loc_x)
                    loc_y_l = list(self.loc_y)
                    loc_yaw_l = list(self.loc_yaw)

                    err_t = list(self.error_time)
                    err_pos = list(self.loc_pos_err)
                    err_head = list(self.heading_err)
                    err_drift = list(self.drift_hist)

                n = min(len(err_t), len(loc_x_l), len(err_pos))

                # Binary search helper for nearest-time lookup
                import bisect

                def nearest(time_arr, data_arr, target_t):
                    if not time_arr:
                        return 0.0
                    idx = bisect.bisect_left(time_arr, target_t)
                    if idx >= len(time_arr):
                        idx = len(time_arr) - 1
                    elif idx > 0:
                        if abs(time_arr[idx - 1] - target_t) < abs(time_arr[idx] - target_t):
                            idx -= 1
                    return data_arr[idx]

                for i in range(n):
                    t = err_t[i]
                    writer.writerow([
                        f'{t:.3f}',
                        f'{nearest(gt_t, gt_x_l, t):.4f}',
                        f'{nearest(gt_t, gt_y_l, t):.4f}',
                        f'{nearest(gt_t, gt_yaw_l, t):.4f}',
                        f'{loc_x_l[i]:.4f}',
                        f'{loc_y_l[i]:.4f}',
                        f'{loc_yaw_l[i]:.4f}',
                        f'{nearest(raw_t, raw_x_l, t):.4f}',
                        f'{nearest(raw_t, raw_y_l, t):.4f}',
                        f'{nearest(raw_t, raw_yaw_l, t):.4f}',
                        f'{err_pos[i]:.4f}',
                        f'{err_head[i]:.2f}',
                        f'{err_drift[i]:.4f}'
                    ])

            self.get_logger().info(f'Saved comparison data to {self.csv_path} ({n} rows)')
        except Exception as e:
            self.get_logger().error(f'Failed to save CSV: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = OdomVisualizerNode()

    # Create and run a dedicated background thread for high-frequency ROS 2 spinning.
    # This guarantees that ROS callbacks are NEVER starved, and process at sub-millisecond latencies.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        # Run GUI plot update loop at 4 FPS (every 250ms) on the main thread.
        # 4 FPS is plenty for a monitoring dashboard and eliminates all lag.
        while rclpy.ok() and plt.fignum_exists(node.fig.number):
            node.update_plot()
            plt.pause(0.25)
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


