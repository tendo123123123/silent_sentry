#!/usr/bin/env python3
"""ROS wrapper for the terrain-referenced navigation matcher."""

import math

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import TransformStamped, Vector3
from nav_msgs.msg import Odometry
from silent_sentry_interfaces.msg import LocalDEM
from std_msgs.msg import Float64

import tf2_ros

from tf_transformations import euler_from_quaternion, quaternion_from_euler

from .trn_core import TRNConfig, TRNCore, TRNMatchCycleResult


class TRNSlamNode(Node):
    """Own ROS interfaces and TF for the extracted TRN core."""

    def __init__(self):
        super().__init__('trn_slam_node')

        self.declare_parameter(
            'global_dem_path',
            '/home/sailesh/silent_sentry/src/2_ugv_fleet_brain/bot_navigation/maps/synthetic_dem.tif',
        )
        self.declare_parameter('global_dem_resolution', 1.0)
        self.declare_parameter('dem_origin_x', 'auto')
        self.declare_parameter('dem_origin_y', 'auto')
        self.declare_parameter('local_grid_nx', 20)
        self.declare_parameter('local_grid_ny', 20)
        self.declare_parameter('local_grid_resolution', 1.0)
        self.declare_parameter('submapping_buffer_seconds', 10.0)
        self.declare_parameter('composite_resolution', 1.0)
        self.declare_parameter('composite_max_width', 80.0)
        self.declare_parameter('composite_max_height', 110.0)
        self.declare_parameter('min_composite_coverage', 0.15)
        self.declare_parameter('bilateral_d', 9)
        self.declare_parameter('bilateral_sigma_color', 15.0)
        self.declare_parameter('bilateral_sigma_space', 75.0)
        self.declare_parameter('diffusion_kappa', 15.0)
        self.declare_parameter('diffusion_iterations', 8)
        self.declare_parameter('diffusion_gamma', 0.15)
        self.declare_parameter('base_search_radius', 50.0)
        self.declare_parameter('max_search_radius', 150.0)
        self.declare_parameter('covariance_scale', 2.0)
        self.declare_parameter('initial_search_radius', 150.0)
        self.declare_parameter('initial_match_count', 5)
        self.declare_parameter('match_rate', 3.0)
        self.declare_parameter('num_particles', 500)
        self.declare_parameter('particle_spread_xy', 0.5)
        self.declare_parameter('particle_spread_yaw', 0.02)
        self.declare_parameter('ncc_patch_radius', 0)
        self.declare_parameter('min_ncc_quality', 0.20)
        self.declare_parameter('ess_threshold', 0.50)
        self.declare_parameter('min_update_ess_ratio', 0.08)
        self.declare_parameter('flatness_std_threshold', 0.05)
        self.declare_parameter('alpha_slow', 0.001)
        self.declare_parameter('alpha_fast', 0.02)
        self.declare_parameter('amcl_max_random_injection', 0.10)
        self.declare_parameter('amcl_global_random_fraction', 0.10)
        self.declare_parameter('amcl_prior_std_scale', 0.25)
        self.declare_parameter('motion_noise_xy_frac', 0.15)
        self.declare_parameter('motion_noise_yaw_frac', 0.10)
        self.declare_parameter('roi_inject_fraction', 0.20)
        self.declare_parameter('entropy_threshold', 2.0)
        self.declare_parameter('degeneracy_covariance', 100.0)
        self.declare_parameter('nominal_covariance', 1.0)
        self.declare_parameter('min_peak_quality', 0.25)
        self.declare_parameter('max_correction_per_cycle', 8.0)
        self.declare_parameter('max_map_shift_per_cycle', 0.5)
        self.declare_parameter('ema_alpha', 0.40)
        self.declare_parameter('tf_publish_rate', 10.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_link_frame', 'base_footprint')

        self.core_config = TRNConfig(
            global_dem_path=self._string_param('global_dem_path'),
            global_res=self._p('global_dem_resolution'),
            local_nx=self._i('local_grid_nx'),
            local_ny=self._i('local_grid_ny'),
            local_res=self._p('local_grid_resolution'),
            min_composite_cov=self._p('min_composite_coverage'),
            bilateral_d=self._i('bilateral_d'),
            bilateral_sc=self._p('bilateral_sigma_color'),
            bilateral_ss=self._p('bilateral_sigma_space'),
            base_search_radius=self._p('base_search_radius'),
            max_search_radius=self._p('max_search_radius'),
            covariance_scale=self._p('covariance_scale'),
            initial_search_radius=self._p('initial_search_radius'),
            initial_match_count=self._i('initial_match_count'),
            match_rate=self._p('match_rate'),
            num_particles=self._i('num_particles'),
            particle_spread_xy=self._p('particle_spread_xy'),
            particle_spread_yaw=self._p('particle_spread_yaw'),
            ess_threshold=self._p('ess_threshold'),
            min_update_ess_ratio=self._p('min_update_ess_ratio'),
            flatness_std_threshold=self._p('flatness_std_threshold'),
            alpha_slow=self._p('alpha_slow'),
            alpha_fast=self._p('alpha_fast'),
            amcl_max_random_injection=self._p('amcl_max_random_injection'),
            amcl_global_random_fraction=self._p('amcl_global_random_fraction'),
            amcl_prior_std_scale=self._p('amcl_prior_std_scale'),
            motion_noise_xy_frac=self._p('motion_noise_xy_frac'),
            motion_noise_yaw_frac=self._p('motion_noise_yaw_frac'),
            roi_inject_fraction=self._p('roi_inject_fraction'),
            entropy_thresh=self._p('entropy_threshold'),
            min_peak_quality=self._p('min_peak_quality'),
            max_correction=self._p('max_correction_per_cycle'),
            max_map_shift_per_cycle=self._p('max_map_shift_per_cycle'),
            ema_alpha=self._p('ema_alpha'),
            tf_publish_rate=self._p('tf_publish_rate'),
            map_frame=self._string_param('map_frame'),
            odom_frame=self._string_param('odom_frame'),
            base_link_frame=self._string_param('base_link_frame'),
            dem_origin_x_str=self._string_param('dem_origin_x'),
            dem_origin_y_str=self._string_param('dem_origin_y'),
        )
        self.core = TRNCore(self.core_config, logger=self.get_logger())

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.quality_pub = self.create_publisher(Float64, '/trn/match_quality', 10)
        self.entropy_pub = self.create_publisher(Float64, '/trn/entropy', 10)
        self.correction_pub = self.create_publisher(Vector3, '/trn/correction', 10)
        self.radius_pub = self.create_publisher(Float64, '/trn/search_radius', 10)
        self.composite_size_pub = self.create_publisher(Vector3, '/trn/composite_size', 10)

        self.create_subscription(
            LocalDEM,
            '/elevation_map/local_dem',
            self.local_dem_callback,
            5,
        )
        self.create_subscription(Odometry, '/odometry/filtered', self.ekf_odom_callback, 10)

        self._local_dem_received = False
        self._odom_received = False
        self._localization_tf_ready = False
        self._wait_logs = set()
        self._ready_logs = set()

        self.match_timer = self.create_timer(
            1.0 / self.core_config.match_rate,
            self.run_trn_match,
        )
        self.tf_timer = self.create_timer(
            1.0 / self.core_config.tf_publish_rate,
            self.publish_map_to_odom_tf,
        )

        self.get_logger().info(
            f'TRN SLAM Node v3 (MCL + MAD) -- '
            f'frontend=rolling_local_dem, '
            f'particles={self.core_config.num_particles}, '
            f'match_rate={self.core_config.match_rate}Hz, '
            f'tf_pub={self.core_config.tf_publish_rate:.1f}Hz, '
            f'bilateral_d={self.core_config.bilateral_d}, '
            f'entropy_thresh={self.core_config.entropy_thresh}, '
            f'flatness_std={self.core_config.flatness_std_threshold:.3f}m, '
            f'EMA_alpha={self.core_config.ema_alpha}, '
            f'max_corr={self.core_config.max_correction:.2f}m, '
            f'max_step={self.core_config.max_map_shift_per_cycle:.2f}m/cycle'
        )

    def _p(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _i(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _string_param(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _log_wait_once(self, key: str, message: str):
        if key in self._wait_logs:
            return
        self._wait_logs.add(key)
        self.get_logger().info(message)

    def _log_ready_once(self, key: str, message: str):
        if key in self._ready_logs:
            return
        self._ready_logs.add(key)
        self.get_logger().info(message)

    def _localization_tf_available(self) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                self.core_config.odom_frame,
                self.core_config.base_link_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            return True
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return False

    def local_dem_callback(self, msg: LocalDEM):
        if not self._local_dem_received:
            self._local_dem_received = True
            self._log_ready_once(
                'local_dem',
                'TRN ready gate satisfied: first typed /elevation_map/local_dem received',
            )
        stamp_ns = (
            msg.acquisition_stamp.sec * 1_000_000_000
            + msg.acquisition_stamp.nanosec
        )
        self.core.update_local_dem(
            msg.data,
            int(msg.height),
            int(msg.width),
            stamp_ns,
            origin_x=msg.origin_x,
            origin_y=msg.origin_y,
            center_x=msg.center_x,
            center_y=msg.center_y,
            resolution=msg.resolution,
        )

    def ekf_odom_callback(self, msg: Odometry):
        if not self._odom_received:
            self._odom_received = True
            self._log_ready_once(
                'odom',
                'TRN ready gate satisfied: first /odometry/filtered sample received',
            )
        quat = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        cov = msg.pose.covariance
        self.core.update_odom(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw,
            cov[0],
            cov[7],
        )

    def _get_map_frame_prior(self) -> tuple[float, float, bool]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.core_config.map_frame,
                self.core_config.base_link_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            return (
                transform.transform.translation.x,
                transform.transform.translation.y,
                True,
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            prior_x, prior_y = self.core.fallback_map_prior()
            return prior_x, prior_y, False

    def run_trn_match(self):
        if not self.core.has_global_dem():
            return
        if not self._local_dem_received or not self.core.has_local_dem():
            self._log_wait_once(
                'wait_local_dem',
                'TRN waiting for first typed local DEM before matching',
            )
            return
        if not self._odom_received:
            self._log_wait_once(
                'wait_odom',
                'TRN waiting for first /odometry/filtered sample before matching',
            )
            return
        if not self._localization_tf_ready:
            if not self._localization_tf_available():
                self._log_wait_once(
                    'wait_tf',
                    f'TRN waiting for TF {self.core_config.odom_frame}->{self.core_config.base_link_frame} before matching',
                )
                return
            self._localization_tf_ready = True
            self._log_ready_once(
                'tf',
                f'TRN ready gate satisfied: TF {self.core_config.odom_frame}->{self.core_config.base_link_frame} available',
            )

        prior_x, prior_y, tf_success = self._get_map_frame_prior()
        result = self.core.run_match_cycle(
            prior_x,
            prior_y,
            tf_success,
            self.get_clock().now().nanoseconds,
        )
        if result is None:
            return

        if result.composite_width_m is not None:
            size_msg = Vector3()
            size_msg.x = result.composite_width_m
            size_msg.y = result.composite_height_m or 0.0
            size_msg.z = result.composite_coverage or 0.0
            self.composite_size_pub.publish(size_msg)

        if result.entropy is not None:
            self._pub_f64(self.entropy_pub, result.entropy)
        if result.search_radius is not None:
            self._pub_f64(self.radius_pub, result.search_radius)
        if result.quality is not None:
            self._pub_f64(self.quality_pub, result.quality)
        if result.odom_correction_x is not None and result.odom_correction_y is not None:
            correction_msg = Vector3()
            correction_msg.x = result.odom_correction_x
            correction_msg.y = result.odom_correction_y
            correction_msg.z = 0.0
            self.correction_pub.publish(correction_msg)

    def publish_map_to_odom_tf(self):
        output = self.core.map_to_odom_output()
        transform = TransformStamped()

        now = self.get_clock().now()
        if output.last_composite_stamp_ns is not None:
            age = (now.nanoseconds - output.last_composite_stamp_ns) / 1e9
            if age < 3.0:
                transform.header.stamp = Time(
                    nanoseconds=output.last_composite_stamp_ns
                ).to_msg()
            else:
                transform.header.stamp = now.to_msg()
        else:
            transform.header.stamp = now.to_msg()

        transform.header.frame_id = self.core_config.map_frame
        transform.child_frame_id = self.core_config.odom_frame
        transform.transform.translation.x = output.x
        transform.transform.translation.y = output.y
        transform.transform.translation.z = 0.0

        quat = quaternion_from_euler(0.0, 0.0, output.yaw)
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(transform)

    def _pub_f64(self, pub, value: float):
        message = Float64()
        message.data = value
        pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = TRNSlamNode()
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
