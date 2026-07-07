#!/usr/bin/env python3
"""ROS wrapper for the factor-graph dead-reckoning backend."""

import os
import sys


def _ensure_venv():
    current = os.path.abspath(__file__)
    for _ in range(10):
        current = os.path.dirname(current)
        site_packages = os.path.join(
            current,
            '.venv',
            'lib',
            'python3.12',
            'site-packages',
        )
        if os.path.isdir(site_packages) and site_packages not in sys.path:
            sys.path.insert(0, site_packages)
            return


_ensure_venv()

import math

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

import tf2_ros

from tf_transformations import quaternion_from_euler

from .factor_graph_core import FactorGraphConfig, FactorGraphCore


class FactorGraphFuser(Node):
    """Own ROS I/O for the extracted factor graph core."""

    def __init__(self):
        super().__init__('factor_graph_fuser')

        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('odom_sigma_xy', 0.10)
        self.declare_parameter('odom_sigma_theta', 0.14)
        self.declare_parameter('imu_yaw_sigma', 0.07)
        self.declare_parameter('imu_roll_pitch_sigma', 0.05)
        self.declare_parameter('imu_accel_sigma', 0.35)
        self.declare_parameter('imu_gyro_sigma', 0.08)
        self.declare_parameter('imu_integration_sigma', 0.01)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('max_velocity', 4.5)
        self.declare_parameter('imu_yaw_jump_gate', 0.5)
        self.declare_parameter('slip_accel_threshold', 1.5)
        self.declare_parameter('slip_cov_multiplier', 25.0)
        self.declare_parameter('keyframe_min_dist', 0.05)
        self.declare_parameter('keyframe_min_angle', 0.02)
        self.declare_parameter('position_noise_per_meter', 0.01)
        self.declare_parameter('heading_variance', 0.001)

        self.core_config = FactorGraphConfig(
            publish_rate=self._pf('publish_rate'),
            odom_sig_xy=self._pf('odom_sigma_xy'),
            imu_sig=self._pf('imu_yaw_sigma'),
            imu_rp_sig=self._pf('imu_roll_pitch_sigma'),
            imu_accel_sig=self._pf('imu_accel_sigma'),
            imu_gyro_sig=self._pf('imu_gyro_sigma'),
            imu_integration_sig=self._pf('imu_integration_sigma'),
            odom_frame=self._string_param('odom_frame'),
            base_frame=self._string_param('base_frame'),
            max_vel=self._pf('max_velocity'),
            yaw_gate=self._pf('imu_yaw_jump_gate'),
            slip_accel_threshold=self._pf('slip_accel_threshold'),
            slip_cov_multiplier=self._pf('slip_cov_multiplier'),
            pos_noise_pm=self._pf('position_noise_per_meter'),
            heading_var=self._pf('heading_variance'),
            kf_min_dist=self._pf('keyframe_min_dist'),
            kf_min_angle=self._pf('keyframe_min_angle'),
        )
        self.core = FactorGraphCore(self.core_config, logger=self.get_logger())

        self.tf_br = tf2_ros.TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 10)

        self._imu_received = False
        self._wheel_odom_received = False
        self._wait_logs = set()

        self.create_subscription(Odometry, '/terramechanic_odom', self._odom_cb, 10)
        self.create_subscription(Imu, '/imu/data_filtered', self._imu_cb, 10)
        self.create_subscription(Float64, '/trn/match_quality', self._trn_quality_cb, 10)
        self.create_subscription(Vector3, '/trn/correction', self._trn_correction_cb, 10)

        self.pub_timer = self.create_timer(
            1.0 / self.core_config.publish_rate,
            self._publish,
        )

        self.get_logger().info(
            'FactorGraphFuser (GTSAM iSAM2, SE(3) IMU preintegration) -- '
            f'rate={self.core_config.publish_rate}Hz, '
            f'wheel_sigma={self.core_config.odom_sig_xy}, '
            f'imu_att=({self.core_config.imu_rp_sig},{self.core_config.imu_sig}), '
            f'imu_preint=('
            f'{self.core_config.imu_accel_sig},'
            f'{self.core_config.imu_gyro_sig},'
            f'{self.core_config.imu_integration_sig}), '
            f'slip_da={self.core_config.slip_accel_threshold:.2f}, '
            f'slip_cov_x{self.core_config.slip_cov_multiplier:.1f}, '
            f'kf_dist={self.core_config.kf_min_dist}m, '
            f'kf_angle={math.degrees(self.core_config.kf_min_angle):.1f}deg'
        )

    def _pf(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _string_param(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _log_wait_once(self, key: str, message: str):
        if key in self._wait_logs:
            return
        self._wait_logs.add(key)
        self.get_logger().info(message)

    def _time_from_msg(self, stamp_msg) -> float:
        if stamp_msg.sec == 0 and stamp_msg.nanosec == 0:
            return self.get_clock().now().nanoseconds / 1e9
        return stamp_msg.sec + stamp_msg.nanosec / 1e9

    def _imu_cb(self, msg: Imu):
        if not self._imu_received:
            self._imu_received = True
            self.get_logger().info(
                'Factor graph ready gate satisfied: first /imu/data_filtered sample received'
            )
        if not self._wheel_odom_received:
            return
        quat = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ],
            dtype=np.float64,
        )
        linear_accel = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=np.float64,
        )
        angular_vel = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=np.float64,
        )
        self.core.process_imu(
            quat,
            linear_accel,
            angular_vel,
            self._time_from_msg(msg.header.stamp),
        )

    def _trn_quality_cb(self, msg: Float64):
        self.core.set_trn_quality(msg.data)

    def _trn_correction_cb(self, msg: Vector3):
        self.core.add_trn_correction_factor(msg.x, msg.y)

    def _odom_cb(self, msg: Odometry):
        if not self._wheel_odom_received:
            self._wheel_odom_received = True
            self.core.reset_to_identity()
            self.get_logger().info(
                'First wheel odometry received — resetting factor graph to identity '
                'and enabling IMU preintegration'
            )
        self.core.process_odom(
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z,
            list(msg.twist.covariance),
            self._time_from_msg(msg.header.stamp),
        )

    def _publish(self):
        if not self._imu_received:
            self._log_wait_once(
                'imu',
                'Factor graph waiting for first /imu/data_filtered sample before publishing odom->base_footprint',
            )
            return
        if not self._wheel_odom_received:
            self._log_wait_once(
                'wheel_odom',
                'Factor graph waiting for first /terramechanic_odom sample before publishing odom->base_footprint',
            )
            return

        output = self.core.build_publish_output()
        if output is None:
            return

        stamp_msg = self.get_clock().now().to_msg()
        quat = quaternion_from_euler(0.0, 0.0, output.theta)
        q_norm = quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2
        if not math.isfinite(q_norm) or q_norm < 0.5:
            quat = (0.0, 0.0, 0.0, 1.0)

        transform = TransformStamped()
        transform.header.stamp = stamp_msg
        transform.header.frame_id = self.core_config.odom_frame
        transform.child_frame_id = self.core_config.base_frame
        transform.transform.translation.x = output.x
        transform.transform.translation.y = output.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        self.tf_br.sendTransform(transform)

        odom = Odometry()
        odom.header.stamp = stamp_msg
        odom.header.frame_id = self.core_config.odom_frame
        odom.child_frame_id = self.core_config.base_frame
        odom.pose.pose.position.x = output.x
        odom.pose.pose.position.y = output.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        odom.pose.covariance = [
            output.pos_cov, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, output.pos_cov, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, self.core_config.heading_var,
        ]
        odom.twist.twist.linear.x = output.vx
        odom.twist.twist.angular.z = output.omega
        odom.twist.covariance = list(output.twist_covariance)
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = FactorGraphFuser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()