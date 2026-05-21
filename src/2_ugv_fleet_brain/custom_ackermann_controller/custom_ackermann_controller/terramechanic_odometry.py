#!/usr/bin/env python3
"""ROS wrapper for the terramechanic odometry estimator."""

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float64
from tf_transformations import quaternion_from_euler

from .terramechanic_core import (
    TerramechanicConfig,
    TerramechanicOdometryCore,
)


class TerramechanicOdometryNode(Node):
    """Own ROS I/O for the terramechanic odometry core."""

    def __init__(self):
        super().__init__('terramechanic_odometry_node')

        self.declare_parameter('wheelbase', 0.9)
        self.declare_parameter('track_width', 0.67)
        self.declare_parameter('wheel_radius_nominal', 0.175)
        self.declare_parameter('tire_width', 0.10)
        self.declare_parameter('front_axle_distance', 0.45)
        self.declare_parameter('rear_axle_distance', 0.45)

        self.declare_parameter('vehicle_mass', 85.0)
        self.declare_parameter('wheel_count', 4)

        self.declare_parameter('bekker_n', 1.1)
        self.declare_parameter('bekker_kc', 0.9)
        self.declare_parameter('bekker_kphi', 1528.0)

        self.declare_parameter('slip_covariance_gain', 50.0)
        self.declare_parameter('max_slip_ratio', 0.95)
        self.declare_parameter('min_wheel_omega', 0.05)

        self.declare_parameter('cornering_stiffness', 500.0)
        self.declare_parameter('understeer_gradient', 0.08)

        self.declare_parameter('gyro_kf_process_noise_omega', 0.1)
        self.declare_parameter('gyro_kf_process_noise_bias', 0.001)
        self.declare_parameter('gyro_kf_meas_noise_kinematic', 0.5)
        self.declare_parameter('gyro_kf_meas_noise_imu', 0.01)
        self.declare_parameter('gyro_fusion_alpha', 0.15)

        self.declare_parameter('zupt_omega_threshold', 0.02)
        self.declare_parameter('zupt_accel_threshold', 0.3)

        self.declare_parameter('imu_velocity_alpha', 0.98)
        self.declare_parameter('imu_accel_bias_alpha', 0.001)
        self.declare_parameter('max_imu_velocity', 5.0)

        self.declare_parameter('stall_detection_enabled', True)
        self.declare_parameter('stall_imu_velocity_threshold', 0.15)
        self.declare_parameter('stall_encoder_velocity_threshold', 0.1)
        self.declare_parameter('stall_duration_threshold', 0.5)
        self.declare_parameter('stall_slip_ratio_threshold', 0.7)
        self.declare_parameter('stall_covariance_multiplier', 100.0)

        self.declare_parameter('sand_slip_coefficient', 0.4)
        self.declare_parameter('tilt_covariance_gain', 3.0)
        self.declare_parameter('yaw_sign', 1.0)

        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')

        self.declare_parameter('left_wheel_joint', 'base_back_left_wheel_joint')
        self.declare_parameter('right_wheel_joint', 'base_back_right_wheel_joint')
        self.declare_parameter('left_steering_joint', 'base_front_left_steering_joint')
        self.declare_parameter('right_steering_joint', 'base_front_right_steering_joint')

        self.declare_parameter('publish_rate', 50.0)

        self.declare_parameter('base_position_variance', 0.005)
        self.declare_parameter('base_orientation_variance', 0.01)
        self.declare_parameter('base_velocity_variance', 0.01)

        self.declare_parameter('velocity_filter_alpha', 0.3)
        self.declare_parameter('max_wheel_acceleration', 5.0)
        self.declare_parameter('deadzone_threshold', 0.003)

        self.core_config = TerramechanicConfig(
            wheelbase=self._p('wheelbase'),
            track_width=self._p('track_width'),
            r_nominal=self._p('wheel_radius_nominal'),
            tire_width=self._p('tire_width'),
            vehicle_mass=self._p('vehicle_mass'),
            wheel_count=self._int_param('wheel_count'),
            bekker_n=self._p('bekker_n'),
            bekker_kc=self._p('bekker_kc') * 1000.0,
            bekker_kphi=self._p('bekker_kphi') * 1000.0,
            k_slip=self._p('slip_covariance_gain'),
            max_slip_ratio=self._p('max_slip_ratio'),
            K_us=self._p('understeer_gradient'),
            gyro_kf_Q_omega=self._p('gyro_kf_process_noise_omega'),
            gyro_kf_Q_bias=self._p('gyro_kf_process_noise_bias'),
            gyro_kf_R_kin=self._p('gyro_kf_meas_noise_kinematic'),
            gyro_kf_R_imu=self._p('gyro_kf_meas_noise_imu'),
            zupt_omega_threshold=self._p('zupt_omega_threshold'),
            zupt_accel_threshold=self._p('zupt_accel_threshold'),
            imu_accel_bias_alpha=self._p('imu_accel_bias_alpha'),
            max_imu_velocity=self._p('max_imu_velocity'),
            stall_detection_enabled=self._bool_param('stall_detection_enabled'),
            stall_imu_vel_thresh=self._p('stall_imu_velocity_threshold'),
            stall_encoder_vel_thresh=self._p('stall_encoder_velocity_threshold'),
            stall_duration_thresh=self._p('stall_duration_threshold'),
            stall_slip_ratio_thresh=self._p('stall_slip_ratio_threshold'),
            stall_cov_multiplier=self._p('stall_covariance_multiplier'),
            sand_slip_coeff=self._p('sand_slip_coefficient'),
            tilt_cov_gain=self._p('tilt_covariance_gain'),
            yaw_sign=self._p('yaw_sign'),
            odom_frame=self._string_param('odom_frame'),
            base_frame=self._string_param('base_frame'),
            left_wheel_joint=self._string_param('left_wheel_joint'),
            right_wheel_joint=self._string_param('right_wheel_joint'),
            left_steering_joint=self._string_param('left_steering_joint'),
            right_steering_joint=self._string_param('right_steering_joint'),
            publish_rate=self._p('publish_rate'),
            base_pos_var=self._p('base_position_variance'),
            base_orient_var=self._p('base_orientation_variance'),
            base_vel_var=self._p('base_velocity_variance'),
            velocity_filter_alpha=self._p('velocity_filter_alpha'),
            max_wheel_accel=self._p('max_wheel_acceleration'),
            deadzone=self._p('deadzone_threshold'),
        )
        self.core = TerramechanicOdometryCore(
            self.core_config,
            logger=self.get_logger(),
            start_time_s=self._now_seconds(),
        )

        self.odom_pub = self.create_publisher(Odometry, '/terramechanic_odom', 10)
        self.slip_ratio_pub = self.create_publisher(
            Float64,
            '/terramechanic/slip_ratio',
            10,
        )
        self.sinkage_pub = self.create_publisher(
            Float64,
            '/terramechanic/sinkage',
            10,
        )
        self.r_eff_pub = self.create_publisher(Float64, '/terramechanic/r_eff', 10)
        self.zupt_pub = self.create_publisher(Bool, '/terramechanic/zupt_active', 10)
        self.omega_fused_pub = self.create_publisher(
            Float64,
            '/terramechanic/omega_fused',
            10,
        )
        self.omega_imu_pub = self.create_publisher(
            Float64,
            '/terramechanic/omega_imu',
            10,
        )
        self.omega_kinem_pub = self.create_publisher(
            Float64,
            '/terramechanic/omega_kinem',
            10,
        )
        self.stall_pub = self.create_publisher(Bool, '/terramechanic/stall_active', 10)

        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.create_subscription(Imu, '/imu/data_filtered', self.imu_callback, 10)

        self.timer = self.create_timer(
            1.0 / self.core_config.publish_rate,
            self.publish_odometry,
        )

        self.get_logger().info(
            'Slip-Aware Terramechanic Odometry Node initialized - '
            f'Bekker-Wong model, '
            f'K_us={self.core_config.K_us} rad*s^2/m^2, '
            f'k_slip={self.core_config.k_slip}, '
            f'stall_detect={self.core_config.stall_detection_enabled}, '
            f'sand_slip={self.core_config.sand_slip_coeff}, '
            f'yaw_sign={self.core_config.yaw_sign}'
        )

    def _p(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _int_param(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _bool_param(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def _string_param(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def imu_callback(self, msg: Imu):
        orientation = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ],
            dtype=np.float64,
        )
        angular_velocity = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=np.float64,
        )
        linear_acceleration = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=np.float64,
        )
        self.core.process_imu(
            orientation,
            angular_velocity,
            linear_acceleration,
            self._now_seconds(),
        )

    def joint_state_callback(self, msg: JointState):
        updated = self.core.process_joint_state(
            list(msg.name),
            list(msg.position),
            self._now_seconds(),
        )
        if updated:
            self._publish_diagnostics()

    def publish_odometry(self):
        output = self.core.build_odometry_output(self._now_seconds())
        if output is None:
            return

        current_time = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = self.core_config.odom_frame
        odom.child_frame_id = self.core_config.base_frame

        odom.pose.pose.position.x = output.x
        odom.pose.pose.position.y = output.y
        odom.pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0.0, 0.0, output.theta)
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        odom.twist.twist.linear.x = output.linear_velocity
        odom.twist.twist.linear.y = output.lateral_velocity
        odom.twist.twist.angular.z = output.angular_velocity

        odom.pose.covariance = [
            output.pos_var, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, output.pos_var, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, output.orient_var,
        ]
        odom.twist.covariance = [
            output.vel_var, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, output.vel_lateral_var, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, output.vel_vertical_var, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, output.vel_var * 2.0,
        ]
        self.odom_pub.publish(odom)

    def _publish_diagnostics(self):
        diagnostics = self.core.diagnostics()

        slip_msg = Float64()
        slip_msg.data = diagnostics.slip_ratio
        self.slip_ratio_pub.publish(slip_msg)

        sinkage_msg = Float64()
        sinkage_msg.data = diagnostics.sinkage
        self.sinkage_pub.publish(sinkage_msg)

        r_eff_msg = Float64()
        r_eff_msg.data = diagnostics.r_eff
        self.r_eff_pub.publish(r_eff_msg)

        zupt_msg = Bool()
        zupt_msg.data = diagnostics.zupt_active
        self.zupt_pub.publish(zupt_msg)

        stall_msg = Bool()
        stall_msg.data = diagnostics.stall_active
        self.stall_pub.publish(stall_msg)

        omega_fused_msg = Float64()
        omega_fused_msg.data = diagnostics.omega_fused
        self.omega_fused_pub.publish(omega_fused_msg)

        omega_imu_msg = Float64()
        omega_imu_msg.data = diagnostics.omega_imu
        self.omega_imu_pub.publish(omega_imu_msg)

        omega_kinem_msg = Float64()
        omega_kinem_msg.data = diagnostics.omega_kinematic
        self.omega_kinem_pub.publish(omega_kinem_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TerramechanicOdometryNode()
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
