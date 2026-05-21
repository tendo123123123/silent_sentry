"""
IMU Covariance Fixer
====================
Gazebo's ros_gz_bridge outputs sensor_msgs/Imu with all-zero covariance matrices,
regardless of the noise parameters defined in the sensor SDF. robot_localization's
UKF requires R > 0 for its measurement noise matrix. When R = 0, the Kalman gain
K = P*H^T * (H*P*H^T + R)^-1 = P*H^T * (H*P*H^T)^-1 drives P to zero after
the first update tick. The UKF then attempts Cholesky decomposition of a near-zero
(or numerically negative-definite) P to generate sigma points → NaN explosion.

This node intercepts /imu/data_filtered (Madgwick output, still zero covariance)
and republishes on /imu/data_filtered_cov with physically realistic diagonal
covariance values derived from the URDF sensor noise model:

  angular_velocity noise stddev  = 2e-4 rad/s   → variance = 4e-8
  linear_acceleration noise stddev = 1.7e-2 m/s² → variance = 2.89e-4
  orientation (Madgwick, no mag)  = ~0.01 rad    → variance = 1e-4
  (Madgwick without magnetometer provides good roll/pitch but yaw drifts;
   0.01 rad is a conservative accuracy estimate for gravity-aligned axes)

The relay is zero-copy for all other message fields (header, stamp, frame_id,
orientation quaternion, angular_velocity, linear_acceleration).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math


# ---------------------------------------------------------------------------
# Covariance values — tuned to match URDF noise model in bot.urdf.xacro
# ---------------------------------------------------------------------------

# Madgwick without magnetometer: roll/pitch gravity-aligned (~0.001 rad accuracy),
# yaw drifts slowly (~0.01 rad/s without correction). Use uniform value for safety.
_ORIENTATION_VAR = 1e-4          # (0.01 rad)^2

# From URDF: angular_velocity noise stddev = 2e-4 rad/s → var = 4e-8
_ANGULAR_VEL_VAR = 4e-8          # (2e-4 rad/s)^2

# From URDF: linear_acceleration noise stddev = 1.7e-2 m/s² → var = 2.89e-4
# (This is the raw accelerometer noise. After gravity subtraction by UKF, residual
#  depends on orientation error: σ_a_residual ≈ 9.81 * σ_orientation ≈ 0.1 m/s²
#  so we inflate slightly to 1e-3 to account for rotation error contribution.)
_LINEAR_ACCEL_VAR = 1e-3         # conservative, covers gravity residual

# 3x3 diagonal covariance matrices (row-major, 9 elements each)
_ORIENTATION_COV = [_ORIENTATION_VAR, 0.0, 0.0,
                    0.0, _ORIENTATION_VAR, 0.0,
                    0.0, 0.0, _ORIENTATION_VAR]

_ANGULAR_VEL_COV = [_ANGULAR_VEL_VAR, 0.0, 0.0,
                    0.0, _ANGULAR_VEL_VAR, 0.0,
                    0.0, 0.0, _ANGULAR_VEL_VAR]

_LINEAR_ACCEL_COV = [_LINEAR_ACCEL_VAR, 0.0, 0.0,
                     0.0, _LINEAR_ACCEL_VAR, 0.0,
                     0.0, 0.0, _LINEAR_ACCEL_VAR]


class ImuCovarianceFixer(Node):
    def __init__(self):
        super().__init__('imu_covariance_fixer')

        self.sub = self.create_subscription(
            Imu, '/imu/data_filtered',
            self._callback, 20)

        self.pub = self.create_publisher(
            Imu, '/imu/data_filtered_cov', 20)

        self.get_logger().info(
            'IMU covariance fixer active: '
            f'/imu/data_filtered → /imu/data_filtered_cov | '
            f'orientation_var={_ORIENTATION_VAR:.2e} '
            f'angular_vel_var={_ANGULAR_VEL_VAR:.2e} '
            f'linear_accel_var={_LINEAR_ACCEL_VAR:.2e}')

    def _callback(self, msg: Imu):
        q = msg.orientation
        # Guard 1: reject NaN anywhere in the message
        vals = [q.x, q.y, q.z, q.w,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        if any(math.isnan(v) for v in vals):
            self.get_logger().warn('Dropping IMU message with NaN values', throttle_duration_sec=2.0)
            return

        # Guard 2: reject zero-norm quaternion (default ROS 2 Quaternion is [0,0,0,0])
        # A valid quaternion must have norm ≈ 1.  Norm < 0.5 indicates uninitialised data.
        qnorm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if qnorm < 0.5:
            self.get_logger().warn(
                f'Dropping IMU message with denormalized quaternion (norm={qnorm:.4f})',
                throttle_duration_sec=2.0)
            return

        out = Imu()
        out.header = msg.header
        out.orientation = msg.orientation
        out.angular_velocity = msg.angular_velocity
        out.linear_acceleration = msg.linear_acceleration

        # Fill covariances — override zeros from Gazebo bridge
        out.orientation_covariance = _ORIENTATION_COV
        out.angular_velocity_covariance = _ANGULAR_VEL_COV
        out.linear_acceleration_covariance = _LINEAR_ACCEL_COV

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImuCovarianceFixer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
