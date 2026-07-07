"""Core terramechanic odometry estimator state and math."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence

import numpy as np
from tf_transformations import euler_from_quaternion


@dataclass(frozen=True)
class TerramechanicConfig:
    wheelbase: float
    track_width: float
    r_nominal: float
    tire_width: float
    vehicle_mass: float
    wheel_count: int
    bekker_n: float
    bekker_kc: float
    bekker_kphi: float
    k_slip: float
    max_slip_ratio: float
    K_us: float
    gyro_kf_Q_omega: float
    gyro_kf_Q_bias: float
    gyro_kf_R_kin: float
    gyro_kf_R_imu: float
    zupt_omega_threshold: float
    zupt_accel_threshold: float
    imu_accel_bias_alpha: float
    max_imu_velocity: float
    stall_detection_enabled: bool
    stall_imu_vel_thresh: float
    stall_encoder_vel_thresh: float
    stall_duration_thresh: float
    stall_slip_ratio_thresh: float
    stall_cov_multiplier: float
    sand_slip_coeff: float
    tilt_cov_gain: float
    yaw_sign: float
    odom_frame: str
    base_frame: str
    left_wheel_joint: str
    right_wheel_joint: str
    left_steering_joint: str
    right_steering_joint: str
    publish_rate: float
    base_pos_var: float
    base_orient_var: float
    base_vel_var: float
    velocity_filter_alpha: float
    max_wheel_accel: float
    deadzone: float


@dataclass(frozen=True)
class TerramechanicDiagnostics:
    slip_ratio: float
    sinkage: float
    r_eff: float
    zupt_active: bool
    stall_active: bool
    omega_fused: float
    omega_imu: float
    omega_kinematic: float


@dataclass(frozen=True)
class TerramechanicOdometryOutput:
    x: float
    y: float
    theta: float
    linear_velocity: float
    lateral_velocity: float
    angular_velocity: float
    pos_var: float
    orient_var: float
    vel_var: float
    vel_lateral_var: float
    vel_vertical_var: float


class TerramechanicOdometryCore:
    """Stateful terramechanic odometry estimator without ROS node ownership."""

    def __init__(
        self,
        config: TerramechanicConfig,
        logger: Optional[Any] = None,
        start_time_s: float = 0.0,
    ):
        self.config = config
        self.logger = logger
        for name, value in vars(config).items():
            setattr(self, name, value)

        self.r_eff, self.sinkage = self._compute_bekker_sinkage()
        self._log_info(
            'Bekker Sinkage Model: '
            f'z0={self.sinkage:.4f}m, '
            f'r_eff={self.r_eff:.4f}m '
            f'(nominal={self.r_nominal:.4f}m, '
            f'delta={((self.r_nominal - self.r_eff) / self.r_nominal) * 100:.1f}% reduction)'
        )

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_time = start_time_s
        self.last_odom_time = None

        self.last_wheel_pos = {'left': 0.0, 'right': 0.0}
        self.joint_names = []
        self.initialized = False

        self.filtered_wheel_vel = {'left': 0.0, 'right': 0.0}
        self.last_raw_wheel_vel = {'left': 0.0, 'right': 0.0}

        self.current_slip_ratio = 0.0
        self.current_steering_angle = 0.0
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        self.v_true_imu = 0.0
        self.imu_accel_bias = 0.0
        self.imu_orientation_q = None
        self.imu_linear_accel_body = np.zeros(3, dtype=np.float64)
        self.last_imu_time = None
        self.imu_received = False

        self.imu_gyro_yaw_rate = 0.0

        self.gyro_kf_x = np.array([0.0, 0.0], dtype=np.float64)
        self.gyro_kf_P = np.diag([0.1, 0.01]).astype(np.float64)
        self.gyro_kf_initialized = False

        self.is_zupt = False
        self.is_stalled = False
        self.stall_accumulator = 0.0

        self.ahrs_roll = 0.0
        self.ahrs_pitch = 0.0
        self.omega_kinematic = 0.0

        self.debug_counter = 0

    def _log_info(self, message: str):
        if self.logger is not None:
            self.logger.info(message)

    def _log_warn(self, message: str):
        if self.logger is not None:
            self.logger.warn(message)

    def _log_error(self, message: str):
        if self.logger is not None:
            self.logger.error(message)

    def _compute_bekker_sinkage(self) -> tuple[float, float]:
        w_load = (self.vehicle_mass * 9.81) / self.wheel_count
        width = self.tire_width
        exponent = 2.0 / (2.0 * self.bekker_n + 1.0)
        numerator = 3.0 * w_load
        denominator = (
            width
            * (2.0 * self.bekker_n + 1.0)
            * (self.bekker_kc / width + self.bekker_kphi)
        )

        if denominator <= 0.0:
            self._log_error('Invalid Bekker parameters: denominator <= 0')
            return self.r_nominal, 0.0

        sinkage = (numerator / denominator) ** exponent
        sinkage = min(sinkage, self.r_nominal * 0.5)
        r_eff = self.r_nominal - sinkage / 2.0
        return r_eff, sinkage

    def process_imu(
        self,
        orientation_q: np.ndarray,
        angular_velocity: np.ndarray,
        linear_acceleration: np.ndarray,
        current_time_s: float,
    ):
        self.imu_orientation_q = np.asarray(orientation_q, dtype=np.float64)

        roll, pitch, _ = euler_from_quaternion(self.imu_orientation_q.tolist())
        self.ahrs_roll = roll
        self.ahrs_pitch = pitch

        self.imu_gyro_yaw_rate = float(angular_velocity[2]) * self.yaw_sign

        a_body = np.asarray(linear_acceleration, dtype=np.float64)
        g_world = np.array([0.0, 0.0, 9.81], dtype=np.float64)
        g_body = self._rotate_vector_by_quaternion_inverse(
            g_world,
            self.imu_orientation_q,
        )

        a_true = a_body - g_body
        self.imu_linear_accel_body = a_true

        if self.last_imu_time is not None:
            dt_imu = current_time_s - self.last_imu_time
            if 0.0 < dt_imu < 0.1:
                a_forward = a_true[0] - self.imu_accel_bias
                if self.is_zupt and abs(a_forward) < self.zupt_accel_threshold:
                    self.v_true_imu = 0.0
                    self.imu_accel_bias += self.imu_accel_bias_alpha * a_forward
                else:
                    self.v_true_imu += a_forward * dt_imu

                self.v_true_imu = float(
                    np.clip(
                        self.v_true_imu,
                        -self.max_imu_velocity,
                        self.max_imu_velocity,
                    )
                )

        self.last_imu_time = current_time_s
        self.imu_received = True

    @staticmethod
    def _rotate_vector_by_quaternion_inverse(
        vec: np.ndarray,
        quat: np.ndarray,
    ) -> np.ndarray:
        quat_conj = np.array([-quat[0], -quat[1], -quat[2], quat[3]])
        vec_quat = np.array([vec[0], vec[1], vec[2], 0.0])
        temp = TerramechanicOdometryCore._quat_multiply(quat_conj, vec_quat)
        result = TerramechanicOdometryCore._quat_multiply(temp, quat)
        return result[:3]

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )

    def _compute_slip_ratio(self, v_encoder: float, omega_avg: float) -> float:
        v_wheel = self.r_eff * abs(omega_avg)

        if not self.imu_received or self.is_zupt:
            return 0.0

        v_true = abs(self.v_true_imu)

        if v_wheel < 0.01 and v_true < 0.01:
            return 0.0

        if v_wheel >= v_true:
            if v_wheel < 0.01:
                return 0.0
            slip_ratio = 1.0 - v_true / v_wheel
        else:
            if v_true < 0.01:
                return 0.0
            slip_ratio = 1.0 - v_wheel / v_true

        return float(
            np.clip(slip_ratio, -self.max_slip_ratio, self.max_slip_ratio)
        )

    def _compute_understeer_kinematic_omega(
        self,
        left_steer: float,
        right_steer: float,
        velocity: float,
    ) -> float:
        delta = self._ackermann_steering(left_steer, right_steer)
        if abs(delta) < 0.001 or abs(velocity) < 0.01:
            return 0.0

        l_effective = self.wheelbase * (1.0 + self.K_us * velocity * velocity)
        return velocity * math.tan(delta) / l_effective

    def _gyro_kf_predict(self, dt: float):
        process_noise = np.diag(
            [self.gyro_kf_Q_omega * dt, self.gyro_kf_Q_bias * dt]
        )
        self.gyro_kf_P += process_noise

    def _gyro_kf_update_kinematic(self, omega_kinematic: float):
        h_mat = np.array([[1.0, 0.0]], dtype=np.float64)
        measurement = np.array([omega_kinematic], dtype=np.float64)
        r_mat = np.array([[self.gyro_kf_R_kin]], dtype=np.float64)

        innovation = measurement - h_mat @ self.gyro_kf_x
        innovation_cov = h_mat @ self.gyro_kf_P @ h_mat.T + r_mat
        kalman_gain = self.gyro_kf_P @ h_mat.T @ np.linalg.inv(innovation_cov)

        self.gyro_kf_x = self.gyro_kf_x + (kalman_gain @ innovation).flatten()
        self.gyro_kf_P = (np.eye(2) - kalman_gain @ h_mat) @ self.gyro_kf_P

    def _gyro_kf_update_imu(self, omega_imu: float):
        h_mat = np.array([[1.0, -1.0]], dtype=np.float64)
        measurement = np.array([omega_imu], dtype=np.float64)
        r_mat = np.array([[self.gyro_kf_R_imu]], dtype=np.float64)

        innovation = measurement - h_mat @ self.gyro_kf_x
        innovation_cov = h_mat @ self.gyro_kf_P @ h_mat.T + r_mat
        kalman_gain = self.gyro_kf_P @ h_mat.T @ np.linalg.inv(innovation_cov)

        self.gyro_kf_x = self.gyro_kf_x + (kalman_gain @ innovation).flatten()
        self.gyro_kf_P = (np.eye(2) - kalman_gain @ h_mat) @ self.gyro_kf_P

    def _fuse_angular_velocity(
        self,
        omega_kinematic: float,
        omega_imu: float,
        dt: float,
    ) -> float:
        if not self.gyro_kf_initialized:
            self.gyro_kf_x = np.array([omega_imu, 0.0], dtype=np.float64)
            self.gyro_kf_P = np.diag([0.1, 0.01]).astype(np.float64)
            self.gyro_kf_initialized = True
            return omega_imu

        self._gyro_kf_predict(dt)
        self._gyro_kf_update_kinematic(omega_kinematic)
        self._gyro_kf_update_imu(omega_imu)
        return float(self.gyro_kf_x[0])

    def _ackermann_steering(
        self,
        left_steer: float,
        right_steer: float,
    ) -> float:
        if abs(left_steer) < 0.001 and abs(right_steer) < 0.001:
            return 0.0

        if abs(left_steer) < 0.001:
            return right_steer * 0.8
        if abs(right_steer) < 0.001:
            return left_steer * 0.8

        try:
            left_radius = self.wheelbase / math.tan(abs(left_steer))
            right_radius = self.wheelbase / math.tan(abs(right_steer))

            turn_direction = 0.0
            if left_steer > 0.001 or right_steer > 0.001:
                turn_direction = 1.0
            elif left_steer < -0.001 or right_steer < -0.001:
                turn_direction = -1.0

            inner_radius = min(left_radius, right_radius)
            center_radius = inner_radius + self.track_width / 2.0
            if center_radius > 0.001:
                return math.atan(self.wheelbase / center_radius) * turn_direction
            return 0.0
        except (ZeroDivisionError, ValueError):
            return (left_steer + right_steer) / 2.0

    def _compute_dynamic_covariance(self) -> tuple[float, float, float, float, float]:
        slip_sq = self.current_slip_ratio ** 2
        slip_scale = 1.0 + self.k_slip * slip_sq

        pos_var = self.base_pos_var * slip_scale
        orient_var = self.base_orient_var * slip_scale
        vel_var = self.base_vel_var * slip_scale

        speed = abs(self.current_linear_velocity)
        pos_var += 0.02 * speed
        vel_var += 0.01 * speed

        steer = abs(self.current_steering_angle)
        pos_var += 0.05 * steer
        orient_var += 0.1 * steer

        tilt_angle = math.sqrt(self.ahrs_roll ** 2 + self.ahrs_pitch ** 2)
        tilt_factor = math.exp(self.tilt_cov_gain * tilt_angle)
        vel_lateral_var = (
            self.base_vel_var
            * tilt_factor
            * (1.0 + 10.0 * abs(self.ahrs_roll))
        )
        vel_vertical_var = (
            self.base_vel_var
            * tilt_factor
            * (1.0 + 10.0 * abs(self.ahrs_pitch))
        )

        vel_var *= 1.0 + 0.5 * tilt_angle

        if self.is_stalled:
            vel_var *= self.stall_cov_multiplier
            vel_lateral_var *= self.stall_cov_multiplier
            vel_vertical_var *= self.stall_cov_multiplier
            pos_var *= 10.0

        return pos_var, orient_var, vel_var, vel_lateral_var, vel_vertical_var

    def process_joint_state(
        self,
        joint_names: Sequence[str],
        positions: Sequence[float],
        current_time_s: float,
    ) -> bool:
        if not self.initialized:
            self.joint_names = list(joint_names)
            self._log_info(f'Available joints: {self.joint_names}')

            required = [
                self.left_wheel_joint,
                self.right_wheel_joint,
                self.left_steering_joint,
                self.right_steering_joint,
            ]
            missing = [name for name in required if name not in self.joint_names]
            if missing:
                self._log_warn(f'Missing joints: {missing}')
                return False

            try:
                left_index = self.joint_names.index(self.left_wheel_joint)
                right_index = self.joint_names.index(self.right_wheel_joint)
                self.last_wheel_pos['left'] = positions[left_index]
                self.last_wheel_pos['right'] = positions[right_index]
                self.last_time = current_time_s
                self.initialized = True
                self._log_info('Terramechanic odometry initialized')
            except (ValueError, IndexError) as exc:
                self._log_error(f'Init failed: {exc}')
                return False

        dt = current_time_s - self.last_time
        self.joint_names = list(joint_names)

        try:
            left_index = self.joint_names.index(self.left_wheel_joint)
            right_index = self.joint_names.index(self.right_wheel_joint)
            left_steer_index = self.joint_names.index(self.left_steering_joint)
            right_steer_index = self.joint_names.index(self.right_steering_joint)

            left_pos = positions[left_index]
            right_pos = positions[right_index]
            # Invert steering polarity: On this machine's URDF/Gazebo setup, positive 
            # joint states mean a RIGHT turn, but kinematics expect positive for LEFT.
            left_steer = -positions[left_steer_index]
            right_steer = -positions[right_steer_index]

            if dt > 0.001:
                delta_left = left_pos - self.last_wheel_pos['left']
                delta_right = right_pos - self.last_wheel_pos['right']
                
                # Wrap wheel position deltas to [-pi, pi] to prevent massive velocity 
                # spikes when Gazebo continuous joints wrap at pi or -pi.
                delta_left = math.atan2(math.sin(delta_left), math.cos(delta_left))
                delta_right = math.atan2(math.sin(delta_right), math.cos(delta_right))

                omega_left = delta_left / dt
                omega_right = delta_right / dt

                raw_left_vel = omega_left * self.r_eff
                raw_right_vel = omega_right * self.r_eff

                max_dv = self.max_wheel_accel * dt
                for side, raw_vel in [
                    ('left', raw_left_vel),
                    ('right', raw_right_vel),
                ]:
                    delta_vel = raw_vel - self.last_raw_wheel_vel[side]
                    if abs(delta_vel) > max_dv:
                        limited = self.last_raw_wheel_vel[side] + math.copysign(
                            max_dv,
                            delta_vel,
                        )
                        if side == 'left':
                            raw_left_vel = limited
                        else:
                            raw_right_vel = limited

                alpha = self.velocity_filter_alpha
                self.filtered_wheel_vel['left'] = (
                    (1 - alpha) * self.filtered_wheel_vel['left']
                    + alpha * raw_left_vel
                )
                self.filtered_wheel_vel['right'] = (
                    (1 - alpha) * self.filtered_wheel_vel['right']
                    + alpha * raw_right_vel
                )

                for side in ('left', 'right'):
                    if abs(self.filtered_wheel_vel[side]) < self.deadzone:
                        self.filtered_wheel_vel[side] = 0.0

                self.last_raw_wheel_vel['left'] = raw_left_vel
                self.last_raw_wheel_vel['right'] = raw_right_vel

                omega_avg = (abs(omega_left) + abs(omega_right)) / 2.0
                self.is_zupt = omega_avg < self.zupt_omega_threshold
                if self.is_zupt:
                    self.filtered_wheel_vel['left'] = 0.0
                    self.filtered_wheel_vel['right'] = 0.0
                    self.v_true_imu = 0.0

                v_encoder = (
                    self.filtered_wheel_vel['left']
                    + self.filtered_wheel_vel['right']
                ) / 2.0

                self.current_slip_ratio = self._compute_slip_ratio(
                    v_encoder,
                    omega_avg,
                )
                self.current_steering_angle = self._ackermann_steering(
                    left_steer,
                    right_steer,
                )

                if self.imu_received and not self.is_zupt:
                    if self.stall_detection_enabled:
                        encoder_moving = (
                            abs(v_encoder) > self.stall_encoder_vel_thresh
                        )
                        imu_stationary = (
                            abs(self.v_true_imu) < self.stall_imu_vel_thresh
                        )
                        high_slip = (
                            abs(self.current_slip_ratio)
                            > self.stall_slip_ratio_thresh
                        )

                        if encoder_moving and imu_stationary and high_slip:
                            self.stall_accumulator += dt
                        else:
                            self.stall_accumulator = max(
                                0.0,
                                self.stall_accumulator - 2.0 * dt,
                            )

                        self.is_stalled = (
                            self.stall_accumulator >= self.stall_duration_thresh
                        )
                    else:
                        self.is_stalled = False

                    # Decoupled raw_odom from IMU: We use pure encoder velocity.
                    # We DO NOT zero velocity based on stall detection because 
                    # high-slip sand driving triggers false positives.
                    self.current_linear_velocity = v_encoder
                else:
                    self.is_stalled = False
                    self.stall_accumulator = 0.0
                    self.current_linear_velocity = v_encoder

                self.omega_kinematic = self._compute_understeer_kinematic_omega(
                    left_steer,
                    right_steer,
                    self.current_linear_velocity,
                )

                # DYNAMIC OBSERVABILITY ARCHITECTURE:
                # We restore pure kinematic yaw rate (omega_kinematic) for the wheel odometry!
                # The C++ Factor Graph now dynamically tightens yaw_sigma when driving straight
                # to learn IMU gyro bias, and relaxes it during turns to let the IMU shine.
                if self.is_zupt:
                    self.current_angular_velocity = 0.0
                else:
                    self.current_angular_velocity = self.omega_kinematic

                self.last_wheel_pos['left'] = left_pos
                self.last_wheel_pos['right'] = right_pos
                self.last_time = current_time_s
                return True
        except (ValueError, IndexError) as exc:
            self._log_warn(f'Joint not found: {exc}')

        return False

    def build_odometry_output(
        self,
        current_time_s: float,
    ) -> Optional[TerramechanicOdometryOutput]:
        if not self.initialized:
            return None

        if self.last_odom_time is None:
            self.last_odom_time = current_time_s
            return None

        dt = current_time_s - self.last_odom_time
        if 0.001 < dt < 0.1:
            dx = self.current_linear_velocity * math.cos(self.theta) * dt
            dy = self.current_linear_velocity * math.sin(self.theta) * dt
            dtheta = self.current_angular_velocity * dt

            self.x += dx
            self.y += dy
            self.theta += dtheta
            self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        self.last_odom_time = current_time_s

        max_pub_speed = 4.5
        if math.isnan(self.current_linear_velocity):
            self._log_warn('NaN detected in linear velocity - zeroing')
            self.current_linear_velocity = 0.0
        elif abs(self.current_linear_velocity) > max_pub_speed:
            self._log_warn(
                f'Velocity {self.current_linear_velocity:.2f} exceeds '
                f'{max_pub_speed} m/s - clamping'
            )
            self.current_linear_velocity = math.copysign(
                max_pub_speed,
                self.current_linear_velocity,
            )

        if math.isnan(self.current_angular_velocity):
            self._log_warn('NaN detected in angular velocity - zeroing')
            self.current_angular_velocity = 0.0

        if math.isnan(self.x) or math.isnan(self.y) or math.isnan(self.theta):
            self._log_error('NaN detected in pose state - resetting to origin')
            self.x = 0.0
            self.y = 0.0
            self.theta = 0.0

        pos_var, orient_var, vel_var, vel_lat_var, vel_vert_var = (
            self._compute_dynamic_covariance()
        )

        lateral_velocity = 0.0
        if abs(self.ahrs_roll) > 0.02:
            lateral_velocity = -(
                9.81 * math.sin(self.ahrs_roll) * self.sand_slip_coeff
            )

        self.debug_counter += 1
        if self.debug_counter % 100 == 0:
            state = 'STALL' if self.is_stalled else ('ZUPT' if self.is_zupt else 'RUN')
            bias = self.gyro_kf_x[1] if self.gyro_kf_initialized else 0.0
            tilt_deg = math.degrees(
                math.sqrt(self.ahrs_roll ** 2 + self.ahrs_pitch ** 2)
            )
            self._log_info(
                f'TerraOdom: x={self.x:.2f} y={self.y:.2f} '
                f'theta={math.degrees(self.theta):.1f}deg '
                f'v={self.current_linear_velocity:.2f}m/s '
                f'omega_fused={math.degrees(self.current_angular_velocity):.1f}deg/s '
                f'omega_kin={math.degrees(self.omega_kinematic):.1f}deg/s '
                f'omega_imu={math.degrees(self.imu_gyro_yaw_rate):.1f}deg/s '
                f'gyro_bias={math.degrees(bias):.2f}deg/s '
                f'slip={self.current_slip_ratio:.3f} '
                f'r_eff={self.r_eff:.4f}m '
                f'tilt={tilt_deg:.1f}deg '
                f'sigma2_pos={pos_var:.4f} [{state}]'
            )

        return TerramechanicOdometryOutput(
            x=self.x,
            y=self.y,
            theta=self.theta,
            linear_velocity=self.current_linear_velocity,
            lateral_velocity=lateral_velocity,
            angular_velocity=self.current_angular_velocity,
            pos_var=pos_var,
            orient_var=orient_var,
            vel_var=vel_var,
            vel_lateral_var=vel_lat_var,
            vel_vertical_var=vel_vert_var,
        )

    def diagnostics(self) -> TerramechanicDiagnostics:
        return TerramechanicDiagnostics(
            slip_ratio=self.current_slip_ratio,
            sinkage=self.sinkage,
            r_eff=self.r_eff,
            zupt_active=self.is_zupt,
            stall_active=self.is_stalled,
            omega_fused=self.current_angular_velocity,
            omega_imu=self.imu_gyro_yaw_rate,
            omega_kinematic=self.omega_kinematic,
        )