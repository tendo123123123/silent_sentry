"""Core GTSAM factor graph odometry fusion logic."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Optional, Sequence

import numpy as np

import gtsam
from gtsam import (
    BetweenFactorPose3,
    ISAM2,
    ISAM2Params,
    ImuFactor,
    NavState,
    NonlinearFactorGraph,
    Pose3,
    PriorFactorConstantBias,
    PriorFactorPose3,
    PriorFactorVector,
    Rot3,
    Values,
    noiseModel,
    symbol,
)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _safe(value: float, default: float = 0.0) -> float:
    return value if math.isfinite(value) else default


def _vec3(x_val: float, y_val: float, z_val: float) -> np.ndarray:
    return np.array([x_val, y_val, z_val], dtype=float)


def _as_vec3(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


def _rot3_from_xyzw(quat_xyzw):
    qx, qy, qz, qw = quat_xyzw
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9 or not math.isfinite(norm):
        return None
    return Rot3.Quaternion(qw / norm, qx / norm, qy / norm, qz / norm)


def _yaw_from_rot3(rot: Rot3) -> float:
    ypr = np.asarray(rot.ypr(), dtype=float).reshape(3)
    return float(ypr[0])


def _X(index: int):
    return symbol('x', index)


def _V(index: int):
    return symbol('v', index)


def _B(index: int):
    return symbol('b', index)


@dataclass(frozen=True)
class FactorGraphConfig:
    publish_rate: float
    odom_sig_xy: float
    imu_sig: float
    imu_rp_sig: float
    imu_accel_sig: float
    imu_gyro_sig: float
    imu_integration_sig: float
    odom_frame: str
    base_frame: str
    max_vel: float
    yaw_gate: float
    slip_accel_threshold: float
    slip_cov_multiplier: float
    pos_noise_pm: float
    heading_var: float
    kf_min_dist: float
    kf_min_angle: float


@dataclass(frozen=True)
class FactorGraphPublishOutput:
    x: float
    y: float
    theta: float
    vx: float
    omega: float
    pos_cov: float
    twist_covariance: Sequence[float]


class FactorGraphCore:
    """Own GTSAM state, preintegration, and estimator updates."""

    def __init__(
        self,
        config: FactorGraphConfig,
        logger: Optional[Any] = None,
    ):
        self.config = config
        self.logger = logger
        for name, value in vars(config).items():
            setattr(self, name, value)

        params = ISAM2Params()
        params.setRelinearizeThreshold(0.1)
        params.relinearizeSkip = 1
        self.isam = ISAM2(params)
        self.graph_inc = NonlinearFactorGraph()
        self.values_inc = Values()
        self.node_idx = 0
        self.bias_key = _B(0)
        self.lock = threading.Lock()

        self.imu_bias = gtsam.imuBias.ConstantBias()
        self.imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        eye3 = np.eye(3, dtype=float)
        self.imu_params.setAccelerometerCovariance(
            eye3 * (self.imu_accel_sig ** 2)
        )
        self.imu_params.setGyroscopeCovariance(
            eye3 * (self.imu_gyro_sig ** 2)
        )
        self.imu_params.setIntegrationCovariance(
            eye3 * (self.imu_integration_sig ** 2)
        )
        self.imu_params.setUse2ndOrderCoriolis(False)
        self.imu_params.setOmegaCoriolis(_vec3(0.0, 0.0, 0.0))
        self.pim = gtsam.PreintegratedImuMeasurements(
            self.imu_params,
            self.imu_bias,
        )

        self.initial_pose_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.02, 0.02, 0.05, 0.01, 0.01, 0.01], dtype=float)
        )
        self.initial_vel_noise = noiseModel.Isotropic.Sigma(3, 0.10)
        self.bias_prior_noise = noiseModel.Isotropic.Sigma(6, 0.10)
        self.imu_attitude_noise = noiseModel.Diagonal.Sigmas(
            np.array(
                [
                    self.imu_rp_sig,
                    self.imu_rp_sig,
                    1e3,
                    1e3,
                    1e3,
                    1e3,
                ],
                dtype=float,
            )
        )

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.raw_vx = 0.0
        self.omega = 0.0
        self.initialized = False
        self.last_odom_stamp = None
        self.last_imu_stamp = None

        self.lkg_x = 0.0
        self.lkg_y = 0.0
        self.lkg_theta = 0.0

        self.imu_roll = 0.0
        self.imu_yaw = None
        self.imu_rotation = None
        self._last_raw_imu_yaw = None
        self.pitch = 0.0
        self.true_accel_x = 0.0
        self._filtered_true_accel_x = None
        self._imu_accel_lpf_alpha = 0.2
        self.imu_yaw_rate = 0.0
        self.imu_received = False
        self._have_preintegrated_imu = False
        self._trn_quality = 0.0

        self.anchor_nav_state = NavState(Pose3(), _vec3(0.0, 0.0, 0.0))
        self.live_pose3 = Pose3()
        self.live_velocity = _vec3(0.0, 0.0, 0.0)

        self._kf_wheel_ds = 0.0
        self._kf_wheel_dtheta = 0.0
        self._kf_noise_scale = 1.0

        self.pos_cov = 0.01
        self.dist_traveled = 0.0

        self.last_twist_cov = [0.0] * 36

        self.is_slipping = False
        self.last_wheel_accel = 0.0
        self._prev_wheel_vx = None
        self._prev_wheel_stamp = None
        self._last_noise_scale = 1.0

        self.tick_count = 0

    def _log_info(self, message: str):
        if self.logger is not None:
            self.logger.info(message)

    def _log_warn(self, message: str):
        if self.logger is not None:
            self.logger.warn(message)

    def set_trn_quality(self, quality: float):
        self._trn_quality = float(quality)

    def _reset_preintegration(self):
        self.pim.resetIntegrationAndSetBias(self.imu_bias)
        self._have_preintegrated_imu = False

    def reset_to_identity(self):
        with self.lock:
            self.isam = ISAM2(self.isam.params())
            self.graph_inc.resize(0)
            self.values_inc.clear()
            self.node_idx = 0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.raw_vx = 0.0
        self.omega = 0.0
        self.initialized = False
        self.last_odom_stamp = None
        # IMU state intentionally preserved so graph re-initialises with current
        # roll/pitch even though planar yaw is now wheel-driven.
        self.lkg_x = 0.0
        self.lkg_y = 0.0
        self.lkg_theta = 0.0
        self.true_accel_x = 0.0
        self._filtered_true_accel_x = None
        self.imu_yaw_rate = 0.0
        self._have_preintegrated_imu = False
        self._trn_quality = 0.0
        self.anchor_nav_state = NavState(Pose3(), _vec3(0.0, 0.0, 0.0))
        self.live_pose3 = Pose3()
        self.live_velocity = _vec3(0.0, 0.0, 0.0)
        self._kf_wheel_ds = 0.0
        self._kf_wheel_dtheta = 0.0
        self._kf_noise_scale = 1.0
        self.pos_cov = 0.01
        self.dist_traveled = 0.0
        self.last_twist_cov = [0.0] * 36
        self.is_slipping = False
        self.last_wheel_accel = 0.0
        self._prev_wheel_vx = None
        self._prev_wheel_stamp = None
        self._last_noise_scale = 1.0
        self.tick_count = 0
        self._reset_preintegration()
        self._log_info('Factor graph reset to identity')

    @staticmethod
    def _wheel_delta_pose(ds: float, dtheta: float = 0.0) -> Pose3:
        return Pose3(Rot3.Yaw(dtheta), _vec3(ds, 0.0, 0.0))

    @staticmethod
    def _compose_planar_seed(
        anchor_pose: Pose3,
        delta_pose: Pose3,
        rot: Rot3,
    ) -> Pose3:
        seeded = anchor_pose.compose(delta_pose)
        return Pose3(rot, _as_vec3(seeded.translation()))

    @staticmethod
    def _rotation_delta_norm(a_rot: Rot3, b_rot: Rot3) -> float:
        delta = a_rot.between(b_rot)
        logmap = np.asarray(Rot3.Logmap(delta), dtype=float).reshape(3)
        return float(np.linalg.norm(logmap))

    def _fused_rotation(self, yaw: float) -> Rot3:
        if not self.imu_received:
            return Rot3.Yaw(yaw)

        roll = _safe(self.imu_roll)
        pitch = _safe(self.pitch)
        return Rot3.RzRyRx(roll, pitch, yaw)

    def _update_planar_projection(self, pose: Pose3):
        translation = _as_vec3(pose.translation())
        yaw = _yaw_from_rot3(pose.rotation())
        self.x = _safe(float(translation[0]), self.x)
        self.y = _safe(float(translation[1]), self.y)
        self.theta = _safe(yaw, self.theta)
        self.lkg_x = self.x
        self.lkg_y = self.y
        self.lkg_theta = self.theta

    def _live_state_prediction(self, vx: float):
        anchor_pose = self.anchor_nav_state.pose()
        wheel_delta = self._wheel_delta_pose(self._kf_wheel_ds, self._kf_wheel_dtheta)
        planar_seed = anchor_pose.compose(wheel_delta)
        planar_yaw = _yaw_from_rot3(planar_seed.rotation())
        if self._have_preintegrated_imu and self.pim.deltaTij() > 1e-4:
            predicted_nav = self.pim.predict(self.anchor_nav_state, self.imu_bias)
            vel = _as_vec3(predicted_nav.velocity())
        else:
            rot = self._fused_rotation(planar_yaw)
            vel = rot.matrix() @ _vec3(vx, 0.0, 0.0)

        rot = self._fused_rotation(planar_yaw)

        pose = self._compose_planar_seed(anchor_pose, wheel_delta, rot)
        return pose, vel

    def _wheel_noise_model(self, ds: float, noise_scale: float, yaw_sig: float = 1e3):
        forward_sig = self.odom_sig_xy * noise_scale * max(1.0, abs(ds) / 0.05)
        lateral_sig = max(0.25, 4.0 * forward_sig)
        vertical_sig = max(0.25, 4.0 * forward_sig)
        return noiseModel.Diagonal.Sigmas(
            np.array(
                [1e3, 1e3, yaw_sig, forward_sig, lateral_sig, vertical_sig],
                dtype=float,
            )
        )

    def _pose3_from_imu(self, translation: np.ndarray, yaw: float) -> Pose3:
        return Pose3(self._fused_rotation(yaw), _as_vec3(translation))

    def _update_covariance(self, vx: float, raw_vx: float, dt: float):
        ds = abs(vx) * dt
        cov_ds = abs(raw_vx) * dt if self.is_slipping else ds
        cov_gain = self.slip_cov_multiplier if self.is_slipping else 1.0
        # TRN quality feedback: high quality → tighter covariance
        trn_scale = max(0.2, 1.0 - self._trn_quality * 0.8)
        self.dist_traveled += ds
        self.pos_cov = min(
            self.pos_cov + cov_ds * self.pos_noise_pm * cov_gain * trn_scale,
            500.0,
        )

    def _log_debug(self, vx: float):
        self.tick_count += 1
        if self.tick_count % 100 == 0:
            imu_state = 'IMU' if self.imu_received else 'NO_IMU'
            self._log_info(
                f'FG: ({self.x:.2f},{self.y:.2f}) '
                f'th={math.degrees(self.theta):.1f} deg '
                f'v={vx:.2f} dist={self.dist_traveled:.1f}m '
                f'nodes={self.node_idx} [{imu_state}] '
                f'pim_dt={self.pim.deltaTij():.3f}s '
                f'slip={self.is_slipping} a_w={self.last_wheel_accel:.2f} '
                f'a_i={self.true_accel_x:.2f} '
                f'noise_x{self._last_noise_scale:.1f}'
            )

    def process_imu(
        self,
        quat_xyzw: Sequence[float],
        linear_accel: np.ndarray,
        angular_vel: np.ndarray,
        stamp_s: float,
    ):
        quat = list(quat_xyzw)
        if not all(math.isfinite(value) for value in quat):
            return
        if sum(value * value for value in quat) < 0.25:
            return

        qx, qy, qz, qw = quat
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        if not math.isfinite(yaw):
            return

        if self.imu_received and self._last_raw_imu_yaw is not None:
            if abs(_wrap(yaw - self._last_raw_imu_yaw)) > self.yaw_gate:
                return

        if not self.imu_received:
            self._log_info(f'IMU yaw initialised at {math.degrees(yaw):.1f} deg')

        self._last_raw_imu_yaw = yaw
        self.imu_roll = roll if math.isfinite(roll) else self.imu_roll
        self.imu_yaw = yaw
        self.pitch = pitch if math.isfinite(pitch) else self.pitch

        rot = _rot3_from_xyzw(quat)
        if rot is not None:
            self.imu_rotation = rot

        measured_acc = _vec3(
            _safe(float(linear_accel[0])),
            _safe(float(linear_accel[1])),
            _safe(float(linear_accel[2])),
        )
        measured_omega = _vec3(
            _safe(float(angular_vel[0])),
            _safe(float(angular_vel[1])),
            _safe(float(angular_vel[2])),
        )

        ax_raw = measured_acc[0]
        if math.isfinite(ax_raw):
            a_true = ax_raw - 9.81 * math.sin(self.pitch)
            if self._filtered_true_accel_x is None:
                self._filtered_true_accel_x = a_true
            else:
                alpha = self._imu_accel_lpf_alpha
                self._filtered_true_accel_x = (
                    alpha * a_true + (1.0 - alpha) * self._filtered_true_accel_x
                )
            self.true_accel_x = self._filtered_true_accel_x

        self.imu_yaw_rate = float(measured_omega[2])
        self.imu_received = True

        if not self.initialized:
            self.last_imu_stamp = stamp_s
            return

        if self.last_imu_stamp is not None:
            dt = stamp_s - self.last_imu_stamp
            if 5e-4 < dt < 0.1:
                self.pim.integrateMeasurement(measured_acc, measured_omega, dt)
                self._have_preintegrated_imu = True
            elif dt >= 0.1:
                self._reset_preintegration()

        self.last_imu_stamp = stamp_s

    def process_odom(
        self,
        linear_x: float,
        angular_z: float,
        twist_covariance: Sequence[float],
        stamp_s: float,
    ):
        raw_vx = _safe(float(linear_x))
        omega = _safe(float(angular_z))
        if abs(raw_vx) > self.max_vel:
            raw_vx = math.copysign(self.max_vel, raw_vx)

        wheel_accel = 0.0
        if self._prev_wheel_stamp is not None and self._prev_wheel_vx is not None:
            wheel_dt = stamp_s - self._prev_wheel_stamp
            if 0.001 < wheel_dt < 0.5:
                wheel_accel = (raw_vx - self._prev_wheel_vx) / wheel_dt
        self._prev_wheel_vx = raw_vx
        self._prev_wheel_stamp = stamp_s

        prev_slip = self.is_slipping
        self.raw_vx = raw_vx
        self.last_wheel_accel = wheel_accel
        if self.imu_received and self._filtered_true_accel_x is not None:
            accel_gap = abs(wheel_accel - self.true_accel_x)
            imu_quiet = abs(self.true_accel_x) < 0.5 * self.slip_accel_threshold
            wheel_spinup = wheel_accel > self.slip_accel_threshold
            self.is_slipping = (
                wheel_spinup and imu_quiet and accel_gap > self.slip_accel_threshold
            )
        else:
            self.is_slipping = False

        if self.is_slipping and not prev_slip:
            self._log_warn(
                'FG slip gate: freezing wheel odom '
                f'(a_wheel={wheel_accel:.2f}, a_imu={self.true_accel_x:.2f}, '
                f'pitch={math.degrees(self.pitch):.1f}deg)'
            )
        elif prev_slip and not self.is_slipping:
            self._log_info(
                'FG slip gate: restored wheel odom '
                f'(a_wheel={wheel_accel:.2f}, a_imu={self.true_accel_x:.2f})'
            )

        vx = 0.0 if self.is_slipping else raw_vx
        self.vx = vx
        self.omega = omega

        if len(twist_covariance) == 36:
            self.last_twist_cov = list(twist_covariance)

        if not self.initialized:
            rot0 = self._fused_rotation(0.0)
            vel0 = rot0.matrix() @ _vec3(vx, 0.0, 0.0)
            pose0 = Pose3(rot0, _vec3(0.0, 0.0, 0.0))
            with self.lock:
                self.graph_inc.add(
                    PriorFactorPose3(_X(0), pose0, self.initial_pose_noise)
                )
                self.graph_inc.add(
                    PriorFactorVector(_V(0), vel0, self.initial_vel_noise)
                )
                self.graph_inc.add(
                    PriorFactorConstantBias(
                        self.bias_key,
                        self.imu_bias,
                        self.bias_prior_noise,
                    )
                )
                self.values_inc.insert(_X(0), pose0)
                self.values_inc.insert(_V(0), vel0)
                self.values_inc.insert(self.bias_key, self.imu_bias)
                try:
                    self.isam.update(self.graph_inc, self.values_inc)
                except Exception as e:
                    self._log_warn(f'iSAM2 init update failed: {e}, retrying next tick')
                    self.graph_inc.resize(0)
                    self.values_inc.clear()
                    self.initialized = False
                    return
                self.graph_inc.resize(0)
                self.values_inc.clear()

            self.anchor_nav_state = NavState(pose0, vel0)
            self.live_pose3 = pose0
            self.live_velocity = _as_vec3(vel0)
            self._update_planar_projection(pose0)
            self.last_odom_stamp = stamp_s
            self.last_imu_stamp = None
            self._reset_preintegration()
            self.initialized = True
            self._log_info(
                f'iSAM2 initialised at theta={math.degrees(self.theta):.1f} deg'
            )
            return

        if self.last_odom_stamp is None:
            self.last_odom_stamp = stamp_s
            return

        dt = stamp_s - self.last_odom_stamp
        self.last_odom_stamp = stamp_s
        if dt <= 0.001 or dt > 0.5:
            return

        odom_vel_var = twist_covariance[0] if len(twist_covariance) > 0 else 0.01
        if math.isfinite(odom_vel_var) and odom_vel_var > 1e-6:
            noise_scale = max(1.0, 1.0 + odom_vel_var * 10.0)
        else:
            noise_scale = 1.0
        if self.is_slipping:
            noise_scale *= self.slip_cov_multiplier
        self._last_noise_scale = noise_scale
        self._kf_noise_scale = max(self._kf_noise_scale, noise_scale)

        self._kf_wheel_ds += vx * dt
        self._kf_wheel_dtheta += omega * dt
        live_pose, live_vel = self._live_state_prediction(vx)
        self.live_pose3 = live_pose
        self.live_velocity = _as_vec3(live_vel)
        self._update_planar_projection(live_pose)

        acc_rot = abs(self._kf_wheel_dtheta)

        _MAX_PIM_AGE = 2.0
        if (
            self._have_preintegrated_imu
            and self.pim.deltaTij() > _MAX_PIM_AGE
            and abs(self._kf_wheel_ds) < self.kf_min_dist
            and acc_rot < self.kf_min_angle
        ):
            self._log_warn(
                f'PIM stale ({self.pim.deltaTij():.1f}s > {_MAX_PIM_AGE}s) — '
                'resetting preintegration (robot likely idle)'
            )
            self._reset_preintegration()

        if abs(self._kf_wheel_ds) < self.kf_min_dist and acc_rot < self.kf_min_angle:
            self._update_covariance(vx, raw_vx, dt)
            self._log_debug(vx)
            return

        if not self._have_preintegrated_imu or self.pim.deltaTij() <= 1e-4:
            self._update_covariance(vx, raw_vx, dt)
            self._log_debug(vx)
            return

        if self.pim.deltaTij() > _MAX_PIM_AGE:
            self._log_warn(
                f'PIM stale ({self.pim.deltaTij():.1f}s > {_MAX_PIM_AGE}s) — '
                'resetting preintegration (robot likely idle)'
            )
            self._reset_preintegration()
            self._update_covariance(vx, raw_vx, dt)
            self._log_debug(vx)
            return

        kf_ds = self._kf_wheel_ds
        kf_dtheta = self._kf_wheel_dtheta
        kf_noise_scale = self._kf_noise_scale
        predicted_nav = self.pim.predict(self.anchor_nav_state, self.imu_bias)
        seed_planar_pose = self.anchor_nav_state.pose().compose(
            self._wheel_delta_pose(kf_ds, kf_dtheta)
        )
        seed_yaw = _yaw_from_rot3(seed_planar_pose.rotation())
        seed_rot = self._fused_rotation(seed_yaw)
        seed_pose = self._compose_planar_seed(
            self.anchor_nav_state.pose(),
            self._wheel_delta_pose(kf_ds, kf_dtheta),
            seed_rot,
        )
        seed_vel = _as_vec3(predicted_nav.velocity())
        new_idx = self.node_idx + 1

        with self.lock:
            self.graph_inc.add(
                ImuFactor(
                    _X(self.node_idx),
                    _V(self.node_idx),
                    _X(new_idx),
                    _V(new_idx),
                    self.bias_key,
                    self.pim,
                )
            )
            # Yaw noise: tight when wheel odometry reports a turn, loose otherwise
            yaw_sig = 0.03 if abs(kf_dtheta) > 0.01 else 1e3
            self.graph_inc.add(
                BetweenFactorPose3(
                    _X(self.node_idx),
                    _X(new_idx),
                    self._wheel_delta_pose(kf_ds, kf_dtheta),
                    self._wheel_noise_model(kf_ds, kf_noise_scale, yaw_sig),
                )
            )

            if self.imu_received and self.imu_rotation is not None:
                self.graph_inc.add(
                    PriorFactorPose3(
                        _X(new_idx),
                        self._pose3_from_imu(seed_pose.translation(), seed_yaw),
                        self.imu_attitude_noise,
                    )
                )

            self.values_inc.insert(_X(new_idx), seed_pose)
            self.values_inc.insert(_V(new_idx), seed_vel)

            try:
                self.isam.update(self.graph_inc, self.values_inc)
            except Exception as e:
                self._log_warn(f'iSAM2 update failed: {e}, resetting preintegration')
                self.graph_inc.resize(0)
                self.values_inc.clear()
                self._reset_preintegration()
                self._kf_wheel_ds = 0.0
                return
            self.graph_inc.resize(0)
            self.values_inc.clear()

            try:
                result = self.isam.calculateEstimate()
                opt_pose = result.atPose3(_X(new_idx))
                opt_vel = _as_vec3(result.atVector(_V(new_idx)))
                self.imu_bias = result.atConstantBias(self.bias_key)
            except Exception as e:
                self._log_warn(f'iSAM2 estimate failed: {e}, falling back to lkg')
                self._reset_preintegration()
                self._kf_wheel_ds = 0.0
                return

        self.node_idx = new_idx
        self.anchor_nav_state = NavState(opt_pose, opt_vel)
        self.live_pose3 = opt_pose
        self.live_velocity = opt_vel
        self._update_planar_projection(opt_pose)
        self._kf_wheel_ds = 0.0
        self._kf_wheel_dtheta = 0.0
        self._kf_noise_scale = 1.0
        self._reset_preintegration()

        self._update_covariance(vx, raw_vx, dt)
        self._log_debug(vx)

    def build_publish_output(self) -> Optional[FactorGraphPublishOutput]:
        with self.lock:
            if not self.initialized:
                return None

            x_pos = self.x if math.isfinite(self.x) else self.lkg_x
            y_pos = self.y if math.isfinite(self.y) else self.lkg_y
            theta = self.theta if math.isfinite(self.theta) else self.lkg_theta

            x_pos = x_pos if math.isfinite(x_pos) else 0.0
            y_pos = y_pos if math.isfinite(y_pos) else 0.0
            theta = theta if math.isfinite(theta) else 0.0

            return FactorGraphPublishOutput(
                x=x_pos,
                y=y_pos,
                theta=theta,
                vx=_safe(self.vx),
                omega=_safe(self.omega),
                pos_cov=max(self.pos_cov, 1e-4),
                twist_covariance=list(self.last_twist_cov),
            )