"""Core terrain-referenced navigation matching logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence
from collections import deque
import math
import time as pytime

import cv2
import numpy as np


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class TRNConfig:
    global_dem_path: str
    global_res: float
    local_nx: int
    local_ny: int
    local_res: float
    min_composite_cov: float
    bilateral_d: int
    bilateral_sc: float
    bilateral_ss: float
    base_search_radius: float
    max_search_radius: float
    covariance_scale: float
    initial_search_radius: float
    initial_match_count: int
    match_rate: float
    num_particles: int
    particle_spread_xy: float
    particle_spread_yaw: float
    ess_threshold: float
    min_update_ess_ratio: float
    flatness_std_threshold: float
    alpha_slow: float
    alpha_fast: float
    amcl_max_random_injection: float
    amcl_global_random_fraction: float
    amcl_prior_std_scale: float
    motion_noise_xy_frac: float
    motion_noise_yaw_frac: float
    roi_inject_fraction: float
    entropy_thresh: float
    min_peak_quality: float
    max_correction: float
    max_map_shift_per_cycle: float
    ema_alpha: float
    tf_publish_rate: float
    map_frame: str
    odom_frame: str
    base_link_frame: str
    dem_origin_x_str: str
    dem_origin_y_str: str


@dataclass(frozen=True)
class TRNMatchCycleResult:
    quality: Optional[float] = None
    entropy: Optional[float] = None
    search_radius: Optional[float] = None
    composite_width_m: Optional[float] = None
    composite_height_m: Optional[float] = None
    composite_coverage: Optional[float] = None
    odom_correction_x: Optional[float] = None
    odom_correction_y: Optional[float] = None


@dataclass(frozen=True)
class TRNMapToOdomOutput:
    x: float
    y: float
    yaw: float
    last_composite_stamp_ns: Optional[int]


class TRNCore:
    """Stateful TRN matcher independent of ROS I/O ownership."""

    def __init__(self, config: TRNConfig, logger: Optional[Any] = None):
        self.config = config
        self.logger = logger
        for name, value in vars(config).items():
            setattr(self, name, value)

        self.global_dem = None
        self.global_dem_origin_x = 0.0
        self.global_dem_origin_y = 0.0
        self._load_global_dem()

        self.latest_local_dem = None
        self.latest_local_origin_x = 0.0
        self.latest_local_origin_y = 0.0
        self.latest_local_center_x = 0.0
        self.latest_local_center_y = 0.0
        self.latest_local_res = self.local_res

        self.ekf_x = 0.0
        self.ekf_y = 0.0
        self.ekf_yaw = 0.0
        self.ekf_cov_x = 0.0
        self.ekf_cov_y = 0.0

        self.map_to_odom_x = 0.0
        self.map_to_odom_y = 0.0
        self.map_to_odom_yaw = 0.0

        self.successful_match_count = 0
        self.last_composite_stamp_ns = None

        self.particles = None
        self.particle_weights = None
        self.odom_snap = None
        self.w_fast = 0.0
        self.w_slow = 0.0

    def _log_info(self, message: str):
        if self.logger is not None:
            self.logger.info(message)

    def _log_warn(self, message: str):
        if self.logger is not None:
            self.logger.warn(message)

    def _log_debug(self, message: str):
        if self.logger is not None:
            self.logger.debug(message)

    def _log_error(self, message: str):
        if self.logger is not None:
            self.logger.error(message)

    def has_global_dem(self) -> bool:
        return self.global_dem is not None

    def has_local_dem(self) -> bool:
        return self.latest_local_dem is not None

    def prior_within_dem_bounds(self, prior_x: float, prior_y: float) -> bool:
        if self.global_dem is None:
            return False
        dem_h, dem_w = self.global_dem.shape
        x_min = self.global_dem_origin_x
        x_max = self.global_dem_origin_x + dem_w * self.global_res
        y_min = self.global_dem_origin_y
        y_max = self.global_dem_origin_y + dem_h * self.global_res
        return (
            x_min <= prior_x <= x_max
            and y_min <= prior_y <= y_max
        )

    def _load_global_dem(self):
        loaded = False
        has_real_georef = False

        try:
            import rasterio

            with rasterio.open(self.global_dem_path) as src:
                self.global_dem = src.read(1).astype(np.float32)
                transform = src.transform
                self.global_dem_origin_x = transform.c
                self.global_dem_origin_y = transform.f
                self.global_res = abs(transform.a)
                has_real_georef = (
                    abs(transform.c) > 0.1 or abs(transform.f) > 0.1
                )
            self.global_dem[self.global_dem < -999] = np.nan
            loaded = True
        except ImportError:
            self._log_warn('rasterio not available, trying OpenCV...')
            try:
                image = cv2.imread(self.global_dem_path, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    if len(image.shape) == 3:
                        image = image[:, :, 0]
                    self.global_dem = image.astype(np.float32)
                    loaded = True
                else:
                    self._log_error(f'Cannot open DEM: {self.global_dem_path}')
            except Exception as exc:
                self._log_error(f'Failed to load global DEM: {exc}')
        except Exception as exc:
            self._log_error(f'Failed to load global DEM: {exc}')

        if not loaded or self.global_dem is None:
            return

        dem_h, dem_w = self.global_dem.shape

        if self.dem_origin_x_str != 'auto':
            self.global_dem_origin_x = float(self.dem_origin_x_str)
        elif not has_real_georef:
            self.global_dem_origin_x = -(dem_w * self.global_res) / 2.0

        if self.dem_origin_y_str != 'auto':
            self.global_dem_origin_y = float(self.dem_origin_y_str)
        elif not has_real_georef:
            self.global_dem_origin_y = -(dem_h * self.global_res) / 2.0

        self._log_info(
            f'Loaded global DEM: {dem_w}x{dem_h} '
            f'@ {self.global_res}m/px from {self.global_dem_path}\n'
            f'  Origin: ({self.global_dem_origin_x:.1f}, '
            f'{self.global_dem_origin_y:.1f})m '
            f'[{"auto-centered" if not has_real_georef else "georeferenced"}]'
        )

    def update_local_dem(
        self,
        data: Sequence[float],
        ny: int,
        nx: int,
        stamp_ns: int,
        origin_x: Optional[float] = None,
        origin_y: Optional[float] = None,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        resolution: Optional[float] = None,
    ):
        try:
            grid = np.array(data, dtype=np.float32).reshape((ny, nx))
            grid[grid < -9990.0] = np.nan

            resolved_res = resolution if resolution is not None else self.local_res
            if resolved_res <= 0.0:
                resolved_res = self.local_res

            half_w = (nx * resolved_res) / 2.0
            half_h = (ny * resolved_res) / 2.0
            self.latest_local_dem = grid.copy()
            self.latest_local_res = resolved_res
            self.latest_local_center_x = (
                center_x if center_x is not None else self.ekf_x
            )
            self.latest_local_center_y = (
                center_y if center_y is not None else self.ekf_y
            )
            self.latest_local_origin_x = (
                origin_x
                if origin_x is not None
                else self.latest_local_center_x - half_w
            )
            self.latest_local_origin_y = (
                origin_y
                if origin_y is not None
                else self.latest_local_center_y - half_h
            )
            self.last_composite_stamp_ns = stamp_ns
        except Exception as exc:
            self._log_warn(f'Failed to parse local DEM: {exc}')

    def update_odom(
        self,
        x_pos: float,
        y_pos: float,
        yaw: float,
        cov_x: float,
        cov_y: float,
    ):
        self.ekf_x = x_pos
        self.ekf_y = y_pos
        self.ekf_yaw = yaw
        self.ekf_cov_x = max(cov_x, 0.0)
        self.ekf_cov_y = max(cov_y, 0.0)

    def fallback_map_prior(self) -> tuple[float, float]:
        cos_yaw = math.cos(self.map_to_odom_yaw)
        sin_yaw = math.sin(self.map_to_odom_yaw)
        map_x = self.map_to_odom_x + self.ekf_x * cos_yaw - self.ekf_y * sin_yaw
        map_y = self.map_to_odom_y + self.ekf_x * sin_yaw + self.ekf_y * cos_yaw
        return map_x, map_y

    def _get_latest_local_dem(self) -> tuple:
        if self.latest_local_dem is None:
            return (None, 0.0, 0.0, 0.0, 0.0, 0, 0, None, False)

        composite = self.latest_local_dem.copy()
        comp_ny, comp_nx = composite.shape
        coverage = float(np.sum(~np.isnan(composite)) / max(comp_nx * comp_ny, 1))
        if coverage < self.min_composite_cov:
            return (
                None,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                self.last_composite_stamp_ns,
                False,
            )

        return (
            composite,
            self.latest_local_origin_x,
            self.latest_local_origin_y,
            self.latest_local_center_x,
            self.latest_local_center_y,
            comp_nx,
            comp_ny,
            self.last_composite_stamp_ns,
            True,
        )

    def _compute_dynamic_radius(self) -> float:
        position_std = math.sqrt(self.ekf_cov_x + self.ekf_cov_y)
        dynamic_radius = self.base_search_radius + self.covariance_scale * position_std
        if self.successful_match_count < self.initial_match_count:
            dynamic_radius = max(dynamic_radius, self.initial_search_radius)
        return max(
            self.base_search_radius,
            min(dynamic_radius, self.max_search_radius),
        )

    def _extract_dynamic_roi(
        self,
        map_x: float,
        map_y: float,
        search_radius_m: float,
    ) -> tuple:
        dem_h, dem_w = self.global_dem.shape
        radius_px = int(math.ceil(search_radius_m / self.global_res))

        px_center = int(round((map_x - self.global_dem_origin_x) / self.global_res))
        py_center = int(round((map_y - self.global_dem_origin_y) / self.global_res))

        x_start = max(0, px_center - radius_px)
        x_end = min(dem_w, px_center + radius_px)
        y_start = max(0, py_center - radius_px)
        y_end = min(dem_h, py_center + radius_px)

        roi_w = x_end - x_start
        roi_h = y_end - y_start
        if roi_w < 10 or roi_h < 10:
            self._log_warn(f'TRN: ROI too small ({roi_w}x{roi_h} px)')
            return (None, 0.0, 0.0, False)

        roi_array = self.global_dem[y_start:y_end, x_start:x_end].copy()
        roi_origin_x = self.global_dem_origin_x + x_start * self.global_res
        roi_origin_y = self.global_dem_origin_y + y_start * self.global_res

        return (roi_array, roi_origin_x, roi_origin_y, True)

    def _bilateral_filter(self, img: np.ndarray) -> np.ndarray:
        result = img.copy().astype(np.float32)
        nan_mask = np.isnan(result)

        if np.any(nan_mask):
            valid_mean = np.nanmean(result)
            if np.isnan(valid_mean):
                valid_mean = 0.0
            result[nan_mask] = valid_mean

        filtered = cv2.bilateralFilter(
            result,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sc,
            sigmaSpace=self.bilateral_ss,
        )
        filtered[nan_mask] = np.nan
        return filtered

    def _resample_to_global_res(
        self,
        composite: np.ndarray,
        comp_res: float,
    ) -> np.ndarray:
        if abs(comp_res - self.global_res) < 0.01:
            return composite

        scale = comp_res / self.global_res
        out_h = max(int(composite.shape[0] * scale), 1)
        out_w = max(int(composite.shape[1] * scale), 1)

        nan_mask = np.isnan(composite)
        temp = composite.copy()
        fill_val = np.nanmean(temp) if not np.all(nan_mask) else 0.0
        temp[nan_mask] = fill_val

        resampled = cv2.resize(temp, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        nan_rs = cv2.resize(
            nan_mask.astype(np.float32),
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        )
        resampled[nan_rs > 0.5] = np.nan
        return resampled

    @staticmethod
    def _compute_spatial_entropy(dem: np.ndarray, n_bins: int = 32) -> float:
        valid = dem[~np.isnan(dem)]
        if len(valid) < 10:
            return 0.0

        counts, _ = np.histogram(valid, bins=n_bins)
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]
        elev_entropy = -np.sum(probabilities * np.log2(probabilities))

        dem_filled = dem.copy()
        nan_mask = np.isnan(dem_filled)
        if nan_mask.any():
            dem_filled[nan_mask] = np.nanmean(dem_filled)

        if dem_filled.shape[0] < 3 or dem_filled.shape[1] < 3:
            return float(elev_entropy)

        gy, gx = np.gradient(dem_filled)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_valid = grad_mag[~nan_mask] if nan_mask.any() else grad_mag.ravel()

        if len(grad_valid) < 10:
            return float(elev_entropy)

        grad_counts, _ = np.histogram(grad_valid, bins=n_bins)
        grad_probs = grad_counts / grad_counts.sum()
        grad_probs = grad_probs[grad_probs > 0]
        grad_entropy = -np.sum(grad_probs * np.log2(grad_probs))

        return float(max(elev_entropy, grad_entropy))

    def _mad_likelihood(
        self,
        composite: np.ndarray,
        global_roi: np.ndarray,
        particle_x: float,
        particle_y: float,
        particle_yaw: float,
        comp_origin_x: float,
        comp_origin_y: float,
        comp_center_x: float,
        comp_center_y: float,
        comp_res: float,
        roi_origin_x: float,
        roi_origin_y: float,
    ) -> float:
        comp_h, comp_w = composite.shape
        roi_h, roi_w = global_roi.shape

        a_vals = []
        b_vals = []

        cos_p = math.cos(particle_yaw)
        sin_p = math.sin(particle_yaw)

        step = max(1, min(comp_h, comp_w) // 30)

        for row in range(0, comp_h, step):
            for col in range(0, comp_w, step):
                elev_comp = composite[row, col]
                if np.isnan(elev_comp):
                    continue

                rel_x = (comp_origin_x + (col + 0.5) * comp_res) - comp_center_x
                rel_y = (comp_origin_y + (row + 0.5) * comp_res) - comp_center_y

                map_x = particle_x + cos_p * rel_x - sin_p * rel_y
                map_y = particle_y + sin_p * rel_x + cos_p * rel_y

                gx_idx = (map_x - roi_origin_x) / self.global_res
                gy_idx = (map_y - roi_origin_y) / self.global_res

                gxi = int(round(gx_idx))
                gyi = int(round(gy_idx))

                if 0 <= gxi < roi_w and 0 <= gyi < roi_h:
                    elev_glob = global_roi[gyi, gxi]
                    if not np.isnan(elev_glob):
                        a_vals.append(elev_comp)
                        b_vals.append(elev_glob)

        if len(a_vals) < 20:
            return -1.0

        a_arr = np.array(a_vals)
        b_arr = np.array(b_vals)

        if float(np.std(a_arr)) < self.flatness_std_threshold:
            return -1.0

        mad_error = float(np.mean(np.abs(a_arr - b_arr)))
        return float(math.exp(-mad_error / 0.25))

    @staticmethod
    def _systematic_resample(weights: np.ndarray) -> np.ndarray:
        count = len(weights)
        positions = (np.arange(count) + np.random.uniform()) / count
        cumulative = np.cumsum(weights)
        indices = np.searchsorted(cumulative, positions)
        return np.clip(indices, 0, count - 1)

    def _sampling_sigma(self, search_radius: float, scale: float = 1.0) -> float:
        base_sigma = (
            self.particle_spread_xy
            if self.particle_spread_xy > 1e-6
            else search_radius / 3.0
        )
        return max(base_sigma * scale, self.global_res)

    def _sample_recovery_particles(
        self,
        count: int,
        prior_x: float,
        prior_y: float,
        search_radius: float,
        roi_x_min: float,
        roi_x_max: float,
        roi_y_min: float,
        roi_y_max: float,
    ) -> np.ndarray:
        if count <= 0:
            return np.empty((0, 3), dtype=np.float64)

        particles = np.empty((count, 3), dtype=np.float64)
        n_global = int(round(count * self.amcl_global_random_fraction))
        n_global = min(max(n_global, 0), count)
        n_prior = count - n_global

        if n_global > 0:
            particles[:n_global, 0] = np.random.uniform(roi_x_min, roi_x_max, n_global)
            particles[:n_global, 1] = np.random.uniform(roi_y_min, roi_y_max, n_global)

        if n_prior > 0:
            sigma = max(search_radius * self.amcl_prior_std_scale, self.global_res)
            start = n_global
            particles[start:, 0] = np.clip(
                prior_x + np.random.randn(n_prior) * sigma,
                roi_x_min,
                roi_x_max,
            )
            particles[start:, 1] = np.clip(
                prior_y + np.random.randn(n_prior) * sigma,
                roi_y_min,
                roi_y_max,
            )

        particles[:, 2] = (
            self.map_to_odom_yaw
            + np.random.randn(count) * max(self.particle_spread_yaw, 1e-3)
        )
        return particles

    def _update_adaptive_recovery(self, average_weight: float) -> float:
        average_weight = max(float(average_weight), 1e-6)

        if self.w_slow <= 0.0:
            self.w_slow = average_weight
        else:
            self.w_slow += self.alpha_slow * (average_weight - self.w_slow)

        if self.w_fast <= 0.0:
            self.w_fast = average_weight
        else:
            self.w_fast += self.alpha_fast * (average_weight - self.w_fast)

        if self.w_slow <= 1e-9:
            return 0.0

        recovery_fraction = max(0.0, 1.0 - (self.w_fast / self.w_slow))
        return min(recovery_fraction, self.amcl_max_random_injection)

    def _prepare_map_correction(self, map_corr_x: float, map_corr_y: float):
        correction = np.array([map_corr_x, map_corr_y], dtype=np.float64)
        if not np.isfinite(correction).all():
            return None

        corr_mag = float(np.linalg.norm(correction))
        if corr_mag > self.max_correction:
            self._log_warn(
                f'TRN: rejecting correction {corr_mag:.2f}m > hard limit {self.max_correction:.2f}m'
            )
            return None

        proposed_step = self.ema_alpha * correction
        step_mag = float(np.linalg.norm(proposed_step))
        if step_mag > self.max_map_shift_per_cycle and step_mag > 1e-6:
            scale = self.max_map_shift_per_cycle / step_mag
            correction *= scale
            step_mag = self.max_map_shift_per_cycle
            self._log_warn(
                f'TRN: slew-limited map step to {self.max_map_shift_per_cycle:.2f}m/cycle'
            )

        return float(correction[0]), float(correction[1]), step_mag

    def _mcl_match(
        self,
        composite: np.ndarray,
        comp_origin_x: float,
        comp_origin_y: float,
        comp_center_x: float,
        comp_center_y: float,
        comp_res: float,
        global_roi: np.ndarray,
        roi_origin_x: float,
        roi_origin_y: float,
        prior_x: float,
        prior_y: float,
        search_radius: float,
    ) -> tuple:
        count = self.num_particles

        roi_w_m = global_roi.shape[1] * self.global_res
        roi_h_m = global_roi.shape[0] * self.global_res
        roi_x_min = roi_origin_x
        roi_x_max = roi_origin_x + roi_w_m
        roi_y_min = roi_origin_y
        roi_y_max = roi_origin_y + roi_h_m

        curr_odom_snap = (self.ekf_x, self.ekf_y, self.ekf_yaw)

        if self.particles is None:
            self.particles = np.empty((count, 3), dtype=np.float64)
            sigma_init = self.global_res
            self.particles[:, 0] = np.clip(
                prior_x + np.random.randn(count) * sigma_init,
                roi_x_min,
                roi_x_max,
            )
            self.particles[:, 1] = np.clip(
                prior_y + np.random.randn(count) * sigma_init,
                roi_y_min,
                roi_y_max,
            )
            self.particles[:, 2] = (
                self.map_to_odom_yaw
                + np.random.randn(count) * self.particle_spread_yaw
            )
            self.particle_weights = np.ones(count, dtype=np.float64) / count

            self._log_info(
                f'MCL known-pose cold-start: {count} particles tightly clustered around '
                f'prior=({prior_x:.2f},{prior_y:.2f}), sigma={sigma_init:.2f}m'
            )
        else:
            if self.odom_snap is not None:
                prev_ox, prev_oy, prev_oyaw = self.odom_snap

                dx_o = self.ekf_x - prev_ox
                dy_o = self.ekf_y - prev_oy
                dyaw_o = _wrap(self.ekf_yaw - prev_oyaw)

                c_m = math.cos(self.map_to_odom_yaw)
                s_m = math.sin(self.map_to_odom_yaw)
                dx_map = c_m * dx_o - s_m * dy_o
                dy_map = s_m * dx_o + c_m * dy_o

                disp = math.hypot(dx_map, dy_map)
                noise_xy = max(disp * self.motion_noise_xy_frac, 0.05)
                noise_yaw = max(abs(dyaw_o) * self.motion_noise_yaw_frac, 0.005)

                self.particles[:, 0] += dx_map + np.random.randn(count) * noise_xy
                self.particles[:, 1] += dy_map + np.random.randn(count) * noise_xy
                self.particles[:, 2] += dyaw_o + np.random.randn(count) * noise_yaw

            in_roi = (
                (self.particles[:, 0] >= roi_x_min)
                & (self.particles[:, 0] <= roi_x_max)
                & (self.particles[:, 1] >= roi_y_min)
                & (self.particles[:, 1] <= roi_y_max)
            )
            dead_idx = np.where(~in_roi)[0]

            n_explore = int(count * self.roi_inject_fraction)
            live_idx = np.where(in_roi)[0]
            if n_explore > len(dead_idx) and len(live_idx) > 0:
                n_extra = min(n_explore - len(dead_idx), len(live_idx))
                extra_replace = np.random.choice(live_idx, n_extra, replace=False)
                replace_idx = np.concatenate([dead_idx, extra_replace])
            else:
                replace_idx = dead_idx

            n_replace = len(replace_idx)
            if n_replace > 0:
                n_unif_inj = n_replace // 2
                n_gauss_inj = n_replace - n_unif_inj

                if n_unif_inj > 0:
                    self.particles[replace_idx[:n_unif_inj], 0] = np.random.uniform(
                        roi_x_min,
                        roi_x_max,
                        n_unif_inj,
                    )
                    self.particles[replace_idx[:n_unif_inj], 1] = np.random.uniform(
                        roi_y_min,
                        roi_y_max,
                        n_unif_inj,
                    )

                if n_gauss_inj > 0:
                    g_sigma = self._sampling_sigma(search_radius, scale=0.75)
                    gx = np.clip(
                        prior_x + np.random.randn(n_gauss_inj) * g_sigma,
                        roi_x_min,
                        roi_x_max,
                    )
                    gy = np.clip(
                        prior_y + np.random.randn(n_gauss_inj) * g_sigma,
                        roi_y_min,
                        roi_y_max,
                    )
                    self.particles[replace_idx[n_unif_inj:], 0] = gx
                    self.particles[replace_idx[n_unif_inj:], 1] = gy

                self.particles[replace_idx, 2] = (
                    self.map_to_odom_yaw
                    + np.random.randn(n_replace) * self.particle_spread_yaw
                )

        self.odom_snap = curr_odom_snap

        scores = np.full(count, -1.0)
        for index in range(count):
            scores[index] = self._mad_likelihood(
                composite,
                global_roi,
                self.particles[index, 0],
                self.particles[index, 1],
                self.particles[index, 2],
                comp_origin_x,
                comp_origin_y,
                comp_center_x,
                comp_center_y,
                comp_res,
                roi_origin_x,
                roi_origin_y,
            )

        valid_mask = scores > -0.99
        n_valid = int(np.sum(valid_mask))
        if n_valid < 10:
            self.particles = self._sample_recovery_particles(
                count,
                prior_x,
                prior_y,
                search_radius,
                roi_x_min,
                roi_x_max,
                roi_y_min,
                roi_y_max,
            )
            self.particle_weights = np.ones(count, dtype=np.float64) / count
            return (prior_x, prior_y, self.map_to_odom_yaw, 0.0, 0.0, 1.0)

        average_score = float(np.mean(scores[valid_mask]))
        recovery_fraction = self._update_adaptive_recovery(average_score)

        scores[~valid_mask] = 1e-6
        weights = scores.copy()
        weight_sum = float(np.sum(weights))
        if weight_sum < 1e-30:
            self.particles = self._sample_recovery_particles(
                count,
                prior_x,
                prior_y,
                search_radius,
                roi_x_min,
                roi_x_max,
                roi_y_min,
                roi_y_max,
            )
            self.particle_weights = np.ones(count, dtype=np.float64) / count
            return (prior_x, prior_y, self.map_to_odom_yaw, 0.0, 0.0, 1.0)

        weights /= weight_sum
        self.particle_weights = weights

        ess = 1.0 / float(np.sum(weights ** 2))
        ess_ratio = ess / count

        best_x = float(np.dot(weights, self.particles[:, 0]))
        best_y = float(np.dot(weights, self.particles[:, 1]))
        sin_yaw = float(np.dot(weights, np.sin(self.particles[:, 2])))
        cos_yaw = float(np.dot(weights, np.cos(self.particles[:, 2])))
        best_yaw = math.atan2(sin_yaw, cos_yaw)
        quality = float(np.max(scores[valid_mask]))

        should_resample = ess_ratio < self.ess_threshold or recovery_fraction > 0.01
        if should_resample:
            resample_idx = self._systematic_resample(weights)
            resampled_particles = self.particles[resample_idx].copy()

            n_recovery = int(round(count * recovery_fraction))
            n_recovery = min(max(n_recovery, 0), count)
            if n_recovery > 0:
                replace_idx = np.random.choice(count, n_recovery, replace=False)
                resampled_particles[replace_idx] = self._sample_recovery_particles(
                    n_recovery,
                    prior_x,
                    prior_y,
                    search_radius,
                    roi_x_min,
                    roi_x_max,
                    roi_y_min,
                    roi_y_max,
                )

            self.particles = resampled_particles
            self.particle_weights = np.ones(count, dtype=np.float64) / count

        return best_x, best_y, best_yaw, quality, ess_ratio, recovery_fraction

    def run_match_cycle(
        self,
        prior_x: float,
        prior_y: float,
        tf_success: bool,
        current_stamp_ns: int,
    ) -> Optional[TRNMatchCycleResult]:
        if self.global_dem is None or self.latest_local_dem is None:
            return None

        start_time = pytime.monotonic()

        result = self._get_latest_local_dem()
        (
            composite,
            comp_orig_x,
            comp_orig_y,
            comp_center_x,
            comp_center_y,
            comp_nx,
            comp_ny,
            composite_stamp_ns,
            success,
        ) = result

        if not success:
            self._log_debug('TRN: Rolling local DEM insufficient, skipping')
            return TRNMatchCycleResult(quality=0.0)

        self.last_composite_stamp_ns = (
            composite_stamp_ns if composite_stamp_ns is not None else current_stamp_ns
        )

        coverage = float(np.sum(~np.isnan(composite)) / max(comp_nx * comp_ny, 1))
        width_m = float(comp_nx * self.latest_local_res)
        height_m = float(comp_ny * self.latest_local_res)

        entropy = self._compute_spatial_entropy(composite)
        if entropy < self.entropy_thresh:
            self._log_info(
                f'TRN: Entropy={entropy:.2f} < {self.entropy_thresh} -- flat, skip'
            )
            return TRNMatchCycleResult(
                quality=0.0,
                entropy=entropy,
                composite_width_m=width_m,
                composite_height_m=height_m,
                composite_coverage=coverage,
            )

        composite_filtered = self._bilateral_filter(composite)
        composite_resampled = self._resample_to_global_res(
            composite_filtered,
            self.latest_local_res,
        )
        comp_res = (
            self.global_res
            if abs(self.latest_local_res - self.global_res) >= 0.01
            else self.latest_local_res
        )

        comp_std = float(np.nanstd(composite_resampled))
        if comp_std < self.flatness_std_threshold:
            self._log_info(
                f'TRN: Composite too flat (std={comp_std:.3f}m < '
                f'{self.flatness_std_threshold:.3f}m) -- skip'
            )
            return TRNMatchCycleResult(
                quality=0.0,
                entropy=entropy,
                composite_width_m=width_m,
                composite_height_m=height_m,
                composite_coverage=coverage,
            )

        if not self.prior_within_dem_bounds(prior_x, prior_y):
            self._log_warn(
                f'TRN: prior ({prior_x:.1f},{prior_y:.1f}) outside DEM bounds — '
                f'skipping match cycle'
            )
            return TRNMatchCycleResult(quality=0.0)

        search_radius = self._compute_dynamic_radius()
        roi_result = self._extract_dynamic_roi(prior_x, prior_y, search_radius)
        roi_array, roi_origin_x, roi_origin_y, roi_success = roi_result
        if not roi_success:
            return TRNMatchCycleResult(
                quality=0.0,
                entropy=entropy,
                search_radius=search_radius,
                composite_width_m=width_m,
                composite_height_m=height_m,
                composite_coverage=coverage,
            )

        global_roi_filtered = self._bilateral_filter(roi_array)

        (
            best_x,
            best_y,
            best_yaw,
            quality,
            ess_ratio,
            recovery_fraction,
        ) = self._mcl_match(
            composite_resampled,
            comp_orig_x,
            comp_orig_y,
            comp_center_x,
            comp_center_y,
            comp_res,
            global_roi_filtered,
            roi_origin_x,
            roi_origin_y,
            prior_x,
            prior_y,
            search_radius,
        )

        if quality < self.min_peak_quality:
            self._log_info(
                f'TRN: MCL quality {quality:.4f} < {self.min_peak_quality} -- skip'
            )
            return TRNMatchCycleResult(
                quality=float(quality),
                entropy=entropy,
                search_radius=search_radius,
                composite_width_m=width_m,
                composite_height_m=height_m,
                composite_coverage=coverage,
            )

        if ess_ratio < self.min_update_ess_ratio:
            self._log_info(
                f'TRN: ESS ratio {ess_ratio:.3f} < {self.min_update_ess_ratio} -- skip'
            )
            return TRNMatchCycleResult(
                quality=float(quality),
                entropy=entropy,
                search_radius=search_radius,
                composite_width_m=width_m,
                composite_height_m=height_m,
                composite_coverage=coverage,
            )

        map_corr_x = best_x - prior_x
        map_corr_y = best_y - prior_y

        correction_result = self._prepare_map_correction(map_corr_x, map_corr_y)
        if correction_result is None:
            return TRNMatchCycleResult(
                quality=float(quality),
                entropy=entropy,
                search_radius=search_radius,
                composite_width_m=width_m,
                composite_height_m=height_m,
                composite_coverage=coverage,
            )

        map_corr_x, map_corr_y, applied_step_mag = correction_result

        self.map_to_odom_x += self.ema_alpha * map_corr_x
        self.map_to_odom_y += self.ema_alpha * map_corr_y
        yaw_correction = math.atan2(
            math.sin(best_yaw - self.map_to_odom_yaw),
            math.cos(best_yaw - self.map_to_odom_yaw),
        )
        self.map_to_odom_yaw += self.ema_alpha * 0.1 * yaw_correction

        self.successful_match_count += 1

        c_inv = math.cos(-self.map_to_odom_yaw)
        s_inv = math.sin(-self.map_to_odom_yaw)
        odom_corr_x = c_inv * map_corr_x - s_inv * map_corr_y
        odom_corr_y = s_inv * map_corr_x + c_inv * map_corr_y

        elapsed_ms = (pytime.monotonic() - start_time) * 1000.0
        self._log_info(
            f'TRN MCL #{self.successful_match_count}: '
            f'comp={comp_nx}x{comp_ny} ({width_m:.0f}x{height_m:.0f}m), '
            f'prior=({prior_x:.1f},{prior_y:.1f}) '
            f'{"[TF2]" if tf_success else "[fallback]"}, '
            f'best=({best_x:.1f},{best_y:.1f}), '
            f'map_corr=({map_corr_x:.3f},{map_corr_y:.3f})m, '
            f'odom_corr=({odom_corr_x:.3f},{odom_corr_y:.3f})m, '
            f'mad_like={quality:.4f}, ess={ess_ratio:.3f}, '
            f'recovery={recovery_fraction:.2f}, step={applied_step_mag:.2f}m, '
            f'map->odom=({self.map_to_odom_x:.3f},{self.map_to_odom_y:.3f}), '
            f'{elapsed_ms:.0f}ms'
        )

        return TRNMatchCycleResult(
            quality=float(quality),
            entropy=entropy,
            search_radius=search_radius,
            composite_width_m=width_m,
            composite_height_m=height_m,
            composite_coverage=coverage,
            odom_correction_x=odom_corr_x,
            odom_correction_y=odom_corr_y,
        )

    def map_to_odom_output(self) -> TRNMapToOdomOutput:
        return TRNMapToOdomOutput(
            x=self.map_to_odom_x,
            y=self.map_to_odom_y,
            yaw=self.map_to_odom_yaw,
            last_composite_stamp_ns=self.last_composite_stamp_ns,
        )