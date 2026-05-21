"""Pure local DEM accumulation and rasterization logic."""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .local_dem_types import (
    LocalDemBuildOutput,
    LocalDemMotionState,
    LocalDemPipelineConfig,
    LocalDemRollingState,
    QueuedCloud,
)

TransformPointsFn = Callable[..., object]


class LocalDemPipelineCore:
    """ROS-free rolling DEM pipeline core."""

    def __init__(self, config: LocalDemPipelineConfig):
        """Store pipeline configuration and initialize mutable state."""
        self.config = config
        self.motion_state = LocalDemMotionState()
        self.submap_state = LocalDemRollingState(config.cloud_queue_size)

    @property
    def latest_cloud_stamp(self):
        """Return the latest queued cloud stamp."""
        return self.submap_state.latest_cloud_stamp

    def update_imu_orientation(self, roll: float, pitch: float):
        """Update the body orientation used for gravity alignment."""
        if math.isfinite(roll) and math.isfinite(pitch):
            self.motion_state.imu_pitch = pitch
            self.motion_state.imu_roll = roll

    def update_body_angular_velocity(self, angular_velocity: np.ndarray):
        """Update body-frame angular velocity used for scan deskew."""
        if np.isfinite(angular_velocity).all():
            self.motion_state.body_angular_velocity = angular_velocity.astype(
                np.float32,
                copy=False,
            )

    def update_body_linear_velocity(self, linear_velocity: np.ndarray):
        """Update body-frame linear velocity used for scan deskew."""
        if np.isfinite(linear_velocity).all():
            self.motion_state.body_linear_velocity = linear_velocity.astype(
                np.float32,
                copy=False,
            )

    def enqueue_cloud(
        self,
        points: np.ndarray,
        stamp_msg,
        source_frame: str,
    ):
        """Queue a parsed cloud for later accumulation."""
        if points is None or len(points) == 0:
            return

        queued_cloud = QueuedCloud(
            points=points,
            stamp_msg=stamp_msg,
            source_frame=source_frame,
        )
        self.submap_state.pending_clouds.append(queued_cloud)
        self.submap_state.latest_cloud_stamp = stamp_msg

    @staticmethod
    def _box_inside_mask(
        points: np.ndarray,
        center: np.ndarray,
        half_extents: np.ndarray,
    ) -> np.ndarray:
        deltas = np.abs(points - center)
        return np.all(deltas <= half_extents, axis=1)

    @staticmethod
    def _cylinder_inside_mask(
        points: np.ndarray,
        center: np.ndarray,
        radius: float,
        length: float,
    ) -> np.ndarray:
        radial_sq = (
            (points[:, 0] - center[0]) ** 2
            + (points[:, 1] - center[1]) ** 2
        )
        axial = np.abs(points[:, 2] - center[2])
        return (radial_sq <= radius ** 2) & (axial <= length * 0.5)

    def _filter_robot_self_hits(
        self,
        points: np.ndarray,
        return_mask: bool = False,
    ):
        if points is None or len(points) == 0:
            if return_mask:
                return points, np.zeros(0, dtype=bool)
            return points

        mask = np.ones(len(points), dtype=bool)

        chassis_center = np.array([0.0, 0.0, 0.175], dtype=np.float32)
        chassis_half = np.array([0.685, 0.185, 0.2075], dtype=np.float32)
        mask &= ~self._box_inside_mask(points, chassis_center, chassis_half)

        deck_center = np.array([0.0, 0.0, 0.4075], dtype=np.float32)
        deck_half = np.array([0.685, 0.40, 0.025], dtype=np.float32)
        mask &= ~self._box_inside_mask(points, deck_center, deck_half)

        mast_center = np.array([0.60, 0.0, 0.5955], dtype=np.float32)
        mask &= ~self._cylinder_inside_mask(
            points,
            mast_center,
            0.020,
            0.426,
        )

        flange_half = np.array([0.040, 0.040, 0.004], dtype=np.float32)
        bottom_flange_center = np.array([0.60, 0.0, 0.3865], dtype=np.float32)
        top_flange_center = np.array([0.60, 0.0, 0.8045], dtype=np.float32)
        mask &= ~self._box_inside_mask(
            points,
            bottom_flange_center,
            flange_half,
        )
        mask &= ~self._box_inside_mask(points, top_flange_center, flange_half)

        lidar_base_center = np.array([0.60, 0.0, 0.8245], dtype=np.float32)
        mask &= ~self._cylinder_inside_mask(
            points,
            lidar_base_center,
            0.0516,
            0.032,
        )

        if return_mask:
            return points[mask], mask
        return points[mask]

    @staticmethod
    def _skew_matrix(vec: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.0, -vec[2], vec[1]],
                [vec[2], 0.0, -vec[0]],
                [-vec[1], vec[0], 0.0],
            ],
            dtype=np.float64,
        )

    def _compute_relative_scan_time(
        self,
        points: np.ndarray,
        sensor_origin_base: np.ndarray,
    ) -> np.ndarray:
        if (
            points is None
            or len(points) == 0
            or self.config.scan_period <= 1e-4
        ):
            length = 0 if points is None else len(points)
            return np.zeros(length, dtype=np.float32)

        rel_points = (
            points.astype(np.float64) - sensor_origin_base.astype(np.float64)
        )
        azimuth = np.mod(
            np.arctan2(rel_points[:, 1], rel_points[:, 0]),
            2.0 * math.pi,
        )
        start_azimuth = float(azimuth[0]) if len(azimuth) > 0 else 0.0

        if self.config.deskew_clockwise:
            delta_azimuth = np.mod(start_azimuth - azimuth, 2.0 * math.pi)
        else:
            delta_azimuth = np.mod(azimuth - start_azimuth, 2.0 * math.pi)

        rel_time = np.clip(
            (delta_azimuth / (2.0 * math.pi)) * self.config.scan_period,
            0.0,
            self.config.scan_period,
        )
        return rel_time.astype(np.float32)

    def _uamc_weights(self, rel_time: np.ndarray) -> np.ndarray:
        if rel_time is None or len(rel_time) == 0:
            return np.zeros(0, dtype=np.float32)

        rel_time64 = rel_time.astype(np.float64)
        weights = np.exp(
            -0.5 * (rel_time64 ** 2) / self.config.uamc_drift_variance
        )
        weights = np.clip(weights, 1e-3, 1.0)
        return weights.astype(np.float32)

    def _deskew_points(
        self,
        points: np.ndarray,
        sensor_origin_base: np.ndarray,
    ):
        if points is None or len(points) == 0:
            return points, np.zeros(0, dtype=np.float32)

        rel_time = self._compute_relative_scan_time(points, sensor_origin_base)
        if self.config.scan_period <= 1e-4:
            return points, rel_time

        linear_vel = self.motion_state.body_linear_velocity.astype(
            np.float64
        )
        angular_vel = self.motion_state.body_angular_velocity.astype(
            np.float64
        )
        if (
            np.linalg.norm(linear_vel) < 1e-4
            and np.linalg.norm(angular_vel) < 1e-4
        ):
            return points, rel_time

        points64 = points.astype(np.float64)
        omega_norm = float(np.linalg.norm(angular_vel))
        if omega_norm < 1e-6:
            deskewed = points64 + rel_time[:, None] * linear_vel[None, :]
            valid = np.isfinite(deskewed).all(axis=1)
            return deskewed[valid].astype(np.float32), rel_time[valid]

        omega = self._skew_matrix(angular_vel)
        omega_sq = omega @ omega
        theta = omega_norm * rel_time

        rot_sin = (np.sin(theta) / omega_norm)[:, None]
        rot_cos = (
            (1.0 - np.cos(theta)) / (omega_norm * omega_norm)
        )[:, None]

        rotated = points64.copy()
        rotated += rot_sin * (points64 @ omega.T)
        rotated += rot_cos * (points64 @ omega_sq.T)

        v_omega = linear_vel @ omega.T
        v_omega_sq = linear_vel @ omega_sq.T
        trans_a = (
            (1.0 - np.cos(theta)) / (omega_norm * omega_norm)
        )[:, None]
        trans_b = ((theta - np.sin(theta)) / (omega_norm ** 3))[:, None]
        translation = rel_time[:, None] * linear_vel[None, :]
        translation += trans_a * v_omega[None, :]
        translation += trans_b * v_omega_sq[None, :]

        deskewed = rotated + translation
        valid = np.isfinite(deskewed).all(axis=1)
        return deskewed[valid].astype(np.float32), rel_time[valid]

    def _gravity_align_points(self, points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return points

        pitch = self.motion_state.imu_pitch
        roll = self.motion_state.imu_roll
        if abs(pitch) <= 0.005 and abs(roll) <= 0.005:
            return points

        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cr = math.cos(roll)
        sr = math.sin(roll)
        rotation = np.array(
            [
                [cp, sp * sr, -sp * cr],
                [0.0, cr, sr],
                [sp, -cp * sr, cp * cr],
            ],
            dtype=np.float32,
        )
        aligned = (rotation @ points.T).T
        valid = np.isfinite(aligned).all(axis=1)
        return aligned[valid]

    def _segment_ground(
        self,
        points: np.ndarray,
        return_mask: bool = False,
    ):
        if len(points) < 10:
            if return_mask:
                return points, np.ones(len(points), dtype=bool)
            return points

        z = points[:, 2]
        height_mask = (
            (z >= self.config.ground_h_min)
            & (z <= self.config.ground_h_max)
        )
        filtered = points[height_mask]
        if len(filtered) < 10:
            if return_mask:
                return filtered, height_mask
            return filtered

        best_inliers = None
        best_count = 0
        for _ in range(self.config.ransac_iters):
            idx = np.random.choice(len(filtered), 3, replace=False)
            p1 = filtered[idx[0]]
            p2 = filtered[idx[1]]
            p3 = filtered[idx[2]]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-6:
                continue
            normal = normal / norm_len

            if abs(normal[2]) < 0.7:
                continue

            d = np.dot(filtered - p1, normal)
            inliers = np.abs(d) < self.config.ransac_dist
            count = int(np.sum(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is not None and best_count > 10:
            final_filtered_mask = best_inliers
        else:
            final_filtered_mask = np.ones(len(filtered), dtype=bool)

        if return_mask:
            final_mask = np.zeros(len(points), dtype=bool)
            height_idx = np.flatnonzero(height_mask)
            final_mask[height_idx[final_filtered_mask]] = True
            return filtered[final_filtered_mask], final_mask
        return filtered[final_filtered_mask]

    def _prune_submap_chunks(self, current_travel: float):
        cutoff = current_travel - self.config.rolling_submap_distance
        while (
            self.submap_state.submap_chunks
            and self.submap_state.submap_chunks[0][0] < cutoff
        ):
            _, chunk_id = self.submap_state.submap_chunks.popleft()
            for bin_key in self.submap_state.chunk_bin_keys.pop(
                chunk_id,
                (),
            ):
                bin_entries = self.submap_state.submap_spatial_bins.get(
                    bin_key
                )
                if bin_entries is None:
                    continue
                bin_entries.pop(chunk_id, None)
                if not bin_entries:
                    self.submap_state.submap_spatial_bins.pop(bin_key, None)

    def _bin_keys_for_bounds(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ):
        bin_size = self.config.submap_spatial_bin_size
        ix_min = int(math.floor(min_x / bin_size))
        ix_max = int(math.floor(max_x / bin_size))
        iy_min = int(math.floor(min_y / bin_size))
        iy_max = int(math.floor(max_y / bin_size))
        return [
            (ix, iy)
            for ix in range(ix_min, ix_max + 1)
            for iy in range(iy_min, iy_max + 1)
        ]

    def _index_submap_chunk(
        self,
        travel: float,
        points_odom: np.ndarray,
        point_weights: np.ndarray,
    ):
        if points_odom is None or len(points_odom) == 0:
            return

        chunk_id = self.submap_state.next_chunk_id
        self.submap_state.next_chunk_id += 1

        points32 = points_odom.astype(np.float32, copy=False)
        weights32 = point_weights.astype(np.float32, copy=False)
        bin_size = self.config.submap_spatial_bin_size
        bx = np.floor(points32[:, 0].astype(np.float64) / bin_size).astype(
            np.int32
        )
        by = np.floor(points32[:, 1].astype(np.float64) / bin_size).astype(
            np.int32
        )
        bin_coords = np.column_stack((bx, by))
        unique_bins, inverse = np.unique(
            bin_coords,
            axis=0,
            return_inverse=True,
        )

        bin_keys = []
        for idx, coords in enumerate(unique_bins):
            mask = inverse == idx
            bin_key = (int(coords[0]), int(coords[1]))
            self.submap_state.submap_spatial_bins[bin_key][chunk_id] = (
                points32[mask].copy(),
                weights32[mask].copy(),
            )
            bin_keys.append(bin_key)

        self.submap_state.chunk_bin_keys[chunk_id] = tuple(bin_keys)
        self.submap_state.submap_chunks.append((travel, chunk_id))

    def _accumulate_ground_cloud(
        self,
        points: np.ndarray,
        stamp_msg,
        source_frame: str,
        transform_points: TransformPointsFn,
        base_frame: str,
        odom_frame: str,
    ) -> bool:
        points_base, sensor_origin = transform_points(
            points,
            source_frame,
            base_frame,
            stamp_msg,
            return_translation=True,
        )
        if points_base is None or len(points_base) < 20:
            return False

        points_base, rel_time = self._deskew_points(
            points_base,
            sensor_origin,
        )
        if points_base is None or len(points_base) < 20:
            return False

        points_base, self_mask = self._filter_robot_self_hits(
            points_base,
            return_mask=True,
        )
        rel_time = rel_time[self_mask]
        if len(points_base) < 20:
            return False

        ranges = np.sqrt(points_base[:, 0] ** 2 + points_base[:, 1] ** 2)
        range_mask = (
            (ranges >= self.config.min_range)
            & (ranges <= self.config.max_range)
        )
        points_base = points_base[range_mask]
        rel_time = rel_time[range_mask]
        if len(points_base) < 20:
            return False

        ground, ground_mask = self._segment_ground(
            points_base,
            return_mask=True,
        )
        rel_time = rel_time[ground_mask]
        if len(ground) < 10:
            return False

        ground = self._gravity_align_points(ground)
        if len(ground) < 10:
            return False

        if len(ground) != len(rel_time):
            valid_ground = min(len(ground), len(rel_time))
            ground = ground[:valid_ground]
            rel_time = rel_time[:valid_ground]

        ground_odom, robot_pose = transform_points(
            ground,
            base_frame,
            odom_frame,
            stamp_msg,
            return_translation=True,
        )
        if ground_odom is None or robot_pose is None or len(ground_odom) < 10:
            return False

        if abs(self.config.spawn_elevation) > 0.01:
            ground_odom[:, 2] += self.config.spawn_elevation

        ground_weights = self._uamc_weights(rel_time)
        if len(ground_weights) != len(ground_odom):
            valid_count = min(len(ground_weights), len(ground_odom))
            ground_odom = ground_odom[:valid_count]
            ground_weights = ground_weights[:valid_count]
        if len(ground_odom) < 10:
            return False

        robot_pose_xy = robot_pose[:2].astype(np.float32)
        if self.submap_state.last_chunk_pose_xy is not None:
            step = float(
                np.hypot(
                    robot_pose_xy[0] - self.submap_state.last_chunk_pose_xy[0],
                    robot_pose_xy[1] - self.submap_state.last_chunk_pose_xy[1],
                )
            )
            if math.isfinite(step):
                self.submap_state.cumulative_travel += step

        self.submap_state.last_chunk_pose_xy = robot_pose_xy
        self._index_submap_chunk(
            self.submap_state.cumulative_travel,
            ground_odom,
            ground_weights,
        )
        self._prune_submap_chunks(self.submap_state.cumulative_travel)
        return True

    def _drain_pending_clouds(
        self,
        transform_points: TransformPointsFn,
        base_frame: str,
        odom_frame: str,
    ) -> int:
        processed = 0
        while self.submap_state.pending_clouds:
            queued_cloud = self.submap_state.pending_clouds.popleft()
            if self._accumulate_ground_cloud(
                queued_cloud.points,
                queued_cloud.stamp_msg,
                queued_cloud.source_frame,
                transform_points,
                base_frame,
                odom_frame,
            ):
                processed += 1
        return processed

    def build_dem(
        self,
        robot_pose: np.ndarray,
        transform_points: TransformPointsFn,
        base_frame: str,
        odom_frame: str,
    ) -> Optional[LocalDemBuildOutput]:
        """Drain queued clouds and rasterize the current local DEM window."""
        processed_clouds = self._drain_pending_clouds(
            transform_points,
            base_frame,
            odom_frame,
        )
        if not self.submap_state.submap_chunks:
            return None

        travel_ref = self.submap_state.cumulative_travel
        if self.submap_state.last_chunk_pose_xy is not None:
            extra = float(
                np.hypot(
                    robot_pose[0] - self.submap_state.last_chunk_pose_xy[0],
                    robot_pose[1] - self.submap_state.last_chunk_pose_xy[1],
                )
            )
            if math.isfinite(extra):
                travel_ref += extra
        self._prune_submap_chunks(travel_ref)
        if not self.submap_state.submap_chunks:
            return None

        half_x = self.config.size_x / 2.0
        half_y = self.config.size_y / 2.0
        origin_x = float(robot_pose[0] - half_x)
        origin_y = float(robot_pose[1] - half_y)
        max_x = origin_x + self.config.size_x
        max_y = origin_y + self.config.size_y

        candidate_clouds = []
        candidate_weights = []
        candidate_chunk_ids = set()
        roi_bin_keys = self._bin_keys_for_bounds(
            origin_x,
            max_x,
            origin_y,
            max_y,
        )
        for bin_key in roi_bin_keys:
            bin_entries = self.submap_state.submap_spatial_bins.get(bin_key)
            if not bin_entries:
                continue
            for chunk_id, (points_bin, weights_bin) in bin_entries.items():
                candidate_clouds.append(points_bin)
                candidate_weights.append(weights_bin)
                candidate_chunk_ids.add(chunk_id)

        if not candidate_clouds:
            return None

        submap_points = np.concatenate(candidate_clouds, axis=0)
        submap_weights = np.concatenate(candidate_weights, axis=0)
        valid = (
            np.isfinite(submap_points).all(axis=1)
            & np.isfinite(submap_weights)
            & (submap_weights > 0.0)
            & (submap_points[:, 0] >= origin_x)
            & (submap_points[:, 0] < max_x)
            & (submap_points[:, 1] >= origin_y)
            & (submap_points[:, 1] < max_y)
        )
        submap_points = submap_points[valid]
        submap_weights = submap_weights[valid]
        if len(submap_points) < 10:
            return None

        gx = (
            (submap_points[:, 0] - origin_x) / self.config.resolution
        ).astype(int)
        gy = (
            (submap_points[:, 1] - origin_y) / self.config.resolution
        ).astype(int)
        grid_valid = (
            (gx >= 0)
            & (gx < self.config.nx)
            & (gy >= 0)
            & (gy < self.config.ny)
        )
        gx = gx[grid_valid]
        gy = gy[grid_valid]
        gz = submap_points[:, 2][grid_valid]
        gw = submap_weights[grid_valid]
        if len(gz) < 10:
            return None

        elevation_grid = np.full(
            (self.config.ny, self.config.nx),
            np.nan,
            dtype=np.float32,
        )
        count_grid = np.zeros((self.config.ny, self.config.nx), dtype=np.int32)
        flat_idx = gy * self.config.nx + gx
        np.add.at(count_grid.ravel(), flat_idx, 1)

        weight_grid = np.zeros(
            (self.config.ny, self.config.nx),
            dtype=np.float64,
        )
        np.add.at(weight_grid.ravel(), flat_idx, gw.astype(np.float64))

        weighted_sum_grid = np.zeros(
            (self.config.ny, self.config.nx),
            dtype=np.float64,
        )
        np.add.at(
            weighted_sum_grid.ravel(),
            flat_idx,
            gz.astype(np.float64) * gw.astype(np.float64),
        )

        valid_cells = (
            (count_grid >= self.config.min_pts)
            & (weight_grid > 1e-9)
        )
        elevation_grid[valid_cells] = (
            weighted_sum_grid[valid_cells] / weight_grid[valid_cells]
        ).astype(np.float32)

        elevation_grid = self._morph_close(elevation_grid)
        elevation_grid = self._reject_obstacle_cells(elevation_grid)

        return LocalDemBuildOutput(
            elevation_grid=elevation_grid,
            origin_x=origin_x,
            origin_y=origin_y,
            processed_clouds=processed_clouds,
            candidate_chunk_count=len(candidate_chunk_ids),
            total_chunk_count=len(self.submap_state.submap_chunks),
            submap_point_count=len(submap_points),
            latest_cloud_stamp=self.submap_state.latest_cloud_stamp,
        )

    @staticmethod
    def _morph_close(grid: np.ndarray, iters: int = 3) -> np.ndarray:
        result = grid.copy()
        ny, nx = grid.shape

        for _ in range(iters):
            nan_mask = np.isnan(result)
            new_result = result.copy()
            for iy in range(1, ny - 1):
                for ix in range(1, nx - 1):
                    if nan_mask[iy, ix]:
                        neighbors = []
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dy == 0 and dx == 0:
                                    continue
                                val = result[iy + dy, ix + dx]
                                if not np.isnan(val):
                                    neighbors.append(val)
                        if len(neighbors) >= 1:
                            new_result[iy, ix] = np.mean(neighbors)
            result = new_result

        return result

    @staticmethod
    def _reject_obstacle_cells(grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        ny, nx = grid.shape
        threshold = 0.4

        for iy in range(1, ny - 1):
            for ix in range(1, nx - 1):
                val = grid[iy, ix]
                if np.isnan(val):
                    continue

                neighbors = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        neighbor = grid[iy + dy, ix + dx]
                        if not np.isnan(neighbor):
                            neighbors.append(neighbor)

                if len(neighbors) >= 2:
                    local_mean = np.mean(neighbors)
                    if val - local_mean > threshold:
                        result[iy, ix] = np.nan

        return result
