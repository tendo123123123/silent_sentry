"""Typed configuration and state containers for the local DEM pipeline."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

BinKey = Tuple[int, int]
ChunkPoints = Tuple[np.ndarray, np.ndarray]
SpatialBins = Dict[BinKey, Dict[int, ChunkPoints]]


@dataclass(frozen=True)
class LocalDemPipelineConfig:
    """Algorithmic configuration for the local DEM core."""

    resolution: float
    size_x: float
    size_y: float
    cloud_queue_size: int
    scan_period: float
    deskew_clockwise: bool
    rolling_submap_distance: float
    submap_spatial_bin_size: float
    uamc_drift_variance: float
    ground_h_min: float
    ground_h_max: float
    obstacle_h: float
    ransac_dist: float
    ransac_iters: int
    min_pts: int
    min_range: float
    max_range: float
    spawn_elevation: float

    @property
    def nx(self) -> int:
        """Return the DEM width in grid cells."""
        return int(self.size_x / self.resolution)

    @property
    def ny(self) -> int:
        """Return the DEM height in grid cells."""
        return int(self.size_y / self.resolution)


@dataclass
class QueuedCloud:
    """Queued LiDAR sweep awaiting transform and accumulation."""

    points: np.ndarray
    stamp_msg: Any
    source_frame: str


@dataclass
class LocalDemMotionState:
    """Motion state used by scan deskew and gravity alignment."""

    imu_pitch: float = 0.0
    imu_roll: float = 0.0
    body_linear_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    body_angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )


@dataclass
class LocalDemRollingState:
    """Rolling indexed submap state used to build local DEM windows."""

    cloud_queue_size: int
    pending_clouds: Deque[QueuedCloud] = field(init=False)
    submap_chunks: Deque[Tuple[float, int]] = field(default_factory=deque)
    submap_spatial_bins: SpatialBins = field(
        default_factory=lambda: defaultdict(dict)
    )
    chunk_bin_keys: Dict[int, Tuple[BinKey, ...]] = field(default_factory=dict)
    next_chunk_id: int = 0
    cumulative_travel: float = 0.0
    last_chunk_pose_xy: Optional[np.ndarray] = None
    latest_cloud_stamp: Any = None

    def __post_init__(self):
        """Initialize the bounded pending-cloud queue."""
        self.pending_clouds = deque(maxlen=max(self.cloud_queue_size, 1))


@dataclass
class LocalDemBuildOutput:
    """Rasterized DEM output returned by the core pipeline."""

    elevation_grid: np.ndarray
    origin_x: float
    origin_y: float
    processed_clouds: int
    candidate_chunk_count: int
    total_chunk_count: int
    submap_point_count: int
    latest_cloud_stamp: Any
