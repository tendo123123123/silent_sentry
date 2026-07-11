#!/usr/bin/env python3
"""Unit tests for the slope-robust ground/obstacle segmentation core."""
import numpy as np

from bot_navigation.ground_segmentation import segment_obstacles


def _flat_ground(n=2000, extent=10.0, z=0.0):
    rng = np.random.default_rng(0)
    xy = rng.uniform(-extent, extent, size=(n, 2))
    zc = np.full((n, 1), z) + rng.normal(0, 0.01, size=(n, 1))  # lidar noise
    return np.hstack([xy, zc])


def test_flat_ground_has_no_obstacles():
    pts = _flat_ground()
    mask = segment_obstacles(pts)
    assert mask.sum() == 0


def test_drivable_slope_is_not_flagged_as_obstacle():
    """A drivable (22-degree) planar slope must read as free ground, not a wall."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 20, size=3000)
    y = rng.uniform(-5, 5, size=3000)
    z = x * np.tan(np.radians(22.0)) + rng.normal(0, 0.01, size=3000)  # smooth incline
    pts = np.column_stack([x, y, z])
    mask = segment_obstacles(pts, cell_size=0.4, height_threshold=0.4, max_height=1e6)
    # Almost nothing should be flagged (a handful of edge cells at most).
    assert mask.mean() < 0.02


def test_rock_on_flat_ground_is_flagged():
    ground = _flat_ground(n=2000)
    # A 0.6 m tall rock cluster at (2, 2).
    rock_xy = np.random.default_rng(2).uniform(-0.15, 0.15, size=(60, 2)) + np.array([2.0, 2.0])
    rock_z = np.full((60, 1), 0.6)
    rock = np.hstack([rock_xy, rock_z])
    pts = np.vstack([ground, rock])
    mask = segment_obstacles(pts, cell_size=0.4, height_threshold=0.4)
    # The rock points (last 60) should be flagged.
    assert mask[-60:].sum() >= 50
    # Flat ground should stay clear.
    assert mask[:2000].sum() == 0


def test_rock_on_slope_is_flagged_but_slope_is_not():
    """The decisive case: a rock ON a drivable 20-degree slope is still an obstacle."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 20, size=3000)
    y = rng.uniform(-5, 5, size=3000)
    z = x * np.tan(np.radians(20.0)) + rng.normal(0, 0.01, size=3000)
    slope = np.column_stack([x, y, z])
    # Rock: 0.7 m above the slope surface at x=10, y=0 (dense cluster).
    rx = np.full(80, 10.0) + rng.uniform(-0.1, 0.1, size=80)
    ry = np.full(80, 0.0) + rng.uniform(-0.1, 0.1, size=80)
    rz = 10.0 * np.tan(np.radians(20.0)) + 0.7
    rock = np.column_stack([rx, ry, np.full(80, rz)])
    pts = np.vstack([slope, rock])
    mask = segment_obstacles(pts, cell_size=0.4, height_threshold=0.4, max_height=1e6)
    assert mask[-80:].sum() >= 60      # rock detected
    assert mask[:3000].mean() < 0.02   # slope not flagged


def test_height_band_filters_extremes():
    pts = np.array([[0, 0, 5.0], [0.1, 0.1, 5.0], [1, 1, -3.0]])  # above/below band
    mask = segment_obstacles(pts, min_height=-1.0, max_height=2.0)
    assert mask.sum() == 0


def test_empty_cloud():
    assert segment_obstacles(np.zeros((0, 3))).shape == (0,)
