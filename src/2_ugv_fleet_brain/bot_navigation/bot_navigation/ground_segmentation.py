#!/usr/bin/env python3
"""
ground_segmentation.py — Slope-robust ground/obstacle segmentation core.
========================================================================
ROS-free math for separating true obstacles (rocks, vegetation, walls) from
drivable sloped terrain in a 3D LiDAR cloud, so Nav2 can avoid obstacles
without treating dune slopes as walls.

Method (grid-based local-ground removal):
  * Work in a gravity-aligned frame (z = world up), so slopes are represented
    by a *smooth* rise in ground height, not by tilted points.
  * Bin points into small (x, y) cells. Within one small cell the ground's
    z-spread is only ~cell_size * tan(slope) — e.g. 0.15 m for a 0.4 m cell on
    a 20 deg slope — so the per-cell minimum z is a good local ground estimate.
  * A point is an OBSTACLE only if it rises MORE than ``height_threshold``
    above its cell's local ground. A slope stays on the ground surface (small
    intra-cell spread) and is NOT flagged; a rock/bush bumps above it and IS.

This is intentionally slope-agnostic: it never compares against a single global
plane or an absolute height, so a 30 deg dune reads as free space while a 0.5 m
rock on that same dune reads as an obstacle.
"""
from __future__ import annotations

import numpy as np


def segment_obstacles(points: np.ndarray,
                      cell_size: float = 0.4,
                      height_threshold: float = 0.4,
                      min_height: float = -1.0,
                      max_height: float = 2.0,
                      min_points_per_cell: int = 2) -> np.ndarray:
    """Return a boolean mask (len N) marking obstacle points.

    Parameters
    ----------
    points : (N, 3) float array of (x, y, z) in a gravity-aligned frame.
    cell_size : ground-estimation cell size (m). Smaller = more slope-robust.
    height_threshold : min rise above local ground to count as an obstacle (m).
    min_height, max_height : absolute z band (m) to pre-filter floor/sky noise.
    min_points_per_cell : cells with fewer points are treated as ground
        (too sparse to trust an obstacle classification).
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Absolute height band pre-filter (drops under-floor returns and high noise).
    band = (z >= min_height) & (z <= max_height)

    # Build a dense 2D grid of per-cell minimum z (the local ground candidate).
    ix = np.floor(x / cell_size).astype(np.int64)
    iy = np.floor(y / cell_size).astype(np.int64)
    ox = ix - ix.min()
    oy = iy - iy.min()
    w = int(ox.max()) + 1
    h = int(oy.max()) + 1

    INF = np.inf
    cell_min = np.full((w, h), INF, dtype=np.float64)
    np.minimum.at(cell_min, (ox, oy), z)
    cell_cnt = np.zeros((w, h), dtype=np.int64)
    np.add.at(cell_cnt, (ox, oy), 1)

    # Ground estimate = minimum z over the 3x3 neighborhood of each cell. This
    # lets an obstacle-only cell (whose ground return is occluded) borrow the
    # true ground level from its neighbors, while remaining slope-safe on
    # drivable inclines (the extra spread over one cell is small).
    ground_grid = cell_min.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.full_like(cell_min, INF)
            xs = slice(max(0, dx), w + min(0, dx))
            ys = slice(max(0, dy), h + min(0, dy))
            xt = slice(max(0, -dx), w + min(0, -dx))
            yt = slice(max(0, -dy), h + min(0, -dy))
            shifted[xt, yt] = cell_min[xs, ys]
            ground_grid = np.minimum(ground_grid, shifted)

    point_ground = ground_grid[ox, oy]
    point_cellcnt = cell_cnt[ox, oy]

    height_above_ground = z - point_ground
    obstacle = (
        band
        & (height_above_ground > height_threshold)
        & (point_cellcnt >= min_points_per_cell)
    )
    return obstacle
