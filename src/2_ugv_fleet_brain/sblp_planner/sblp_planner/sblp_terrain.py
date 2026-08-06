#!/usr/bin/env python3
"""
sblp_terrain.py — Terrain traversability sampler for SBLP waypoint gating.
==========================================================================
Loads the a-priori traversability costmap (the same one used by the Nav2
global costmap / GIS pipeline) and exposes ``cost_at(x, y) -> [0, 1]`` where
0 = fully traversable and 1 = lethal. SBLPCore uses this to reject Lévy
waypoints that land on impassable slopes, so the patrol only *chooses* goals
on drivable terrain; Nav2 still plans the obstacle-free *path* to each goal.

Costmap convention (matches bot_navigation/maps/continuous_costmap):
    raw pixel 255 = traversable, 0 = lethal  ->  we invert & normalize to
    cost 0 (traversable) .. 1 (lethal).

Loading is dependency-tolerant: it tries GDAL, then Pillow, then a NumPy
``.npy``; if none succeed it falls back to an all-traversable field so patrol
never hard-fails on a missing map.
"""
from __future__ import annotations

import math

import numpy as np


class TerrainCostmap:
    """Samples the a-priori traversability raster.

    ROW-ORDER WARNING (unresolved repo-wide inconsistency)
    ------------------------------------------------------
    Two subsystems disagree about the vertical orientation of the terrain data:

      * ``synthetic_dem.bin`` is indexed by TRN (``ugv_trn``) and the obstacle
        detector (``ugv_obstacle``) as ``row = (y - origin_y) / res``, i.e.
        image row 0 is treated as MINIMUM y.
      * ``nav2_map_server`` loads ``continuous_planner_map.pgm`` and, per the
        ROS map convention, treats image row 0 as MAXIMUM y -- it flips
        internally.

    Both files are written by ``tif_to_bin.py`` / ``gdal_translate`` from the
    same GeoTIFF WITHOUT a flip, and they measurably share row order
    (slope correlation 0.93 as-is vs 0.02 flipped). Therefore the Nav2 global
    static layer is vertically MIRRORED relative to the DEM that TRN matches
    against. One of the two is wrong; which one requires empirical
    confirmation against Gazebo ground truth (compare ``/ground_truth/pose``
    z with the DEM lookup under each convention).

    ``flip_y`` defaults to True so that SBLP agrees with **map_server**, since
    SBLP's job is to pick goals the Nav2 planner can actually path to. Set it
    False to match the TRN/DEM convention instead.
    """

    def __init__(self, grid: np.ndarray, origin_x: float, origin_y: float,
                 resolution: float, raw_is_traversable_high: bool = True,
                 flip_y: bool = True):
        # Normalize to cost in [0,1], 1 = lethal.
        g = np.asarray(grid, dtype=np.float32)
        gmin, gmax = float(g.min()), float(g.max())
        if gmax > gmin:
            g = (g - gmin) / (gmax - gmin)
        else:
            g = np.zeros_like(g)
        self.cost = (1.0 - g) if raw_is_traversable_high else g
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.res = float(resolution)
        self.flip_y = bool(flip_y)
        self.h, self.w = self.cost.shape

    def cost_at(self, x: float, y: float) -> float:
        """Nearest-cell terrain cost at world (x, y). Out-of-bounds = lethal.

        See the class docstring for the ``flip_y`` row-order caveat.
        """
        col = int((x - self.origin_x) / self.res)
        iy = int((y - self.origin_y) / self.res)
        row = (self.h - 1 - iy) if self.flip_y else iy
        if col < 0 or row < 0 or col >= self.w or row >= self.h:
            return 1.0
        return float(self.cost[row, col])


def _load_raster(path: str) -> np.ndarray | None:
    # Try GDAL first (handles GeoTIFF + geotransform elsewhere).
    try:
        from osgeo import gdal  # type: ignore
        ds = gdal.Open(path)
        if ds is not None:
            return np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.float32)
    except Exception:
        pass
    # Then Pillow (PNG/PGM/TIFF).
    try:
        from PIL import Image  # type: ignore
        return np.array(Image.open(path), dtype=np.float32)
    except Exception:
        pass
    # Then a raw .npy.
    try:
        return np.load(path).astype(np.float32)
    except Exception:
        return None


def load_terrain_costmap(path: str, origin_x: float = 0.0, origin_y: float = 0.0,
                         resolution: float = 1.0,
                         raw_is_traversable_high: bool = True,
                         flip_y: bool = True) -> TerrainCostmap:
    """Load a costmap raster, or return an all-traversable fallback (1x1)."""
    grid = _load_raster(path)
    if grid is None:
        grid = np.full((1, 1), 255.0, dtype=np.float32)  # all traversable
    if grid.ndim == 3:  # RGB(A) -> single band
        grid = grid[:, :, 0]
    return TerrainCostmap(grid, origin_x, origin_y, resolution,
                          raw_is_traversable_high, flip_y)
