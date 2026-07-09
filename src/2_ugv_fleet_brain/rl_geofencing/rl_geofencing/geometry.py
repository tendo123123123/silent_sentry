#!/usr/bin/env python3
"""
geometry.py — Polygon and terrain-sampling utilities for elastic geo-fencing.
=============================================================================
Pure, ROS-free helpers used by both the RL environment (training) and the
Base Station node (inference). Kept dependency-light (numpy only) so the same
math runs identically offline and online.
"""
from __future__ import annotations

import numpy as np

Polygon = np.ndarray  # shape (M, 2), float, ordered vertices in meters


def polygon_area(poly: Polygon) -> float:
    """Signed area via the shoelace formula (absolute value returned)."""
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(poly: Polygon) -> np.ndarray:
    """Area-weighted centroid of a simple polygon. Falls back to vertex mean."""
    if len(poly) < 3:
        return poly.mean(axis=0) if len(poly) else np.zeros(2)
    x = poly[:, 0]
    y = poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() * 0.5
    if abs(a) < 1e-9:
        return poly.mean(axis=0)
    cx = np.dot(x + np.roll(x, -1), cross) / (6.0 * a)
    cy = np.dot(y + np.roll(y, -1), cross) / (6.0 * a)
    return np.array([cx, cy])


def point_in_polygon(pt: np.ndarray, poly: Polygon) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    if n < 3:
        return False
    x, y = float(pt[0]), float(pt[1])
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def sample_points_in_polygon(poly: Polygon, spacing: float) -> np.ndarray:
    """Regular grid of interior sample points (used for coverage/terrain scoring)."""
    if len(poly) < 3:
        return np.empty((0, 2))
    min_xy = poly.min(axis=0)
    max_xy = poly.max(axis=0)
    xs = np.arange(min_xy[0], max_xy[0] + spacing, spacing)
    ys = np.arange(min_xy[1], max_xy[1] + spacing, spacing)
    pts = []
    for x in xs:
        for y in ys:
            p = np.array([x, y])
            if point_in_polygon(p, poly):
                pts.append(p)
    return np.array(pts) if pts else np.empty((0, 2))


def rasterize_coverage(polys: list[Polygon], bounds: tuple, res: float) -> np.ndarray:
    """
    Rasterize the union of polygons into a boolean coverage grid over `bounds`
    (min_x, min_y, max_x, max_y). Returns a 2D bool array; True where covered.
    """
    min_x, min_y, max_x, max_y = bounds
    nx = max(1, int(np.ceil((max_x - min_x) / res)))
    ny = max(1, int(np.ceil((max_y - min_y) / res)))
    grid = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        cy = min_y + (j + 0.5) * res
        for i in range(nx):
            cx = min_x + (i + 0.5) * res
            p = np.array([cx, cy])
            for poly in polys:
                if point_in_polygon(p, poly):
                    grid[j, i] = True
                    break
    return grid


def scale_polygon_about(poly: Polygon, center: np.ndarray,
                        sx: float, sy: float) -> Polygon:
    """Affine-scale a polygon about `center` (used to 'stretch' a sector)."""
    return (poly - center) * np.array([sx, sy]) + center


def translate_polygon(poly: Polygon, delta: np.ndarray) -> Polygon:
    return poly + delta
