#!/usr/bin/env python3
"""
sblp_metrics.py — Coverage & unpredictability metrics for SBLP evaluation.
==========================================================================
ROS-free utilities that quantify how well (and how unpredictably) a patrol
path covers its sector. Used to produce the paper's "SBLP Coverage and
Unpredictability Metrics" results by comparing the heavy-tailed Lévy walk
against a deterministic lawnmower baseline.

All functions operate on plain lists of (x, y) positions and use only the
standard library so they run in unit tests without heavy dependencies.
"""
from __future__ import annotations

import math


def visitation_grid(positions, bounds, resolution: float):
    """Bin a path's positions into a 2D visitation-count grid.

    Parameters
    ----------
    positions : list[(x, y)]
    bounds    : (min_x, min_y, max_x, max_y)
    resolution: cell size in metres.

    Returns
    -------
    (grid, nx, ny) where ``grid`` is a row-major list of ints of length nx*ny.
    """
    min_x, min_y, max_x, max_y = bounds
    nx = max(1, int(math.ceil((max_x - min_x) / resolution)))
    ny = max(1, int(math.ceil((max_y - min_y) / resolution)))
    grid = [0] * (nx * ny)
    for x, y in positions:
        if not (min_x <= x < max_x and min_y <= y < max_y):
            continue
        cx = int((x - min_x) / resolution)
        cy = int((y - min_y) / resolution)
        cx = min(cx, nx - 1)
        cy = min(cy, ny - 1)
        grid[cy * nx + cx] += 1
    return grid, nx, ny


def coverage_fraction(grid) -> float:
    """Fraction of grid cells visited at least once (in [0, 1])."""
    if not grid:
        return 0.0
    visited = sum(1 for c in grid if c > 0)
    return visited / len(grid)


def shannon_entropy(grid) -> float:
    """Shannon entropy (bits) of the normalized visitation distribution.

    Higher entropy = more uniform spatial spread. Zero when all visits fall in
    a single cell. This is our proxy for spatial *unpredictability*.
    """
    total = sum(grid)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in grid:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def normalized_entropy(grid) -> float:
    """Entropy normalized by log2(num_cells) → [0, 1]. 1 = perfectly uniform."""
    n = len(grid)
    if n <= 1:
        return 0.0
    max_h = math.log2(n)
    return shannon_entropy(grid) / max_h if max_h > 0 else 0.0


def turn_angle_entropy(headings, num_bins: int = 12) -> float:
    """Entropy (bits) of the distribution of heading changes along a path.

    A deterministic lawnmower produces only a few discrete turn angles (low
    entropy); a Lévy walk spreads turns broadly (high entropy). Quantifies
    *directional* unpredictability.
    """
    if len(headings) < 2:
        return 0.0
    deltas = []
    for i in range(1, len(headings)):
        d = headings[i] - headings[i - 1]
        d = math.atan2(math.sin(d), math.cos(d))  # wrap to [-pi, pi]
        deltas.append(d)
    bins = [0] * num_bins
    span = 2.0 * math.pi
    for d in deltas:
        idx = int((d + math.pi) / span * num_bins)
        idx = min(max(idx, 0), num_bins - 1)
        bins[idx] += 1
    return shannon_entropy(bins)


def lawnmower_path(bounds, spacing: float, step: float = 1.0):
    """Generate a deterministic boustrophedon (lawnmower) baseline path.

    Sweeps back and forth in x at ``spacing`` intervals in y. ``step`` is the
    inter-sample spacing along each sweep.
    """
    min_x, min_y, max_x, max_y = bounds
    path = []
    y = min_y
    direction = 1
    n_steps = max(1, int((max_x - min_x) / step))
    while y <= max_y:
        xs = [min_x + i * step for i in range(n_steps + 1)]
        if direction < 0:
            xs = list(reversed(xs))
        for x in xs:
            path.append((x, y))
        y += spacing
        direction *= -1
    return path
