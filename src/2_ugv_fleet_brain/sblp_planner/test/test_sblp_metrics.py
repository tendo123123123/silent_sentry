#!/usr/bin/env python3
"""Unit tests for SBLP coverage / unpredictability metrics."""
import math
import random

import pytest

from sblp_planner.sblp_core import SBLPConfig, SBLPCore
from sblp_planner.sblp_metrics import (
    coverage_fraction,
    lawnmower_path,
    normalized_entropy,
    shannon_entropy,
    turn_angle_entropy,
    visitation_grid,
)

BOUNDS = (0.0, 0.0, 20.0, 20.0)


def test_visitation_grid_counts():
    grid, nx, ny = visitation_grid([(1.0, 1.0), (1.0, 1.0), (19.0, 19.0)],
                                   BOUNDS, resolution=5.0)
    assert nx == 4 and ny == 4
    assert sum(grid) == 3
    assert grid[0] == 2  # both (1,1) samples in cell (0,0)


def test_visitation_grid_ignores_out_of_bounds():
    grid, _, _ = visitation_grid([(100.0, 100.0), (-5.0, 0.0)], BOUNDS, 5.0)
    assert sum(grid) == 0


def test_coverage_fraction():
    grid = [0, 1, 2, 0]
    assert coverage_fraction(grid) == pytest.approx(0.5)
    assert coverage_fraction([]) == 0.0


def test_shannon_entropy_single_cell_is_zero():
    assert shannon_entropy([10, 0, 0, 0]) == pytest.approx(0.0)


def test_shannon_entropy_uniform_is_log2n():
    grid = [4, 4, 4, 4]
    assert shannon_entropy(grid) == pytest.approx(math.log2(4))


def test_normalized_entropy_range():
    assert normalized_entropy([1, 1, 1, 1]) == pytest.approx(1.0)
    assert normalized_entropy([9, 0, 0, 0]) == pytest.approx(0.0)


def test_turn_angle_entropy_constant_heading_is_zero():
    headings = [0.0, 0.0, 0.0, 0.0]
    assert turn_angle_entropy(headings) == pytest.approx(0.0)


def test_lawnmower_covers_bounds():
    path = lawnmower_path(BOUNDS, spacing=5.0, step=2.0)
    grid, _, _ = visitation_grid(path, BOUNDS, resolution=5.0)
    # A dense lawnmower should visit most cells.
    assert coverage_fraction(grid) > 0.7


def test_levy_walk_more_directionally_unpredictable_than_lawnmower():
    """Core claim: the Lévy walk has higher turn-angle entropy than a lawnmower."""
    # Lévy heading sequence from the core.
    core = SBLPCore(SBLPConfig(
        geofence_polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]),
        rng=random.Random(11))
    x, y, yaw = 50.0, 50.0, 0.0
    levy_headings = []
    for _ in range(200):
        wp = core.generate_waypoint(x, y, yaw)
        levy_headings.append(wp.yaw)
        x, y, yaw = wp.x, wp.y, wp.yaw

    # Lawnmower heading sequence: alternating 0 / pi (two discrete turns).
    lawn = lawnmower_path(BOUNDS, spacing=5.0, step=2.0)
    lawn_headings = []
    for i in range(1, len(lawn)):
        dx = lawn[i][0] - lawn[i - 1][0]
        dy = lawn[i][1] - lawn[i - 1][1]
        lawn_headings.append(math.atan2(dy, dx))

    assert turn_angle_entropy(levy_headings) > turn_angle_entropy(lawn_headings)
