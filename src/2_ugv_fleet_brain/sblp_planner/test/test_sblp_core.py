#!/usr/bin/env python3
"""Unit tests for the ROS-free SBLP core (sblp_planner.sblp_core)."""
import math
import random

import pytest

from sblp_planner.sblp_core import (
    SBLPConfig,
    SBLPCore,
    point_in_polygon,
    polygon_centroid,
    reshape_flat_polygon,
)

SQUARE = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]


# ── Geometry helpers ────────────────────────────────────────────────────────
def test_point_in_polygon_inside():
    assert point_in_polygon(0.0, 0.0, SQUARE) is True
    assert point_in_polygon(5.0, -5.0, SQUARE) is True


def test_point_in_polygon_outside():
    assert point_in_polygon(20.0, 0.0, SQUARE) is False
    assert point_in_polygon(0.0, 100.0, SQUARE) is False


def test_point_in_polygon_degenerate():
    assert point_in_polygon(0.0, 0.0, [(0.0, 0.0), (1.0, 1.0)]) is False


def test_polygon_centroid():
    cx, cy = polygon_centroid(SQUARE)
    assert cx == pytest.approx(0.0)
    assert cy == pytest.approx(0.0)


def test_reshape_flat_polygon_flat():
    poly = reshape_flat_polygon([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    assert poly == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]


def test_reshape_flat_polygon_paired_passthrough():
    poly = reshape_flat_polygon([(0.0, 0.0), (1.0, 1.0)])
    assert poly == [(0.0, 0.0), (1.0, 1.0)]


def test_reshape_flat_polygon_odd_raises():
    with pytest.raises(ValueError):
        reshape_flat_polygon([0.0, 1.0, 2.0])


# ── Lévy sampling ─────────────────────────────────────────────────────────────
def _core(seed=0, **kw):
    cfg = SBLPConfig(geofence_polygon=SQUARE, **kw)
    return SBLPCore(cfg, rng=random.Random(seed))


def test_levy_step_within_bounds():
    core = _core(seed=1, l_min=3.0, l_max=35.0)
    for _ in range(2000):
        s = core.sample_levy_step()
        assert 3.0 <= s <= 35.0 + 1e-9


def test_levy_step_is_heavy_tailed():
    """Mean step should exceed l_min and the tail should occasionally saturate."""
    core = _core(seed=2, l_min=3.0, l_max=35.0, levy_beta=1.8)
    samples = [core.sample_levy_step() for _ in range(5000)]
    assert sum(samples) / len(samples) > 3.0
    # Heavy tail → some draws near the truncation limit.
    assert max(samples) > 20.0


def test_levy_step_degenerate_beta():
    core = _core(seed=3, l_min=3.0, l_max=35.0, levy_beta=1.0)  # beta<=1 → uniform fallback
    for _ in range(500):
        s = core.sample_levy_step()
        assert 3.0 <= s <= 35.0 + 1e-9


# ── Waypoint generation ───────────────────────────────────────────────────────
def test_generated_waypoint_inside_polygon():
    core = _core(seed=4)
    for _ in range(500):
        wp = core.generate_waypoint(0.0, 0.0, 0.0)
        if wp.source == 'levy_flight':
            assert point_in_polygon(wp.x, wp.y, SQUARE)


def test_waypoint_deterministic_with_seed():
    a = _core(seed=123).generate_waypoint(0.0, 0.0, 0.0)
    b = _core(seed=123).generate_waypoint(0.0, 0.0, 0.0)
    assert (a.x, a.y, a.yaw) == (b.x, b.y, b.yaw)


def test_waypoint_recovery_when_outside():
    """Robot far outside the polygon → recovery aimed toward the centroid."""
    core = _core(seed=5, max_rejection_attempts=5)
    wp = core.generate_waypoint(1000.0, 1000.0, 0.0)
    assert wp.source == 'recovery'
    # Recovery heading should point back toward the centroid (down-left).
    assert math.cos(wp.yaw) < 0 and math.sin(wp.yaw) < 0


def test_waypoint_never_lands_inside_turning_circle():
    """Every Lévy waypoint must be reachable: >= 2*r_min from the robot.

    Guards against the orbit trap where the pure-pursuit controller is stuck
    on a target that lies inside its minimum turning circle.
    """
    core = _core(seed=42)  # defaults: l_min=8.0, r_min=2.5 → 2*r_min=5.0
    r_min = core.cfg.max_linear_vel / core.cfg.max_angular_vel
    for _ in range(500):
        wp = core.generate_waypoint(0.0, 0.0, 0.0)
        if wp.source == 'levy_flight':
            assert math.hypot(wp.x, wp.y) >= 2.0 * r_min - 1e-6


def test_correlated_turn_biases_forward():
    """A correlated random walk should keep most steps within a forward cone."""
    core = _core(seed=7, turn_sigma_rad=1.0, reorient_probability=0.0)
    turns = [core.sample_turn_angle() for _ in range(2000)]
    # With sigma=1 rad, ~68% of turns should be within ±1 rad of straight ahead.
    within_one = sum(1 for t in turns if abs(t) < 1.0)
    assert within_one / len(turns) > 0.55


def test_correlated_turn_reorient_gives_full_coverage():
    """With reorient_probability=1, turns fill all four quadrants."""
    core = _core(seed=8, reorient_probability=1.0)
    turns = [core.sample_turn_angle() for _ in range(2000)]
    assert any(t < -1.0 for t in turns)
    assert any(t > 1.0 for t in turns)


# ── Navigation ────────────────────────────────────────────────────────────────
def test_reached():
    core = _core()
    assert core.reached(0.0, 0.0, (1.0, 0.0), tol=1.5) is True
    assert core.reached(0.0, 0.0, (5.0, 0.0), tol=1.5) is False


def test_pure_pursuit_straight_ahead():
    core = _core()
    speed, angular = core.pure_pursuit(0.0, 0.0, 0.0, (10.0, 0.0))
    assert speed == pytest.approx(core.cfg.max_linear_vel)
    assert abs(angular) < 1e-6


def test_pure_pursuit_turns_left_for_left_target():
    core = _core()
    _, angular = core.pure_pursuit(0.0, 0.0, 0.0, (10.0, 5.0))
    assert angular > 0.0  # +z = CCW = left


def test_pure_pursuit_turns_right_for_right_target():
    core = _core()
    _, angular = core.pure_pursuit(0.0, 0.0, 0.0, (10.0, -5.0))
    assert angular < 0.0


def test_pure_pursuit_angular_clamp():
    core = _core()  # max_angular_vel default 0.6
    _, angular = core.pure_pursuit(0.0, 0.0, 0.0, (0.0, 4.0))
    assert abs(angular) <= core.cfg.max_angular_vel + 1e-9
    assert abs(angular) == pytest.approx(core.cfg.max_angular_vel)


def test_pure_pursuit_slows_near_waypoint():
    core = _core()
    speed, _ = core.pure_pursuit(0.0, 0.0, 0.0, (1.0, 0.0))
    assert 0.3 <= speed < core.cfg.max_linear_vel


# ── Micro-burst application ───────────────────────────────────────────────────
def test_micro_burst_dict_updates_fields():
    core = _core()
    applied = core.apply_micro_burst({'beta': 2.5, 'l_max': 50.0})
    assert core.beta == 2.5
    assert core.l_max == 50.0
    assert applied == {'beta': 2.5, 'l_max': 50.0}


def test_micro_burst_json_string_and_polygon():
    core = _core()
    applied = core.apply_micro_burst('{"polygon": [0,0, 20,0, 20,20, 0,20]}')
    assert applied['polygon_vertices'] == 4
    assert core.polygon == [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]


def test_micro_burst_rejects_bad_beta():
    core = _core()
    with pytest.raises(ValueError):
        core.apply_micro_burst({'beta': 0.5})


def test_micro_burst_rejects_non_object():
    core = _core()
    with pytest.raises(ValueError):
        core.apply_micro_burst('[1, 2, 3]')


def test_terrain_gating_rejects_lethal_waypoints():
    """With a terrain-cost function, waypoints only land on traversable cells."""
    # Lethal in the entire right half-plane (x > 0), traversable on the left.
    def cost_fn(x, y):
        return 1.0 if x > 0.0 else 0.0

    cfg = SBLPConfig(geofence_polygon=SQUARE, terrain_cost_threshold=0.7)
    core = SBLPCore(cfg, rng=random.Random(9), terrain_cost_fn=cost_fn)
    for _ in range(300):
        wp = core.generate_waypoint(-5.0, 0.0, math.pi)  # start left, face left
        if wp.source == 'levy_flight':
            assert wp.x <= 0.0  # never selects the lethal right half


def test_terrain_gating_absent_allows_all():
    """Without a cost function, terrain gating is skipped (geo-fence only)."""
    core = _core(seed=10)
    assert core.terrain_cost_fn is None
    wp = core.generate_waypoint(0.0, 0.0, 0.0)
    assert wp is not None


def test_terrain_cost_lookup_failure_is_conservative():
    """A raising cost function is treated as lethal (candidate rejected)."""
    def bad(x, y):
        raise RuntimeError('costmap unavailable')

    cfg = SBLPConfig(geofence_polygon=SQUARE)
    core = SBLPCore(cfg, rng=random.Random(1), terrain_cost_fn=bad)
    # Every candidate is rejected → falls through to recovery.
    wp = core.generate_waypoint(0.0, 0.0, 0.0)
    assert wp.source == 'recovery'


def test_micro_burst_polygon_takes_effect_on_sampling():
    """After stretching the polygon, waypoints respect the NEW bounds."""
    core = _core(seed=7)
    core.apply_micro_burst({'polygon': [0, 0, 5, 0, 5, 5, 0, 5]})
    for _ in range(300):
        wp = core.generate_waypoint(2.5, 2.5, 0.0)
        if wp.source == 'levy_flight':
            assert point_in_polygon(wp.x, wp.y, core.polygon)
