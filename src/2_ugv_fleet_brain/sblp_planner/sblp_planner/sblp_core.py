#!/usr/bin/env python3
"""
sblp_core.py — ROS-free core for the Spatially-Bounded Lévy Patrol (SBLP).
==========================================================================
All SBLP mathematics (heavy-tailed step sampling, geo-fenced rejection
sampling, boundary recovery, pure-pursuit steering, and micro-burst parameter
application) live here, decoupled from ROS so they can be unit-tested
deterministically. The ROS node (sblp_node.py) is a thin I/O wrapper around
this class.

Design notes
------------
* Randomness is injected via a ``random.Random`` instance so tests seed it for
  reproducible waypoints.
* Geo-fence polygons are plain ``[(x, y), ...]`` lists in the map/odom frame.
* All public methods are pure w.r.t. inputs except for the RNG and the mutable
  ``beta`` / ``l_max`` / ``polygon`` fields, which the Base Station may stretch
  at runtime via :meth:`apply_micro_burst`.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field


# ── Geometry helpers ────────────────────────────────────────────────────────
def point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray-casting point-in-polygon test. ``polygon`` is a list of (x, y)."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def polygon_centroid(polygon):
    """Vertex-mean centroid (robust for the convex sector polygons used here)."""
    n = len(polygon)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    return (cx, cy)


def reshape_flat_polygon(flat):
    """Convert a flat ``[x0, y0, x1, y1, ...]`` list into ``[(x0, y0), ...]``.

    Accepts an already-paired list unchanged. Raises ``ValueError`` on a flat
    list of odd length.
    """
    if not flat:
        return []
    if isinstance(flat[0], (list, tuple)):
        return [(float(p[0]), float(p[1])) for p in flat]
    if len(flat) % 2 != 0:
        raise ValueError('Flat polygon must have an even number of values')
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]


# ── Configuration & result types ────────────────────────────────────────────
@dataclass
class SBLPConfig:
    max_linear_vel: float = 1.5
    max_angular_vel: float = 0.6
    levy_beta: float = 1.8          # exponent in P(l) ~ l^(-beta), 1 < beta <= 3
    # l_min MUST exceed 2 * r_min (r_min = max_linear/max_angular) so that every
    # Lévy waypoint lies outside the UGV's minimum turning circle. With the
    # defaults above, r_min = 2.5 m, so l_min >= 5 m; 8 m adds safety margin.
    l_min: float = 8.0
    l_max: float = 60.0
    waypoint_tolerance: float = 2.0
    # Watchdog: if a waypoint isn't reached in this many seconds, regenerate a
    # new one. Prevents residual unreachability traps.
    waypoint_timeout_s: float = 30.0
    max_rejection_attempts: int = 50
    # Correlated random walk: heading changes drawn from a wrapped normal
    # centered at 0 with std deviation ``turn_sigma_rad``. Larger sigma ->
    # closer to a pure (isotropic) Lévy flight; smaller sigma -> straighter
    # paths. A ~90° std keeps the walk exploratory but Ackermann-friendly.
    turn_sigma_rad: float = 1.4
    # Occasional isotropic reorientation to guarantee full-plane coverage
    # (this preserves the mathematical Lévy character over long horizons).
    reorient_probability: float = 0.05
    # Terrain gating: reject candidate waypoints whose terrain traversal cost
    # (in [0, 1], 1 = lethal) exceeds this threshold. Only applied when a
    # terrain-cost function is provided to SBLPCore.
    terrain_cost_threshold: float = 0.7
    # Default patrol sector: 200 x 150 m rectangle centered on the odom origin.
    geofence_polygon: list = field(
        default_factory=lambda: [(-100.0, -75.0), (100.0, -75.0),
                                 (100.0, 75.0), (-100.0, 75.0)])


@dataclass
class Waypoint:
    x: float
    y: float
    yaw: float
    source: str          # "levy_flight" | "recovery"
    attempts: int        # rejection-sampler iterations used
    step_len: float      # drawn Lévy step length (m)
    turn_angle: float    # heading change from current yaw (rad)


# ── SBLP core ───────────────────────────────────────────────────────────────
class SBLPCore:
    def __init__(self, config: SBLPConfig | None = None,
                 rng: random.Random | None = None,
                 terrain_cost_fn=None):
        """``terrain_cost_fn(x, y) -> float in [0, 1]`` (1 = lethal), optional.

        When provided, waypoint candidates whose terrain cost exceeds
        ``config.terrain_cost_threshold`` are rejected, so the Lévy patrol never
        selects a goal on impassable slopes. When ``None``, terrain gating is
        skipped (geo-fence only).
        """
        self.cfg = config or SBLPConfig()
        self.rng = rng or random.Random()
        self.terrain_cost_fn = terrain_cost_fn
        # Runtime-mutable fields (stretched by Base Station micro-bursts).
        self.beta = float(self.cfg.levy_beta)
        self.l_max = float(self.cfg.l_max)
        self.polygon = reshape_flat_polygon(self.cfg.geofence_polygon)

    def _terrain_ok(self, x: float, y: float) -> bool:
        """True if terrain at (x, y) is traversable (or no cost fn is set)."""
        if self.terrain_cost_fn is None:
            return True
        try:
            return self.terrain_cost_fn(x, y) <= self.cfg.terrain_cost_threshold
        except Exception:
            # A costmap lookup failure should not crash patrol; treat as lethal
            # so we conservatively reject the candidate.
            return False

    # ── Lévy sampling ────────────────────────────────────────────────────
    def sample_levy_step(self) -> float:
        """Draw a step length from P(l) ~ l^(-beta) via inverse-transform.

        For a Pareto tail with exponent ``beta`` and minimum ``l_min``:
            l = l_min * u^(-1 / (beta - 1)),   u ~ Uniform(0, 1),
        truncated at ``l_max``.
        """
        if self.beta <= 1.0:
            # Degenerate exponent: fall back to a uniform draw in [l_min, l_max].
            return self.rng.uniform(self.cfg.l_min, self.l_max)
        u = self.rng.uniform(1e-3, 1.0 - 1e-3)
        step = self.cfg.l_min * (u ** (-1.0 / (self.beta - 1.0)))
        return min(step, self.l_max)

    def sample_turn_angle(self) -> float:
        """Draw a heading change for a *correlated* Lévy walk.

        Real foraging trajectories are not isotropic: consecutive step
        directions are correlated (the animal doesn't teleport around).
        Using a wrapped-normal turn distribution respects the UGV's Ackermann
        kinematics while preserving the heavy-tailed step length that makes
        the walk Lévy. An occasional isotropic reorientation restores full
        angular coverage over long horizons.
        """
        if self.rng.random() < self.cfg.reorient_probability:
            return self.rng.uniform(-math.pi, math.pi)
        d = self.rng.gauss(0.0, self.cfg.turn_sigma_rad)
        return math.atan2(math.sin(d), math.cos(d))  # wrap to [-pi, pi]

    def generate_waypoint(self, x: float, y: float, yaw: float) -> Waypoint:
        """Rejection-sample the next geo-fenced Lévy waypoint.

        Guarantees the returned waypoint lies inside the polygon when a valid
        candidate is found; otherwise returns a boundary-recovery waypoint
        aimed at the polygon centroid.
        """
        attempts = 0
        for attempts in range(1, self.cfg.max_rejection_attempts + 1):
            step = self.sample_levy_step()
            turn = self.sample_turn_angle()
            heading = yaw + turn
            cx = x + step * math.cos(heading)
            cy = y + step * math.sin(heading)
            if point_in_polygon(cx, cy, self.polygon) and self._terrain_ok(cx, cy):
                return Waypoint(cx, cy, heading, 'levy_flight',
                                attempts, step, turn)

        # Rejection sampling exhausted → recover toward centroid.
        cent_x, cent_y = polygon_centroid(self.polygon)
        rec_heading = math.atan2(cent_y - y, cent_x - x)
        rec_step = min(self.cfg.l_min * 2.0, math.hypot(cent_x - x, cent_y - y))
        rec_x = x + rec_step * math.cos(rec_heading)
        rec_y = y + rec_step * math.sin(rec_heading)
        return Waypoint(rec_x, rec_y, rec_heading, 'recovery',
                        attempts, rec_step, 0.0)

    # ── Navigation ───────────────────────────────────────────────────────
    def distance_to(self, x: float, y: float, target) -> float:
        return math.hypot(target[0] - x, target[1] - y)

    def reached(self, x: float, y: float, target, tol: float | None = None) -> bool:
        tol = self.cfg.waypoint_tolerance if tol is None else tol
        return self.distance_to(x, y, target) < tol

    def pure_pursuit(self, x: float, y: float, yaw: float, target):
        """Continuous-curvature Ackermann steering toward ``target``.

        Returns ``(linear_vel, angular_vel)`` with graceful slowdown near the
        waypoint and angular velocity clamped to the platform limit.
        """
        dx = target[0] - x
        dy = target[1] - y
        dist = math.hypot(dx, dy)

        speed = self.cfg.max_linear_vel
        slow_zone = self.cfg.waypoint_tolerance * 2.5
        if dist < slow_zone:
            speed = max(0.3, self.cfg.max_linear_vel * (dist / slow_zone))

        # Transform target into body frame; curvature = 2*y_local / d^2.
        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)
        y_local = dx * sin_y + dy * cos_y
        curvature = (2.0 * y_local) / (dist * dist) if dist > 0.1 else 0.0

        angular = speed * curvature
        angular = max(-self.cfg.max_angular_vel,
                      min(self.cfg.max_angular_vel, angular))
        return speed, angular

    # ── Micro-burst parameter stretching ─────────────────────────────────
    def apply_micro_burst(self, payload) -> dict:
        """Apply a Base Station micro-burst (dict or JSON string).

        Recognized keys: ``beta``, ``l_max``, ``polygon`` (paired or flat).
        Returns a dict describing what was applied. Malformed input raises
        ``ValueError`` (the ROS wrapper catches and logs it).
        """
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise ValueError('micro-burst payload must be a JSON object')

        applied = {}
        if 'beta' in payload:
            beta = float(payload['beta'])
            if beta <= 1.0:
                raise ValueError('beta must be > 1.0')
            self.beta = beta
            applied['beta'] = beta
        if 'l_max' in payload:
            l_max = float(payload['l_max'])
            if l_max <= 0.0:
                raise ValueError('l_max must be positive')
            self.l_max = l_max
            applied['l_max'] = l_max
        if 'polygon' in payload and isinstance(payload['polygon'], list):
            poly = reshape_flat_polygon(payload['polygon'])
            if len(poly) < 3:
                raise ValueError('polygon must have at least 3 vertices')
            self.polygon = poly
            applied['polygon_vertices'] = len(poly)
        return applied
