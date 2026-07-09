#!/usr/bin/env python3
"""
reward.py — Reward terms for RL-optimized elastic geo-fencing.
==============================================================
Implements the paper's reward function

    R = W1 * C_coverage  -  W2 * E_travel  -  W3 * T_terrain

as three composable, normalized terms in [0, 1] so the weights W1..W3 carry
their intended relative meaning. All functions are pure (numpy only).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import geometry as geom


@dataclass
class RewardWeights:
    w_coverage: float = 1.0   # W1 — reward border coverage
    w_travel: float = 0.3     # W2 — penalize repositioning effort
    w_terrain: float = 0.6    # W3 — penalize lethal-slope routing


class TerrainSampler:
    """Bilinear lookup into a global traversability/slope costmap.

    Cost convention (matches the a-priori GIS costmap): 0 = fully traversable,
    1 = lethal. The raw costmap stores 255 = traversable, 0 = lethal, so we
    invert and normalize on load.
    """

    def __init__(self, costmap: np.ndarray, origin: tuple, resolution: float):
        # costmap: 2D array, values in [0, 255] (255 = traversable).
        self.cost = 1.0 - (np.asarray(costmap, dtype=np.float32) / 255.0)
        self.origin = np.asarray(origin, dtype=np.float32)  # (x0, y0) of cell (0,0)
        self.res = float(resolution)
        self.h, self.w = self.cost.shape

    def cost_at(self, x: float, y: float) -> float:
        col = (x - self.origin[0]) / self.res
        row = (y - self.origin[1]) / self.res
        if col < 0 or row < 0 or col >= self.w - 1 or row >= self.h - 1:
            return 1.0  # out of bounds treated as lethal
        c0, r0 = int(np.floor(col)), int(np.floor(row))
        fc, fr = col - c0, row - r0
        v = (self.cost[r0, c0] * (1 - fc) * (1 - fr)
             + self.cost[r0, c0 + 1] * fc * (1 - fr)
             + self.cost[r0 + 1, c0] * (1 - fc) * fr
             + self.cost[r0 + 1, c0 + 1] * fc * fr)
        return float(v)


def coverage_term(new_polys: list[np.ndarray], border_bounds: tuple,
                  res: float = 5.0) -> float:
    """Fraction of the border region covered by the union of reassigned sectors."""
    grid = geom.rasterize_coverage(new_polys, border_bounds, res)
    return float(grid.mean()) if grid.size else 0.0


def travel_term(prev_polys: list[np.ndarray], new_polys: list[np.ndarray],
                normalizer: float) -> float:
    """Normalized mean centroid displacement each agent must incur (in [0, 1])."""
    if not new_polys:
        return 0.0
    dists = []
    for prev, new in zip(prev_polys, new_polys):
        dists.append(np.linalg.norm(geom.polygon_centroid(new)
                                    - geom.polygon_centroid(prev)))
    mean_d = float(np.mean(dists)) if dists else 0.0
    return min(1.0, mean_d / max(normalizer, 1e-6))


def terrain_term(new_polys: list[np.ndarray], sampler: TerrainSampler,
                 spacing: float = 5.0) -> float:
    """Mean traversal cost (0..1) sampled over the interiors of the new sectors."""
    costs = []
    for poly in new_polys:
        pts = geom.sample_points_in_polygon(poly, spacing)
        for p in pts:
            costs.append(sampler.cost_at(p[0], p[1]))
    return float(np.mean(costs)) if costs else 1.0


def compute_reward(prev_polys, new_polys, border_bounds, sampler,
                   weights: RewardWeights, travel_norm: float = 100.0):
    """Full reward R = W1*C - W2*E - W3*T, returning (R, components dict)."""
    c = coverage_term(new_polys, border_bounds)
    e = travel_term(prev_polys, new_polys, travel_norm)
    t = terrain_term(new_polys, sampler)
    r = weights.w_coverage * c - weights.w_travel * e - weights.w_terrain * t
    return r, {'coverage': c, 'travel': e, 'terrain': t}
