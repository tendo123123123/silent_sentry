#!/usr/bin/env python3
"""
policy.py — Reallocation policy for the Base Station.
=====================================================
Provides a uniform `reallocate()` interface with two backends:

  1. LearnedPolicy   — loads an offline-trained Stable-Baselines3 model and
                       runs a single forward pass (used in the paper).
  2. AnalyticalPolicy — a closed-form, greedy fallback that stretches the
                       neighbours of the breached sector toward the vacated
                       zone while respecting terrain cost. Runs with zero ML
                       dependencies so the stack is functional before training.

Both return a list of (robot_id, new_polygon) tuples for the remaining agents.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geom
from .reward import RewardWeights, TerrainSampler, compute_reward


class AnalyticalPolicy:
    """Greedy, terrain-aware elastic stretch. Deterministic, no training needed."""

    def __init__(self, weights: RewardWeights | None = None):
        self.weights = weights or RewardWeights()

    def reallocate(self, sectors: dict[str, np.ndarray], breached_id: str,
                   sampler: TerrainSampler, border_bounds: tuple):
        breach_poly = sectors[breached_id]
        breach_c = geom.polygon_centroid(breach_poly)

        remaining = {rid: p for rid, p in sectors.items() if rid != breached_id}
        if not remaining:
            return [], 0.0

        # Assign the vacated sector to the geometrically nearest neighbour, and
        # let the others expand modestly toward the gap.
        dists = {rid: np.linalg.norm(geom.polygon_centroid(p) - breach_c)
                 for rid, p in remaining.items()}
        nearest = min(dists, key=dists.get)

        directives = []
        new_polys = []
        prev_polys = []
        for rid, poly in remaining.items():
            c = geom.polygon_centroid(poly)
            if rid == nearest:
                # Expand to encompass its own sector plus the vacated one.
                merged = np.vstack([poly, breach_poly])
                min_xy = merged.min(axis=0)
                max_xy = merged.max(axis=0)
                new_poly = np.array([
                    [min_xy[0], min_xy[1]], [max_xy[0], min_xy[1]],
                    [max_xy[0], max_xy[1]], [min_xy[0], max_xy[1]],
                ], dtype=np.float32)
            else:
                # Gentle stretch toward the breach centroid.
                direction = breach_c - c
                norm = np.linalg.norm(direction) + 1e-9
                shift = 0.25 * direction / norm * np.linalg.norm(
                    poly.max(axis=0) - poly.min(axis=0))
                new_poly = geom.scale_polygon_about(poly, c, 1.15, 1.15)
                new_poly = geom.translate_polygon(new_poly, shift)
            directives.append((rid, new_poly))
            new_polys.append(new_poly)
            prev_polys.append(poly)

        reward, _ = compute_reward(prev_polys, new_polys, border_bounds,
                                   sampler, self.weights)
        return directives, float(reward)


class LearnedPolicy:
    """Wraps an offline-trained SB3 model; falls back to analytical on failure."""

    def __init__(self, model_path: str, n_agents: int = 3,
                 weights: RewardWeights | None = None):
        self.n_agents = n_agents
        self.weights = weights or RewardWeights()
        self.model = None
        self._fallback = AnalyticalPolicy(weights)
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
        except Exception:
            # Model or SB3 unavailable — analytical path will be used.
            self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def reallocate(self, sectors, breached_id, sampler, border_bounds):
        if self.model is None:
            return self._fallback.reallocate(sectors, breached_id,
                                             sampler, border_bounds)
        # Build the same observation the env produces, run one forward pass,
        # and decode the action into polygons. Import here to avoid a hard dep.
        from .environment import ElasticGeofenceEnv
        env = ElasticGeofenceEnv(n_agents=self.n_agents, border=border_bounds,
                                 weights=self.weights)
        env._costmap = (1.0 - sampler.cost) * 255.0
        env._sampler = sampler
        ordered_ids = list(sectors.keys())
        env._sectors = [sectors[k] for k in ordered_ids]
        env._breached = ordered_ids.index(breached_id)
        obs = env._obs()
        action, _ = self.model.predict(obs, deterministic=True)
        new_polys = env._apply_action(action)
        remaining_ids = [k for k in ordered_ids if k != breached_id]
        directives = list(zip(remaining_ids, new_polys))
        prev = [sectors[k] for k in remaining_ids]
        reward, _ = compute_reward(prev, new_polys, border_bounds,
                                   sampler, self.weights)
        return directives, float(reward)


def build_policy(model_path: str | None, n_agents: int,
                 weights: RewardWeights | None = None):
    """Factory: use the learned policy if a model path is given, else analytical."""
    if model_path:
        return LearnedPolicy(model_path, n_agents=n_agents, weights=weights)
    return AnalyticalPolicy(weights=weights)
