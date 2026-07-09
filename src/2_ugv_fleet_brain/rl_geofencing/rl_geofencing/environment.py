#!/usr/bin/env python3
"""
environment.py — Elastic geo-fencing MDP for offline RL training.
=================================================================
A Gymnasium environment that simulates breach scenarios: at reset, a random
fleet of sector polygons is placed over a border strip on a random costmap,
and one sentinel is marked as breached (its sector vacated). The agent outputs
per-remaining-agent stretch/translation parameters; the reward is the paper's
R = W1*C_coverage - W2*E_travel - W3*T_terrain.

Gymnasium and the RL trainer are optional imports so the rest of the package
(the Base Station node and its analytical fallback policy) works without them.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geom
from .reward import RewardWeights, TerrainSampler, compute_reward

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except Exception:  # pragma: no cover - optional dependency
    gym = object  # type: ignore
    spaces = None  # type: ignore
    _HAVE_GYM = False


class ElasticGeofenceEnv(gym.Env if _HAVE_GYM else object):
    """Single-step (contextual bandit style) reallocation environment.

    Observation: flattened [downsampled costmap | per-agent sector centroids |
                 per-agent sector extents | breach centroid | active mask].
    Action:      per remaining agent -> (scale_x, scale_y, dx, dy) in [-1, 1],
                 mapped to bounded stretch factors and translations.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_agents: int = 3, grid: int = 32,
                 border: tuple = (0.0, 0.0, 900.0, 300.0),
                 weights: RewardWeights | None = None, seed: int | None = None):
        super().__init__()
        self.n_agents = n_agents
        self.grid = grid
        self.border = border
        self.weights = weights or RewardWeights()
        self.rng = np.random.default_rng(seed)

        self._costmap = None
        self._sampler = None
        self._sectors: list[np.ndarray] = []
        self._breached = 0

        if _HAVE_GYM:
            obs_dim = grid * grid + n_agents * 4 + 2 + n_agents
            self.observation_space = spaces.Box(-np.inf, np.inf,
                                                shape=(obs_dim,), dtype=np.float32)
            # 4 params per (potentially) reallocated agent
            self.action_space = spaces.Box(-1.0, 1.0,
                                           shape=(n_agents * 4,), dtype=np.float32)

    # ── Scenario generation ───────────────────────────────────────────────
    def _random_costmap(self) -> np.ndarray:
        """Smooth random traversability field in [0,255] (255 = traversable)."""
        g = self.grid
        noise = self.rng.random((g, g)).astype(np.float32)
        # cheap smoothing via repeated box blur
        for _ in range(3):
            noise = 0.25 * (noise
                            + np.roll(noise, 1, 0)
                            + np.roll(noise, 1, 1)
                            + np.roll(noise, -1, 0))
        noise = (noise - noise.min()) / (noise.ptp() + 1e-9)
        return (noise * 255.0).astype(np.float32)

    def _tile_sectors(self) -> list[np.ndarray]:
        """Partition the border strip into n contiguous rectangular sectors."""
        min_x, min_y, max_x, max_y = self.border
        width = (max_x - min_x) / self.n_agents
        secs = []
        for i in range(self.n_agents):
            x0 = min_x + i * width
            x1 = x0 + width
            secs.append(np.array([[x0, min_y], [x1, min_y],
                                  [x1, max_y], [x0, max_y]], dtype=np.float32))
        return secs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._costmap = self._random_costmap()
        # Map grid to border extent for the sampler
        min_x, min_y, max_x, max_y = self.border
        res_x = (max_x - min_x) / self.grid
        self._sampler = TerrainSampler(self._costmap, (min_x, min_y), res_x)
        self._sectors = self._tile_sectors()
        self._breached = int(self.rng.integers(0, self.n_agents))
        return self._obs(), {}

    # ── Observation encoding ──────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        cost = (self._costmap / 255.0).flatten()
        feats = []
        for i, sec in enumerate(self._sectors):
            c = geom.polygon_centroid(sec)
            ext = sec.max(axis=0) - sec.min(axis=0)
            feats.extend([c[0], c[1], ext[0], ext[1]])
        breach_c = geom.polygon_centroid(self._sectors[self._breached])
        mask = [0.0 if i == self._breached else 1.0 for i in range(self.n_agents)]
        return np.concatenate([cost, np.array(feats, np.float32),
                               breach_c.astype(np.float32),
                               np.array(mask, np.float32)]).astype(np.float32)

    # ── Action application ────────────────────────────────────────────────
    def _apply_action(self, action: np.ndarray) -> list[np.ndarray]:
        a = np.asarray(action, dtype=np.float32).reshape(self.n_agents, 4)
        new_polys = []
        for i, sec in enumerate(self._sectors):
            if i == self._breached:
                continue
            sx = 1.0 + 0.75 * a[i, 0]        # stretch 0.25x .. 1.75x
            sy = 1.0 + 0.75 * a[i, 1]
            dx = 60.0 * a[i, 2]              # translate up to +-60 m
            dy = 30.0 * a[i, 3]
            c = geom.polygon_centroid(sec)
            stretched = geom.scale_polygon_about(sec, c, sx, sy)
            stretched = geom.translate_polygon(stretched, np.array([dx, dy]))
            new_polys.append(stretched)
        return new_polys

    def step(self, action):
        new_polys = self._apply_action(action)
        prev_polys = [s for i, s in enumerate(self._sectors) if i != self._breached]
        r, comp = compute_reward(prev_polys, new_polys, self.border,
                                 self._sampler, self.weights)
        # Single-step episode (contextual optimization).
        return self._obs(), float(r), True, False, comp
