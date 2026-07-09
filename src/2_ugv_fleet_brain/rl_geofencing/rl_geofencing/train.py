#!/usr/bin/env python3
"""
train.py — Offline PPO training for the elastic geo-fencing policy.
===================================================================
Trains a policy over randomized breach scenarios in ElasticGeofenceEnv and
saves the model for the Base Station node to load at runtime.

This is an OFFLINE tool (run on a workstation), not a ROS node. Requires
`gymnasium` and `stable-baselines3`:

    pip install gymnasium stable-baselines3

Usage:
    python3 -m rl_geofencing.train --timesteps 2000000 --out models/geofence_ppo
"""
from __future__ import annotations

import argparse

from .environment import ElasticGeofenceEnv, _HAVE_GYM
from .reward import RewardWeights


def main():
    ap = argparse.ArgumentParser(description='Train elastic geo-fencing PPO policy.')
    ap.add_argument('--timesteps', type=int, default=2_000_000)
    ap.add_argument('--n-agents', type=int, default=3)
    ap.add_argument('--out', type=str, default='models/geofence_ppo')
    ap.add_argument('--w-coverage', type=float, default=1.0)
    ap.add_argument('--w-travel', type=float, default=0.3)
    ap.add_argument('--w-terrain', type=float, default=0.6)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if not _HAVE_GYM:
        raise SystemExit(
            'gymnasium is not installed. Install training deps:\n'
            '    pip install gymnasium stable-baselines3')
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f'stable-baselines3 required for training: {exc}')

    weights = RewardWeights(args.w_coverage, args.w_travel, args.w_terrain)

    def _factory():
        return ElasticGeofenceEnv(n_agents=args.n_agents, weights=weights,
                                  seed=args.seed)

    env = make_vec_env(_factory, n_envs=8, seed=args.seed)
    model = PPO('MlpPolicy', env, verbose=1, seed=args.seed,
                n_steps=256, batch_size=256, gae_lambda=0.95, gamma=0.99)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f'Saved trained policy to {args.out}.zip')


if __name__ == '__main__':
    main()
