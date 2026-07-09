#!/usr/bin/env python3
"""
base_station_node.py — RL Elastic Geo-Fencing brain (Base Station only).
========================================================================
Runs exclusively on the Base Station. It maintains the last-known geo-fence of
each sentinel, listens for breach reports (delivered as single encrypted Zenoh
micro-bursts when a sentinel leaves its sector), runs the offline-trained RL
policy to compute healed sector polygons, and broadcasts ONE reallocation
message to the remaining fleet — preserving EMCON stealth for the UGVs.

Transport note:
  * Typed I/O uses silent_sentry_interfaces (BreachReport / FleetReallocation).
  * For drop-in compatibility with the existing SBLP planner, the healed
    polygons are ALSO emitted as JSON on each robot's /<robot>/sblp/micro_burst
    (std_msgs/String), which SBLPPlanner already consumes.

Subscribes:
  /fleet/breach_report      (silent_sentry_interfaces/BreachReport)
Publishes:
  /fleet/reallocation       (silent_sentry_interfaces/FleetReallocation)
  /<robot>/sblp/micro_burst (std_msgs/String, JSON)  — per remaining sentinel
"""
from __future__ import annotations

import json
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point32

from .reward import RewardWeights, TerrainSampler
from .policy import build_policy

try:
    from silent_sentry_interfaces.msg import (
        BreachReport, GeoFenceDirective, FleetReallocation,
    )
    _HAVE_MSGS = True
except Exception:  # pragma: no cover - interfaces not yet built
    _HAVE_MSGS = False


def _load_costmap(path: str, grid: int = 64):
    """Load a global costmap (.npy) or synthesize a flat traversable field."""
    try:
        arr = np.load(path)
        return arr.astype(np.float32)
    except Exception:
        return np.full((grid, grid), 255.0, dtype=np.float32)


class BaseStationRLNode(Node):
    def __init__(self):
        super().__init__('base_station_rl_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('model_path', '')          # '' -> analytical fallback
        self.declare_parameter('costmap_path', '')        # .npy global costmap
        self.declare_parameter('border_bounds', [0.0, 0.0, 900.0, 300.0])
        self.declare_parameter('costmap_origin', [0.0, 0.0])
        self.declare_parameter('costmap_resolution', 1.0)
        self.declare_parameter('robot_ids', ['alpha', 'beta', 'gamma'])
        self.declare_parameter('w_coverage', 1.0)
        self.declare_parameter('w_travel', 0.3)
        self.declare_parameter('w_terrain', 0.6)
        # Flat [x0,y0,x1,y1,...] initial sectors per robot (optional).
        self.declare_parameter('initial_sectors', [])

        gp = self.get_parameter
        model_path = gp('model_path').value or None
        self.border = tuple(gp('border_bounds').value)
        self.robot_ids = list(gp('robot_ids').value)
        weights = RewardWeights(
            w_coverage=gp('w_coverage').value,
            w_travel=gp('w_travel').value,
            w_terrain=gp('w_terrain').value,
        )

        costmap = _load_costmap(gp('costmap_path').value)
        self.sampler = TerrainSampler(
            costmap,
            tuple(gp('costmap_origin').value),
            float(gp('costmap_resolution').value),
        )
        self.policy = build_policy(model_path, n_agents=len(self.robot_ids),
                                   weights=weights)

        # ── Sector state ──────────────────────────────────────────────────
        self.sectors = self._init_sectors(gp('initial_sectors').value)

        # ── ROS I/O ───────────────────────────────────────────────────────
        self.micro_burst_pubs = {
            rid: self.create_publisher(String, f'/{rid}/sblp/micro_burst', 10)
            for rid in self.robot_ids
        }
        if _HAVE_MSGS:
            self.realloc_pub = self.create_publisher(
                FleetReallocation, '/fleet/reallocation', 10)
            self.create_subscription(
                BreachReport, '/fleet/breach_report', self._on_breach, 10)
        else:
            self.get_logger().warn(
                'silent_sentry_interfaces not built — typed breach I/O disabled. '
                'Build the interfaces package to enable /fleet/breach_report.')

        backend = ('learned' if getattr(self.policy, 'loaded', False)
                   else 'analytical')
        self.get_logger().info(
            f'Base Station RL brain ready. policy={backend}, '
            f'fleet={self.robot_ids}, border={self.border}')

    # ── Sector initialization ─────────────────────────────────────────────
    def _init_sectors(self, flat) -> dict[str, np.ndarray]:
        sectors: dict[str, np.ndarray] = {}
        if flat and len(flat) >= 8:
            # Interpret as one rectangle [x0,y0,x1,y1] per robot.
            for i, rid in enumerate(self.robot_ids):
                base = i * 4
                if base + 4 <= len(flat):
                    x0, y0, x1, y1 = flat[base:base + 4]
                    sectors[rid] = np.array(
                        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        if not sectors:
            # Default: tile the border strip evenly.
            min_x, min_y, max_x, max_y = self.border
            w = (max_x - min_x) / len(self.robot_ids)
            for i, rid in enumerate(self.robot_ids):
                x0 = min_x + i * w
                sectors[rid] = np.array(
                    [[x0, min_y], [x0 + w, min_y],
                     [x0 + w, max_y], [x0, max_y]], np.float32)
        return sectors

    # ── Breach handling ────────────────────────────────────────────────────
    def _on_breach(self, msg):  # type: ignore[no-untyped-def]
        breached = msg.robot_id
        if breached not in self.sectors:
            self.get_logger().warn(f'Breach from unknown sentinel "{breached}"')
            return

        # If the report carries the vacated polygon, trust it over our cache.
        if msg.vacated_sector:
            self.sectors[breached] = np.array(
                [[p.x, p.y] for p in msg.vacated_sector], np.float32)

        self.get_logger().warn(
            f'BREACH: "{breached}" left its sector (reason="{msg.reason}"). '
            f'Computing elastic reallocation...')

        directives, reward = self.policy.reallocate(
            self.sectors, breached, self.sampler, self.border)

        self._broadcast(directives, reward, breached)

        # Update our cache so subsequent breaches reason about the new layout.
        for rid, poly in directives:
            self.sectors[rid] = np.asarray(poly, np.float32)

    # ── Single broadcast ─────────────────────────────────────────────────
    def _broadcast(self, directives, reward, triggered_by):
        # 1) Typed fleet reallocation (single message).
        if _HAVE_MSGS:
            fr = FleetReallocation()
            fr.header = Header()
            fr.header.stamp = self.get_clock().now().to_msg()
            fr.triggered_by = triggered_by
            fr.expected_reward = float(reward)
            for rid, poly in directives:
                d = GeoFenceDirective()
                d.robot_id = rid
                d.polygon = [Point32(x=float(p[0]), y=float(p[1]), z=0.0)
                             for p in poly]
                d.levy_beta = -1.0   # unchanged
                d.l_max = -1.0
                fr.directives.append(d)
            self.realloc_pub.publish(fr)

        # 2) JSON micro-burst per remaining sentinel (SBLP-compatible).
        for rid, poly in directives:
            payload = {'polygon': [[float(p[0]), float(p[1])] for p in poly]}
            self.micro_burst_pubs[rid].publish(String(data=json.dumps(payload)))

        self.get_logger().info(
            f'Reallocation broadcast: {len(directives)} sector(s) updated, '
            f'expected R={reward:.3f}')


def main(args=None):
    rclpy.init(args=args)
    node = BaseStationRLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
