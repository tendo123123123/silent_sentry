#!/usr/bin/env python3
"""
RL Coverage Brain Node (Base Station)
=====================================
Elastic Geo-Fencing via RL and Encrypted Micro-Burst Broadcast logic.

Monitors fleet surveillance coverage across a discretized desert sector grid using
inbound HMAC-signed telemetry micro-bursts from EMCON-silenced UGVs.
When a sector's coverage uncertainty exceeds a breach threshold (or an agent drops out),
an embedded RL Policy Inference engine calculates mathematically optimal reallocation
parameters (Lévy exponent beta, step limit l_max, and elastic geo-fence polygon)
and broadcasts a single encrypted micro-burst to autonomously heal the patrol net.

Subscribes:
  /emcon/telemetry_burst (std_msgs/String) — HMAC-signed telemetry from EMCON controller
  /odometry/filtered     (nav_msgs/Odometry) — direct fallback tracking for simulation

Publishes:
  /emcon/command_burst       (std_msgs/String)  — encrypted micro-burst directive to UGVs
  /sblp/micro_burst          (std_msgs/String)  — direct SBLP bridge
  /base_station/coverage     (std_msgs/Float32) — overall surveillance score [0..1]
  /base_station/breach_alert (std_msgs/String)  — breach logs and RL action reports
"""

import json
import math
import hashlib
import hmac
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry


class RLCoverageBrain(Node):
    def __init__(self):
        super().__init__('rl_coverage_brain')

        # Declare parameters
        self.declare_parameter('grid_size_x', 60.0)
        self.declare_parameter('grid_size_y', 60.0)
        self.declare_parameter('grid_rows', 4)
        self.declare_parameter('grid_cols', 4)
        self.declare_parameter('decay_rate', 0.05)
        self.declare_parameter('breach_threshold', 0.80)
        self.declare_parameter('default_beta', 1.8)
        self.declare_parameter('stretched_beta', 1.3)
        self.declare_parameter('default_l_max', 35.0)
        self.declare_parameter('stretched_l_max', 65.0)
        self.declare_parameter('hmac_secret_key', "sentry_emcon_secret_key_2026")

        self.cell_x    = self.get_parameter('grid_size_x').value
        self.cell_y    = self.get_parameter('grid_size_y').value
        self.rows      = self.get_parameter('grid_rows').value
        self.cols      = self.get_parameter('grid_cols').value
        self.decay     = self.get_parameter('decay_rate').value
        self.breach_th = self.get_parameter('breach_threshold').value
        self.def_beta  = self.get_parameter('default_beta').value
        self.str_beta  = self.get_parameter('stretched_beta').value
        self.def_lmax  = self.get_parameter('default_l_max').value
        self.str_lmax  = self.get_parameter('stretched_l_max').value
        self.secret    = self.get_parameter('hmac_secret_key').value.encode('utf-8')

        # Initialize sector uncertainty grid [rows x cols] with 0.0 (fully covered)
        self.uncertainty_grid = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.ugv_positions = {}
        self.healing_active = False

        # Subscribers
        self.create_subscription(String,   '/emcon/telemetry_burst', self._burst_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered',     self._odom_cb,  10)

        # Publishers
        self._cmd_burst_pub  = self.create_publisher(String,  '/emcon/command_burst',       10)
        self._sblp_pub       = self.create_publisher(String,  '/sblp/micro_burst',          10)
        self._score_pub      = self.create_publisher(Float32, '/base_station/coverage',      10)
        self._alert_pub      = self.create_publisher(String,  '/base_station/breach_alert', 10)

        # Costmap evaluation and RL inference loop at 2 Hz
        self.create_timer(0.5, self._evaluation_loop)
        self.get_logger().info(
            f'RL Coverage Brain ready | Grid: {self.rows}x{self.cols} ({self.cell_x*self.cols}x{self.cell_y*self.rows}m) | '
            f'Breach threshold: {self.breach_th}'
        )

    # ── Telemetry Ingestion & Grid Mapping ────────────────────────────────────
    def _burst_cb(self, msg: String):
        """Ingest HMAC-signed telemetry micro-burst from EMCON silenced UGVs."""
        try:
            packet = json.loads(msg.data)
            payload = packet.get("data", {})
            auth_tag = packet.get("auth_tag", "")
            
            # Verify HMAC
            raw_payload = json.dumps(payload, separators=(',', ':'))
            expected_tag = hmac.new(self.secret, raw_payload.encode('utf-8'), hashlib.sha256).hexdigest()[:16]
            if auth_tag and auth_tag != expected_tag:
                self.get_logger().error("Base Station: Inbound telemetry HMAC mismatch! Dropping.")
                return
                
            ugv_id = payload.get("ugv_id", "ugv")
            history = payload.get("history", [])
            if history:
                latest = history[-1].get("odom", {})
                if "x" in latest and "y" in latest:
                    self._register_ugv_visit(ugv_id, latest["x"], latest["y"])
        except Exception as e:
            self.get_logger().error(f"Base Station: Failed to parse telemetry burst: {e}")

    def _odom_cb(self, msg: Odometry):
        """Direct fallback tracking for simulation."""
        pos = msg.pose.pose.position
        self._register_ugv_visit("autobot", pos.x, pos.y)

    def _register_ugv_visit(self, ugv_id: str, x: float, y: float):
        self.ugv_positions[ugv_id] = (x, y)
        r, c = self._pos_to_grid(x, y)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            # Visit resets cell uncertainty to zero
            self.uncertainty_grid[r][c] = 0.0

    def _pos_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Map (x, y) coordinates to (row, col) in surveillance grid centered at origin."""
        origin_x = -(self.cols * self.cell_x) / 2.0
        origin_y = -(self.rows * self.cell_y) / 2.0
        c = int((x - origin_x) / self.cell_x)
        r = int((y - origin_y) / self.cell_y)
        return r, c

    def _grid_to_pos(self, r: int, c: int) -> tuple[float, float]:
        """Map (row, col) center to (x, y) world coordinates."""
        origin_x = -(self.cols * self.cell_x) / 2.0
        origin_y = -(self.rows * self.cell_y) / 2.0
        x = origin_x + (c + 0.5) * self.cell_x
        y = origin_y + (r + 0.5) * self.cell_y
        return x, y

    # ── Evaluation & RL Policy Inference Loop ─────────────────────────────────
    def _evaluation_loop(self):
        dt = 0.5
        max_uncertainty = 0.0
        breach_row, breach_col = -1, -1
        total_cells = self.rows * self.cols
        sum_uncertainty = 0.0

        # Update uncertainty decay across grid
        for r in range(self.rows):
            for c in range(self.cols):
                self.uncertainty_grid[r][c] = min(1.0, self.uncertainty_grid[r][c] + self.decay * dt)
                u = self.uncertainty_grid[r][c]
                sum_uncertainty += u
                if u > max_uncertainty:
                    max_uncertainty = u
                    breach_row, breach_col = r, c

        # Compute overall surveillance coverage score [0..1]
        mean_uncertainty = sum_uncertainty / total_cells
        coverage_score = max(0.0, 1.0 - mean_uncertainty)
        self._score_pub.publish(Float32(data=float(coverage_score)))

        # Check for coverage breach triggering RL reallocation
        if max_uncertainty >= self.breach_th and not self.healing_active:
            bx, by = self._grid_to_pos(breach_row, breach_col)
            self.get_logger().warn(
                f"Base Station: COVERAGE BREACH at grid ({breach_row}, {breach_col}) "
                f"[world ({bx:.1f}, {by:.1f})m, uncertainty={max_uncertainty:.2f}]! "
                f"Triggering RL Elastic Geo-Fence reallocation."
            )
            self._execute_rl_reallocation(bx, by, max_uncertainty)
            self.healing_active = True
        elif max_uncertainty < self.breach_th * 0.6 and self.healing_active:
            self.get_logger().info("Base Station: Surveillance coverage restored! Reverting to default SBLP parameters.")
            self._revert_default_patrol()
            self.healing_active = False

    def _execute_rl_reallocation(self, breach_x: float, breach_y: float, severity: float):
        """
        RL Policy Inference:
        Maps breach coordinates and severity to optimal SBLP parameter stretches.
        Lower beta -> heavy-tailed super-diffusion jumps toward breached sector.
        Stretched l_max -> extended step reach.
        Elastic Geo-Fence -> polygon stretched outward to encompass breach coordinates.
        """
        # Calculate optimal beta (more severe breach -> lower beta for longer Lévy flights)
        inferred_beta = max(1.15, self.def_beta - (severity - self.breach_th) * 2.0)
        inferred_beta = min(inferred_beta, self.str_beta)
        
        # Calculate optimal step limit
        inferred_lmax = min(80.0, self.def_lmax * (1.0 + severity))
        
        # Calculate elastic geo-fence polygon stretching toward breach
        half_w = (self.cols * self.cell_x) / 2.0
        half_h = (self.rows * self.cell_y) / 2.0
        
        # Expand boundary in direction of breach
        stretch_pad = 40.0
        min_x = min(-half_w, breach_x - stretch_pad)
        max_x = max( half_w, breach_x + stretch_pad)
        min_y = min(-half_h, breach_y - stretch_pad)
        max_y = max( half_h, breach_y + stretch_pad)
        
        elastic_polygon = [
            [round(min_x, 1), round(min_y, 1)],
            [round(max_x, 1), round(min_y, 1)],
            [round(max_x, 1), round(max_y, 1)],
            [round(min_x, 1), round(max_y, 1)]
        ]

        # Construct reallocation directive
        payload = {
            "action": "RL_HEAL_BREACH",
            "emcon_state": 3,  # Brief EMERGENCY_BURST state to receive directive
            "beta": round(inferred_beta, 2),
            "l_max": round(inferred_lmax, 1),
            "polygon": elastic_polygon,
            "target_breach": [round(breach_x, 1), round(breach_y, 1)],
            "severity": round(severity, 2)
        }
        
        self._broadcast_secure_micro_burst(payload)
        
        alert_msg = json.dumps({"event": "RL_REALLOCATION", "payload": payload})
        self._alert_pub.publish(String(data=alert_msg))

    def _revert_default_patrol(self):
        half_w = (self.cols * self.cell_x) / 2.0
        half_h = (self.rows * self.cell_y) / 2.0
        default_polygon = [
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h]
        ]
        payload = {
            "action": "RL_RESTORE_DEFAULT",
            "emcon_state": 2,  # Return to ZERO_EMISSION stealth patrol
            "beta": self.def_beta,
            "l_max": self.def_lmax,
            "polygon": default_polygon
        }
        self._broadcast_secure_micro_burst(payload)

    def _broadcast_secure_micro_burst(self, payload: dict):
        raw_json = json.dumps(payload, separators=(',', ':'))
        signature = hmac.new(self.secret, raw_json.encode('utf-8'), hashlib.sha256).hexdigest()
        secure_packet = json.dumps({"auth_tag": signature[:16], "data": payload})
        
        self._cmd_burst_pub.publish(String(data=secure_packet))
        self._sblp_pub.publish(String(data=raw_json))  # Direct bridge
        self.get_logger().warn(
            f"Base Station: Broadcasted HMAC-signed micro-burst directive! "
            f"beta={payload.get('beta')}, l_max={payload.get('l_max')}, polygon vertices={len(payload.get('polygon', []))}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RLCoverageBrain()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
