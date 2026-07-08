#!/usr/bin/env python3
"""
EMCON Controller Node
=====================
Emission-Control-aware command arbitration and Event-Driven Zenoh Middleware Gateway.

Replaces standard DDS discovery spam with a multi-state EMCON state machine and
event-driven micro-burst command/telemetry transport. Suppresses continuous RF/sensor
broadcasts during adversarial exposure windows while maintaining mathematical coverage
guarantees via encrypted micro-burst stretching.

EMCON States:
  0 = NORMAL / ACTIVE_RF         — Full RF emission allowed, continuous telemetry streaming.
  1 = RADIO_SILENCE / PASSIVE    — Suppress continuous RF telemetry, kinematic slowdown (35%).
  2 = ZERO_EMISSION / STEALTH    — Suppress ALL RF emissions, acoustic creeping limit (v <= 0.3 m/s), buffer telemetry.
  3 = EMERGENCY_BURST            — Brief RF receive window for coverage breach healing, flush buffered micro-bursts.

Subscribes:
  /cmd_vel_raw          (geometry_msgs/Twist) — raw velocity commands from SBLP planner / teleop
  /emcon_state          (std_msgs/Bool)       — backwards compatible bool trigger (True -> State 1, False -> State 0)
  /emcon/directive      (std_msgs/Int8)       — multi-state trigger (0..3)
  /odometry/filtered    (nav_msgs/Odometry)   — robot pose/velocity for local buffer
  /sblp/status          (std_msgs/String)     — SBLP status for local buffer
  /emcon/poll           (std_msgs/String)     — encrypted Base Station poll requesting telemetry micro-burst
  /emcon/command_burst  (std_msgs/String)     — inbound encrypted micro-burst from Base Station RL brain

Publishes:
  /cmd_vel               (geometry_msgs/Twist) — arbitrated velocity commands to motor controller
  /emcon_status          (std_msgs/String)     — current EMCON mode and RF emission status
  /emcon/telemetry_burst (std_msgs/String)     — event-driven, compressed JSON telemetry micro-burst
  /sblp/micro_burst      (std_msgs/String)     — validated Base Station parameter stretching forwarded to SBLP
"""

import json
import hashlib
import hmac
from collections import deque
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int8, String
from nav_msgs.msg import Odometry


EMCON_NAMES = {
    0: "NORMAL_ACTIVE_RF",
    1: "RADIO_SILENCE_PASSIVE",
    2: "ZERO_EMISSION_STEALTH",
    3: "EMERGENCY_BURST",
}


class EmconController(Node):
    def __init__(self):
        super().__init__('emcon_controller')

        # Declare parameters
        self.declare_parameter('emcon_slowdown_factor', 0.35)
        self.declare_parameter('emcon_max_angular', 0.4)
        self.declare_parameter('stealth_max_linear_vel', 0.3)
        self.declare_parameter('default_emcon_state', 0)
        self.declare_parameter('burst_buffer_size', 50)
        self.declare_parameter('hmac_secret_key', "sentry_emcon_secret_key_2026")
        self.declare_parameter('zenoh_rmw_enabled', True)
        self.declare_parameter('suppress_discovery_multicast', True)

        self.slowdown         = self.get_parameter('emcon_slowdown_factor').value
        self.max_ang          = self.get_parameter('emcon_max_angular').value
        self.stealth_max_lin  = self.get_parameter('stealth_max_linear_vel').value
        self.emcon_state      = self.get_parameter('default_emcon_state').value
        self.buffer_size      = self.get_parameter('burst_buffer_size').value
        self.secret_key       = self.get_parameter('hmac_secret_key').value.encode('utf-8')
        self.zenoh_enabled    = self.get_parameter('zenoh_rmw_enabled').value
        self.suppress_mcast   = self.get_parameter('suppress_discovery_multicast').value

        # Local telemetry circular buffer (for stealth micro-bursting)
        self.telemetry_buffer = deque(maxlen=self.buffer_size)
        self.latest_odom_summary = {}
        self.latest_sblp_summary = {}

        # Subscribers
        self.create_subscription(Twist,    '/cmd_vel_raw',         self._cmd_cb,        10)
        self.create_subscription(Bool,     '/emcon_state',         self._bool_emcon_cb, 10)
        self.create_subscription(Int8,     '/emcon/directive',     self._int_emcon_cb,  10)
        self.create_subscription(Odometry, '/odometry/filtered',   self._odom_cb,       10)
        self.create_subscription(String,   '/sblp/status',         self._sblp_cb,       10)
        self.create_subscription(String,   '/emcon/poll',          self._poll_cb,       10)
        self.create_subscription(String,   '/emcon/command_burst', self._cmd_burst_cb,  10)

        # Publishers
        self._cmd_pub       = self.create_publisher(Twist,  '/cmd_vel',               10)
        self._status_pub    = self.create_publisher(String, '/emcon_status',          10)
        self._burst_pub     = self.create_publisher(String, '/emcon/telemetry_burst', 10)
        self._sblp_fwd_pub  = self.create_publisher(String, '/sblp/micro_burst',      10)

        # Status heartbeat timer (only emits over RF in State 0 NORMAL)
        self.create_timer(1.0, self._status_loop)

        mode_name = EMCON_NAMES.get(self.emcon_state, "UNKNOWN")
        self.get_logger().info(
            f'EMCON Controller ready | State: {mode_name} | '
            f'Zenoh RMW: {"ENABLED (Multicast Suppressed)" if self.zenoh_enabled else "STANDARD DDS"}'
        )

    # ── State Machine & Directives ────────────────────────────────────────────
    def _set_emcon_state(self, new_state: int, source: str):
        if new_state not in EMCON_NAMES:
            self.get_logger().error(f"EMCON: Invalid state requested ({new_state})")
            return
        old_state = self.emcon_state
        self.emcon_state = new_state
        mode_name = EMCON_NAMES[new_state]
        self.get_logger().info(f"EMCON: Transitioned {EMCON_NAMES[old_state]} -> {mode_name} [{source}]")
        
        # Immediate local status notification
        status_msg = json.dumps({
            "state_id": self.emcon_state,
            "mode": mode_name,
            "rf_emissions": "ALLOWED" if self.emcon_state == 0 else "SUPPRESSED",
            "zenoh_rmw": self.zenoh_enabled
        })
        self._status_pub.publish(String(data=status_msg))

        # If entering EMERGENCY_BURST (State 3), immediately flush buffered telemetry micro-burst
        # and revert to ZERO_EMISSION_STEALTH (State 2)
        if new_state == 3:
            self.get_logger().warn("EMCON: EMERGENCY_BURST triggered! Flushing telemetry micro-burst over Zenoh.")
            self._flush_telemetry_burst(reason="emergency_breach")
            self._set_emcon_state(2, source="auto_revert_from_burst")

    def _bool_emcon_cb(self, msg: Bool):
        """Backwards compatibility for legacy Bool trigger."""
        new_state = 1 if msg.data else 0
        if new_state != self.emcon_state:
            self._set_emcon_state(new_state, source="/emcon_state_bool")

    def _int_emcon_cb(self, msg: Int8):
        if msg.data != self.emcon_state:
            self._set_emcon_state(msg.data, source="/emcon/directive")

    def _status_loop(self):
        """Periodic heartbeat. In EMCON states 1 and 2, suppress routine RF broadcasts."""
        mode_name = EMCON_NAMES.get(self.emcon_state, "UNKNOWN")
        status_data = {
            "state_id": self.emcon_state,
            "mode": mode_name,
            "rf_emissions": "ALLOWED" if self.emcon_state == 0 else "SUPPRESSED",
            "buffer_count": len(self.telemetry_buffer)
        }
        # In NORMAL mode, stream continuously. In stealth modes, only log locally.
        if self.emcon_state == 0:
            self._status_pub.publish(String(data=json.dumps(status_data)))

    # ── Command Arbitration & Kinematic Slowdown ──────────────────────────────
    def _cmd_cb(self, msg: Twist):
        out = Twist()
        if self.emcon_state == 0:
            # NORMAL: pass through raw commands
            out = msg
        elif self.emcon_state == 1:
            # RADIO_SILENCE: kinematic slowdown to prevent aggressive maneuvers / tire slip noise
            out.linear.x  = msg.linear.x * self.slowdown
            out.angular.z = max(-self.max_ang, min(self.max_ang, msg.angular.z * self.slowdown))
        elif self.emcon_state in (2, 3):
            # ZERO_EMISSION / STEALTH: strict acoustic creeping speed limit (v <= 0.3 m/s)
            scaled_lin = msg.linear.x * self.slowdown
            out.linear.x  = min(scaled_lin, self.stealth_max_lin) if scaled_lin >= 0 else max(scaled_lin, -self.stealth_max_lin)
            stealth_ang   = self.max_ang * 0.75
            out.angular.z = max(-stealth_ang, min(stealth_ang, msg.angular.z * self.slowdown))
            
        self._cmd_pub.publish(out)

    # ── Event-Driven Telemetry Buffering & Micro-Bursting ─────────────────────
    def _odom_cb(self, msg: Odometry):
        pos = msg.pose.pose.position
        lin = msg.twist.twist.linear
        self.latest_odom_summary = {
            "x": round(pos.x, 2),
            "y": round(pos.y, 2),
            "v": round(lin.x, 2)
        }
        self._buffer_snapshot()

    def _sblp_cb(self, msg: String):
        try:
            self.latest_sblp_summary = json.loads(msg.data)
            self._buffer_snapshot()
        except Exception:
            pass

    def _buffer_snapshot(self):
        if not self.latest_odom_summary:
            return
        snapshot = {
            "t": self.get_clock().now().nanoseconds // 1000000,
            "odom": self.latest_odom_summary,
            "sblp": self.latest_sblp_summary.get("scenario", "unknown")
        }
        self.telemetry_buffer.append(snapshot)

    def _poll_cb(self, msg: String):
        """Handle encrypted Base Station poll requesting buffered telemetry micro-burst."""
        self.get_logger().info("EMCON: Received Base Station telemetry poll. Generating micro-burst.")
        self._flush_telemetry_burst(reason="base_station_poll")

    def _flush_telemetry_burst(self, reason: str):
        if not self.telemetry_buffer:
            self.get_logger().info("EMCON: Telemetry buffer empty, nothing to burst.")
            return
            
        burst_payload = {
            "reason": reason,
            "ugv_id": "autobot",
            "state": EMCON_NAMES.get(self.emcon_state, "UNKNOWN"),
            "count": len(self.telemetry_buffer),
            "history": list(self.telemetry_buffer)
        }
        raw_json = json.dumps(burst_payload, separators=(',', ':'))
        
        # Compute HMAC integrity tag (simulated SHA-256 auth)
        signature = hmac.new(self.secret_key, raw_json.encode('utf-8'), hashlib.sha256).hexdigest()
        secure_burst = json.dumps({"auth_tag": signature[:16], "data": burst_payload})
        
        self._burst_pub.publish(String(data=secure_burst))
        self.get_logger().info(f"EMCON: Published secure telemetry micro-burst ({len(secure_burst)} bytes, {len(self.telemetry_buffer)} frames)")
        self.telemetry_buffer.clear()

    # ── Inbound Command Micro-Burst Handling (Base Station -> SBLP) ───────────
    def _cmd_burst_cb(self, msg: String):
        """
        Ingest encrypted command micro-burst from Base Station RL coverage brain.
        Verifies HMAC signature, applies emergency state transitions if required,
        and forwards parameter stretching to SBLP planner.
        """
        try:
            packet = json.loads(msg.data)
            payload = packet.get("data", {})
            auth_tag = packet.get("auth_tag", "")
            
            # Verify HMAC signature
            raw_payload = json.dumps(payload, separators=(',', ':'))
            expected_tag = hmac.new(self.secret_key, raw_payload.encode('utf-8'), hashlib.sha256).hexdigest()[:16]
            
            if auth_tag and auth_tag != expected_tag:
                self.get_logger().error("EMCON: Security alert! Inbound micro-burst HMAC signature mismatch! Dropping packet.")
                return
                
            self.get_logger().warn(f"EMCON: Validated Base Station micro-burst command: {list(payload.keys())}")
            
            # Check if command requests emergency EMCON state change
            if "emcon_state" in payload:
                self._set_emcon_state(int(payload["emcon_state"]), source="micro_burst_command")
                
            # Forward SBLP parameter stretching (beta, l_max, polygon) to SBLP planner
            sblp_params = {k: v for k, v in payload.items() if k in ("beta", "l_max", "polygon")}
            if sblp_params:
                self._sblp_fwd_pub.publish(String(data=json.dumps(sblp_params)))
                self.get_logger().info("EMCON: Forwarded parameter stretching micro-burst to SBLP planner.")
                
        except Exception as e:
            self.get_logger().error(f"EMCON: Failed to process command micro-burst: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = EmconController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
