"""Shared benchmark-frame helpers for localization diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import rclpy
from rclpy.duration import Duration

import tf2_ros

from tf_transformations import euler_from_quaternion


@dataclass(frozen=True)
class LocalizedPose:
    x: float
    y: float
    yaw: float
    source: str


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(quat) -> float:
    return euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]


def rotate_into_reference(dx: float, dy: float, reference_yaw: float) -> tuple[float, float]:
    cos_ref = math.cos(-reference_yaw)
    sin_ref = math.sin(-reference_yaw)
    return dx * cos_ref - dy * sin_ref, dx * sin_ref + dy * cos_ref


def align_pose_to_reference(
    x: float,
    y: float,
    yaw: float,
    initial_x: float,
    initial_y: float,
    reference_yaw: float,
) -> tuple[float, float, float]:
    dx = x - initial_x
    dy = y - initial_y
    aligned_x, aligned_y = rotate_into_reference(dx, dy, reference_yaw)
    return aligned_x, aligned_y, wrap_angle(yaw - reference_yaw)


def lookup_localized_pose(
    tf_buffer: tf2_ros.Buffer,
    odom_x: float,
    odom_y: float,
    odom_yaw: float,
    map_frame: str = 'map',
    odom_frame: str = 'odom',
    base_frame: str = 'base_footprint',
    timeout_sec: float = 0.02,
) -> LocalizedPose | None:
    timeout = Duration(seconds=timeout_sec)

    try:
        transform = tf_buffer.lookup_transform(
            map_frame,
            base_frame,
            rclpy.time.Time(),
            timeout=timeout,
        )
        return LocalizedPose(
            x=transform.transform.translation.x,
            y=transform.transform.translation.y,
            yaw=quaternion_to_yaw(transform.transform.rotation),
            source='map_to_base',
        )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ):
        pass

    try:
        transform = tf_buffer.lookup_transform(
            map_frame,
            odom_frame,
            rclpy.time.Time(),
            timeout=timeout,
        )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ):
        return None

    map_to_odom_yaw = quaternion_to_yaw(transform.transform.rotation)
    cos_yaw = math.cos(map_to_odom_yaw)
    sin_yaw = math.sin(map_to_odom_yaw)
    return LocalizedPose(
        x=transform.transform.translation.x + cos_yaw * odom_x - sin_yaw * odom_y,
        y=transform.transform.translation.y + sin_yaw * odom_x + cos_yaw * odom_y,
        yaw=wrap_angle(map_to_odom_yaw + odom_yaw),
        source='map_to_odom_plus_odom',
    )