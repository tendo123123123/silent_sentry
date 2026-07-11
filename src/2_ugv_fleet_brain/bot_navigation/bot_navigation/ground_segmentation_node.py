#!/usr/bin/env python3
"""
Ground Segmentation Node
========================
Converts the 3D LiDAR cloud into an OBSTACLE-ONLY cloud for Nav2, removing
drivable sloped terrain so dunes are not mistaken for walls.

Pipeline:
  1. Subscribe /scan/points (3D PointCloud2, sensor frame).
  2. Transform points into the gravity-aligned odom frame via TF (so slope is
     a smooth height rise, not tilted points).
  3. Run grid-based local-ground removal (ground_segmentation.segment_obstacles).
  4. Publish the surviving obstacle points on /scan/obstacles (PointCloud2,
     odom frame) for the Nav2 costmap voxel/obstacle layer.

Because only above-ground points survive, the Nav2 costmap sees rocks and
vegetation but NOT the slope itself. Terrain-slope avoidance is handled
separately by the a-priori terrain static layer (the slope costmap).

Subscribes:  /scan/points (sensor_msgs/PointCloud2)
Publishes:   /scan/obstacles (sensor_msgs/PointCloud2, in odom frame)
"""
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import tf2_ros
from tf_transformations import quaternion_matrix

from bot_navigation.ground_segmentation import segment_obstacles


class GroundSegmentationNode(Node):
    def __init__(self):
        super().__init__('ground_segmentation_node')

        self.declare_parameter('input_topic', '/scan/points')
        self.declare_parameter('output_topic', '/scan/obstacles')
        self.declare_parameter('target_frame', 'odom')  # gravity-aligned
        self.declare_parameter('cell_size', 0.4)
        self.declare_parameter('height_threshold', 0.4)
        self.declare_parameter('min_height', -1.0)
        self.declare_parameter('max_height', 2.0)
        self.declare_parameter('min_points_per_cell', 2)
        self.declare_parameter('max_range', 30.0)

        gp = self.get_parameter
        self.target_frame = gp('target_frame').value
        self.cell_size = gp('cell_size').value
        self.height_threshold = gp('height_threshold').value
        self.min_height = gp('min_height').value
        self.max_height = gp('max_height').value
        self.min_points_per_cell = gp('min_points_per_cell').value
        self.max_range = gp('max_range').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointCloud2, gp('input_topic').value, self._cloud_cb, 10)
        self.pub = self.create_publisher(PointCloud2, gp('output_topic').value, 10)
        self.get_logger().info(
            f'Ground segmentation ready: {gp("input_topic").value} -> '
            f'{gp("output_topic").value} (frame {self.target_frame}, '
            f'cell={self.cell_size}m, thresh={self.height_threshold}m)')

    def _cloud_cb(self, msg: PointCloud2):
        # Read x,y,z (skip NaNs).
        pts = point_cloud2.read_points_numpy(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if pts.shape[0] == 0:
            return

        # Range gate (drop far returns that add noise to ground estimation).
        rng = np.linalg.norm(pts[:, :2], axis=1)
        pts = pts[rng <= self.max_range]
        if pts.shape[0] == 0:
            return

        # Transform to the gravity-aligned target frame.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id,
                rclpy.time.Time())  # latest available
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            self.get_logger().warn(
                f'No TF {self.target_frame} <- {msg.header.frame_id}; skipping cloud',
                throttle_duration_sec=5.0)
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        mat = quaternion_matrix([q.x, q.y, q.z, q.w])
        mat[0:3, 3] = [t.x, t.y, t.z]
        homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
        world = (mat @ homog.T).T[:, :3]

        # Classify in the gravity-aligned frame (slope-robust)...
        mask = segment_obstacles(
            world,
            cell_size=self.cell_size,
            height_threshold=self.height_threshold,
            min_height=self.min_height,
            max_height=self.max_height,
            min_points_per_cell=self.min_points_per_cell)

        # ...but publish the surviving points in the ORIGINAL sensor frame, so
        # the Nav2 costmap raytraces clearing from the true sensor origin.
        obstacles = pts[mask]
        out = point_cloud2.create_cloud_xyz32(msg.header, obstacles.tolist())
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GroundSegmentationNode()
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
