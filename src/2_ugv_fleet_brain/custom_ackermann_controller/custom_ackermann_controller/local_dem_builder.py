#!/usr/bin/env python3
"""ROS wrapper for the rolling local DEM frontend used by TRN."""

import math

from geometry_msgs.msg import Pose

from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, PointCloud2

from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

import tf2_ros

from tf_transformations import euler_from_quaternion

from .local_dem_pipeline import LocalDemPipelineCore
from .local_dem_types import LocalDemPipelineConfig


class LocalDEMBuilderNode(Node):
    """Own ROS I/O, TF, and message publishing for local DEM generation."""

    def __init__(self):
        """Initialize parameters, ROS interfaces, and the extracted core."""
        super().__init__('local_dem_builder')

        self.declare_parameter('grid_resolution', 1.0)
        self.declare_parameter('grid_size_x', 20.0)
        self.declare_parameter('grid_size_y', 20.0)
        self.declare_parameter('lidar_topic', '/scan/points')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('cloud_queue_size', 20)

        self.declare_parameter('deskew_scan_period', 0.1)
        self.declare_parameter('deskew_clockwise', False)
        self.declare_parameter('rolling_submap_distance', 50.0)
        self.declare_parameter('submap_spatial_bin_size', 5.0)
        self.declare_parameter('uamc_drift_variance', 0.01)

        self.declare_parameter('ground_height_min', -0.5)
        self.declare_parameter('ground_height_max', 1.5)
        self.declare_parameter('obstacle_height_threshold', 0.5)
        self.declare_parameter('ransac_distance_threshold', 0.15)
        self.declare_parameter('ransac_iterations', 50)
        self.declare_parameter('min_points_per_cell', 1)

        self.declare_parameter('min_range', 0.5)
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('spawn_elevation', 0.0)

        self.resolution = self._p('grid_resolution')
        self.size_x = self._p('grid_size_x')
        self.size_y = self._p('grid_size_y')
        self.lidar_topic = self.get_parameter(
            'lidar_topic'
        ).get_parameter_value().string_value
        self.base_frame = self.get_parameter(
            'base_frame'
        ).get_parameter_value().string_value
        self.odom_frame = self.get_parameter(
            'odom_frame'
        ).get_parameter_value().string_value
        self.publish_rate = self._p('publish_rate')
        self.cloud_queue_size = max(self._i('cloud_queue_size'), 1)

        self.scan_period = self._p('deskew_scan_period')
        self.deskew_clockwise = self._b('deskew_clockwise')
        self.rolling_submap_distance = self._p('rolling_submap_distance')
        self.submap_spatial_bin_size = max(
            self._p('submap_spatial_bin_size'),
            self.resolution,
        )
        self.uamc_drift_variance = max(
            self._p('uamc_drift_variance'),
            1e-6,
        )

        self.ground_h_min = self._p('ground_height_min')
        self.ground_h_max = self._p('ground_height_max')
        self.obstacle_h = self._p('obstacle_height_threshold')
        self.ransac_dist = self._p('ransac_distance_threshold')
        self.ransac_iters = self._i('ransac_iterations')
        self.min_pts = self._i('min_points_per_cell')

        self.min_range = self._p('min_range')
        self.max_range = self._p('max_range')
        self.spawn_elevation = self._p('spawn_elevation')

        self.nx = int(self.size_x / self.resolution)
        self.ny = int(self.size_y / self.resolution)

        self.pipeline_config = LocalDemPipelineConfig(
            resolution=self.resolution,
            size_x=self.size_x,
            size_y=self.size_y,
            cloud_queue_size=self.cloud_queue_size,
            scan_period=self.scan_period,
            deskew_clockwise=self.deskew_clockwise,
            rolling_submap_distance=self.rolling_submap_distance,
            submap_spatial_bin_size=self.submap_spatial_bin_size,
            uamc_drift_variance=self.uamc_drift_variance,
            ground_h_min=self.ground_h_min,
            ground_h_max=self.ground_h_max,
            obstacle_h=self.obstacle_h,
            ransac_dist=self.ransac_dist,
            ransac_iters=self.ransac_iters,
            min_pts=self.min_pts,
            min_range=self.min_range,
            max_range=self.max_range,
            spawn_elevation=self.spawn_elevation,
        )
        self.core = LocalDemPipelineCore(self.pipeline_config)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.occ_pub = self.create_publisher(
            OccupancyGrid,
            '/elevation_map/local',
            5,
        )
        self.float_pub = self.create_publisher(
            Float32MultiArray,
            '/elevation_map/local_float',
            5,
        )
        self.info_pub = self.create_publisher(String, '/elevation_map/info', 5)

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.cloud_callback,
            5,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data_filtered',
            self._imu_callback,
            5,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/terramechanic_odom',
            self._odom_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.build_and_publish_dem,
        )

        self.get_logger().info(
            'Local DEM Builder initialized: '
            f'{self.nx}x{self.ny} grid, '
            f'{self.resolution}m/cell, '
            f'rolling_submap={self.rolling_submap_distance:.1f}m, '
            f'spatial_bin={self.submap_spatial_bin_size:.1f}m, '
            f'deskew_scan={self.scan_period:.3f}s, '
            f'uamc_var={self.uamc_drift_variance:.4f}'
        )

    def _p(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _i(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _b(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def _lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        stamp_msg,
    ):
        if not source_frame or source_frame == target_frame:
            return None

        lookup_times = []
        if stamp_msg is not None:
            lookup_times.append(rclpy.time.Time.from_msg(stamp_msg))
        lookup_times.append(rclpy.time.Time())

        last_exc = None
        for lookup_time in lookup_times:
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    lookup_time,
                    timeout=rclpy.duration.Duration(seconds=0.05),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as exc:
                last_exc = exc

        self.get_logger().warn(
            f'Local DEM: failed TF lookup {source_frame}->{target_frame}: '
            f'{last_exc}'
        )
        return None

    @staticmethod
    def _rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-9:
            return np.eye(3, dtype=np.float32)

        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        return np.array(
            [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ],
                [
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ],
                [
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float32,
        )

    def _transform_points(
        self,
        points: np.ndarray,
        source_frame: str,
        target_frame: str,
        stamp_msg,
        return_translation: bool = False,
    ):
        if points is None or len(points) == 0:
            if return_translation:
                return points, np.zeros(3, dtype=np.float32)
            return points

        if not source_frame or source_frame == target_frame:
            if return_translation:
                return points, np.zeros(3, dtype=np.float32)
            return points

        transform = self._lookup_transform(
            target_frame,
            source_frame,
            stamp_msg,
        )
        if transform is None:
            if return_translation:
                return None, None
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        vector = np.array(
            [translation.x, translation.y, translation.z],
            dtype=np.float32,
        )
        if not np.isfinite(vector).all():
            if return_translation:
                return None, None
            return None

        quat = np.array(
            [rotation.x, rotation.y, rotation.z, rotation.w],
            dtype=np.float32,
        )
        if not np.isfinite(quat).all():
            if return_translation:
                return None, None
            return None

        rot_mat = self._rotation_matrix_from_quaternion(quat)
        transformed = points @ rot_mat.T + vector
        valid = np.isfinite(transformed).all(axis=1)
        transformed = transformed[valid]

        if return_translation:
            return transformed, vector
        return transformed

    def _imu_callback(self, msg: Imu):
        q = msg.orientation
        try:
            roll, pitch, _ = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.core.update_imu_orientation(roll, pitch)
        except (TypeError, ValueError):
            pass

        ang_vel = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=np.float32,
        )
        self.core.update_body_angular_velocity(ang_vel)

    def _odom_callback(self, msg: Odometry):
        linear_vel = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )
        self.core.update_body_linear_velocity(linear_vel)

    def cloud_callback(self, msg: PointCloud2):
        """Parse and queue incoming LiDAR sweeps for the extracted core."""
        points = self._parse_pointcloud2(msg)
        if points is None or len(points) == 0:
            return

        self.core.enqueue_cloud(points, msg.header.stamp, msg.header.frame_id)

    @staticmethod
    def _parse_pointcloud2(msg: PointCloud2) -> np.ndarray:
        field_map = {
            field.name: (field.offset, field.datatype)
            for field in msg.fields
        }
        if (
            'x' not in field_map
            or 'y' not in field_map
            or 'z' not in field_map
        ):
            return None

        x_off, x_type = field_map['x']
        y_off, y_type = field_map['y']
        z_off, z_type = field_map['z']

        if x_type != 7 or y_type != 7 or z_type != 7:
            return None

        n_points = msg.width * msg.height
        if n_points <= 0:
            return None

        dtype = np.dtype(
            {
                'names': ['x', 'y', 'z'],
                'formats': ['<f4', '<f4', '<f4'],
                'offsets': [x_off, y_off, z_off],
                'itemsize': msg.point_step,
            }
        )
        raw = np.frombuffer(msg.data, dtype=dtype, count=n_points)
        points = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(
            np.float32,
            copy=False,
        )
        valid = np.isfinite(points).all(axis=1)
        return points[valid]

    def _lookup_robot_pose_in_odom(self):
        transform = self._lookup_transform(
            self.odom_frame,
            self.base_frame,
            None,
        )
        if transform is None:
            return None

        translation = transform.transform.translation
        pose = np.array(
            [translation.x, translation.y, translation.z],
            dtype=np.float32,
        )
        if not np.isfinite(pose).all():
            return None
        return pose

    def build_and_publish_dem(self):
        """Build the latest local DEM window and publish ROS messages."""
        robot_pose = self._lookup_robot_pose_in_odom()
        if robot_pose is None:
            return

        build_output = self.core.build_dem(
            robot_pose,
            self._transform_points,
            self.base_frame,
            self.odom_frame,
        )
        if build_output is None:
            return

        elevation_grid = build_output.elevation_grid
        stamp_msg = build_output.latest_cloud_stamp
        stamp_ns = self.get_clock().now().nanoseconds
        if stamp_msg is not None:
            stamp_ns = stamp_msg.sec * 1_000_000_000 + stamp_msg.nanosec

        center_x = build_output.origin_x + (self.size_x / 2.0)
        center_y = build_output.origin_y + (self.size_y / 2.0)
        row_label = (
            'rows;'
            f'origin_x={build_output.origin_x:.6f};'
            f'origin_y={build_output.origin_y:.6f};'
            f'center_x={center_x:.6f};'
            f'center_y={center_y:.6f};'
            f'resolution={self.resolution:.6f};'
            f'stamp_ns={stamp_ns}'
        )

        float_msg = Float32MultiArray()
        float_msg.layout.dim = [
            MultiArrayDimension(
                label=row_label,
                size=self.ny,
                stride=self.ny * self.nx,
            ),
            MultiArrayDimension(
                label='cols',
                size=self.nx,
                stride=self.nx,
            ),
        ]
        publish_grid = np.where(
            np.isnan(elevation_grid),
            -9999.0,
            elevation_grid,
        )
        float_msg.data = publish_grid.ravel().tolist()
        self.float_pub.publish(float_msg)

        occ_msg = OccupancyGrid()
        occ_msg.header.stamp = (
            stamp_msg
            if stamp_msg
            else self.get_clock().now().to_msg()
        )
        occ_msg.header.frame_id = self.odom_frame

        occ_msg.info = MapMetaData()
        occ_msg.info.resolution = self.resolution
        occ_msg.info.width = self.nx
        occ_msg.info.height = self.ny
        occ_msg.info.origin = Pose()
        occ_msg.info.origin.position.x = build_output.origin_x
        occ_msg.info.origin.position.y = build_output.origin_y
        occ_msg.info.origin.position.z = 0.0

        valid_vals = elevation_grid[~np.isnan(elevation_grid)]
        if len(valid_vals) > 0:
            e_min = np.min(valid_vals)
            e_max = np.max(valid_vals)
            e_range = e_max - e_min if e_max > e_min else 1.0

            occ_data = np.full(self.ny * self.nx, -1, dtype=np.int8)
            flat_elev = elevation_grid.ravel()
            valid_flat = ~np.isnan(flat_elev)
            occ_data[valid_flat] = (
                ((flat_elev[valid_flat] - e_min) / e_range) * 100
            ).astype(np.int8)
            occ_msg.data = occ_data.tolist()
        else:
            occ_msg.data = [-1] * (self.nx * self.ny)

        self.occ_pub.publish(occ_msg)

        n_valid = int(np.sum(~np.isnan(elevation_grid)))
        n_total = self.nx * self.ny
        coverage = n_valid / n_total * 100 if n_total > 0 else 0.0
        info_msg = String()
        info_msg.data = (
            f'Rolling DEM: {n_valid}/{n_total} cells '
            f'({coverage:.1f}% coverage), '
            f'{build_output.submap_point_count} odom-frame ground pts from '
            f'{build_output.candidate_chunk_count}/'
            f'{build_output.total_chunk_count} indexed chunks, '
            f'{build_output.processed_clouds} new clouds'
        )
        self.info_pub.publish(info_msg)


def main(args=None):
    """Run the local DEM builder ROS node."""
    rclpy.init(args=args)
    node = LocalDEMBuilderNode()
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
