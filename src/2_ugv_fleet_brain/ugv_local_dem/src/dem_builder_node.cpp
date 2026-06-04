/**
 * @file dem_builder_node.cpp
 * @brief ROS 2 Lifecycle Node wrapping the local DEM builder core.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#include "ugv_local_dem/dem_builder_node.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <Eigen/Geometry>

#include <chrono>

namespace ugv_local_dem {

LocalDEMBuilderNode::LocalDEMBuilderNode(const rclcpp::NodeOptions& options)
    : rclcpp_lifecycle::LifecycleNode("local_dem_builder", options),
      odom_travel_(0.0),
      odom_initialized_(false),
      base_frame_("base_footprint"),
      odom_frame_("odom"),
      lidar_topic_("/scan/points")
{
    // Declare Configuration Parameters with defaults matching the Python builder
    this->declare_parameter<double>("grid_resolution", 1.0);
    this->declare_parameter<double>("grid_size_x", 20.0);
    this->declare_parameter<double>("grid_size_y", 20.0);
    this->declare_parameter<std::string>("lidar_topic", "/scan/points");
    this->declare_parameter<std::string>("base_frame", "base_footprint");
    this->declare_parameter<std::string>("odom_frame", "odom");
    this->declare_parameter<double>("publish_rate", 2.0);
    this->declare_parameter<int>("cloud_queue_size", 20);

    this->declare_parameter<double>("deskew_scan_period", 0.1);
    this->declare_parameter<bool>("deskew_clockwise", false);
    this->declare_parameter<double>("rolling_submap_distance", 50.0);
    this->declare_parameter<double>("submap_spatial_bin_size", 5.0);
    this->declare_parameter<double>("uamc_drift_variance", 0.01);

    this->declare_parameter<double>("ground_height_min", -0.5);
    this->declare_parameter<double>("ground_height_max", 1.5);
    this->declare_parameter<double>("obstacle_height_threshold", 0.4);
    this->declare_parameter<double>("ransac_distance_threshold", 0.15);
    this->declare_parameter<int>("ransac_iterations", 50);
    this->declare_parameter<int>("min_points_per_cell", 1);

    this->declare_parameter<double>("min_range", 0.5);
    this->declare_parameter<double>("max_range", 30.0);
    this->declare_parameter<double>("spawn_elevation", 0.0);
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LocalDEMBuilderNode::on_configure(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "LocalDEMBuilderNode [on_configure]: Parsing configuration parameters.");

    // Retrieve Parameters
    DEMBuilderConfig config;
    this->get_parameter("grid_resolution", config.grid_resolution);
    this->get_parameter("grid_size_x", config.grid_size_x);
    this->get_parameter("grid_size_y", config.grid_size_y);
    this->get_parameter("deskew_scan_period", config.deskew_scan_period);
    this->get_parameter("deskew_clockwise", config.deskew_clockwise);
    this->get_parameter("rolling_submap_distance", config.rolling_submap_distance);
    this->get_parameter("submap_spatial_bin_size", config.submap_spatial_bin_size);
    this->get_parameter("uamc_drift_variance", config.uamc_drift_variance);
    this->get_parameter("ground_height_min", config.ground_height_min);
    this->get_parameter("ground_height_max", config.ground_height_max);
    this->get_parameter("obstacle_height_threshold", config.obstacle_height_threshold);
    this->get_parameter("ransac_distance_threshold", config.ransac_distance_threshold);
    this->get_parameter("ransac_iterations", config.ransac_iterations);
    this->get_parameter("min_points_per_cell", config.min_points_per_cell);
    this->get_parameter("min_range", config.min_range);
    this->get_parameter("max_range", config.max_range);
    this->get_parameter("spawn_elevation", config.spawn_elevation);

    this->get_parameter("base_frame", base_frame_);
    this->get_parameter("odom_frame", odom_frame_);
    this->get_parameter("lidar_topic", lidar_topic_);

    // Instantiate pure C++ pipeline core (Layer 2)
    dem_core_ = std::make_unique<DEMBuilderCore>(config);

    // Setup Publishers
    dem_pub_ = this->create_publisher<silent_sentry_interfaces::msg::LocalDEM>("/elevation_map/local_dem", 10);
    occupancy_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/elevation_map/local", 10);

    // Setup TF Listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    RCLCPP_INFO(get_logger(), "LocalDEMBuilderNode [on_configure]: Successfully configured.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LocalDEMBuilderNode::on_activate(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "LocalDEMBuilderNode [on_activate]: Activating communications.");

    odom_travel_ = 0.0;
    odom_initialized_ = false;
    robot_pose_odom_ = Eigen::Vector3f::Zero();

    // Activate lifecycle publishers
    dem_pub_->on_activate();
    occupancy_pub_->on_activate();

    // Setup Topic Subscribers
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data_filtered", 10, std::bind(&LocalDEMBuilderNode::imu_callback, this, std::placeholders::_1)
    );

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", 10, std::bind(&LocalDEMBuilderNode::odom_callback, this, std::placeholders::_1)
    );

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, 10, std::bind(&LocalDEMBuilderNode::cloud_callback, this, std::placeholders::_1)
    );

    double publish_rate = 2.0;
    this->get_parameter("publish_rate", publish_rate);

    // Setup 2Hz (default) publisher timer
    publish_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
        std::bind(&LocalDEMBuilderNode::publish_timer_callback, this)
    );

    RCLCPP_INFO(get_logger(), "LocalDEMBuilderNode [on_activate]: Successfully activated.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LocalDEMBuilderNode::on_deactivate(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "LocalDEMBuilderNode [on_deactivate]: Deactivating.");

    cloud_sub_.reset();
    imu_sub_.reset();
    odom_sub_.reset();
    publish_timer_.reset();

    dem_pub_->on_deactivate();
    occupancy_pub_->on_deactivate();

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LocalDEMBuilderNode::on_cleanup(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "LocalDEMBuilderNode [on_cleanup]: Cleaning up core resources.");
    dem_core_.reset();
    dem_pub_.reset();
    occupancy_pub_.reset();
    tf_buffer_.reset();
    tf_listener_.reset();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LocalDEMBuilderNode::on_shutdown(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "LocalDEMBuilderNode [on_shutdown]: Shutting down.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

void LocalDEMBuilderNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
    // Extract roll/pitch orientational drift from Gravity-aligned IMU
    const auto& q = msg->orientation;
    
    double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    double roll = std::atan2(sinr_cosp, cosr_cosp);

    double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    double pitch = 0.0;
    if (std::abs(sinp) >= 1.0) {
        pitch = std::copysign(M_PI / 2.0, sinp);
    } else {
        pitch = std::asin(sinp);
    }

    // Call mathematical core to update gravity angle matrices (Layer 1 -> Layer 2 boundary)
    dem_core_->update_imu_orientation(roll, pitch);

    // Also update body-frame linear and angular velocity vectors for motion deskewing
    Eigen::Vector3f linear_vel(
        static_cast<float>(msg->linear_acceleration.x), 
        static_cast<float>(msg->linear_acceleration.y), 
        static_cast<float>(msg->linear_acceleration.z)
    );
    Eigen::Vector3f angular_vel(
        static_cast<float>(msg->angular_velocity.x), 
        static_cast<float>(msg->angular_velocity.y), 
        static_cast<float>(msg->angular_velocity.z)
    );

    // Fast-pass to core
    dem_core_->update_body_velocity(linear_vel, angular_vel);
}

void LocalDEMBuilderNode::odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
    const float cx = msg->pose.pose.position.x;
    const float cy = msg->pose.pose.position.y;
    
    // Extract yaw angle
    const auto& q = msg->pose.pose.orientation;
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    const float cyaw = static_cast<float>(std::atan2(siny_cosp, cosy_cosp));

    robot_pose_odom_ = Eigen::Vector3f(cx, cy, cyaw);

    if (!odom_initialized_) {
        odom_travel_ = 0.0;
        odom_initialized_ = true;
        return;
    }

    // Capture simple planar travel delta increments
    static float last_x = cx;
    static float last_y = cy;
    float dist = std::hypot(cx - last_x, cy - last_y);
    if (std::isfinite(dist)) {
        odom_travel_ += dist;
    }
    last_x = cx;
    last_y = cy;
}

void LocalDEMBuilderNode::cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
    if (!odom_initialized_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "DEM: Waiting for EKF odometry warm-start.");
        return;
    }

    // Create high-speed standard ROS point cloud iterators (zero memory copies, fast index lookups)
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    std::vector<Eigen::Vector3f> raw_points;
    raw_points.reserve(msg->width * msg->height);

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        if (std::isfinite(*iter_x) && std::isfinite(*iter_y) && std::isfinite(*iter_z)) {
            raw_points.push_back(Eigen::Vector3f(*iter_x, *iter_y, *iter_z));
        }
    }

    if (raw_points.empty()) {
        return;
    }

    // Lookup physical transform: lidar physical frame -> chassis base_footprint frame
    geometry_msgs::msg::TransformStamped tf_sensor_to_base;
    try {
        tf_sensor_to_base = tf_buffer_->lookupTransform(base_frame_, msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.05));
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "DEM: LiDAR to body TF lookup failed: %s", ex.what());
        return;
    }

    Eigen::Vector3f sensor_origin(
        tf_sensor_to_base.transform.translation.x,
        tf_sensor_to_base.transform.translation.y,
        tf_sensor_to_base.transform.translation.z
    );
    Eigen::Quaternionf sensor_rot(
        tf_sensor_to_base.transform.rotation.w,
        tf_sensor_to_base.transform.rotation.x,
        tf_sensor_to_base.transform.rotation.y,
        tf_sensor_to_base.transform.rotation.z
    );

    // Convert flat arrays to aligned base-frame Eigen points matrix
    Eigen::MatrixXf points_base(raw_points.size(), 3);
    for (size_t i = 0; i < raw_points.size(); ++i) {
        points_base.row(i) = sensor_rot * raw_points[i] + sensor_origin;
    }

    // Call mathematical core to process and queue (Layer 1 -> Layer 2 boundary)
    dem_core_->accumulate_ground_cloud(points_base, sensor_origin, robot_pose_odom_, odom_travel_);
}

void LocalDEMBuilderNode::publish_timer_callback()
{
    if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
        return;
    }

    if (!odom_initialized_) {
        return;
    }

    Eigen::MatrixXf local_grid;
    double origin_x = 0.0;
    double origin_y = 0.0;

    // Trigger grid generation in pure C++ core (Layer 1 -> Layer 2 boundary)
    bool success = dem_core_->build_dem(robot_pose_odom_, local_grid, origin_x, origin_y);
    if (!success) {
        return;
    }

    const int rows = local_grid.rows();
    const int cols = local_grid.cols();

    double grid_resolution = 1.0;
    this->get_parameter("grid_resolution", grid_resolution);

    // 1. Publish Typed LocalDEM message (Primary downstream consumer interface)
    silent_sentry_interfaces::msg::LocalDEM dem_msg;
    dem_msg.header.stamp = this->get_clock()->now();
    dem_msg.header.frame_id = odom_frame_;

    dem_msg.acquisition_stamp = dem_msg.header.stamp;
    dem_msg.width = cols;
    dem_msg.height = rows;
    dem_msg.resolution = static_cast<float>(grid_resolution);
    dem_msg.origin_x = origin_x;
    dem_msg.origin_y = origin_y;
    dem_msg.center_x = robot_pose_odom_.x();
    dem_msg.center_y = robot_pose_odom_.y();
    dem_msg.no_data_value = std::numeric_limits<float>::quiet_NaN();

    dem_msg.data.resize(rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dem_msg.data[r * cols + c] = local_grid(r, c);
        }
    }

    dem_pub_->publish(dem_msg);

    // 2. Publish Legacy OccupancyGrid message (Standard ROS/RViz visualizer interface)
    nav_msgs::msg::OccupancyGrid occ_msg;
    occ_msg.header = dem_msg.header;
    occ_msg.info.map_load_time = dem_msg.header.stamp;
    occ_msg.info.resolution = static_cast<float>(grid_resolution);
    occ_msg.info.width = cols;
    occ_msg.info.height = rows;
    occ_msg.info.origin.position.x = origin_x;
    occ_msg.info.origin.position.y = origin_y;
    occ_msg.info.origin.position.z = 0.0;
    occ_msg.info.origin.orientation.w = 1.0;

    occ_msg.data.resize(rows * cols, -1); // Initialize as unobserved (-1)

    // Compute bounds for elevation visualization scaling
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (int i = 0; i < rows * cols; ++i) {
        float val = dem_msg.data[i];
        if (std::isfinite(val)) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }

    const float range = max_val - min_val;
    for (int i = 0; i < rows * cols; ++i) {
        float val = dem_msg.data[i];
        if (std::isfinite(val) && range > 1e-4f) {
            // Scale heights proportionally between [0, 100] for standard Occupancy Grid grayscale maps
            occ_msg.data[i] = static_cast<int8_t>((val - min_val) / range * 100.0f);
        }
    }

    occupancy_pub_->publish(occ_msg);
}

} // namespace ugv_local_dem

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ugv_local_dem::LocalDEMBuilderNode>();
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
