/**
 * @file dem_builder_node.hpp
 * @brief ROS 2 Lifecycle Node wrapping the rolling local DEM builder core.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#ifndef UGV_LOCAL_DEM__DEM_BUILDER_NODE_HPP_
#define UGV_LOCAL_DEM__DEM_BUILDER_NODE_HPP_

#include "ugv_local_dem/dem_builder_core.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <silent_sentry_interfaces/msg/local_dem.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <memory>
#include <string>

namespace ugv_local_dem {

/**
 * @class LocalDEMBuilderNode
 * @brief ROS 2 Lifecycle Node coordinating point-cloud processing and LocalDEM rasterization.
 * 
 * Implements Layer 1 of the local DEM pipeline. Its responsibilities are strictly limited to:
 * 1. Handling ROS 2 lifecycle states (Configure, Activate, Deactivate, Cleanup, Shutdown).
 * 2. Listening to TF tree to transform LiDAR scans into the vehicle body frame and odometry frame.
 * 3. Extracting PointCloud2 binary payloads using standard high-speed iterators into Eigen matrices.
 * 4. Triggering the grid rasterization loop at 2Hz and publishing typed LocalDEM messages.
 */
class LocalDEMBuilderNode : public rclcpp_lifecycle::LifecycleNode {
public:
    /**
     * @brief Construct a new LocalDEMBuilderNode lifecycle node.
     * @param options ROS 2 node options.
     */
    explicit LocalDEMBuilderNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    /**
     * @brief Destructor.
     */
    ~LocalDEMBuilderNode() override = default;

    /**
     * @brief Lifecycle configure callback.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle activate callback.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle deactivate callback.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle cleanup callback.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_cleanup(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle shutdown callback.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_shutdown(const rclcpp_lifecycle::State& state) override;

private:
    /**
     * @brief High-frequency point cloud subscriber callback.
     */
    void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

    /**
     * @brief High-frequency IMU orientation subscriber callback.
     */
    void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);

    /**
     * @brief High-frequency EKF /odometry/filtered subscriber callback.
     */
    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);

    /**
     * @brief Low-frequency timer callback to rasterize and publish the DEM.
     */
    void publish_timer_callback();

    // Core Pipeline Object (Layer 2)
    std::unique_ptr<DEMBuilderCore> dem_core_;             ///< Pure C++ pipeline engine

    // ROS 2 Communication Handles
    rclcpp_lifecycle::LifecyclePublisher<silent_sentry_interfaces::msg::LocalDEM>::SharedPtr dem_pub_; ///< LocalDEM typed publisher
    rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_pub_;     ///< Legacy OccupancyGrid publisher

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;                        ///< LiDAR PointCloud2 subscription
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;                                  ///< IMU subscription
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;                                ///< Odom subscription

    rclcpp::TimerBase::SharedPtr publish_timer_;                                                      ///< 2Hz publisher timer

    // TF Listeners (to obtain active sensor->base and base->odom transforms)
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // Tracker States
    Eigen::Vector3f robot_pose_odom_;                                                                 ///< Active (x, y, yaw) pose in odom frame
    double odom_travel_;                                                                              ///< Cumulative travel distance (meters)
    bool odom_initialized_;                                                                           ///< Warm-start flag

    // Frame Names
    std::string base_frame_;                                                                          ///< Vehicle body frame name ("base_footprint")
    std::string odom_frame_;                                                                          ///< Local drifting odom frame name ("odom")
    std::string lidar_topic_;                                                                         ///< Scan topic name ("_scan_points")
};

} // namespace ugv_local_dem

#endif // UGV_LOCAL_DEM__DEM_BUILDER_NODE_HPP_
