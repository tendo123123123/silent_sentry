/**
 * @file fuser_node.hpp
 * @brief High-frequency SE(3) pose estimation ROS 2 lifecycle node wrapping the fuser core.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#ifndef UGV_ESTIMATION__FUSER_NODE_HPP_
#define UGV_ESTIMATION__FUSER_NODE_HPP_

#include "ugv_estimation/se3_fuser_core.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <memory>
#include <string>

namespace ugv_estimation {

/**
 * @class FuserNode
 * @brief ROS 2 Lifecycle Node wrapping the SE3FuserCore.
 * 
 * Implements Layer 1 of the architecture. Its responsibilities are strictly limited to:
 * 1. Handling ROS 2 lifecycle states (Configure, Activate, Deactivate, Cleanup, Shutdown).
 * 2. Converting incoming ROS 2 sensor messages (IMU, Wheel Odom, TRN) into pure Eigen/GTSAM types.
 * 3. Broadcasting the single authoritative map -> odom TF and publishing filtered high-frequency odometry.
 * 
 * Inside this class, NO core SLAM or kinematics mathematics are computed directly; 
 * all execution is delegated to Layer 2 (SE3FuserCore).
 */
class FuserNode : public rclcpp_lifecycle::LifecycleNode {
public:
    /**
     * @brief Construct a new FuserNode lifecycle node.
     * @param options ROS 2 node options.
     */
    explicit FuserNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    /**
     * @brief Destructor. Cleans up callbacks, subscribers, and publishers.
     */
    ~FuserNode() override = default;

    /**
     * @brief Lifecycle configure callback.
     * 
     * In this stage:
     * - Parameter files are parsed.
     * - Memory is preallocated.
     * - Core math engine (SE3FuserCore) is constructed and configured.
     * - ROS Publishers, subscribers, and TF broadcasters are declared.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle activate callback.
     * 
     * In this stage:
     * - Dynamic subscribers are registered.
     * - Publishers are activated.
     * - GTSAM solver state is anchored at zero-pose.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle deactivate callback.
     * 
     * In this stage:
     * - Subscribers are torn down.
     * - Publishers are transitionally disabled.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle cleanup callback.
     * 
     * Releases preallocated fuser buffers and core handles.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_cleanup(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle shutdown callback.
     * 
     * Safely closes open communication threads and handles.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_shutdown(const rclcpp_lifecycle::State& state) override;

private:
    /**
     * @brief 400Hz IMU callback. Converts ROS message and integrates measurement in core.
     * @param msg ROS 2 sensor_msgs::msg::Imu shared pointer.
     */
    void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);

    /**
     * @brief Wheel odometry callback. Handles raw encoder tick displacement.
     * @param msg ROS 2 nav_msgs::msg::Odometry shared pointer.
     */
    void wheel_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);

    /**
     * @brief Low-frequency Terrain-Referenced Navigation (TRN) loop closure callback.
     * @param msg ROS 2 geometry_msgs::msg::PoseWithCovarianceStamped shared pointer.
     * 
     * When a match occurs, converts the TRN global pose to gtsam::Pose3, evaluates
     * the slip gate, triggers iSAM2 factor graph optimization in SE3FuserCore, and 
     * broadcasts the resulting map -> odom TF translation and rotation.
     */
    void trn_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);

    /**
     * @brief Periodically publishes 50Hz odometry filtered output.
     */
    void publish_odometry();

    // Core SLAM Object (Layer 2)
    std::unique_ptr<SE3FuserCore> fuser_;     ///< Pure C++ mathematical SLAM fuser

    // ROS 2 Communication Handles
    rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_; ///< Odometry publisher
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;                    ///<Authoritative TF broadcaster

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;                   ///< High-rate IMU subscription
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_sub_;               ///< Wheel odometry subscription
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr trn_sub_; ///< TRN prior subscription

    rclcpp::TimerBase::SharedPtr pub_timer_;                                           ///< Publishing rate timer

    // Local State Tracking for Accumulators
    double last_imu_time_;                    ///< Timestamp of last IMU msg
    double last_wheel_time_;                  ///< Timestamp of last wheel msg
    bool imu_initialized_;                     ///< IMU warm-start tracker
    bool wheel_initialized_;                   ///< Wheel warm-start tracker

    // Acceleration statistics for the slip-gating filter
    double last_wheel_vx_;                     ///< Last wheel speed measurement
    double accum_wheel_accel_x_;              ///< Rolling sum of wheel accelerations
    double accum_imu_accel_x_;                ///< Rolling sum of IMU longitudinal accelerations
    uint64_t accum_accel_count_;              ///< Number of samples in rolling acceleration window

    // Frame Names
    std::string map_frame_;                   ///< Global map frame ("map")
    std::string odom_frame_;                  ///< Drifting dead-reckoned frame ("odom")
    std::string base_frame_;                  ///< Vehicle body footprint frame ("base_footprint")

    // State Tracking
    std::mutex state_mtx_;                    ///< Mutex protecting local odometry state
    gtsam::Pose3 pure_odom_to_base_;          ///< Continuous, jump-free wheel odometry frame
    gtsam::Pose3 odom_at_last_keyframe_;      ///< Snapshot of odom frame at the last global correction
};

} // namespace ugv_estimation

#endif // UGV_ESTIMATION__FUSER_NODE_HPP_
