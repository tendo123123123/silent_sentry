/**
 * @file trn_node.hpp
 * @brief ROS 2 Lifecycle Node wrapping the Terrain-Referenced Navigation (TRN) core.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#ifndef UGV_TRN__TRN_NODE_HPP_
#define UGV_TRN__TRN_NODE_HPP_

#include "ugv_trn/trn_core.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <silent_sentry_interfaces/msg/local_dem.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <memory>
#include <string>

namespace ugv_trn {

/**
 * @class TRNNode
 * @brief ROS 2 Lifecycle Node coordinating global TRN MCL estimation.
 * 
 * Implements Layer 1 of the TRN system. Its responsibilities are strictly limited to:
 * 1. Handling ROS 2 lifecycle states (Configure, Activate, Deactivate, Cleanup, Shutdown).
 * 2. Converting incoming LocalDEM messages and EKF odometry into clean Eigen structures for Layer 2.
 * 3. Listening to the active map->odom coordinate transform from the TF tree.
 * 4. Triggering the particle filter match cycle at 3Hz and publishing global corrections on high confidence.
 */
class TRNNode : public rclcpp_lifecycle::LifecycleNode {
public:
    /**
     * @brief Construct a new TRNNode lifecycle node.
     * @param options ROS 2 node options.
     */
    explicit TRNNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    /**
     * @brief Destructor.
     */
    ~TRNNode() override = default;

    /**
     * @brief Lifecycle configure callback.
     * 
     * Parses parameter configuration, preallocates particle filter memory,
     * constructs and loads the offline global reference DEM, and declares publishers and subscribers.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle activate callback.
     * 
     * Registers sensor subscribers, activates lifecycle publishers, and starts the 3Hz match timer.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle deactivate callback.
     * 
     * Closes subscribers, disables transitional publishers, and halts match timers.
     */
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State& state) override;

    /**
     * @brief Lifecycle cleanup callback.
     * 
     * Releases active matching core buffers and structures.
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
     * @brief High-frequency /odometry/filtered callback.
     * @param msg Filtered dead-reckoning odometry.
     * 
     * Calculates incremental displacement and rotation deltas since the last callback,
     * and propagates the particle cloud on the SE(2) manifold inside the math core.
     */
    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);

    /**
     * @brief High-speed local scan elevation map callback.
     * @param msg LocalDEM height grid message.
     * 
     * Extracts and reformats the 1D flat float array into a dense Eigen::MatrixXf local DEM
     * centered on the sensor, mapping no-data pixels to NaN, and passes it to the TRNCore.
     */
    void local_dem_callback(const silent_sentry_interfaces::msg::LocalDEM::ConstSharedPtr msg);

    /**
     * @brief Periodic 3Hz TRN match cycle timer callback.
     * 
     * Polls the TF tree to retrieve the active map->odom transform, evaluates particle 
     * likelihood scores across the local DEM, executes systematic resampling, and 
     * publishes a high-confidence global correction pose and covariance.
     */
    void match_timer_callback();

    // Core SLAM Object (Layer 2)
    std::unique_ptr<TRNCore> trn_core_;              ///< Pure C++ mathematical particle filter engine

    // ROS 2 Communication Handles
    rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr correction_pub_; ///< Pose correction publisher
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64>::SharedPtr quality_pub_;                          ///< Match quality publisher

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;                                           ///< High-rate EKF subscription
    rclcpp::Subscription<silent_sentry_interfaces::msg::LocalDEM>::SharedPtr local_dem_sub_;                      ///< Local scan subscription

    rclcpp::TimerBase::SharedPtr match_timer_;                                                                    ///< 3Hz match execution timer

    // TF Listeners (to obtain active map->odom transforms)
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // Incremental Motion Accumulation State
    double last_odom_time_;                           ///< Timestamp of last EKF msg (seconds)
    double last_odom_x_;                              ///< Last EKF position X (meters)
    double last_odom_y_;                              ///< Last EKF position Y (meters)
    double last_odom_yaw_;                            ///< Last EKF yaw (radians)
    bool odom_initialized_;                           ///< Warm-start EKF tracker

    // Frame Names
    std::string map_frame_;                           ///< Global map frame name ("map")
    std::string odom_frame_;                          ///< Drifting local frame name ("odom")
    std::string base_frame_;                          ///< Vehicle body frame name ("base_footprint")
};

} // namespace ugv_trn

#endif // UGV_TRN__TRN_NODE_HPP_
