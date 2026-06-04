/**
 * @file fuser_node.cpp
 * @brief High-frequency SE(3) pose estimation ROS 2 lifecycle node implementation.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#include "ugv_estimation/fuser_node.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <lifecycle_msgs/msg/state.hpp>

#include <chrono>

namespace ugv_estimation {

FuserNode::FuserNode(const rclcpp::NodeOptions& options)
    : rclcpp_lifecycle::LifecycleNode("fuser_node", options),
      last_imu_time_(0.0),
      last_wheel_time_(0.0),
      imu_initialized_(false),
      wheel_initialized_(false),
      accum_wheel_ds_(0.0),
      accum_wheel_dtheta_(0.0),
      accum_wheel_dt_(0.0),
      last_wheel_vx_(0.0),
      accum_wheel_accel_x_(0.0),
      accum_imu_accel_x_(0.0),
      accum_accel_count_(0),
      map_frame_("map"),
      odom_frame_("odom"),
      base_frame_("base_footprint")
{
    // Declare Configuration Parameters with defaults
    this->declare_parameter<double>("imu_accel_noise", 0.35);
    this->declare_parameter<double>("imu_gyro_noise", 0.08);
    this->declare_parameter<double>("imu_integration_noise", 0.01);
    this->declare_parameter<double>("imu_accel_bias_noise", 0.005);
    this->declare_parameter<double>("imu_gyro_bias_noise", 0.001);

    this->declare_parameter<double>("initial_pose_sigma", 1e-4);
    this->declare_parameter<double>("initial_vel_sigma", 1e-4);
    this->declare_parameter<double>("initial_bias_sigma", 1e-4);

    this->declare_parameter<double>("slip_accel_threshold", 1.5);
    this->declare_parameter<double>("slip_cov_multiplier", 100.0);

    this->declare_parameter<std::vector<double>>("odom_sigmas", {0.05, 0.05, 0.01, 0.10, 0.25, 0.25});

    this->declare_parameter<std::string>("map_frame", "map");
    this->declare_parameter<std::string>("odom_frame", "odom");
    this->declare_parameter<std::string>("base_frame", "base_footprint");
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FuserNode::on_configure(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "FuserNode [on_configure]: Parsing configuration parameters.");

    // Retrieve Parameters
    FuserConfig config;
    this->get_parameter("imu_accel_noise", config.imu_accel_noise);
    this->get_parameter("imu_gyro_noise", config.imu_gyro_noise);
    this->get_parameter("imu_integration_noise", config.imu_integration_noise);
    this->get_parameter("imu_accel_bias_noise", config.imu_accel_bias_noise);
    this->get_parameter("imu_gyro_bias_noise", config.imu_gyro_bias_noise);
    this->get_parameter("initial_pose_sigma", config.initial_pose_sigma);
    this->get_parameter("initial_vel_sigma", config.initial_vel_sigma);
    this->get_parameter("initial_bias_sigma", config.initial_bias_sigma);
    this->get_parameter("slip_accel_threshold", config.slip_accel_threshold);
    this->get_parameter("slip_cov_multiplier", config.slip_cov_multiplier);

    std::vector<double> sigmas;
    this->get_parameter("odom_sigmas", sigmas);
    if (sigmas.size() == 6) {
        config.odom_sigmas << sigmas[0], sigmas[1], sigmas[2], sigmas[3], sigmas[4], sigmas[5];
    } else {
        RCLCPP_ERROR(get_logger(), "Odom sigmas size mismatch! Expected 6, got %zu. Using default values.", sigmas.size());
        config.odom_sigmas << 0.05, 0.05, 0.01, 0.10, 0.25, 0.25;
    }

    this->get_parameter("map_frame", map_frame_);
    this->get_parameter("odom_frame", odom_frame_);
    this->get_parameter("base_frame", base_frame_);

    // Allocate core estimation object (Layer 2)
    fuser_ = std::make_unique<SE3FuserCore>(config);

    // Setup Publishers
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/filtered", 10);

    // Setup TF Broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    RCLCPP_INFO(get_logger(), "FuserNode [on_configure]: Successfully configured.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FuserNode::on_activate(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "FuserNode [on_activate]: Transitioning to active state.");

    // Warm-start core solver
    fuser_->initialize_graph();

    // Initialize map->odom transform to identity
    latest_map_to_odom_ = gtsam::Pose3();

    // Reset accumulators
    accum_wheel_ds_ = 0.0;
    accum_wheel_dtheta_ = 0.0;
    accum_wheel_dt_ = 0.0;
    accum_wheel_accel_x_ = 0.0;
    accum_imu_accel_x_ = 0.0;
    accum_accel_count_ = 0;
    imu_initialized_ = false;
    wheel_initialized_ = false;

    // Active lifecycle publishers
    odom_pub_->on_activate();

    // Broadcast initial map -> odom TF of identity to connect TF tree on startup
    geometry_msgs::msg::TransformStamped initial_tf;
    initial_tf.header.stamp = this->get_clock()->now();
    initial_tf.header.frame_id = map_frame_;
    initial_tf.child_frame_id = odom_frame_;
    initial_tf.transform.translation.x = 0.0;
    initial_tf.transform.translation.y = 0.0;
    initial_tf.transform.translation.z = 0.0;
    initial_tf.transform.rotation.x = 0.0;
    initial_tf.transform.rotation.y = 0.0;
    initial_tf.transform.rotation.z = 0.0;
    initial_tf.transform.rotation.w = 1.0;
    tf_broadcaster_->sendTransform(initial_tf);

    // Setup Topic Subscriptions
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data_filtered", 10, std::bind(&FuserNode::imu_callback, this, std::placeholders::_1)
    );

    wheel_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/terramechanic_odom", 10, std::bind(&FuserNode::wheel_callback, this, std::placeholders::_1)
    );

    trn_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/trn/global_correction", 10, std::bind(&FuserNode::trn_callback, this, std::placeholders::_1)
    );

    // Setup 50Hz publishing timer
    pub_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(20), std::bind(&FuserNode::publish_odometry, this)
    );

    RCLCPP_INFO(get_logger(), "FuserNode [on_activate]: Active and subscribing.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FuserNode::on_deactivate(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "FuserNode [on_deactivate]: Deactivating subscriptions and publishers.");

    // Tear down subscribers
    imu_sub_.reset();
    wheel_sub_.reset();
    trn_sub_.reset();

    // Tear down timers
    pub_timer_.reset();

    // Deactivate publishers
    odom_pub_->on_deactivate();

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FuserNode::on_cleanup(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "FuserNode [on_cleanup]: Resetting core estimator.");
    fuser_.reset();
    odom_pub_.reset();
    tf_broadcaster_.reset();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FuserNode::on_shutdown(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "FuserNode [on_shutdown]: Shuting down.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

void FuserNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
    const double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    if (!imu_initialized_) {
        last_imu_time_ = timestamp;
        imu_initialized_ = true;
        return;
    }

    const double dt = timestamp - last_imu_time_;
    last_imu_time_ = timestamp;

    if (dt <= 0.0 || dt > 0.1) {
        return;
    }

    // Convert ROS Imu vectors to Eigen structures (Layer 1 -> Layer 2 boundary)
    Eigen::Vector3d accel(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    Eigen::Vector3d gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

    // Call mathematical core
    fuser_->add_imu_measurement(dt, accel, gyro);

    // Accumulate IMU linear acceleration statistics for slip filter
    accum_imu_accel_x_ += msg->linear_acceleration.x;
}

void FuserNode::wheel_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
    const double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    if (!wheel_initialized_) {
        last_wheel_time_ = timestamp;
        last_wheel_vx_ = msg->twist.twist.linear.x;
        wheel_initialized_ = true;
        return;
    }

    const double dt = timestamp - last_wheel_time_;
    last_wheel_time_ = timestamp;

    if (dt <= 0.0 || dt > 0.2) {
        return;
    }

    const double vx = msg->twist.twist.linear.x;
    const double omega_z = msg->twist.twist.angular.z;

    // Kinematic integration of displacement and yaw rotation (tangent vector components)
    const double ds = vx * dt;
    const double dtheta = omega_z * dt;

    accum_wheel_ds_ += ds;
    accum_wheel_dtheta_ += dtheta;
    accum_wheel_dt_ += dt;

    // Compute longitudinal acceleration from wheel encoders
    const double wheel_accel = (vx - last_wheel_vx_) / dt;
    last_wheel_vx_ = vx;

    // Accumulate statistics for the slip-gating filter
    accum_wheel_accel_x_ += wheel_accel;
    accum_accel_count_++;
}

void FuserNode::trn_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
{
    RCLCPP_INFO(get_logger(), "FuserNode [trn_callback]: Received global TRN correction.");

    // Convert ROS Pose definition to GTSAM Pose3
    const auto& pos = msg->pose.pose.position;
    const auto& ori = msg->pose.pose.orientation;
    
    gtsam::Rot3 trn_rot = gtsam::Rot3::Quaternion(ori.w, ori.x, ori.y, ori.z);
    gtsam::Point3 trn_pos(pos.x, pos.y, pos.z);
    gtsam::Pose3 trn_pose(trn_rot, trn_pos);

    // Convert 6x6 Row-major covariance matrix to Eigen::Matrix (ROS order: [Trans, Rot])
    Eigen::Matrix<double, 6, 6> ros_cov = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(msg->pose.covariance.data());

    // Swap 3x3 blocks to map ROS [Trans, Rot] to GTSAM [Rot, Trans]
    // GTSAM tangent space ordering: [Rotation, Translation]
    // ROS 2 PoseWithCovariance tangent space ordering: [Translation, Rotation]
    Eigen::Matrix<double, 6, 6> trn_covariance;
    trn_covariance.block<3,3>(0,0).noalias() = ros_cov.block<3,3>(3,3); // ROS Rot->Rot to GTSAM Rot->Rot
    trn_covariance.block<3,3>(3,3).noalias() = ros_cov.block<3,3>(0,0); // ROS Trans->Trans to GTSAM Trans->Trans
    trn_covariance.block<3,3>(0,3).noalias() = ros_cov.block<3,3>(3,0); // ROS Rot->Trans to GTSAM Rot->Trans
    trn_covariance.block<3,3>(3,0).noalias() = ros_cov.block<3,3>(0,3); // ROS Trans->Rot to GTSAM Trans->Rot

    // Extract statistics and average them over the epoch interval
    double mean_wheel_accel = 0.0;
    double mean_imu_accel = 0.0;
    if (accum_accel_count_ > 0) {
        mean_wheel_accel = accum_wheel_accel_x_ / static_cast<double>(accum_accel_count_);
        mean_imu_accel = accum_imu_accel_x_ / static_cast<double>(accum_accel_count_);
    }

    // Call mathematical core to execute iSAM2 optimization update
    gtsam::Pose3 map_to_odom = fuser_->add_global_correction(
        trn_pose,
        trn_covariance,
        accum_wheel_ds_,
        accum_wheel_dtheta_,
        accum_wheel_dt_,
        mean_wheel_accel,
        mean_imu_accel
    );

    // Cache latest map->odom transform
    latest_map_to_odom_ = map_to_odom;

    // Broadcast the single authoritative map -> odom TF immediately as well
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = msg->header.stamp;
    tf_msg.header.frame_id = map_frame_;
    tf_msg.child_frame_id = odom_frame_;

    const auto& t = map_to_odom.translation();
    const auto& q = map_to_odom.rotation().toQuaternion();

    tf_msg.transform.translation.x = t.x();
    tf_msg.transform.translation.y = t.y();
    tf_msg.transform.translation.z = t.z();
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(tf_msg);

    // Reset accumulators for the next global optimization epoch
    accum_wheel_ds_ = 0.0;
    accum_wheel_dtheta_ = 0.0;
    accum_wheel_dt_ = 0.0;
    accum_wheel_accel_x_ = 0.0;
    accum_imu_accel_x_ = 0.0;
    accum_accel_count_ = 0;
}

void FuserNode::publish_odometry()
{
    if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
        return;
    }

    // Get current high-frequency dead-reckoned state from math core
    gtsam::Pose3 pose = fuser_->get_current_pose();
    Eigen::Vector3d velocity = fuser_->get_current_velocity();

    rclcpp::Time now = this->get_clock()->now();

    // Publish high-frequency filtered Odometry
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = now;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = base_frame_;

    const auto& t = pose.translation();
    const auto& q = pose.rotation().toQuaternion();

    odom_msg.pose.pose.position.x = t.x();
    odom_msg.pose.pose.position.y = t.y();
    odom_msg.pose.pose.position.z = t.z();
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();

    odom_msg.twist.twist.linear.x = velocity.x();
    odom_msg.twist.twist.linear.y = velocity.y();
    odom_msg.twist.twist.linear.z = velocity.z();

    odom_pub_->publish(odom_msg);

    // Broadcast continuous high-frequency odom -> base TF
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = now;
    tf_msg.header.frame_id = odom_frame_;
    tf_msg.child_frame_id = base_frame_;

    tf_msg.transform.translation.x = t.x();
    tf_msg.transform.translation.y = t.y();
    tf_msg.transform.translation.z = t.z();
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(tf_msg);

    // Broadcast authoritative map -> odom TF continuously at 50Hz to keep TF tree alive and in sync
    geometry_msgs::msg::TransformStamped map_tf;
    map_tf.header.stamp = now;
    map_tf.header.frame_id = map_frame_;
    map_tf.child_frame_id = odom_frame_;

    const auto& mt = latest_map_to_odom_.translation();
    const auto& mq = latest_map_to_odom_.rotation().toQuaternion();

    map_tf.transform.translation.x = mt.x();
    map_tf.transform.translation.y = mt.y();
    map_tf.transform.translation.z = mt.z();
    map_tf.transform.rotation.x = mq.x();
    map_tf.transform.rotation.y = mq.y();
    map_tf.transform.rotation.z = mq.z();
    map_tf.transform.rotation.w = mq.w();

    tf_broadcaster_->sendTransform(map_tf);
}

} // namespace ugv_estimation

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ugv_estimation::FuserNode>();
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
