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
#include <csignal>
#include <execinfo.h>
#include <cstdlib>
#include <unistd.h>

// ─── SIGSEGV crash handler ───────────────────────────────────────────────
static void crash_handler(int sig)
{
    const char msg[] = "\n\n=== FUSER_NODE CRASH HANDLER ==="
                       "\nSignal: ";
    write(STDERR_FILENO, msg, sizeof(msg) - 1);

    // Write signal number
    char sigbuf[4];
    int s = sig;
    int i = 0;
    if (s >= 10) { sigbuf[i++] = '0' + (s / 10); s %= 10; }
    sigbuf[i++] = '0' + s;
    sigbuf[i++] = '\n';
    write(STDERR_FILENO, sigbuf, i);

    // Print backtrace (max 64 frames)
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);

    const char end[] = "=== END CRASH HANDLER ===\n";
    write(STDERR_FILENO, end, sizeof(end) - 1);

    // Restore default handler and re-raise to get core dump
    signal(sig, SIG_DFL);
    raise(sig);
}

namespace ugv_estimation {

FuserNode::FuserNode(const rclcpp::NodeOptions& options)
    : rclcpp_lifecycle::LifecycleNode("fuser_node", options),
      last_imu_time_(0.0),
      last_wheel_time_(0.0),
      imu_initialized_(false),
      wheel_initialized_(false),
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

    // Allocate core estimation object (Layer 2) using explicit new to preserve Eigen AVX 32-byte alignment
    fuser_.reset(new SE3FuserCore(config));

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

    {
        std::lock_guard<std::mutex> lock(state_mtx_);
        // Initialize continuous odometry frames to identity
        pure_odom_to_base_ = gtsam::Pose3();
        odom_at_last_keyframe_ = gtsam::Pose3();

        // Reset accumulators
        accum_wheel_accel_x_ = 0.0;
        accum_imu_accel_x_ = 0.0;
        accum_accel_count_ = 0;
        imu_initialized_ = false;
        wheel_initialized_ = false;
    }

    // Active lifecycle publishers
    odom_pub_->on_activate();

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
    
    if (!std::isfinite(timestamp)) return;

    if (!imu_initialized_) {
        const auto& q = msg->orientation;
        double norm_sq = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
        if (!std::isfinite(norm_sq) || norm_sq < 0.5) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "FuserNode [imu_callback]: Invalid initial quaternion (norm_sq=%f). Waiting for filter convergence.", norm_sq);
            return;
        }

        last_imu_time_ = timestamp;
        imu_initialized_ = true;

        // Warm-start core solver with the true pitch and roll of the IMU, but enforce 0.0 starting yaw
        // so that the map frame perfectly aligns with the odom frame at startup.
        RCLCPP_INFO(get_logger(), "[TRACE] FuserNode: Converting msg->orientation to gtsam::Rot3.");
        gtsam::Rot3 full_rot = gtsam::Rot3::Quaternion(q.w, q.x, q.y, q.z);
        
        RCLCPP_INFO(get_logger(), "[TRACE] FuserNode: Computing Ypr from full_rot.");
        gtsam::Rot3 initial_rot = gtsam::Rot3::Ypr(0.0, full_rot.pitch(), full_rot.roll());
        
        RCLCPP_INFO(get_logger(), "[TRACE] FuserNode: Calling fuser_->initialize_graph().");
        fuser_->initialize_graph(initial_rot);
        
        RCLCPP_INFO(get_logger(), "FuserNode [imu_callback]: Warm-started solver with IMU initial orientation (w=%f, x=%f, y=%f, z=%f)", q.w, q.x, q.y, q.z);
        return;
    }

    const double dt = timestamp - last_imu_time_;
    last_imu_time_ = timestamp;

    if (!std::isfinite(dt) || dt <= 0.0 || dt > 0.5) {
        return;
    }

    // Convert ROS Imu vectors to Eigen structures (Layer 1 -> Layer 2 boundary)
    Eigen::Vector3d accel(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    Eigen::Vector3d gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

    if (!accel.allFinite() || !gyro.allFinite()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "FuserNode [imu_callback]: Non-finite IMU accel or gyro. Dropping message.");
        return;
    }

    // Call mathematical core
    fuser_->add_imu_measurement(dt, accel, gyro);

    // Accumulate IMU linear acceleration statistics for slip filter
    {
        std::lock_guard<std::mutex> lock(state_mtx_);
        accum_imu_accel_x_ += msg->linear_acceleration.x;
    }
}

void FuserNode::wheel_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
    const double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    
    if (!std::isfinite(timestamp)) return;

    std::lock_guard<std::mutex> lock(state_mtx_);
    if (!wheel_initialized_) {
        last_wheel_time_ = timestamp;
        last_wheel_vx_ = msg->twist.twist.linear.x;
        wheel_initialized_ = true;
        return;
    }

    const double dt = timestamp - last_wheel_time_;
    last_wheel_time_ = timestamp;

    if (!std::isfinite(dt) || dt <= 0.0 || dt > 1.0) {
        return;
    }

    const double vx = msg->twist.twist.linear.x;
    const double omega_z = msg->twist.twist.angular.z;

    if (!std::isfinite(vx) || !std::isfinite(omega_z)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "FuserNode [wheel_callback]: Non-finite Odometry velocity. Dropping message.");
        return;
    }

    // Wheel odometry natively provides zero-velocity updates when stationary.
    // We do NOT reset pim_ here because it destroys the continuous IMU factor
    // needed for the next iSAM2 optimization cycle!

    // Kinematic integration of displacement and yaw rotation (tangent vector components)
    const double ds = vx * dt;
    const double dtheta = omega_z * dt;

    // Track pure continuous dead-reckoning for the odom->base TF
    gtsam::Pose3 delta(gtsam::Rot3::Yaw(dtheta), gtsam::Point3(ds, 0.0, 0.0));
    pure_odom_to_base_ = pure_odom_to_base_.compose(delta);

    // Compute longitudinal acceleration from wheel encoders
    const double wheel_accel = (vx - last_wheel_vx_) / dt;
    last_wheel_vx_ = vx;

    // Accumulate statistics for the slip-gating filter
    if (std::isfinite(wheel_accel)) {
        accum_wheel_accel_x_ += wheel_accel;
        accum_accel_count_++;
    }
}

void FuserNode::trn_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
{
    std::cerr << "[DIAG] trn_callback ENTER" << std::endl;

    // Convert ROS Pose definition to GTSAM Pose3
    const auto& pos = msg->pose.pose.position;
    const auto& ori = msg->pose.pose.orientation;
    
    std::cerr << "[DIAG] trn_callback: TRN pos=(" << pos.x << "," << pos.y << "," << pos.z
              << ") ori=(" << ori.w << "," << ori.x << "," << ori.y << "," << ori.z << ")" << std::endl;

    gtsam::Rot3 trn_rot = gtsam::Rot3::Quaternion(ori.w, ori.x, ori.y, ori.z);
    gtsam::Point3 trn_pos(pos.x, pos.y, pos.z);
    gtsam::Pose3 trn_pose(trn_rot, trn_pos);
    std::cerr << "[DIAG] trn_callback: Pose3 constructed" << std::endl;

    // Convert 6x6 Row-major covariance matrix to Eigen::Matrix (ROS order: [Trans, Rot])
    Eigen::Matrix<double, 6, 6> ros_cov = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(msg->pose.covariance.data());
    std::cerr << "[DIAG] trn_callback: ros_cov mapped" << std::endl;

    // Swap 3x3 blocks to map ROS [Trans, Rot] to GTSAM [Rot, Trans]
    Eigen::Matrix<double, 6, 6> trn_covariance;
    trn_covariance.block<3,3>(0,0).noalias() = ros_cov.block<3,3>(3,3);
    trn_covariance.block<3,3>(3,3).noalias() = ros_cov.block<3,3>(0,0);
    trn_covariance.block<3,3>(0,3).noalias() = ros_cov.block<3,3>(3,0);
    trn_covariance.block<3,3>(3,0).noalias() = ros_cov.block<3,3>(0,3);

    // Scale TRN Covariance for Factor Graph (Fix 5 from diagnosis)
    // The particle filter search area inflates uncertainty. 
    // Scaling by 0.1 trusts TRN 10x more relative to wheel odometry.
    double trn_cov_scale = 0.1;
    trn_covariance *= trn_cov_scale;

    std::cerr << "[DIAG] trn_callback: cov swapped and scaled, diag=(" 
              << trn_covariance(0,0) << "," << trn_covariance(1,1) << ","
              << trn_covariance(2,2) << "," << trn_covariance(3,3) << ","
              << trn_covariance(4,4) << "," << trn_covariance(5,5) << ")" << std::endl;

    // Extract statistics and average them over the epoch interval
    double mean_wheel_accel = 0.0;
    double mean_imu_accel = 0.0;
    gtsam::Pose3 wheel_delta;
    
    {
        std::lock_guard<std::mutex> lock(state_mtx_);
        if (accum_accel_count_ > 0) {
            mean_wheel_accel = accum_wheel_accel_x_ / static_cast<double>(accum_accel_count_);
            mean_imu_accel = accum_imu_accel_x_ / static_cast<double>(accum_accel_count_);
        }
        // Compute precise SE(3) relative motion since the last global correction
        wheel_delta = odom_at_last_keyframe_.between(pure_odom_to_base_);
    }
    std::cerr << "[DIAG] trn_callback: wheel_delta computed, accel_count=" << accum_accel_count_ << std::endl;

    // Call mathematical core to execute iSAM2 optimization update
    std::cerr << "[DIAG] trn_callback: CALLING add_global_correction" << std::endl;
    fuser_->add_global_correction(
        trn_pose,
        trn_covariance,
        wheel_delta,
        mean_wheel_accel,
        mean_imu_accel
    );
    std::cerr << "[DIAG] trn_callback: add_global_correction RETURNED" << std::endl;

    {
        std::lock_guard<std::mutex> lock(state_mtx_);
        // Snapshot the current pure odom state as the anchor for the next interval
        odom_at_last_keyframe_ = pure_odom_to_base_;

        // Reset acceleration accumulators (since ds and dtheta are no longer needed)
        accum_wheel_accel_x_ = 0.0;
        accum_imu_accel_x_ = 0.0;
        accum_accel_count_ = 0;
    }
}


void FuserNode::publish_odometry()
{
    if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE || !imu_initialized_) {
        return;
    }

    gtsam::Pose3 wheel_delta;
    gtsam::Pose3 local_pure_odom;
    double current_vx;
    {
        std::lock_guard<std::mutex> lock(state_mtx_);
        wheel_delta = odom_at_last_keyframe_.between(pure_odom_to_base_);
        local_pure_odom = pure_odom_to_base_;
        current_vx = last_wheel_vx_;
    }

    // Get current high-frequency dead-reckoned state from math core using true wheel odometry
    // This returns the precise map->base pose (with exact TRN updates applied)
    gtsam::Pose3 map_to_base = fuser_->get_current_pose(wheel_delta);
    Eigen::Vector3d velocity = fuser_->get_current_velocity(current_vx);

    // Dynamically compute the jump-correction TF (map -> odom)
    // T_{map->base} = T_{map->odom} * T_{odom->base}
    // T_{map->odom} = T_{map->base} * (T_{odom->base})^-1
    gtsam::Pose3 map_to_odom = map_to_base.compose(local_pure_odom.inverse());

    rclcpp::Time now = this->get_clock()->now();

    // Publish high-frequency filtered Odometry using the continuous odom frame
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = now;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = base_frame_;

    // Odometry is published in the pure odom frame, but to avoid the robot pitching 
    // down through the floor in Rviz, we fuse IMU pitch/roll into the published msg.
    gtsam::Rot3 imu_rot = map_to_base.rotation();
    gtsam::Rot3 odom_rot = gtsam::Rot3::Ypr(local_pure_odom.rotation().yaw(), imu_rot.pitch(), imu_rot.roll());

    const auto& t = local_pure_odom.translation();
    const auto& q = odom_rot.toQuaternion();

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

    // Broadcast authoritative map -> odom TF continuously
    geometry_msgs::msg::TransformStamped map_tf;
    map_tf.header.stamp = now;
    map_tf.header.frame_id = map_frame_;
    map_tf.child_frame_id = odom_frame_;

    const auto& mt = map_to_odom.translation();
    const auto& mq = map_to_odom.rotation().toQuaternion();

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
    // Install crash handler BEFORE anything else
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    signal(SIGFPE,  crash_handler);
    signal(SIGBUS,  crash_handler);
    std::cerr << "[DIAG] fuser_node main(): crash handler installed" << std::endl;

    rclcpp::init(argc, argv);
    auto node = std::shared_ptr<ugv_estimation::FuserNode>(new ugv_estimation::FuserNode());
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
