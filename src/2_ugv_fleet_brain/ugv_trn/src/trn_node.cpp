/**
 * @file trn_node.cpp
 * @brief ROS 2 Lifecycle Node wrapping the TRN core implementation.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#include "ugv_trn/trn_node.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <lifecycle_msgs/msg/state.hpp>

#include <chrono>

namespace ugv_trn {

TRNNode::TRNNode(const rclcpp::NodeOptions& options)
    : rclcpp_lifecycle::LifecycleNode("trn_node", options),
      last_odom_time_(0.0),
      last_odom_x_(0.0),
      last_odom_y_(0.0),
      last_odom_yaw_(0.0),
      odom_initialized_(false),
      map_frame_("map"),
      odom_frame_("odom"),
      base_frame_("base_footprint")
{
    // Declare Configuration Parameters with defaults
    this->declare_parameter<int>("num_particles", 800);
    this->declare_parameter<double>("min_peak_quality", 0.40);
    this->declare_parameter<double>("flatness_std_threshold", 0.05);
    this->declare_parameter<double>("entropy_threshold", 1.8);
    this->declare_parameter<double>("ess_threshold", 0.40);
    this->declare_parameter<double>("motion_noise_xy_frac", 0.15);
    this->declare_parameter<double>("motion_noise_yaw_frac", 0.05);

    this->declare_parameter<double>("base_search_radius", 15.0);
    this->declare_parameter<double>("max_search_radius", 30.0);
    this->declare_parameter<double>("initial_search_radius", 15.0);

    this->declare_parameter<double>("bilateral_sigma_color", 15.0);
    this->declare_parameter<double>("bilateral_sigma_space", 75.0);
    this->declare_parameter<int>("bilateral_d", 9);

    this->declare_parameter<double>("amcl_random_fraction", 0.10);

    this->declare_parameter<std::string>("global_dem_path", "");
    this->declare_parameter<double>("global_dem_res", 1.0);
    this->declare_parameter<double>("global_dem_origin_x", 0.0);
    this->declare_parameter<double>("global_dem_origin_y", 0.0);

    this->declare_parameter<std::string>("map_frame", "map");
    this->declare_parameter<std::string>("odom_frame", "odom");
    this->declare_parameter<std::string>("base_frame", "base_footprint");
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TRNNode::on_configure(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "TRNNode [on_configure]: Parsing configuration parameters.");

    // Retrieve Parameters
    TRNConfig config;
    this->get_parameter("num_particles", config.num_particles);
    this->get_parameter("min_peak_quality", config.min_peak_quality);
    this->get_parameter("flatness_std_threshold", config.flatness_std_threshold);
    this->get_parameter("entropy_threshold", config.entropy_threshold);
    this->get_parameter("ess_threshold", config.ess_threshold);
    this->get_parameter("motion_noise_xy_frac", config.motion_noise_xy_frac);
    this->get_parameter("motion_noise_yaw_frac", config.motion_noise_yaw_frac);
    this->get_parameter("base_search_radius", config.base_search_radius);
    this->get_parameter("max_search_radius", config.max_search_radius);
    this->get_parameter("initial_search_radius", config.initial_search_radius);
    this->get_parameter("bilateral_sigma_color", config.bilateral_sigma_color);
    this->get_parameter("bilateral_sigma_space", config.bilateral_sigma_space);
    this->get_parameter("bilateral_d", config.bilateral_d);
    this->get_parameter("amcl_random_fraction", config.amcl_random_fraction);

    std::string global_dem_path;
    double global_dem_res = 1.0;
    double global_dem_origin_x = 0.0;
    double global_dem_origin_y = 0.0;

    this->get_parameter("global_dem_path", global_dem_path);
    this->get_parameter("global_dem_res", global_dem_res);
    this->get_parameter("global_dem_origin_x", global_dem_origin_x);
    this->get_parameter("global_dem_origin_y", global_dem_origin_y);

    this->get_parameter("map_frame", map_frame_);
    this->get_parameter("odom_frame", odom_frame_);
    this->get_parameter("base_frame", base_frame_);

    // Create math engine (Layer 2)
    trn_core_ = std::make_unique<TRNCore>(config);

    // Load Offline Global DEM Reference
    if (!trn_core_->load_global_dem(global_dem_path, global_dem_res, global_dem_origin_x, global_dem_origin_y)) {
        RCLCPP_ERROR(get_logger(), "TRNNode [on_configure]: FAILED to load Global DEM reference! Node configuration failed.");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    // Setup Publishers
    correction_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/trn/global_correction", 10);
    quality_pub_ = this->create_publisher<std_msgs::msg::Float64>("/trn/match_quality", 10);

    // Setup TF Listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    RCLCPP_INFO(get_logger(), "TRNNode [on_configure]: Successfully configured.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TRNNode::on_activate(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "TRNNode [on_activate]: Activating node communication.");

    odom_initialized_ = false;
    last_odom_time_ = 0.0;
    last_odom_x_ = 0.0;
    last_odom_y_ = 0.0;
    last_odom_yaw_ = 0.0;

    // Activate lifecycle publishers
    correction_pub_->on_activate();
    quality_pub_->on_activate();

    // Setup Topic Subscribers
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", 10, std::bind(&TRNNode::odom_callback, this, std::placeholders::_1)
    );

    local_dem_sub_ = this->create_subscription<silent_sentry_interfaces::msg::LocalDEM>(
        "/elevation_map/local_dem", 10, std::bind(&TRNNode::local_dem_callback, this, std::placeholders::_1)
    );

    // Setup low-frequency 3Hz matching cycle timer
    match_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(333), std::bind(&TRNNode::match_timer_callback, this)
    );

    RCLCPP_INFO(get_logger(), "TRNNode [on_activate]: Successfully activated. Subscribers registered.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TRNNode::on_deactivate(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "TRNNode [on_deactivate]: Deactivating subscriptions and publishers.");

    // Tear down subscribers
    odom_sub_.reset();
    local_dem_sub_.reset();

    // Tear down timers
    match_timer_.reset();

    // Deactivate publishers
    correction_pub_->on_deactivate();
    quality_pub_->on_deactivate();

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TRNNode::on_cleanup(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "TRNNode [on_cleanup]: Cleaning up core matching resources.");
    trn_core_.reset();
    correction_pub_.reset();
    quality_pub_.reset();
    tf_buffer_.reset();
    tf_listener_.reset();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TRNNode::on_shutdown(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "TRNNode [on_shutdown]: Shutting down.");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

void TRNNode::odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
    const double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    const double cx = msg->pose.pose.position.x;
    const double cy = msg->pose.pose.position.y;
    
    // Extract yaw rotation from odom orientation
    const auto& q = msg->pose.pose.orientation;
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    const double cyaw = std::atan2(siny_cosp, cosy_cosp);

    if (!odom_initialized_) {
        last_odom_time_ = timestamp;
        last_odom_x_ = cx;
        last_odom_y_ = cy;
        last_odom_yaw_ = cyaw;
        odom_initialized_ = true;
        return;
    }

    const double dt = timestamp - last_odom_time_;
    if (dt <= 0.0 || dt > 1.0) {
        last_odom_time_ = timestamp;
        return;
    }

    // Compute dead-reckoning coordinate deltas (tangent space projection)
    const double dx = cx - last_odom_x_;
    const double dy = cy - last_odom_y_;
    const double dyaw = std::remainder(cyaw - last_odom_yaw_, 2.0 * M_PI);

    // Call mathematical core to propagate the global particles (Layer 1 -> Layer 2 boundary)
    trn_core_->propagate_particles(dx, dy, dyaw);

    // Update EKF state values
    last_odom_time_ = timestamp;
    last_odom_x_ = cx;
    last_odom_y_ = cy;
    last_odom_yaw_ = cyaw;
}

void TRNNode::local_dem_callback(const silent_sentry_interfaces::msg::LocalDEM::ConstSharedPtr msg)
{
    const int cols = msg->width;
    const int rows = msg->height;

    if (msg->data.size() != static_cast<size_t>(cols * rows)) {
        RCLCPP_WARN(this->get_logger(), "TRN: LocalDEM data flat size (%zu) mismatch with dimensions %dx%d!", msg->data.size(), cols, rows);
        return;
    }

    // Reformat 1D flat message data into a dense 2D Eigen float matrix
    Eigen::MatrixXf local_grid(rows, cols);
    const float no_data_val = msg->no_data_value;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float val = msg->data[r * cols + c];
            if (std::abs(val - no_data_val) < 1e-4) {
                local_grid(r, c) = std::numeric_limits<float>::quiet_NaN();
            } else {
                local_grid(r, c) = val;
            }
        }
    }

    // Update local grid inside pure C++ core (Layer 1 -> Layer 2 boundary)
    trn_core_->update_local_dem(local_grid, msg->resolution, msg->origin_x, msg->origin_y);
}

void TRNNode::match_timer_callback()
{
    if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
        return;
    }

    if (!odom_initialized_) {
        RCLCPP_WARN(this->get_logger(), "TRN: Waiting for initial /odometry/filtered to start matching cycle.");
        return;
    }

    // Look up the current map->odom transform from the TF tree
    gtsam::Pose3 map_to_odom_current = gtsam::Pose3::Identity();
    try {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg = tf_buffer_->lookupTransform(map_frame_, odom_frame_, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.05));
        
        gtsam::Rot3 rot = gtsam::Rot3::Quaternion(
            tf_msg.transform.rotation.w, 
            tf_msg.transform.rotation.x, 
            tf_msg.transform.rotation.y, 
            tf_msg.transform.rotation.z
        );
        gtsam::Point3 pos(
            tf_msg.transform.translation.x, 
            tf_msg.transform.translation.y, 
            tf_msg.transform.translation.z
        );
        map_to_odom_current = gtsam::Pose3(rot, pos);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "TRN: map_to_odom transform lookup failed: %s. Defaulting to identity.", ex.what());
    }

    // Setup latest EKF odometry pose prior in the local odom frame
    gtsam::Pose3 odom_prior(gtsam::Rot3::Yaw(last_odom_yaw_), gtsam::Point3(last_odom_x_, last_odom_y_, 0.0));

    gtsam::Pose3 global_match_pose;
    Eigen::Matrix<double, 6, 6> global_match_cov;
    double match_quality = 0.0;

    // Trigger matching cycle inside pure C++ core (Layer 1 -> Layer 2 boundary)
    bool matched = trn_core_->execute_match_cycle(
        odom_prior,
        map_to_odom_current,
        global_match_pose,
        global_match_cov,
        match_quality
    );

    // Publish Match Quality
    std_msgs::msg::Float64 quality_msg;
    quality_msg.data = match_quality;
    quality_pub_->publish(quality_msg);

    if (matched) {
        RCLCPP_INFO(get_logger(), "TRN: Global DEM match success. Quality: %.4f. Publishing correction.", match_quality);

        // Publish Pose with Covariance correction in map frame
        geometry_msgs::msg::PoseWithCovarianceStamped corr_msg;
        corr_msg.header.stamp = this->get_clock()->now();
        corr_msg.header.frame_id = map_frame_;

        const auto& t = global_match_pose.translation();
        const auto& q = global_match_pose.rotation().toQuaternion();

        corr_msg.pose.pose.position.x = t.x();
        corr_msg.pose.pose.position.y = t.y();
        corr_msg.pose.pose.position.z = t.z();
        corr_msg.pose.pose.orientation.x = q.x();
        corr_msg.pose.pose.orientation.y = q.y();
        corr_msg.pose.pose.orientation.z = q.z();
        corr_msg.pose.pose.orientation.w = q.w();

        // Swap 3x3 blocks to map GTSAM [Rot, Trans] order to ROS 2 [Trans, Rot] order
        // GTSAM tangent space ordering: [Rotation, Translation]
        // ROS 2 PoseWithCovariance tangent space ordering: [Translation, Rotation]
        Eigen::Matrix<double, 6, 6> ros_cov;
        ros_cov.block<3,3>(0,0).noalias() = global_match_cov.block<3,3>(3,3); // GTSAM Trans->Trans to ROS Trans->Trans
        ros_cov.block<3,3>(3,3).noalias() = global_match_cov.block<3,3>(0,0); // GTSAM Rot->Rot to ROS Rot->Rot
        ros_cov.block<3,3>(0,3).noalias() = global_match_cov.block<3,3>(3,0); // GTSAM Trans->Rot to ROS Trans->Rot
        ros_cov.block<3,3>(3,0).noalias() = global_match_cov.block<3,3>(0,3); // GTSAM Rot->Trans to ROS Rot->Trans

        // Flatten the ROS-ordered matrix into corr_msg.pose.covariance
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(corr_msg.pose.covariance.data()).noalias() = ros_cov;

        correction_pub_->publish(corr_msg);
    } else {
        RCLCPP_DEBUG(get_logger(), "TRN: Matching cycle completed with no valid high-confidence global match.");
    }
}

} // namespace ugv_trn

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ugv_trn::TRNNode>();
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
