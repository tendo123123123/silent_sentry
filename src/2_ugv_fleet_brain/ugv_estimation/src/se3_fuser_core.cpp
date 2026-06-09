/**
 * @file se3_fuser_core.cpp
 * @brief High-performance, ROS-independent SE(3) pose estimation core implementation.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#include "ugv_estimation/se3_fuser_core.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/NavState.h>

#include <cmath>

namespace ugv_estimation {

// Short-hand symbol helpers
using gtsam::symbol_shorthand::X; // Pose3 keys
using gtsam::symbol_shorthand::V; // Velocity keys
using gtsam::symbol_shorthand::B; // Bias keys

SE3FuserCore::SE3FuserCore(const FuserConfig& config)
    : config_(config),
      keyframe_index_(0),
      is_initialized_(false),
      current_pose_(gtsam::Pose3()),
      current_velocity_(gtsam::Vector3::Zero()),
      current_bias_(gtsam::imuBias::ConstantBias())
{
    // Initialize IMU parameters
    // MakeSharedU(9.81) constructs parameters with vertical gravity along negative Z-axis.
    imu_params_ = gtsam::PreintegrationParams::MakeSharedU(9.81);

    // Accel noise density
    const double acc_var = config_.imu_accel_noise * config_.imu_accel_noise;
    imu_params_->setAccelerometerCovariance(gtsam::Matrix33::Identity() * acc_var);

    // Gyro noise density
    const double gyro_var = config_.imu_gyro_noise * config_.imu_gyro_noise;
    imu_params_->setGyroscopeCovariance(gtsam::Matrix33::Identity() * gyro_var);

    // Continuous-time integration covariance
    const double int_var = config_.imu_integration_noise * config_.imu_integration_noise;
    imu_params_->setIntegrationCovariance(gtsam::Matrix33::Identity() * int_var);

    // Set 2nd-order Coriolis to false for maximum computational efficiency on local scales
    imu_params_->setUse2ndOrderCoriolis(false);
    imu_params_->setOmegaCoriolis(gtsam::Vector3::Zero());

    // Create the preintegration buffer
    pim_ = std::unique_ptr<gtsam::PreintegratedImuMeasurements>(new gtsam::PreintegratedImuMeasurements(imu_params_, current_bias_));
}

void SE3FuserCore::initialize_graph(const gtsam::Rot3& initial_rotation)
{
    std::cout << "[TRACE] SE3FuserCore: Entering initialize_graph lock." << std::endl;
    std::lock_guard<std::mutex> lock(mtx_);
    if (is_initialized_) {
        return;
    }

    std::cout << "[TRACE] SE3FuserCore: Setting initial states." << std::endl;
    // Set starting states with initial orientation to compensate gravity correctly
    current_pose_ = gtsam::Pose3(initial_rotation, gtsam::Point3(0.0, 0.0, 0.0));
    current_velocity_ = gtsam::Vector3::Zero();
    current_bias_ = gtsam::imuBias::ConstantBias();

    std::cout << "[TRACE] SE3FuserCore: Configuring noise models." << std::endl;
    // Configure initial noise models
    auto pose_noise = gtsam::noiseModel::Isotropic::Sigma(6, config_.initial_pose_sigma);
    auto vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, config_.initial_vel_sigma);
    auto bias_noise = gtsam::noiseModel::Isotropic::Sigma(6, config_.initial_bias_sigma);

    std::cout << "[TRACE] SE3FuserCore: Adding prior factors." << std::endl;
    // Add prior factors to mathematically anchor keyframe 0
    graph_.addPrior(X(0), current_pose_, pose_noise);
    graph_.addPrior(V(0), current_velocity_, vel_noise);
    graph_.addPrior(B(0), current_bias_, bias_noise);

    std::cout << "[TRACE] SE3FuserCore: Inserting initial values." << std::endl;
    // Add initial values for optimization
    initial_values_.insert(X(0), current_pose_);
    initial_values_.insert(V(0), current_velocity_);
    initial_values_.insert(B(0), current_bias_);

    std::cout << "[TRACE] SE3FuserCore: Calling isam2_.update()." << std::endl;
    // Run initial ISAM2 update
    isam2_.update(graph_, initial_values_);
    
    std::cout << "[TRACE] SE3FuserCore: Clearing graph buffers." << std::endl;
    graph_.resize(0);
    initial_values_.clear();

    keyframe_index_ = 0;
    is_initialized_ = true;
    std::cout << "[TRACE] SE3FuserCore: initialize_graph COMPLETE." << std::endl;
}

void SE3FuserCore::add_imu_measurement(double dt, const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_initialized_) {
        return;
    }
    pim_->integrateMeasurement(accel, gyro, dt);
}

Eigen::Matrix<double, 6, 6> SE3FuserCore::evaluate_slip_gate(double wheel_accel_x, double imu_accel_x) const
{
    // Compare structural wheel acceleration with physical accelerometer measurement
    const double accel_diff = std::abs(wheel_accel_x - imu_accel_x);
    const bool is_slipping = (accel_diff > config_.slip_accel_threshold);

    // Initialize sigmas with base tight parameters
    Eigen::Matrix<double, 6, 1> sigmas = config_.odom_sigmas;

    if (is_slipping) {
        // Inflate the translational translation noise sigmas (indices 3, 4, 5) by the multiplier
        // This dynamically isolates the wheel factor from corrupting the graph during slippage.
        sigmas(3) *= config_.slip_cov_multiplier; // forward x
        sigmas(4) *= config_.slip_cov_multiplier; // lateral y
        sigmas(5) *= config_.slip_cov_multiplier; // vertical z
    }

    // Convert sigmas to a diagonal covariance matrix: Cov = diag(sigmas^2)
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
    covariance.diagonal().noalias() = sigmas.array().square().matrix();

    return covariance;
}

gtsam::Pose3 SE3FuserCore::add_global_correction(
    const gtsam::Pose3& trn_pose,
    const Eigen::Matrix<double, 6, 6>& trn_covariance,
    const gtsam::Pose3& wheel_delta,
    double wheel_accel_x,
    double imu_accel_x)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_initialized_) {
        return gtsam::Pose3();
    }

    const uint64_t prev_idx = keyframe_index_;
    const uint64_t next_idx = keyframe_index_ + 1;

    // 1. Predict state using IMU preintegration
    gtsam::NavState anchor_state(current_pose_, current_velocity_);
    gtsam::NavState predicted_state = pim_->predict(anchor_state, current_bias_);

    // 2. Construct and add the IMU Factor spanning the entire optimization interval
    gtsam::ImuFactor imu_factor(
        X(prev_idx), V(prev_idx),
        X(next_idx), V(next_idx),
        B(prev_idx), *pim_
    );
    graph_.add(imu_factor);

    // 3. Construct and add the slip-gated wheel BetweenFactor
    Eigen::Matrix<double, 6, 6> wheel_cov = evaluate_slip_gate(wheel_accel_x, imu_accel_x);
    auto wheel_noise = gtsam::noiseModel::Gaussian::Covariance(wheel_cov);
    
    gtsam::BetweenFactor<gtsam::Pose3> odom_factor(X(prev_idx), X(next_idx), wheel_delta, wheel_noise);
    graph_.add(odom_factor);

    // 4. Construct and add the global TRN PriorFactor in map frame
    auto trn_noise = gtsam::noiseModel::Gaussian::Covariance(trn_covariance);
    gtsam::PriorFactor<gtsam::Pose3> trn_prior(X(next_idx), trn_pose, trn_noise);
    graph_.add(trn_prior);

    // 5. Insert initial value predictions for the new node states
    initial_values_.insert(X(next_idx), predicted_state.pose());
    initial_values_.insert(V(next_idx), predicted_state.v());
    initial_values_.insert(B(next_idx), current_bias_);

    // 6. Execute ISAM2 optimization update
    isam2_.update(graph_, initial_values_);
    graph_.resize(0);
    initial_values_.clear();

    // 7. Extract the fully optimized states at next_idx
    gtsam::Values results = isam2_.calculateEstimate();
    current_pose_ = results.at<gtsam::Pose3>(X(next_idx));
    current_velocity_ = results.at<gtsam::Vector3>(V(next_idx));
    current_bias_ = results.at<gtsam::imuBias::ConstantBias>(B(next_idx));

    // 8. Reset the preintegration buffer using the newly optimized bias
    pim_->resetIntegrationAndSetBias(current_bias_);
    keyframe_index_ = next_idx;

    // Return the new current_pose_
    return current_pose_;
}

gtsam::Pose3 SE3FuserCore::get_current_pose(const gtsam::Pose3& wheel_delta) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_initialized_) {
        return gtsam::Pose3();
    }
    // High-frequency propagation: predict from last keyframe using active IMU buffer to get good pitch/roll
    gtsam::NavState anchor_state(current_pose_, current_velocity_);
    gtsam::NavState predicted_state = pim_->predict(anchor_state, current_bias_);
    
    // Compose wheel odometry for exact forward displacement and yaw rotation
    gtsam::Pose3 wheel_predicted = current_pose_.compose(wheel_delta);
    
    // Combine IMU pitch and roll with the wheel's translation and yaw
    gtsam::Rot3 imu_rot = predicted_state.pose().rotation();
    gtsam::Rot3 combined_rot = gtsam::Rot3::Ypr(wheel_predicted.rotation().yaw(), imu_rot.pitch(), imu_rot.roll());
    
    return gtsam::Pose3(combined_rot, wheel_predicted.translation());
}

Eigen::Vector3d SE3FuserCore::get_current_velocity(double wheel_vx) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_initialized_) {
        return Eigen::Vector3d::Zero();
    }
    // Return the actual wheel linear velocity in the body frame.
    // UGV is non-holonomic, so vy and vz are zero.
    return Eigen::Vector3d(wheel_vx, 0.0, 0.0);
}

gtsam::imuBias::ConstantBias SE3FuserCore::get_current_bias() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return current_bias_;
}

void SE3FuserCore::reset_preintegration()
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_initialized_) {
        return;
    }
    // Lock the last predicted pose to prevent snapping back to the origin
    gtsam::NavState anchor_state(current_pose_, current_velocity_);
    gtsam::NavState predicted_state = pim_->predict(anchor_state, current_bias_);
    current_pose_ = predicted_state.pose();

    pim_->resetIntegrationAndSetBias(current_bias_);
    current_velocity_ = gtsam::Vector3::Zero();
}

void SE3FuserCore::update_config(const FuserConfig& config)
{
    std::lock_guard<std::mutex> lock(mtx_);
    config_ = config;

    // Update continuous integration parameters
    const double acc_var = config_.imu_accel_noise * config_.imu_accel_noise;
    imu_params_->setAccelerometerCovariance(gtsam::Matrix33::Identity() * acc_var);

    const double gyro_var = config_.imu_gyro_noise * config_.imu_gyro_noise;
    imu_params_->setGyroscopeCovariance(gtsam::Matrix33::Identity() * gyro_var);

    const double int_var = config_.imu_integration_noise * config_.imu_integration_noise;
    imu_params_->setIntegrationCovariance(gtsam::Matrix33::Identity() * int_var);
}

} // namespace ugv_estimation
