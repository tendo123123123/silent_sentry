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
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    isam2_.reset(new gtsam::ISAM2(params));

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

    std::cout << "[TRACE] SE3FuserCore: Defereing isam2_->update() until first IMU edge is added to prevent COLAMD segfault on disconnected graph." << std::endl;
    // DO NOT call isam2_->update() or clear the buffers! 
    // Leave the Priors in graph_ and initial_values_ so they are committed 
    // simultaneously with the first ImuFactor in add_global_correction().

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

gtsam::SharedNoiseModel SE3FuserCore::evaluate_slip_gate(double wheel_accel_x, double imu_accel_x, double /* wheel_yaw_delta */) const
{
    // Compare structural wheel acceleration with physical accelerometer measurement
    const double accel_diff = std::abs(wheel_accel_x - imu_accel_x);
    const bool is_slipping = (accel_diff > config_.slip_accel_threshold);

    // Initialize sigmas with base tight parameters
    Eigen::Matrix<double, 6, 1> sigmas = config_.odom_sigmas;

    // --- PURE IMU HEADING & 3D OBSERVABILITY ---
    // Wheel encoders ONLY measure longitudinal (forward) movement along the chassis.
    // They have absolutely zero observability over Roll, Pitch, lateral slip (Y), or vertical bounce (Z).
    // If we constrain these to 0 (which the flat 2D wheel_delta assumes), the Factor Graph 
    // will violently fight the IMU when the robot climbs a dune, resulting in massive fake biases.
    // We completely decouple all non-longitudinal constraints from the wheel factor.
    sigmas(0) = 1e6; // Ignore wheel Roll
    sigmas(1) = 1e6; // Ignore wheel Pitch
    sigmas(2) = 1e6; // Ignore wheel Yaw
    
    // sigmas(3) is Forward X (WE KEEP THIS!)
    
    sigmas(4) = 1e6; // Ignore wheel lateral Y
    sigmas(5) = 1e6; // Ignore wheel vertical Z

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

    return gtsam::noiseModel::Gaussian::Covariance(covariance);
}

gtsam::Pose3 SE3FuserCore::add_global_correction(
    const gtsam::Pose3& trn_pose,
    const Eigen::Matrix<double, 6, 6>& trn_covariance,
    const gtsam::Pose3& wheel_delta,
    double wheel_accel_x,
    double imu_accel_x,
    const gtsam::Rot3& ahrs_rotation)
{
    std::cerr << "[DIAG] add_global_correction ENTER" << std::endl;
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_initialized_) {
        std::cerr << "[DIAG] add_global_correction: NOT INITIALIZED, returning" << std::endl;
        return gtsam::Pose3();
    }

    const uint64_t prev_idx = keyframe_index_;
    const uint64_t next_idx = keyframe_index_ + 1;
    std::cerr << "[DIAG] add_global_correction: prev=" << prev_idx << " next=" << next_idx << std::endl;

    // 1. Predict state using IMU preintegration
    std::cerr << "[DIAG] step 1: pim_->predict" << std::endl;
    gtsam::NavState anchor_state(current_pose_, current_velocity_);
    gtsam::NavState predicted_state = pim_->predict(anchor_state, current_bias_);
    std::cerr << "[DIAG] step 1: predict DONE" << std::endl;

    // 2. Add ImuFactor via emplace_shared
    std::cerr << "[DIAG] step 2: emplace ImuFactor" << std::endl;
    graph_.emplace_shared<gtsam::ImuFactor>(
        X(prev_idx), V(prev_idx),
        X(next_idx), V(next_idx),
        B(prev_idx), *pim_
    );
    std::cerr << "[DIAG] step 2: ImuFactor DONE" << std::endl;

    // 3. Add slip-gated wheel BetweenFactor via emplace_shared
    std::cerr << "[DIAG] step 3: evaluate_slip_gate" << std::endl;
    auto wheel_noise = evaluate_slip_gate(wheel_accel_x, imu_accel_x, wheel_delta.rotation().yaw());
    std::cerr << "[DIAG] step 3: emplace BetweenFactor<Pose3>" << std::endl;
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(prev_idx), X(next_idx), wheel_delta, wheel_noise);
    std::cerr << "[DIAG] step 3: BetweenFactor DONE" << std::endl;

    // 3.5 Add BetweenFactor for IMU Bias Random Walk
    std::cerr << "[DIAG] step 3.5: bias random walk" << std::endl;
    gtsam::Matrix66 bias_cov = gtsam::Matrix66::Zero();
    bias_cov.block<3,3>(0,0) = gtsam::Matrix33::Identity() * (config_.imu_accel_bias_noise * config_.imu_accel_bias_noise);
    bias_cov.block<3,3>(3,3) = gtsam::Matrix33::Identity() * (config_.imu_gyro_bias_noise * config_.imu_gyro_bias_noise);
    auto bias_noise_model = gtsam::noiseModel::Gaussian::Covariance(bias_cov);
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
        B(prev_idx), B(next_idx), gtsam::imuBias::ConstantBias(), bias_noise_model);
    std::cerr << "[DIAG] step 3.5: bias DONE" << std::endl;

    // 4. Add global TRN PriorFactor via emplace_shared
    std::cerr << "[DIAG] step 4: TRN prior" << std::endl;
    auto trn_noise = gtsam::noiseModel::Gaussian::Covariance(trn_covariance);
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        X(next_idx), trn_pose, trn_noise);
    std::cerr << "[DIAG] step 4: TRN prior DONE" << std::endl;

    // 4.5 Add absolute heading prior from AHRS filter (ALIGNED to map frame)
    std::cerr << "[DIAG] step 4.5: AHRS orientation prior" << std::endl;
    // We constrain the 3D rotation, but leave translation completely unconstrained (1e6)
    auto ahrs_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 1e6, 1e6, 1e6).finished());
    gtsam::Pose3 ahrs_pose(ahrs_rotation, gtsam::Point3(0.0, 0.0, 0.0));
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        X(next_idx), ahrs_pose, ahrs_noise);
    std::cerr << "[DIAG] step 4.5: AHRS prior DONE" << std::endl;

    // 5. Insert initial value predictions for the new node states
    std::cerr << "[DIAG] step 5: insert initial values" << std::endl;
    initial_values_.insert(X(next_idx), predicted_state.pose());
    initial_values_.insert(V(next_idx), predicted_state.v());
    initial_values_.insert(B(next_idx), current_bias_);
    std::cerr << "[DIAG] step 5: initial values DONE, graph size=" << graph_.size() << " values size=" << initial_values_.size() << std::endl;

    // 6. Execute ISAM2 optimization update
    std::cerr << "[DIAG] step 6: isam2_->update()" << std::endl;
    try {
        isam2_->update(graph_, initial_values_);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] isam2_->update() exception: " << e.what() << std::endl;
        throw;
    }
    std::cerr << "[DIAG] step 6: isam2 update DONE" << std::endl;
    graph_.resize(0);
    initial_values_.clear();

    // 7. Extract the fully optimized states at next_idx
    std::cerr << "[DIAG] step 7: calculateEstimate" << std::endl;
    try {
        gtsam::Values results = isam2_->calculateEstimate();
        current_pose_ = results.at<gtsam::Pose3>(X(next_idx));
        current_velocity_ = results.at<gtsam::Vector3>(V(next_idx));
        current_bias_ = results.at<gtsam::imuBias::ConstantBias>(B(next_idx));
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] calculateEstimate exception: " << e.what() << std::endl;
        throw;
    }
    std::cerr << "[DIAG] step 7: estimate DONE" << std::endl;

    // 8. Reset the preintegration buffer using the newly optimized bias
    pim_->resetIntegrationAndSetBias(current_bias_);
    keyframe_index_ = next_idx;

    std::cerr << "[DIAG] add_global_correction COMPLETE, keyframe=" << keyframe_index_ << std::endl;
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
    
    // As requested: In dead reckoning, both Rotation and Translation should be taken from IMU data.
    // The continuous smooth trajectory relies 100% on the preintegrated IMU, while the 
    // Factor Graph fuses the IMU and Encoders periodically to correct the IMU biases.
    return predicted_state.pose();
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
