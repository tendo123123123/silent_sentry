/**
 * @file se3_fuser_core.hpp
 * @brief High-performance, ROS-independent SE(3) pose estimation core using GTSAM iSAM2.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#ifndef UGV_ESTIMATION__SE3_FUSER_CORE_HPP_
#define UGV_ESTIMATION__SE3_FUSER_CORE_HPP_

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Dense>
#include <memory>
#include <mutex>

namespace ugv_estimation {

using PreintegrationParamsPtr = decltype(gtsam::PreintegrationParams::MakeSharedU(double()));

/**
 * @struct FuserConfig
 * @brief Algorithmic parameters for the SE(3) state estimator.
 */
struct FuserConfig {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double imu_accel_noise;      ///< Accel noise density (m/s^1.5)
    double imu_gyro_noise;       ///< Gyro noise density (rad/s^0.5)
    double imu_integration_noise; ///< Continuous-time integration noise
    double imu_accel_bias_noise; ///< Accel bias random walk (m/s^2.5)
    double imu_gyro_bias_noise;  ///< Gyro bias random walk (rad/s^1.5)

    double initial_pose_sigma;   ///< Standard deviation for initial pose prior
    double initial_vel_sigma;    ///< Standard deviation for initial velocity prior
    double initial_bias_sigma;   ///< Standard deviation for initial bias prior

    double slip_accel_threshold; ///< Slip gate trigger threshold (m/s^2)
    double slip_cov_multiplier;  ///< Factor by which wheel covariance is inflated during slip

    Eigen::Matrix<double, 6, 1> odom_sigmas; ///< Base standard deviations for wheel factors [rot, trans]
};

/**
 * @class SE3FuserCore
 * @brief Pure C++ engine managing the GTSAM factor graph optimization and preintegration.
 * 
 * This class operates directly on the SE(3) Lie manifold, leveraging GTSAM's 
 * high-performance Lie algebra representations. It isolates all robotics mathematics 
 * from the middleware layer, achieving perfect separation of concerns and maximum 
 * deterministic execution speed.
 */
class SE3FuserCore {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /**
     * @brief Construct a new SE3FuserCore object.
     * @param config Configuration parameters containing noise models and slip gates.
     */
    explicit SE3FuserCore(const FuserConfig& config);

    /**
     * @brief Default Destructor. Handles cleanup of raw pointers inside GTSAM constructs.
     */
    ~SE3FuserCore() = default;

    /**
     * @brief Initialize the factor graph with zero-state priors at keyframe 0.
     * @param initial_rotation Optional initial orientation of the vehicle (defaults to Identity).
     * 
     * Mathematically anchors the start state on the manifold to avoid global gage ambiguity:
     * \f[
     *   \mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \Sigma_0)
     * \f]
     */
    void initialize_graph(const gtsam::Rot3& initial_rotation = gtsam::Rot3());

    /**
     * @brief Integrate a high-frequency IMU sample into the preintegration buffer.
     * @param dt Elapsed time since last IMU sample (seconds).
     * @param accel Raw linear acceleration vector (m/s^2) in UGV body frame.
     * @param gyro Raw angular velocity vector (rad/s) in UGV body frame.
     * 
     * Integrates raw measurements into a PreintegratedImuMeasurements buffer,
     * tracking the local state on the \f$ SE(3) \times \mathbb{R}^3 \f$ manifold.
     */
    void add_imu_measurement(double dt, const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro);

    /**
     * @brief Evaluate UGV slippage on soft desert sand and compute the gated covariance.
     * @param wheel_accel_x Longitudinal acceleration estimated from wheel encoders (m/s^2).
     * @param imu_accel_x True longitudinal acceleration measured by the accelerometer (m/s^2).
     * @return Eigen::Matrix<double, 6, 6> 6x6 Pose3 covariance matrix.
     * 
     * If UGV wheels spin up while the IMU detects no physical acceleration, we identify slip:
     * \f[
     *   |a_{\text{wheel}} - a_{\text{IMU}}| > \text{threshold}
     * \f]
     * Under slip conditions, the wheel odometry factor is heavily down-weighted by inflating 
     * its covariance by the slip_cov_multiplier to force iSAM2 to follow IMU and near-zero motion cues.
     * Furthermore, if driving straight, yaw constraint is tightened to learn IMU gyro bias.
     */
    gtsam::SharedNoiseModel evaluate_slip_gate(double wheel_accel_x, double imu_accel_x, double wheel_yaw_delta) const;

    /**
     * @brief Fuses a new global TRN loop closure correction and optimizes the Factor Graph.
     * @param trn_pose Global absolute pose (map frame) estimated by TRN MCL.
     * @param trn_covariance 6x6 uncertainty covariance of the TRN match.
     * @param wheel_delta SE(3) relative pose from last keyframe.
     * @param wheel_accel_x Mean wheel acceleration over the interval.
     * @param imu_accel_x Mean IMU acceleration over the interval.
     * @return gtsam::Pose3 The resulting residual transform \f$ T_{\text{map} \to \text{odom}} \f$.
     */
    gtsam::Pose3 add_global_correction(
        const gtsam::Pose3& trn_pose,
        const Eigen::Matrix<double, 6, 6>& trn_covariance,
        const gtsam::Pose3& wheel_delta,
        double wheel_accel_x,
        double imu_accel_x,
        const gtsam::Rot3& ahrs_rotation
    );

    /**
     * @brief Propagate the current dead-reckoning state on the manifold using wheel odometry.
     * @param wheel_delta SE(3) relative pose from last keyframe.
     * @return gtsam::Pose3 Current high-frequency estimated pose.
     */
    gtsam::Pose3 get_current_pose(const gtsam::Pose3& wheel_delta) const;

    /**
     * @brief Retrieve vehicle twist velocity in body frame.
     * @param wheel_vx The actual measured wheel linear velocity.
     */
    Eigen::Vector3d get_current_velocity(double wheel_vx) const;

    /**
     * @brief Retrieve optimized IMU biases.
     */
    gtsam::imuBias::ConstantBias get_current_bias() const;

    /**
     * @brief Reset the preintegration buffer and velocity to zero (used when robot is stationary).
     */
    void reset_preintegration();

    /**
     * @brief Thread-safe configuration update.
     */
    void update_config(const FuserConfig& config);

private:
    FuserConfig config_;                    ///< Configuration parameters
    mutable std::mutex mtx_;                ///< Thread-safety guard

    std::unique_ptr<gtsam::ISAM2> isam2_;   ///< Incremental iSAM2 Bayes Tree solver
    gtsam::NonlinearFactorGraph graph_;     ///< Factor graph container for incremental steps
    gtsam::Values initial_values_;          ///< Initial values for new nodes

    uint64_t keyframe_index_;               ///< Index of the latest keyframe node X(k)
    bool is_initialized_;                   ///< Initialization flag

    // Current Optimized States
    gtsam::Pose3 current_pose_;              ///< Optimized pose (map frame coordinate space)
    gtsam::Vector3 current_velocity_;        ///< Optimized velocity (m/s)
    gtsam::imuBias::ConstantBias current_bias_; ///< Optimized IMU biases

    // IMU Preintegration Handle
    PreintegrationParamsPtr imu_params_;
    std::unique_ptr<gtsam::PreintegratedImuMeasurements> pim_;
};

} // namespace ugv_estimation

#endif // UGV_ESTIMATION__SE3_FUSER_CORE_H_
