/**
 * @file trn_core.hpp
 * @brief High-performance, ROS-independent Terrain-Referenced Navigation (TRN) core.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#ifndef UGV_TRN__TRN_CORE_HPP_
#define UGV_TRN__TRN_CORE_HPP_

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <vector>
#include <memory>
#include <mutex>
#include <random>

namespace ugv_trn {

/**
 * @struct Particle
 * @brief Candidate state representation on the SE(2) planar manifold.
 */
struct Particle {
    double x;     ///< Position X in global map frame (meters)
    double y;     ///< Position Y in global map frame (meters)
    double yaw;   ///< Heading orientation in global map frame (radians)
    double weight; ///< Normalized likelihood weight [0.0, 1.0]
};

/**
 * @struct TRNConfig
 * @brief Parameters governing the TRN particle filter.
 */
struct TRNConfig {
    int num_particles;             ///< Number of active particles (e.g., 800)
    double min_peak_quality;       ///< Minimum maximum likelihood score to accept match [0.0, 1.0]
    double flatness_std_threshold; ///< Standard deviation elevation floor to avoid flat matching (meters)
    double entropy_threshold;      ///< Local DEM spatial entropy threshold
    double ess_threshold;          ///< Effective Sample Size (ESS) ratio trigger for resampling

    double motion_noise_xy_frac;   ///< Accrued translation noise fraction per meter
    double motion_noise_yaw_frac;  ///< Accrued rotation noise fraction per radian

    double base_search_radius;     ///< Nominal search boundary radius (meters)
    double max_search_radius;      ///< Maximum expanded search boundary (meters)
    double initial_search_radius;  ///< Search radius used during startup cold-start (meters)

    double bilateral_sigma_color;  ///< Bilateral filter sigma in elevation domain (meters)
    double bilateral_sigma_space;  ///< Bilateral filter sigma in spatial domain (pixels)
    int bilateral_d;               ///< Bilateral filter diameter (pixels)

    double amcl_random_fraction;   ///< AMCL style random recovery injection fraction per resample
};

/**
 * @class TRNCore
 * @brief Standard-compliant pure C++ engine computing Terrain-Referenced Monte Carlo Localization.
 * 
 * Computes global alignment on unstructured, sloped, or dune terrain. Incorporates 
 * high-performance Eigen and OpenCV pipelines to filter local grids and evaluate scores 
 * across particle sets, fully decoupled from ROS.
 */
class TRNCore {
public:
    /**
     * @brief Construct a new TRNCore object.
     * @param config Analytical settings of the filter.
     */
    explicit TRNCore(const TRNConfig& config);

    /**
     * @brief Default Destructor.
     */
    ~TRNCore() = default;

    /**
     * @brief Load the global reference height grid.
     * @param filepath Path to global DEM binary/CSV file.
     * @param resolution Grid spacing (meters per pixel).
     * @param origin_x Map X-origin of bottom-left corner of the DEM (meters).
     * @param origin_y Map Y-origin of bottom-left corner of the DEM (meters).
     */
    bool load_global_dem(const std::string& filepath, double resolution, double origin_x, double origin_y);

    /**
     * @brief Update the latest local DEM scan.
     * @param local_grid 2D height map scan. NaNs represent unobserved/invalid cells.
     * @param resolution Grid spacing (meters per pixel).
     * @param origin_x Local scan X-origin in sensor-centered frame (meters).
     * @param origin_y Local scan Y-origin in sensor-centered frame (meters).
     */
    void update_local_dem(const Eigen::MatrixXf& local_grid, double resolution, double origin_x, double origin_y);

    /**
     * @brief Propagate the particle cloud using local wheel-inertial dead-reckoning deltas.
     * @param dx Displacement delta along local X-axis (meters).
     * @param dy Displacement delta along local Y-axis (meters).
     * @param dyaw Yaw angle rotation delta (radians).
     * 
     * Applies a non-linear motion model on the SE(2) tangent space with Gaussian noise:
     * \f[
     *   x_k^{(t+1)} = x_k^{(t)} + dx \cos(\theta_k^{(t)}) - dy \sin(\theta_k^{(t)}) + w_x
     * \f]
     */
    void propagate_particles(double local_dx, double local_dy, double dyaw);

    /**
     * @brief Perform high-fidelity bilateral filtering to smooth local height scanning noise.
     * @param input Raw height matrix.
     * @return Eigen::MatrixXf Filtered grid.
     * 
     * Preserves sharp topographical ridges while eliminating high-frequency LiDAR measurement noise:
     * \f[
     *   I^{\text{filtered}}(\mathbf{x}) = \frac{1}{W_p} \sum_{\mathbf{y}} I(\mathbf{y}) g_s(\|\mathbf{x}-\mathbf{y}\|) g_c(|I(\mathbf{x})-I(\mathbf{y})|)
     * \f]
     */
    Eigen::MatrixXf bilateral_filter(const Eigen::MatrixXf& input) const;

    /**
     * @brief Compute the spatial entropy of a height grid.
     * @param grid 2D height matrix.
     * @return double Calculated Shannon entropy.
     * 
     * Checks if the scanned terrain possesses enough structural texture to warrant matching:
     * \f[
     *   H = -\sum_{i} p_i \log_2(p_i)
     * \f]
     */
    double compute_spatial_entropy(const Eigen::MatrixXf& grid) const;

    /**
     * @brief Runs the complete matching, likelihood evaluation, and resampling loop.
     * @param odom_prior Current robot pose prediction (odom frame coordinate).
     * @param map_to_odom_current Current estimate of the map->odom coordinate transform.
     * @param out_pose Fused global pose output (map frame).
     * @param out_covariance 6x6 uncertainty covariance of the global pose estimate.
     * @param match_quality Highest particle likelihood score achieved in this epoch.
     * @return bool True if a valid high-confidence global match was acquired; false if skipped.
     */
    bool execute_match_cycle(
        const gtsam::Pose3& odom_prior,
        const gtsam::Pose3& map_to_odom_current,
        gtsam::Pose3& out_pose,
        Eigen::Matrix<double, 6, 6>& out_covariance,
        double& match_quality
    );

    /**
     * @brief Thread-safe configuration update.
     */
    void update_config(const TRNConfig& config);

    /**
     * @brief Retrieve copy of active particles (for visualization/diagnostics).
     */
    std::vector<Particle> get_particles() const;

private:
    /**
     * @brief Compute the Mean Absolute Difference (MAD) score of a particle.
     */
    double evaluate_particle_likelihood(const Particle& p, const Eigen::MatrixXf& local_dem_filtered, const gtsam::Pose3& odom_prior) const;

    /**
     * @brief Perform systematic resampling of the particles.
     * 
     * Re-samples particles proportionally to their normalized weights using a single random trigger:
     * \f[
     *   U_k = \frac{r + k - 1}{N}, \quad r \sim \mathcal{U}(0, 1)
     * \f]
     */
    void systematic_resample();

    /**
     * @brief Inject random recovery particles near the prior (AMCL style) to handle kidnapping.
     */
    void inject_recovery_particles(const gtsam::Pose3& prior, double radius);

    TRNConfig config_;                       ///< Analytical settings
    mutable std::mutex mtx_;                 ///< Thread-safety guard

    std::vector<Particle> particles_;        ///< Active particle set
    bool is_initialized_;                    ///< Solver warm-start flag

    // Digital Elevation Models (DEMs)
    Eigen::MatrixXf global_dem_;             ///< Global offline elevation map
    double global_res_;                      ///< Resolution of global DEM (m/px)
    double global_origin_x_;                 ///< Origin X of global DEM (m)
    double global_origin_y_;                 ///< Origin Y of global DEM (m)

    Eigen::MatrixXf latest_local_dem_;       ///< Scan-centered local elevation map
    double local_res_;                       ///< Resolution of local DEM (m/px)
    double local_origin_x_;                  ///< Origin X of local DEM (m)
    double local_origin_y_;                  ///< Origin Y of local DEM (m)
    bool local_dem_ready_;                   ///< Flag tracking active local scanning

    // Random Number Engine
    mutable std::mt19937 rand_engine_;       ///< Deterministic pseudorandom engine
};

} // namespace ugv_trn

#endif // UGV_TRN__TRN_CORE_HPP_
