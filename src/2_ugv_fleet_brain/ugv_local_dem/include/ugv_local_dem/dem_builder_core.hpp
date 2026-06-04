/**
 * @file dem_builder_core.hpp
 * @brief High-performance, ROS-independent rolling local Digital Elevation Model (DEM) builder core.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#ifndef UGV_LOCAL_DEM__DEM_BUILDER_CORE_HPP_
#define UGV_LOCAL_DEM__DEM_BUILDER_CORE_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace ugv_local_dem {

/**
 * @struct DEMBuilderConfig
 * @brief Parameters governing local DEM generation.
 */
struct DEMBuilderConfig {
    double grid_resolution;          ///< Size of each cell (meters, e.g., 1.0)
    double grid_size_x;              ///< Width of local grid (meters, e.g., 20.0)
    double grid_size_y;              ///< Height of local grid (meters, e.g., 20.0)
    
    double deskew_scan_period;       ///< LiDAR spin period (seconds, e.g., 0.1 for 10Hz)
    bool deskew_clockwise;           ///< Clockwise scan direction flag
    double rolling_submap_distance;  ///< Maximum horizontal travel distance for old chunk pruning (meters)
    double submap_spatial_bin_size;  ///< Size of indexing spatial hash bins (meters)
    double uamc_drift_variance;      ///< Drift variance for Uncertainty-Aware Motion Correction weights
    
    double ground_height_min;        ///< Floor for ground filtering relative to chassis (meters)
    double ground_height_max;        ///< Ceiling for ground filtering relative to chassis (meters)
    double obstacle_height_threshold; ///< Delta threshold to reject steep obstacle spikes (meters)
    double ransac_distance_threshold; ///< Max normal distance for RANSAC ground model inliers (meters)
    int ransac_iterations;           ///< Maximum RANSAC trials per scan
    int min_points_per_cell;         ///< Minimum point count to mark grid cell observed

    double min_range;                ///< Inner lidar blind range (meters)
    double max_range;                ///< Outer active range limit (meters)
    double spawn_elevation;          ///< Metric elevation offset at spawn
};

/**
 * @struct SubmapChunk
 * @brief Aligned and weighted point-cloud chunk.
 */
struct SubmapChunk {
    Eigen::MatrixXf points;          ///< (N, 3) coordinates in map/odom frame
    Eigen::VectorXf weights;         ///< (N, 1) confidence weights [0.0, 1.0]
};

/**
 * @class DEMBuilderCore
 * @brief Core geometry and linear algebra engine for generating local digital elevation maps.
 * 
 * Fully decoupled from ROS. Implements fast OpenMP-parallelized point filters,
 * azimuth-based scan deskewing, RANSAC-driven terrain plane segmentation, and spatial hash binning.
 */
class DEMBuilderCore {
public:
    /**
     * @brief Construct a new DEMBuilderCore object.
     * @param config Analytical settings of the pipeline.
     */
    explicit DEMBuilderCore(const DEMBuilderConfig& config);

    /**
     * @brief Default Destructor.
     */
    ~DEMBuilderCore() = default;

    /**
     * @brief Update chassis orientation from gravity vector estimators.
     * @param roll Pitch alignment orientation (radians)
     * @param pitch Pitch alignment orientation (radians)
     */
    void update_imu_orientation(double roll, double pitch);

    /**
     * @brief Update chassis-frame translational and rotational velocities.
     * @param linear Body-frame velocity vector (m/s)
     * @param angular Body-frame rotational velocity vector (rad/s)
     */
    void update_body_velocity(const Eigen::Vector3f& linear, const Eigen::Vector3f& angular);

    /**
     * @brief Queue a raw 3D scan points array for background processing.
     * @param points Matrix of (N, 3) raw point coordinates in LiDAR sensor frame.
     */
    void enqueue_cloud(const Eigen::MatrixXf& points);

    /**
     * @brief Feed, align, filter, and register a ground-detected cloud chunk into spatial bins.
     * @param points_base Raw cloud converted to base vehicle body frame.
     * @param sensor_origin Body-frame LiDAR physical translation vector.
     * @param robot_pose_odom Active robot metric pose in local odometry frame.
     * @param odom_travel Cumulative forward distance traversed (meters).
     * @return bool True if the chunk was successfully registered; false if skipped/invalid.
     */
    bool accumulate_ground_cloud(
        const Eigen::MatrixXf& points_base,
        const Eigen::Vector3f& sensor_origin,
        const Eigen::Vector3f& robot_pose_odom,
        double odom_travel
    );

    /**
     * @brief Retrieve, rasterize, smooth, and export the rolling local elevation map.
     * @param robot_pose_odom Active robot metric pose in local odometry frame.
     * @param out_grid Dense 2D output grid representing elevation heights.
     * @param out_origin_x Calculated grid minimum-X global coordinate (meters).
     * @param out_origin_y Calculated grid minimum-Y global coordinate (meters).
     * @return bool True if a valid local DEM map was built; false if empty or incomplete.
     */
    bool build_dem(
        const Eigen::Vector3f& robot_pose_odom,
        Eigen::MatrixXf& out_grid,
        double& out_origin_x,
        double& out_origin_y
    );

    /**
     * @brief Perform spatial closing morphology on the elevation grid.
     */
    Eigen::MatrixXf morph_close(const Eigen::MatrixXf& grid, int iterations = 3) const;

    /**
     * @brief Scan and clear grid cells corresponding to sharp non-negotiable cliff obstacle spikes.
     */
    Eigen::MatrixXf reject_obstacle_cells(const Eigen::MatrixXf& grid) const;

    /**
     * @brief Thread-safe configuration update.
     */
    void update_config(const DEMBuilderConfig& config);

private:
    /**
     * @brief Mask out points striking the vehicle's body structures (chassis, deck, mast, flanges).
     */
    Eigen::MatrixXf filter_self_hits(const Eigen::MatrixXf& points) const;

    /**
     * @brief Interpolate relative timestamp offset [0, scan_period] based on point azimuth.
     */
    Eigen::VectorXf compute_relative_scan_times(const Eigen::MatrixXf& points, const Eigen::Vector3f& sensor_origin) const;

    /**
     * @brief Generate dynamic weights based on uncertainty propagation decay.
     */
    Eigen::VectorXf compute_uamc_weights(const Eigen::VectorXf& rel_times) const;

    /**
     * @brief Apply a continuous-time rigid transform to eliminate LiDAR motion distortion.
     */
    Eigen::MatrixXf deskew_points(
        const Eigen::MatrixXf& points,
        const Eigen::Vector3f& sensor_origin,
        const Eigen::VectorXf& rel_times
    ) const;

    /**
     * @brief Re-project points into gravity-aligned frame using IMU roll and pitch.
     */
    Eigen::MatrixXf gravity_align_points(const Eigen::MatrixXf& points) const;

    /**
     * @brief Extract ground inliers using 3-Point RANSAC.
     */
    Eigen::MatrixXf segment_ground(const Eigen::MatrixXf& points, std::vector<bool>& out_mask) const;

    /**
     * @brief Index ground points into spatial hash bins.
     */
    void index_submap_chunk(double travel, const Eigen::MatrixXf& points_odom, const Eigen::VectorXf& weights);

    /**
     * @brief Discard old submap chunks beyond horizontal rolling boundary distance.
     */
    void prune_submap_chunks(double travel);

    // Spatial Hashing types
    typedef std::pair<int, int> BinKey;
    struct BinKeyHash {
        std::size_t operator()(const BinKey& k) const {
            return (std::hash<int>()(k.first) ^ (std::hash<int>()(k.second) << 1));
        }
    };

    DEMBuilderConfig config_;                  ///< Settings
    mutable std::mutex mtx_;                   ///< Mutex thread-safety guard

    // Internal Motion States
    double imu_roll_;                          ///< Roll angle (radians)
    double imu_pitch_;                         ///< Pitch angle (radians)
    Eigen::Vector3f body_linear_vel_;          ///< Translation velocity (m/s)
    Eigen::Vector3f body_angular_vel_;         ///< Rotation velocity (rad/s)

    // Queued pending raw scans
    std::deque<Eigen::MatrixXf> pending_clouds_; ///< Scan deque

    // Rolling Submap Buffers
    uint64_t next_chunk_id_;                   ///< Global counter of chunk IDs
    double cumulative_travel_;                 ///< Track total odometric movement
    Eigen::Vector2f last_chunk_pose_xy_;       ///< Tracks increments
    bool last_chunk_pose_initialized_;         ///< Flag

    std::deque<std::pair<double, uint64_t>> submap_chunks_; ///< Queue tracking chunk age: (travel, chunk_id)
    std::unordered_map<uint64_t, std::vector<BinKey>> chunk_bin_keys_; ///< Map: chunk_id -> list of bin keys it resides in
    
    // Hash map for spatial bins: BinKey -> Map of (chunk_id -> chunk data)
    std::unordered_map<BinKey, std::unordered_map<uint64_t, SubmapChunk>, BinKeyHash> submap_spatial_bins_;
};

} // namespace ugv_local_dem

#endif // UGV_LOCAL_DEM__DEM_BUILDER_CORE_HPP_
