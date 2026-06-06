/**
 * @file trn_core.cpp
 * @brief High-performance, ROS-independent Terrain-Referenced Navigation (TRN) core implementation.
 * @author Silent Sentry Robotics Team
 * @date June 2026
 */

#include "ugv_trn/trn_core.hpp"

#include <gtsam/geometry/Pose2.h>

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace ugv_trn {

// Helper to wrap angle between [-pi, pi]
static double wrap_angle(double angle) {
    return std::remainder(angle, 2.0 * M_PI);
}

TRNCore::TRNCore(const TRNConfig& config)
    : config_(config),
      is_initialized_(false),
      global_res_(1.0),
      global_origin_x_(0.0),
      global_origin_y_(0.0),
      local_res_(1.0),
      local_origin_x_(0.0),
      local_origin_y_(0.0),
      local_dem_ready_(false)
{
    // Initialize random engine with deterministic seed for reproducibility in defense trials
    std::random_device rd;
    rand_engine_.seed(12345); // Fixed seed for deterministic execution
}

bool TRNCore::load_global_dem(const std::string& filepath, double resolution, double origin_x, double origin_y)
{
    std::lock_guard<std::mutex> lock(mtx_);
    global_res_ = resolution;
    global_origin_x_ = origin_x;
    global_origin_y_ = origin_y;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "TRN: Global DEM file not found at: " << filepath 
                  << ". Generating synthetic desert dune topography for simulation safety!" << std::endl;
        
        // Generate a 500x500 synthetic sand dune elevation grid
        const int rows = 500;
        const int cols = 500;
        global_dem_ = Eigen::MatrixXf::Zero(rows, cols);

        // If origins are default 0.0, center the synthetic map around the origin
        if (std::abs(origin_x) < 1e-4 && std::abs(origin_y) < 1e-4) {
            global_origin_x_ = -(cols * resolution) / 2.0;
            global_origin_y_ = -(rows * resolution) / 2.0;
        }

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                double gx = global_origin_x_ + c * resolution;
                double gy = global_origin_y_ + r * resolution;
                // Double-sine wave representing linear wind-swept sand dunes
                global_dem_(r, c) = static_cast<float>(
                    10.0 * std::sin(gx / 60.0) * std::cos(gy / 60.0) +
                     2.0 * std::sin(gx / 15.0)
                );
            }
        }
        return true;
    }

    // Binary height format: expects 2D raw float array
    // Read width, height, then raw data
    uint32_t cols = 0, rows = 0;
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));

    if (cols == 0 || rows == 0 || cols > 10000 || rows > 10000) {
        std::cerr << "TRN: Invalid DEM dimension binary headers! Cols: " << cols << ", Rows: " << rows << std::endl;
        return false;
    }

    global_dem_ = Eigen::MatrixXf(rows, cols);
    file.read(reinterpret_cast<char*>(global_dem_.data()), rows * cols * sizeof(float));
    file.close();

    std::cout << "TRN: Successfully loaded Global DEM grid (" << cols << "x" << rows << ") from binary file." << std::endl;
    return true;
}

void TRNCore::update_local_dem(const Eigen::MatrixXf& local_grid, double resolution, double origin_x, double origin_y)
{
    std::lock_guard<std::mutex> lock(mtx_);
    latest_local_dem_ = local_grid;
    local_res_ = resolution;
    local_origin_x_ = origin_x;
    local_origin_y_ = origin_y;
    local_dem_ready_ = true;
}

void TRNCore::propagate_particles(double local_dx, double local_dy, double dyaw)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (particles_.empty()) {
        return;
    }

    const double dist = std::hypot(local_dx, local_dy);
    const double std_dev_xy = dist * config_.motion_noise_xy_frac + 0.05;
    const double std_dev_yaw = std::abs(dyaw) * config_.motion_noise_yaw_frac + 0.005;

    std::normal_distribution<double> noise_x(0.0, std_dev_xy);
    std::normal_distribution<double> noise_y(0.0, std_dev_xy);
    std::normal_distribution<double> noise_yaw(0.0, std_dev_yaw);

    // local_dx, local_dy are in the robot's base frame.
    // We project them into the map frame using each particle's own map->base yaw.
    for (auto& p : particles_) {
        double cos_p = std::cos(p.yaw);
        double sin_p = std::sin(p.yaw);
        
        p.x += local_dx * cos_p - local_dy * sin_p + noise_x(rand_engine_);
        p.y += local_dx * sin_p + local_dy * cos_p + noise_y(rand_engine_);
        p.yaw = wrap_angle(p.yaw + dyaw + noise_yaw(rand_engine_));
    }
}

Eigen::MatrixXf TRNCore::bilateral_filter(const Eigen::MatrixXf& input) const
{
    // Reformat Eigen Matrix to OpenCV float matrix
    cv::Mat cv_input(input.rows(), input.cols(), CV_32FC1, const_cast<float*>(input.data()));

    // Bilateral filter cannot process NaNs directly.
    // We create a valid mask, temporarily replace NaNs with the mean value, and restore NaNs in output.
    cv::Mat mask = (cv_input == cv_input); // NaN check: NaN != NaN
    double mean_val = 0.0;
    
    cv::Scalar mean_scalar = cv::mean(cv_input, mask);
    mean_val = mean_scalar[0];

    cv::Mat cv_filled = cv_input.clone();
    cv_filled.setTo(mean_val, ~mask);

    cv::Mat cv_filtered;
    // Apply OpenCV high-speed bilateral filter
    cv::bilateralFilter(
        cv_filled, 
        cv_filtered, 
        config_.bilateral_d, 
        config_.bilateral_sigma_color, 
        config_.bilateral_sigma_space
    );

    Eigen::MatrixXf output(input.rows(), input.cols());
    std::memcpy(output.data(), cv_filtered.data, input.rows() * input.cols() * sizeof(float));

    // Restore NaNs for cell validity tracking
    for (int r = 0; r < input.rows(); ++r) {
        for (int c = 0; c < input.cols(); ++c) {
            if (std::isnan(input(r, c))) {
                output(r, c) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    return output;
}

double TRNCore::compute_spatial_entropy(const Eigen::MatrixXf& grid) const
{
    std::vector<float> heights;
    heights.reserve(grid.size());

    float min_h = std::numeric_limits<float>::max();
    float max_h = std::numeric_limits<float>::lowest();

    for (int i = 0; i < grid.size(); ++i) {
        float val = grid.data()[i];
        if (std::isfinite(val)) {
            heights.push_back(val);
            min_h = std::min(min_h, val);
            max_h = std::max(max_h, val);
        }
    }

    if (heights.size() < 100 || min_h >= max_h) {
        return 0.0;
    }

    // Classify observed terrain heights into 50 histogram bins
    const int num_bins = 50;
    std::vector<double> hist(num_bins, 0.0);
    const double range = max_h - min_h;

    for (float h : heights) {
        int bin = static_cast<int>((h - min_h) / range * (num_bins - 1));
        bin = std::clamp(bin, 0, num_bins - 1);
        hist[bin] += 1.0;
    }

    // Compute Shannon Entropy
    double entropy = 0.0;
    const double total_samples = static_cast<double>(heights.size());
    for (double count : hist) {
        if (count > 0.0) {
            double p = count / total_samples;
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

double TRNCore::evaluate_particle_likelihood(const Particle& p, const Eigen::MatrixXf& local_dem_filtered, const gtsam::Pose3& odom_prior) const
{
    double accum_diff = 0.0;
    uint64_t valid_overlap_count = 0;

    const int local_rows = local_dem_filtered.rows();
    const int local_cols = local_dem_filtered.cols();

    // The particle p represents the robot's base_footprint pose in the map frame (map->base).
    // The grid cells are aligned to the odom frame.
    // To transform an odom coordinate to the map frame using the particle's hypothesis,
    // we compute the implied map->odom transform: T_{map->odom} = T_{map->base} * (T_{odom->base})^-1
    
    gtsam::Pose2 map_to_base(p.x, p.y, p.yaw);
    gtsam::Pose2 odom_to_base(odom_prior.x(), odom_prior.y(), odom_prior.rotation().yaw());
    gtsam::Pose2 map_to_odom = map_to_base.compose(odom_to_base.inverse());

    const double cos_yaw = std::cos(map_to_odom.theta());
    const double sin_yaw = std::sin(map_to_odom.theta());
    const double tx = map_to_odom.x();
    const double ty = map_to_odom.y();

    // Loop over the local scan cells
    for (int r = 0; r < local_rows; ++r) {
        for (int c = 0; c < local_cols; ++c) {
            float local_height = local_dem_filtered(r, c);
            if (!std::isfinite(local_height)) {
                continue;
            }

            // Convert cell grid index to metric coordinates in the odom frame
            double u = local_origin_x_ + c * local_res_;
            double v = local_origin_y_ + r * local_res_;

            // Rigid transform onto map frame using the implied map->odom transform
            double gx = tx + u * cos_yaw - v * sin_yaw;
            double gy = ty + u * sin_yaw + v * cos_yaw;

            // Map continuous metric coordinates to nearest global reference pixel index
            int gc = static_cast<int>(std::round((gx - global_origin_x_) / global_res_));
            int gr = static_cast<int>(std::round((gy - global_origin_y_) / global_res_));

            if (gc >= 0 && gc < global_dem_.cols() && gr >= 0 && gr < global_dem_.rows()) {
                float global_height = global_dem_(gr, gc);
                if (std::isfinite(global_height)) {
                    accum_diff += std::abs(local_height - global_height);
                    valid_overlap_count++;
                }
            }
        }
    }

    // Count the total number of valid local cells to compute a fair overlap percentage
    uint64_t total_valid_local = 0;
    for (int r = 0; r < local_rows; ++r) {
        for (int c = 0; c < local_cols; ++c) {
            if (std::isfinite(local_dem_filtered(r, c))) {
                total_valid_local++;
            }
        }
    }

    // Reject particles with insufficient grid overlap (defense criteria: < 50% of valid local cells or < 10 absolute cells)
    if (total_valid_local < 10) {
        return -1.0;
    }
    
    const uint64_t min_overlap = std::max(static_cast<uint64_t>(10), static_cast<uint64_t>(total_valid_local * 0.50));
    if (valid_overlap_count < min_overlap) {
        return -1.0;
    }

    // Gaussian likelihood from Mean Absolute Difference (MAD)
    double mad = accum_diff / static_cast<double>(valid_overlap_count);
    double score = std::exp(-2.0 * mad);
    return std::max(score, 1e-12);
}

void TRNCore::systematic_resample()
{
    const int n = config_.num_particles;
    std::vector<double> cdf(n, 0.0);
    cdf[0] = particles_[0].weight;
    for (int i = 1; i < n; ++i) {
        cdf[i] = cdf[i - 1] + particles_[i].weight;
    }

    // Low-variance systematic resampling
    std::uniform_real_distribution<double> dist(0.0, 1.0 / static_cast<double>(n));
    const double start_val = dist(rand_engine_);

    std::vector<Particle> resampled;
    resampled.reserve(n);

    int idx = 0;
    for (int i = 0; i < n; ++i) {
        double target = start_val + static_cast<double>(i) / static_cast<double>(n);
        while (target > cdf[idx] && idx < n - 1) {
            idx++;
        }
        Particle p = particles_[idx];
        p.weight = 1.0 / static_cast<double>(n);
        resampled.push_back(p);
    }

    particles_ = std::move(resampled);
}

void TRNCore::inject_recovery_particles(const gtsam::Pose3& prior, double radius)
{
    const double px = prior.x();
    const double py = prior.y();
    const double pyaw = prior.rotation().yaw();

    std::normal_distribution<double> dist_x(px, radius * 0.5);
    std::normal_distribution<double> dist_y(py, radius * 0.5);
    std::normal_distribution<double> dist_yaw(pyaw, config_.motion_noise_yaw_frac * 5.0);

    // Inject 80% random recovery, keep 20% at current positions with small noise
    int n_inject = static_cast<int>(particles_.size() * 0.8);
    for (int i = 0; i < n_inject; ++i) {
        particles_[i].x = dist_x(rand_engine_);
        particles_[i].y = dist_y(rand_engine_);
        particles_[i].yaw = wrap_angle(dist_yaw(rand_engine_));
        particles_[i].weight = 1.0 / static_cast<double>(particles_.size());
    }
    std::normal_distribution<double> keep_noise(0.0, radius * 0.05);
    for (int i = n_inject; i < static_cast<int>(particles_.size()); ++i) {
        particles_[i].x += keep_noise(rand_engine_);
        particles_[i].y += keep_noise(rand_engine_);
        particles_[i].yaw = wrap_angle(particles_[i].yaw + keep_noise(rand_engine_));
        particles_[i].weight = 1.0 / static_cast<double>(particles_.size());
    }
}

bool TRNCore::execute_match_cycle(
    const gtsam::Pose3& odom_prior,
    const gtsam::Pose3& map_to_odom_current,
    gtsam::Pose3& out_pose,
    Eigen::Matrix<double, 6, 6>& out_covariance,
    double& match_quality)
{
    std::lock_guard<std::mutex> lock(mtx_);
    match_quality = 0.0;

    if (global_dem_.size() == 0 || !local_dem_ready_) {
        return false;
    }

    // Predict current map-frame coordinate prior
    gtsam::Pose3 map_prior = map_to_odom_current.compose(odom_prior);

    // Compute Shannon Entropy on raw composite grid
    double entropy = compute_spatial_entropy(latest_local_dem_);
    if (entropy < config_.entropy_threshold) {
        return false; // Flatten surface: abort matching cycle
    }

    // Flatness standard-deviation gate (config param was previously unused)
    double height_sum = 0.0, height_sq_sum = 0.0;
    int valid_cells = 0;
    for (int i = 0; i < latest_local_dem_.size(); ++i) {
        float val = latest_local_dem_.data()[i];
        if (std::isfinite(val)) {
            height_sum += val;
            height_sq_sum += val * val;
            valid_cells++;
        }
    }
    if (valid_cells > 10) {
        double mean = height_sum / valid_cells;
        double std_dev = std::sqrt(std::max(0.0, height_sq_sum / valid_cells - mean * mean));
        if (std_dev < config_.flatness_std_threshold) {
            return false; // Too flat: no texture for matching
        }
    }

    // Apply high-fidelity bilateral filtering to smooth lidar scans
    Eigen::MatrixXf local_filtered = bilateral_filter(latest_local_dem_);

    const int n = config_.num_particles;
    
    // Cold-start initialization (first iteration)
    if (!is_initialized_ || particles_.empty()) {
        particles_.resize(n);
        double cold_start_radius = config_.initial_search_radius;
        std::normal_distribution<double> cold_x(map_prior.x(), cold_start_radius * 0.25);
        std::normal_distribution<double> cold_y(map_prior.y(), cold_start_radius * 0.25);
        std::normal_distribution<double> cold_yaw(map_prior.rotation().yaw(), 0.10);

        for (auto& p : particles_) {
            p.x = cold_x(rand_engine_);
            p.y = cold_y(rand_engine_);
            p.yaw = wrap_angle(cold_yaw(rand_engine_));
            p.weight = 1.0 / static_cast<double>(n);
        }
        is_initialized_ = true;
    }

    // Evaluate scores for active particles
    std::vector<double> scores(n, 0.0);
    double score_sum = 0.0;
    double max_score = 0.0;
    int valid_scores_count = 0;

    for (int i = 0; i < n; ++i) {
        double likelihood = evaluate_particle_likelihood(particles_[i], local_filtered, odom_prior);
        if (likelihood >= 0.0) {
            scores[i] = likelihood;
            score_sum += likelihood;
            max_score = std::max(max_score, likelihood);
            valid_scores_count++;
        } else {
            scores[i] = 1e-12; // Out of bounds or invalid overlap
        }
    }

    match_quality = max_score;

    // Kidney / Lost recovery check
    if (valid_scores_count < static_cast<int>(n * 0.05) || score_sum < 1e-10) {
        std::cerr << "TRN: Insufficient valid particle coverage! Triggering global kidnapping recovery." << std::endl;
        inject_recovery_particles(map_prior, config_.max_search_radius);
        return false;
    }

    // Normalize weights
    for (int i = 0; i < n; ++i) {
        particles_[i].weight = scores[i] / score_sum;
    }

    // Compute Effective Sample Size (ESS) to track degeneration
    double ess_denom = 0.0;
    for (const auto& p : particles_) {
        ess_denom += p.weight * p.weight;
    }
    double ess = 1.0 / ess_denom;
    double ess_ratio = ess / static_cast<double>(n);

    // Execute systematic resampling if ESS drops below threshold
    if (ess_ratio < config_.ess_threshold) {
        systematic_resample();
        
        // AMCL random recovery injection to prevent local convergence lock
        if (config_.amcl_random_fraction > 0.0) {
            int n_recovery = static_cast<int>(n * config_.amcl_random_fraction);
            std::normal_distribution<double> rec_x(map_prior.x(), config_.base_search_radius * 0.5);
            std::normal_distribution<double> rec_y(map_prior.y(), config_.base_search_radius * 0.5);
            std::normal_distribution<double> rec_yaw(map_prior.rotation().yaw(), 0.15);

            for (int i = 0; i < n_recovery; ++i) {
                int index = std::uniform_int_distribution<int>(0, n - 1)(rand_engine_);
                particles_[index].x = rec_x(rand_engine_);
                particles_[index].y = rec_y(rand_engine_);
                particles_[index].yaw = wrap_angle(rec_yaw(rand_engine_));
                particles_[index].weight = 1.0 / static_cast<double>(n);
            }
        }
    }

    // Reject matches below peak quality threshold
    if (max_score < config_.min_peak_quality) {
        return false;
    }

    // Compute weighted mean pose (Global Output Pose on SE(2) manifold)
    double mean_x = 0.0;
    double mean_y = 0.0;
    double sin_yaw_sum = 0.0;
    double cos_yaw_sum = 0.0;

    for (const auto& p : particles_) {
        mean_x += p.weight * p.x;
        mean_y += p.weight * p.y;
        sin_yaw_sum += p.weight * std::sin(p.yaw);
        cos_yaw_sum += p.weight * std::cos(p.yaw);
    }
    double mean_yaw = std::atan2(sin_yaw_sum, cos_yaw_sum);

    // Compute weighted covariance matrix on SE(2)
    double var_x = 0.0;
    double var_y = 0.0;
    double var_yaw = 0.0;
    double cov_xy = 0.0;

    for (const auto& p : particles_) {
        double dx = p.x - mean_x;
        double dy = p.y - mean_y;
        double dyaw = wrap_angle(p.yaw - mean_yaw);

        var_x += p.weight * dx * dx;
        var_y += p.weight * dy * dy;
        var_yaw += p.weight * dyaw * dyaw;
        cov_xy += p.weight * dx * dy;
    }

    // Configure 6x6 Pose3 covariance matrix
    out_covariance.setZero();
    // Set planar orientation variances
    out_covariance(0, 0) = 1e-4; // Roll variance (tight planar constraint)
    out_covariance(1, 1) = 1e-4; // Pitch variance (tight planar constraint)
    out_covariance(2, 2) = std::max(var_yaw, 1e-4); // Yaw variance

    // Set planar position variances
    out_covariance(3, 3) = std::max(var_x, 0.04);
    out_covariance(4, 4) = std::max(var_y, 0.04);
    out_covariance(3, 4) = cov_xy;
    out_covariance(4, 3) = cov_xy;
    out_covariance(5, 5) = 1e-4; // Vertical Z variance (tight planar constraint)

    out_pose = gtsam::Pose3(gtsam::Rot3::Yaw(mean_yaw), gtsam::Point3(mean_x, mean_y, map_prior.z()));

    return true;
}

void TRNCore::update_config(const TRNConfig& config)
{
    std::lock_guard<std::mutex> lock(mtx_);
    config_ = config;
}

std::vector<Particle> TRNCore::get_particles() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return particles_;
}

} // namespace ugv_trn
