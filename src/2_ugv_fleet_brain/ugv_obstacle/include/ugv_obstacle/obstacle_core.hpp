// DEM-prior obstacle detection core (ROS-free, Eigen-only).
//
// An obstacle is a LiDAR return that rises above the KNOWN terrain surface.
// We use two complementary signals:
//   1. DEM-prior differencing (slope-invariant): the a-priori global DEM
//      already contains the dune shape, so a point on any slope matches it
//      (diff ~ 0). Only rocks/vegetation rise above -> obstacle. Requires a
//      reasonable map->base pose; its threshold is widened when localization
//      confidence is low.
//   2. Local-jump (localization-free): a point far above the local
//      neighbourhood minimum indicates a vertical structure regardless of the
//      prior or pose drift.
// A point is an obstacle if EITHER fires. Points within self_radius of the
// sensor (robot body/near ground) are dropped.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace ugv_obstacle {

struct ObstacleConfig {
  double dem_res = 1.0;          // global DEM resolution (m/px)
  double dem_origin_x = -450.0;  // DEM bottom-left corner X (m, map frame)
  double dem_origin_y = -150.0;  // DEM bottom-left corner Y (m, map frame)
  double tau_prior = 0.4;        // height above DEM prior -> obstacle (m)
  double tau_local = 0.4;        // local neighbourhood jump -> obstacle (m)
  double self_radius = 1.2;      // drop returns within this of sensor (m)
  double max_range = 30.0;       // drop returns beyond this (m)
  double min_height = -2.0;      // map-frame z band, low cut (m)
  double max_height = 4.0;       // map-frame z band, high cut (m)
  double cell_size = 0.4;        // local-jump grid cell (m)
  int min_points_per_cell = 2;   // sparse cells not classified as obstacle
  double low_conf_relax = 0.6;   // extra tau_prior added when confidence = 0
};

class ObstacleCore {
 public:
  ObstacleCore() = default;
  explicit ObstacleCore(const ObstacleConfig& cfg) : cfg_(cfg) {}

  const ObstacleConfig& config() const { return cfg_; }

  // Load global DEM from the TRN binary format:
  //   uint32 cols, uint32 rows, then rows*cols float32 (column-major).
  bool load_global_dem(const std::string& path, double res,
                       double origin_x, double origin_y);

  // Inject a DEM directly (rows=Y index, cols=X index). Used by tests.
  void set_global_dem(const Eigen::MatrixXf& dem, double res,
                      double origin_x, double origin_y);

  bool has_dem() const { return dem_.size() > 0; }

  // Bilinear DEM elevation at map (x, y). Returns NaN if out of bounds / no DEM.
  float dem_at(double x, double y) const;

  // Classify map-frame points (Nx3). sensor_(x,y) = sensor origin in map.
  // confidence in [0,1] (e.g. TRN match quality). Returns a mask (len N),
  // 1 = obstacle.
  std::vector<uint8_t> classify(const Eigen::MatrixXf& pts_map,
                                double sensor_x, double sensor_y,
                                double confidence) const;

 private:
  ObstacleConfig cfg_;
  Eigen::MatrixXf dem_;  // rows(Y) x cols(X)
  double res_ = 1.0;
  double origin_x_ = 0.0;
  double origin_y_ = 0.0;
  int rows_ = 0;
  int cols_ = 0;
};

}  // namespace ugv_obstacle
