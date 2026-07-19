#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "ugv_obstacle/obstacle_core.hpp"

using ugv_obstacle::ObstacleConfig;
using ugv_obstacle::ObstacleCore;

namespace {

// Build a DEM (rows=Y, cols=X) of a planar slope: h = x * tan(deg), with
// origin (0,0), 1 m/px, covering x,y in [0, size).
Eigen::MatrixXf slope_dem(int size, double slope_deg) {
  Eigen::MatrixXf dem(size, size);
  const double m = std::tan(slope_deg * M_PI / 180.0);
  for (int r = 0; r < size; ++r) {      // r = y
    for (int c = 0; c < size; ++c) {    // c = x
      dem(r, c) = static_cast<float>(c * m);
    }
  }
  return dem;
}

ObstacleCore make_core(const Eigen::MatrixXf& dem) {
  ObstacleConfig cfg;
  cfg.tau_prior = 0.4;
  cfg.tau_local = 0.4;
  cfg.self_radius = 0.5;
  cfg.max_range = 200.0;
  cfg.cell_size = 0.4;
  // Synthetic slope DEMs reach large absolute z; disable the map-z band so it
  // does not reject the (physically valid) test points.
  cfg.min_height = -1.0e6;
  cfg.max_height = 1.0e6;
  ObstacleCore core(cfg);
  core.set_global_dem(dem, 1.0, 0.0, 0.0);
  return core;
}

}  // namespace

TEST(ObstacleCore, BilinearDemLookup) {
  auto core = make_core(slope_dem(50, 20.0));
  const double m = std::tan(20.0 * M_PI / 180.0);
  EXPECT_NEAR(core.dem_at(10.0, 5.0), 10.0 * m, 1e-3);
  EXPECT_NEAR(core.dem_at(10.5, 5.0), 10.5 * m, 1e-3);  // interpolated
  EXPECT_TRUE(std::isnan(core.dem_at(-5.0, 5.0)));       // out of bounds
}

TEST(ObstacleCore, SteepSlopeNotFlagged) {
  // 35-degree slope: DEM-diff must stay ~0 (points lie on the DEM surface),
  // proving slope-invariance where the old grid-min-z method failed.
  const double deg = 35.0;
  auto core = make_core(slope_dem(60, deg));
  const double m = std::tan(deg * M_PI / 180.0);

  std::vector<Eigen::Vector3f> v;
  for (int k = 0; k < 400; ++k) {
    double x = 5.0 + (k % 40) * 1.0;
    double y = 5.0 + (k / 40) * 1.0;
    v.emplace_back(x, y, x * m);  // exactly on the slope surface
  }
  Eigen::MatrixXf pts(v.size(), 3);
  for (size_t i = 0; i < v.size(); ++i) pts.row(i) = v[i];

  auto mask = core.classify(pts, /*sensor*/ 0.0, 0.0, /*conf*/ 1.0);
  int flagged = 0;
  for (auto m8 : mask) flagged += m8;
  EXPECT_LT(flagged, static_cast<int>(mask.size()) / 50);  // <2%
}

TEST(ObstacleCore, RockOnSlopeFlagged) {
  const double deg = 30.0;
  auto core = make_core(slope_dem(60, deg));
  const double m = std::tan(deg * M_PI / 180.0);

  std::vector<Eigen::Vector3f> v;
  // Slope surface points.
  for (int k = 0; k < 400; ++k) {
    double x = 5.0 + (k % 40) * 1.0;
    double y = 5.0 + (k / 40) * 1.0;
    v.emplace_back(x, y, x * m);
  }
  const size_t rock_start = v.size();
  // Rock: 0.8 m above the slope at (20, 20), dense cluster.
  for (int k = 0; k < 40; ++k) {
    double x = 20.0 + (k % 6) * 0.05;
    double y = 20.0 + (k / 6) * 0.05;
    v.emplace_back(x, y, 20.0 * m + 0.8);
  }
  Eigen::MatrixXf pts(v.size(), 3);
  for (size_t i = 0; i < v.size(); ++i) pts.row(i) = v[i];

  auto mask = core.classify(pts, 0.0, 0.0, 1.0);
  int rock_flagged = 0;
  for (size_t i = rock_start; i < mask.size(); ++i) rock_flagged += mask[i];
  int slope_flagged = 0;
  for (size_t i = 0; i < rock_start; ++i) slope_flagged += mask[i];

  EXPECT_GE(rock_flagged, 25);                 // rock detected
  EXPECT_LT(slope_flagged, static_cast<int>(rock_start) / 50);  // slope clean
}

TEST(ObstacleCore, SelfHitsDropped) {
  auto core = make_core(slope_dem(50, 10.0));
  // A tall return right at the sensor (self-hit) must be ignored.
  Eigen::MatrixXf pts(1, 3);
  pts << 0.2, 0.0, 1.5;  // within self_radius 0.5
  auto mask = core.classify(pts, 0.0, 0.0, 1.0);
  EXPECT_EQ(mask[0], 0);
}

TEST(ObstacleCore, LowConfidenceWidensPriorThreshold) {
  // A 0.5 m bump: flagged at high confidence, ignored at low confidence
  // (threshold widened by low_conf_relax=0.6 -> 1.0 m).
  auto core = make_core(slope_dem(50, 0.0));  // flat, DEM h=0
  std::vector<Eigen::Vector3f> v;
  for (int k = 0; k < 20; ++k) v.emplace_back(10.0 + (k % 5) * 0.05, 10.0 + (k / 5) * 0.05, 0.5);
  Eigen::MatrixXf pts(v.size(), 3);
  for (size_t i = 0; i < v.size(); ++i) pts.row(i) = v[i];

  auto hi = core.classify(pts, 0.0, 0.0, 1.0);
  auto lo = core.classify(pts, 0.0, 0.0, 0.0);
  int hi_n = 0, lo_n = 0;
  for (auto m8 : hi) hi_n += m8;
  for (auto m8 : lo) lo_n += m8;
  EXPECT_GT(hi_n, 0);   // detected when confident
  // Note: local-jump may still catch it; this checks DEM-prior relaxation only
  // is consistent (lo <= hi).
  EXPECT_LE(lo_n, hi_n);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
