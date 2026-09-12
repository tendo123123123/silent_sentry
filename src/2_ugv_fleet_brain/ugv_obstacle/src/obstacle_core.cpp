#include "ugv_obstacle/obstacle_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <vector>

namespace ugv_obstacle {

namespace {
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Encode integer cell coords into one 64-bit key.
inline int64_t cell_key(int ix, int iy) {
  return (static_cast<int64_t>(ix) << 32) ^ (static_cast<int64_t>(iy) & 0xffffffff);
}
}  // namespace

void ObstacleCore::set_global_dem(const Eigen::MatrixXf& dem, double res,
                                  double origin_x, double origin_y) {
  dem_ = dem;
  res_ = res;
  origin_x_ = origin_x;
  origin_y_ = origin_y;
  rows_ = static_cast<int>(dem.rows());
  cols_ = static_cast<int>(dem.cols());
}

bool ObstacleCore::load_global_dem(const std::string& path, double res,
                                   double origin_x, double origin_y) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }
  uint32_t cols = 0, rows = 0;
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  if (cols == 0 || rows == 0 || cols > 100000 || rows > 100000) {
    return false;
  }
  // TRN binary is column-major float32; Eigen default is column-major.
  Eigen::MatrixXf dem(rows, cols);
  file.read(reinterpret_cast<char*>(dem.data()),
            static_cast<std::streamsize>(rows) * cols * sizeof(float));
  if (!file) {
    return false;
  }
  set_global_dem(dem, res, origin_x, origin_y);
  return true;
}

float ObstacleCore::dem_at(double x, double y) const {
  if (rows_ < 2 || cols_ < 2) {
    return kNaN;
  }
  const double col = (x - origin_x_) / res_;  // X index
  const double row = (y - origin_y_) / res_;  // Y index
  if (col < 0.0 || row < 0.0 || col > cols_ - 1 || row > rows_ - 1) {
    return kNaN;
  }
  const int c0 = static_cast<int>(std::floor(col));
  const int r0 = static_cast<int>(std::floor(row));
  const int c1 = std::min(c0 + 1, cols_ - 1);
  const int r1 = std::min(r0 + 1, rows_ - 1);
  const double fc = col - c0;
  const double fr = row - r0;
  const float v00 = dem_(r0, c0);
  const float v01 = dem_(r0, c1);
  const float v10 = dem_(r1, c0);
  const float v11 = dem_(r1, c1);
  const double top = v00 * (1.0 - fc) + v01 * fc;
  const double bot = v10 * (1.0 - fc) + v11 * fc;
  return static_cast<float>(top * (1.0 - fr) + bot * fr);
}

std::vector<uint8_t> ObstacleCore::classify(const Eigen::MatrixXf& pts_map,
                                            double sensor_x, double sensor_y,
                                            double confidence) const {
  const int n = static_cast<int>(pts_map.rows());
  std::vector<uint8_t> mask(n, 0);
  if (n == 0) {
    return mask;
  }

  const double conf = std::max(0.0, std::min(1.0, confidence));
  const double tau_prior_eff = cfg_.tau_prior + (1.0 - conf) * cfg_.low_conf_relax;
  const double self_r2 = cfg_.self_radius * cfg_.self_radius;
  const double max_r2 = cfg_.max_range * cfg_.max_range;
  const double inv_cell = 1.0 / cfg_.cell_size;

  // Setup 2D Flat Binned Array centered on sensor position.
  // Replaces std::unordered_map to provide O(1) cache-friendly memory access.
  const double grid_span = cfg_.max_range * 2.0 + 2.0;
  const int grid_dim = static_cast<int>(std::ceil(grid_span * inv_cell));
  const double grid_origin_x = sensor_x - cfg_.max_range - 1.0;
  const double grid_origin_y = sensor_y - cfg_.max_range - 1.0;

  std::vector<float> grid_min(grid_dim * grid_dim, std::numeric_limits<float>::infinity());
  std::vector<uint16_t> grid_cnt(grid_dim * grid_dim, 0);

  const double half_len = cfg_.chassis_length * 0.5;
  const double half_wid = cfg_.chassis_width * 0.5;

  std::vector<uint8_t> valid(n, 0);  // passed self/range/band gates
  std::vector<int> pt_col(n, -1);
  std::vector<int> pt_row(n, -1);

  // Pass 1: O(N) Populate 2D binned array with cell min Z and point counts.
  for (int i = 0; i < n; ++i) {
    const double x = pts_map(i, 0);
    const double y = pts_map(i, 1);
    const double z = pts_map(i, 2);
    const double dx = x - sensor_x;
    const double dy = y - sensor_y;
    const double r2 = dx * dx + dy * dy;

    // Range gates
    if (r2 < self_r2 || r2 > max_r2) continue;
    if (z < cfg_.min_height || z > cfg_.max_height) continue;

    // OBB chassis box filter (if enabled)
    if (cfg_.use_chassis_box_filter && std::abs(dx) < half_len && std::abs(dy) < half_wid) {
      continue;
    }

    const int col = static_cast<int>((x - grid_origin_x) * inv_cell);
    const int row = static_cast<int>((y - grid_origin_y) * inv_cell);

    if (col < 0 || col >= grid_dim || row < 0 || row >= grid_dim) continue;

    valid[i] = 1;
    pt_col[i] = col;
    pt_row[i] = row;

    const int idx = row * grid_dim + col;
    if (z < grid_min[idx]) {
      grid_min[idx] = static_cast<float>(z);
    }
    if (grid_cnt[idx] < 65535) {
      grid_cnt[idx]++;
    }
  }

  // Estimate DEM elevation offset (median Z_min - DEM_z)
  double dem_offset = 0.0;
  if (rows_ >= 2 && cols_ >= 2) {
    std::vector<double> offs;
    offs.reserve(grid_dim * grid_dim / 4);
    for (int r = 0; r < grid_dim; ++r) {
      for (int c = 0; c < grid_dim; ++c) {
        const int idx = r * grid_dim + c;
        if (grid_cnt[idx] < cfg_.min_points_per_cell) continue;
        const double cx = grid_origin_x + (c + 0.5) * cfg_.cell_size;
        const double cy = grid_origin_y + (r + 0.5) * cfg_.cell_size;
        const float h = dem_at(cx, cy);
        if (std::isfinite(h)) {
          offs.push_back(static_cast<double>(grid_min[idx]) - h);
        }
      }
    }
    if (offs.size() >= 8) {
      const size_t mid = offs.size() / 2;
      std::nth_element(offs.begin(), offs.begin() + mid, offs.end());
      dem_offset = offs[mid];
    }
  }

  // Pass 2: Classify with fast O(1) 2D array lookups and slope-compensated local jump.
  for (int i = 0; i < n; ++i) {
    if (!valid[i]) continue;
    const double x = pts_map(i, 0);
    const double y = pts_map(i, 1);
    const double z = pts_map(i, 2);
    const int col = pt_col[i];
    const int row = pt_row[i];
    const int self_idx = row * grid_dim + col;

    // 3x3 local neighborhood ground estimation
    float min_ground = std::numeric_limits<float>::infinity();
    float max_ground = -std::numeric_limits<float>::infinity();
    int valid_neighbors = 0;

    for (int dr = -1; dr <= 1; ++dr) {
      const int r = row + dr;
      if (r < 0 || r >= grid_dim) continue;
      for (int dc = -1; dc <= 1; ++dc) {
        const int c = col + dc;
        if (c < 0 || c >= grid_dim) continue;
        const int idx = r * grid_dim + c;
        const float g = grid_min[idx];
        if (std::isfinite(g)) {
          min_ground = std::min(min_ground, g);
          max_ground = std::max(max_ground, g);
          valid_neighbors++;
        }
      }
    }

    const uint16_t cnt = grid_cnt[self_idx];
    const bool enough = cnt >= cfg_.min_points_per_cell;

    // Dune Slope Compensation:
    // If the 3x3 neighborhood covers a natural slope, max_ground - min_ground
    // reflects terrain slope drop across ~1.2m. We compensate ground baseline.
    float slope_comp = 0.0f;
    if (valid_neighbors >= 3 && std::isfinite(max_ground) && std::isfinite(min_ground)) {
      slope_comp = 0.5f * (max_ground - min_ground);
    }

    const float effective_ground = min_ground + slope_comp;
    const bool local_obs =
        enough && std::isfinite(min_ground) && (z - effective_ground > cfg_.tau_local);

    // DEM-prior difference (slope-invariant).
    bool dem_obs = false;
    const float h = dem_at(x, y);
    if (std::isfinite(h)) {
      dem_obs = enough && (z - (h + dem_offset) > tau_prior_eff);
    }

    mask[i] = (local_obs || dem_obs) ? 1 : 0;
  }
  return mask;
}

}  // namespace ugv_obstacle
