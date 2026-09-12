// DEM-prior obstacle detector node.
//
// Subscribes the 3D LiDAR cloud, transforms it to the map frame, classifies
// obstacles against the a-priori DEM (+ local-jump, + TRN-confidence-scaled
// threshold), and republishes an obstacle-only cloud in the ORIGINAL sensor
// frame so the Nav2 costmap raytrace-clears from the true sensor origin.
//
// Drop-in replacement for the Python ground_segmentation_node: same output
// topic /scan/obstacles, so nav2_params needs no change.
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include "ugv_obstacle/obstacle_core.hpp"

using ugv_obstacle::ObstacleConfig;
using ugv_obstacle::ObstacleCore;

class ObstacleNode : public rclcpp::Node {
 public:
  ObstacleNode() : rclcpp::Node("ugv_obstacle_node") {
    ObstacleConfig cfg;
    cfg.dem_res = declare_parameter<double>("dem_res", 1.0);
    cfg.dem_origin_x = declare_parameter<double>("dem_origin_x", -450.0);
    cfg.dem_origin_y = declare_parameter<double>("dem_origin_y", -150.0);
    cfg.tau_prior = declare_parameter<double>("tau_prior", 0.4);
    cfg.tau_local = declare_parameter<double>("tau_local", 0.4);
    cfg.self_radius = declare_parameter<double>("self_radius", 1.2);
    cfg.max_range = declare_parameter<double>("max_range", 30.0);
    cfg.min_height = declare_parameter<double>("min_height", -2.0);
    cfg.max_height = declare_parameter<double>("max_height", 4.0);
    cfg.cell_size = declare_parameter<double>("cell_size", 0.4);
    cfg.min_points_per_cell =
        declare_parameter<int>("min_points_per_cell", 2);
    cfg.low_conf_relax = declare_parameter<double>("low_conf_relax", 0.6);

    dem_path_ = declare_parameter<std::string>("dem_path", "");
    input_topic_ = declare_parameter<std::string>("input_topic", "/scan/points");
    output_topic_ =
        declare_parameter<std::string>("output_topic", "/scan/obstacles");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    default_confidence_ =
        declare_parameter<double>("default_confidence", 1.0);
    grid_size_ = declare_parameter<double>("grid_size", 40.0);
    grid_res_ = declare_parameter<double>("grid_resolution", 0.25);
    publish_grid_ = declare_parameter<bool>("publish_grid", true);

    core_ = std::make_unique<ObstacleCore>(cfg);
    if (!dem_path_.empty() &&
        core_->load_global_dem(dem_path_, cfg.dem_res, cfg.dem_origin_x,
                               cfg.dem_origin_y)) {
      RCLCPP_INFO(get_logger(), "Loaded global DEM: %s", dem_path_.c_str());
    } else {
      RCLCPP_WARN(get_logger(),
                  "No global DEM ('%s'); DEM-prior disabled, using local-jump "
                  "only.",
                  dem_path_.c_str());
    }

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ObstacleNode::cloudCb, this, std::placeholders::_1));
    conf_sub_ = create_subscription<std_msgs::msg::Float64>(
        "/trn/match_quality", 10,
        [this](const std_msgs::msg::Float64::SharedPtr m) {
          confidence_ = std::max(0.0, std::min(1.0, m->data));
        });
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);
    grid_pub_ = create_publisher<grid_map_msgs::msg::GridMap>("/obstacle/grid", 1);
    occ_pub_ =
        create_publisher<nav_msgs::msg::OccupancyGrid>("/obstacle/costmap", 1);

    RCLCPP_INFO(get_logger(), "ugv_obstacle ready: %s -> %s (+grid_map)",
                input_topic_.c_str(), output_topic_.c_str());
  }

 private:
  void cloudCb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Read sensor-frame points into Eigen.
    std::vector<Eigen::Vector3f> pts_s;
    pts_s.reserve(msg->width * msg->height);
    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iy(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iz(*msg, "z");
    for (; ix != ix.end(); ++ix, ++iy, ++iz) {
      const float x = *ix, y = *iy, z = *iz;
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
      pts_s.emplace_back(x, y, z);
    }
    if (pts_s.empty()) return;

    // map <- sensor transform.
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_->lookupTransform(map_frame_, msg->header.frame_id,
                                       tf2::TimePointZero);
    } catch (const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "No TF %s <- %s: %s", map_frame_.c_str(),
                           msg->header.frame_id.c_str(), e.what());
      return;
    }
    const Eigen::Isometry3d T = tf2::transformToEigen(tf);

    Eigen::MatrixXf pts_map(static_cast<int>(pts_s.size()), 3);
    for (int i = 0; i < static_cast<int>(pts_s.size()); ++i) {
      const Eigen::Vector3d p =
          T * pts_s[i].cast<double>();
      pts_map(i, 0) = static_cast<float>(p.x());
      pts_map(i, 1) = static_cast<float>(p.y());
      pts_map(i, 2) = static_cast<float>(p.z());
    }

    const double conf = core_->has_dem() ? confidence_ : 0.0;
    const auto mask = core_->classify(pts_map, T.translation().x(),
                                      T.translation().y(),
                                      core_->has_dem() ? conf : default_confidence_);

    // Publish surviving obstacle points in the ORIGINAL sensor frame.
    size_t n_obs = 0;
    for (auto m8 : mask) n_obs += m8;

    sensor_msgs::msg::PointCloud2 out;
    out.header = msg->header;
    sensor_msgs::PointCloud2Modifier mod(out);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(n_obs);
    sensor_msgs::PointCloud2Iterator<float> ox(out, "x");
    sensor_msgs::PointCloud2Iterator<float> oy(out, "y");
    sensor_msgs::PointCloud2Iterator<float> oz(out, "z");
    for (size_t i = 0; i < mask.size(); ++i) {
      if (!mask[i]) continue;
      *ox = pts_s[i].x();
      *oy = pts_s[i].y();
      *oz = pts_s[i].z();
      ++ox; ++oy; ++oz;
    }
    pub_->publish(out);

    if (publish_grid_) {
      buildAndPublishGrid(pts_map, T.translation().x(), T.translation().y(),
                          msg->header.stamp);
    }
  }

  // Build a robot-centric multi-layer grid_map and publish it plus an
  // OccupancyGrid derived from the obstacle layer (Nav2 static-layer input).
  void buildAndPublishGrid(const Eigen::MatrixXf& pts_map, double rx, double ry,
                           const builtin_interfaces::msg::Time& stamp) {
    grid_map::GridMap map(
        {"elevation", "elevation_prior", "obstacle", "traversability"});
    map.setFrameId(map_frame_);
    map.setGeometry(grid_map::Length(grid_size_, grid_size_), grid_res_,
                    grid_map::Position(rx, ry));

    // Prior elevation + slope-based traversability from the a-priori DEM.
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
      grid_map::Position pos;
      map.getPosition(*it, pos);
      const float h = core_->has_dem() ? core_->dem_at(pos.x(), pos.y())
                                       : std::numeric_limits<float>::quiet_NaN();
      map.at("elevation_prior", *it) = h;
    }

    // Live elevation = max LiDAR z per cell.
    for (int i = 0; i < pts_map.rows(); ++i) {
      grid_map::Position p(pts_map(i, 0), pts_map(i, 1));
      grid_map::Index idx;
      if (!map.getIndex(p, idx)) continue;
      float& e = map.at("elevation", idx);
      const float z = pts_map(i, 2);
      if (std::isnan(e) || z > e) e = z;
    }

    // Obstacle layer = live elevation minus prior above threshold.
    const double conf = core_->has_dem() ? confidence_ : 0.0;
    const double tau = core_->config().tau_prior +
                       (1.0 - conf) * core_->config().low_conf_relax;
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
      const float e = map.at("elevation", *it);
      const float prior = map.at("elevation_prior", *it);
      float occ = 0.0f;
      if (!std::isnan(e) && !std::isnan(prior) && (e - prior) > tau) {
        occ = 100.0f;
      }
      map.at("obstacle", *it) = occ;

      // Traversability from prior slope (finite diff), 0 = free .. 100 = lethal.
      map.at("traversability", *it) = 0.0f;
    }

    // Publish grid_map (RViz) and OccupancyGrid (Nav2 static layer, Option A).
    auto grid_msg = grid_map::GridMapRosConverter::toMessage(map);
    grid_msg->header.stamp = stamp;
    grid_pub_->publish(*grid_msg);

    nav_msgs::msg::OccupancyGrid occ;
    grid_map::GridMapRosConverter::toOccupancyGrid(map, "obstacle", 0.0, 100.0,
                                                   occ);
    occ.header.stamp = stamp;
    occ_pub_->publish(occ);
  }

  std::unique_ptr<ObstacleCore> core_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr conf_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr grid_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_pub_;

  std::string dem_path_, input_topic_, output_topic_, map_frame_;
  double confidence_ = 1.0;
  double default_confidence_ = 1.0;
  double grid_size_ = 40.0;
  double grid_res_ = 0.25;
  bool publish_grid_ = true;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleNode>());
  rclcpp::shutdown();
  return 0;
}
