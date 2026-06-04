cmake_minimum_required(VERSION 3.8)
project(ugv_local_dem)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization Flags for High-Performance Defense-Grade Code
add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic)

# Find Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(silent_sentry_interfaces REQUIRED)

# Include Directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

# 1. Core DEM Pipeline Core Mathematical Library (Zero ROS Headers)
add_library(dem_builder_core
  src/dem_builder_core.cpp
)
target_include_directories(dem_builder_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(dem_builder_core
  Eigen3::Eigen
)

# 2. ROS 2 Lifecycle Node Wrapper Executable
add_executable(dem_builder_node
  src/dem_builder_node.cpp
)
target_link_libraries(dem_builder_node
  dem_builder_core
)
ament_target_dependencies(dem_builder_node
  rclcpp
  rclcpp_lifecycle
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  silent_sentry_interfaces
)

# Install Rules
install(TARGETS dem_builder_core
  EXPORT export_ugv_local_dem
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(TARGETS dem_builder_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_ugv_local_dem HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  rclcpp_lifecycle
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  Eigen3
  silent_sentry_interfaces
)

ament_package()
