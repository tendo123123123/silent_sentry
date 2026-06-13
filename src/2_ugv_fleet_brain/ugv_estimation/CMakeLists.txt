cmake_minimum_required(VERSION 3.8)
project(ugv_estimation)

# Enable C++17 or C++20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -O2 is the standard safe optimization level for Eigen/GTSAM code.
add_compile_options(-O2 -Wall -Wextra -Wpedantic)

# Find Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)

# Include Directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
)

# 1. Core Mathematical and Algorithmic Library (Zero ROS Headers)
add_library(se3_fuser_core
  src/se3_fuser_core.cpp
)
target_include_directories(se3_fuser_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(se3_fuser_core
  gtsam
  Eigen3::Eigen
)

# 2. ROS 2 Lifecycle Node Wrapper Executable
add_executable(fuser_node
  src/fuser_node.cpp
)
target_link_libraries(fuser_node
  se3_fuser_core
)
ament_target_dependencies(fuser_node
  rclcpp
  rclcpp_lifecycle
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
)
# Export ALL symbols for backtrace_symbols_fd crash diagnostics
target_link_options(fuser_node PRIVATE -rdynamic)

# Install Rules
install(TARGETS se3_fuser_core
  EXPORT export_ugv_estimation
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(TARGETS fuser_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_ugv_estimation HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  rclcpp_lifecycle
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  Eigen3
  GTSAM
)

ament_package()
