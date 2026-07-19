cmake_minimum_required(VERSION 3.8)
project(ugv_obstacle)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_costmap_2d REQUIRED)

# ROS-free core.
add_library(obstacle_core src/obstacle_core.cpp)
target_include_directories(obstacle_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(obstacle_core Eigen3::Eigen)
# grid_map_core injects a project-wide Eigen plugin define; every Eigen-
# compiling target needs its include path (grid_map_core is ROS-free).
ament_target_dependencies(obstacle_core grid_map_core)

# Node.
add_executable(obstacle_node src/obstacle_node.cpp)
target_include_directories(obstacle_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(obstacle_node obstacle_core Eigen3::Eigen)
ament_target_dependencies(obstacle_node
  rclcpp sensor_msgs std_msgs geometry_msgs nav_msgs tf2 tf2_ros tf2_eigen
  grid_map_core grid_map_ros grid_map_msgs grid_map_costmap_2d)

install(TARGETS obstacle_node
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/
  DESTINATION include)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_obstacle_core test/test_obstacle_core.cpp)
  target_link_libraries(test_obstacle_core obstacle_core Eigen3::Eigen)
  ament_target_dependencies(test_obstacle_core grid_map_core)
  target_include_directories(test_obstacle_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

ament_package()
