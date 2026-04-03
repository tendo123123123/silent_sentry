// Copyright 2026 Silent Sentry Project
// SPDX-License-Identifier: Apache-2.0
//
// EMCON — ros2_control SystemInterface
// Bridges joint commands across ROS_DOMAIN_IDs via gz-transport.

#ifndef EMCON_HARDWARE_INTERFACE__EMCON_SYSTEM_INTERFACE_HPP_
#define EMCON_HARDWARE_INTERFACE__EMCON_SYSTEM_INTERFACE_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"

// gz-transport — must be fully included (no forward decl due to inline templates)
#include <gz/transport/Node.hh>
#include <gz/msgs/model.pb.h>
#include <gz/msgs/double.pb.h>

namespace emcon_hardware_interface
{

/// Joint data holder for internal bookkeeping
struct JointData
{
  std::string name;
  double position = 0.0;
  double velocity = 0.0;
  double command  = 0.0;
  /// "position" or "velocity"
  std::string command_interface_type;

  /// Publisher for this joint's commands
  gz::transport::Node::Publisher pub;
};

class EmconSystemInterface : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(EmconSystemInterface)

  // lifecycle callbacks
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & hardware_info) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;

  // realtime loop
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // gz-transport callback
  void on_gz_joint_state(const gz::msgs::Model & msg);

  // data
  std::vector<JointData> joints_;

  /// Gazebo model name (set from URDF hardware param "bot_name")
  std::string bot_name_;

  /// Gazebo world name (set from URDF hardware param "world_name")
  std::string world_name_;

  /// gz-transport node (heap-allocated to defer construction)
  std::unique_ptr<gz::transport::Node> gz_node_;

  /// Protects joints_ state data written by the gz-transport callback thread
  std::mutex state_mutex_;
};

}  // namespace emcon_hardware_interface

#endif  // EMCON_HARDWARE_INTERFACE__EMCON_SYSTEM_INTERFACE_HPP_
