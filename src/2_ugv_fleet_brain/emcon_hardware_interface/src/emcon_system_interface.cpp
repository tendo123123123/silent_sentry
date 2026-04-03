// Copyright 2026 Silent Sentry Project
// SPDX-License-Identifier: Apache-2.0
//
// EMCON — ros2_control SystemInterface implementation.
// Uses gz-transport to communicate joint commands/states directly with
// the Gazebo physics engine, bypassing ROS 2 DDS entirely.

#include "emcon_hardware_interface/emcon_system_interface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"

namespace emcon_hardware_interface
{

//  on_init — Parse joint configuration from URDF HardwareInfo
hardware_interface::CallbackReturn EmconSystemInterface::on_init(
  const hardware_interface::HardwareInfo & hardware_info)
{
  // Call the parent on_init (stores hardware_info into info_)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  if (hardware_interface::SystemInterface::on_init(hardware_info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }
#pragma GCC diagnostic pop

  // Read custom hardware parameters from URDF <ros2_control><hardware><param>
  auto it_bot = info_.hardware_parameters.find("bot_name");
  bot_name_ = (it_bot != info_.hardware_parameters.end()) ? it_bot->second : "autobot";

  auto it_world = info_.hardware_parameters.find("world_name");
  world_name_ = (it_world != info_.hardware_parameters.end()) ? it_world->second : "thar_desert";

  RCLCPP_INFO(
    get_logger(),
    "[EMCON] Initializing Data Diode for model='%s' in world='%s'",
    bot_name_.c_str(), world_name_.c_str());

  // Parse joints from the URDF <ros2_control> block
  joints_.resize(info_.joints.size());
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    joints_[i].name = info_.joints[i].name;
    joints_[i].position = 0.0;
    joints_[i].velocity = 0.0;
    joints_[i].command  = 0.0;

    // Determine command type from the URDF command interfaces
    if (!info_.joints[i].command_interfaces.empty()) {
      joints_[i].command_interface_type = info_.joints[i].command_interfaces[0].name;
    } else {
      // Read-only joint (e.g. front wheel encoders)
      joints_[i].command_interface_type = "";
    }

    RCLCPP_INFO(
      get_logger(),
      "[EMCON]   Joint[%zu]: '%s'  cmd_type='%s'",
      i, joints_[i].name.c_str(), joints_[i].command_interface_type.c_str());
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

//  on_configure — Stand up the gz-transport node and subscribe to joint states
hardware_interface::CallbackReturn EmconSystemInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "[EMCON] Creating gz-transport node...");
  gz_node_ = std::make_unique<gz::transport::Node>();

  // Subscribe to the Gazebo joint state topic for this model
  // Gazebo publishes Model messages at: /world/<world>/model/<model>/joint_state
  const std::string joint_state_topic =
    "/world/" + world_name_ + "/model/" + bot_name_ + "/joint_state";

  if (!gz_node_->Subscribe(joint_state_topic, &EmconSystemInterface::on_gz_joint_state, this)) {
    RCLCPP_ERROR(
      get_logger(),
      "[EMCON] Failed to subscribe to gz-transport topic: %s",
      joint_state_topic.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }

  RCLCPP_INFO(
    get_logger(),
    "[EMCON] Subscribed to gz-transport: %s", joint_state_topic.c_str());

  // Advertise joint command topics
  for (auto & joint : joints_) {
    if (joint.command_interface_type.empty()) {
      continue;
    }

    std::string cmd_suffix;
    if (joint.command_interface_type == hardware_interface::HW_IF_POSITION) {
      cmd_suffix = "cmd_pos";
    } else if (joint.command_interface_type == hardware_interface::HW_IF_VELOCITY) {
      cmd_suffix = "cmd_vel";
    } else {
      continue;
    }

    const std::string topic =
      "/model/" + bot_name_ + "/joint/" + joint.name + "/" + cmd_suffix;

    joint.pub = gz_node_->Advertise<gz::msgs::Double>(topic);
    RCLCPP_INFO(get_logger(), "[EMCON] Advertised command topic: %s", topic.c_str());
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

//  on_activate — Ready for real-time loop
hardware_interface::CallbackReturn EmconSystemInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "[EMCON] Activating — Data Diode is live.");

  // Initialise commands to current state so there's no jerk on activation
  std::lock_guard<std::mutex> lock(state_mutex_);
  for (auto & joint : joints_) {
    if (joint.command_interface_type == hardware_interface::HW_IF_POSITION) {
      joint.command = joint.position;
    } else if (joint.command_interface_type == hardware_interface::HW_IF_VELOCITY) {
      joint.command = 0.0;
    }
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

//  on_deactivate
hardware_interface::CallbackReturn EmconSystemInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "[EMCON] Deactivating — stopping commands.");
  return hardware_interface::CallbackReturn::SUCCESS;
}

//  on_cleanup — Tear down gz-transport
hardware_interface::CallbackReturn EmconSystemInterface::on_cleanup(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "[EMCON] Cleaning up gz-transport node.");
  gz_node_.reset();
  return hardware_interface::CallbackReturn::SUCCESS;
}

//  read — Copy latest gz-transport state into ros2_control state interfaces
hardware_interface::return_type EmconSystemInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  std::lock_guard<std::mutex> lock(state_mutex_);

  for (size_t i = 0; i < joints_.size(); ++i) {
    const auto & joint = joints_[i];
    // Write into the framework's state interfaces via the base class helpers.
    // The key format is "<joint_name>/<interface_name>".
    const std::string pos_key = joint.name + "/" + hardware_interface::HW_IF_POSITION;
    const std::string vel_key = joint.name + "/" + hardware_interface::HW_IF_VELOCITY;

    if (has_state(pos_key)) {
      set_state(pos_key, joint.position);
    }
    if (has_state(vel_key)) {
      set_state(vel_key, joint.velocity);
    }
  }

  return hardware_interface::return_type::OK;
}

//  write — Beam commands to Gazebo via gz-transport
hardware_interface::return_type EmconSystemInterface::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  for (size_t i = 0; i < joints_.size(); ++i) {
    auto & joint = joints_[i];

    // Skip read-only joints (no command interface)
    if (joint.command_interface_type.empty()) {
      continue;
    }

    // Read the latest command from the framework
    const std::string cmd_key = joint.name + "/" + joint.command_interface_type;
    if (has_command(cmd_key)) {
      joint.command = get_command<double>(cmd_key);
    }

    if (!joint.pub.Valid()) {
      continue;
    }

    if (!std::isnan(joint.command)) {
      gz::msgs::Double msg;
      msg.set_data(joint.command);

      // Non-blocking publish
      joint.pub.Publish(msg);
    }
  }

  return hardware_interface::return_type::OK;
}

//  gz-transport callback — Runs on gz-transport's I/O thread
void EmconSystemInterface::on_gz_joint_state(const gz::msgs::Model & msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);

  for (int j = 0; j < msg.joint_size(); ++j) {
    const auto & gz_joint = msg.joint(j);
    const std::string & gz_name = gz_joint.name();

    // Find the matching joint in our list
    for (auto & joint : joints_) {
      if (joint.name == gz_name) {
        // Gazebo Model message: axis1 contains the joint state
        if (gz_joint.has_axis1()) {
          joint.position = gz_joint.axis1().position();
          joint.velocity = gz_joint.axis1().velocity();
        }
        break;
      }
    }
  }
}

}  // namespace emcon_hardware_interface

// pluginlib registration 
PLUGINLIB_EXPORT_CLASS(
  emcon_hardware_interface::EmconSystemInterface,
  hardware_interface::SystemInterface)
