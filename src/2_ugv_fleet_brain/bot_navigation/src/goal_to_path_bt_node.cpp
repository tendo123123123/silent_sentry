#include <string>
#include "behaviortree_cpp/action_node.h"
#include "behaviortree_cpp/bt_factory.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav2_behavior_tree/bt_utils.hpp"

namespace nav2_behavior_tree
{

class GoalToPath : public BT::SyncActionNode
{
public:
  GoalToPath(
    const std::string & action_name,
    const BT::NodeConfig & conf)
  : BT::SyncActionNode(action_name, conf)
  {
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<geometry_msgs::msg::PoseStamped>("input_goal", "Target goal pose"),
      BT::OutputPort<nav_msgs::msg::Path>("output_path", "Path message containing target goal pose")
    };
  }

  BT::NodeStatus tick() override
  {
    geometry_msgs::msg::PoseStamped goal;
    if (!getInput("input_goal", goal)) {
      return BT::NodeStatus::FAILURE;
    }

    nav_msgs::msg::Path path;
    path.header = goal.header;
    path.poses.push_back(goal);

    setOutput("output_path", path);
    return BT::NodeStatus::SUCCESS;
  }
};

}  // namespace nav2_behavior_tree

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<nav2_behavior_tree::GoalToPath>("GoalToPath");
}
