#include <rrtc_interface/rrtc_interface.hpp>
#include <moveit/utils/logger.hpp>

namespace rrtc_interface
{

RRTCInterface::RRTCInterface(const rclcpp::Node::SharedPtr& node)
  : rrtc::RRTConnectPlanner(), node_(node)
{
  loadParams();
}

void RRTCInterface::loadParams()
{
  node_->get_parameter_or("rrtc.planning_time_limit", params_.planning_time_limit_, 0.0);
  node_->get_parameter_or("rrtc.max_iterations", params_.max_iterations_, 10000);
  node_->get_parameter_or("rrtc.configuration_space_step", params_.configuration_space_step_, 0.03);
  node_->get_parameter_or("rrtc.iters_between_goal_checks", params_.iters_between_goal_checks_, 3);
}
}  // namespace rrtc_interface
