#include <rrtc_interface/rrtc_planning_context.hpp>
#include <moveit/robot_state/conversions.hpp>

namespace rrtc_interface
{
RRTCPlanningContext::RRTCPlanningContext(const std::string& name, const std::string& group,
                                           const moveit::core::RobotModelConstPtr& model,
                                           const rclcpp::Node::SharedPtr& node)
  : planning_interface::PlanningContext(name, group), robot_model_(model)
{
  rrtc_interface_ = std::make_shared<RRTCInterface>(node);
}

void RRTCPlanningContext::solve(planning_interface::MotionPlanDetailedResponse& res)
{
  rrtc_interface_->solve(planning_scene_, request_, rrtc_interface_->getParams(), res);
}

void RRTCPlanningContext::solve(planning_interface::MotionPlanResponse& res)
{
  planning_interface::MotionPlanDetailedResponse res_detailed;
  solve(res_detailed);
  if (res_detailed.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
  {
    res.trajectory = res_detailed.trajectory[0];
    res.planning_time = res_detailed.processing_time[0];
  }
  res.error_code = res_detailed.error_code;
}

bool RRTCPlanningContext::terminate()
{
  // TODO - make interruptible
  return true;
}

void RRTCPlanningContext::clear()
{
}

}  // namespace rrtc_interface
