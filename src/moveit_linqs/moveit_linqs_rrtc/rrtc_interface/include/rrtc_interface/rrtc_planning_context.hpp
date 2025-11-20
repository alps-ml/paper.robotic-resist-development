#pragma once

#include <rrtc_interface/rrtc_interface.hpp>
#include <moveit/planning_interface/planning_interface.hpp>

#include <rclcpp/rclcpp.hpp>

namespace rrtc_interface
{
MOVEIT_CLASS_FORWARD(RRTCPlanningContext);  // Defines RRTCPlanningContextPtr, ConstPtr, WeakPtr... etc

class RRTCPlanningContext : public planning_interface::PlanningContext
{
public:
  void solve(planning_interface::MotionPlanResponse& res) override;
  void solve(planning_interface::MotionPlanDetailedResponse& res) override;

  void clear() override;
  bool terminate() override;

  RRTCPlanningContext(const std::string& name, const std::string& group, const moveit::core::RobotModelConstPtr& model,
                       const rclcpp::Node::SharedPtr& node);

  ~RRTCPlanningContext() override = default;

  void initialize();

private:
  RRTCInterfacePtr rrtc_interface_;
  moveit::core::RobotModelConstPtr robot_model_;
};

}  // namespace rrtc_interface
