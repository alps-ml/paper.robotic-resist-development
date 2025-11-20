#pragma once

#include <rrtc_motion_planner/rrtc_parameters.hpp>
#include <rrtc_motion_planner/rrtc_planner.hpp>

#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/utils/moveit_error_code.hpp>

namespace rrtc_interface
{
MOVEIT_CLASS_FORWARD(RRTCInterface);  // Defines RRTCInterfacePtr, ConstPtr, WeakPtr... etc

class RRTCInterface : public rrtc::RRTConnectPlanner
{
public:
  RRTCInterface(const rclcpp::Node::SharedPtr& node);

  const rrtc::RRTConnectParameters& getParams() const
  {
    return params_;
  }

protected:
  /** @brief Configure everything using the param server */
  void loadParams();

  std::shared_ptr<rclcpp::Node> node_;  /// The ROS node

  rrtc::RRTConnectParameters params_;
};
}  // namespace rrtc_interface
