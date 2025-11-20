/* Author: S. Gyger */

#pragma once

#include <moveit/planning_interface/planning_request.hpp>
#include <moveit/planning_interface/planning_response.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_state/robot_state.hpp>

#include <rrtc_motion_planner/rrtc_parameters.hpp>
#include <rrtc_motion_planner/rrtc_utils.hpp>

namespace rrtc
{
class RRTConnectPlanner
{
public:
  RRTConnectPlanner() = default;
  virtual ~RRTConnectPlanner() = default;

  void solve(const planning_scene::PlanningSceneConstPtr& planning_scene,
             const planning_interface::MotionPlanRequest& req, 
             const RRTConnectParameters& params,
             planning_interface::MotionPlanDetailedResponse& res) const;
};
}  // namespace rrtc
