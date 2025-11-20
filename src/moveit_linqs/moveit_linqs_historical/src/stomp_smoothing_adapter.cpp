/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2023, PickNik Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PickNik Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/** @file
 * @author Henning Kayser
 * @brief: A PlanningResponseAdapter plugin for optimizing already solved trajectories with STOMP.
 */

// ROS
#include <rclcpp/rclcpp.hpp>
#include <class_loader/class_loader.hpp>

// MoveIt
#include <moveit/utils/logger.hpp>
#include <moveit/planning_interface/planning_response_adapter.hpp>
#include <moveit/planning_interface/planning_interface.hpp>

// STOMP
#include <linqs_moveit_historical/stomp_moveit_planning_context.hpp>
#include <moveit_planners_stomp/stomp_moveit_parameters.hpp>

using namespace planning_interface;

namespace stomp_moveit
{
/** @brief This adapter uses STOMP for optimizing pre-solved trajectories */
class SmoothingAdapter : public planning_interface::PlanningResponseAdapter
{
public:
  SmoothingAdapter()
    : planning_interface::PlanningResponseAdapter()
    , logger_(moveit::getLogger("moveit.planners.stomp.smoothing_adapter"))
  {
  }

  void initialize(const rclcpp::Node::SharedPtr& node, const std::string& parameter_namespace) override
  {
    param_listener_ = std::make_shared<stomp_moveit::ParamListener>(node, parameter_namespace);
  }

  std::string getDescription() const override
  {
    return "Stomp Smoothing Adapter";
  }

  void adapt(const planning_scene::PlanningSceneConstPtr& ps,
             [[maybe_unused]] const planning_interface::MotionPlanRequest& req,
             planning_interface::MotionPlanResponse& res) const override
  {
    RCLCPP_DEBUG(logger_, "Running '%s': adapt", getDescription().c_str());

    if (!res.trajectory || res.trajectory->empty())
    {
      RCLCPP_WARN(logger_, "Input trajectory is empty or null. Cannot optimize with STOMP.");
      if (res.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
      {
        res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
      }
      return;
    }

    const size_t seed_waypoint_count = res.trajectory->getWayPointCount();
    if (seed_waypoint_count == 0)
    {
      RCLCPP_WARN(logger_, "Input trajectory has 0 waypoints. Cannot optimize with STOMP.");
      if (res.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
      {
        res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
      }
      return;
    }

    const std::vector<std::string> joint_names = res.trajectory->getFirstWayPoint()
                                                     .getJointModelGroup(req.group_name)
                                                     ->getActiveJointModelNames();
    const size_t joint_count = joint_names.size();

    planning_interface::MotionPlanRequest seed_req = req;
    seed_req.trajectory_constraints.constraints.clear();
    seed_req.trajectory_constraints.constraints.resize(seed_waypoint_count);
    for (size_t i = 0; i < seed_waypoint_count; ++i)
    {
      seed_req.trajectory_constraints.constraints[i].joint_constraints.resize(joint_count);
      for (size_t j = 0; j < joint_count; ++j)
      {
        seed_req.trajectory_constraints.constraints[i].joint_constraints[j].joint_name = joint_names[j];
        seed_req.trajectory_constraints.constraints[i].joint_constraints[j].position =
            res.trajectory->getWayPoint(i).getVariablePosition(joint_names[j]);
      }
    }

    stomp_moveit::Params stomp_params = param_listener_->get_params();

    PlanningContextPtr planning_context =
        std::make_shared<stomp_moveit::StompPlanningContext>("STOMP", req.group_name, stomp_params);
    planning_context->clear();
    planning_context->setPlanningScene(ps);
    planning_context->setMotionPlanRequest(seed_req);

    RCLCPP_INFO(logger_, "Smoothing trajectory with STOMP (input waypoints: %zu)", seed_waypoint_count);
    planning_interface::MotionPlanResponse stomp_res;
    planning_context->solve(stomp_res); // Assume solve now returns void

    bool success = (stomp_res.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
    if (success && stomp_res.trajectory && !stomp_res.trajectory->empty())
    {
      RCLCPP_DEBUG(logger_, "STOMP optimization successful.");
      res.trajectory = stomp_res.trajectory;
      res.planning_time += stomp_res.planning_time;
      res.error_code = stomp_res.error_code;
      if (res.error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
      {
        RCLCPP_WARN(logger_,
                    "STOMP solve reported success, but error code is not SUCCESS (%d). Using STOMP's error code.",
                    stomp_res.error_code.val);
      }
    }
    else
    {
      if (!success)
      {
        RCLCPP_WARN(logger_, "STOMP optimization failed. Error code: %d", stomp_res.error_code.val);
      }
      else
      {
        RCLCPP_ERROR(logger_, "STOMP optimization reported success but produced no trajectory.");
      }
      // If STOMP failed, retain its error code, unless it was SUCCESS (which would be contradictory)
      if (stomp_res.error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
      {
        res.error_code = stomp_res.error_code;
      }
      else // STOMP claimed success but produced no trajectory or 'success' was false due to other reasons
      {
        res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
        RCLCPP_WARN(logger_, "STOMP optimization considered failed. Setting error code to PLANNING_FAILED.");
      }
    }
  }

private:
  std::shared_ptr<stomp_moveit::ParamListener> param_listener_;
  rclcpp::Logger logger_;
};
}  // namespace stomp_moveit

CLASS_LOADER_REGISTER_CLASS(stomp_moveit::SmoothingAdapter, planning_interface::PlanningResponseAdapter);