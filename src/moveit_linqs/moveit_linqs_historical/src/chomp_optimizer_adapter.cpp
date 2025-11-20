/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Raghavender Sahdev.
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
 *   * Neither the name of Raghavender Sahdev nor the names of its
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

/* Author: Raghavender Sahdev */

#include <moveit/utils/logger.hpp>

#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_interface/planning_response_adapter.hpp>

#include <moveit/collision_distance_field/collision_detector_allocator_hybrid.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/trajectory_processing/trajectory_tools.hpp>

#include <chomp_motion_planner/chomp_parameters.hpp>
#include <chomp_motion_planner/chomp_planner.hpp>

#include <eigen3/Eigen/Core>
#include <pluginlib/class_list_macros.hpp>
#include <vector>

#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_ros_planning/default_response_adapter_parameters.hpp>


namespace chomp
{
  
class OptimizerAdapter : public planning_interface::PlanningResponseAdapter
{
public:
  OptimizerAdapter() : logger_(moveit::getLogger("moveit.planners.chomp.planner"))
  {
  }
  
  ~OptimizerAdapter() = default; // was override before but seems to give compilation errors with other library versions.

  void initialize(const rclcpp::Node::SharedPtr& node, const std::string& /* unused */) override
  {
    if (!node->get_parameter("chomp.planning_time_limit", params_.planning_time_limit_))
    {
      params_.planning_time_limit_ = 10.0;
      RCLCPP_DEBUG(logger_, "Param planning_time_limit was not set. Using default value: %f",
                   params_.planning_time_limit_);
    }
    if (!node->get_parameter("chomp.max_iterations", params_.max_iterations_))
    {
      params_.max_iterations_ = 200;
      RCLCPP_DEBUG(logger_, "Param max_iterations was not set. Using default value: %d", params_.max_iterations_);
    }
    if (!node->get_parameter("chomp.max_iterations_after_collision_free", params_.max_iterations_after_collision_free_))
    {
      params_.max_iterations_after_collision_free_ = 5;
      RCLCPP_DEBUG(logger_, "Param max_iterations_after_collision_free was not set. Using default value: %d",
                   params_.max_iterations_after_collision_free_);
    }
    if (!node->get_parameter("chomp.smoothness_cost_weight", params_.smoothness_cost_weight_))
    {
      params_.smoothness_cost_weight_ = 0.1;
      RCLCPP_DEBUG(logger_, "Param smoothness_cost_weight was not set. Using default value: %f",
                   params_.smoothness_cost_weight_);
    }
    if (!node->get_parameter("chomp.obstacle_cost_weight", params_.obstacle_cost_weight_))
    {
      params_.obstacle_cost_weight_ = 1.0;
      RCLCPP_DEBUG(logger_, "Param obstacle_cost_weight was not set. Using default value: %f",
                   params_.obstacle_cost_weight_);
    }
    if (!node->get_parameter("chomp.learning_rate", params_.learning_rate_))
    {
      params_.learning_rate_ = 0.01;
      RCLCPP_DEBUG(logger_, "Param learning_rate was not set. Using default value: %f", params_.learning_rate_);
    }
    if (!node->get_parameter("chomp.smoothness_cost_velocity", params_.smoothness_cost_velocity_))
    {
      params_.smoothness_cost_velocity_ = 0.0;
      RCLCPP_DEBUG(logger_, "Param smoothness_cost_velocity was not set. Using default value: %f",
                   params_.smoothness_cost_velocity_);
    }
    if (!node->get_parameter("chomp.smoothness_cost_acceleration", params_.smoothness_cost_acceleration_))
    {
      params_.smoothness_cost_acceleration_ = 1.0;
      RCLCPP_DEBUG(logger_, "Param smoothness_cost_acceleration was not set. Using default value: %f",
                   params_.smoothness_cost_acceleration_);
    }
    if (!node->get_parameter("chomp.smoothness_cost_jerk", params_.smoothness_cost_jerk_))
    {
      params_.smoothness_cost_jerk_ = 0.0;
      RCLCPP_DEBUG(logger_, "Param smoothness_cost_jerk_ was not set. Using default value: %f",
                   params_.smoothness_cost_jerk_);
    }
    if (!node->get_parameter("chomp.ridge_factor", params_.ridge_factor_))
    {
      params_.ridge_factor_ = 0.0;
      RCLCPP_DEBUG(logger_, "Param ridge_factor_ was not set. Using default value: %f", params_.ridge_factor_);
    }
    if (!node->get_parameter("chomp.use_pseudo_inverse", params_.use_pseudo_inverse_))
    {
      params_.use_pseudo_inverse_ = 0.0;
      RCLCPP_DEBUG(logger_, "Param use_pseudo_inverse_ was not set. Using default value: %d",
                   params_.use_pseudo_inverse_);
    }
    if (!node->get_parameter("chomp.pseudo_inverse_ridge_factor", params_.pseudo_inverse_ridge_factor_))
    {
      params_.pseudo_inverse_ridge_factor_ = 1e-4;
      RCLCPP_DEBUG(logger_, "Param pseudo_inverse_ridge_factor was not set. Using default value: %f",
                   params_.pseudo_inverse_ridge_factor_);
    }
    if (!node->get_parameter("chomp.joint_update_limit", params_.joint_update_limit_))
    {
      params_.joint_update_limit_ = 0.1;
      RCLCPP_DEBUG(logger_, "Param joint_update_limit was not set. Using default value: %f", params_.joint_update_limit_);
    }
    if (!node->get_parameter("chomp.collision_clearance", params_.min_clearance_))
    {
      params_.min_clearance_ = 0.2;
      RCLCPP_DEBUG(logger_, "Param collision_clearance was not set. Using default value: %f", params_.min_clearance_);
    }
    if (!node->get_parameter("chomp.collision_threshold", params_.collision_threshold_))
    {
      params_.collision_threshold_ = 0.07;
      RCLCPP_DEBUG(logger_, "Param collision_threshold_ was not set. Using default value: %f",
                   params_.collision_threshold_);
    }
    if (!node->get_parameter("chomp.use_stochastic_descent", params_.use_stochastic_descent_))
    {
      params_.use_stochastic_descent_ = true;
      RCLCPP_DEBUG(logger_, "Param use_stochastic_descent was not set. Using default value: %d",
                   params_.use_stochastic_descent_);
    }
    params_.trajectory_initialization_method_ = "quintic-spline";
    std::string method;
    if (node->get_parameter("chomp.trajectory_initialization_method", method) &&
        !params_.setTrajectoryInitializationMethod(method))
    {
      RCLCPP_ERROR(logger_,
                   "Attempted to set trajectory_initialization_method to invalid value '%s'. Using default "
                   "'%s' instead.",
                   method.c_str(), params_.trajectory_initialization_method_.c_str());
    }
  }

  [[nodiscard]] std::string getDescription() const override
  {
    return std::string("CHOMP Optimizer");;
  }

  void adapt(const planning_scene::PlanningSceneConstPtr& ps,
             const planning_interface::MotionPlanRequest& req,
             planning_interface::MotionPlanResponse& res) const override
  {
    RCLCPP_DEBUG(logger_, "Running '%s': adapt", getDescription().c_str());

    if (!res.trajectory || res.trajectory->empty())
    {
      RCLCPP_WARN(logger_, "Input trajectory is empty or null. Cannot optimize.");
      // If there's no trajectory to optimize, return the current error code or indicate failure.
      if (res.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
      {
         // If previous planner said success but gave no plan, it's an issue.
        res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
      }
      return;
    }

    // create a hybrid collision detector to set the collision checker as hybrid
    collision_detection::CollisionDetectorAllocatorPtr hybrid_cd(
        collision_detection::CollisionDetectorAllocatorHybrid::create());

    // create a writable planning scene
    planning_scene::PlanningScenePtr planning_scene = ps->diff();
    RCLCPP_DEBUG(logger_, "Configuring Planning Scene for CHOMP ...");
    planning_scene->allocateCollisionDetector(hybrid_cd);

    chomp::ChompPlanner chomp_planner;
    planning_interface::MotionPlanDetailedResponse res_detailed;
    res_detailed.trajectory.push_back(res.trajectory);

    chomp_planner.solve(planning_scene, req, params_, res_detailed);

    res.error_code = res_detailed.error_code;
    if (res.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
    {
      if (!res_detailed.trajectory.empty() && res_detailed.trajectory[0])
      {
        res.trajectory = res_detailed.trajectory[0];
      }
      else
      {
        RCLCPP_ERROR(logger_, "CHOMP optimization succeeded but produced no trajectory.");
        res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
      }
      if (!res_detailed.processing_time.empty())
      {
        // Add CHOMP's processing time to the total planning time
        res.planning_time += res_detailed.processing_time[0];
      }
    }
  }

private:
  chomp::ChompParameters params_;
  rclcpp::Logger logger_;
};
}  // namespace chomp

PLUGINLIB_EXPORT_CLASS(chomp::OptimizerAdapter, planning_interface::PlanningResponseAdapter)
