#include <rrtc_motion_planner/rrtc_planner.hpp>
#include <rrtc_motion_planner/rrtc_utils.hpp>
#include <rrtc_motion_planner/rrtc_parameters.hpp>

#include <moveit/robot_state/conversions.hpp>
#include <moveit/utils/logger.hpp>

#include <rrtconnect/rrt_connect.h>

#include <chrono>

namespace rrtc
{
namespace
{
rclcpp::Logger getLogger()
{
  return moveit::getLogger("moveit.planners.rrtc.planner");
}
}  // namespace

void RRTConnectPlanner::solve(const planning_scene::PlanningSceneConstPtr& planning_scene,
                              const planning_interface::MotionPlanRequest& req, 
                              const RRTConnectParameters& params,
                              planning_interface::MotionPlanDetailedResponse& res) const
{
  auto start_time = std::chrono::system_clock::now();
  res.planner_id = std::string("rrtc");

  if (!planning_scene)
  {
    RCLCPP_ERROR(getLogger(), "No planning scene initialized.");
    res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
    return;
  }

  // get the specified start state
  moveit::core::RobotState start_state = planning_scene->getCurrentState();
  moveit::core::robotStateMsgToRobotState(planning_scene->getTransforms(), req.start_state, start_state);

  if (!start_state.satisfiesBounds())
  {
    RCLCPP_ERROR(getLogger(), "Start state violates joint limits");
    res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_ROBOT_STATE;
    return;
  }

  const moveit::core::JointModelGroup* model_group =
      planning_scene->getRobotModel()->getJointModelGroup(req.group_name);
  if (!model_group)
  {
    RCLCPP_ERROR(getLogger(), "Invalid joint model group: %s", req.group_name.c_str());
    res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
    return;
  }
  rrtconnect::ConfigWaypoint start_config;
  robotStateToWaypoint(start_state, model_group, start_config);

  if (req.goal_constraints.size() != 1)
  {
    RCLCPP_ERROR(getLogger(), "Expecting exactly one goal constraint, got: %zd", req.goal_constraints.size());
    res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
    return;
  }

  if (req.goal_constraints[0].joint_constraints.empty() || !req.goal_constraints[0].position_constraints.empty() ||
      !req.goal_constraints[0].orientation_constraints.empty())
  {
    RCLCPP_ERROR(getLogger(), "Only joint-space goals are supported");
    res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
    return;
  }

  moveit::core::RobotState goal_state(start_state);
  for (const moveit_msgs::msg::JointConstraint& joint_constraint : req.goal_constraints[0].joint_constraints)
    goal_state.setVariablePosition(joint_constraint.joint_name, joint_constraint.position);

  if (!goal_state.satisfiesBounds())
  {
    RCLCPP_ERROR(getLogger(), "Goal state violates joint limits");
    res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_ROBOT_STATE;
    return;
  }

  rrtconnect::ConfigWaypoint goal_config;
  robotStateToWaypoint(goal_state, model_group, goal_config);

  // fix the goal to move the shortest angular distance for wrap-around joints:
  for (size_t i = 0; i < model_group->getActiveJointModels().size(); ++i)
  {
    const moveit::core::JointModel* jm = model_group->getActiveJointModels()[i];
    const moveit::core::RevoluteJointModel* revolute_joint = dynamic_cast<const moveit::core::RevoluteJointModel*>(jm);

    if (revolute_joint != nullptr && revolute_joint->isContinuous())
    {
      double start_angle = start_config(i);
      double end_angle = goal_config(i);
      goal_config(i) = start_angle + shortestAngularDistance(start_angle, end_angle);
    }
  }

  rrtconnect::ConfigWaypoint lower_limits, upper_limits;
  getJointLimits(model_group, lower_limits, upper_limits);

  auto validate_state = [&](const absl::optional<rrtconnect::ConfigWaypoint>& q_origin, 
                            const rrtconnect::ConfigWaypoint& q_dest) -> bool {
    // Caller should validate q_dest, and optionally validate that q_origin
    // (if present) has a valid transition to q_dest.
    static moveit::core::RobotState validation_state(planning_scene->getCurrentState());
    static moveit::core::RobotState origin_robot_state(planning_scene->getCurrentState());
    
    auto ajm = model_group->getActiveJointModels();
    setJointPositions(q_dest, ajm, validation_state);
    validation_state.update();

    if (!validation_state.satisfiesBounds(model_group))
    {
      return false;
    }

    if (planning_scene->isStateColliding(validation_state, req.group_name))
    {
      return false;
    }
    
    // If q_origin is present, validate the transition from q_origin to q_dest
    if (q_origin.has_value())
    {
      setJointPositions(q_origin.value(), ajm, origin_robot_state);
      origin_robot_state.update();
      robot_trajectory::RobotTrajectory segment_trajectory(planning_scene->getRobotModel(), req.group_name);
      segment_trajectory.addSuffixWayPoint(origin_robot_state, 0.0); // dt can be 0.0 for path checking
      segment_trajectory.addSuffixWayPoint(validation_state, 0.0);   // dt can be 0.0 for path checking
      
      // Check for collision and joint limit violations on the path segment
      if (!planning_scene->isPathValid(segment_trajectory, req.group_name))
      {
        RCLCPP_DEBUG(getLogger(), "State validation: Path segment from q_origin to q_dest is invalid (collision or bounds).");
        return false;
      }
    }
    return true;
  };

  RCLCPP_INFO(getLogger(), "Calling rrtconnect::RRTConnectPlanner::Plan");

  rrtconnect::PlanResult plan_result = rrtconnect::RRTConnectPlanner::Plan(
      start_config, goal_config, 
      lower_limits, upper_limits, 
      validate_state,
      params.max_iterations_,                               // iters
      params.planning_time_limit_,                          // timeout_s
      params.configuration_space_step_,                     // configuration_space_step
      params.iters_between_goal_checks_,                    // iters_between_goal_checks
      nullptr,                                              // start_tree
      nullptr                                               // goal_tree
  );

  auto total_duration = std::chrono::duration<double>(std::chrono::system_clock::now() - start_time).count();
  res.processing_time.resize(1);
  res.processing_time[0] = total_duration;

  if (!plan_result.path.empty())
  {
    RCLCPP_INFO(getLogger(), "RRT-Connect found a solution with %zu points in %f seconds. (%d iterations)",
                plan_result.path.size(), total_duration, plan_result.planning_iterations);

    auto result_traj =
        std::make_shared<robot_trajectory::RobotTrajectory>(planning_scene->getRobotModel(), req.group_name);

    for (const auto& waypoint_state : plan_result.path)
    {
      moveit::core::RobotStatePtr robot_state = std::make_shared<moveit::core::RobotState>(start_state);
      
      setJointPositions(waypoint_state, model_group, *robot_state);
      result_traj->addSuffixWayPoint(robot_state, 0.1);  // Using a nominal dt, can be refined
    }

    res.trajectory.resize(1);
    res.trajectory[0] = result_traj;
    res.description.resize(1);
    res.description[0] = "plan";
    res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

    // check that final state is within goal tolerances
    kinematic_constraints::JointConstraint jc(planning_scene->getRobotModel());
    const moveit::core::RobotState& last_state = result_traj->getLastWayPoint();
    for (const moveit_msgs::msg::JointConstraint& constraint : req.goal_constraints[0].joint_constraints)
    {
      if (!jc.configure(constraint) || !jc.decide(last_state).satisfied)
      {
        RCLCPP_ERROR(getLogger(), "Goal constraints are violated by RRT-Connect path: %s",
                     constraint.joint_name.c_str());
        res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::GOAL_CONSTRAINTS_VIOLATED;
        return;
      }
    }
  }
  else
  {
    RCLCPP_ERROR(getLogger(), "RRT-Connect planning failed or returned an empty path. (%d iterations)", 
                              plan_result.planning_iterations);
    res.error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
  }

  RCLCPP_DEBUG(getLogger(), "Serviced planning request in %f wall-seconds", total_duration);
}

}  // namespace rrtc
