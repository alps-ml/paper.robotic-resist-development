#pragma once

#include <iostream>

#include <moveit/planning_scene/planning_scene.hpp>
#include <rrtconnect/rrt_connect.h>

// Note: The file conversion_functions.hpp from moveit_planners/stomp could be useful if we want to refactor this functions.

namespace rrtc
{
using Joints = std::vector<const moveit::core::JointModel*>;

static inline void robotStateToWaypoint(const moveit::core::RobotState& state,
                                        const moveit::core::JointModelGroup* jmg,
                                        rrtconnect::ConfigWaypoint& joint_array)
{
  const unsigned int num_active_dof = jmg->getActiveVariableCount();
  joint_array.resize(num_active_dof);
  
  size_t joint_index = 0;
  for (const moveit::core::JointModel* jm : jmg->getActiveJointModels())
    joint_array[joint_index++] = state.getVariablePosition(jm->getFirstVariableIndex());
}

static inline void robotStateToWaypoint(const moveit::core::RobotState& state, 
                                        const std::string& planning_group_name,
                                        rrtconnect::ConfigWaypoint& joint_array)
{
  const moveit::core::JointModelGroup* jmg = state.getJointModelGroup(planning_group_name);
  robotStateToWaypoint(state, jmg, joint_array);
}

/**
 * Writes the provided position values into a robot state.
 *
 * This function requires the dimension of values and joints to be equal!
 *
 * @param values The joint position values to copy from
 * @param joints The joints that should be considered
 * @param state  The robot state to update with the new joint values
 */
static inline void setJointPositions(const rrtconnect::ConfigWaypoint& values, 
                                     const Joints& joints,
                                     moveit::core::RobotState& state)
{
  for (size_t joint_index = 0; joint_index < joints.size(); ++joint_index)
  {
    state.setJointPositions(joints[joint_index], &values[joint_index]);
  }
}

/**
 * Writes the provided position values into a robot state.
 *
 * This function requires the dimension of values and joints to be equal!
 *
 * @param values The joint position values to copy from
 * @param jmg The JointModelGroup that should be considered.
 * @param state  The robot state to update with the new joint values
 */
static inline void setJointPositions(const rrtconnect::ConfigWaypoint& values, 
                                     const moveit::core::JointModelGroup* jmg,
                                     moveit::core::RobotState& state)
{
  setJointPositions(values, jmg->getActiveJointModels(), state);
}

void getJointLimits(const moveit::core::JointModelGroup* jmg,
                    rrtconnect::ConfigWaypoint& lower_limits,
                    rrtconnect::ConfigWaypoint& upper_limits);

// copied from geometry/angles/angles.h
static inline double normalizeAnglePositive(double angle)
{
  return fmod(fmod(angle, 2.0 * M_PI) + 2.0 * M_PI, 2.0 * M_PI);
}

static inline double normalizeAngle(double angle)
{
  double a = normalizeAnglePositive(angle);
  if (a > M_PI)
    a -= 2.0 * M_PI;
  return a;
}

static inline double shortestAngularDistance(double start, double end)
{
  double res = normalizeAnglePositive(normalizeAnglePositive(end) - normalizeAnglePositive(start));
  if (res > M_PI)
  {
    res = -(2.0 * M_PI - res);
  }
  return normalizeAngle(res);
}
}  // rrtc
