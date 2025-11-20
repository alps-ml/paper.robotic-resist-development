#include <rrtc_motion_planner/rrtc_utils.hpp>
#include <rrtconnect/rrt_connect.h>

#include <moveit/robot_model/joint_model.hpp>
#include <vector>

namespace rrtc
{

void getJointLimits(const moveit::core::JointModelGroup* jmg,
                    rrtconnect::ConfigWaypoint& lower_limits,
                    rrtconnect::ConfigWaypoint& upper_limits)
{
    const unsigned int num_active_dof = jmg->getActiveVariableCount();

    lower_limits.resize(num_active_dof);
    upper_limits.resize(num_active_dof);

    unsigned int current_var_idx = 0;
    for (const moveit::core::JointModel* jm : jmg->getActiveJointModels())
    {
    const std::vector<moveit::core::VariableBounds>& var_bounds = jm->getVariableBounds();
    for (size_t i = 0; i < jm->getVariableCount(); ++i)
    {
        const auto& bound = var_bounds[i];
        lower_limits(current_var_idx) = bound.min_position_;
        upper_limits(current_var_idx) = bound.max_position_;

        // Planners often require finite bounds. If MoveIt reports infinite bounds,
        // they might need to be clamped to large finite numbers.
        if (!bound.position_bounded_)
        {
            if (bound.min_position_ == -std::numeric_limits<double>::infinity())
            {
            lower_limits(current_var_idx) = -1e10; // A large negative number
            }
            if (bound.max_position_ == std::numeric_limits<double>::infinity())
            {
            upper_limits(current_var_idx) = 1e10;  // A large positive number
            }
            // For continuous revolute joints, MoveIt often sets bounds like [-PI, PI]
            // and position_bounded_ is false. The shortestAngularDistance logic
            // in your planner handles the wrapping for these.
        }
        }
        current_var_idx++;
    }
}

}  // namespace rrtc