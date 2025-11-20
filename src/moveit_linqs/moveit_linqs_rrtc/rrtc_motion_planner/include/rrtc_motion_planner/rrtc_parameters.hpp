#include <string>
#include <vector>

#pragma once

namespace rrtc
{
class RRTConnectParameters
{
public:
  RRTConnectParameters() = default;
  virtual ~RRTConnectParameters() = default;

public:
  // Default maximum configuration-space step in radians.
  static constexpr double kDefaultConfigurationSpaceStep = 0.1;

  double planning_time_limit_ = 0 ; /*!< the maximum time (in seconds) to spend planning, set to zero to disable. */
  int max_iterations_ = 10000;  /*!< the maximum number of iterations to use when sampling new positions. */
  double configuration_space_step_ = kDefaultConfigurationSpaceStep; /*!< no configuration space step will be larger in magnitude than this value. */
  int iters_between_goal_checks_ = 3;  /*!< the number of samples between occasions when a direct path to the goal is tried. If <2 it is never tried.*/
};

}  // namespace rrtc
