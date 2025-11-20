#include <vector>
#include <pluginlib/class_list_macros.hpp>

#include <moveit/collision_distance_field/collision_detector_allocator_hybrid.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/utils/logger.hpp>

#include <rrtc_interface/rrtc_planning_context.hpp>


namespace rrtc_interface
{
namespace
{
rclcpp::Logger getLogger()
{
  return moveit::getLogger("moveit.linqs.rrtc.planner_manager");
}
}  // namespace

class RRTCPlannerManager : public planning_interface::PlannerManager
{
public:
  RRTCPlannerManager() : planning_interface::PlannerManager()
  {
  }

  bool initialize(const moveit::core::RobotModelConstPtr& model, 
                  const rclcpp::Node::SharedPtr& node,
                  const std::string& /* unused */) override
  {
    for (const std::string& group : model->getJointModelGroupNames())
    {
      planning_contexts_[group] = std::make_shared<RRTCPlanningContext>("rrtc_planning_context", group, model, node);
    }
    return true;
  }

  planning_interface::PlanningContextPtr
  getPlanningContext(const planning_scene::PlanningSceneConstPtr& planning_scene,
                     const planning_interface::MotionPlanRequest& req,
                     moveit_msgs::msg::MoveItErrorCodes& error_code) const override
  {
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

    if (req.group_name.empty())
    {
      RCLCPP_ERROR(getLogger(), "No group specified to plan for");
      error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
      return planning_interface::PlanningContextPtr();
    }

    if (!planning_scene)
    {
      RCLCPP_ERROR(getLogger(), "No planning scene supplied as input");
      error_code.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
      return planning_interface::PlanningContextPtr();
    }

    // create PlanningScene using hybrid collision detector
    planning_scene::PlanningScenePtr ps = planning_scene->diff();
    ps->allocateCollisionDetector(collision_detection::CollisionDetectorAllocatorHybrid::create());

    // retrieve and configure existing context
    const RRTCPlanningContextPtr& context = planning_contexts_.at(req.group_name);
    context->setPlanningScene(ps);
    context->setMotionPlanRequest(req);
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    return context;
  }

  bool canServiceRequest(const planning_interface::MotionPlanRequest& /*req*/) const override
  {
    // TODO: this is a dummy implementation
    //      capabilities.dummy = false;
    return true;
  }

  std::string getDescription() const override
  {
    return "RRTC";
  }

  void getPlanningAlgorithms(std::vector<std::string>& algs) const override
  {
    algs.resize(1);
    algs[0] = "RRTC";
  }

protected:
  std::map<std::string, RRTCPlanningContextPtr> planning_contexts_;
};

}  // namespace rrtc_interface

PLUGINLIB_EXPORT_CLASS(rrtc_interface::RRTCPlannerManager, planning_interface::PlannerManager)
