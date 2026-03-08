#include "easy_simulation/suction_gripper_gazebo.hpp"

using namespace easy_simulation_plugins;

void SuctionGripperPlugin::Configure(
    const gz::sim::Entity &entity,
    const std::shared_ptr<const sdf::Element> &sdf,
    gz::sim::EntityComponentManager &ecm,
    gz::sim::EventManager &)
{
  // Suction link
  auto matches = gz::sim::entitiesFromScopedName(
      "end_effector_link_fixed_joint_lump__gripper_collision",
      ecm);

  if (!matches.empty())
    suction_link_collision_ = *matches.begin();
  else
    std::cerr << "[SuctionGripperPlugin] Suction link not found!" << std::endl;

  // Subscribe to suction command
  node_.Subscribe("cmd_suction", &SuctionGripperPlugin::OnCmd, this);
  cmd_suction_state_pub_ = node_.Advertise<gz::msgs::Boolean>("cmd_suction_state");
  suction_state_pub_ = node_.Advertise<gz::msgs::Boolean>("suction_state");

  std::cout << "[SuctionGripperPlugin] Plugin initialized, suction link id: " << suction_link_collision_ << std::endl;
}

void SuctionGripperPlugin::OnCmd(const gz::msgs::Boolean &msg)
{
  std::cout << "[SuctionGripperPlugin] Received suction command: " << msg.data() << std::endl;
  suction_on_ = msg.data();

  // Publish current cmd suction state
  gz::msgs::Boolean state_msg;
  state_msg.set_data(suction_on_);
  cmd_suction_state_pub_.Publish(state_msg);
}

void SuctionGripperPlugin::PreUpdate(
    const gz::sim::UpdateInfo &,
    gz::sim::EntityComponentManager &ecm)
{
  // Suction OFF → detach if needed
  if (!suction_on_ && attached_object_ != gz::sim::kNullEntity)
  {
    RemoveConstraint(ecm);
    return;
  }

  // Already attached → nothing to do
  if (attached_object_ != gz::sim::kNullEntity)
    return;

  // Suction ON → check contact
  // Update box objects list when turning suction on
  if (suction_on_)
  {
    // std::cout << "[SuctionGripperPlugin] Suction ON, scanning for boxes..." << std::endl;
    box_objects_.clear();
    // Find all "box" objects
    for (auto ent : ecm.EntitiesByComponents(gz::sim::components::Model()))
    {
      auto name = gz::sim::scopedName(ent, ecm, "::", false);
      if (name.find("box") != std::string::npos)
        box_objects_.push_back(ent);
    }
    // std::cout << "[SuctionGripperPlugin] Found " << box_objects_.size() << " box objects:" << std::endl;
    // for (const auto &box : box_objects_)
    // {
    //   std::cout << " - " << gz::sim::scopedName(box, ecm) << std::endl;
    // }
    ScanForContactAndAttach(ecm);
  }
}

void SuctionGripperPlugin::ScanForContactAndAttach(
    gz::sim::EntityComponentManager &ecm)
{
  // Get the contact sensor component from the suction link
  std::cout << "[SuctionGripperPlugin] Get parent link of the collision" << std::endl;
  auto suction_link = ecm.Component<gz::sim::components::ParentEntity>(suction_link_collision_)->Data();
  std::cout << "[SuctionGripperPlugin] Scanning for contacts..." << std::endl;
  auto contacts =
      ecm.Component<gz::sim::components::ContactSensorData>(suction_link_collision_);

  if (!contacts)
  {
    std::cerr << "[SuctionGripperPlugin] No contact sensor data found on suction link!" << std::endl;
    return;
  }

  std::cout << "[SuctionGripperPlugin] Number of contacts: " << contacts->Data().contact_size() << std::endl;

  // Iterate safely
  for (const auto &c : contacts->Data().contact())
  {
    auto coll1Opt = ecm.EntityByName(c.collision1().name());
    auto coll2Opt = ecm.EntityByName(c.collision2().name());

    std::cout << "[SuctionGripperPlugin] Contact between: "
              << c.collision1().id() << " and " << c.collision2().id() << std::endl;

    gz::sim::Entity coll1Entity = c.collision1().id();
    gz::sim::Entity coll2Entity = c.collision2().id();

    // Skip if nothing valid
    if (coll1Entity == gz::sim::kNullEntity && coll2Entity == gz::sim::kNullEntity)
      continue;
    const auto &normal_msg = c.normal(0);
    // Get suction axis in world frame (example: +Z of suction link)
    auto suction_pose = gz::sim::worldPose(suction_link, ecm);
    gz::math::Vector3d suction_dir = suction_pose.Rot().RotateVector(gz::math::Vector3d(0,0,1));
    gz::math::Vector3d contact_normal(normal_msg.x(), normal_msg.y(), normal_msg.z());

    double dot_product = suction_dir.Dot(contact_normal);
    std::cout << "[SuctionGripperPlugin] Contact entities: "
              << coll1Entity << " and " << coll2Entity << std::endl;
    std::cout << "[SuctionGripperPlugin] Contact normal: "
              << normal_msg.x() << ", " << normal_msg.y() << ", " << normal_msg.z() << std::endl;
    std::cout << "[SuctionGripperPlugin] Suction direction: "
              << suction_dir.X() << ", " << suction_dir.Y() << ", " << suction_dir.Z() << std::endl;
    std::cout << "[SuctionGripperPlugin] Dot product between suction direction and contact normal: "
              << dot_product << std::endl;

    // Check if contact normal aligns with suction direction
    if (std::abs(dot_product) < 0.98) // Adjust threshold as needed
    {
      std::cerr << "[SuctionGripperPlugin] Contact normal does not align with suction direction." << std::endl;
      continue;
    }

    // Pick one that is not the suction link
    gz::sim::Entity other =
        (coll1Entity != suction_link_collision_) ? coll1Entity : coll2Entity;

    if (other == suction_link_collision_ || other == gz::sim::kNullEntity)
      continue;

    std::cout << "[SuctionGripperPlugin] Other contact entity: "
              << other << std::endl;

    // Match model that contains "box"
    // Get parent model
    auto parentComp = ecm.Component<gz::sim::components::ParentEntity>(other);
    auto grandParentComp = ecm.Component<gz::sim::components::ParentEntity>(parentComp->Data());
    if (!parentComp)
      continue;

    std::cout << "[SuctionGripperPlugin] Parent entity: "
              << parentComp->Data() << std::endl;

    gz::sim::Entity model = grandParentComp->Data();

    if (std::find(box_objects_.begin(), box_objects_.end(), model) != box_objects_.end())
    {
      CreateConstraint(ecm, parentComp->Data());
      return;
    }

    std::cerr << "[SuctionGripperPlugin] No valid box found to attach." << std::endl;
  }
}

void SuctionGripperPlugin::CreateConstraint(
    gz::sim::EntityComponentManager &ecm,
    gz::sim::Entity object)
{
  attached_object_ = object;
  auto suction_link = ecm.Component<gz::sim::components::ParentEntity>(suction_link_collision_)->Data();

  // Create a new joint entity
  constraint_joint_ = ecm.CreateEntity();

  ecm.CreateComponent(constraint_joint_,
                      gz::sim::components::DetachableJoint({suction_link, object, "fixed"}));

  std::cout << "[SuctionGripper] Attached "
            << gz::sim::scopedName(object, ecm)
            << " to suction link "
            << gz::sim::scopedName(suction_link, ecm)
            << " by entity " << constraint_joint_
            << std::endl;

  gz::msgs::Boolean state_msg;
  state_msg.set_data(true);
  suction_state_pub_.Publish(state_msg);
}

void SuctionGripperPlugin::RemoveConstraint(
    gz::sim::EntityComponentManager &ecm)
{
  if (constraint_joint_ != gz::sim::kNullEntity)
  {
    std::cout << "[SuctionGripper] Detaching object "
              << gz::sim::scopedName(attached_object_, ecm)
              << " from suction link by removing joint entity "
              << constraint_joint_
              << std::endl;

    ecm.RequestRemoveEntity(constraint_joint_);
    gz::msgs::Boolean state_msg;
    state_msg.set_data(false);
    suction_state_pub_.Publish(state_msg);
  }

  attached_object_ = gz::sim::kNullEntity;
  constraint_joint_ = gz::sim::kNullEntity;
}

#include <gz/plugin/Register.hh>

GZ_ADD_PLUGIN(
    easy_simulation_plugins::SuctionGripperPlugin,
    gz::sim::System,
    easy_simulation_plugins::SuctionGripperPlugin::ISystemConfigure,
    easy_simulation_plugins::SuctionGripperPlugin::ISystemPreUpdate)