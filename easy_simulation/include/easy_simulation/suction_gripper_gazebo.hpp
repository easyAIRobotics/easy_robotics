#ifndef SUCTION_GRIPPER_GAZEBO_HPP_
#define SUCTION_GRIPPER_GAZEBO_HPP_

#include <gz/sim/System.hh>
#include <gz/msgs.hh>
#include <gz/transport.hh>

#include "gz/sim/Conversions.hh"
#include "gz/sim/EntityComponentManager.hh"
#include "gz/sim/Util.hh"
#include "gz/sim/components/DetachableJoint.hh"
#include "gz/sim/components/Collision.hh"
#include "gz/sim/components/ContactSensor.hh"
#include "gz/sim/components/ContactSensorData.hh"
#include "gz/sim/components/Link.hh"
#include "gz/sim/components/Name.hh"
#include "gz/sim/components/ParentEntity.hh"
#include "gz/sim/components/Joint.hh"
#include "gz/sim/components/JointType.hh"
#include "gz/sim/components/Model.hh"

#include "gz/math/Vector3.hh"


namespace easy_simulation_plugins
{

  class SuctionGripperPlugin
      : public gz::sim::System,
        public gz::sim::ISystemConfigure,
        public gz::sim::ISystemPreUpdate
  {
  public:
    void Configure(const gz::sim::Entity &entity,
                   const std::shared_ptr<const sdf::Element> &sdf,
                   gz::sim::EntityComponentManager &ecm,
                   gz::sim::EventManager &) override;

    void PreUpdate(const gz::sim::UpdateInfo &info,
                   gz::sim::EntityComponentManager &ecm) override;

  private:
    gz::transport::Node node_;
    gz::transport::Node::Publisher cmd_suction_state_pub_;
    gz::transport::Node::Publisher suction_state_pub_;

    bool suction_on_ = false;
    void OnCmd(const gz::msgs::Boolean &msg);

    // Suction link
    gz::sim::Entity suction_link_collision_;

    // List of objects matching "box"
    std::vector<gz::sim::Entity> box_objects_;

    // Attached object
    gz::sim::Entity attached_object_{gz::sim::kNullEntity};
    gz::sim::Entity constraint_joint_{gz::sim::kNullEntity};

    void ScanForContactAndAttach(gz::sim::EntityComponentManager &ecm);
    void CreateConstraint(gz::sim::EntityComponentManager &ecm, gz::sim::Entity object);
    void RemoveConstraint(gz::sim::EntityComponentManager &ecm);
  };

} // namespace easy_simulation

#endif // SUCTION_GRIPPER_GAZEBO_HPP_
