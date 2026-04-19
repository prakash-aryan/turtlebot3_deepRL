// SDF configuration:
//   <plugin filename="obstacle_animator" name="turtlebot3_drlnav::ObstacleAnimator">
//     <duration>160.0</duration>
//     <loop>true</loop>
//     <keyframe time="0"  x="0.0"  y="0.0"  yaw="0.0"/>
//     <keyframe time="10" x="-0.5" y="-1.0"/>
//   </plugin>

#include <algorithm>
#include <chrono>
#include <memory>
#include <vector>

#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/PoseCmd.hh>
#include <gz/sim/components/World.hh>
#include <sdf/Element.hh>

namespace turtlebot3_drlnav
{

struct KeyFrame
{
  double time;
  double x;
  double y;
  double z;
  double yaw;
};

class ObstacleAnimator
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity & _entity,
                 const std::shared_ptr<const sdf::Element> & _sdf,
                 gz::sim::EntityComponentManager & _ecm,
                 gz::sim::EventManager & /*_eventMgr*/) override
  {
    this->model = gz::sim::Model(_entity);
    if (!this->model.Valid(_ecm)) {
      gzerr << "[ObstacleAnimator] Parent entity is not a model; plugin disabled.\n";
      return;
    }

    auto sdf = _sdf->Clone();
    this->duration = sdf->Get<double>("duration", 10.0).first;
    this->loop = sdf->Get<bool>("loop", true).first;

    for (auto e = sdf->GetFirstElement(); e; e = e->GetNextElement()) {
      if (e->GetName() != "keyframe") {
        continue;
      }
      KeyFrame kf;
      kf.time = e->Get<double>("time", 0.0).first;
      kf.x = e->Get<double>("x", 0.0).first;
      kf.y = e->Get<double>("y", 0.0).first;
      kf.z = e->Get<double>("z", 0.0).first;
      kf.yaw = e->Get<double>("yaw", 0.0).first;
      this->keyframes.push_back(kf);
    }
    std::sort(this->keyframes.begin(), this->keyframes.end(),
              [](const KeyFrame & a, const KeyFrame & b) { return a.time < b.time; });

    if (this->keyframes.empty()) {
      gzerr << "[ObstacleAnimator] No <keyframe> entries found; plugin disabled.\n";
      return;
    }

    auto poseComp = _ecm.Component<gz::sim::components::Pose>(_entity);
    if (poseComp) {
      this->origin = poseComp->Data();
    }
    this->initialized = true;
  }

  void PreUpdate(const gz::sim::UpdateInfo & _info,
                 gz::sim::EntityComponentManager & _ecm) override
  {
    if (!this->initialized || _info.paused) {
      return;
    }

    const double t_sec =
        std::chrono::duration<double>(_info.simTime).count();
    double t = t_sec;
    if (this->loop && this->duration > 0.0) {
      t = std::fmod(t_sec, this->duration);
    } else if (t > this->duration) {
      t = this->duration;
    }

    const gz::math::Pose3d offset = this->InterpolatePose(t);
    const gz::math::Pose3d target(this->origin.Pos() + offset.Pos(),
                                  this->origin.Rot() * offset.Rot());

    auto entity = this->model.Entity();
    auto poseComp = _ecm.Component<gz::sim::components::Pose>(entity);
    if (!poseComp) {
      _ecm.CreateComponent(entity, gz::sim::components::Pose(target));
    } else {
      *poseComp = gz::sim::components::Pose(target);
    }

    ++this->step_counter;
    if (this->step_counter >= kMarkInterval) {
      _ecm.SetChanged(entity, gz::sim::components::Pose::typeId,
                      gz::sim::ComponentState::OneTimeChange);
      this->step_counter = 0;
    }
  }

private:
  gz::math::Pose3d InterpolatePose(double t) const
  {
    const auto & kfs = this->keyframes;
    if (t <= kfs.front().time) {
      return PoseFromKey(kfs.front());
    }
    if (t >= kfs.back().time) {
      return PoseFromKey(kfs.back());
    }
    for (size_t i = 1; i < kfs.size(); ++i) {
      if (t <= kfs[i].time) {
        const auto & a = kfs[i - 1];
        const auto & b = kfs[i];
        const double span = b.time - a.time;
        const double frac = span > 1e-9 ? (t - a.time) / span : 0.0;
        const double x = a.x + (b.x - a.x) * frac;
        const double y = a.y + (b.y - a.y) * frac;
        const double z = a.z + (b.z - a.z) * frac;
        const double yaw = a.yaw + (b.yaw - a.yaw) * frac;
        return gz::math::Pose3d(x, y, z, 0.0, 0.0, yaw);
      }
    }
    return PoseFromKey(kfs.back());
  }

  static gz::math::Pose3d PoseFromKey(const KeyFrame & kf)
  {
    return gz::math::Pose3d(kf.x, kf.y, kf.z, 0.0, 0.0, kf.yaw);
  }

  gz::sim::Model model{gz::sim::kNullEntity};
  gz::math::Pose3d origin{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<KeyFrame> keyframes;
  double duration{10.0};
  bool loop{true};
  bool initialized{false};

  // Throttle SetChanged to ~scene-broadcaster rate to avoid publishing
  // stale/fresh pose pairs within a single broadcast cycle.
  static constexpr int kMarkInterval{16};
  int step_counter{0};
};

}  // namespace turtlebot3_drlnav

GZ_ADD_PLUGIN(turtlebot3_drlnav::ObstacleAnimator,
              gz::sim::System,
              turtlebot3_drlnav::ObstacleAnimator::ISystemConfigure,
              turtlebot3_drlnav::ObstacleAnimator::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(turtlebot3_drlnav::ObstacleAnimator,
                    "turtlebot3_drlnav::ObstacleAnimator")
