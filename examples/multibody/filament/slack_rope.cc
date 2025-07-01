#include <memory>

#include <gflags/gflags.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/sine.h"

DEFINE_double(simulation_time, 10.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(link_size, 0.05, "Size of the links [m].");
DEFINE_double(link_position, 0.5, "Position of the links [m].");
DEFINE_double(link_density, 1000, "Mass density of the links [kg/m³].");
DEFINE_double(rope_diameter, 5e-3, "Diameter of the rope [m].");
DEFINE_double(rope_E, 1e7, "Young's modulus of the rope [Pa].");
DEFINE_double(rope_density, 1000, "Mass density of the rope [kg/m³].");
DEFINE_int32(rope_N, 100, "Number of spatial discretization of the rope.");
DEFINE_double(P_gain, 1e6, "Proportional gain for position controller.");

namespace drake {
namespace examples {
namespace filament {
namespace {

using drake::geometry::Box;
using drake::geometry::Filament;
using drake::geometry::SceneGraph;
using drake::geometry::Sphere;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::JointActuator;
using drake::multibody::JointActuatorIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::PdControllerGains;
using drake::multibody::PrismaticJoint;
using drake::multibody::RevoluteJoint;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using drake::systems::LeafSystem;
using drake::systems::Simulator;
using drake::systems::Sine;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using math::RigidTransformd;
using math::RotationMatrixd;

template <typename T>
class DeriredStateSource : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeriredStateSource);

  DeriredStateSource(const T& amplitude, const T& frequency, int num_actuators)
      : amplitude_(amplitude),
        frequency_(frequency),
        num_actuators_(num_actuators) {
    this->DeclareVectorOutputPort(systems::kUseDefaultName, num_actuators * 2,
                                  &DeriredStateSource<T>::CalcOutput);
  }

 private:
  void CalcOutput(const Context<T>& context,
                  systems::BasicVector<T>* output) const {
    Eigen::VectorBlock<VectorX<T>> source = output->get_mutable_value();
    const double t = context.get_time();
    const T omega = 2 * M_PI * frequency_;
    const T& A = 0.5 * amplitude_;
    source.head(num_actuators_).array() = (1 - cos(omega * t)) * A;
    source.tail(num_actuators_).array() = omega * sin(omega * t) * A;
  }

  T amplitude_;
  T frequency_;
  int num_actuators_;
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();

  /* Add link 1. */
  const RigidBody<double>& link1 =
      plant.AddRigidBody("link1", SpatialInertia<double>::SolidBoxWithDensity(
                                      FLAGS_link_density, FLAGS_link_size,
                                      FLAGS_link_size, FLAGS_link_size * 0.1));
  const RigidTransformd X_WL1(RotationMatrixd::MakeYRotation(M_PI / 2));
  const PrismaticJoint<double>& joint1 =
      plant.AddJoint<PrismaticJoint>("joint1", plant.world_body(), X_WL1, link1,
                                     RigidTransformd(), Vector3d(0, 0, 1));
  JointActuatorIndex joint1_index =
      plant.AddJointActuator("actuator1", joint1).index();
  plant.get_mutable_joint_actuator(joint1_index)
      .set_controller_gains(PdControllerGains{FLAGS_P_gain, 0.0});
  plant.RegisterVisualGeometry(
      link1, RigidTransformd(),
      Box(FLAGS_link_size, FLAGS_link_size, FLAGS_link_size * 0.1), "link1",
      Vector4d(1, 0, 0, 1));

  /* Add link 2. */
  const RigidBody<double>& link2 =
      plant.AddRigidBody("link2", SpatialInertia<double>::SolidBoxWithDensity(
                                      FLAGS_link_density, FLAGS_link_size,
                                      FLAGS_link_size, FLAGS_link_size * 0.1));
  const RigidTransformd X_WL2(RotationMatrixd::MakeYRotation(-M_PI / 2),
                              Vector3d(FLAGS_link_position, 0, 0));
  const PrismaticJoint<double>& joint2 =
      plant.AddJoint<PrismaticJoint>("joint2", plant.world_body(), X_WL2, link2,
                                     RigidTransformd(), Vector3d(0, 0, 1));
  JointActuatorIndex joint2_index =
      plant.AddJointActuator("actuator2", joint2).index();
  plant.get_mutable_joint_actuator(joint2_index)
      .set_controller_gains(PdControllerGains{FLAGS_P_gain, 0.0});
  plant.RegisterVisualGeometry(
      link2, RigidTransformd(),
      Box(FLAGS_link_size, FLAGS_link_size, FLAGS_link_size * 0.1), "link2",
      Vector4d(0, 1, 0, 1));

  /* Add a rope. */
  const double edge_length = FLAGS_link_position / FLAGS_rope_N;
  Eigen::Matrix3Xd node_pos(3, 2);
  node_pos.col(0) = Vector3d(0, 0, 0);
  node_pos.col(1) = Vector3d(FLAGS_link_position, 0, 0);
  Filament filament(
      false, node_pos,
      Filament::CircularCrossSection{.diameter = FLAGS_rope_diameter});
  auto geometry_instance = std::make_unique<geometry::GeometryInstance>(
      RigidTransformd(), filament, "rope");
  geometry::IllustrationProperties illus_props;
  illus_props.AddProperty("phong", "diffuse", Vector4d(0, 0, 1, 1));
  geometry_instance->set_illustration_properties(std::move(illus_props));

  DeformableBodyConfig<double> config;
  config.set_youngs_modulus(FLAGS_rope_E);
  config.set_poissons_ratio(0.25);
  config.set_mass_density(FLAGS_rope_density);
  config.set_mass_damping_coefficient(1.0);

  DeformableBodyId rope_body_id = deformable_model.RegisterDeformableBody(
      std::move(geometry_instance), config, edge_length);

  /* Attach the flexible shaft to the input link and output link. */
  const Sphere bbox(edge_length * 0.1);
  deformable_model.AddFixedConstraint(rope_body_id, link1, X_WL1.inverse(),
                                      bbox, RigidTransformd());
  deformable_model.AddFixedConstraint(rope_body_id, link2, X_WL2.inverse(),
                                      bbox, RigidTransformd());

  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  geometry::DrakeVisualizer<double>::AddToBuilder(&builder, scene_graph,
                                                  nullptr, params);
  /* Make the actuator track a sine position. */
  DeriredStateSource<double>* source =
      builder.AddSystem<DeriredStateSource<double>>(FLAGS_link_position * 0.3,
                                                    0.5 /* Hz */, 2);
  builder.Connect(
      source->get_output_port(),
      plant.get_desired_state_input_port(multibody::ModelInstanceIndex(1)));
  auto diagram = builder.Build();

  Simulator<double> simulator(*diagram);
  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);

  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace filament
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase the modeling of a slack rope using "
      "a deformable filament. Refer to README for instructions on meldis as "
      "well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::filament::do_main();
}
