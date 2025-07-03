#include <memory>

#include <gflags/gflags.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/multiplexer.h"
#include "drake/systems/primitives/sine.h"

DEFINE_double(simulation_time, 30.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(link_size, 0.05, "Size of the input and output links [m].");
DEFINE_double(link_position, 0.4,
              "Position of the input and output links [m].");
DEFINE_double(link_density, 1000,
              "Mass density of the input and output links [kg/m³].");
DEFINE_double(shaft_diameter, 5e-3, "Diameter of the flexible shaft [m].");
DEFINE_double(shaft_E, 1e6, "Young's modulus of the flexible shaft [Pa].");
DEFINE_double(shaft_density, 1000,
              "Mass density of the flexible shaft [kg/m³].");
DEFINE_double(shaft_N, 100,
              "Number of spatial discretization of the flexible shaft.");

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
using drake::multibody::RevoluteJoint;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using drake::systems::Multiplexer;
using drake::systems::Simulator;
using drake::systems::Sine;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using math::RigidTransformd;
using math::RotationMatrixd;

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();

  /* Add input link. */
  const RigidBody<double>& link1 = plant.AddRigidBody(
      "input_link", SpatialInertia<double>::SolidBoxWithDensity(
                        FLAGS_link_density, FLAGS_link_size, FLAGS_link_size,
                        FLAGS_link_size * 0.1));
  const RigidTransformd X_WL1;
  const RevoluteJoint<double>& joint1 =
      plant.AddJoint<RevoluteJoint>("joint1", plant.world_body(), X_WL1, link1,
                                    RigidTransformd(), Vector3d(0, 0, 1));
  plant.RegisterVisualGeometry(
      link1, RigidTransformd(),
      Box(FLAGS_link_size, FLAGS_link_size, FLAGS_link_size * 0.1),
      "input_link", Vector4d(1, 0, 0, 1));

  /* Add output link. */
  const RigidBody<double>& link2 = plant.AddRigidBody(
      "output_link", SpatialInertia<double>::SolidBoxWithDensity(
                         FLAGS_link_density, FLAGS_link_size, FLAGS_link_size,
                         FLAGS_link_size * 0.1));
  const RigidTransformd X_WL2(RotationMatrixd::MakeYRotation(M_PI),
                              Vector3d(0, 0, FLAGS_link_position));
  plant.AddJoint<RevoluteJoint>("joint2", plant.world_body(), X_WL2, link2,
                                RigidTransformd(), Vector3d(0, 0, 1));
  plant.RegisterVisualGeometry(
      link2, RigidTransformd(),
      Box(FLAGS_link_size, FLAGS_link_size, FLAGS_link_size * 0.1),
      "output_link", Vector4d(0, 1, 0, 1));

  /* Add flexible shaft. */
  const double edge_length = FLAGS_link_position / FLAGS_shaft_N;
  Eigen::Matrix3Xd node_pos(3, 2);
  node_pos.col(0) = Vector3d(0, 0, 0);
  node_pos.col(1) = Vector3d(0, 0, FLAGS_link_position);
  Filament filament(
      false, node_pos,
      Filament::RectangularCrossSection{.width = FLAGS_shaft_diameter,
                                        .height = FLAGS_shaft_diameter},
      Vector3d(1, 0, 0));
  auto geometry_instance = std::make_unique<geometry::GeometryInstance>(
      RigidTransformd(), filament, "twisting shaft");
  geometry::IllustrationProperties illus_props;
  illus_props.AddProperty("phong", "diffuse", Vector4d(0, 0, 1, 1));
  geometry_instance->set_illustration_properties(std::move(illus_props));

  DeformableBodyConfig<double> config;
  config.set_youngs_modulus(FLAGS_shaft_E);
  config.set_poissons_ratio(0.25);
  config.set_mass_density(FLAGS_shaft_density);
  config.set_mass_damping_coefficient(1.0);

  DeformableBodyId shaft_body_id = deformable_model.RegisterDeformableBody(
      std::move(geometry_instance), config, edge_length);

  /* Attach the flexible shaft to the input link and output link. */
  const Sphere bbox(edge_length * 1.0001);
  deformable_model.AddFixedConstraint(shaft_body_id, link1, X_WL1.inverse(),
                                      bbox, RigidTransformd());
  deformable_model.AddFixedConstraint(shaft_body_id, link2, X_WL2.inverse(),
                                      bbox, RigidTransformd());

  /* Add actuator driving the input link. */
  JointActuatorIndex actuator_index =
      plant.AddJointActuator("motor", joint1).index();
  plant.get_mutable_joint_actuator(actuator_index)
      .set_controller_gains(PdControllerGains{10.0, 0.0});

  plant.mutable_gravity_field().set_gravity_vector(Vector3d::Zero());
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  geometry::DrakeVisualizer<double>::AddToBuilder(&builder, scene_graph,
                                                  nullptr, params);
  /* Make the motor track a position of -cos(t). */
  Sine<double>* source =
      builder.AddSystem<Sine<double>>(2 * M_PI, 1, -M_PI / 2, 1);
  Multiplexer<double>* mux = builder.AddSystem<Multiplexer<double>>(2);
  builder.Connect(source->get_output_port(0), mux->get_input_port(0));
  builder.Connect(source->get_output_port(1), mux->get_input_port(1));
  builder.Connect(
      mux->get_output_port(),
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
      "This is a demo used to showcase the modeling of a twisting shaft using "
      "a "
      "deformable filament. Refer to README for instructions on meldis as "
      "well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::filament::do_main();
}
