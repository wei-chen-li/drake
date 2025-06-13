#include <memory>

#include <gflags/gflags.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 3.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-3,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 4e4, "Young's modulus of the deformable bodies [Pa].");
DEFINE_double(G, 2e4, "Shear modulus of the deformable bodies [Pa].");
DEFINE_double(rho, 50, "Mass density of the deformable bodies [kg/m³].");
DEFINE_double(length, 0.6, "Length of the rope [m].");
DEFINE_double(diameter, 0.015, "Diameter of the rope [m].");
DEFINE_int32(num_edges, 50,
             "Number of edges the rope is spatially discretized.");
DEFINE_double(ball_mass, 1e-3, "Mass of the ball [kg].");
DEFINE_double(ball_radius, 0.03, "Radius of the ball [m].");
DEFINE_string(contact_approximation, "lagged",
              "Type of convex contact approximation. See "
              "multibody::DiscreteContactApproximation for details. Options "
              "are: 'sap', 'lagged', and 'similar'.");

namespace drake {
namespace examples {
namespace {

using drake::geometry::Box;
using drake::geometry::Filament;
using drake::geometry::SceneGraph;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::ForceDensityField;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using drake::systems::Simulator;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using math::RigidTransformd;
using math::RotationMatrixd;

DeformableBodyId RegisterRope(DeformableModel<double>* deformable_model) {
  DRAKE_THROW_UNLESS(FLAGS_num_edges > 0);

  const bool closed = false;
  Eigen::Matrix3Xd node_pos(3, 2);
  node_pos.col(0) = Vector3d(0, 0, 0);
  node_pos.col(1) = Vector3d(0, 0, FLAGS_length);

  Filament filament(closed, node_pos,
                    Filament::CircularCrossSection{.diameter = FLAGS_diameter});

  /* Create the geometry instance from the shape shifted by z = +0.5. */
  const RigidTransformd X_WG(RotationMatrixd::Identity(),
                             Vector3d(0, 0, FLAGS_length * 0.1));
  auto geometry_instance = std::make_unique<geometry::GeometryInstance>(
      X_WG, filament, "cantilever beam");

  /* Add a minimal illustration property for visualization. */
  geometry::IllustrationProperties illus_props;
  illus_props.AddProperty("phong", "diffuse", Vector4d(0.7, 0.5, 0.4, 1.0));
  geometry_instance->set_illustration_properties(std::move(illus_props));

  /* Add a minimal proximity property for collision detection. */
  geometry::ProximityProperties proximity_props;
  const CoulombFriction<double> surface_friction(0.8, 0.8);
  AddContactMaterial({}, {}, surface_friction, &proximity_props);
  geometry_instance->set_proximity_properties(proximity_props);

  /* Set the material properties. Notice G = E / 2(1+𝜈). */
  DeformableBodyConfig<double> config;
  config.set_youngs_modulus(FLAGS_E);
  config.set_poissons_ratio(0.5 * FLAGS_E / FLAGS_G - 1);
  config.set_mass_density(FLAGS_rho);
  config.set_mass_damping_coefficient(1.0);

  /* Add the geometry instance to the deformable model. The filament geometry is
   further discretized based on resolution_hint. */
  const double edge_length = FLAGS_length / FLAGS_num_edges;
  DeformableBodyId body_id = deformable_model->RegisterDeformableBody(
      std::move(geometry_instance), config,
      /* resolution_hint = */ edge_length);

  return body_id;
}

void RegisterGround(MultibodyPlant<double>* plant) {
  const RigidBody<double>& ground = plant->AddRigidBody("ground");
  const double h = 0.01;
  plant->WeldFrames(plant->world_frame(), ground.body_frame(),
                    RigidTransformd(RotationMatrixd(), Vector3d(0, 0, -h / 2)));
  plant->RegisterVisualGeometry(ground, RigidTransformd::Identity(),
                                Box(2, 2, h), "ground",
                                Vector4d(0, 01.0, 1.0, 0.8));
  plant->RegisterCollisionGeometry(ground, RigidTransformd::Identity(),
                                   Box(2, 2, h), "ground",
                                   CoulombFriction(0.8, 0.8));
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();
  RegisterGround(&plant);
  RegisterRope(&deformable_model);
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  geometry::DrakeVisualizer<double>::AddToBuilder(&builder, scene_graph,
                                                  nullptr, params);
  auto diagram = builder.Build();

  Simulator<double> simulator(*diagram);
  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);

  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase the modeling of a cantilever beam using "
      "a deformable filament. Refer to README for instructions on meldis as "
      "well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
