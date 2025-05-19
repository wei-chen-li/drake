#include <memory>

#include <gflags/gflags.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/multibody/der/der_model.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 5.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e9, "Young's modulus of the deformable bodies [Pa].");
DEFINE_double(G, 0.3e9, "Shear modulus of the deformable bodies [Pa].");
DEFINE_double(rho, 50, "Mass density of the deformable bodies [kg/m³].");
DEFINE_double(length, 1.0, "Length of the cantilever beam [m].");
DEFINE_double(radius, 0.15, "Radius of the cantilever beam [m].");
DEFINE_int32(num_edges, 300,
             "Number of edges the cantilever beam is spatially discretized.");
DEFINE_string(contact_approximation, "lagged",
              "Type of convex contact approximation. See "
              "multibody::DiscreteContactApproximation for details. Options "
              "are: 'sap', 'lagged', and 'similar'.");
DEFINE_double(
    contact_damping, 10.0,
    "Hunt and Crossley damping for the deformable body, only used when "
    "'contact_approximation' is set to 'lagged' or 'similar' [s/m].");

namespace drake {
namespace examples {
namespace {

using drake::multibody::AddMultibodyPlant;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::der::DerEdgeIndex;
using drake::multibody::der::DerModel;
using drake::multibody::der::DerNodeIndex;
using drake::systems::Context;
using Eigen::Vector3d;

void RegisterCantileverBeam(DeformableModel<double>* deformable_model) {
  DRAKE_THROW_UNLESS(FLAGS_num_edges > 0);
  DerModel<double>::Builder builder;
  const Vector3d d1_0 = Vector3d(0, 1, 0);
  const double dx = FLAGS_length / FLAGS_num_edges;
  builder.AddFirstEdge(Vector3d(0, 0, 0), 0, Vector3d(dx, 0, 0), d1_0);
  builder.FixNode(DerNodeIndex(0));
  builder.FixEdge(DerEdgeIndex(0));
  builder.FixNode(DerNodeIndex(1));
  for (int i = 2; i < FLAGS_num_edges + 1; ++i) {
    builder.AddEdge(0, Vector3d(dx * i, 0, 0));
  }
  builder.SetZeroUndeformedCurvatureAndTwist();
  builder.SetMaterialProperties(FLAGS_E, FLAGS_G, FLAGS_rho);
  builder.SetCircularCrossSection(FLAGS_radius);
  builder.SetDampingCoefficients(0.0, 0.0);

  std::unique_ptr<DerModel<double>> model = builder.Build();
  unused(deformable_model);
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();
  RegisterCantileverBeam(&deformable_model);

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  geometry::DrakeVisualizer<double>::AddToBuilder(&builder, scene_graph,
                                                  nullptr, params);

  auto diagram = builder.Build();

  systems::Simulator<double> simulator(*diagram);

  Context<double>& mutable_root_context = simulator.get_mutable_context();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, &mutable_root_context);

  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);

  simulator.AdvanceTo(FLAGS_simulation_time);
  unused(plant_context);

  return 0;
}

}  // namespace
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase enabling/disabling of deformable bodies."
      "Deformable torus bodies are stacked on top of each other and enabled "
      "one-by-one. Refer to README for instructions on meldis as well as "
      "optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
