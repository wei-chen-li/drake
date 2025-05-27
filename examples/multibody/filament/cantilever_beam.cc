#include <memory>

#include <gflags/gflags.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/force_density_field.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 10.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e7, "Young's modulus of the deformable bodies [Pa].");
DEFINE_double(G, 5e7, "Shear modulus of the deformable bodies [Pa].");
DEFINE_double(rho, 50, "Mass density of the deformable bodies [kg/m³].");
DEFINE_double(length, 1.0, "Length of the cantilever beam [m].");
DEFINE_double(width, 0.02, "Width of the cantilever beam [m].");
DEFINE_int32(num_edges, 100,
             "Number of edges the cantilever beam is spatially discretized.");
DEFINE_string(shape, "line", "Shape of the beam. \"line\" or \"circle\"");
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

using drake::geometry::Filament;
using drake::geometry::SceneGraph;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::ForceDensityField;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using drake::systems::Simulator;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using math::RigidTransform;
using math::RotationMatrix;

DeformableBodyId RegisterCantileverBeam(
    DeformableModel<double>* deformable_model, bool circular = false) {
  DRAKE_THROW_UNLESS(FLAGS_num_edges > 0);

  bool closed;
  Eigen::Matrix3Xd node_pos;
  Vector3d first_edge_m1;
  if (!circular) {
    /* The beam has an initial shape of a line from (0,0,0) to (length,0,0). */
    closed = false;
    node_pos.resize(3, 2);
    node_pos.col(0) = Vector3d(0, 0, 0);
    node_pos.col(1) = Vector3d(FLAGS_length, 0, 0);
    first_edge_m1 = Vector3d(0, 1, 0);
  } else {
    /* The beam has an initial shape of a circle. */
    closed = true;
    const int num_nodes = FLAGS_num_edges;
    const auto theta =
        VectorXd::LinSpaced(num_nodes + 1, 0, 2 * M_PI).head(num_nodes).array();
    node_pos.resize(3, num_nodes);
    node_pos.row(0) = 0.5 * FLAGS_length * (cos(theta) + 1);
    node_pos.row(1) = 0.5 * FLAGS_length * sin(theta);
    node_pos.row(2) = VectorXd::Zero(num_nodes);
    first_edge_m1 = Vector3d(-1, 0, 0);
  }
  Filament filament(closed, node_pos, first_edge_m1,
                    Filament::CrossSection{.type = Filament::kRectangular,
                                           .width = FLAGS_width,
                                           .height = FLAGS_width});

  /* Create the geometry instance from the shape shifted by z = +0.5. */
  const RigidTransform<double> X_WG(RotationMatrix<double>::Identity(),
                                    Vector3d(0, 0, 0.5));
  auto geometry_instance = std::make_unique<geometry::GeometryInstance>(
      X_WG, filament, "cantilever beam");

  /* Add a minimal illustration property for visualization. */
  geometry::IllustrationProperties illus_props;
  illus_props.AddProperty("phong", "diffuse", Vector4d(0.7, 0.5, 0.4, 1.0));
  geometry_instance->set_illustration_properties(std::move(illus_props));

  /* Add a minimal proximity property for collision detection. */
  geometry::ProximityProperties proximity_props;
  geometry_instance->set_proximity_properties(proximity_props);

  /* Set the material properties. Notice G = E / 2(1+𝜈). */
  DeformableBodyConfig<double> config;
  config.set_youngs_modulus(FLAGS_E);
  config.set_poissons_ratio(0.5 * FLAGS_E / FLAGS_G - 1);
  config.set_mass_density(FLAGS_rho);
  config.set_mass_damping_coefficient(1.0);

  /* Add the geometry instance to the deformable model. The filament geometry is
   further discretized based on resolution_hint. */
  const double edge_length =
      (circular ? M_PI * FLAGS_length : FLAGS_length) / FLAGS_num_edges;
  DeformableBodyId body_id = deformable_model->RegisterDeformableBody(
      std::move(geometry_instance), config,
      /* resolution_hint = */ edge_length);

  /* Fix the first two nodes effectively makes the beam have a clamped end. */
  deformable_model->SetWallBoundaryCondition(
      body_id, Vector3d(edge_length * 1.001, 0, 0), Vector3d(1, 0, 0));

  return body_id;
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();

  DRAKE_THROW_UNLESS(FLAGS_shape == "line" || FLAGS_shape == "circle");
  DeformableBodyId body_id =
      RegisterCantileverBeam(&deformable_model, FLAGS_shape != "line");
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  geometry::DrakeVisualizer<double>::AddToBuilder(&builder, scene_graph,
                                                  nullptr, params);

  auto diagram = builder.Build();

  Simulator<double> simulator(*diagram);

  Context<double>& mutable_root_context = simulator.get_mutable_context();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, &mutable_root_context);
  deformable_model.is_enabled(body_id, plant_context);

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
