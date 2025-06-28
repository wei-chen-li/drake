#include <fstream>
#include <sstream>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/multibody/filament/filament_common.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/force_density_field.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 0.02, "Desired duration of the simulation [s].");
DEFINE_double(time_step, 1e-5,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 2e8, "Young's modulus of the filament [Pa].");
DEFINE_double(G, 1e8, "Shear modulus of the filament [Pa].");
DEFINE_double(rho, 1e3, "Mass density of the filament [kg/m³].");
DEFINE_double(mu, 0.01, "Friction coefficient of the filament [unitless].");
DEFINE_double(force, 3e9, "Pulling force on the filament.");
DEFINE_string(
    knot_configuration, "n4",
    "The knot configuration. Options are: 'n1', 'n2', 'n3', and 'n4'.");
DEFINE_string(contact_approximation, "lagged",
              "Type of convex contact approximation. See "
              "multibody::DiscreteContactApproximation for details. Options "
              "are: 'sap', 'lagged', and 'similar'.");

namespace drake {
namespace examples {
namespace filament {
namespace {

using drake::geometry::Filament;
using drake::geometry::SceneGraph;
using drake::geometry::Sphere;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::ForceDensityField;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::der::DerModel;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using drake::systems::Simulator;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using math::RigidTransformd;
using math::RotationMatrixd;

class ConstantPullingForce final : public ForceDensityField<double> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ConstantPullingForce);

  ConstantPullingForce(DeformableBodyId id, double force)
      : id_(id), force_(force) {}

 private:
  void DoDeclareCacheEntries(MultibodyPlant<double>* plant) override {
    deformable_model_ = &plant->deformable_model();
  }

  Vector3d DoEvaluateAt(const Context<double>& context,
                        const Vector3d& p_WQ) const override {
    DRAKE_DEMAND(deformable_model_ != nullptr);
    const Eigen::Matrix3Xd node_positions =
        deformable_model_->GetPositions(context, id_);
    const Vector3d start = node_positions.col(0);
    const Vector3d end = node_positions.template rightCols<1>();
    const Vector3d v = (end - start).normalized();
    const double kEpsilon = 1e-3;
    if ((p_WQ - start).norm() <= kEpsilon) {
      return -v * force_;
    } else if ((p_WQ - end).norm() <= kEpsilon) {
      return v * force_;
    } else {
      return Vector3d::Zero();
    }
  };

  std::unique_ptr<ForceDensityFieldBase<double>> DoClone() const override {
    return std::make_unique<ConstantPullingForce>(*this);
  }

  DeformableBodyId id_{};
  double force_{};
  const DeformableModel<double>* deformable_model_{};
};

DeformableBodyId RegisterFilament(DeformableModel<double>* deformable_model) {
  Filament filament = LoadFilament(FLAGS_knot_configuration);

  auto geometry_instance = std::make_unique<geometry::GeometryInstance>(
      RigidTransformd::Identity(), filament, "filament_with_knot");

  /* Add a minimal illustration property for visualization. */
  geometry::IllustrationProperties illus_props;
  illus_props.AddProperty("phong", "diffuse", Vector4d(0.7, 0.5, 0.4, 1.0));
  geometry_instance->set_illustration_properties(std::move(illus_props));

  /* Add a minimal proximity property for collision detection. */
  geometry::ProximityProperties proximity_props;
  const CoulombFriction<double> surface_friction(FLAGS_mu, FLAGS_mu);
  AddContactMaterial({}, {}, surface_friction, &proximity_props);
  geometry_instance->set_proximity_properties(proximity_props);

  /* Set the material properties. Notice G = E / 2(1+𝜈). */
  DeformableBodyConfig<double> config;
  config.set_youngs_modulus(FLAGS_E);
  config.set_poissons_ratio(0.5 * FLAGS_E / FLAGS_G - 1);
  config.set_mass_density(FLAGS_rho);
  config.set_mass_damping_coefficient(1.0);

  const double unused_resolution_hint = 9999;
  DeformableBodyId body_id = deformable_model->RegisterDeformableBody(
      std::move(geometry_instance), config, unused_resolution_hint);
  deformable_model->GetMutableBody(body_id).set_default_pose(
      math::RigidTransformd(Vector3d(0, 0, 0.15)));

  return body_id;
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  plant.mutable_gravity_field().set_gravity_vector(Vector3d::Zero());

  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();
  DeformableBodyId id = RegisterFilament(&deformable_model);
  deformable_model.AddExternalForce(
      std::make_unique<ConstantPullingForce>(id, FLAGS_force));
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  params.publish_period = FLAGS_time_step;
  geometry::DrakeVisualizer<double>::AddToBuilder(&builder, scene_graph,
                                                  nullptr, params);
  auto diagram = builder.Build();

  Simulator<double> simulator(*diagram);
  simulator.Initialize();

  simulator.AdvanceTo(FLAGS_simulation_time);
  return 0;
}

}  // namespace
}  // namespace filament
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase self-contact resolution during knot "
      "tying. Refer to README for instructions on meldis as well as optional "
      "flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::filament::do_main();
}
