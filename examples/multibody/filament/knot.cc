#include <fstream>
#include <sstream>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/text_logging.h"
#include "drake/examples/multibody/filament/filament_common.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/force_density_field.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(time_step, 1e-5, "Discrete time step for the system [s].");
DEFINE_double(E, 2.4e8, "Young's modulus of the filament [Pa].");
DEFINE_double(G, 1.0e8, "Shear modulus of the filament [Pa].");
DEFINE_double(rho, 1e3, "Mass density of the filament [kg/m³].");
DEFINE_double(diameter, 0.003, "Diameter of the filament [m].");
DEFINE_double(mu, 0.0, "Friction coefficient of the filament [unitless].");
DEFINE_double(P_gain, 1e6, "Proportional gain for position controller.");
DEFINE_double(pull_speed, 80, "Pulling speed at the two ends [m/s].");
DEFINE_double(pull_distance, 0.52, "Pulling distance [m].");
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

using drake::geometry::Box;
using drake::geometry::Filament;
using drake::geometry::SceneGraph;
using drake::geometry::Sphere;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::ForceDensityField;
using drake::multibody::JointActuator;
using drake::multibody::JointActuatorIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::PdControllerGains;
using drake::multibody::PrismaticJoint;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::der::DerModel;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using drake::systems::LeafSystem;
using drake::systems::Simulator;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using math::RigidTransformd;
using math::RotationMatrixd;

template <typename T>
class DeriredStateSource : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeriredStateSource);

  DeriredStateSource(const T& speed, int num_actuators)
      : speed_(speed), num_actuators_(num_actuators) {
    this->DeclareVectorOutputPort(systems::kUseDefaultName, num_actuators * 2,
                                  &DeriredStateSource<T>::CalcOutput);
  }

 private:
  void CalcOutput(const Context<T>& context,
                  systems::BasicVector<T>* output) const {
    Eigen::VectorBlock<VectorX<T>> source = output->get_mutable_value();
    const double t = context.get_time();
    drake::log()->info("time {:.5f}", t);
    source.head(num_actuators_).array() = speed_ * t;
    source.tail(num_actuators_).array() = speed_;
  }

  T speed_;
  int num_actuators_;
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();

  /* Add filament. */
  Filament filament = LoadFilament(FLAGS_knot_configuration, FLAGS_diameter);
  auto geometry_instance = std::make_unique<geometry::GeometryInstance>(
      RigidTransformd::Identity(), filament, "filament");

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
  DeformableBodyId body_id = deformable_model.RegisterDeformableBody(
      std::move(geometry_instance), config, unused_resolution_hint);

  const VectorXd node_positions =
      deformable_model.GetReferencePositions(body_id);
  const Vector3d end_pos1 = node_positions.template head<3>();
  const Vector3d end_pos2 = node_positions.template tail<3>();

  /* Add link 1. */
  const RigidBody<double>& link1 =
      plant.AddRigidBody("link1", SpatialInertia<double>::MakeUnitary());
  const int kZaxis = 2;
  const RigidTransformd X_WL1(
      RotationMatrixd::MakeFromOneVector(end_pos1 - end_pos2, kZaxis),
      end_pos1);
  const PrismaticJoint<double>& joint1 =
      plant.AddJoint<PrismaticJoint>("joint1", plant.world_body(), X_WL1, link1,
                                     RigidTransformd(), Vector3d(0, 0, 1));
  JointActuatorIndex joint1_index =
      plant.AddJointActuator("actuator1", joint1).index();
  plant.get_mutable_joint_actuator(joint1_index)
      .set_controller_gains(PdControllerGains{FLAGS_P_gain, 0.0});
  plant.RegisterVisualGeometry(
      link1, RigidTransformd(),
      Box(FLAGS_diameter * 5, FLAGS_diameter * 5, FLAGS_diameter), "link1",
      Vector4d(1, 0, 0, 1));

  /* Add link 2. */
  const RigidBody<double>& link2 =
      plant.AddRigidBody("link2", SpatialInertia<double>::MakeUnitary());
  const RigidTransformd X_WL2(
      RotationMatrixd::MakeFromOneVector(end_pos2 - end_pos1, kZaxis),
      end_pos2);
  const PrismaticJoint<double>& joint2 =
      plant.AddJoint<PrismaticJoint>("joint2", plant.world_body(), X_WL2, link2,
                                     RigidTransformd(), Vector3d(0, 0, 1));
  JointActuatorIndex joint2_index =
      plant.AddJointActuator("actuator2", joint2).index();
  plant.get_mutable_joint_actuator(joint2_index)
      .set_controller_gains(PdControllerGains{FLAGS_P_gain, 0.0});
  plant.RegisterVisualGeometry(
      link2, RigidTransformd(),
      Box(FLAGS_diameter * 5, FLAGS_diameter * 5, FLAGS_diameter * 0.5),
      "link2", Vector4d(1, 0, 0, 1));

  /* Attach the filament to the links. */
  deformable_model.AddFixedConstraint(body_id, link1, X_WL1.inverse(),
                                      Sphere(1e-6), RigidTransformd());
  deformable_model.AddFixedConstraint(body_id, link2, X_WL2.inverse(),
                                      Sphere(1e-6), RigidTransformd());

  plant.mutable_gravity_field().set_gravity_vector(Vector3d::Zero());
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  params.publish_period = FLAGS_time_step;
  geometry::DrakeVisualizer<double>::AddToBuilder(&builder, scene_graph,
                                                  nullptr, params);

  /* Make the actuators track a ramp position. */
  DeriredStateSource<double>* source =
      builder.AddSystem<DeriredStateSource<double>>(FLAGS_pull_speed / 2, 2);
  builder.Connect(
      source->get_output_port(),
      plant.get_desired_state_input_port(multibody::ModelInstanceIndex(1)));
  auto diagram = builder.Build();

  Simulator<double> simulator(*diagram);
  simulator.Initialize();

  simulator.AdvanceTo(FLAGS_pull_distance / FLAGS_pull_speed);
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
