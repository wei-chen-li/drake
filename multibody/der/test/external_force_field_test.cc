#include "drake/multibody/der/external_force_field.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using Eigen::Vector3d;
using Eigen::VectorXd;
using test::LimitMalloc;

template <typename T>
class DummySystem final : public systems::LeafSystem<T> {
 public:
  DummySystem() {}
};

class ExternalForceFieldTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    const bool has_closed_ends = GetParam();

    const auto [E, G, rho] = std::make_tuple(3e9, 0.8e9, 910);
    prop_ = DerStructuralProperty<double>::FromCircularCrossSection(1e-3, E, G,
                                                                    rho);

    std::vector<Vector3d> node_positions;
    std::vector<double> edge_angles;
    const double l = 0.01;
    if (!has_closed_ends) {
      node_positions = {Vector3d(0, 0, 0), Vector3d(l, 0, l * 0.5),
                        Vector3d(l, l, l * 1.5), Vector3d(0, l, l * 3.0)};
      edge_angles = {0, 0.1, 0.2};
      undeformed_ = DerUndeformedState<double>::ZeroCurvatureAndTwist(
          has_closed_ends, {l * 1.2, l * 0.8, l * 1.0});
    } else {
      node_positions = {Vector3d(0, 0, 0), Vector3d(l, 0, l),
                        Vector3d(l, l, l * 1.5), Vector3d(0, l, l)};
      edge_angles = {0, 0.1, 0.2, 0.1};
      undeformed_ = DerUndeformedState<double>::ZeroCurvatureAndTwist(
          has_closed_ends, {l * 1.2, l * 0.8, l * 1.0, l * 0.6});
    }
    der_state_system_ = std::make_unique<DerStateSystem<double>>(
        has_closed_ends, node_positions, edge_angles, std::nullopt);

    state_ = std::make_unique<DerState<double>>(der_state_system_.get());
  }

  std::optional<DerStructuralProperty<double>> prop_;
  std::optional<DerUndeformedState<double>> undeformed_;
  std::unique_ptr<DerStateSystem<double>> der_state_system_;
  std::unique_ptr<DerState<double>> state_;
};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds, ExternalForceFieldTest,
                         ::testing::Values(false, true));

TEST_P(ExternalForceFieldTest, MassMatrix) {
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> mass =
      ComputeMassMatrix(*prop_, *undeformed_);
  EXPECT_EQ(mass.rows(), undeformed_->num_dofs());
  EXPECT_EQ(mass.cols(), undeformed_->num_dofs());

  const double l = 0.01;
  const double rhoAl = prop_->rhoA() * l;
  const double rhoJl = prop_->rhoJ() * l;
  std::vector<double> nodes_mass;
  std::vector<double> edges_mass;
  if (!state_->has_closed_ends()) {
    nodes_mass = {0.6 * rhoAl, 1.0 * rhoAl, 0.9 * rhoAl, 0.5 * rhoAl};
    edges_mass = {1.2 * rhoJl, 0.8 * rhoJl, 1.0 * rhoJl};
  } else {
    nodes_mass = {0.9 * rhoAl, 1.0 * rhoAl, 0.9 * rhoAl, 0.8 * rhoAl};
    edges_mass = {1.2 * rhoJl, 0.8 * rhoJl, 1.0 * rhoJl, 0.6 * rhoJl};
  }
  for (int i = 0; i < undeformed_->num_nodes(); ++i) {
    EXPECT_TRUE(CompareMatrices(mass.diagonal().template segment<3>(4 * i),
                                Eigen::Vector3d::Constant(nodes_mass[i]),
                                1e-16));
  }
  for (int i = 0; i < undeformed_->num_edges(); ++i) {
    EXPECT_NEAR(mass.diagonal()[4 * i + 3], edges_mass[i], 1e-16);
  }
}

TEST_P(ExternalForceFieldTest, GravitationalForce) {
  const double g = 9.81;
  const double rho = 910;
  GravityForceField<double> gravity_force_field(Vector3d(0, 0, -g), rho);

  auto dummy_context = DummySystem<double>().CreateDefaultContext();
  ExternalForceField<double> external_force_field(dummy_context.get(),
                                                  {&gravity_force_field});

  VectorXd force = external_force_field(*prop_, *undeformed_, *state_).eval();

  Eigen::DiagonalMatrix<double, Eigen::Dynamic> mass =
      ComputeMassMatrix(*prop_, *undeformed_);
  VectorXd acceleration = mass.inverse() * force;

  constexpr double kTol = 1e-14;
  for (int i = 0; i < state_->num_nodes(); ++i) {
    EXPECT_TRUE(CompareMatrices(acceleration.template segment<3>(4 * i),
                                Vector3d(0, 0, -g), kTol));
  }
  for (int i = 0; i < state_->num_edges(); ++i) {
    EXPECT_NEAR(acceleration[4 * i + 3], 0.0, kTol);
  }

  {
    LimitMalloc guard;
    force.setZero();
    force += external_force_field(*prop_, *undeformed_, *state_);
  }
  EXPECT_TRUE(CompareMatrices(force, mass * acceleration, kTol));
  {
    LimitMalloc guard;
    force.setZero();
    force -= external_force_field(*prop_, *undeformed_, *state_);
  }
  EXPECT_TRUE(CompareMatrices(force, mass * -acceleration, kTol));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
