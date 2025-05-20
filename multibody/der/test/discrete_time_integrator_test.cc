#include "drake/multibody/der/discrete_time_integrator.h"

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using Eigen::Vector3d;
using Eigen::VectorXd;

/* A dummy implementation of the DiscreteTimeIntegrator class. Used in the unit
 test in this file only. */
class DummyScheme final : public DiscreteTimeIntegrator<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DummyScheme);

  explicit DummyScheme(double dt) : DiscreteTimeIntegrator<double>(dt) {}

  ~DummyScheme() = default;

 private:
  std::unique_ptr<DiscreteTimeIntegrator<double>> DoClone() const final {
    return std::make_unique<DummyScheme>(this->dt());
  }

  std::array<double, 3> DoGetWeights() const final { return {1, 2, 3}; }

  // Dummy implementation to return the position.
  const Eigen::VectorXd& DoGetUnknowns(
      const DerState<double>& state) const final {
    return state.get_position();
  }

  // Dummy implementation to set the position to `q + dz`.
  void DoAdjustStateFromChangeInUnknowns(
      const Eigen::Ref<const Eigen::VectorXd>& dz,
      DerState<double>* state) const final {
    state->AdjustPositionWithinStep(state->get_position() + dz);
  }

  // Dummy implementation to set the position of the state to the entry-wise
  // product of previous state's position and the unknown variable.
  void DoAdvanceOneTimeStep(const DerState<double>& prev_state,
                            const Eigen::Ref<const Eigen::VectorXd>& z,
                            DerState<double>* state) const final {
    state->AdvancePositionToNextStep(prev_state.get_position().cwiseProduct(z));
  }
};

class DiscreteTimeIntegratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    bool has_closed_ends = false;
    std::vector<Vector3d> node_positions = {
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(2, 1, 0)};
    std::vector<double> edge_angles = {M_PI / 6, M_PI / 6};
    Vector3d d1_0(0, 1, 0);
    der_state_system_ = std::make_unique<DerStateSystem<double>>(
        has_closed_ends, node_positions, edge_angles, d1_0);
  }

  std::unique_ptr<DerStateSystem<double>> der_state_system_;
  static constexpr double kDt = 0.01;
  DummyScheme scheme_{kDt};
};  // namespace

TEST_F(DiscreteTimeIntegratorTest, Dt) {
  EXPECT_EQ(scheme_.dt(), kDt);
}

TEST_F(DiscreteTimeIntegratorTest, Weights) {
  EXPECT_EQ(scheme_.GetWeights(), (std::array<double, 3>{1, 2, 3}));
}

TEST_F(DiscreteTimeIntegratorTest, GetUnknowns) {
  const DerState<double> state(der_state_system_.get());
  EXPECT_EQ(scheme_.GetUnknowns(state), state.get_position());
}

TEST_F(DiscreteTimeIntegratorTest, AdvanceOneTimeStep) {
  const DerState<double> state0(der_state_system_.get());
  DerState<double> state(der_state_system_.get());
  const VectorXd z = VectorXd::LinSpaced(state.num_dofs(), 0.0, 1.0);
  scheme_.AdvanceOneTimeStep(state0, z, &state);
  EXPECT_EQ(state.get_position(), state0.get_position().cwiseProduct(z));
}

TEST_F(DiscreteTimeIntegratorTest, AdjustStateFromChangeInUnknowns) {
  DerState<double> state(der_state_system_.get());
  const VectorXd dz = VectorXd::LinSpaced(state.num_dofs(), 0.0, 1.0);
  const VectorXd expected = state.get_position() + dz;
  scheme_.AdjustStateFromChangeInUnknowns(dz, &state);
  EXPECT_EQ(state.get_position(), expected);
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
