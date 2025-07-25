#include "drake/multibody/der/der_solver.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/multibody/der/velocity_newmark_scheme.h"
#include "drake/multibody/tree/force_density_field.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

using Eigen::Vector3d;
using Eigen::VectorXd;

/* Friend class to access private members of DerSolver. */
class DerSolverTester {
 public:
  DerSolverTester() = delete;

  template <typename T>
  static const DerModel<T>* model(const DerSolver<T>& solver) {
    return solver.model_;
  }

  template <typename T>
  static const DiscreteTimeIntegrator<T>& integrator(
      const DerSolver<T>& solver) {
    return *solver.integrator_;
  }

  template <typename T>
  static void set_max_iterations(DerSolver<T>* solver, int max_iters) {
    solver->max_newton_iters_ = max_iters;
  }

  template <typename T>
  static int max_iterations(const DerSolver<T>& solver) {
    return solver.max_newton_iters_;
  }

  template <typename T>
  static void scaramble_owned_state(DerSolver<T>* solver) {
    DerState<T>& state = *solver->state_;
    const int num_dofs = state.num_dofs();
    state.AdvancePositionToNextStep(VectorXd::LinSpaced(num_dofs, 0.0, 1.0));
    state.AdvancePositionToNextStep(VectorXd::LinSpaced(num_dofs, 1.0, 2.0));
    state.AdjustPositionWithinStep(VectorXd::LinSpaced(num_dofs, 2.0, 3.0));
    state.SetVelocity(VectorXd::Constant(num_dofs, 1.2));
    state.SetAcceleration(VectorXd::Constant(num_dofs, 3.4));
  }
};

namespace {

class DerSolverTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    const bool has_closed_ends = GetParam();

    DerModel<double>::Builder builder;
    if (has_closed_ends) {
      const double l = 0.01;
      Vector3d d1_0(0, 1, 0);
      builder.AddFirstEdge(Vector3d(0, 0, 0), 0.0, Vector3d(l, 0, l), d1_0);
      builder.AddEdge(0.1, Vector3d(l, l, l * 1.5));
      builder.AddEdge(0.2, Vector3d(0, l, l));
      builder.AddEdge(0.1, Vector3d(0, 0, 0));
    } else {
      const double l = 0.01;
      Vector3d d1_0(0, 1, 0);
      builder.AddFirstEdge(Vector3d(0, 0, 0), 0.0, Vector3d(l, 0, l), d1_0);
      builder.AddEdge(0.1, Vector3d(l, l, l * 1.5));
      builder.AddEdge(0.2, Vector3d(0, l, l * 3.0));
    }

    builder.SetZeroUndeformedCurvatureAndTwist();
    const auto [E, G, rho] = std::make_tuple(3e9, 0.8e9, 910);
    builder.SetMaterialProperties(E, G, rho);
    const double radius = 1e-3;
    builder.SetCircularCrossSection(radius);
    builder.SetDampingCoefficients(0.0, 0.0);
    der_model_ = builder.Build();

    const double dt = 0.01;
    const double gamma = 0.5;
    const double beta = 0.25;
    integrator_ =
        std::make_unique<VelocityNewmarkScheme<double>>(dt, gamma, beta);

    solver_ = std::make_unique<DerSolver<double>>(der_model_.get(),
                                                  integrator_.get());

    const double g = 9.81;
    force_density_field_ =
        std::make_unique<GravityForceField<double>>(Vector3d(0, 0, -g), rho);
  }

  std::unique_ptr<DerModel<double>> der_model_;
  std::unique_ptr<DiscreteTimeIntegrator<double>> integrator_;
  std::unique_ptr<DerSolver<double>> solver_;
  std::unique_ptr<ForceDensityField<double>> force_density_field_;
};

INSTANTIATE_TEST_SUITE_P(DerModelHasClosedEnds, DerSolverTest,
                         ::testing::Values(false, true));

template <typename T>
class DummySystem final : public systems::LeafSystem<T> {
 public:
  DummySystem() {}
};

/* Tests that AdvanceOneTimeStep() throws an error message if the Newton-Raphson
 solver doesn't converge within the max number of iterations. */
TEST_P(DerSolverTest, Nonconvergence) {
  std::unique_ptr<DerState<double>> state = der_model_->CreateDerState();
  auto dummy_context = DummySystem<double>().CreateDefaultContext();
  ExternalForceField external_force_field(dummy_context.get(),
                                          {force_density_field_.get()});

  DerSolverTester::set_max_iterations(solver_.get(), 0);
  DRAKE_EXPECT_THROWS_MESSAGE(
      solver_->AdvanceOneTimeStep(*state, external_force_field),
      ".*failed to converge.*");
}

TEST_P(DerSolverTest, Tolerance) {
  /* Default values. */
  EXPECT_EQ(solver_->relative_tolerance(), 1e-4);
  EXPECT_EQ(solver_->absolute_tolerance(), 1e-6);
  /* Test Setters. */
  const double rel_tolerance = 1e-8;
  const double abs_tolerance = 1e-10;
  solver_->set_relative_tolerance(rel_tolerance);
  solver_->set_absolute_tolerance(abs_tolerance);
  EXPECT_EQ(solver_->relative_tolerance(), rel_tolerance);
  EXPECT_EQ(solver_->absolute_tolerance(), abs_tolerance);
}

TEST_P(DerSolverTest, Clone) {
  /* Scramble the solver owned state. */
  DerSolverTester::scaramble_owned_state(solver_.get());
  /* Scramble the solver tolerances. */
  solver_->set_relative_tolerance(1e-8);
  solver_->set_absolute_tolerance(1e-10);
  /* Scramble the solver max iterations. */
  DerSolverTester::set_max_iterations(solver_.get(), 1000);

  std::unique_ptr<DerSolver<double>> cloned = solver_->Clone();
  /* Check model and integrator are the same. */
  EXPECT_EQ(DerSolverTester::model(*cloned), DerSolverTester::model(*solver_));
  EXPECT_EQ(DerSolverTester::integrator(*cloned).dt(),
            DerSolverTester::integrator(*solver_).dt());
  /* Check owned states have same values. */
  EXPECT_EQ(cloned->get_state().get_position(),
            solver_->get_state().get_position());
  EXPECT_EQ(cloned->get_state().get_velocity(),
            solver_->get_state().get_velocity());
  EXPECT_EQ(cloned->get_state().get_acceleration(),
            solver_->get_state().get_acceleration());
  EXPECT_EQ(cloned->get_state().get_reference_frame_d1(),
            solver_->get_state().get_reference_frame_d1());
  EXPECT_EQ(cloned->get_state().get_twist(), solver_->get_state().get_twist());
  /* Check tolerance and max iterations are the same. */
  EXPECT_EQ(cloned->relative_tolerance(), solver_->relative_tolerance());
  EXPECT_EQ(cloned->absolute_tolerance(), solver_->absolute_tolerance());
  EXPECT_EQ(DerSolverTester::max_iterations(*cloned),
            DerSolverTester::max_iterations(*solver_));
}

/* The following tests build a DerModel representing a cantilever beam and use
 DerSolver to simulate its static and dynamic response. Simulation results are
 then compared with analytical solutions. The tests are parameterized on mode
 number, where mode number = 0 indicates the static case. */
class DerSolverCantileverBeamTest : public ::testing::TestWithParam<int> {
 protected:
  std::unique_ptr<DerModel<double>> CreateBeam(bool clamp_end = true) {
    DerModel<double>::Builder builder;
    const Vector3d d1_0 = Vector3d(0, 1, 0);
    const double dx = kLength / (kNumNodes - 1);
    builder.AddFirstEdge(Vector3d(0, 0, 0), 0, Vector3d(dx, 0, 0), d1_0);
    for (int i = 2; i < kNumNodes; ++i) {
      builder.AddEdge(0, Vector3d(dx * i, 0, 0));
    }
    builder.SetZeroUndeformedCurvatureAndTwist();
    builder.SetMaterialProperties(kE, kG, kRho);
    builder.SetRectangularCrossSection(kWidth, kHeight);
    builder.SetDampingCoefficients(0.0, 0.0);
    std::unique_ptr<DerModel<double>> model = builder.Build();

    model->FixPosition(DerNodeIndex(0));
    if (clamp_end) {
      model->FixPosition(DerEdgeIndex(0));
      model->FixPosition(DerNodeIndex(1));
    }
    return model;
  }

  void TestStaticBending();
  void TestStaticStretching();

  const double kE = 1e9;               // Young's modulus = 1 GPa.
  const double kG = 0.3e9;             // Shear modulus = 0.3 GPa.
  const double kRho = 50;              // Mass density = 50 kg/m³.
  const double kWidth = 0.02;          // Rod width = 2 cm.
  const double kHeight = 0.015;        // Rod height = 1.5 cm.
  const double kLength = 1.0;          // Rod length = 1 m.
  const int kNumNodes = 301;           // Rod is discretized into 300 edges.
  const double kA = kWidth * kHeight;  // Area of cross-section.
  const double kI = kWidth * pow(kHeight, 3) / 12;  // Inertia of cross-section.
};

INSTANTIATE_TEST_SUITE_P(ModeNumber, DerSolverCantileverBeamTest,
                         ::testing::Values(0, 1, 2, 3));  // Mode numbers.

/* Simulate a cantilever beam bending under self weight. Compare the bending
 shape with the analytical solution from Euler-Bernoulli beam theory. */
void DerSolverCantileverBeamTest::TestStaticBending() {
  /* Euler-Bernoulli beam under uniform load:
       z(x) = 𝑓/(24EI) ⋅ x² (6L² - 4Lx + x²),
   where 𝑓 is the force per length. */
  const double g = 9.81;
  const double force_per_length = kRho * kA * -g;
  auto x = VectorXd::LinSpaced(kNumNodes, 0.0, kLength).array();
  const VectorXd static_shape =
      force_per_length / (24 * kE * kI) * (x * x) *
      (6 * kLength * kLength - 4 * kLength * x + x * x);

  /* Create the beam DerModel. */
  std::unique_ptr<DerModel<double>> model = CreateBeam();

  /* Set the integrator to use midpoint rule q = q₀ + δt/2 * (v₀ + v). */
  const double dt = 0.01;
  VelocityNewmarkScheme<double> integrator(dt, 1.0, 0.5);

  /* Under gravatational force. */
  auto dummy_context = DummySystem<double>().CreateDefaultContext();
  GravityForceField<double> gravity_force_field(Vector3d(0, 0, -g), kRho);
  ExternalForceField external_force_field(dummy_context.get(),
                                          {&gravity_force_field});

  /* Solve the cantilever beam long enough to reach static equilibrium. */
  std::unique_ptr<DerState<double>> state = model->CreateDerState();
  DerSolver<double> solver(model.get(), &integrator);
  double t = 0.0;
  const double t_end = 1.0;
  while (t < t_end) {
    solver.AdvanceOneTimeStep(*state, external_force_field);
    state->CopyFrom(solver.get_state());
    t += dt;
  }

  /* Simulated bending shape should match the analytical solution. */
  VectorXd sim_shape(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i)
    sim_shape[i] = state->get_position()[4 * i + 2];
  EXPECT_TRUE(CompareMatrices(
      sim_shape, static_shape,
      7e-3 * static_shape.cwiseAbs().maxCoeff()));  // At most 0.7% difference.
}

/* Simulate a vibrating cantilever beam. Compare the time period and bending
 shape with the analytical values from Euler-Bernoulli beam theory. */
TEST_P(DerSolverCantileverBeamTest, Bending) {
  if (GetParam() == 0) return TestStaticBending();

  /* Euler-Bernoulli beam dynamic response:
   z(x,t) = ∑ₙ Aₙ ((sinh(λₙ) + sin(λₙ)) (cosh(λₙx/L) - cos(λₙx/L)) -
                  (cosh(λₙ) + cos(λₙ)) (sinh(λₙx/L) - sin(λₙx/L))) exp(jωₙt),
   where ωₙ = λₙ² √(EI/ρAL⁴).  */
  const double eigenvalues[3] = {1.875, 4.694, 7.855};
  const int mode_number = GetParam();                  // n.
  const double lambda = eigenvalues[mode_number - 1];  // λₙ.
  auto lambda_xbar = lambda * VectorXd::LinSpaced(kNumNodes, 0.0, 1.0).array();
  VectorXd mode_shape =
      (sinh(lambda) + sin(lambda)) * (cosh(lambda_xbar) - cos(lambda_xbar)) -
      (cosh(lambda) + cos(lambda)) * (sinh(lambda_xbar) - sin(lambda_xbar));
  mode_shape /= mode_shape[kNumNodes - 1];

  const double omega =
      pow(lambda, 2) * sqrt(kE * kI / (kRho * kA * pow(kLength, 4)));  // ωₙ.
  const double T = (2 * M_PI) / omega;                                 // Tₙ.
  const double dt = T / 100;
  const double t_end = T * 3.25;
  const double v_tip_init = 1e-4;  // Initial tip velocity = 0.1 mm/s.

  /* Create the beam DerModel. */
  std::unique_ptr<DerModel<double>> model = CreateBeam();

  /* Set the integrator to use midpoint rule q = q₀ + δt/2 * (v₀ + v). */
  VelocityNewmarkScheme<double> integrator(dt, 1.0, 0.5);

  /* Under no external forces. */
  auto dummy_context = DummySystem<double>().CreateDefaultContext();
  ExternalForceField external_force_field(dummy_context.get(), {});

  /* Create a DerState and set the initial velocity. */
  std::unique_ptr<DerState<double>> state = model->CreateDerState();
  VectorXd velocity = VectorXd::Zero(state->num_dofs());
  for (int i = 0; i < kNumNodes; ++i)
    velocity[4 * i + 2] = mode_shape[i] * v_tip_init;
  state->SetVelocity(velocity);

  /* Simulate the vibrating cantilever beam. */
  DerSolver<double> solver(model.get(), &integrator);
  double t = 0.0;
  std::vector<double> t_crossing = {t};
  double prev_z_tip = 0.0;
  while (t < t_end) {
    solver.AdvanceOneTimeStep(*state, external_force_field);
    state->CopyFrom(solver.get_state());
    t += dt;
    // Check boundary conditions are satisfied.
    ASSERT_TRUE(state->get_velocity().head(7).isZero());
    ASSERT_TRUE(state->get_acceleration().head(7).isZero());

    const double z_tip = state->get_position()[state->num_dofs() - 1];
    if (prev_z_tip * z_tip < 0) t_crossing.push_back(t);
    prev_z_tip = z_tip;
  }

  /* The simulated time period should match the analytical value. */
  const VectorXd t_stamps =
      Eigen::Map<VectorXd>(t_crossing.data(), t_crossing.size());
  const double sim_T = 2 * (t_stamps.tail(t_stamps.size() - 1) -
                            t_stamps.head(t_stamps.size() - 1))
                               .mean();
  EXPECT_NEAR(sim_T, T, dt / 2);

  /* The simulated beam shape should match the analytical mode shape. */
  VectorXd sim_shape(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i)
    sim_shape[i] = state->get_position()[4 * i + 2];
  const double kTolerance[3] = {2e-3, 4e-3, 9e-3};  // At most 0.9% difference.
  EXPECT_TRUE(CompareMatrices(sim_shape / sim_shape[kNumNodes - 1], mode_shape,
                              kTolerance[mode_number - 1]));
}

/* Simulate a beam stretching under self weight. Compare the stretching
 displacement with the analytical solution. */
void DerSolverCantileverBeamTest::TestStaticStretching() {
  /* Stretching displecement u of a beam under self weight satisfies:
       ∂/∂x (E ∂u(x)/∂x) - ρg = 0,
   The solution is:
       u(x) = ρg/2E (2Lx - x²). */
  const double g = 9.81;
  auto x = VectorXd::LinSpaced(kNumNodes, 0.0, kLength).array();
  const VectorXd static_disp = kRho * g / (2 * kE) * (2 * kLength * x - x * x);

  /* Create the beam DerModel. */
  std::unique_ptr<DerModel<double>> model = CreateBeam(/* clamp_end = */ false);

  /* Set the integrator to use midpoint rule q = q₀ + δt/2 * (v₀ + v). */
  const double dt = 0.01;
  VelocityNewmarkScheme<double> integrator(dt, 1.0, 0.5);

  /* Under gravatational force. */
  auto dummy_context = DummySystem<double>().CreateDefaultContext();
  GravityForceField<double> gravity_force_field(Vector3d(g, 0, 0), kRho);
  ExternalForceField external_force_field(dummy_context.get(),
                                          {&gravity_force_field});

  /* Solve the beam long enough to reach static equilibrium. */
  std::unique_ptr<DerState<double>> state = model->CreateDerState();
  DerSolver<double> solver(model.get(), &integrator);
  double t = 0.0;
  const double t_end = 1.0;
  while (t < t_end) {
    solver.AdvanceOneTimeStep(*state, external_force_field);
    state->CopyFrom(solver.get_state());
    t += dt;
  }

  /* Simulated dispalcement should match the analytical solution. */
  VectorXd sim_disp(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i)
    sim_disp[i] = state->get_position()[4 * i] - x[i];
  EXPECT_TRUE(CompareMatrices(
      sim_disp, static_disp,
      8e-4 * static_disp.cwiseAbs().maxCoeff()));  // At most 0.08% difference.
}

/* Simulate a periodic stretching beam with top end fixed and bottom end free.
 Compare the time period and shape with the analytical solution. */
TEST_P(DerSolverCantileverBeamTest, Stretching) {
  if (GetParam() == 0) return TestStaticStretching();

  /* The dynamic response of the stretching displecement u of a beam satisfies:
       ρ ∂²u/∂t² = E ∂²u/∂x².
   The solution is:
       u(x,t) = ∑ₙ Aₙ sin(λₙx/L) exp(jωₙt),
   where λₙ = π(2n-1)/2 and ωₙ = λₙ √(E/ρ) */
  const int mode_number = GetParam();                      // n.
  const double lambda = M_PI * (2 * mode_number - 1) / 2;  // λₙ.
  const VectorXd x = VectorXd::LinSpaced(kNumNodes, 0.0, kLength);
  auto lambda_xbar = lambda * x.array() / kLength;
  VectorXd mode_disp = sin(lambda_xbar);
  mode_disp /= mode_disp[kNumNodes - 1];

  const double omega = lambda * sqrt(kE / kRho);  // ωₙ.
  const double T = (2 * M_PI) / omega;            // Tₙ.
  const double dt = T / 100;
  const double t_end = T * 3.25;
  const double v_tip_init = 1e-3;  // Initial tip velocity = 1 mm/s.

  /* Create the beam DerModel. */
  std::unique_ptr<DerModel<double>> model = CreateBeam(/* clamp_end = */ false);

  /* Set the integrator to use midpoint rule q = q₀ + δt/2 * (v₀ + v). */
  VelocityNewmarkScheme<double> integrator(dt, 1.0, 0.5);

  /* Under no external forces. */
  auto dummy_context = DummySystem<double>().CreateDefaultContext();
  ExternalForceField external_force_field(dummy_context.get(), {});

  /* Create a DerState and set the initial velocity. */
  std::unique_ptr<DerState<double>> state = model->CreateDerState();
  VectorXd velocity = VectorXd::Zero(state->num_dofs());
  for (int i = 0; i < kNumNodes; ++i)
    velocity[4 * i] = mode_disp[i] * v_tip_init;
  state->SetVelocity(velocity);

  /* Simulate the periodic stretching beam. */
  DerSolver<double> solver(model.get(), &integrator);
  double t = 0.0;
  std::vector<double> t_crossing = {t};
  double prev_disp_tip = 0.0;
  while (t < t_end) {
    solver.AdvanceOneTimeStep(*state, external_force_field);
    state->CopyFrom(solver.get_state());
    t += dt;
    // Check boundary conditions are satisfied.
    ASSERT_TRUE(state->get_velocity().head(3).isZero());
    ASSERT_TRUE(state->get_acceleration().head(3).isZero());

    const double disp_tip =
        state->get_position()[4 * (kNumNodes - 1)] - kLength;
    if (prev_disp_tip * disp_tip < 0) t_crossing.push_back(t);
    prev_disp_tip = disp_tip;
  }

  /* The simulated time period should match the analytical value. */
  const VectorXd t_stamps =
      Eigen::Map<VectorXd>(t_crossing.data(), t_crossing.size());
  const double sim_T = 2 * (t_stamps.tail(t_stamps.size() - 1) -
                            t_stamps.head(t_stamps.size() - 1))
                               .mean();
  EXPECT_NEAR(sim_T, T, dt / 2);

  /* The simulated beam displacement should match the analytical solution. */
  VectorXd sim_disp(kNumNodes);
  for (int i = 0; i < kNumNodes; ++i)
    sim_disp[i] = state->get_position()[4 * i] - x[i];
  EXPECT_TRUE(CompareMatrices(sim_disp / sim_disp[kNumNodes - 1], mode_disp,
                              1e-5));  // At most 0.01% difference.
}

/* Simulate a periodic twisting beam with top end fixed and bottom end free.
 Compare the time period and angle with the analytical solution. */
TEST_P(DerSolverCantileverBeamTest, Twisting) {
  if (GetParam() == 0) return;

  /* The dynamic response of the twisting angle θ of a beam satisfies:
       ρ ∂²θ/∂t² = G ∂²θ/∂x²,
   The solution is:
       θ(x,t) = ∑ₙ Aₙ sin(λₙx/L) exp(jωₙt),
   where λₙ = π(2n-1)/2 and ωₙ = λₙ √(G/ρ) */
  const int mode_number = GetParam();                      // n.
  const double lambda = M_PI * (2 * mode_number - 1) / 2;  // λₙ.
  const double dx = kLength / (kNumNodes - 1);
  const VectorXd x = VectorXd::LinSpaced(kNumNodes - 1, 0.0, kLength - dx);
  auto lambda_xbar = lambda * x.array() / kLength;
  VectorXd mode_angle = sin(lambda_xbar);
  mode_angle /= mode_angle[kNumNodes - 2];

  const double omega = lambda * sqrt(kG / kRho);  // ωₙ.
  const double T = (2 * M_PI) / omega;            // Tₙ.
  const double dt = T / 100;
  const double t_end = T * 3.25;
  const double v_tip_init = 1.0;  // Initial tip velocity = 1 rad/s.

  /* Create the beam DerModel. */
  std::unique_ptr<DerModel<double>> model = CreateBeam();

  /* Set the integrator to use midpoint rule q = q₀ + δt/2 * (v₀ + v). */
  VelocityNewmarkScheme<double> integrator(dt, 1.0, 0.5);

  /* Under no external forces. */
  auto dummy_context = DummySystem<double>().CreateDefaultContext();
  ExternalForceField external_force_field(dummy_context.get(), {});

  /* Create a DerState and set the initial velocity. */
  std::unique_ptr<DerState<double>> state = model->CreateDerState();
  VectorXd velocity = VectorXd::Zero(state->num_dofs());
  for (int i = 0; i < kNumNodes - 1; ++i)
    velocity[4 * i + 3] = mode_angle[i] * v_tip_init;
  state->SetVelocity(velocity);

  /* Simulate the periodic twisting beam. */
  DerSolver<double> solver(model.get(), &integrator);
  double t = 0.0;
  std::vector<double> t_crossing = {t};
  double prev_angle_tip = 0.0;
  while (t < t_end) {
    solver.AdvanceOneTimeStep(*state, external_force_field);
    state->CopyFrom(solver.get_state());
    t += dt;
    // Check boundary conditions are satisfied.
    ASSERT_TRUE(state->get_velocity().head(7).isZero());
    ASSERT_TRUE(state->get_acceleration().head(7).isZero());

    const double angle_tip = state->get_position()[4 * (kNumNodes - 2) + 3];
    if (prev_angle_tip * angle_tip < 0) t_crossing.push_back(t);
    prev_angle_tip = angle_tip;
  }

  /* The simulated time period should match the analytical value. */
  const VectorXd t_stamps =
      Eigen::Map<VectorXd>(t_crossing.data(), t_crossing.size());
  const double sim_T = 2 * (t_stamps.tail(t_stamps.size() - 1) -
                            t_stamps.head(t_stamps.size() - 1))
                               .mean();
  EXPECT_NEAR(sim_T, T, dt / 2);

  /* The simulated beam angle should match the analytical solution. */
  VectorXd sim_angle(kNumNodes - 1);
  for (int i = 0; i < kNumNodes - 1; ++i)
    sim_angle[i] = state->get_position()[4 * i + 3];
  const double kTolerance[3] = {0.001, 0.012,
                                0.041};  // At most 4.1% difference.
  EXPECT_TRUE(CompareMatrices(sim_angle / sim_angle[kNumNodes - 2], mode_angle,
                              kTolerance[mode_number - 1]));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
