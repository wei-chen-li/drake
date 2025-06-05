#include "drake/multibody/der/der_model.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/multibody/der/elastic_energy.h"
#include "drake/multibody/tree/force_density_field.h"

namespace drake {
namespace multibody {
namespace der {

/* Friend class for accessing DerModel private members. */
class DerModelTester {
 public:
  DerModelTester() = delete;

  template <typename T>
  static const internal::DerStructuralProperty<T>& get_der_structural_property(
      const DerModel<T>& der_model) {
    return der_model.der_structural_property_;
  }

  template <typename T>
  static const internal::DerUndeformedState<T>& get_der_undeformed_state(
      const DerModel<T>& der_model) {
    return der_model.der_undeformed_state_;
  }

  template <typename T>
  static const internal::DampingModel<T>& get_damping_model(
      const DerModel<T>& der_model) {
    return der_model.damping_model_;
  }
};

namespace {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using test::LimitMalloc;

enum RodConfigurationTests { kOpenEnds, kClosedEnds };
enum UndeformedStateTests { kZeroCurvatureTwist, kSameAsInitialState };
enum CrossSectionTests { kCircular, kRectangular, kElliptical };
enum DampingCoefficientTests { kZeroDamping, kNonZeroDamping };

class DerModelBuilderTest
    : public ::testing::TestWithParam<
          std::tuple<RodConfigurationTests, UndeformedStateTests,
                     CrossSectionTests, DampingCoefficientTests>> {
 protected:
  void Test() {
    builder_ = std::make_unique<DerModel<double>::Builder>();
    AddRodConfiguration();
    AddUndeformedState();
    AddCrossSection();
    AddDampingCoefficient();
    model_ = builder_->Build();
    CheckRodConfiguration();
    CheckUndeformedState();
    CheckCrossSection();
    CheckDampingCoefficient();
  }

  void AddRodConfiguration() {
    RodConfigurationTests opt = std::get<0>(GetParam());
    if (opt == kOpenEnds) {
      const double l = 0.01;
      Vector3d d1_0(0, 1, 0);
      auto indexes1 = builder_->AddFirstEdge(Vector3d(0, 0, 0), 0.0,
                                             Vector3d(l, 0, l), d1_0);
      auto indexes2 = builder_->AddEdge(0.1, Vector3d(l, l, l * 1.5));
      auto indexes3 = builder_->AddEdge(0.2, Vector3d(0, l, l * 3.0));

      EXPECT_EQ(indexes1, std::make_tuple(DerNodeIndex(0),  //
                                          DerEdgeIndex(0), DerNodeIndex(1)));
      EXPECT_EQ(indexes2, std::make_tuple(DerEdgeIndex(1), DerNodeIndex(2)));
      EXPECT_EQ(indexes3, std::make_tuple(DerEdgeIndex(2), DerNodeIndex(3)));
    } else if (opt == kClosedEnds) {
      const double l = 0.01;
      Vector3d d1_0(0, 1, 0);
      auto indexes1 = builder_->AddFirstEdge(Vector3d(0, 0, 0), 0.0,
                                             Vector3d(l, 0, l), d1_0);
      auto indexes2 = builder_->AddEdge(0.1, Vector3d(l, l, l * 1.5));
      auto indexes3 = builder_->AddEdge(0.2, Vector3d(0, l, l));
      auto indexes4 = builder_->AddEdge(0.1, Vector3d(0, 0, 0));

      EXPECT_EQ(indexes1, std::make_tuple(DerNodeIndex(0),  //
                                          DerEdgeIndex(0), DerNodeIndex(1)));
      EXPECT_EQ(indexes2, std::make_tuple(DerEdgeIndex(1), DerNodeIndex(2)));
      EXPECT_EQ(indexes3, std::make_tuple(DerEdgeIndex(2), DerNodeIndex(3)));
      EXPECT_EQ(indexes4, std::make_tuple(DerEdgeIndex(3), DerNodeIndex(0)));
    }
  }

  void CheckRodConfiguration() {
    std::unique_ptr<internal::DerState<double>> state =
        model_->CreateDerState();
    const auto& q = state->get_position();
    const auto& d1 = state->get_reference_frame_d1();

    RodConfigurationTests opt = std::get<0>(GetParam());
    if (opt == kOpenEnds) {
      const double l = 0.01;
      EXPECT_FALSE(state->has_closed_ends());
      EXPECT_EQ(state->num_nodes(), 4);
      EXPECT_TRUE(CompareMatrices(q.template segment<3>(4 * 0),  //
                                  Vector3d(0, 0, 0)));
      EXPECT_TRUE(CompareMatrices(q.template segment<3>(4 * 1),  //
                                  Vector3d(l, 0, l)));
      EXPECT_TRUE(CompareMatrices(q.template segment<3>(4 * 2),
                                  Vector3d(l, l, l * 1.5)));
      EXPECT_TRUE(CompareMatrices(q.template segment<3>(4 * 3),
                                  Vector3d(0, l, l * 3.0)));
      EXPECT_EQ(q[4 * 0 + 3], 0.0);
      EXPECT_EQ(q[4 * 1 + 3], 0.1);
      EXPECT_EQ(q[4 * 2 + 3], 0.2);
      EXPECT_TRUE(CompareMatrices(d1.col(0), Vector3d(0, 1, 0)));
    } else if (opt == kClosedEnds) {
      const double l = 0.01;
      EXPECT_TRUE(state->has_closed_ends());
      EXPECT_EQ(state->num_nodes(), 4);
      EXPECT_TRUE(CompareMatrices(q.template segment<3>(4 * 0),  //
                                  Vector3d(0, 0, 0)));
      EXPECT_TRUE(CompareMatrices(q.template segment<3>(4 * 1),  //
                                  Vector3d(l, 0, l)));
      EXPECT_TRUE(CompareMatrices(q.template segment<3>(4 * 2),
                                  Vector3d(l, l, l * 1.5)));
      EXPECT_TRUE(CompareMatrices(q.template segment<3>(4 * 3),  //
                                  Vector3d(0, l, l)));
      EXPECT_EQ(q[4 * 0 + 3], 0.0);
      EXPECT_EQ(q[4 * 1 + 3], 0.1);
      EXPECT_EQ(q[4 * 2 + 3], 0.2);
      EXPECT_EQ(q[4 * 3 + 3], 0.1);
      EXPECT_TRUE(CompareMatrices(d1.col(0), Vector3d(0, 1, 0)));
    }
  }

  void AddUndeformedState() {
    UndeformedStateTests opt = std::get<1>(GetParam());
    if (opt == kZeroCurvatureTwist) {
      builder_->SetZeroUndeformedCurvatureAndTwist();
    } else if (opt == kSameAsInitialState) {
      builder_->SetUndeformedStateToInitialState();
    }
  }

  void CheckUndeformedState() {
    const internal::DerUndeformedState<double>& undeformed =
        DerModelTester::get_der_undeformed_state(*model_);
    std::unique_ptr<internal::DerState<double>> state =
        model_->CreateDerState();

    UndeformedStateTests opt = std::get<1>(GetParam());
    if (opt == kZeroCurvatureTwist) {
      EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(),
                                  state->get_edge_length()));
      auto zero = Eigen::RowVectorXd::Zero(state->num_internal_nodes());
      EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(), zero));
      EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(), zero));
      EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), zero));
    } else if (opt == kSameAsInitialState) {
      EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(),
                                  state->get_edge_length()));
      EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(),
                                  state->get_curvature_kappa1()));
      EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(),
                                  state->get_curvature_kappa2()));
      EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), state->get_twist()));
    }
  }

  void AddCrossSection() {
    CrossSectionTests opt = std::get<2>(GetParam());
    if (opt == kCircular) {
      const double r = 1e-3;
      builder_->SetCircularCrossSection(r);
    } else if (opt == kRectangular) {
      const auto [width, height] = std::make_tuple(1.38e-3, 6e-3);
      builder_->SetRectangularCrossSection(width, height);
    } else if (opt == kElliptical) {
      const auto [a, b] = std::make_tuple(2.0e-3, 1.5e-3);
      builder_->SetEllipticalCrossSection(a, b);
    }

    const auto [E, G, rho] = std::make_tuple(3e9, 0.8e9, 910);
    builder_->SetMaterialProperties(E, G, rho);
  }

  void CheckCrossSection() {
    const internal::DerStructuralProperty<double>& der_structural_property =
        DerModelTester::get_der_structural_property(*model_);

    const auto [E, G, rho] = std::make_tuple(3e9, 0.8e9, 910);
    std::optional<internal::DerStructuralProperty<double>> expected;

    CrossSectionTests opt = std::get<2>(GetParam());
    if (opt == kCircular) {
      expected =
          internal::DerStructuralProperty<double>::FromCircularCrossSection(
              1e-3, E, G, rho);
    } else if (opt == kRectangular) {
      expected =
          internal::DerStructuralProperty<double>::FromRectangularCrossSection(
              1.38e-3, 6e-3, E, G, rho);
    } else if (opt == kElliptical) {
      expected =
          internal::DerStructuralProperty<double>::FromEllipticalCrossSection(
              2.0e-3, 1.5e-3, E, G, rho);
    }
    EXPECT_EQ(der_structural_property.EA(), expected->EA());
    EXPECT_EQ(der_structural_property.EI1(), expected->EI1());
    EXPECT_EQ(der_structural_property.EI2(), expected->EI2());
    EXPECT_EQ(der_structural_property.GJ(), expected->GJ());
    EXPECT_EQ(der_structural_property.rhoA(), expected->rhoA());
    EXPECT_EQ(der_structural_property.rhoJ(), expected->rhoJ());
  }

  void AddDampingCoefficient() {
    DampingCoefficientTests opt = std::get<3>(GetParam());
    if (opt == kZeroDamping) {
      builder_->SetDampingCoefficients(0.0, 0.0);
    } else if (opt == kNonZeroDamping) {
      builder_->SetDampingCoefficients(0.1, 0.2);
    }
  }

  void CheckDampingCoefficient() {
    const internal::DampingModel<double>& damping_model =
        DerModelTester::get_damping_model(*model_);

    DampingCoefficientTests opt = std::get<3>(GetParam());
    if (opt == kZeroDamping) {
      EXPECT_EQ(damping_model.mass_coeff_alpha(), 0.0);
      EXPECT_EQ(damping_model.stiffness_coeff_beta(), 0.0);
    } else if (opt == kNonZeroDamping) {
      EXPECT_EQ(damping_model.mass_coeff_alpha(), 0.1);
      EXPECT_EQ(damping_model.stiffness_coeff_beta(), 0.2);
    }
  }

 private:
  std::unique_ptr<DerModel<double>::Builder> builder_;
  std::unique_ptr<DerModel<double>> model_;
};

TEST_P(DerModelBuilderTest, Test) {
  Test();
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations, DerModelBuilderTest,
    ::testing::Combine(::testing::Values(kOpenEnds, kClosedEnds),
                       ::testing::Values(kZeroCurvatureTwist,
                                         kSameAsInitialState),
                       ::testing::Values(kCircular, kRectangular, kElliptical),
                       ::testing::Values(kZeroDamping, kNonZeroDamping)));

template <typename T>
class DummySystem final : public systems::LeafSystem<T> {
 public:
  DummySystem() {}
};

class DerModelTest : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    const auto [has_closed_ends, have_bc] = GetParam();

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
    const auto [width, height] = std::make_tuple(1.38e-3, 6e-3);
    builder.SetRectangularCrossSection(width, height);
    builder.SetDampingCoefficients(0.1, 0.2);
    der_model_ = builder.Build();

    if (have_bc) {
      fixed_nodes_ = {DerNodeIndex(0), DerNodeIndex(2)};
      fixed_edges_ = {DerEdgeIndex(0)};
      for (DerNodeIndex index : fixed_nodes_)
        der_model_->FixPositionOrAngle(index);
      for (DerEdgeIndex index : fixed_edges_)
        der_model_->FixPositionOrAngle(index);
    }

    const double g = 9.81;
    force_density_field_ =
        std::make_unique<GravityForceField<double>>(Vector3d(0, 0, -g), rho);
  }

  std::vector<DerNodeIndex> fixed_nodes_;
  std::vector<DerEdgeIndex> fixed_edges_;
  std::unique_ptr<DerModel<double>> der_model_;
  std::unique_ptr<ForceDensityField<double>> force_density_field_;
};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds_HaveBC, DerModelTest,
                         ::testing::Combine(::testing::Values(false, true),
                                            ::testing::Values(false, true)));

TEST_P(DerModelTest, ComputeResidual) {
  auto state = der_model_->CreateDerState();
  state->SetVelocity(VectorXd::LinSpaced(state->num_dofs(), 0.0, 1.0));
  state->SetAcceleration(VectorXd::LinSpaced(state->num_dofs(), 1.0, 2.0));

  auto dummy_context = DummySystem<double>().CreateDefaultContext();
  internal::ExternalForceField external_force_field(
      dummy_context.get(), {force_density_field_.get()});

  auto scratch = der_model_->MakeScratch();
  const VectorXd* residual;
  {
    LimitMalloc guard;
    residual = &der_model_->ComputeResidual(*state, external_force_field,
                                            scratch.get());
  }

  // Compute and expected residual and compare to `residual`.
  const auto& prop = DerModelTester::get_der_structural_property(*der_model_);
  const auto& undeformed =
      DerModelTester::get_der_undeformed_state(*der_model_);
  const auto& damping = DerModelTester::get_damping_model(*der_model_);
  const int num_dofs = state->num_dofs();

  VectorXd dEdq(num_dofs);
  internal::ComputeElasticEnergyJacobian<double>(prop, undeformed, *state,
                                                 &dEdq);
  auto d2Edq2 = internal::MakeEnergyHessianMatrix<double>(
      state->has_closed_ends(), state->num_nodes(), state->num_edges());
  internal::ComputeElasticEnergyHessian(prop, undeformed, *state, &d2Edq2);

  MatrixXd M = ComputeMassMatrix(prop, undeformed).toDenseMatrix();
  MatrixXd K = d2Edq2.MakeDenseMatrix().topLeftCorner(num_dofs, num_dofs);
  VectorXd qdot = state->get_velocity();
  VectorXd qddot = state->get_acceleration();
  double alpha = damping.mass_coeff_alpha();
  double beta = damping.stiffness_coeff_beta();
  VectorXd F_ext = external_force_field(prop, undeformed, *state).eval();

  VectorXd expected = M * qddot + dEdq + (alpha * M + beta * K) * qdot - F_ext;
  for (DerNodeIndex index : fixed_nodes_) {
    expected.template segment<3>(4 * index).setZero();
  }
  for (DerEdgeIndex index : fixed_edges_) {
    expected(4 * index + 3) = 0.0;
  }

  EXPECT_TRUE(CompareMatrices(*residual, expected, 1e-9));
}

TEST_P(DerModelTest, ComputeTangentMatrix) {
  auto state = der_model_->CreateDerState();
  std::array<double, 3> weights = {1.2, 3.4, 5.6};

  auto scratch = der_model_->MakeScratch();
  const internal::Block4x4SparseSymmetricMatrix<double>* tangent_matrix;
  {
    LimitMalloc guard;
    tangent_matrix =
        &der_model_->ComputeTangentMatrix(*state, weights, scratch.get());
  }

  // Compute the expected tangent matrix and compare to `tangent_matrix`.
  const auto& prop = DerModelTester::get_der_structural_property(*der_model_);
  const auto& undeformed =
      DerModelTester::get_der_undeformed_state(*der_model_);
  const auto& damping = DerModelTester::get_damping_model(*der_model_);
  const int num_dofs = state->num_dofs();

  auto d2Edq2 = internal::MakeEnergyHessianMatrix<double>(
      state->has_closed_ends(), state->num_nodes(), state->num_edges());
  internal::ComputeElasticEnergyHessian(prop, undeformed, *state, &d2Edq2);

  MatrixXd M = ComputeMassMatrix(prop, undeformed).toDenseMatrix();
  MatrixXd K = d2Edq2.MakeDenseMatrix().topLeftCorner(num_dofs, num_dofs);
  double alpha = damping.mass_coeff_alpha();
  double beta = damping.stiffness_coeff_beta();

  MatrixXd expected =
      weights[0] * K + weights[1] * (alpha * M + beta * K) + weights[2] * M;
  for (DerNodeIndex index : fixed_nodes_) {
    expected.template middleCols<3>(4 * index).setZero();
    expected.template middleRows<3>(4 * index).setZero();
    expected.template block<3, 3>(4 * index, 4 * index).setIdentity();
  }
  for (DerEdgeIndex index : fixed_edges_) {
    expected.template middleCols<1>(4 * index + 3).setZero();
    expected.template middleRows<1>(4 * index + 3).setZero();
    expected(4 * index + 3, 4 * index + 3) = 1.0;
  }

  const double kTol = 1e-9;
  if (der_model_->has_closed_ends()) {
    EXPECT_TRUE(
        CompareMatrices(tangent_matrix->MakeDenseMatrix(), expected, kTol));
    return;
  }
  const MatrixXd result = tangent_matrix->MakeDenseMatrix();
  EXPECT_TRUE(CompareMatrices(result.topLeftCorner(num_dofs, num_dofs),
                              expected, kTol));
  /* Last diagonal entry must be 1 per ComputeTangentMatrix() documentation. */
  EXPECT_EQ(result(num_dofs, num_dofs), 1.0);
  EXPECT_TRUE(result.topRightCorner(num_dofs, 1).isZero());
  EXPECT_TRUE(result.bottomLeftCorner(1, num_dofs).isZero());
}

TEST_P(DerModelTest, ApplyBoundaryCondition) {
  auto state = der_model_->CreateDerState();
  state->SetVelocity(VectorXd::Ones(state->num_dofs()));
  state->SetAcceleration(VectorXd::Ones(state->num_dofs()));

  VectorXd q_expected = VectorXd::Ones(state->num_dofs());
  VectorXd v_expected = state->get_velocity();
  VectorXd a_expected = state->get_acceleration();
  for (DerNodeIndex index : fixed_nodes_) {
    q_expected.template segment<3>(4 * index) =
        state->get_position().template segment<3>(4 * index);
    v_expected.template segment<3>(4 * index).setZero();
    a_expected.template segment<3>(4 * index).setZero();
  }
  for (DerEdgeIndex index : fixed_edges_) {
    q_expected(4 * index + 3) = state->get_position()(4 * index + 3);
    v_expected(4 * index + 3) = 0.0;
    a_expected(4 * index + 3) = 0.0;
  }
  state->AdvancePositionToNextStep(VectorXd::Ones(state->num_dofs()));
  der_model_->ApplyBoundaryCondition(state.get());

  EXPECT_TRUE(CompareMatrices(state->get_position(), q_expected));
  EXPECT_TRUE(CompareMatrices(state->get_velocity(), v_expected));
  EXPECT_TRUE(CompareMatrices(state->get_acceleration(), a_expected));
}

TEST_P(DerModelTest, CloneAndScalarConversion) {
  auto der_model2 = der_model_->Clone();
  auto der_model3 = der_model2->ToScalarType<AutoDiffXd>();
  auto der_model4 = der_model3->Clone();
  auto der_model5 = der_model4->ToScalarType<double>();
}

}  // namespace
}  // namespace der
}  // namespace multibody
}  // namespace drake
