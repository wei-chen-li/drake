#include "drake/multibody/der/der_state.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/frame_transport.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

// This unit test is based on [Jawed, 2018, §4.5].
// [Jawed, 2018] M. K. Jawed, A. Novelia, and O. M. O'Reilly, A Primer on the
// Kinematics of Discrete Elastic Rods. Springer, 2018.

using Eigen::Vector3d;

static constexpr double kTol = 1e-15;

class DerStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    bool has_closed_ends = false;

    std::vector<Vector3d> node_positions(3);
    node_positions[0] = Vector3d(0, 0, 0);
    node_positions[1] = Vector3d(1, 0, 0);
    node_positions[2] = Vector3d(2, 1, 0);

    std::vector<double> edge_angles = {M_PI / 6, M_PI / 6};

    Vector3d d1_0(0, 1, 0);

    der_state_system_ = std::make_unique<DerStateSystem<double>>(
        has_closed_ends, node_positions, edge_angles, d1_0);
    state_ = std::make_unique<DerState<double>>(der_state_system_.get());
  }

  std::unique_ptr<DerState<double>> state_{};

 private:
  std::unique_ptr<const DerStateSystem<double>> der_state_system_{};
};

TEST_F(DerStateTest, Position) {
  Eigen::VectorXd q_expected(11);
  const double ang = M_PI / 6;
  q_expected << 0, 0, 0, ang, 1, 0, 0, ang, 2, 1, 0;

  const auto& q = state_->get_position();
  EXPECT_TRUE(CompareMatrices(q, q_expected));
}

TEST_F(DerStateTest, Velocity) {
  Eigen::VectorXd qdot = Eigen::VectorXd::Ones(11);
  state_->SetVelocity(qdot);
  EXPECT_TRUE(CompareMatrices(state_->get_velocity(), qdot));
}

TEST_F(DerStateTest, Acceleration) {
  Eigen::VectorXd qddot = Eigen::VectorXd::Ones(11);
  state_->SetAcceleration(qddot);
  EXPECT_TRUE(CompareMatrices(state_->get_acceleration(), qddot));
}

TEST_F(DerStateTest, EdgeLength) {
  const auto& edge_length = state_->get_edge_length();
  EXPECT_NEAR(edge_length[0], 1.0, kTol);
  EXPECT_NEAR(edge_length[1], sqrt(2), kTol);
}

TEST_F(DerStateTest, Tangent) {
  const auto& t = state_->get_tangent();
  EXPECT_TRUE(CompareMatrices(t.col(0), Vector3d(1, 0, 0), kTol));
  EXPECT_TRUE(CompareMatrices(t.col(1), Vector3d(1, 1, 0).normalized(), kTol));
}

TEST_F(DerStateTest, ReferenceFrame) {
  const auto& d1 = state_->get_reference_frame_d1();
  const auto& d2 = state_->get_reference_frame_d2();

  EXPECT_TRUE(CompareMatrices(d1.col(0), Vector3d(0, 1, 0), kTol));
  EXPECT_TRUE(CompareMatrices(d2.col(0), Vector3d(0, 0, 1), kTol));

  EXPECT_TRUE(
      CompareMatrices(d1.col(1), Vector3d(-1, 1, 0).normalized(), kTol));
  EXPECT_TRUE(CompareMatrices(d2.col(1), Vector3d(0, 0, 1), kTol));
}

TEST_F(DerStateTest, MaterialFrame) {
  const double a = sqrt(2) / 2;
  auto [sin30, cos30] = std::make_pair(1.0 / 2, sqrt(3) / 2);

  const auto& m1 = state_->get_material_frame_m1();
  const auto& m2 = state_->get_material_frame_m2();

  EXPECT_TRUE(CompareMatrices(m1.col(0), Vector3d(0, cos30, sin30), kTol));
  EXPECT_TRUE(CompareMatrices(m2.col(0), Vector3d(0, -sin30, cos30), kTol));

  EXPECT_TRUE(
      CompareMatrices(m1.col(1), Vector3d(-a * cos30, a * cos30, sin30), kTol));
  EXPECT_TRUE(
      CompareMatrices(m2.col(1), Vector3d(a * sin30, -a * sin30, cos30), kTol));
}

TEST_F(DerStateTest, Curvature) {
  const auto& m1 = state_->get_material_frame_m1();
  const auto& m2 = state_->get_material_frame_m2();
  Vector3d curvature_expected = 2 / (1 + sqrt(2)) * Vector3d(0, 0, 1);
  double kappa1_expected = curvature_expected.dot((m2.col(0) + m2.col(1)) / 2);
  double kappa2_expected = -curvature_expected.dot((m1.col(0) + m1.col(1)) / 2);

  const auto& curvature = state_->get_discrete_integrated_curvature();
  EXPECT_TRUE(CompareMatrices(curvature.col(0), curvature_expected, kTol));

  const auto& kappa1 = state_->get_curvature_kappa1();
  EXPECT_NEAR(kappa1[0], kappa1_expected, kTol);

  const auto& kappa2 = state_->get_curvature_kappa2();
  EXPECT_NEAR(kappa2[0], kappa2_expected, kTol);
}

TEST_F(DerStateTest, Twist) {
  // The twist and reference twist are related by τᵢ = γⁱ⁺¹ - γⁱ + τᵢ,ᵣₑ. The
  // initial reference twist is zero. And since γⁱ and γⁱ⁺¹ have the same angle
  // here, τᵢ = 0.
  const auto& twist = state_->get_twist();
  EXPECT_NEAR(twist[0], 0.0, kTol);
}

class DerStateAfterDeformationTest : public DerStateTest {
 private:
  void SetUp() override {
    DerStateTest::SetUp();

    Eigen::VectorXd q = state_->get_position();
    q.template segment<3>(4 * 2) = Vector3d(2, 1, e_);
    state_->AdvancePositionToNextStep(q);
  }

 protected:
  const double e_ = 0.37;
};

TEST_F(DerStateAfterDeformationTest, Position) {
  Eigen::VectorXd q_expected(11);
  const double ang = M_PI / 6;
  q_expected << 0, 0, 0, ang, 1, 0, 0, ang, 2, 1, e_;

  const auto& q = state_->get_position();
  EXPECT_TRUE(CompareMatrices(q, q_expected));
}

TEST_F(DerStateAfterDeformationTest, EdgeLength) {
  const auto& edge_length = state_->get_edge_length();
  EXPECT_NEAR(edge_length[0], 1.0, kTol);
  EXPECT_NEAR(edge_length[1], sqrt(2 + e_ * e_), kTol);
}

TEST_F(DerStateAfterDeformationTest, Tangent) {
  const auto& t = state_->get_tangent();
  EXPECT_TRUE(CompareMatrices(t.col(0), Vector3d(1, 0, 0), kTol));
  EXPECT_TRUE(CompareMatrices(t.col(1), Vector3d(1, 1, e_).normalized(), kTol));
}

TEST_F(DerStateAfterDeformationTest, ReferenceFrame) {
  const auto& d1 = state_->get_reference_frame_d1();
  const auto& d2 = state_->get_reference_frame_d2();

  EXPECT_TRUE(CompareMatrices(d1.col(0), Vector3d(0, 1, 0), kTol));
  EXPECT_TRUE(CompareMatrices(d2.col(0), Vector3d(0, 0, 1), kTol));

  EXPECT_TRUE(
      CompareMatrices(d1.col(1), Vector3d(-1, 1, 0).normalized(), kTol));
  EXPECT_TRUE(
      CompareMatrices(d2.col(1), Vector3d(-e_, -e_, 2).normalized(), kTol));
}

TEST_F(DerStateAfterDeformationTest, MaterialFrame) {
  const double d = 1 / sqrt(e_ * e_ + 2);
  const double c = d * e_ / 2;
  auto [sin30, cos30] = std::make_pair(1.0 / 2, sqrt(3) / 2);

  const auto& m1 = state_->get_material_frame_m1();
  const auto& m2 = state_->get_material_frame_m2();

  EXPECT_TRUE(CompareMatrices(m1.col(0), Vector3d(0, cos30, sin30), kTol));
  EXPECT_TRUE(CompareMatrices(m2.col(0), Vector3d(0, -sin30, cos30), kTol));

  EXPECT_TRUE(CompareMatrices(
      m1.col(1), Vector3d(-c - cos30, -c + cos30, d).normalized(), kTol));
  EXPECT_TRUE(CompareMatrices(
      m2.col(1), Vector3d(1, 1, e_).cross(m1.col(1)).normalized(), kTol));
}

TEST_F(DerStateAfterDeformationTest, Curvature) {
  const auto& m1 = state_->get_material_frame_m1();
  const auto& m2 = state_->get_material_frame_m2();
  Vector3d curvature_expected =
      2 / (1 + sqrt(2 + e_ * e_)) * Vector3d(0, -e_, 1);
  double kappa1_expected = curvature_expected.dot((m2.col(0) + m2.col(1)) / 2);
  double kappa2_expected = -curvature_expected.dot((m1.col(0) + m1.col(1)) / 2);

  const auto& curvature = state_->get_discrete_integrated_curvature();
  EXPECT_TRUE(CompareMatrices(curvature.col(0), curvature_expected, kTol));

  const auto& kappa1 = state_->get_curvature_kappa1();
  EXPECT_NEAR(kappa1[0], kappa1_expected, kTol);

  const auto& kappa2 = state_->get_curvature_kappa2();
  EXPECT_NEAR(kappa2[0], kappa2_expected, kTol);
}

TEST_F(DerStateAfterDeformationTest, ReferenceTwist) {
  // The twist and reference twist are related by τᵢ = γⁱ⁺¹ - γⁱ + τᵢ,ᵣₑ, where
  // γⁱ are the edge angles. Since γⁱ and γⁱ⁺¹ have the same angle here (both
  // are π/6), we have τᵢ,ᵣₑ = τᵢ.
  const auto& ref_twist = state_->get_twist();

  double e2 = e_ * e_;
  double cos_ref_twist_expected =
      (2 + e2 * (1 + sqrt(2 + e2))) / (sqrt(2) * (1 + e2) * sqrt(2 + e2));

  EXPECT_NEAR(cos(ref_twist[0]), cos_ref_twist_expected, kTol);
  EXPECT_EQ(ref_twist[0] > 0, e_ > 0);
}

class DerStateAdjustPositionWithinStepTest : public DerStateTest {
 private:
  void SetUp() override {
    DerStateTest::SetUp();

    Eigen::VectorXd q(11);
    const double ang = M_PI / 6;

    Eigen::VectorXd q1(11);
    q1 << 0, 0, 0, ang, 1, 1, 0, ang, 2, 1, 0;
    state_->AdjustPositionWithinStep(q1);

    /* Because we call AdjustPositionWithinStep() here, any arbitrary calls to
     AdjustPositionWithinStep() before does not affect the result. */
    Eigen::VectorXd q2(11);
    q2 << 0, 0, 0, ang, 1, 0, 0, ang, 2, 1, e_;
    state_->AdjustPositionWithinStep(q2);
  }

 protected:
  double e_ = -0.23;
};

TEST_F(DerStateAdjustPositionWithinStepTest, ReferenceFrame) {
  const auto& d1 = state_->get_reference_frame_d1();
  const auto& d2 = state_->get_reference_frame_d2();

  EXPECT_TRUE(CompareMatrices(d1.col(0), Vector3d(0, 1, 0), kTol));
  EXPECT_TRUE(CompareMatrices(d2.col(0), Vector3d(0, 0, 1), kTol));

  EXPECT_TRUE(
      CompareMatrices(d1.col(1), Vector3d(-1, 1, 0).normalized(), kTol));
  EXPECT_TRUE(
      CompareMatrices(d2.col(1), Vector3d(-e_, -e_, 2).normalized(), kTol));
}

TEST_F(DerStateAdjustPositionWithinStepTest, ReferenceTwist) {
  const auto& ref_twist = state_->get_twist();

  double e2 = e_ * e_;
  double cos_ref_twist_expected =
      (2 + e2 * (1 + sqrt(2 + e2))) / (sqrt(2) * (1 + e2) * sqrt(2 + e2));

  EXPECT_NEAR(cos(ref_twist[0]), cos_ref_twist_expected, kTol);
  EXPECT_EQ(ref_twist[0] > 0, e_ > 0);
}

class DerStateAdvancePositionToNextStepTest : public DerStateTest {
 private:
  void SetUp() override {
    DerStateTest::SetUp();

    const double ang = M_PI / 6;

    Eigen::VectorXd q1(11);
    q1 << 0, 0, 0, ang, 1, 1, 0, ang, 2, 1, 0;
    state_->AdjustPositionWithinStep(q1);

    /* Because we call AdvancePositionToNextStep() here, future calculations of
     some quantities will depend on q1. */
    Eigen::VectorXd q2(11);
    q2 << 0, 0, 0, ang, 1, 0, 0, ang, 2, 1, e_;
    state_->AdvancePositionToNextStep(q2);
  }

 protected:
  double e_ = -0.23;
};

TEST_F(DerStateAdvancePositionToNextStepTest, ReferenceFrame) {
  const auto& d1 = state_->get_reference_frame_d1();
  const auto& d2 = state_->get_reference_frame_d2();

  EXPECT_TRUE(CompareMatrices(d1.col(0), Vector3d(0, 1, 0), kTol));
  EXPECT_TRUE(CompareMatrices(d2.col(0), Vector3d(0, 0, 1), kTol));

  /* d₁¹ does not equal (−1/√2, 1/√2, 0) as is in the test
   DerStateAdjustPositionWithinStepTest.ReferenceFrame above. */
  EXPECT_FALSE(
      CompareMatrices(d1.col(1), Vector3d(-1, 1, 0).normalized(), kTol));

  Vector3d d1_1_expected;
  math::FrameTransport<double>(Vector3d(1, 0, 0), Vector3d(0, 1, 0),
                               Vector3d(1, 1, e_).normalized(), d1_1_expected);
  EXPECT_TRUE(CompareMatrices(d1.col(1), d1_1_expected, kTol));
  EXPECT_TRUE(CompareMatrices(
      d2.col(1), Vector3d(1, 1, e_).cross(d1_1_expected).normalized(), kTol));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
