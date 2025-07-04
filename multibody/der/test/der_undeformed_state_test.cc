#include "drake/multibody/der/der_undeformed_state.h"

#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

class DerUndeformedStateTest : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds, DerUndeformedStateTest,
                         ::testing::Values(false, true));

TEST_P(DerUndeformedStateTest, ZeroCurvatureAndTwist) {
  const bool has_closed_ends = GetParam();

  const int num_edges = 3;
  Eigen::RowVectorXd edge_length(num_edges);
  edge_length << 0.10, 0.12, 0.08;
  const auto undeformed = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, edge_length);
  EXPECT_EQ(undeformed.has_closed_ends(), has_closed_ends);

  EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(), edge_length));

  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  Eigen::RowVectorXd voronoi_length(num_internal_nodes);
  if (has_closed_ends) {
    voronoi_length << 0.11, 0.10, 0.09;
  } else {
    voronoi_length << 0.11, 0.10;
  }
  EXPECT_TRUE(CompareMatrices(undeformed.get_voronoi_length(), voronoi_length));

  const auto zero = Eigen::RowVectorXd::Zero(num_internal_nodes);
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(), zero));
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(), zero));
  EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), zero));
}

TEST_P(DerUndeformedStateTest, set_edge_length) {
  const bool has_closed_ends = GetParam();

  const int num_edges = 3;
  Eigen::RowVectorXd edge_length = Eigen::RowVectorXd::Constant(num_edges, 0.1);
  auto undeformed = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, edge_length);

  edge_length << 0.10, 0.12, 0.08;
  undeformed.set_edge_length(edge_length);
  EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(), edge_length));

  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  Eigen::RowVectorXd voronoi_length(num_internal_nodes);
  if (has_closed_ends) {
    voronoi_length << 0.11, 0.10, 0.09;
  } else {
    voronoi_length << 0.11, 0.10;
  }
  EXPECT_TRUE(CompareMatrices(undeformed.get_voronoi_length(), voronoi_length));

  const auto zero = Eigen::RowVectorXd::Zero(num_internal_nodes);
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(), zero));
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(), zero));
  EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), zero));
}

TEST_P(DerUndeformedStateTest, set_curvature_kappa) {
  const bool has_closed_ends = GetParam();

  const int num_edges = 3;
  const auto edge_length = Eigen::RowVectorXd::Constant(num_edges, 0.1);
  auto undeformed = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, edge_length);

  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  const auto kappa1 =
      Eigen::RowVectorXd::LinSpaced(num_internal_nodes, 0.0, 1.0);
  const auto kappa2 =
      Eigen::RowVectorXd::LinSpaced(num_internal_nodes, 1.0, 2.0);
  undeformed.set_curvature_kappa(kappa1, kappa2);

  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(), kappa1));
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(), kappa2));
  EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(), edge_length));
  const auto zero = Eigen::RowVectorXd::Zero(num_internal_nodes);
  EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), zero));
}

TEST_P(DerUndeformedStateTest, set_twist) {
  const bool has_closed_ends = GetParam();

  const int num_edges = 3;
  const auto edge_length = Eigen::RowVectorXd::Constant(num_edges, 0.1);
  auto undeformed = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, edge_length);

  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  const auto twist =
      Eigen::RowVectorXd::LinSpaced(num_internal_nodes, 0.0, 1.0);
  undeformed.set_twist(twist);

  EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), twist));
  EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(), edge_length));
  const auto zero = Eigen::RowVectorXd::Zero(num_internal_nodes);
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(), zero));
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(), zero));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
