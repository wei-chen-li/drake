#include "drake/multibody/der/der_undeformed_state.h"

#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

template <typename T>
class DerUndeformedStateTest : public ::testing::Test {};

using NonsymbolicScalarTypes = ::testing::Types<double, AutoDiffXd>;
TYPED_TEST_SUITE(DerUndeformedStateTest, NonsymbolicScalarTypes);

TYPED_TEST(DerUndeformedStateTest, ZeroCurvatureAndTwist1) {
  using T = TypeParam;

  for (const bool has_closed_ends : {false, true}) {
    const int num_edges = 3;
    const double edge_length = 0.01;
    auto rest = DerUndeformedState<T>::ZeroCurvatureAndTwist(
        has_closed_ends, num_edges, edge_length);
    EXPECT_EQ(rest.has_closed_ends(), has_closed_ends);

    EXPECT_TRUE(
        CompareMatrices(rest.get_edge_length(),
                        Eigen::RowVectorX<T>::Ones(num_edges) * edge_length));

    const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
    EXPECT_TRUE(CompareMatrices(
        rest.get_voronoi_length(),
        Eigen::RowVectorX<T>::Ones(num_internal_nodes) * edge_length));

    auto zero = Eigen::RowVectorX<T>::Zero(num_internal_nodes);
    EXPECT_TRUE(CompareMatrices(rest.get_curvature_kappa1(), zero));
    EXPECT_TRUE(CompareMatrices(rest.get_curvature_kappa2(), zero));
    EXPECT_TRUE(CompareMatrices(rest.get_twist(), zero));
  }
}

TYPED_TEST(DerUndeformedStateTest, ZeroCurvatureAndTwist2) {
  using T = TypeParam;

  for (const bool has_closed_ends : {false, true}) {
    const int num_edges = 3;
    std::vector<T> edge_length = {0.10, 0.12, 0.08};
    auto rest = DerUndeformedState<T>::ZeroCurvatureAndTwist(has_closed_ends,
                                                             edge_length);
    EXPECT_EQ(rest.has_closed_ends(), has_closed_ends);

    EXPECT_TRUE(CompareMatrices(rest.get_edge_length(),
                                Eigen::Map<Eigen::RowVectorX<T>>(
                                    edge_length.data(), edge_length.size())));

    const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
    Eigen::RowVectorX<T> voronoi_length(num_internal_nodes);
    if (has_closed_ends) {
      voronoi_length << 0.11, 0.10, 0.09;
    } else {
      voronoi_length << 0.11, 0.10;
    }
    EXPECT_TRUE(CompareMatrices(rest.get_voronoi_length(), voronoi_length));

    auto zero = Eigen::RowVectorX<T>::Zero(num_internal_nodes);
    EXPECT_TRUE(CompareMatrices(rest.get_curvature_kappa1(), zero));
    EXPECT_TRUE(CompareMatrices(rest.get_curvature_kappa2(), zero));
    EXPECT_TRUE(CompareMatrices(rest.get_twist(), zero));
  }
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
