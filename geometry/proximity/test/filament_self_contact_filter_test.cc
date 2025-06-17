#include "drake/geometry/proximity/filament_self_contact_filter.h"

#include <numeric>
#include <random>

#include <gtest/gtest.h>

namespace drake {
namespace geometry {
namespace internal {
namespace filament {
namespace {

constexpr double kC = 1.0;

GTEST_TEST(FilamentSelfContactFilterTest, OpenEnds1Edge) {
  Eigen::RowVectorXd edge_lengths(1);
  edge_lengths << 1.1;
  FilamentSelfContactFilter filter(false, edge_lengths, kC);
  EXPECT_FALSE(filter.ShouldCollide(0, 0));
}

GTEST_TEST(FilamentSelfContactFilterTest, OpenEnds2Edges) {
  Eigen::RowVectorXd edge_lengths(2);
  edge_lengths << 1.1, 1.1;
  FilamentSelfContactFilter filter(false, edge_lengths, kC);
  EXPECT_FALSE(filter.ShouldCollide(0, 0));
  EXPECT_FALSE(filter.ShouldCollide(0, 1));
  EXPECT_FALSE(filter.ShouldCollide(1, 1));
}

GTEST_TEST(FilamentSelfContactFilterTest, OpenEnds3Edges) {
  Eigen::RowVectorXd edge_lengths(3);
  edge_lengths << 1.1, 1.0, 1.1;
  FilamentSelfContactFilter filter_a(false, edge_lengths, kC);
  EXPECT_FALSE(filter_a.ShouldCollide(0, 1));
  EXPECT_FALSE(filter_a.ShouldCollide(1, 2));
  EXPECT_FALSE(filter_a.ShouldCollide(0, 2));

  edge_lengths << 1.1, 1.1, 1.1;
  FilamentSelfContactFilter filter_b(false, edge_lengths, kC);
  EXPECT_FALSE(filter_b.ShouldCollide(0, 1));
  EXPECT_FALSE(filter_b.ShouldCollide(1, 2));
  EXPECT_TRUE(filter_b.ShouldCollide(0, 2));
}

GTEST_TEST(FilamentSelfContactFilterTest, OpenEnds100Edges) {
  unsigned int seed = 0;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  const int num_edges = 100;
  Eigen::RowVectorXd edge_lengths(num_edges);
  for (int i = 0; i < num_edges; ++i) {
    edge_lengths[i] = 1.0 - dist(gen);
  }

  FilamentSelfContactFilter filter(false, edge_lengths, kC);
  for (int i = 0; i < num_edges; ++i) {
    for (int j = i + 1; j < num_edges; ++j) {
      // ∑ₖ₌ᵢ₊₁ʲ⁻¹ edge_lengths[k].
      const double sum = std::accumulate(edge_lengths.cbegin() + i + 1,
                                         edge_lengths.cbegin() + j, 0.0);
      EXPECT_EQ(filter.ShouldCollide(i, j), sum > kC);
      EXPECT_EQ(filter.ShouldCollide(j, i), sum > kC);
    }
  }
}

GTEST_TEST(FilamentSelfContactFilterTest, ClosedEnds2Edges) {
  Eigen::RowVectorXd edge_lengths(2);
  edge_lengths << 1.1, 1.1;
  FilamentSelfContactFilter filter(true, edge_lengths, kC);
  EXPECT_FALSE(filter.ShouldCollide(0, 0));
  EXPECT_FALSE(filter.ShouldCollide(0, 1));
  EXPECT_FALSE(filter.ShouldCollide(1, 1));
}

GTEST_TEST(FilamentSelfContactFilterTest, ClosedEnds3Edges) {
  Eigen::RowVectorXd edge_lengths(3);
  edge_lengths << 1.1, 1.1, 1.1;
  FilamentSelfContactFilter filter(true, edge_lengths, kC);
  EXPECT_FALSE(filter.ShouldCollide(0, 0));
  EXPECT_FALSE(filter.ShouldCollide(0, 1));
  EXPECT_FALSE(filter.ShouldCollide(0, 2));
  EXPECT_FALSE(filter.ShouldCollide(1, 1));
  EXPECT_FALSE(filter.ShouldCollide(1, 2));
  EXPECT_FALSE(filter.ShouldCollide(2, 2));
}

GTEST_TEST(FilamentSelfContactFilterTest, ClosedEnds100Edges) {
  unsigned int seed = 0;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  const int num_edges = 100;
  Eigen::RowVectorXd edge_lengths(num_edges);
  for (int i = 0; i < num_edges; ++i) {
    edge_lengths[i] = 1.0 - dist(gen);
  }

  FilamentSelfContactFilter filter(true, edge_lengths, kC);
  for (int i = 0; i < num_edges; ++i) {
    for (int j = i + 1; j < num_edges; ++j) {
      // ∑ₖ₌ᵢ₊₁ʲ⁻¹ edge_lengths[k].
      const double sum1 = std::accumulate(edge_lengths.cbegin() + i + 1,
                                          edge_lengths.cbegin() + j, 0.0);
      const double sum2 =
          std::accumulate(edge_lengths.cbegin(), edge_lengths.cend(), 0.0) -
          sum1 - edge_lengths[i] - edge_lengths[j];
      EXPECT_EQ(filter.ShouldCollide(i, j), sum1 > kC && sum2 > kC);
      EXPECT_EQ(filter.ShouldCollide(j, i), sum1 > kC && sum2 > kC);
    }
  }
}

}  // namespace
}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
