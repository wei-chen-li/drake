#include "drake/geometry/proximity/filament_self_contact_filter.h"

#include <numeric>
#include <random>

#include <gtest/gtest.h>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {
namespace {

/* Creates a filament with open-ends, have edge legths given by `edge_lengths`,
 and a cross-section diameter given by `cross_section_diameter`. */
Filament MakeOpenEndsFilament(const std::vector<double>& edge_lengths,
                              double cross_section_diameter) {
  DRAKE_THROW_UNLESS(ssize(edge_lengths) >= 1);
  const int num_edges = ssize(edge_lengths);
  const int num_nodes = num_edges + 1;
  Eigen::Matrix3Xd node_pos(3, num_nodes);
  node_pos.setZero();
  for (int i = 0; i < num_edges; ++i) {
    node_pos.col(i + 1)[0] = node_pos.col(i)[0] + edge_lengths[i];
  }
  return Filament(
      false, std::move(node_pos),
      Filament::CircularCrossSection{.diameter = cross_section_diameter});
}

/* Creates a filament with open-ends, have edge legths given by `edge_lengths`,
 and a cross-section diameter given by `cross_section_diameter`. */
Filament MakeClosedEndsFilament(const std::vector<double>& edge_lengths,
                                double cross_section_diameter) {
  DRAKE_THROW_UNLESS(ssize(edge_lengths) >= 2);
  const int num_edges = ssize(edge_lengths);
  const int num_nodes = num_edges;

  solvers::MathematicalProgram prog;
  auto alpha = prog.NewContinuousVariables(num_edges, "alpha");
  auto r = prog.NewContinuousVariables(1, "r")[0];
  for (int i = 0; i < num_edges; ++i) {
    prog.AddConstraint(r * sin(alpha[i]) == edge_lengths[i] / 2);
  }
  prog.AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(num_edges), M_PI,
                                   alpha);

  Eigen::VectorXd initial_guess(num_edges + 1);
  initial_guess.head(num_edges).setConstant(M_PI / num_edges);
  initial_guess[num_edges] =
      std::accumulate(edge_lengths.begin(), edge_lengths.end(), 0.0) /
      (2 * M_PI);
  solvers::MathematicalProgramResult result =
      solvers::Solve(prog, initial_guess);
  DRAKE_DEMAND(result.is_success());

  Eigen::Matrix3Xd node_pos(3, num_nodes);
  node_pos.setZero();
  const double r_val = result.GetSolution(r);
  double theta_val = 0.0;
  for (int i = 0; i < num_nodes; ++i) {
    node_pos.col(i)[0] = r_val * cos(theta_val);
    node_pos.col(i)[1] = r_val * sin(theta_val);
    theta_val += result.GetSolution(alpha[i]) * 2;
  }
  return Filament(
      true, std::move(node_pos),
      Filament::CircularCrossSection{.diameter = cross_section_diameter});
}

GTEST_TEST(FilamentSelfContactFilterTEST, OpenEnds1Edge) {
  Filament filament = MakeOpenEndsFilament({1.1}, 1.0);
  FilamentSelfContactFilter filter(filament);
  EXPECT_FALSE(filter.ShouldCollide(0, 0));
}

GTEST_TEST(FilamentSelfContactFilterTEST, OpenEnds2Edges) {
  Filament filament = MakeOpenEndsFilament({1.1, 1.1}, 1.0);
  FilamentSelfContactFilter filter(filament);
  EXPECT_FALSE(filter.ShouldCollide(0, 0));
  EXPECT_FALSE(filter.ShouldCollide(0, 1));
  EXPECT_FALSE(filter.ShouldCollide(1, 1));
}

GTEST_TEST(FilamentSelfContactFilterTEST, OpenEnds3Edges) {
  Filament filament_a = MakeOpenEndsFilament({1.1, 1.0, 1.1}, 1.0);
  FilamentSelfContactFilter filter_a(filament_a);
  EXPECT_FALSE(filter_a.ShouldCollide(0, 1));
  EXPECT_FALSE(filter_a.ShouldCollide(1, 2));
  EXPECT_FALSE(filter_a.ShouldCollide(0, 2));

  Filament filament_b = MakeOpenEndsFilament({1.1, 1.1, 1.1}, 1.0);
  FilamentSelfContactFilter filter_b(filament_b);
  EXPECT_FALSE(filter_b.ShouldCollide(0, 1));
  EXPECT_FALSE(filter_b.ShouldCollide(1, 2));
  EXPECT_TRUE(filter_b.ShouldCollide(0, 2));
}

GTEST_TEST(FilamentSelfContactFilterTEST, OpenEnds100Edges) {
  unsigned int seed = 0;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  const int num_edges = 100;
  std::vector<double> edge_lengths(num_edges);
  for (int i = 0; i < num_edges; ++i) {
    edge_lengths[i] = 1.0 - dist(gen);
  }

  const double d = 1.0;
  Filament filament = MakeOpenEndsFilament(edge_lengths, d);
  FilamentSelfContactFilter filter(filament);
  for (int i = 0; i < num_edges; ++i) {
    for (int j = i + 1; j < num_edges; ++j) {
      // ∑ₖ₌ᵢ₊₁ʲ⁻¹ edge_lengths[k].
      const double sum = std::accumulate(edge_lengths.cbegin() + i + 1,
                                         edge_lengths.cbegin() + j, 0.0);
      EXPECT_EQ(filter.ShouldCollide(i, j), sum > d);
      EXPECT_EQ(filter.ShouldCollide(j, i), sum > d);
    }
  }
}

GTEST_TEST(FilamentSelfContactFilterTEST, ClosedEnds2Edges) {
  Filament filament = MakeClosedEndsFilament({1.1, 1.1}, 1.0);
  FilamentSelfContactFilter filter(filament);
  EXPECT_FALSE(filter.ShouldCollide(0, 0));
  EXPECT_FALSE(filter.ShouldCollide(0, 1));
  EXPECT_FALSE(filter.ShouldCollide(1, 1));
}

GTEST_TEST(FilamentSelfContactFilterTEST, ClosedEnds3Edges) {
  Filament filament = MakeClosedEndsFilament({1.1, 1.1, 1.1}, 1.0);
  FilamentSelfContactFilter filter(filament);
  EXPECT_FALSE(filter.ShouldCollide(0, 0));
  EXPECT_FALSE(filter.ShouldCollide(0, 1));
  EXPECT_FALSE(filter.ShouldCollide(0, 2));
  EXPECT_FALSE(filter.ShouldCollide(1, 1));
  EXPECT_FALSE(filter.ShouldCollide(1, 2));
  EXPECT_FALSE(filter.ShouldCollide(2, 2));
}

GTEST_TEST(FilamentSelfContactFilterTEST, ClosedEnds100Edges) {
  unsigned int seed = 0;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  const int num_edges = 100;
  std::vector<double> edge_lengths(num_edges);
  for (int i = 0; i < num_edges; ++i) {
    edge_lengths[i] = 1.0 - dist(gen);
  }

  const double d = 1.0;
  Filament filament = MakeClosedEndsFilament(edge_lengths, d);
  FilamentSelfContactFilter filter(filament);
  for (int i = 0; i < num_edges; ++i) {
    for (int j = i + 1; j < num_edges; ++j) {
      // ∑ₖ₌ᵢ₊₁ʲ⁻¹ edge_lengths[k].
      const double sum1 = std::accumulate(edge_lengths.cbegin() + i + 1,
                                          edge_lengths.cbegin() + j, 0.0);
      const double sum2 =
          std::accumulate(edge_lengths.cbegin(), edge_lengths.cend(), 0.0) -
          sum1 - edge_lengths[i] - edge_lengths[j];
      EXPECT_EQ(filter.ShouldCollide(i, j), sum1 > d && sum2 > d);
      EXPECT_EQ(filter.ShouldCollide(j, i), sum1 > d && sum2 > d);
    }
  }
}

}  // namespace
}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
