#include "drake/multibody/der/constraint_participation.h"

#include <algorithm>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using multibody::contact_solvers::internal::PartialPermutation;
using testing::UnorderedElementsAreArray;

class ConstraintParticipationTest : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds, ConstraintParticipationTest,
                         ::testing::Values(false, true));

static void CheckConstraintParticipatingDofs(
    const ConstraintParticipation& constraint_participation, int num_dofs,
    const std::unordered_set<int>& participating_dofs) {
  EXPECT_THAT(constraint_participation.ComputeParticipatingDofs(),
              UnorderedElementsAreArray(participating_dofs));

  PartialPermutation dof_permutation =
      constraint_participation.ComputeDofPermutation();
  EXPECT_EQ(dof_permutation.domain_size(), num_dofs);

  int permuted_index = 0;
  for (int i = 0; i < num_dofs; ++i) {
    EXPECT_EQ(dof_permutation.participates(i), participating_dofs.contains(i));
    if (dof_permutation.participates(i)) {
      EXPECT_EQ(dof_permutation.permuted_index(i), permuted_index++);
    }
  }
}

TEST_P(ConstraintParticipationTest, ParticipateEdgesAndAdjacentNodes) {
  const bool has_closed_ends = GetParam();
  const int num_edges = 4;
  const int num_nodes = has_closed_ends ? num_edges : num_edges + 1;
  const int num_dofs = num_nodes * 3 + num_edges;
  ConstraintParticipation constraint_participation(has_closed_ends, num_nodes,
                                                   num_edges);

  constraint_participation.ParticipateEdgesAndAdjacentNodes({1, 3});

  std::unordered_set<int> participating_dofs;
  if (!has_closed_ends) {
    participating_dofs.insert({4, 5, 6, 7, 8, 9, 10});
    participating_dofs.insert({12, 13, 14, 15, 16, 17, 18});
  } else {
    participating_dofs.insert({4, 5, 6, 7, 8, 9, 10});
    participating_dofs.insert({12, 13, 14, 15, 0, 1, 2});
  }
  CheckConstraintParticipatingDofs(constraint_participation, num_dofs,
                                   participating_dofs);
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
