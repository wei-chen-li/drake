#include "drake/multibody/der/constraint_participation.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

ConstraintParticipation::ConstraintParticipation(bool has_closed_ends,
                                                 int num_nodes, int num_edges)
    : num_nodes_(num_nodes), num_edges_(num_edges) {
  DRAKE_THROW_UNLESS(num_nodes > 0);
  DRAKE_THROW_UNLESS(num_edges > 0);
  DRAKE_THROW_UNLESS(num_edges ==
                     (has_closed_ends ? num_nodes : num_nodes - 1));
}

void ConstraintParticipation::ParticipateEdgesAndAdjacentNodes(
    const std::unordered_set<int>& edges) {
  for (int i : edges) {
    DRAKE_THROW_UNLESS(0 <= i && i < num_edges_);
    participating_edges_.insert(i);
    /* The two adjacent nodes also participate in constraint. */
    participating_nodes_.insert(i);
    participating_nodes_.insert((i + 1) % num_nodes_);
  }
}

multibody::contact_solvers::internal::PartialPermutation
ConstraintParticipation::ComputeDofPermutation() const {
  const int num_dofs = num_nodes_ * 3 + num_edges_;
  std::vector<int> permuted_dof_indexes(num_dofs, -1);
  int permuted_dof_index = 0;
  for (int i = 0; i < num_nodes_; ++i) {
    if (participating_nodes_.contains(i)) {
      const int dof = 4 * i;
      permuted_dof_indexes[dof] = permuted_dof_index++;
      permuted_dof_indexes[dof + 1] = permuted_dof_index++;
      permuted_dof_indexes[dof + 2] = permuted_dof_index++;
    }
    if (i >= num_edges_) continue;
    if (participating_edges_.contains(i)) {
      const int dof = 4 * i + 3;
      permuted_dof_indexes[dof] = permuted_dof_index++;
    }
  }
  return multibody::contact_solvers::internal::PartialPermutation(
      std::move(permuted_dof_indexes));
}

bool ConstraintParticipation::no_participation() const {
  return participating_nodes_.empty() && participating_edges_.empty();
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
