#pragma once

#include <unordered_set>
#include <vector>

#include "drake/multibody/contact_solvers/sap/partial_permutation.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

class ConstraintParticipation {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ConstraintParticipation);

  /* Constructs a ConstraintParticipation for a discrete elastic rod/filament
   geometry.
   @pre `num_nodes > 0`.
   @pre `num_edges > 0`.
   @pre `num_edges == (has_closed_ends ? num_nodes : num_nodes-1)`. */
  ConstraintParticipation(bool has_closed_ends, int num_nodes, int num_edges);

  /* Mark the given edges and the nodes adjacent to the edges as participating
   in constraint.
   @pre Each entry in `edges` is non-negative and less than `num_edges` provided
   at construction. */
  void ParticipateEdgesAndAdjacentNodes(const std::unordered_set<int>& edges);

  /* Returns the DoF partial permutation. The DoF partial permutation p is such
   that p(i) gives the permuted DoF index for DoF i, if DoF i is participating
   in constraint. If both DoF i and DoF j are participating in contact,
   and i < j, then p(i) < p(j). */
  multibody::contact_solvers::internal::PartialPermutation
  ComputeDofPermutation() const;

  /* Returns true if no nodes and no edges participate in constraint. */
  bool no_participation() const;

 private:
  int num_nodes_;
  int num_edges_;
  std::unordered_set<int> participating_nodes_;
  std::unordered_set<int> participating_edges_;
};

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
