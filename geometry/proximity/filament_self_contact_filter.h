#pragma once

#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

/* FilamentSelfContactFilter provides logic to report if two edges of the same
 filament should be checked for collision.

 Given a reference filament with reference edge lengths. Edge i and edge j
 (assuming i ≤ j) should be checked for collision if and only if

    ∑ₖ₌ᵢ₊₁ʲ⁻¹ reference_edge_lengths[k] > C,

 where C is a characteristic diameter of the filament. From this definition, it
 is obvious that adjacent edges are never checked for collision; their
 interactions are instead governed by internal elastic forces due to bending. */
class FilamentSelfContactFilter {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FilamentSelfContactFilter)

  /* Constructs a self-contact filter.
   @param closed                 Flag indicating if the filament is closed.
   @param reference_edge_lengths The reference edge lengths of the filament.
   @param C                      A characteristic diameter of the filament. */
  FilamentSelfContactFilter(
      bool closed,
      const Eigen::Ref<const Eigen::RowVectorXd>& reference_edge_lengths,
      double C);

  /* Returns true if edge i and edge j is a candidate pair to be checked for
   collision. */
  bool ShouldCollide(int edge_i, int edge_j) const;

 private:
  /* Whether the filament is closed. */
  bool closed_;
  /* Number of edges of the filament. */
  int num_edges_;
  /* The edges (i,j) with i ≤ j is a candidate pair to be checked for collision
   if j ≥ i_to_smallest_j[i] > i. */
  std::vector<int> i_to_smallest_j_;
};

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
