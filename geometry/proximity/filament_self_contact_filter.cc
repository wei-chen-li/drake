#include "drake/geometry/proximity/filament_self_contact_filter.h"

#include <limits>

#include "drake/common/drake_throw.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

FilamentSelfContactFilter::FilamentSelfContactFilter(
    bool closed,
    const Eigen::Ref<const Eigen::RowVectorXd>& reference_edge_lengths,
    double C)
    : closed_(closed), num_edges_(reference_edge_lengths.size()) {
  i_to_smallest_j_.resize(num_edges_, std::numeric_limits<int>::max());
  if (num_edges_ <= 2) return;

  int j = 1;
  double sum = 0;  // sum = ∑ₖ₌ᵢ₊₁ʲ⁻¹ edge_lengths[k].
  for (int i = 0; i < num_edges_; ++i) {
    if (i != 0) {
      sum -= reference_edge_lengths[i];
    }
    while (C >= sum) {
      if (!closed_ && j == num_edges_) break;
      sum += reference_edge_lengths[j++ % num_edges_];
    }
    if (C < sum)
      i_to_smallest_j_[i] = j;
    else
      break;
  }
}

bool FilamentSelfContactFilter::ShouldCollide(int i, int j) const {
  DRAKE_THROW_UNLESS(0 <= i && i < num_edges_);
  DRAKE_THROW_UNLESS(0 <= j && j < num_edges_);
  if (i > j) std::swap(i, j);
  if (!closed_) {
    return j >= i_to_smallest_j_[i];
  } else {
    return (j >= i_to_smallest_j_[i]) &&
           (i + num_edges_ >= i_to_smallest_j_[j]);
  }
}

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
