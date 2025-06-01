#include "drake/geometry/proximity/filament_self_contact_filter.h"

#include <limits>

#include "drake/common/overloaded.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

FilamentSelfContactFilter::FilamentSelfContactFilter(const Filament& filament)
    : closed_(filament.closed()), num_edges_(filament.edge_m1().cols()) {
  i_to_smallest_j_.resize(num_edges_, std::numeric_limits<int>::max());
  if (num_edges_ <= 2) return;

  std::vector<double> edge_lengths(num_edges_);
  for (int i = 0; i < num_edges_; ++i) {
    const int ip1 = (i + 1) % filament.node_pos().cols();
    edge_lengths[i] =
        (filament.node_pos().col(ip1) - filament.node_pos().col(i)).norm();
  }
  const double d =
      std::visit(overloaded{[](const Filament::CircularCrossSection& cs) {
                              return cs.diameter;
                            },
                            [](const Filament::RectangularCrossSection& cs) {
                              return std::hypot(cs.width, cs.height);
                            }},
                 filament.cross_section());

  int j = 1;
  double sum = 0;  // sum = ∑ₖ₌ᵢ₊₁ʲ⁻¹ edge_lengths[k].
  for (int i = 0; i < num_edges_; ++i) {
    if (i != 0) {
      sum -= edge_lengths.at(i);
    }
    while (d >= sum) {
      if (!closed_ && j == num_edges_) break;
      sum += edge_lengths.at(j++ % num_edges_);
    }
    if (d < sum)
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
