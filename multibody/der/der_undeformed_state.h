#pragma once

#include <vector>

#include "drake/multibody/der/der_state.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/*
 @p DerUndeformedState describes the undeformed state of a discrete elsatic
 rod, that is the state without external forces and under static equilibrium.
 For example, the undeformed state of a rope may be in a straight line (zero
 curvature and zero twist); or the undeformed state of a rubber band in a
 circular shape (constant curvature and zero twist).

 @tparam_default_scalar
 */
template <typename T>
class DerUndeformedState {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(DerUndeformedState);

  /* Creates an undeformed state where all edges have the same specified
   length, with zero curvature and zero twist. */
  static DerUndeformedState<T> ZeroCurvatureAndTwist(bool has_closed_ends,
                                                     int num_edges,
                                                     T every_edge_length);

  /* Creates an undeformed state with the specified edge lengths, zero
   curvature, and zero twist. */
  static DerUndeformedState<T> ZeroCurvatureAndTwist(
      bool has_closed_ends, std::vector<T> edge_length);

  /* Creates an undeformed state from the `state`'s current edge length,
   curvature, and twist. */
  static DerUndeformedState<T> FromCurrentDerState(const DerState<T>& state);

  bool has_closed_ends() const { return has_closed_ends_; }
  int num_edges() const { return edge_length_.size(); }
  int num_nodes() const {
    return has_closed_ends_ ? num_edges() : num_edges() + 1;
  }
  int num_internal_nodes() const {
    return has_closed_ends_ ? num_nodes() : num_nodes() - 2;
  }
  int num_dofs() const { return num_nodes() * 3 + num_edges(); }

  /*
   @name Property of edges when undeformed
   @{
   */
  const Eigen::RowVectorX<T>& get_edge_length() const { return edge_length_; }
  // @}

  /*
   @name Properties of internal nodes when undeformed
   @{
   */
  const Eigen::RowVectorX<T>& get_voronoi_length() const {
    return voronoi_length_;
  }

  const Eigen::RowVectorX<T>& get_curvature_kappa1() const { return kappa1_; }

  const Eigen::RowVectorX<T>& get_curvature_kappa2() const { return kappa2_; }

  const Eigen::RowVectorX<T>& get_twist() const { return twist_; }
  // @}

  template <typename U>
  DerUndeformedState<U> ToScalarType() const;

 private:
  template <typename U>
  friend class DerUndeformedState;

  DerUndeformedState(bool has_closed_ends, Eigen::RowVectorX<T> edge_length,
                     Eigen::RowVectorX<T> voronoi_length,
                     Eigen::RowVectorX<T> kappa1, Eigen::RowVectorX<T> kappa2,
                     Eigen::RowVectorX<T> twist);

  bool has_closed_ends_{};
  /* Properties associated with edges. */
  Eigen::RowVectorX<T> edge_length_;
  /* Properties associated with internal nodes. */
  Eigen::RowVectorX<T> voronoi_length_;
  Eigen::RowVectorX<T> kappa1_;
  Eigen::RowVectorX<T> kappa2_;
  Eigen::RowVectorX<T> twist_;
};

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DerUndeformedState);
