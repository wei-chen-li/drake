#include "drake/multibody/der/der_undeformed_state.h"

namespace drake {
namespace multibody {
namespace der {

namespace internal {

template <typename T>
DerUndeformedState<T> DerUndeformedState<T>::ZeroCurvatureAndTwist(
    bool has_closed_ends,
    const Eigen::Ref<const Eigen::RowVectorX<T>>& edge_length) {
  const int num_edges = edge_length.size();
  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  DRAKE_THROW_UNLESS(num_edges > 0);
  DRAKE_THROW_UNLESS((edge_length.array() > 0).all());

  auto kappa1 = Eigen::RowVectorX<T>::Zero(num_internal_nodes);
  auto kappa2 = Eigen::RowVectorX<T>::Zero(num_internal_nodes);
  auto twist = Eigen::RowVectorX<T>::Zero(num_internal_nodes);

  return DerUndeformedState<T>(has_closed_ends, edge_length, kappa1, kappa2,
                               twist);
}

template <typename T>
DerUndeformedState<T> DerUndeformedState<T>::FromCurrentDerState(
    const DerState<T>& state) {
  return DerUndeformedState<T>(state.has_closed_ends(), state.get_edge_length(),
                               state.get_curvature_kappa1(),
                               state.get_curvature_kappa2(), state.get_twist());
}

template <typename T>
void DerUndeformedState<T>::set_edge_length(
    const Eigen::Ref<const Eigen::RowVectorX<T>>& edge_length) {
  DRAKE_THROW_UNLESS(edge_length.size() == num_edges());
  DRAKE_THROW_UNLESS((edge_length.array() > 0).all());
  edge_length_ = edge_length;
  for (int i = 0; i < num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % num_edges();
    voronoi_length_[i] = 0.5 * (edge_length_[i] + edge_length_[ip1]);
  }
}

template <typename T>
void DerUndeformedState<T>::set_curvature_kappa(
    const Eigen::Ref<const Eigen::RowVectorX<T>>& kappa1,
    const Eigen::Ref<const Eigen::RowVectorX<T>>& kappa2) {
  DRAKE_THROW_UNLESS(kappa1.size() == num_internal_nodes());
  DRAKE_THROW_UNLESS(kappa2.size() == num_internal_nodes());
  kappa1_ = kappa1;
  kappa2_ = kappa2;
}

template <typename T>
void DerUndeformedState<T>::set_twist(
    const Eigen::Ref<const Eigen::RowVectorX<T>>& twist) {
  DRAKE_THROW_UNLESS(twist.size() == num_internal_nodes());
  twist_ = twist;
}

template <typename T>
DerUndeformedState<T>::DerUndeformedState(bool has_closed_ends,
                                          Eigen::RowVectorX<T> edge_length,
                                          Eigen::RowVectorX<T> kappa1,
                                          Eigen::RowVectorX<T> kappa2,
                                          Eigen::RowVectorX<T> twist)
    : has_closed_ends_(has_closed_ends),
      edge_length_(std::move(edge_length)),
      kappa1_(std::move(kappa1)),
      kappa2_(std::move(kappa2)),
      twist_(std::move(twist)) {
  DRAKE_THROW_UNLESS(num_edges() ==
                     (has_closed_ends_ ? num_nodes() : num_nodes() - 1));
  DRAKE_THROW_UNLESS(num_internal_nodes() ==
                     (has_closed_ends_ ? num_nodes() : num_nodes() - 2));
  DRAKE_THROW_UNLESS(edge_length_.size() == num_edges());
  DRAKE_THROW_UNLESS(kappa1_.size() == num_internal_nodes());
  DRAKE_THROW_UNLESS(kappa2_.size() == num_internal_nodes());
  DRAKE_THROW_UNLESS(twist_.size() == num_internal_nodes());

  voronoi_length_.resize(num_internal_nodes());
  for (int i = 0; i < num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % num_edges();
    voronoi_length_[i] = 0.5 * (edge_length_[i] + edge_length_[ip1]);
  }
}

template <typename T>
template <typename U>
DerUndeformedState<U> DerUndeformedState<T>::ToScalarType() const {
  static_assert(!std::is_same_v<T, U>);
  return DerUndeformedState<U>(
      has_closed_ends_, ExtractDoubleOrThrow(edge_length_),
      ExtractDoubleOrThrow(kappa1_), ExtractDoubleOrThrow(kappa2_),
      ExtractDoubleOrThrow(twist_));
}

using symbolic::Expression;
template DerUndeformedState<AutoDiffXd>
DerUndeformedState<double>::ToScalarType<AutoDiffXd>() const;
template DerUndeformedState<Expression>
DerUndeformedState<double>::ToScalarType<Expression>() const;
template DerUndeformedState<double>
DerUndeformedState<AutoDiffXd>::ToScalarType<double>() const;
template DerUndeformedState<Expression>
DerUndeformedState<AutoDiffXd>::ToScalarType<Expression>() const;
template DerUndeformedState<double>
DerUndeformedState<Expression>::ToScalarType<double>() const;
template DerUndeformedState<AutoDiffXd>
DerUndeformedState<Expression>::ToScalarType<AutoDiffXd>() const;

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DerUndeformedState);
