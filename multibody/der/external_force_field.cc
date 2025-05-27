#include "drake/multibody/der/external_force_field.h"

#include <array>

namespace drake {
namespace multibody {
namespace der {
namespace internal {

namespace {

struct QuadraturePair {
  double abscissa;
  double weight;

  constexpr QuadraturePair(double abscissa_in, double weight_in)
      : abscissa(abscissa_in), weight(weight_in) {}
};

const std::array<QuadraturePair, 5>& GaussQuadrature() {
  /* Use Gauss–Lobatto quadrature so that end points are included. */
  static constexpr std::array<QuadraturePair, 5> quadrature_pairs = {
      QuadraturePair{0.0, 32 / 45.0},
      QuadraturePair{-0.654653670707977, 49 / 90.0},
      QuadraturePair{+0.654653670707977, 49 / 90.0},
      QuadraturePair{-1.0, 1 / 10.0},  //
      QuadraturePair{+1.0, 1 / 10.0}};
  return quadrature_pairs;
}

}  // namespace

template <typename T>
ExternalForceField<T>::ExternalForceField(
    const systems::Context<T>* plant_context,
    std::vector<const ForceDensityField<T>*> force_density_fields)
    : plant_context_(plant_context),
      force_density_fields_(std::move(force_density_fields)) {
  DRAKE_THROW_UNLESS(plant_context != nullptr);
  for (int i = 0; i < ssize(force_density_fields_); ++i)
    DRAKE_THROW_UNLESS(force_density_fields_[i] != nullptr);
}

template <typename T>
ExternalForceVector<T> ExternalForceField<T>::operator()(
    const DerStructuralProperty<T>& prop,
    const DerUndeformedState<T>& undeformed, const DerState<T>& state) const {
  return ExternalForceVector<T>(plant_context_, &force_density_fields_, &prop,
                                &undeformed, &state);
}

template <typename T>
ExternalForceVector<T>::ExternalForceVector(
    const systems::Context<T>* plant_context,
    const std::vector<const ForceDensityField<T>*>* force_density_fields,
    const DerStructuralProperty<T>* prop,
    const DerUndeformedState<T>* undeformed, const DerState<T>* state)
    : plant_context_(plant_context),
      force_density_fields_(force_density_fields),
      prop_(prop),
      undeformed_(undeformed),
      state_(state) {
  DRAKE_THROW_UNLESS(plant_context != nullptr);
  DRAKE_THROW_UNLESS(force_density_fields != nullptr);
  for (int i = 0; i < ssize(*force_density_fields); ++i)
    DRAKE_THROW_UNLESS((*force_density_fields)[i] != nullptr);
  DRAKE_THROW_UNLESS(prop != nullptr);
  DRAKE_THROW_UNLESS(undeformed != nullptr);
  DRAKE_THROW_UNLESS(state != nullptr);
  DRAKE_THROW_UNLESS(undeformed->has_closed_ends() == state->has_closed_ends());
  DRAKE_THROW_UNLESS(undeformed->num_nodes() == state->num_nodes());
}

template <typename T>
Eigen::VectorX<T> ExternalForceVector<T>::eval() const {
  Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(state_->num_dofs());
  this->ScaleAndAddToVector(1.0, &result);
  return result;
}

template <typename T>
Eigen::Ref<Eigen::VectorX<T>> ExternalForceVector<T>::ScaleAndAddToVector(
    const T& scale, EigenPtr<Eigen::VectorX<T>> other) const {
  DRAKE_THROW_UNLESS(other != nullptr);
  DRAKE_THROW_UNLESS(other->size() == state_->num_dofs());

  auto& q = state_->get_position();
  auto& t = state_->get_tangent();
  auto& l = state_->get_edge_length();
  auto& l_undeformed = undeformed_->get_edge_length();

  for (const ForceDensityField<T>* force_density_field :
       *force_density_fields_) {
    for (int i = 0; i < state_->num_edges(); ++i) {
      const int ip1 = (i + 1) % state_->num_nodes();

      Eigen::Vector3<T> rod_i_force = Eigen::Vector3<T>::Zero();
      Eigen::Vector3<T> rod_i_moment = Eigen::Vector3<T>::Zero();
      for (const QuadraturePair& pair : GaussQuadrature()) {
        Eigen::Vector3<T> vec = pair.abscissa * 0.5 * l[i] * t.col(i);
        Eigen::Vector3<T> point = 0.5 * (q.template segment<3>(4 * i) +
                                         q.template segment<3>(4 * ip1)) +
                                  vec;
        Eigen::Vector3<T> force =
            force_density_field->EvaluateAt(*plant_context_, point);

        rod_i_force += pair.weight * force;
        rod_i_moment += pair.weight * vec.cross(force);
      }
      /* Multiply by the volume to convert force density to force. Divide
       by 2 because the Gauss quadrature weights sum to 2. */
      rod_i_force *= prop_->A() * l_undeformed[i] * 0.5;
      rod_i_moment *= prop_->A() * l_undeformed[i] * 0.5;

      auto node_i_force = other->template segment<3>(4 * i);
      auto node_ip1_force = other->template segment<3>(4 * ip1);
      /* Force on the rod is distributed to the two nodes. */
      node_i_force += (0.5 * rod_i_force) * scale;
      node_ip1_force += (0.5 * rod_i_force) * scale;
      /* Moment on the rod (which is perpendicular to the axis of the rod) is
       converted to an equivalent force couple and applied to the two nodes. */
      node_i_force -= (rod_i_moment.cross(t.col(i)) / l[i]) * scale;
      node_ip1_force += (rod_i_moment.cross(t.col(i)) / l[i]) * scale;
    }
  }
  return *other;
}

template <typename T>
Eigen::DiagonalMatrix<T, Eigen::Dynamic> ComputeMassMatrix(
    const DerStructuralProperty<T>& prop,
    const DerUndeformedState<T>& undeformed) {
  Eigen::VectorX<T> generalized_mass(undeformed.num_dofs());
  auto& l_undeformed = undeformed.get_edge_length();
  const int num_edges = undeformed.num_edges();

  /* The generalized mass assiciated with the node position DoF is half the
   total mass of the adjacent rod(s). Two rods if the node is on the internal,
   only one rod if the DER has open-ends and the node is on either end. */
  for (int i = 0; i < undeformed.num_nodes(); ++i) {
    T effective_length = 0;
    if (undeformed.has_closed_ends()) {
      const int im1 = (i - 1 + num_edges) % num_edges;
      effective_length = 0.5 * (l_undeformed[im1] + l_undeformed[i]);
    } else {
      if (i - 1 >= 0) effective_length += 0.5 * l_undeformed[i - 1];
      if (i < num_edges) effective_length += 0.5 * l_undeformed[i];
    }
    generalized_mass.template segment<3>(4 * i).setConstant(effective_length *
                                                            prop.rhoA());
  }

  /* The generalized mass associated with the edge angle DoF is the axial
   rotational inertial of the rod. */
  for (int i = 0; i < num_edges; ++i) {
    generalized_mass(4 * i + 3) = l_undeformed[i] * prop.rhoJ();
  }

  return generalized_mass.asDiagonal();
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::ExternalForceField);

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::ExternalForceVector);

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (&::drake::multibody::der::internal::ComputeMassMatrix<T>));
