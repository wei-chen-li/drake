#include "drake/multibody/der/discrete_time_integrator.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
std::array<T, 3> DiscreteTimeIntegrator<T>::GetWeights() const {
  return DoGetWeights();
}

template <typename T>
const Eigen::VectorX<T>& DiscreteTimeIntegrator<T>::GetUnknowns(
    const DerState<T>& state) const {
  return DoGetUnknowns(state);
}

template <typename T>
void DiscreteTimeIntegrator<T>::AdvanceOneTimeStep(
    const DerState<T>& prev_state, const Eigen::Ref<const Eigen::VectorX<T>>& z,
    DerState<T>* state) const {
  DRAKE_THROW_UNLESS(state != nullptr);
  DRAKE_THROW_UNLESS(prev_state.num_dofs() == state->num_dofs());
  DRAKE_THROW_UNLESS(prev_state.num_dofs() == z.size());
  DoAdvanceOneTimeStep(prev_state, z, state);
}

template <typename T>
void DiscreteTimeIntegrator<T>::AdjustStateFromChangeInUnknowns(
    const Eigen::Ref<const Eigen::VectorX<T>>& dz, DerState<T>* state) const {
  DRAKE_THROW_UNLESS(state != nullptr);
  DRAKE_THROW_UNLESS(dz.size() == state->num_dofs());
  DoAdjustStateFromChangeInUnknowns(dz, state);
}

template <typename T>
DiscreteTimeIntegrator<T>::DiscreteTimeIntegrator(double dt) : dt_(dt) {
  DRAKE_THROW_UNLESS(dt > 0);
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DiscreteTimeIntegrator);
