#include "drake/multibody/der/acceleration_newmark_scheme.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
AccelerationNewmarkScheme<T>::AccelerationNewmarkScheme(double dt, double gamma,
                                                        double beta)
    : DiscreteTimeIntegrator<T>(dt), gamma_(gamma), beta_(beta) {
  DRAKE_THROW_UNLESS(0.5 <= gamma && gamma <= 1);
  DRAKE_THROW_UNLESS(0 <= beta && beta <= 0.5);
}

template <typename T>
AccelerationNewmarkScheme<T>::~AccelerationNewmarkScheme() = default;

template <typename T>
std::unique_ptr<DiscreteTimeIntegrator<T>>
AccelerationNewmarkScheme<T>::DoClone() const {
  return std::make_unique<AccelerationNewmarkScheme<T>>(this->dt(), gamma_,
                                                        beta_);
}

template <typename T>
std::array<T, 3> AccelerationNewmarkScheme<T>::DoGetWeights() const {
  const double dt = this->dt();
  return {beta_ * dt * dt, gamma_ * dt, 1.0};
}

template <typename T>
void AccelerationNewmarkScheme<T>::DoAdvanceOneTimeStep(
    const DerState<T>& prev_state, const Eigen::Ref<const Eigen::VectorX<T>>& z,
    DerState<T>* state) const {
  const double dt = this->dt();
  const Eigen::VectorX<T>& an = prev_state.get_acceleration();
  const Eigen::VectorX<T>& vn = prev_state.get_velocity();
  const Eigen::VectorX<T>& qn = prev_state.get_position();
  const Eigen::Ref<const Eigen::VectorX<T>>& a = z;
  /* Make `state` a copy of `prev_state` . This needs to be done before
   advancing `state`. */
  state->CopyFrom(prev_state);
  /* Note that the partials of the next time step's (q, v, a) w.r.t. z are
   (δt²*β, δt*γ, 1), and they must match the weights given by DoGetWeights(). */
  state->AdvancePositionToNextStep(qn + dt * vn +
                                   dt * dt * (beta_ * a + (0.5 - beta_) * an));
  state->SetVelocity(vn + dt * (gamma_ * a + (1.0 - gamma_) * an));
  state->SetAcceleration(a);
}

template <typename T>
void AccelerationNewmarkScheme<T>::DoAdjustStateFromChangeInUnknowns(
    const Eigen::Ref<const Eigen::VectorX<T>>& dz, DerState<T>* state) const {
  const Eigen::VectorX<T>& a = state->get_acceleration();
  const Eigen::VectorX<T>& v = state->get_velocity();
  const Eigen::VectorX<T>& q = state->get_position();
  const std::array<T, 3> weights = this->GetWeights();
  state->AdjustPositionWithinStep(q + weights[0] * dz);
  state->SetVelocity(v + weights[1] * dz);
  state->SetAcceleration(a + weights[2] * dz);
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::AccelerationNewmarkScheme);
