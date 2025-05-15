#include "drake/multibody/der/velocity_newmark_scheme.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
VelocityNewmarkScheme<T>::VelocityNewmarkScheme(double dt, double gamma,
                                                double beta)
    : DiscreteTimeIntegrator<T>(dt),
      gamma_(gamma),
      beta_over_gamma_(beta / gamma),
      one_over_dt_gamma_(1.0 / (dt * gamma)) {
  DRAKE_DEMAND(0.5 <= gamma && gamma <= 1);
  DRAKE_DEMAND(0 <= beta && beta <= 0.5);
}

template <typename T>
VelocityNewmarkScheme<T>::~VelocityNewmarkScheme() = default;

template <typename T>
std::unique_ptr<DiscreteTimeIntegrator<T>> VelocityNewmarkScheme<T>::DoClone()
    const {
  return std::make_unique<VelocityNewmarkScheme<T>>(this->dt(), gamma_,
                                                    beta_over_gamma_ * gamma_);
}

template <typename T>
std::array<T, 3> VelocityNewmarkScheme<T>::DoGetWeights() const {
  const double dt = this->dt();
  return {beta_over_gamma_ * dt, 1.0, one_over_dt_gamma_};
}

template <typename T>
void VelocityNewmarkScheme<T>::DoAdvanceOneTimeStep(
    const DerState<T>& prev_state, const Eigen::Ref<const Eigen::VectorX<T>>& z,
    DerState<T>* state) const {
  const double dt = this->dt();
  const Eigen::VectorX<T>& an = prev_state.get_acceleration();
  const Eigen::VectorX<T>& vn = prev_state.get_velocity();
  const Eigen::VectorX<T>& qn = prev_state.get_position();
  const Eigen::Ref<const Eigen::VectorX<T>>& v = z;
  /* Make `state` a copy of `prev_state` . This needs to be done before
   advancing `state`. */
  state->CopyFrom(prev_state);
  /* Note that the partials of the next time step's (q, v, a) w.r.t. z are
   (β*δt/γ, 1, 1/(δt*γ)), and they must match the weights given by
   DoGetWeights(). */
  state->AdvancePositionToNextStep(
      qn + dt * (beta_over_gamma_ * v + (1.0 - beta_over_gamma_) * vn) +
      dt * dt * (0.5 - beta_over_gamma_) * an);
  state->SetAcceleration(one_over_dt_gamma_ * (v - vn) -
                         (1.0 - gamma_) / gamma_ * an);
  state->SetVelocity(v);
}

template <typename T>
void VelocityNewmarkScheme<T>::DoAdjustStateFromChangeInUnknowns(
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
    class ::drake::multibody::der::internal::VelocityNewmarkScheme);
