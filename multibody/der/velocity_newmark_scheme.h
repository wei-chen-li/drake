#pragma once

#include <array>
#include <memory>

#include "drake/multibody/der/discrete_time_integrator.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/* Implements the interface DiscreteTimeIntegrator with Newmark-beta time
 integration scheme. Given the value for the next time step velocity `vₙ₊₁`,
 the states are calculated from states from the previous time step according to
 the following equations:

        qₙ₊₁ = qₙ + δt ⋅ (β/γ ⋅ vₙ₊₁ +  (1 - β/γ) ⋅ vₙ) + δt² ⋅ (½ − β/γ) ⋅ aₙ
        aₙ₊₁ = (vₙ₊₁ - vₙ) / (δt ⋅ γ) - (1 − γ) / γ ⋅ aₙ.

 Note that the scheme is unconditionally unstable for gamma < 0.5 and therefore
 we require gamma >= 0.5.
 See AccelerationNewmarkScheme for the same integration scheme implemented with
 acceleration as the unknown variable.
 See [Newmark, 1959] for the original reference for the method.

 [Newmark, 1959] Newmark, Nathan M. "A method of computation for structural
 dynamics." Journal of the engineering mechanics division 85.3 (1959): 67-94.
 @tparam_default_scalar */
template <typename T>
class VelocityNewmarkScheme final : public DiscreteTimeIntegrator<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(VelocityNewmarkScheme);

  /* Constructs a Newmark scheme with velocity as the unknown variable.
   @pre dt > 0.
   @pre 0.5 <= gamma <= 1.
   @pre 0 <= beta <= 0.5. */
  VelocityNewmarkScheme(double dt, double gamma, double beta);

  ~VelocityNewmarkScheme() override;

 private:
  std::unique_ptr<DiscreteTimeIntegrator<T>> DoClone() const final;

  std::array<T, 3> DoGetWeights() const final;

  const Eigen::VectorX<T>& DoGetUnknowns(const DerState<T>& state) const final {
    return state.get_velocity();
  }

  void DoAdvanceOneTimeStep(const DerState<T>& prev_state,
                            const Eigen::Ref<const Eigen::VectorX<T>>& z,
                            DerState<T>* state) const final;

  void DoAdjustStateFromChangeInUnknowns(
      const Eigen::Ref<const Eigen::VectorX<T>>& dz,
      DerState<T>* state) const final;

  const double gamma_{};
  const double beta_over_gamma_{};
  const double one_over_dt_gamma_{};
};

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::VelocityNewmarkScheme);
