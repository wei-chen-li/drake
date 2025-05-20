#pragma once

#include <array>
#include <memory>

#include "drake/multibody/der/der_state.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/* DiscreteTimeIntegrator is an abstract class that encapsulates discrete time
 integrations schemes for second order ODEs. When a second order ODE

     f(q, v = q̇, a = q̈) = 0

 is discretized in time, the quantities of interest evaluated at the next time
 step can often be expressed as an affine mapping on a single variable z, i.e.

        qₙ₊₁ = αₚ z + bₚ
        vₙ₊₁ = αᵥ z + bᵥ
        aₙ₊₁ = αₐ z + bₐ

 For example, for the Newmark-beta scheme, where

        qₙ₊₁ = qₙ + δt ⋅ vₙ + δt² ⋅ ((½ − β) ⋅ aₙ + β ⋅ aₙ₊₁)
        vₙ₊₁ = vₙ + δt ⋅ ((1 − γ) ⋅ aₙ + γ ⋅ aₙ₊₁)
        aₙ₊₁ = f(qₙ₊₁,vₙ₊₁),

 if one chooses z = a, we get

        qₙ₊₁ = qₙ + δt ⋅ vₙ + δt² ⋅ (β ⋅ z + (½ - β) ⋅ aₙ)
        vₙ₊₁ = vₙ + δt ⋅ (γ ⋅ z + (1−γ) ⋅ aₙ)
        aₙ₊₁ = z;

 On the other hand, if one chooses z = v instead for the same scheme, we get

        qₙ₊₁ = qₙ + δt ⋅ (β/γ ⋅ z +  (1 - β/γ) ⋅ vₙ) + δt² ⋅ (½ − β/γ) ⋅ aₙ
        vₙ₊₁ = z
        aₙ₊₁ = (z - vₙ) / (δt ⋅ γ) - (1 − γ) / γ ⋅ aₙ.

 DiscreteTimeIntegrator provides the interface to query the relationship between
 the states (q, v = q̇, a = q̈) and the unknown variable z.
 @tparam_default_scalar */
template <typename T>
class DiscreteTimeIntegrator {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DiscreteTimeIntegrator);

  virtual ~DiscreteTimeIntegrator() = default;

  /* Returns an identical copy of `this` DiscreteTimeIntegrator. */
  std::unique_ptr<DiscreteTimeIntegrator<T>> Clone() const { return DoClone(); }

  /* Returns (αₚ, αᵥ, αₐ), the derivative of (q, v, a) with respect to the
   unknown variable z (See class documentation). These weights can be used to
   combine stiffness, damping, and mass matrices to form the tangent
   matrix (see DerModel::ComputeTangentMatrix). */
  std::array<T, 3> GetWeights() const;

  /* Extracts the unknown variable `z` from the given DER `state`. */
  const Eigen::VectorX<T>& GetUnknowns(const DerState<T>& state) const;

  /* Advances `prev_state` by one time step to the `state` with the given value
   of the unknown variable z.

   @param[in]  prev_state  The state at the previous time step. Need this
                           because `state` cannot be modified in-place.
   @param[in]  z           The value of the unknown variable z.
   @param[out] state       The result after advancing the next time step.

   @pre `next_state != nullptr`.
   @pre The sizes of `prev_state`, `z`, and `state` are compatible. */
  void AdvanceOneTimeStep(const DerState<T>& prev_state,
                          const Eigen::Ref<const Eigen::VectorX<T>>& z,
                          DerState<T>* state) const;

  /* Adjusts the DerState `state` given the change in the unknown variables.
   More specifically, it sets the given `state` to the following values.

        q = αₚ (z + dz) + bₚ
        v = αᵥ (z + dz) + bᵥ
        a = αₐ (z + dz) + bₐ

   @pre `state != nullptr`.
   @pre `dz.size() == state->num_dofs()`. */
  void AdjustStateFromChangeInUnknowns(
      const Eigen::Ref<const Eigen::VectorX<T>>& dz, DerState<T>* state) const;

  /* Returns the discrete time step of the integration scheme. */
  double dt() const { return dt_; }

 protected:
  explicit DiscreteTimeIntegrator(double dt);

  /* Derived classes must override this method to implement the NVI
   DoClone(). */
  virtual std::unique_ptr<DiscreteTimeIntegrator<T>> DoClone() const = 0;

  /* Derived classes must override this method to implement the NVI
   GetWeights(). */
  virtual std::array<T, 3> DoGetWeights() const = 0;

  /* Derived classes must override this method to implement the NVI
   GetUnknowns(). */
  virtual const Eigen::VectorX<T>& DoGetUnknowns(
      const DerState<T>& state) const = 0;

  /* Derived classes must override this method to implement the NVI
   AdvanceOneTimeStep(). */
  virtual void DoAdvanceOneTimeStep(
      const DerState<T>& prev_state,
      const Eigen::Ref<const Eigen::VectorX<T>>& z,
      DerState<T>* next_state) const = 0;

  /* Derived classes must override this method to implement the NVI
   AdjustStateFromChangeInUnknowns(). */
  virtual void DoAdjustStateFromChangeInUnknowns(
      const Eigen::Ref<const Eigen::VectorX<T>>& dz,
      DerState<T>* state) const = 0;

  double dt_{0.0};
};

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DiscreteTimeIntegrator);
