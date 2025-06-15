#pragma once

#include <memory>
#include <unordered_set>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/multibody/contact_solvers/block_sparse_cholesky_solver.h"
#include "drake/multibody/der/der_model.h"
#include "drake/multibody/der/discrete_time_integrator.h"
#include "drake/multibody/der/energy_hessian_matrix_utility.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/*
 @p DerSolver solves the dynamic discrete elastic rod (DER) problem. The
 governing PDE of the dynamics is spatially discretized by DerModel and
 temporally discretized by DiscreteTimeIntegrator. DerSolver provides the
 `AdvanceOneTimeStep()` function that advances the free-motion states (i.e.
 without considering contacts or constraints) of the spatially discretized DER
 model by one time step according to the prescribed discrete time integration
 scheme.

 @tparam_double_only
 */
template <typename T>
class DerSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DerSolver);

  /* Constructs a DerSolver that solves the given `model` with the `integrator`
   provided to advance time.
   @note The `model` and `integrator` must outlive this DerSolver.
   @pre `model != nullptr`.
   @pre `integrator != nullptr`. */
  DerSolver(const DerModel<T>* model,
            const DiscreteTimeIntegrator<T>* integrator);

  /* Advances `prev_state` by one time step to the internally owned DerState
   so that M q̈ - Fᵢₙₜ(q, q̇) - Fₑₓₜ ≈ 0 is satisfied at the new time step.

   @param[in] prev_state The state at the previous time step.
   @param[in] external_force_field The external force field to evaluate under.
   @param[in] generalized_external_force Optional. A generalized external force
                                         additional to the external force field.
   @returns the number of iterations the solver takes to converge.

   @pre `prev_state` is created from the model prescribed at construction.
   @pre `generalized_external_force` has size 0 or `model->num_dofs()`.
   @throws std::expection if the solver fails to factorize the tangent matrix or
   the solver fails to converge. */
  int AdvanceOneTimeStep(const DerState<T>& prev_state,
                         const ExternalForceField<T>& external_force_field,
                         const Eigen::Ref<const VectorX<T>>&
                             generalized_external_force = VectorX<T>());

  /* Sets the internally owned DerState to `state`.
   @pre `state` is created from the model prescribed at construction. */
  void set_state(const DerState<T>& state);

  /* Returns the internally owned DerState. */
  const DerState<T>& get_state() const { return *state_; }

  /* Computes the Schur complement of the tangent matrix evaluated at the
   internally owned state.
   @param[in] participating_dofs The DoFs of the DER model that participate in
                                 constraint.
   @pre All DoF indices in `participating_dofs` are greater than or equal to
        zero and less thant num_dofs(). */
  void ComputeTangentMatrixSchurComplement(
      const std::unordered_set<int>& participating_dofs);

  /* Returns the internally owned tangent matrix Schur complement. */
  const SchurComplement<T>& get_tangent_matrix_schur_complement() const {
    return tangent_matrix_schur_complement_;
  }

  /* Sets the relative tolerance, unitless. See solver_converged() for how
   the tolerance is used. The default value is 1e-4. */
  void set_relative_tolerance(double tolerance) {
    relative_tolerance_ = tolerance;
  }
  double relative_tolerance() const { return relative_tolerance_; }

  /* Sets the absolute tolerance with unit Newton. See solver_converged() for
   how the tolerance is used. The default value is 1e-6. */
  void set_absolute_tolerance(double tolerance) {
    absolute_tolerance_ = tolerance;
  }
  double absolute_tolerance() const { return absolute_tolerance_; }

  /* Returns a copy of this DerSolver. */
  std::unique_ptr<DerSolver<T>> Clone() const;

 private:
  /* Friend class to facilitate testing. */
  friend class DerSolverTester;

  /* Scales the edge angle DoFs of `residual`, which has unit N⋅m, by the
   inverse of a characteric radius so that all entries have unit N. Then returns
   the 2-norm of the unit adjusted vector. */
  T unit_adjusted_norm(const Eigen::Ref<const Eigen::VectorX<T>>& residual);

  /* The solver is considered as converged if ‖r‖ < max(εᵣ * ‖r₀‖, εₐ) where
    r and r₀ are `residual_norm` and `initial_residual_norm` respectively,
    and εᵣ and εₐ are relative and absolute tolerance respectively. */
  bool solver_converged(const T& residual_norm,
                        const T& initial_residual_norm) const;

  /* Pointer to DerModel and DiscreteTimeIntegrator set at construction. */
  const DerModel<T>* const model_{};
  const DiscreteTimeIntegrator<T>* const integrator_{};
  /* Owned DerState. */
  copyable_unique_ptr<internal::DerState<T>> state_;
  /* Owned SchurComplement */
  SchurComplement<T> tangent_matrix_schur_complement_;
  /* Tolerance for convergence. */
  double relative_tolerance_{1e-4};  // unitless.
  double absolute_tolerance_{1e-6};  // unit N.
  /* Max number of Newton-Raphson iterations before giving up. */
  int max_iterations_{100};
  /* Instance of struct holding preallocated memory. */
  struct Scratch {
    std::unique_ptr<typename DerModel<T>::Scratch,
                    typename DerModel<T>::ScratchDeleter>
        der_model_scratch;
    contact_solvers::internal::BlockSparseCholeskySolver<Matrix4<T>>
        linear_solver;
    Eigen::VectorX<T> b;
  } scratch_;
};

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
