#include "drake/multibody/der/der_solver.h"

#include <algorithm>

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
DerSolver<T>::DerSolver(const DerModel<T>* model,
                        const DiscreteTimeIntegrator<T>* integrator)
    : model_(model), integrator_(integrator) {
  DRAKE_THROW_UNLESS(model != nullptr);
  DRAKE_THROW_UNLESS(integrator != nullptr);
  state_ = model_->CreateDerState();
  /* Allocate the scratch for the DerModel. */
  scratch_.der_model_scratch = model_->MakeScratch();
  /* Tell BlockSparseCholeskySolver the block sparsity pattern. */
  const Block4x4SparseSymmetricMatrix<T>& A = model_->ComputeTangentMatrix(
      *state_, integrator_->GetWeights(), scratch_.der_model_scratch.get());
  scratch_.linear_solver.SetMatrix(A);
  scratch_.prev_is_contact_energy_enabled = model_->IsContactEnergyEnabled();
  /* Set b to size that matches the tangent matrix. */
  scratch_.b = Eigen::VectorX<T>::Zero(A.rows());
}

template <typename T>
int DerSolver<T>::AdvanceOneTimeStep(
    const DerState<T>& prev_state,
    const ExternalForceField<T>& external_force_field) {
  model_->ValidateDerState(prev_state);

  DerState<T>& state = *state_;
  typename DerModel<T>::Scratch* der_model_scratch =
      scratch_.der_model_scratch.get();
  contact_solvers::internal::BlockSparseCholeskySolver<Matrix4<T>>&
      linear_solver = scratch_.linear_solver;
  Eigen::VectorX<T>& b = scratch_.b;
  const int num_dofs = model_->num_dofs();
  DRAKE_DEMAND(der_model_scratch != nullptr);
  DRAKE_DEMAND(b.size() ==
               (model_->has_closed_ends() ? num_dofs : num_dofs + 1));

  const Eigen::VectorX<T>& z = integrator_->GetUnknowns(prev_state);
  integrator_->AdvanceOneTimeStep(prev_state, z, &state);
  model_->ApplyBoundaryCondition(&state);
  b.head(num_dofs) =
      -model_->ComputeResidual(state, external_force_field, der_model_scratch);
  T residual_norm = unit_adjusted_norm(b);
  const T initial_residual_norm = residual_norm;
  int iter = 0;
  /* For Newton-Raphson solver, we iterate until any of the following is true:
   1. The max number of allowed iterations is reached;
   2. The norm of the residual is smaller than the absolute tolerance.
   3. The relative error (the norm of the residual divided by the norm of the
      initial residual) is smaller than the dimensionless relative tolerance. */
  while (iter < max_iterations_ &&
         !solver_converged(residual_norm, initial_residual_norm)) {
    const internal::Block4x4SparseSymmetricMatrix<T>& tangent_matrix =
        model_->ComputeTangentMatrix(state, integrator_->GetWeights(),
                                     der_model_scratch);
    /* If the contact energy is not enabled, the sparsity pattern of the tangent
     matrix does not change and thus we call UpdateMatrix(). */
    if (!model_->IsContactEnergyEnabled()) {
      if (scratch_.prev_is_contact_energy_enabled)
        linear_solver.SetMatrix(tangent_matrix);
      else
        linear_solver.UpdateMatrix(tangent_matrix);
      scratch_.prev_is_contact_energy_enabled = false;
    } else {
      linear_solver.SetMatrix(tangent_matrix);
      scratch_.prev_is_contact_energy_enabled = true;
    }
    const bool factored = linear_solver.Factor();
    if (!factored) {
      throw std::runtime_error(
          "Tangent matrix factorization failed in DerSolver because the DER "
          "tangent matrix is not symmetric positive definite. This may be "
          "triggered by a combination of a stiff constitutive model and a "
          "large timestep.");
    }
    linear_solver.SolveInPlace(&b);
    auto dz = b.head(num_dofs);
    integrator_->AdjustStateFromChangeInUnknowns(dz, &state);
    b.head(num_dofs) = -model_->ComputeResidual(state, external_force_field,
                                                der_model_scratch);
    residual_norm = unit_adjusted_norm(b);
    ++iter;
  }
  if (!solver_converged(residual_norm, initial_residual_norm)) {
    throw std::runtime_error(
        "DerSolver::AdvanceOneTimeStep() failed to converge. Consider using a "
        "smaller timestep or reduce the stiffness of the material.");
  }
  return iter;
}

template <typename T>
void DerSolver<T>::ComputeTangentMatrixSchurComplement(
    const std::unordered_set<int>& participating_dofs) {
  DRAKE_THROW_UNLESS(std::all_of(participating_dofs.begin(),
                                 participating_dofs.end(), [&](int dof) {
                                   return 0 <= dof && dof < model_->num_dofs();
                                 }));
  DerState<T>& state = *state_;
  typename DerModel<T>::Scratch* der_model_scratch =
      scratch_.der_model_scratch.get();
  DRAKE_DEMAND(der_model_scratch != nullptr);

  const internal::Block4x4SparseSymmetricMatrix<T>& tangent_matrix =
      model_->ComputeTangentMatrix(state, integrator_->GetWeights(),
                                   der_model_scratch);
  tangent_matrix_schur_complement_ =
      ComputeSchurComplement(tangent_matrix, participating_dofs);
}

template <typename T>
void DerSolver<T>::set_state(const DerState<T>& state) {
  model_->ValidateDerState(state);
  state_->CopyFrom(state);
}

template <typename T>
T DerSolver<T>::unit_adjusted_norm(
    const Eigen::Ref<const Eigen::VectorX<T>>& residual) {
  const DerStructuralProperty<T>& prop = model_->structural_property();
  const T r_squared_inv = prop.rhoA() / prop.rhoJ();
  T squared_sum = 0;
  for (int i = 0; i < residual.size(); ++i) {
    squared_sum +=
        residual[i] * residual[i] * (i % 4 == 3 ? r_squared_inv : 1.0);
  }
  return sqrt(squared_sum);
}

template <typename T>
bool DerSolver<T>::solver_converged(const T& residual_norm,
                                    const T& initial_residual_norm) const {
  return residual_norm < std::max(relative_tolerance_ * initial_residual_norm,
                                  absolute_tolerance_);
}

template <typename T>
std::unique_ptr<DerSolver<T>> DerSolver<T>::Clone() const {
  auto clone = std::make_unique<DerSolver<T>>(this->model_, this->integrator_);
  /* Copy the owned DerState. */
  clone->state_->CopyFrom(*this->state_);
  /* Copy the owned SchurComplement. */
  clone->tangent_matrix_schur_complement_ =
      this->tangent_matrix_schur_complement_;
  /* Copy the solver parameters. */
  clone->relative_tolerance_ = this->relative_tolerance_;
  clone->absolute_tolerance_ = this->absolute_tolerance_;
  clone->max_iterations_ = this->max_iterations_;
  /* Scratch variables do not need to be copied. */
  return clone;
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

template class drake::multibody::der::internal::DerSolver<double>;
