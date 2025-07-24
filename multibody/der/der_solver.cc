#include "drake/multibody/der/der_solver.h"

#include <algorithm>
#include <limits>

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
DerSolver<T>::DerSolver(const DerModel<T>* model,
                        const DiscreteTimeIntegrator<T>& integrator)
    : model_(model),
      integrator_(integrator.Clone()),
      dt_max_(integrator.dt()),
      dt_min_(dt_max_ * 0.001),
      dt_(dt_max_) {
  DRAKE_THROW_UNLESS(model != nullptr);
  state_ = model_->CreateDerState();
  /* Allocate a DerState. */
  scratch_.prev_state = model_->CreateDerState();
  /* Allocate the scratch for the DerModel. */
  scratch_.der_model_scratch = model_->MakeScratch();
  /* Initialize the linear solver. */
  const int num_dofs = model_->num_dofs();
  scratch_.linear_solver =
      std::make_unique<EnergyHessianMatrixLinearSolver<double>>(num_dofs);
  /* Set b and dz to have size num_dofs. */
  scratch_.b = Eigen::VectorX<T>::Zero(num_dofs);
  scratch_.dz = Eigen::VectorX<T>::Zero(num_dofs);
}

template <typename T>
int DerSolver<T>::AdvanceOneTimeStep(
    const DerState<T>& prev_state_in,
    const ExternalForceField<T>& external_force_field) {
  model_->ValidateDerState(prev_state_in);
  DRAKE_DEMAND(scratch_.prev_state != nullptr);
  DerState<T>& prev_state = *scratch_.prev_state;
  prev_state.CopyFrom(prev_state_in);

  double t = 0.0;
  const double t_target = dt_max_;
  int num_total_iters = 0;

  while (t < t_target - 8 * std::numeric_limits<double>::epsilon()) {
    const int newton_iters =
        AdvanceDt(prev_state, (t + dt_ < t_target) ? dt_ : (t_target - t),
                  external_force_field);
    const bool success = (newton_iters >= 0);
    if (success) {
      t += dt_;
      num_total_iters += newton_iters;
      prev_state.CopyFrom(*state_);
      if (newton_iters < 4) {
        dt_ = std::min(dt_ * 1.5, dt_max_);
      } else if (newton_iters > 10) {
        dt_ = std::max(dt_ * 0.7, dt_min_);
      }
    } else {
      dt_ *= 0.5;
      if (dt_ < dt_min_) {
        throw std::runtime_error(fmt::format(
            "DerSolver::AdvanceOneTimeStep() failed to converge even with a "
            "shrinked time-step of {}. Consider using a smaller nominal "
            "time-step size than the current value {}.",
            dt_, dt_max_));
      }
    }
  }

  state_->CopyFrom(prev_state_in);
  state_->AdvancePositionToNextStep(prev_state.get_position());
  state_->SetVelocity(prev_state.get_velocity());
  state_->SetAcceleration(prev_state.get_acceleration());
  return num_total_iters;
}

template <typename T>
int DerSolver<T>::AdvanceDt(const DerState<T>& prev_state, double dt,
                            const ExternalForceField<T>& external_force_field) {
  DerState<T>& state = *state_;
  typename DerModel<T>::Scratch* der_model_scratch =
      scratch_.der_model_scratch.get();
  EnergyHessianMatrixLinearSolver<double>& linear_solver =
      *scratch_.linear_solver;
  Eigen::VectorX<T>& b = scratch_.b;
  Eigen::VectorX<T>& dz = scratch_.dz;
  DRAKE_DEMAND(der_model_scratch != nullptr);

  integrator_->set_dt(dt);
  const Eigen::VectorX<T>& z = integrator_->GetUnknowns(prev_state);
  integrator_->AdvanceDt(prev_state, z, &state);
  model_->ApplyBoundaryCondition(&state);
  b = model_->ComputeResidual(state, external_force_field, der_model_scratch);
  T residual_norm = unit_adjusted_norm(b);
  const T initial_residual_norm = residual_norm;
  int iter = 0;
  /* For Newton-Raphson solver, we iterate until any of the following is true:
   1. The max number of allowed iterations is reached;
   2. The norm of the residual is smaller than the absolute tolerance.
   3. The relative error (the norm of the residual divided by the norm of the
      initial residual) is smaller than the dimensionless relative tolerance. */
  while (iter < max_newton_iters_ &&
         !solver_converged(residual_norm, initial_residual_norm)) {
    const EnergyHessianMatrix<T>& tangent_matrix = model_->ComputeTangentMatrix(
        state, integrator_->GetWeights(), der_model_scratch);
    bool success = linear_solver.Solve(tangent_matrix, -b, &dz);
    if (!success) return -1;
    integrator_->AdjustStateFromChangeInUnknowns(dz, &state);
    b = model_->ComputeResidual(state, external_force_field, der_model_scratch);
    residual_norm = unit_adjusted_norm(b);
    ++iter;
  }
  if (!solver_converged(residual_norm, initial_residual_norm)) {
    return -1;
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

  integrator_->set_dt(dt_max_);
  const EnergyHessianMatrix<T>& tangent_matrix = model_->ComputeTangentMatrix(
      state, integrator_->GetWeights(), der_model_scratch);

  tangent_matrix_schur_complement_ =
      tangent_matrix.ComputeSchurComplement(participating_dofs);
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
  integrator_->set_dt(dt_max_);
  auto clone = std::make_unique<DerSolver<T>>(this->model_, *integrator_);
  /* Copy the owned DerState. */
  clone->state_->CopyFrom(*this->state_);
  /* Copy the owned SchurComplement. */
  clone->tangent_matrix_schur_complement_ =
      this->tangent_matrix_schur_complement_;
  /* Copy the solver parameters. */
  clone->dt_ = this->dt_;
  clone->relative_tolerance_ = this->relative_tolerance_;
  clone->absolute_tolerance_ = this->absolute_tolerance_;
  clone->max_newton_iters_ = this->max_newton_iters_;
  /* Scratch variables do not need to be copied. */
  return clone;
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

template class drake::multibody::der::internal::DerSolver<double>;
