#pragma once

#include <memory>

#include "drake/multibody/der/der_state.h"
#include "drake/multibody/der/der_structural_property.h"
#include "drake/multibody/der/der_undeformed_state.h"
#include "drake/multibody/der/energy_hessian_matrix_util.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/* Computes E = Eₛ + Eₙ + Eₜ.

 @pre `undeformed` and `state` are compatible.
 @tparam_default_scalar */
template <typename T>
T ComputeElasticEnergy(const DerStructuralProperty<T>& prop,
                       const DerUndeformedState<T>& undeformed,
                       const DerState<T>& state);

/* Computes ∂E/∂q.

 @pre `undeformed` and `state` are compatible.
 @pre `jacobian != nullptr`.
 @pre `jacobian->size() == state.num_dofs()`.
 @tparam_default_scalar */
template <typename T>
void ComputeElasticEnergyJacobian(const DerStructuralProperty<T>& prop,
                                  const DerUndeformedState<T>& undeformed,
                                  const DerState<T>& state,
                                  EigenPtr<Eigen::VectorX<T>> jacobian);

/* Computes ∂²E/∂q².

 If `!state.has_closed_ends()`, ∂²E/∂q² occupies the top-left `num_dof() ×
 num_dof()` submatrix of `hessian` (size is off by 1).
 If `state.has_closed_ends()`, the size is exact.

 @pre `undeformed` and `state` are compatible.
 @pre `hessian != nullptr`.
 @pre `hessia` is created from MakeEnergyHessianMatrix().
 @tparam_default_scalar */
template <typename T>
void ComputeElasticEnergyHessian(const DerStructuralProperty<T>& prop,
                                 const DerUndeformedState<T>& undeformed,
                                 const DerState<T>& state,
                                 Block4x4SparseSymmetricMatrix<T>* hessian);

/* Creates a Hessian matrix with the appropriate sparsity pattern, initialized
 to zero.
 @tparam_default_scalar */
template <typename T>
Block4x4SparseSymmetricMatrix<T> MakeEnergyHessianMatrix(bool has_closed_ends,
                                                         int num_nodes);

/* Computes Eₛ. */
template <typename T>
T ComputeStretchingEnergy(const DerStructuralProperty<T>& prop,
                          const DerUndeformedState<T>& undeformed,
                          const DerState<T>& state);

/* Adds ∂Eₛ/∂q to `jacobian`. */
template <typename T>
void AddStretchingEnergyJacobian(const DerStructuralProperty<T>& prop,
                                 const DerUndeformedState<T>& undeformed,
                                 const DerState<T>& state,
                                 EigenPtr<Eigen::VectorX<T>> jacobian);

/* Adds ∂²Eₛ/∂q² to `hessian`. */
template <typename T>
void AddStretchingEnergyHessian(const DerStructuralProperty<T>& prop,
                                const DerUndeformedState<T>& undeformed,
                                const DerState<T>& state,
                                Block4x4SparseSymmetricMatrix<T>* hessian);

/* Computes Eₜ. */
template <typename T>
T ComputeTwistingEnergy(const DerStructuralProperty<T>& prop,
                        const DerUndeformedState<T>& undeformed,
                        const DerState<T>& state);

/* Adds ∂Eₜ/∂q to `jacobian`. */
template <typename T>
void AddTwistingEnergyJacobian(const DerStructuralProperty<T>& prop,
                               const DerUndeformedState<T>& undeformed,
                               const DerState<T>& state,
                               EigenPtr<Eigen::VectorX<T>> jacobian);

/* Adds ∂²Eₜ/∂q² to `hessian`. */
template <typename T>
void AddTwistingEnergyHessian(const DerStructuralProperty<T>& prop,
                              const DerUndeformedState<T>& undeformed,
                              const DerState<T>& state,
                              Block4x4SparseSymmetricMatrix<T>* hessian);

/* Computes Eₙ. */
template <typename T>
T ComputeBendingEnergy(const DerStructuralProperty<T>& prop,
                       const DerUndeformedState<T>& undeformed,
                       const DerState<T>& state);

/* Adds ∂Eₙ/∂q to `jacobian`. */
template <typename T>
void AddBendingEnergyJacobian(const DerStructuralProperty<T>& prop,
                              const DerUndeformedState<T>& undeformed,
                              const DerState<T>& state,
                              EigenPtr<Eigen::VectorX<T>> jacobian);

/* Adds ∂²Eₙ/∂q² to `hessian`. */
template <typename T>
void AddBendingEnergyHessian(const DerStructuralProperty<T>& prop,
                             const DerUndeformedState<T>& undeformed,
                             const DerState<T>& state,
                             Block4x4SparseSymmetricMatrix<T>* hessian);

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
