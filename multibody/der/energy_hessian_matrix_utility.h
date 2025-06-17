#pragma once

#include <unordered_set>

#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"
#include "drake/multibody/fem/schur_complement.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

using contact_solvers::internal::Block4x4SparseSymmetricMatrix;
using multibody::fem::internal::SchurComplement;

/* Converts a Block4x4SparseSymmetricMatrix into an Eigen::SparseMatrix.
 @pre `dest != nullptr`.
 @pre `dest->rows() == dest->cols()`.
 @pre `source.rows() == dest->rows() ||
       source.rows() == dest->rows() + 1`.
 @pre If `source.rows() == dest->rows() + 1`, the last row and last column of
      `source` are zero. */
template <typename T>
void Convert(const Block4x4SparseSymmetricMatrix<T>& source,
             Eigen::SparseMatrix<T>* dest);

/* Given a system of linear equations that can be written in block form as:
     Ax + By  =  a     (1)
     Bᵀx + Dy =  0     (2)
 The indices in `participating_dofs` form the matrix A. One can solve the system
 using Schur complement, where
     (A - BD⁻¹Bᵀ)x = a.
 After a solution for x is obtained, y can be recovered from
     y = -D⁻¹Bᵀx.
 Returns a structure that contains the D complement (i.e., A - BD⁻¹Bᵀ) and a
 method to recover y.

 @pre `mat.rows() == mat.cols()`.
 @pre Every dof index in `participating_dofs` is greater than or equal to zero
      and less than `mat.rows()`.
 @throws if `mat` is not positive definite.
 @tparam_nonsymbolic_scalar */
template <typename T>
SchurComplement<T> ComputeSchurComplement(
    const Eigen::SparseMatrix<T>& mat,
    const std::unordered_set<int> participating_dofs);

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
