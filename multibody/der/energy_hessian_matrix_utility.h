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

/* Forward declaration. */
template <typename T>
class Block4x4SparseSymmetricMatrixVectorProduct;

/* Multiplies a Block4x4SparseSymmetricMatrix with a vector. The actual
 computation not performed until the returned
 Block4x4SparseSymmetricMatrixVectorProduct is assigned to a Eigen::VectorX.

 @param[in] rhs_mat The block-sparse symmetric matrix.
 @param[in] rhs_vec The vector to be multiplied by.

 @pre `mat.rows() == vec.size() ||
       mat.rows() == vec.size() + 1`.
 @pre If `mat.rows() == vec.size() + 1`, the last row and last column of `mat`
      should be zero.
 @tparam_default_scalar */
template <typename T>
Block4x4SparseSymmetricMatrixVectorProduct<T> operator*(
    const Block4x4SparseSymmetricMatrix<T>& mat, const Eigen::VectorX<T>& vec) {
  return Block4x4SparseSymmetricMatrixVectorProduct(&mat, &vec);
}

/* Class returned by the above operator*(). */
template <typename T>
class Block4x4SparseSymmetricMatrixVectorProduct {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(
      Block4x4SparseSymmetricMatrixVectorProduct);

  /* T times Block4x4SparseSymmetricMatrixVectorProduct<T>. */
  friend Block4x4SparseSymmetricMatrixVectorProduct<T> operator*(
      const T& scalar,
      const Block4x4SparseSymmetricMatrixVectorProduct<T>& self) {
    return Block4x4SparseSymmetricMatrixVectorProduct<T>(self.mat_, self.vec_,
                                                         self.scale_ * scalar);
  }

  /* Block4x4SparseSymmetricMatrixVectorProduct<T> times T. */
  friend Block4x4SparseSymmetricMatrixVectorProduct<T> operator*(
      const Block4x4SparseSymmetricMatrixVectorProduct<T>& self,
      const T& scalar) {
    return Block4x4SparseSymmetricMatrixVectorProduct<T>(self.mat_, self.vec_,
                                                         self.scale_ * scalar);
  }

  /* Eigen::VectorX<T> += Block4x4SparseSymmetricMatrixVectorProduct<T>. */
  friend Eigen::Ref<Eigen::VectorX<T>> operator+=(
      Eigen::Ref<Eigen::VectorX<T>> lhs,
      const Block4x4SparseSymmetricMatrixVectorProduct<T>& self) {
    return self.AddToVector(&lhs);
  }

 private:
  friend Block4x4SparseSymmetricMatrixVectorProduct<T> operator*
      <T>(const Block4x4SparseSymmetricMatrix<T>&, const Eigen::VectorX<T>&);

  Block4x4SparseSymmetricMatrixVectorProduct(
      const Block4x4SparseSymmetricMatrix<T>* mat, const Eigen::VectorX<T>* vec,
      const T& scale = 1.0);

  /* Computes the vector represented by this class and add it to `lhs`.
   @return `*lhs`. */
  Eigen::Ref<Eigen::VectorX<T>> AddToVector(
      EigenPtr<Eigen::VectorX<T>> lhs) const;

  const Block4x4SparseSymmetricMatrix<T>* mat_;
  const Eigen::VectorX<T>* vec_;
  T scale_;
};

/* Adds a scaled diagonal matrix to a Block4x4SparseSymmetricMatrix.

 @param[in,out] lhs The block-sparse symmetric matrix to be modified.
 @param[in] rhs The diagonal matrix to add.
 @param[in] scale Optional scaling factor (default = 1.0).

 @pre `lhs != nullptr`.
 @pre `lhs.rows() == rhs.rows() || lhs.rows() == rhs.rows() + 1`.
 @pre If `lhs.rows() == rhs.rows() + 1`, the last row and last column of `mat`
      should be zero.
 @tparam_default_scalar */
template <typename T>
void AddScaledMatrix(Block4x4SparseSymmetricMatrix<T>* lhs,
                     const Eigen::DiagonalMatrix<T, Eigen::Dynamic>& rhs,
                     const T& scale = 1.0);

/* Adds a scaled Block4x4SparseSymmetricMatrix to another.

 @param[in,out] lhs The destination matrix to be modified.
 @param[in] rhs The matrix to scale and add.
 @param[in] scale Optional scaling factor (default = 1.0).

 @pre `lhs != nullptr`.
 @pre `lhs` should have compatible sparsity pattern with `rhs`.
 @tparam_default_scalar */
template <typename T>
void AddScaledMatrix(Block4x4SparseSymmetricMatrix<T>* lhs,
                     const Block4x4SparseSymmetricMatrix<T>& rhs,
                     const T& scale = 1.0);

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

 @pre Every dof index in `participating_dofs` is greater than or equal to zero
      and less than `mat.rows()`.
 @throws if `mat` is not positive definite.
 @tparam_nonsymbolic_scalar */
template <typename T>
SchurComplement<T> ComputeSchurComplement(
    const Block4x4SparseSymmetricMatrix<T>& mat,
    const std::unordered_set<int> participating_dofs);

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
