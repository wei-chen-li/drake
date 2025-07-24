#pragma once

#include <unordered_set>

#include "drake/common/default_scalars.h"
#include "drake/multibody/contact_solvers/block_sparse_ldlt_solver.h"
#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"
#include "drake/multibody/der/der_indexes.h"
#include "drake/multibody/der/schur_complement.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/* Forward declaration. */
template <typename T>
class EnergyHessianMatrixLinearSolver;
template <typename T>
class EnergyHessianMatrixVectorProduct;

/*
 @p EnergyHessianMatrix represents the Hessian matrix of the internal elastic
 energy of a discrete elastic rod.

 @tparam_default_scalar
 */
template <typename T>
class EnergyHessianMatrix {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(EnergyHessianMatrix);

  /* Allocates an energy Hessian matrix based on the `num_dofs` of the discrete
  elastic rod.
  @pre `num_dofs >= 7`.
  @pre `num_dofs % 4 == 0 || num_dofs % 4 == 3`. */
  static EnergyHessianMatrix<T> Allocate(int num_dofs);

  int rows() const { return num_dofs_; }
  int cols() const { return num_dofs_; }

  /* Multiplies an this with a vector. The actual computation not performed
   until the returned result is assigned to an Eigen::VectorX.
   @pre `vec.size() == cols()`. */
  EnergyHessianMatrixVectorProduct<T> operator*(
      const Eigen::VectorX<T>& vec) const;

  /* Returns the diagonal of this matrix. */
  Eigen::VectorX<T> Diagonal() const;

  /* Sets the numerical values of all nonzero blocks to zero without changing
   the sparsity pattern. */
  void SetZero();

  /* Inserts mat = ∂²E/∂(xᵢ)(xⱼ) into this hessian. */
  void Insert(DerNodeIndex i, DerNodeIndex j, Matrix3<T> mat);

  /* Inserts vec = ∂²E/∂(xᵢ)(γʲ) into this hessian. */
  void Insert(DerNodeIndex i, DerEdgeIndex j, Vector3<T> vec);

  /* Inserts val = ∂²E/∂(γⁱ)(γʲ) into this hessian.  */
  void Insert(DerEdgeIndex i, DerEdgeIndex j, T val);

  /* Adds a scaled diagonal matrix to this matrix.
   @pre `rhs.rows() == rows()`. */
  void AddScaledMatrix(const Eigen::DiagonalMatrix<T, Eigen::Dynamic>& rhs,
                       const T& scale);

  /* Adds a scaled EnergyHessianMatrix to this matrix.
   @pre `rhs.rows() == rows()`. */
  void AddScaledMatrix(const EnergyHessianMatrix<T>& rhs, const T& scale);

  /* Zero out rows and columns corresponding to the DoFs associated with the
   node specified by `node_index`. Furthermore, the diagonal entries for those
   DoFs will be set to one.*/
  void ApplyBoundaryCondition(DerNodeIndex node_index);

  /* Zero out row and column corresponding to the DoF associated with the edge
   specified by `edge_index`. Furthermore, the diagonal entrie for the DoF will
   be set to one.*/
  void ApplyBoundaryCondition(DerEdgeIndex edge_index);

  /* Given a system of linear equations that can be written in block form as:
       Ax + By  =  a     (1)
       Bᵀx + Dy =  0     (2)
   The indices in `participating_dofs` arranged in increasing order forms the
   matrix A. One can solve the system using Schur complement, where
   (A - BD⁻¹Bᵀ)x = a. After a solution for x is obtained, y can be recovered
   from y = -D⁻¹Bᵀx. Returns a structure that contains the D complement
   (i.e., A - BD⁻¹Bᵀ) and a method to recover y.

   @pre Every dof index in `participating_dofs` is greater than or equal to zero
        and less than `rows()`.
   @throws if `this` is not positive definite. */
  template <typename T1 = T>
  std::enable_if_t<std::is_same_v<T1, double>, SchurComplement<T>>
  ComputeSchurComplement(
      const std::unordered_set<int>& participating_dofs) const;

  /* Makes a dense representation of this matrix. Useful for debugging purposes.
   */
  Eigen::MatrixX<T> MakeDenseMatrix() const;

 private:
  /* Friend class to facilitate testing. */
  friend class EnergyHessianMatrixTester;

  /* Allow helper class to access private member. */
  friend class EnergyHessianMatrixLinearSolver<T>;
  friend class EnergyHessianMatrixVectorProduct<T>;

  /* Private constructor. */
  EnergyHessianMatrix(
      int num_dofs,
      contact_solvers::internal::Block4x4SparseSymmetricMatrix<T>&& data);

  /* Returns true if the `data_.rows() == num_dofs_`;
   returns false if `data_.rows() == num_dofs_ + 1`.  */
  bool is_data_size_exact() const;

  int num_dofs_{};
  contact_solvers::internal::Block4x4SparseSymmetricMatrix<T> data_;
};

/*
 @p EnergyHessianMatrixLinearSolver solves linear systems with a
 EnergyHessianMatrix as the system matrix.

 @tparam_double_only
 */
template <typename T>
class EnergyHessianMatrixLinearSolver {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(EnergyHessianMatrixLinearSolver);

  /* Constructs a EnergyHessianMatrixLinearSolver which can be used to solve
  linear systems with `num_dofs`.
  @pre `num_dofs >= 7`.
  @pre `num_dofs % 4 == 0 || num_dofs % 4 == 3`. */
  explicit EnergyHessianMatrixLinearSolver(int num_dofs);

  /* Solves A x = b. Returns false if factorization of A fails.
   @pre `A.rows() == num_dofs_`.
   @pre `b.size() == num_dofs_`.
   @pre `x->size() == num_dofs_`. */
  [[nodiscard]] bool Solve(const EnergyHessianMatrix<T>& A,
                           const Eigen::Ref<const VectorX<T>> b,
                           EigenPtr<VectorX<T>> x);

  /* Regularizes the matrix A to a positive definite matrix and return the
   result.
   @pre `A.rows() == num_dofs_`. */
  EnergyHessianMatrix<T> RegularizeToPositiveDefinite(
      const EnergyHessianMatrix<T>& A);

 private:
  int num_dofs_;
  contact_solvers::internal::BlockSparseLdltSolver<Matrix4<T>> ldlt_solver_;
};

/* Class returned by EnergyHessianMatrix<T> times Eigen::VectorX<T>. */
template <typename T>
class EnergyHessianMatrixVectorProduct {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(EnergyHessianMatrixVectorProduct);

  EnergyHessianMatrixVectorProduct(const EnergyHessianMatrix<T>* mat,
                                   const Eigen::VectorX<T>* vec,
                                   const T& scale = 1.0)
      : mat_(mat), vec_(vec), scale_(scale) {
    DRAKE_THROW_UNLESS(mat != nullptr);
    DRAKE_THROW_UNLESS(vec != nullptr);
    DRAKE_THROW_UNLESS(mat->cols() == vec->size());
  }

  friend EnergyHessianMatrixVectorProduct<T> operator*(
      const T& scalar, const EnergyHessianMatrixVectorProduct<T>& self) {
    return EnergyHessianMatrixVectorProduct<T>(self.mat_, self.vec_,
                                               self.scale_ * scalar);
  }

  friend EnergyHessianMatrixVectorProduct<T> operator*(
      const EnergyHessianMatrixVectorProduct<T>& self, const T& scalar) {
    return EnergyHessianMatrixVectorProduct<T>(self.mat_, self.vec_,
                                               self.scale_ * scalar);
  }

  friend Eigen::Ref<Eigen::VectorX<T>> operator+=(
      Eigen::Ref<Eigen::VectorX<T>> lhs,
      const EnergyHessianMatrixVectorProduct<T>& self) {
    return self.AddToVector(&lhs);
  }

 private:
  /* Computes the vector represented by this class and add it to `lhs`.
   @return `*lhs`. */
  Eigen::Ref<Eigen::VectorX<T>> AddToVector(
      EigenPtr<Eigen::VectorX<T>> lhs) const;

  const EnergyHessianMatrix<T>* mat_;
  const Eigen::VectorX<T>* vec_;
  T scale_;
};

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::EnergyHessianMatrix);
