#pragma once

#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/common/reset_after_move.h"
#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

/* A LDLᵀ solver for solving the symmetric system
   A⋅x = b
 where A is block sparse.

 @tparam BlockType The matrix type for individual block matrices;
 MatrixX<double> or Matrix4<double>. The fixed-size matrix version is preferred
 if you know the sizes of blocks are uniform and fixed. */
template <typename BlockType>
class BlockSparseLdltSolver {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BlockSparseLdltSolver);

  /* The state of the solver. */
  enum class SolverMode {
    kEmpty,     // Matrix A is yet to be set.
    kAnalyzed,  // Matrix A is symbolically analyzed and ready to be factored,
                // but not yet numerically factored.
    kFactored,  // Matrix A is factored in place into L and D..
  };

  using Scalar = typename BlockType::Scalar;
  using SymmetricMatrix =
      BlockSparseLowerTriangularOrSymmetricMatrix<BlockType, true>;
  using LowerTriangularMatrix =
      BlockSparseLowerTriangularOrSymmetricMatrix<BlockType, false>;

  /* Constructs a BlockSparseLdltSolver. */
  BlockSparseLdltSolver();

  ~BlockSparseLdltSolver();

  /* Sets the matrix to be factored and analyzes its sparsity pattern to find an
   efficient elimination ordering. Factor() may be called after call to
   SetMatrix(). SetMatrix() can be called repeatedly in a row, but prefer the
   following pattern when the sparsity patterns don't change.

     solver.SetMatrix(A0);
     ...
     solver.UpdateMatrix(A1);
     ...
     solver.UpdateMatrix(A2);
     ...
     solver.UpdateMatrix(A3);
     ...
   See UpdateMatrix().

   @pre A is positive definite.
   @post solver_mode() == SolverMode::kAnalyzed. */
  void SetMatrix(const SymmetricMatrix& A);

  /* Updates the matrix to be factored. This is useful for solving a series of
   matrices with the same sparsity pattern using the same elimination ordering.
   For example, with matrices A and B with the same sparisty pattern. It's more
   efficient to call
     solver.SetMatrix(A);
     solver.Factor();
     solver.Solve(...)
     solver.UpdateMatrix(B);
     solver.Factor();
     solver.Solve(...)
   than to call
     solver.SetMatrix(A);
     solver.Factor();
     solver.Solve(...)
     solver.SetMatrix(B);
     solver.Factor();
     solver.Solve(...)
   Factor() may be called after call to UpdateMatrix().
   @pre SetMatrix() has been invoked and the argument to the last call of
   SetMatrix() has the same sparsity pattern of A.
   @pre A is positive definite.
   @post solver_mode() == SolverMode::kAnalyzed. */
  void UpdateMatrix(const SymmetricMatrix& A);

  /* Computes the block sparse LDLᵀ factorization. Returns true if factorization
   succeeds, otherwise returns false. Failure is triggered by the fact that
   individual blocks of D have eigenvalues that are too small.
   @throws std::exception if solver_mode() is not SolverMode::kAnalyzed.
   @post solver_mode() is SolverMode::kFactored if factorization is successful
         and is SolverMode::kEmpty otherwise. */
  [[nodiscard]] bool Factor();

  /* Computes the block sparse LDLᵀ factorization, but regularizes the matrix D
   so that it is positive definite.
   @param regularized_D_min_eigenvalue  The eigenvalues for each diagonal block
                                        of D smaller than this value will be set
                                        to this value.
   @pre `regularized_D_min_eigenvalue > 0`.
   @throws std::exception if solver_mode() is not SolverMode::kAnalyzed.
   @post solver_mode() == SolverMode::kFactored. */
  void FactorForcePositiveDefinite(Scalar regularized_D_min_eigenvalue);

  /* Solves the system A⋅x = b and returns x.
   @throws std::exception if b.size() is incompatible with the size of the
   matrix set by SetMatrix().
   @throws std::exception unless solver_mode() == SolverMode::kFactored. */
  VectorX<Scalar> Solve(const Eigen::Ref<const VectorX<Scalar>>& b) const;

  /* Returns the current mode of the solver. See SolverMode. */
  SolverMode solver_mode() const { return solver_mode_; }

  /* Returns a matrix A, computed from A = Pᵀ⋅L⋅D⋅Lᵀ⋅P.
   @note The returned A may be different from the A set via SetMatrix() or
   UpdateMatrix() if FactorWithRegularization() is called later. In such cases,
   the returned matrix A is guaranteed to be positive definite (because L is
   full-rank and D is regularized to be positive definite).
   @throws std::exception unless solver_mode() == SolverMode::kFactored.  */
  SymmetricMatrix A() const;

  /* Returns (the lower triangular) Cholesky factorization matrix L.
   L is defined by L⋅D⋅Lᵀ = P⋅A⋅Pᵀ.
   @throws std::exception unless solver_mode() == SolverMode::kFactored. */
  LowerTriangularMatrix L() const;

  /* Returns block diagonal matrix D.
   D is defined by L⋅D⋅Lᵀ = P⋅A⋅Pᵀ.
   @throws std::exception unless solver_mode() == SolverMode::kFactored. */
  SymmetricMatrix D() const;

  /* Returns the permutation matrix P induced by the elimination ordering.
   P is defined by L⋅D⋅Lᵀ = P⋅A⋅Pᵀ.
   @throws std::exception if solver_mode() == SolverMode::kEmpty. */
  Eigen::PermutationMatrix<Eigen::Dynamic> P() const;

 private:
  /* Helper for SetMatrix() to set the matrix given its elimination ordering and
   the sparsity pattern of its L matrix. It performs the
   following:
    1. sets `A_sparsity_pattern_`;
    2. sets `block_permutation_`;
    3. sets `scalar_permutation_`;
    4. allocates for `L_`, `D_` and `Dinv_`;
    5. calls UpdateMatrix(A) to copy the numeric values of A to L_.
   @param[in] A                     The matrix to be factored.
   @param[in] elimination_ordering  Elimination ordering of the blocks of A.
                                    Must be a permutation of {0, 1, ...,
                                    A.block_cols() - 1}.
   @param[in] L_pattern             The block sparsity pattern of the lower
                                    triangular matrix L if A is to be factored
                                    with the given elimination ordering.  */
  void SetMatrixImpl(const SymmetricMatrix& A,
                     const std::vector<int>& elimination_ordering,
                     BlockSparsityPattern&& L_pattern);

  /* Sets `scalar_permutation_` given the matrix A and the prescribed
   elimination ordering.
   @param[in] A                     The matrix to be factored.
   @param[in] elimination_ordering  Elimination ordering of the blocks of A.
                                    Must be a permutation of {0, 1, ...,
                                    A.block_cols() - 1}. */
  void SetScalarPermutation(const SymmetricMatrix& A,
                            const std::vector<int>& elimination_ordering);

  /* Returns the block sparsity pattern of the L matrix from a block sparse
   Cholesky factorization following the prescribed elimination ordering.
   @param[in] A                     The matrix to be factored.
   @param[in] elimination_ordering  Elimination ordering of the blocks of A.
                                    Must be a permutation of {0, 1, ...,
                                    A.block_cols() - 1}. */
  BlockSparsityPattern SymbolicFactor(
      const SymmetricMatrix& A, const std::vector<int>& elimination_ordering);

  /* Factorizes matrix A.
   @param regularized_D_min_eigenvalue  If specified, every D(j,j) matrix will
                                        have its eigenvalues clamped below by
                                        this value.
   @note this function does not modify solver mode.
   @pre solver_mode() == kAnalyzed. */
  bool ComputeFactorization(
      std::optional<Scalar> regularized_D_min_eigenvalue = std::nullopt);

  /* Performs L(j+1:, j+1:) -= L(j+1:, j) * D(j, j) * L(j+1:, j).transpose().
   @pre 0 <= j < L.block_cols(). */
  void RightLookingSymmetricRank1Update(int j);

  /* Permutes the given matrix A with `block_permutation_` p and set L such that
   the lower triangular part of L satisfies L(p(i), p(j)) = A(i, j).
   @pre SetMarix() has been called. */
  void PermuteAndCopyToL(const SymmetricMatrix& A);

  /* The LDLᵀ factorization of the permuted matrix, i.e. L⋅D⋅Lᵀ = P⋅A⋅Pᵀ,
   where P is the permutation matrix induced by the `scalar_permutation_`. */
  copyable_unique_ptr<LowerTriangularMatrix> L_;
  std::vector<BlockType> D_;
  std::vector<BlockType> Dinv_;

  /* Block and scalar representations of the permutation matrix P (see
   CalcPermutationMatrix()). */
  /* Permutation for block indices, same size as A.block_cols(). For a given
   block index i into A, `block_permutation_[i]` gives the permuted block index
   into L_. */
  PartialPermutation block_permutation_;
  /* Permutation for scarlar indices, same size as A.cols(). For a given
   scalar index i into A, `scalar_permutation_[i]` gives the permuted scalar
   index into L_. */
  PartialPermutation scalar_permutation_;

  /* The sparsity pattern of the A matrix. */
  BlockSparsityPattern A_sparsity_pattern_;

  reset_after_move<SolverMode> solver_mode_{SolverMode::kEmpty};
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
