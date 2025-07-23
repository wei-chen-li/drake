#include "drake/multibody/contact_solvers/block_sparse_ldlt_solver.h"

#include <memory>
#include <numeric>
#include <utility>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/unused.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;

class BlockSparseLdltSolverTest : public ::testing::Test {
 private:
  /* Makes an arbitrary SPD matrix with the following nonzero pattern.
    X X | O O O | O O O O | O O O
    X X | O O O | O O O O | O O O
    --- | ----- |---------| -----
    O O | X X X | X X X X | X X X
    O O | X X X | X X X X | X X X
    O O | X X X | X X X X | X X X
    --- | ----- |---------| -----
    O O | X X X | X X X X | O O O
    O O | X X X | X X X X | O O O
    O O | X X X | X X X X | O O O
    O O | X X X | X X X X | O O O
    --- | ----- |---------| -----
    O O | X X X | O O O O | X X X
    O O | X X X | O O O O | X X X
    O O | X X X | O O O O | X X X
  */
  BlockSparseSymmetricMatrix RandomSymmetricMatrix(int seed = 0) const {
    srand(seed);

    std::vector<std::vector<int>> sparsity;
    sparsity.emplace_back(std::vector<int>{0});
    sparsity.emplace_back(std::vector<int>{1, 2, 3});
    sparsity.emplace_back(std::vector<int>{2});
    sparsity.emplace_back(std::vector<int>{3});
    std::vector<int> block_sizes = {2, 3, 4, 3};
    BlockSparsityPattern block_pattern(block_sizes, sparsity);

    BlockSparseSymmetricMatrix A(std::move(block_pattern));
    for (int j = 0; j < A.block_cols(); ++j) {
      for (int i : A.block_row_indices(j)) {
        MatrixXd block = MatrixXd::Random(block_sizes[i], block_sizes[j]);
        if (i == j) block = (block + block.transpose()).eval() / 2;
        A.SetBlock(i, j, block);
      }
    }
    return A;
  }

  /* Performs A += scalar * I. */
  static void AddScaledIdentity(BlockSparseSymmetricMatrix* A, double scalar) {
    DRAKE_THROW_UNLESS(A != nullptr);
    for (int j = 0; j < A->block_cols(); ++j) {
      const int size = A->sparsity_pattern().block_sizes()[j];
      A->AddToBlock(j, j, scalar * MatrixXd::Identity(size, size));
    }
  }

 protected:
  /* Returns a symmetric matrix with positive eigenvalues. */
  BlockSparseSymmetricMatrix MakePositiveDefiniteMatrix() const {
    BlockSparseSymmetricMatrix A = RandomSymmetricMatrix();
    const VectorXd eigenvalues =
        Eigen::SelfAdjointEigenSolver<MatrixXd>(A.MakeDenseMatrix(),
                                                Eigen::EigenvaluesOnly)
            .eigenvalues();
    AddScaledIdentity(&A,
                      (eigenvalues[0] <= 1.0) ? (1.0 - eigenvalues[0]) : 0.0);
    return A;
  }

  /* Returns a  symmetric matrix with one zero eigenvalue. */
  BlockSparseSymmetricMatrix MakePositiveSemiDefiniteMatrix() const {
    BlockSparseSymmetricMatrix A = RandomSymmetricMatrix();
    const VectorXd eigenvalues =
        Eigen::SelfAdjointEigenSolver<MatrixXd>(A.MakeDenseMatrix(),
                                                Eigen::EigenvaluesOnly)
            .eigenvalues();
    AddScaledIdentity(&A, -eigenvalues[0]);
    return A;
  }

  /* Returns a symmetric matrix with pisitive and negative eigenvalues, but no
   * zero eigenvalues. */
  BlockSparseSymmetricMatrix MakeIndefiniteMatrix() const {
    BlockSparseSymmetricMatrix A = RandomSymmetricMatrix();
    const VectorXd eigenvalues =
        Eigen::SelfAdjointEigenSolver<MatrixXd>(A.MakeDenseMatrix(),
                                                Eigen::EigenvaluesOnly)
            .eigenvalues();
    const int mid = eigenvalues.size() / 2;
    DRAKE_DEMAND(eigenvalues[mid + 1] - eigenvalues[mid] > 1e-12);
    AddScaledIdentity(&A, -(eigenvalues[mid] + eigenvalues[mid + 1]) / 2);
    return A;
  }
};

TEST_F(BlockSparseLdltSolverTest, Solve) {
  BlockSparseLdltSolver<MatrixXd> solver;
  BlockSparseSymmetricMatrix A = MakePositiveDefiniteMatrix();
  MatrixX<double> dense_A = A.MakeDenseMatrix();
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kEmpty);
  solver.SetMatrix(A);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kAnalyzed);
  bool success = solver.Factor();
  EXPECT_TRUE(success);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kFactored);

  const VectorXd b1 = VectorXd::LinSpaced(A.cols(), 0.0, 1.0);
  const VectorXd x1 = solver.Solve(b1);
  const VectorXd expected_x1 = dense_A.llt().solve(b1);
  EXPECT_TRUE(CompareMatrices(x1, expected_x1, 1e-13));

  /* Solve for a different right hand side without refactoring. */
  const VectorXd b2 = VectorXd::LinSpaced(A.cols(), 0.0, 10.0);
  const VectorXd x2 = solver.Solve(b2);
  const VectorXd expected_x2 = dense_A.llt().solve(b2);
  EXPECT_TRUE(CompareMatrices(x2, expected_x2, 1e-13));

  /* Update the matrix with an indefinite matrix but the same sparsity pattern.
   */
  BlockSparseSymmetricMatrix A2 = MakeIndefiniteMatrix();
  MatrixX<double> dense_A2 = A2.MakeDenseMatrix();
  solver.UpdateMatrix(A2);
  success = solver.Factor();
  EXPECT_TRUE(success);
  const VectorXd b3 = VectorXd::LinSpaced(A2.cols(), 0.0, 10.0);
  const VectorXd x3 = solver.Solve(b3);
  const VectorXd expected_x3 = dense_A2.ldlt().solve(b3);
  EXPECT_TRUE(CompareMatrices(x3, expected_x3, 1e-10));

  /* Update the matrix with an positive semi-definite matrix. Factorization
   should fail due to having eigenvalue of zero. */
  BlockSparseSymmetricMatrix A3 = MakePositiveSemiDefiniteMatrix();
  solver.SetMatrix(A3);
  success = solver.Factor();
  EXPECT_FALSE(success);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kEmpty);
}

TEST_F(BlockSparseLdltSolverTest, FactorForcePositiveDefinite) {
  BlockSparseLdltSolver<MatrixXd> solver;
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kEmpty);
  const double regularized_D_min_eigenvalue = 1e-12;

  /* For a positive definite matrix, regularization does not take effect. */
  BlockSparseSymmetricMatrix A1 = MakePositiveDefiniteMatrix();
  MatrixX<double> dense_A1 = A1.MakeDenseMatrix();
  solver.SetMatrix(A1);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kAnalyzed);
  solver.FactorForcePositiveDefinite(regularized_D_min_eigenvalue);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kFactored);

  const VectorXd b1 = VectorXd::LinSpaced(A1.cols(), 0.0, 1.0);
  const VectorXd x1 = solver.Solve(b1);
  const VectorXd expected_x1 = dense_A1.llt().solve(b1);
  EXPECT_TRUE(CompareMatrices(x1, expected_x1, 1e-13));

  /* Update the matrix with a positive definite matrix. The solver should
   regularize the eigenvalues of D. */
  BlockSparseSymmetricMatrix A2 = MakePositiveSemiDefiniteMatrix();
  solver.UpdateMatrix(A2);
  solver.FactorForcePositiveDefinite(regularized_D_min_eigenvalue);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kFactored);

  BlockSparseSymmetricMatrix D = solver.D();
  for (int j = 0; j < A2.block_cols(); ++j) {
    const MatrixXd D_block = D.diagonal_block(j);
    const VectorXd D_block_eigenvalues =
        Eigen::SelfAdjointEigenSolver<MatrixXd>(D_block, Eigen::EigenvaluesOnly)
            .eigenvalues();
    for (int k = 0; k < D_block_eigenvalues.size(); ++k) {
      ASSERT_GE(D_block_eigenvalues[k], regularized_D_min_eigenvalue * 0.999);
    }
  }
}

TEST_F(BlockSparseLdltSolverTest, FactorBeforeSetMatrixThrows) {
  BlockSparseLdltSolver<MatrixXd> solver;
  EXPECT_THROW(unused(solver.Factor()), std::exception);
}

TEST_F(BlockSparseLdltSolverTest, SolveBeforeFactorThrows) {
  BlockSparseLdltSolver<MatrixXd> solver;
  BlockSparseSymmetricMatrix A = MakePositiveDefiniteMatrix();
  solver.SetMatrix(A);
  VectorXd b = VectorXd::LinSpaced(A.cols(), 0.0, 10.0);
  EXPECT_THROW(solver.Solve(b), std::exception);
}

TEST_F(BlockSparseLdltSolverTest, LDLt_equals_PAPt) {
  BlockSparseLdltSolver<MatrixXd> solver;

  /* Exact factorization for a positive definite matrix. */
  BlockSparseSymmetricMatrix A1 = MakePositiveDefiniteMatrix();
  MatrixXd A1_dense = A1.MakeDenseMatrix();
  solver.SetMatrix(A1);
  /* Trying to get A, L, and D before factorization is an exception. */
  EXPECT_THROW(solver.A(), std::exception);
  EXPECT_THROW(solver.L(), std::exception);
  EXPECT_THROW(solver.D(), std::exception);
  const bool success = solver.Factor();
  EXPECT_TRUE(success);
  {
    const MatrixXd L = solver.L().MakeDenseMatrix();
    const MatrixXd D = solver.D().MakeDenseMatrix();
    const Eigen::PermutationMatrix<Eigen::Dynamic> P = solver.P();
    const MatrixXd lhs = L * D * L.transpose();
    const MatrixXd rhs = P * A1_dense * P.transpose();
    EXPECT_TRUE(CompareMatrices(lhs, rhs, 1e-14));
    EXPECT_TRUE(CompareMatrices(solver.A().MakeDenseMatrix(), A1_dense, 1e-14));
  }

  /* Factorize and regularize a positive semi-definite matrix. */
  BlockSparseSymmetricMatrix A2 = MakePositiveSemiDefiniteMatrix();
  solver.UpdateMatrix(A2);
  /* Trying to get A, L, and D before factorization is an exception. */
  EXPECT_THROW(solver.A(), std::exception);
  EXPECT_THROW(solver.L(), std::exception);
  EXPECT_THROW(solver.D(), std::exception);
  solver.FactorForcePositiveDefinite(1e-12);
  {
    const MatrixXd A = solver.A().MakeDenseMatrix();
    const MatrixXd L = solver.L().MakeDenseMatrix();
    const MatrixXd D = solver.D().MakeDenseMatrix();
    const Eigen::PermutationMatrix<Eigen::Dynamic> P = solver.P();
    const MatrixXd lhs = L * D * L.transpose();
    const MatrixXd rhs = P * A * P.transpose();
    EXPECT_TRUE(CompareMatrices(lhs, rhs, 1e-14));
    EXPECT_FALSE(CompareMatrices(A, A2.MakeDenseMatrix(), 1e-14));
  }
}

TEST_F(BlockSparseLdltSolverTest, PermutationMatrixPrecondition) {
  BlockSparseLdltSolver<MatrixXd> solver;
  BlockSparseSymmetricMatrix A = MakePositiveDefiniteMatrix();
  /* CalcPermutationMatrix() before setting the matrix throws. */
  EXPECT_THROW(solver.P(), std::exception);
  /* After setting the matrix, CalcPermutatoinMatrix() returns the same result
   before and after factorization. */
  solver.SetMatrix(A);
  const Eigen::PermutationMatrix<Eigen::Dynamic> P0 = solver.P();
  const bool success = solver.Factor();
  EXPECT_TRUE(success);
  const Eigen::PermutationMatrix<Eigen::Dynamic> P1 = solver.P();
  EXPECT_EQ(MatrixXi(P0), MatrixXi(P1));
}

TEST_F(BlockSparseLdltSolverTest, SolverModeAfterMove) {
  BlockSparseLdltSolver<MatrixXd> solver;
  BlockSparseSymmetricMatrix A = MakePositiveDefiniteMatrix();
  solver.SetMatrix(A);
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kAnalyzed);
  BlockSparseLdltSolver<MatrixXd> new_solver(std::move(solver));
  EXPECT_EQ(new_solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kAnalyzed);
  /* The mode of the old solver resets to kEmpty. */
  EXPECT_EQ(solver.solver_mode(),
            BlockSparseLdltSolver<MatrixXd>::SolverMode::kEmpty);
}

}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
