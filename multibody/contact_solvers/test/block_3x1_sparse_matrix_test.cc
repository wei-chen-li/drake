#include "drake/multibody/contact_solvers/block_3x1_sparse_matrix.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/contact_solvers/block_3x3_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

/* Returns an arbitrary non-zero matrix of size m-by-n.*/
MatrixXd MakeArbitraryMatrix(int m, int n) {
  MatrixXd A(m, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      A(i, j) = 3 * i + 4 * j;
    }
  }
  return A;
}

/* Returns a dummy 3x1 matrix with all entries in the matrix equal to the given
 value. */
Vector3d MakeVector(double value) {
  return value * Vector3d::Ones();
}

/* Returns an arbitrary Block3x1SparseMatrix with size 12-by-7. */
Block3x1SparseMatrix<double> MakeBlockSparseMatrix() {
  Block3x1SparseMatrix<double> sparse_matrix(4, 7);
  std::vector<Block3x1SparseMatrix<double>::Triplet> triplets;
  triplets.emplace_back(0, 0, MakeVector(1.0));
  triplets.emplace_back(0, 0, MakeVector(2.0));
  triplets.emplace_back(0, 1, MakeVector(3.0));
  triplets.emplace_back(0, 1, MakeVector(4.0));
  triplets.emplace_back(2, 4, MakeVector(5.0));
  triplets.emplace_back(2, 5, MakeVector(6.0));
  triplets.emplace_back(3, 6, MakeVector(7.0));
  triplets.emplace_back(3, 3, MakeVector(8.0));
  sparse_matrix.SetFromTriplets(triplets);
  EXPECT_EQ(sparse_matrix.num_blocks(), 6);
  return sparse_matrix;
}

GTEST_TEST(Block3x1SparseMatrixTest, Size) {
  const Block3x1SparseMatrix<double> sparse_matrix(4, 7);
  EXPECT_EQ(sparse_matrix.rows(), 12);
  EXPECT_EQ(sparse_matrix.cols(), 7);
  EXPECT_EQ(sparse_matrix.block_rows(), 4);
  EXPECT_EQ(sparse_matrix.block_cols(), 7);
}

GTEST_TEST(Block3x1SparseMatrixTest, SetFromTriplets) {
  Block3x1SparseMatrix<double> sparse_matrix = MakeBlockSparseMatrix();
  MatrixXd expected_matrix = MatrixXd::Zero(12, 7);
  expected_matrix.block<3, 1>(0, 0) = MakeVector(3.0);
  expected_matrix.block<3, 1>(0, 1) = MakeVector(7.0);
  expected_matrix.block<3, 1>(6, 4) = MakeVector(5.0);
  expected_matrix.block<3, 1>(6, 5) = MakeVector(6.0);
  expected_matrix.block<3, 1>(9, 3) = MakeVector(8.0);
  expected_matrix.block<3, 1>(9, 6) = MakeVector(7.0);
  EXPECT_EQ(sparse_matrix.MakeDenseMatrix(), expected_matrix);

  /* Verify that setting the sparse matrix with a new vector of triplets resets
   the matrix. */
  std::vector<Block3x1SparseMatrix<double>::Triplet> triplets;
  triplets.emplace_back(2, 2, MakeVector(4.0));
  sparse_matrix.SetFromTriplets(triplets);
  expected_matrix.setZero();
  expected_matrix.block<3, 1>(6, 2) = MakeVector(4.0);
  EXPECT_EQ(sparse_matrix.MakeDenseMatrix(), expected_matrix);
  EXPECT_EQ(sparse_matrix.num_blocks(), 1);
}

GTEST_TEST(Block3x1SparseMatrixTest, MultiplyAndAddTo) {
  const MatrixXd A = MakeArbitraryMatrix(7, 6);
  const Block3x1SparseMatrix<double> sparse_matrix = MakeBlockSparseMatrix();
  const MatrixXd dense_matrix = sparse_matrix.MakeDenseMatrix();

  /* Set the destinations to compatible-sized non-zero matrices. */
  MatrixXd y1 = MakeArbitraryMatrix(12, 6);
  MatrixXd y2 = y1;

  sparse_matrix.MultiplyAndAddTo(A, &y1);
  y2 += dense_matrix * A;
  EXPECT_TRUE(CompareMatrices(y1, y2));
}

GTEST_TEST(Block3x1SparseMatrixTest, LeftMultiplyAndAddTo) {
  const Block3x1SparseMatrix<double> sparse_matrix = MakeBlockSparseMatrix();
  const MatrixXd dense_matrix = sparse_matrix.MakeDenseMatrix();

  const MatrixXd A = MakeArbitraryMatrix(6, 12);

  /* Set the destinations to compatible-sized non-zero matrices. */
  MatrixXd y1 = MakeArbitraryMatrix(6, 7);
  MatrixXd y2 = y1;

  sparse_matrix.LeftMultiplyAndAddTo(A, &y1);
  y2 += A * dense_matrix;
  EXPECT_TRUE(CompareMatrices(y1, y2));
}

GTEST_TEST(Block3x1SparseMatrixTest, TransposeAndMultiplyAndAddTo) {
  const Block3x1SparseMatrix<double> sparse_matrix = MakeBlockSparseMatrix();
  const MatrixXd dense_matrix = sparse_matrix.MakeDenseMatrix();

  /* Test multiplication with dense. */
  const MatrixXd A = MakeArbitraryMatrix(12, 6);
  /* Set the destinations to compatible-sized non-zero matrices. */
  MatrixXd y1 = MakeArbitraryMatrix(7, 6);
  MatrixXd y2 = y1;

  sparse_matrix.TransposeAndMultiplyAndAddTo(A, &y1);
  y2 += dense_matrix.transpose() * A;
  EXPECT_TRUE(CompareMatrices(y1, y2));

  /* Test multiplication with 3x1 sparse. */
  Block3x1SparseMatrix<double> A_3x1sparse(4, 6);
  {
    std::vector<Block3x1SparseMatrix<double>::Triplet> triplets;
    triplets.emplace_back(0, 1, MakeVector(4.0));
    triplets.emplace_back(2, 1, MakeVector(5.0));
    A_3x1sparse.SetFromTriplets(triplets);
  }
  sparse_matrix.TransposeAndMultiplyAndAddTo(A_3x1sparse, &y1);
  y2 += dense_matrix.transpose() * A_3x1sparse.MakeDenseMatrix();
  EXPECT_TRUE(CompareMatrices(y1, y2));

  /* Test multiplication with 3x3 sparse. */
  Block3x3SparseMatrix<double> A_3x3sparse(4, 2);
  {
    std::vector<Block3x3SparseMatrix<double>::Triplet> triplets;
    triplets.emplace_back(0, 1, Matrix3d::Constant(4.0));
    triplets.emplace_back(2, 1, Matrix3d::Constant(5.0));
    A_3x3sparse.SetFromTriplets(triplets);
  }
  sparse_matrix.TransposeAndMultiplyAndAddTo(A_3x3sparse, &y1);
  y2 += dense_matrix.transpose() * A_3x3sparse.MakeDenseMatrix();
  EXPECT_TRUE(CompareMatrices(y1, y2));
}

GTEST_TEST(Block3x1SparseMatrixTest, MultiplyWithScaledTransposeAndAddTo) {
  const Block3x1SparseMatrix<double> sparse_matrix = MakeBlockSparseMatrix();
  const MatrixXd dense_matrix = sparse_matrix.MakeDenseMatrix();
  const VectorXd scale = VectorXd::LinSpaced(7, 0.0, 1.0);

  /* Set the destinations to compatible-sized non-zero matrices. */
  MatrixXd y1 = MakeArbitraryMatrix(12, 12);
  MatrixXd y2 = y1;

  sparse_matrix.MultiplyWithScaledTransposeAndAddTo(scale, &y1);
  y2 += dense_matrix * scale.asDiagonal() * dense_matrix.transpose();
  EXPECT_TRUE(CompareMatrices(y1, y2));
}

}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
