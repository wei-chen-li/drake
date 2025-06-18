#include "drake/multibody/der/energy_hessian_matrix_utility.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/multibody/der/elastic_energy.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using test::LimitMalloc;

class EnergyHessianMatrixUtilTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override { srand(0); }

  Block4x4SparseSymmetricMatrix<double> MakeRandomMatrix(
      bool fill_empty_diagonal_with_one = false) {
    Block4x4SparseSymmetricMatrix<double> mat =
        MakeElasticEnergyHessianMatrix<double>(has_closed_ends_, num_nodes_,
                                               num_edges_);
    EXPECT_EQ(mat.rows(), has_closed_ends_ ? num_dofs_ : num_dofs_ + 1);
    for (int j = 0; j < mat.block_cols(); ++j) {
      for (int i : mat.sparsity_pattern().neighbors()[j]) {  // i ≥ j
        Eigen::Matrix4d block = Eigen::Matrix4d::Random();
        if (i == j) {
          block = 0.5 * (block + block.transpose()).eval() +
                  10 * Eigen::Matrix4d::Identity();
        }
        if (!has_closed_ends_) {
          if (i == mat.block_rows() - 1)
            block.template bottomRows<1>().setZero();
          if (j == mat.block_cols() - 1)
            block.template rightCols<1>().setZero();
        }
        mat.SetBlock(i, j, block);
      }
    }
    if (!has_closed_ends_ && fill_empty_diagonal_with_one) {
      const int i = mat.block_rows() - 1;
      Eigen::Matrix4d block = mat.block(i, i);
      block(3, 3) = 1.0;
      mat.SetBlock(i, i, block);
    }
    return mat;
  }

  const bool has_closed_ends_ = GetParam();
  const int num_nodes_ = 8;
  const int num_edges_ = has_closed_ends_ ? num_nodes_ : num_nodes_ - 1;
  const int num_dofs_ = num_nodes_ * 3 + num_edges_;
};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds, EnergyHessianMatrixUtilTest,
                         ::testing::Values(false, true));

TEST_P(EnergyHessianMatrixUtilTest, MatrixVectorProduct) {
  const Block4x4SparseSymmetricMatrix<double> mat = MakeRandomMatrix();
  const VectorXd vec = VectorXd::LinSpaced(num_dofs_, 0.0, 1.0);

  VectorXd result = VectorXd::Ones(num_dofs_);
  {
    LimitMalloc guard;
    result += mat * vec * 1.2;
  }

  const VectorXd expected =
      VectorXd::Ones(num_dofs_) +
      mat.MakeDenseMatrix().topLeftCorner(num_dofs_, num_dofs_) * vec * 1.2;
  EXPECT_TRUE(CompareMatrices(result, expected, 1e-12));
}

TEST_P(EnergyHessianMatrixUtilTest, AddDiagonalMatrix) {
  Block4x4SparseSymmetricMatrix<double> lhs = MakeRandomMatrix();
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> rhs(
      VectorXd::LinSpaced(num_dofs_, 0.0, 1.0));
  const double scale = 1.23;

  const MatrixXd expected =
      lhs.MakeDenseMatrix().topLeftCorner(num_dofs_, num_dofs_) +
      (rhs * scale).toDenseMatrix();

  {
    LimitMalloc guard;
    AddScaledMatrix(&lhs, rhs, scale);
  }
  EXPECT_TRUE(
      CompareMatrices(lhs.MakeDenseMatrix().topLeftCorner(num_dofs_, num_dofs_),
                      expected, 1e-12));
  if (lhs.rows() == num_dofs_ + 1) {
    EXPECT_TRUE(lhs.MakeDenseMatrix().rightCols<1>().isZero());
    EXPECT_TRUE(lhs.MakeDenseMatrix().bottomRows<1>().isZero());
  }
}

TEST_P(EnergyHessianMatrixUtilTest, AddBlock4x4SparseSymmetricMatrix) {
  Block4x4SparseSymmetricMatrix<double> lhs = MakeRandomMatrix();
  const Block4x4SparseSymmetricMatrix<double> rhs = MakeRandomMatrix();
  const double scale = 1.23;

  const MatrixXd expected =
      (lhs.MakeDenseMatrix() + rhs.MakeDenseMatrix() * scale)
          .topLeftCorner(num_dofs_, num_dofs_);

  {
    LimitMalloc guard;
    AddScaledMatrix(&lhs, rhs, scale);
  }
  EXPECT_TRUE(
      CompareMatrices(lhs.MakeDenseMatrix().topLeftCorner(num_dofs_, num_dofs_),
                      expected, 1e-12));
  if (lhs.rows() == num_dofs_ + 1) {
    EXPECT_TRUE(lhs.MakeDenseMatrix().rightCols<1>().isZero());
    EXPECT_TRUE(lhs.MakeDenseMatrix().bottomRows<1>().isZero());
  }
}

TEST_P(EnergyHessianMatrixUtilTest, SumMatrices) {
  const Block4x4SparseSymmetricMatrix<double> mat1 = MakeRandomMatrix();
  const Block4x4SparseSymmetricMatrix<double> mat2 = MakeRandomMatrix();
  const Block4x4SparseSymmetricMatrix<double> sum =
      SumMatrices(mat1, 1.2, mat2, 3.4);
  EXPECT_TRUE(CompareMatrices(
      sum.MakeDenseMatrix(),
      mat1.MakeDenseMatrix() * 1.2 + mat2.MakeDenseMatrix() * 3.4));
}

TEST_P(EnergyHessianMatrixUtilTest, ComputeSchurComplement) {
  const std::unordered_set<int> participating_dofs = {2, 3, 5, 7, 11, 13, 17};
  const Block4x4SparseSymmetricMatrix<double> mat =
      MakeRandomMatrix(/* fill_empty_diagonal_with_one = */ true);
  const SchurComplement<double> schur_complement =
      ComputeSchurComplement(mat, participating_dofs);

  const int num_rows = mat.rows();
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(num_rows);
  int permuted_index = 0;
  for (int i = 0; i < num_rows; ++i) {
    if (participating_dofs.contains(i)) P.indices()[i] = permuted_index++;
  }
  for (int i = 0; i < num_rows; ++i) {
    if (!participating_dofs.contains(i)) P.indices()[i] = permuted_index++;
  }
  const MatrixXd permuted_mat = P * mat.MakeDenseMatrix() * P.transpose();

  const int num_participating_dofs = participating_dofs.size();
  const MatrixXd A = permuted_mat.topLeftCorner(num_participating_dofs,
                                                num_participating_dofs);
  const MatrixXd B = permuted_mat.topRightCorner(
      num_participating_dofs, num_rows - num_participating_dofs);
  const MatrixXd D = permuted_mat.bottomRightCorner(
      num_rows - num_participating_dofs, num_rows - num_participating_dofs);

  EXPECT_TRUE(CompareMatrices(schur_complement.get_D_complement(),
                              A - B * D.inverse() * B.transpose(), 1e-12));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
