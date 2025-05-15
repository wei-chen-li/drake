#include "drake/multibody/der/energy_hessian_matrix_util.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/multibody/der/elastic_energy.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using Eigen::VectorXd;
using test::LimitMalloc;

class EnergyHessianMatrixUtilTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override { srand(0); }

  Block4x4SparseSymmetricMatrix<double> MakeRandomMatrix() {
    Block4x4SparseSymmetricMatrix<double> mat =
        MakeEnergyHessianMatrix<double>(has_closed_ends_, num_nodes_);
    EXPECT_EQ(mat.rows(), has_closed_ends_ ? num_dofs_ : num_dofs_ + 1);
    for (int j = 0; j < mat.block_cols(); ++j) {
      for (int i : mat.sparsity_pattern().neighbors()[j]) {  // i ≥ j
        Eigen::Matrix4d block = Eigen::Matrix4d::Random();
        if (i == j) block = 0.5 * (block + block.transpose()).eval();
        if (!has_closed_ends_) {
          if (i == mat.block_rows() - 1)
            block.template bottomRows<1>().setZero();
          if (j == mat.block_cols() - 1)
            block.template rightCols<1>().setZero();
        }
        mat.SetBlock(i, j, block);
      }
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
  EXPECT_TRUE(CompareMatrices(result, expected, 1e-15));
}

TEST_P(EnergyHessianMatrixUtilTest, AddDiagonalMatrix) {
  Block4x4SparseSymmetricMatrix<double> lhs = MakeRandomMatrix();
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> rhs(
      VectorXd::LinSpaced(num_dofs_, 0.0, 1.0));
  const double scale = 1.23;

  const Eigen::MatrixXd expected =
      lhs.MakeDenseMatrix().topLeftCorner(num_dofs_, num_dofs_) +
      (rhs * scale).toDenseMatrix();

  {
    LimitMalloc guard;
    AddScaledMatrix(&lhs, rhs, scale);
  }
  EXPECT_TRUE(CompareMatrices(
      lhs.MakeDenseMatrix().topLeftCorner(num_dofs_, num_dofs_), expected));
  if (lhs.rows() == num_dofs_ + 1) {
    EXPECT_TRUE(lhs.MakeDenseMatrix().rightCols<1>().isZero());
    EXPECT_TRUE(lhs.MakeDenseMatrix().bottomRows<1>().isZero());
  }
}

TEST_P(EnergyHessianMatrixUtilTest, AddBlock4x4SparseSymmetricMatrix) {
  Block4x4SparseSymmetricMatrix<double> lhs = MakeRandomMatrix();
  const Block4x4SparseSymmetricMatrix<double> rhs = MakeRandomMatrix();
  const double scale = 1.23;

  const Eigen::MatrixXd expected =
      (lhs.MakeDenseMatrix() + rhs.MakeDenseMatrix() * scale)
          .topLeftCorner(num_dofs_, num_dofs_);

  {
    LimitMalloc guard;
    AddScaledMatrix(&lhs, rhs, scale);
  }
  EXPECT_TRUE(CompareMatrices(
      lhs.MakeDenseMatrix().topLeftCorner(num_dofs_, num_dofs_), expected));
  if (lhs.rows() == num_dofs_ + 1) {
    EXPECT_TRUE(lhs.MakeDenseMatrix().rightCols<1>().isZero());
    EXPECT_TRUE(lhs.MakeDenseMatrix().bottomRows<1>().isZero());
  }
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
