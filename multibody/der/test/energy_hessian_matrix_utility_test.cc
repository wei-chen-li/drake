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

  Block4x4SparseSymmetricMatrix<double> MakeSPDBlockSparseMatrix() {
    Block4x4SparseSymmetricMatrix<double> mat = MakeEnergyHessianMatrix<double>(
        has_closed_ends_, num_nodes_, num_edges_);
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
    return mat;
  }

  Eigen::SparseMatrix<double> MakeSPDSparseMatrix() {
    const Block4x4SparseSymmetricMatrix<double> mat =
        MakeSPDBlockSparseMatrix();
    Eigen::SparseMatrix<double> result(num_dofs_, num_dofs_);
    Convert(mat, &result);
    return result;
  }

  const bool has_closed_ends_ = GetParam();
  const int num_nodes_ = 8;
  const int num_edges_ = has_closed_ends_ ? num_nodes_ : num_nodes_ - 1;
  const int num_dofs_ = num_nodes_ * 3 + num_edges_;
};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds, EnergyHessianMatrixUtilTest,
                         ::testing::Values(false, true));

TEST_P(EnergyHessianMatrixUtilTest, Convert) {
  const Block4x4SparseSymmetricMatrix<double> source =
      MakeSPDBlockSparseMatrix();
  Eigen::SparseMatrix<double> dest(num_dofs_, num_dofs_);
  Convert<double>(source, &dest);
  EXPECT_TRUE(CompareMatrices(
      MatrixXd(dest),
      source.MakeDenseMatrix().topLeftCorner(num_dofs_, num_dofs_)));
}

TEST_P(EnergyHessianMatrixUtilTest, ComputeSchurComplement) {
  const std::unordered_set<int> participating_dofs = {2, 3, 5, 7, 11, 13, 17};
  const Eigen::SparseMatrix<double> mat = MakeSPDSparseMatrix();
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
  const MatrixXd permuted_mat = P * MatrixXd(mat) * P.transpose();

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
