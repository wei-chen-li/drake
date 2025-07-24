#include "drake/multibody/der/energy_hessian_matrix.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/limit_malloc.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using test::LimitMalloc;

/* Friend class for accessing EnergyHessianMatrix private members. */
class EnergyHessianMatrixTester {
 public:
  EnergyHessianMatrixTester() = delete;

  template <typename T>
  static const contact_solvers::internal::Block4x4SparseSymmetricMatrix<T>&
  get_data(const EnergyHessianMatrix<T>& hessian) {
    return hessian.data_;
  }

  template <typename T>
  static contact_solvers::internal::Block4x4SparseSymmetricMatrix<T>& get_data(
      EnergyHessianMatrix<T>* hessian) {
    DRAKE_THROW_UNLESS(hessian != nullptr);
    return hessian->data_;
  }
};

namespace {

class EnergyHessianMatrixTest : public ::testing::TestWithParam<bool> {
 private:
  void SetUp() override { srand(0); }

 protected:
  EnergyHessianMatrix<double> MakeRandomMatrix() {
    const bool has_closed_ends = GetParam();
    const int num_nodes = 8;
    const int num_edges = has_closed_ends ? num_nodes : num_nodes - 1;
    const int num_dofs = num_nodes * 3 + num_edges;

    EnergyHessianMatrix<double> result =
        EnergyHessianMatrix<double>::Allocate(num_dofs);
    EXPECT_EQ(result.rows(), num_dofs);
    contact_solvers::internal::Block4x4SparseSymmetricMatrix<double>& data =
        EnergyHessianMatrixTester::get_data(&result);

    for (int j = 0; j < data.block_cols(); ++j) {
      for (int i : data.sparsity_pattern().neighbors()[j]) {  // i ≥ j
        Eigen::Matrix4d block = Eigen::Matrix4d::Random();
        if (i == j) {
          block = 0.5 * (block + block.transpose()).eval() +
                  10 * Eigen::Matrix4d::Identity();
        }
        if (!has_closed_ends) {
          if (i == data.block_rows() - 1)
            block.template bottomRows<1>().setZero();
          if (j == data.block_cols() - 1)
            block.template rightCols<1>().setZero();
        }
        data.SetBlock(i, j, block);
      }
    }
    return result;
  }
};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds, EnergyHessianMatrixTest,
                         ::testing::Values(false, true));

TEST_P(EnergyHessianMatrixTest, Allocate) {
  const bool has_closed_ends = GetParam();
  const int num_nodes = 301;
  const int num_edges = has_closed_ends ? num_nodes : num_nodes - 1;
  const int num_internal_nodes = has_closed_ends ? num_nodes : num_nodes - 2;
  const int num_dofs = num_nodes * 3 + num_edges;

  /* pattern[block_i] stores the set of block_j indices that are nonzero. */
  std::vector<std::set<int>> pattern(num_nodes);

  for (int i = 0; i < num_edges; ++i) {
    const int node_i = i;
    const int node_ip1 = (i + 1) % num_nodes;
    pattern[node_i].insert(node_i);
    pattern[node_i].insert(node_ip1);
    pattern[node_ip1].insert(node_i);
    pattern[node_ip1].insert(node_ip1);
  }
  for (int i = 0; i < num_internal_nodes; ++i) {
    const int node_i = i;
    const int node_ip1 = (i + 1) % num_nodes;
    const int node_ip2 = (i + 2) % num_nodes;
    pattern[node_i].insert(node_i);
    pattern[node_i].insert(node_ip1);
    pattern[node_i].insert(node_ip2);
    pattern[node_ip1].insert(node_i);
    pattern[node_ip1].insert(node_ip1);
    pattern[node_ip1].insert(node_ip2);
    pattern[node_ip1].insert(node_i);
    pattern[node_ip2].insert(node_ip1);
    pattern[node_ip2].insert(node_ip2);

    const int edge_i = i;
    const int edge_ip1 = (i + 1) % num_edges;
    pattern[edge_i].insert(edge_i);
    pattern[edge_i].insert(edge_ip1);
    pattern[edge_ip1].insert(edge_i);
    pattern[edge_ip1].insert(edge_ip1);

    pattern[node_i].insert(edge_i);
    pattern[node_i].insert(edge_ip1);
    pattern[node_ip1].insert(edge_i);
    pattern[node_ip1].insert(edge_ip1);
    pattern[node_ip2].insert(edge_i);
    pattern[node_ip2].insert(edge_ip1);

    pattern[edge_i].insert(node_i);
    pattern[edge_i].insert(node_ip1);
    pattern[edge_i].insert(node_ip2);
    pattern[edge_ip1].insert(node_i);
    pattern[edge_ip1].insert(node_ip1);
    pattern[edge_ip1].insert(node_ip2);
  }

  EnergyHessianMatrix<double> hessian =
      EnergyHessianMatrix<double>::Allocate(num_dofs);
  const contact_solvers::internal::Block4x4SparseSymmetricMatrix<double>& data =
      EnergyHessianMatrixTester::get_data(hessian);

  for (int i = 0; i < ssize(pattern); ++i) {
    std::set<int>& row_pattern = pattern[i];

    /* Remove indices in row_pattern that are smaller than i (leave only upper
     triangle indices). */
    row_pattern.erase(row_pattern.begin(), row_pattern.lower_bound(i));

    EXPECT_THAT(data.sparsity_pattern().neighbors()[i],
                ::testing::UnorderedElementsAreArray(row_pattern));
  }

  /* Check that the matrix is initialized to zero. And if the data size is
   larger than the hessian size, the last diagonal entry is set to 1.0. */
  for (int test = 0; test < 1; ++test) {
    if (data.rows() == hessian.rows()) {
      EXPECT_TRUE(data.MakeDenseMatrix().isZero());
    } else {
      EXPECT_EQ(data.rows(), num_dofs + 1);
      EXPECT_TRUE(data.MakeDenseMatrix().leftCols(num_dofs).isZero());
      EXPECT_TRUE(data.MakeDenseMatrix().topRows(num_dofs).isZero());
      EXPECT_EQ(data.MakeDenseMatrix()(num_dofs, num_dofs), 1.0);
    }

    /* Setting the EnergyHessianMatrix to zero should retain the extra
     diagonal 1.0 entry. */
    hessian.SetZero();
  }
}

TEST_P(EnergyHessianMatrixTest, MatrixVectorProduct) {
  const EnergyHessianMatrix<double> mat = MakeRandomMatrix();
  const VectorXd vec = VectorXd::LinSpaced(mat.cols(), 0.0, 1.0);

  VectorXd result = VectorXd::Ones(mat.rows());
  {
    LimitMalloc guard;
    result += mat * vec * 1.2;
  }

  const VectorXd expected =
      VectorXd::Ones(mat.rows()) + mat.MakeDenseMatrix() * vec * 1.2;
  EXPECT_TRUE(CompareMatrices(result, expected, 1e-12));
}

TEST_P(EnergyHessianMatrixTest, Diagonal) {
  const EnergyHessianMatrix<double> mat = MakeRandomMatrix();
  EXPECT_TRUE(
      CompareMatrices(mat.Diagonal(), mat.MakeDenseMatrix().diagonal()));
}

TEST_P(EnergyHessianMatrixTest, AddScaledDiagonalMatrix) {
  EnergyHessianMatrix<double> lhs = MakeRandomMatrix();
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> rhs(
      VectorXd::LinSpaced(lhs.rows(), 0.0, 1.0));
  const double scale = 1.23;

  const MatrixXd expected =
      lhs.MakeDenseMatrix() + (rhs * scale).toDenseMatrix();

  {
    LimitMalloc guard;
    lhs.AddScaledMatrix(rhs, scale);
  }
  EXPECT_TRUE(CompareMatrices(lhs.MakeDenseMatrix(), expected, 1e-12));
}

TEST_P(EnergyHessianMatrixTest, AddScaledEnergyHessianMatrix) {
  EnergyHessianMatrix<double> lhs = MakeRandomMatrix();
  const EnergyHessianMatrix<double> rhs = MakeRandomMatrix();
  const double scale = 1.23;

  const MatrixXd expected =
      lhs.MakeDenseMatrix() + rhs.MakeDenseMatrix() * scale;

  {
    LimitMalloc guard;
    lhs.AddScaledMatrix(rhs, scale);
  }
  EXPECT_TRUE(CompareMatrices(lhs.MakeDenseMatrix(), expected, 1e-12));
}

TEST_P(EnergyHessianMatrixTest, ComputeSchurComplement) {
  const std::unordered_set<int> participating_dofs = {2, 3, 5, 7, 11, 13, 17};
  const EnergyHessianMatrix<double> mat = MakeRandomMatrix();
  const SchurComplement<double> schur_complement =
      mat.ComputeSchurComplement(participating_dofs);

  const int num_dofs = mat.rows();
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(num_dofs);
  int permuted_index = 0;
  for (int i = 0; i < num_dofs; ++i) {
    if (participating_dofs.contains(i)) P.indices()[i] = permuted_index++;
  }
  for (int i = 0; i < num_dofs; ++i) {
    if (!participating_dofs.contains(i)) P.indices()[i] = permuted_index++;
  }
  const MatrixXd permuted_mat = P * mat.MakeDenseMatrix() * P.transpose();

  const int num_participating_dofs = participating_dofs.size();
  const MatrixXd A = permuted_mat.topLeftCorner(num_participating_dofs,
                                                num_participating_dofs);
  const MatrixXd B = permuted_mat.topRightCorner(
      num_participating_dofs, num_dofs - num_participating_dofs);
  const MatrixXd D = permuted_mat.bottomRightCorner(
      num_dofs - num_participating_dofs, num_dofs - num_participating_dofs);

  EXPECT_TRUE(CompareMatrices(schur_complement.get_D_complement(),
                              A - B * D.inverse() * B.transpose(), 1e-12));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
