#include "drake/multibody/der/energy_hessian_matrix_utility.h"

#include <algorithm>

#include "drake/common/default_scalars.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

namespace {

using multibody::contact_solvers::internal::PartialPermutation;

/* Reports if the last row and last column of `mat` are all zero. */
template <typename T>
[[nodiscard]] bool IsLastRowAndColZero(
    const Block4x4SparseSymmetricMatrix<T>& mat) {
  for (int j = 0; j < mat.block_cols(); ++j) {
    for (int i : mat.sparsity_pattern().neighbors()[j]) {  // i ≥ j
      if (i < mat.block_rows() - 1) continue;
      const Eigen::Matrix4d block = ExtractDoubleOrThrow(mat.block(i, j));
      if (i == j) {
        if (!block.template rightCols<1>().isZero() ||
            !block.template bottomRows<1>().isZero())
          return false;
      } else {
        if (!block.template bottomRows<1>().isZero()) return false;
      }
    }
  }
  return true;
}

/* Validates if the number of rows of `mat` is equal to `num_dofs` or
 `num_dofs + 1`. If number of rows equal `num_dofs + 1`, assert that the last
 row and last column of `mat` are all zero. */
template <typename T>
void ValidateMatrixSize(const Block4x4SparseSymmetricMatrix<T>& mat,
                        int num_dofs) {
  DRAKE_THROW_UNLESS(num_dofs > 0);
  DRAKE_THROW_UNLESS(mat.rows() == mat.cols());
  DRAKE_THROW_UNLESS(mat.rows() == num_dofs || mat.rows() == num_dofs + 1);
  if (mat.rows() == num_dofs + 1) DRAKE_ASSERT(IsLastRowAndColZero(mat));
}

}  // namespace

template <typename T>
Block4x4SparseSymmetricMatrixVectorProduct<T>::
    Block4x4SparseSymmetricMatrixVectorProduct(
        const Block4x4SparseSymmetricMatrix<T>* mat,
        const Eigen::VectorX<T>* vec, const T& scale)
    : mat_(mat), vec_(vec), scale_(scale) {
  DRAKE_THROW_UNLESS(mat != nullptr);
  DRAKE_THROW_UNLESS(vec != nullptr);
  ValidateMatrixSize(*mat, vec->size());
}

template <typename T>
Eigen::Ref<Eigen::VectorX<T>>
Block4x4SparseSymmetricMatrixVectorProduct<T>::AddToVector(
    EigenPtr<Eigen::VectorX<T>> lhs) const {
  DRAKE_THROW_UNLESS(lhs->size() == vec_->size());
  const bool size_off_by_1 = mat_->rows() != vec_->size();

  for (int j = 0; j < mat_->block_cols(); ++j) {
    for (int i : mat_->sparsity_pattern().neighbors()[j]) {  // i ≥ j
      if (!size_off_by_1 || i < mat_->block_rows() - 1) {
        if (i == j) {
          lhs->template segment<4>(4 * i) +=
              mat_->block(i, j) * vec_->template segment<4>(4 * j) * scale_;
        } else {
          lhs->template segment<4>(4 * i) +=
              mat_->block(i, j) * vec_->template segment<4>(4 * j) * scale_;
          lhs->template segment<4>(4 * j) += mat_->block(i, j).transpose() *
                                             vec_->template segment<4>(4 * i) *
                                             scale_;
        }
      } else {
        if (i == j) {
          lhs->template segment<3>(4 * i) +=
              mat_->block(i, j).template topLeftCorner<3, 3>() *
              vec_->template segment<3>(4 * j) * scale_;
        } else {
          lhs->template segment<3>(4 * i) +=
              mat_->block(i, j).template topLeftCorner<3, 4>() *
              vec_->template segment<4>(4 * j) * scale_;
          lhs->template segment<4>(4 * j) +=
              mat_->block(i, j).template topLeftCorner<3, 4>().transpose() *
              vec_->template segment<3>(4 * i) * scale_;
        }
      }
    }
  }
  return *lhs;
}

template <typename T>
void AddScaledMatrix(Block4x4SparseSymmetricMatrix<T>* lhs,
                     const Eigen::DiagonalMatrix<T, Eigen::Dynamic>& rhs,
                     const T& scale) {
  DRAKE_THROW_UNLESS(lhs != nullptr);
  ValidateMatrixSize(*lhs, rhs.rows());
  const bool size_off_by_1 = lhs->rows() != rhs.rows();
  if (scale == 0.0) return;

  for (int i = 0; i < lhs->block_rows(); ++i) {
    if (!size_off_by_1 || i < lhs->block_rows() - 1) {
      auto vec = rhs.diagonal().template segment<4>(4 * i) * scale;
      lhs->AddToBlock(i, i, vec.asDiagonal().toDenseMatrix());
    } else {
      Eigen::Vector4<T> vec = Eigen::Vector4<T>::Zero();
      vec.template segment<3>(0) =
          rhs.diagonal().template segment<3>(4 * i) * scale;
      lhs->AddToBlock(i, i, vec.asDiagonal().toDenseMatrix());
    }
  }
}

template <typename T>
void AddScaledMatrix(Block4x4SparseSymmetricMatrix<T>* lhs,
                     const Block4x4SparseSymmetricMatrix<T>& rhs,
                     const T& scale) {
  DRAKE_THROW_UNLESS(lhs != nullptr);
  DRAKE_THROW_UNLESS(lhs->rows() == rhs.rows() && lhs->cols() == rhs.cols());
  if (scale == 0.0) return;
  for (int j = 0; j < rhs.block_cols(); ++j) {
    for (int i : rhs.sparsity_pattern().neighbors()[j]) {  // i ≥ j
      lhs->AddToBlock(i, j, rhs.block(i, j) * scale);
    }
  }
}

template <typename T>
Block4x4SparseSymmetricMatrix<T> SumMatrices(
    const Block4x4SparseSymmetricMatrix<T>& mat1, const T& scalar1,
    const Block4x4SparseSymmetricMatrix<T>& mat2, const T& scalar2) {
  DRAKE_THROW_UNLESS(mat1.rows() == mat2.rows());
  const int block_rows = mat1.block_rows();

  std::vector<std::set<int>> pattern(block_rows);
  for (int i = 0; i < block_rows; ++i) {
    pattern[i].insert(mat1.sparsity_pattern().neighbors()[i].begin(),
                      mat1.sparsity_pattern().neighbors()[i].end());
    pattern[i].insert(mat2.sparsity_pattern().neighbors()[i].begin(),
                      mat2.sparsity_pattern().neighbors()[i].end());
  }
  std::vector<std::vector<int>> neighbors(block_rows);
  for (int i = 0; i < block_rows; ++i) {
    neighbors[i].insert(neighbors[i].end(), pattern[i].begin(),
                        pattern[i].end());
  }
  std::vector<int> block_sizes(block_rows, 4);
  contact_solvers::internal::BlockSparsityPattern block_sparsity_pattern(
      std::move(block_sizes), std::move(neighbors));
  Block4x4SparseSymmetricMatrix<T> result(std::move(block_sparsity_pattern));

  for (int j = 0; j < mat1.block_cols(); ++j) {
    for (int i : mat1.sparsity_pattern().neighbors()[j]) {  // i ≥ j
      result.AddToBlock(i, j, mat1.block(i, j) * scalar1);
    }
  }
  for (int j = 0; j < mat2.block_cols(); ++j) {
    for (int i : mat2.sparsity_pattern().neighbors()[j]) {  // i ≥ j
      result.AddToBlock(i, j, mat2.block(i, j) * scalar2);
    }
  }
  return result;
}

template <typename T>
static void FillInTriplets(int block_i, int block_j,
                           const Eigen::Ref<const Matrix4<T>>& block,
                           const PartialPermutation& permutation, int A_size,
                           std::vector<Eigen::Triplet<T>>* A_triplets,
                           std::vector<Eigen::Triplet<T>>* Bt_triplets,
                           std::vector<Eigen::Triplet<T>>* D_triplets) {
  constexpr int block_size = 4;
  for (int i = 0; i < block_size; ++i) {
    for (int j = 0; j < block_size; ++j) {
      const int dof_i = block_size * block_i + i;
      const int dof_j = block_size * block_j + j;
      const int permuted_dof_i = permutation.permuted_index(dof_i);
      const int permuted_dof_j = permutation.permuted_index(dof_j);
      if (permuted_dof_i < A_size && permuted_dof_j < A_size) {
        A_triplets->emplace_back(permuted_dof_i, permuted_dof_j, block(i, j));
      } else if (permuted_dof_i >= A_size && permuted_dof_j >= A_size) {
        D_triplets->emplace_back(permuted_dof_i - A_size,
                                 permuted_dof_j - A_size, block(i, j));
      } else if (permuted_dof_i >= A_size && permuted_dof_j < A_size) {
        Bt_triplets->emplace_back(permuted_dof_i - A_size, permuted_dof_j,
                                  block(i, j));
      } else {  // Do nothing since we don't need to fill in B.
      }
    }
  }
}

template <typename T>
SchurComplement<T> ComputeSchurComplement(
    const Block4x4SparseSymmetricMatrix<T>& mat,
    const std::unordered_set<int> participating_dofs) {
  const int num_rows = mat.rows();
  DRAKE_THROW_UNLESS(std::all_of(participating_dofs.begin(),
                                 participating_dofs.end(), [&](int dof) {
                                   return 0 <= dof && dof < num_rows;
                                 }));
  const int num_participating_dofs = participating_dofs.size();
  std::vector<int> permuted_dof_indexes(num_rows, -1);
  int permuted_dof_index = 0;
  for (int dof = 0; dof < num_rows; ++dof) {
    if (participating_dofs.contains(dof))
      permuted_dof_indexes[dof] = permuted_dof_index++;
  }
  PartialPermutation permutation(std::move(permuted_dof_indexes));
  permutation.ExtendToFullPermutation();

  std::vector<Eigen::Triplet<T>> A_triplets;
  std::vector<Eigen::Triplet<T>> Bt_triplets;
  std::vector<Eigen::Triplet<T>> D_triplets;

  for (int block_j = 0; block_j < mat.block_cols(); ++block_j) {
    for (int block_i :
         mat.sparsity_pattern().neighbors()[block_j]) {  // block_i ≥ block_j
      const Matrix4<T>& block = mat.block(block_i, block_j);
      FillInTriplets<T>(block_i, block_j, block, permutation,
                        num_participating_dofs, &A_triplets, &Bt_triplets,
                        &D_triplets);
      if (block_i == block_j) continue;
      FillInTriplets<T>(block_j, block_i, block.transpose(), permutation,
                        num_participating_dofs, &A_triplets, &Bt_triplets,
                        &D_triplets);
    }
  }

  Eigen::SparseMatrix<T> A(num_participating_dofs, num_participating_dofs);
  A.setFromTriplets(A_triplets.begin(), A_triplets.end());
  Eigen::SparseMatrix<T> Bt(num_rows - num_participating_dofs,
                            num_participating_dofs);
  Bt.setFromTriplets(Bt_triplets.begin(), Bt_triplets.end());
  Eigen::SparseMatrix<T> D(num_rows - num_participating_dofs,
                           num_rows - num_participating_dofs);
  D.setFromTriplets(D_triplets.begin(), D_triplets.end());
  return SchurComplement<T>(A, Bt, D);
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (static_cast<void (*)(Block4x4SparseSymmetricMatrix<T>*,
                          const Eigen::DiagonalMatrix<T, Eigen::Dynamic>&,
                          const T&)>(&AddScaledMatrix<T>),
     static_cast<void (*)(Block4x4SparseSymmetricMatrix<T>*,
                          const Block4x4SparseSymmetricMatrix<T>&,  //
                          const T&)>(&AddScaledMatrix<T>),
     &SumMatrices<T>));

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&ComputeSchurComplement<T>));

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::
        Block4x4SparseSymmetricMatrixVectorProduct);
