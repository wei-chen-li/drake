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

template <typename T>
std::vector<int> NumNonzeroPerColumn(
    const Block4x4SparseSymmetricMatrix<T>& mat) {
  std::vector<int> num_nonzero_block_per_block_column(mat.block_cols(), 0);
  for (int j = 0; j < mat.block_cols(); ++j) {
    for (int i : mat.sparsity_pattern().neighbors()[j]) {  // i ≥ j
      ++num_nonzero_block_per_block_column[j];
      if (i != j) ++num_nonzero_block_per_block_column[i];
    }
  }
  std::vector<int> num_nonzero_per_column(mat.cols());
  for (int j = 0; j < mat.block_cols(); ++j) {
    for (int s = 0; s < 4; ++s) {
      num_nonzero_per_column[4 * j + s] =
          num_nonzero_block_per_block_column[j] * 4;
    }
  }
  return num_nonzero_per_column;
}

}  // namespace

template <typename T>
void Convert(const Block4x4SparseSymmetricMatrix<T>& source,
             Eigen::SparseMatrix<T>* dest) {
  DRAKE_THROW_UNLESS(dest != nullptr);
  DRAKE_THROW_UNLESS(dest->rows() == dest->cols());
  DRAKE_THROW_UNLESS(dest->rows() != 0);
  const int num_dofs = dest->rows();
  DRAKE_THROW_UNLESS(source.rows() == num_dofs ||
                     source.rows() == num_dofs + 1);

  dest->setZero();
  dest->reserve(NumNonzeroPerColumn(source));
  for (int block_j = 0; block_j < source.block_cols(); ++block_j) {
    for (int block_i : source.sparsity_pattern().neighbors()[block_j]) {
      // block_i ≥ block_j
      const Matrix4<T>& block = source.block(block_i, block_j);
      for (int u = 0; u < 4; ++u) {
        for (int v = 0; v < 4; ++v) {
          const int i = 4 * block_i + u;
          const int j = 4 * block_j + v;
          if (i >= num_dofs || j >= num_dofs) {
            DRAKE_ASSERT(block(u, v) == 0.0);
            continue;
          }
          dest->insert(i, j) = block(u, v);
          if (block_i == block_j) continue;
          dest->insert(j, i) = block(u, v);
        }
      }
    }
  }
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
    const Eigen::SparseMatrix<T>& mat,
    const std::unordered_set<int> participating_dofs) {
  DRAKE_THROW_UNLESS(mat.rows() == mat.cols());
  const int num_dofs = mat.rows();
  DRAKE_THROW_UNLESS(std::all_of(participating_dofs.begin(),
                                 participating_dofs.end(), [&](int dof) {
                                   return 0 <= dof && dof < num_dofs;
                                 }));
  const int num_participating_dofs = participating_dofs.size();
  std::vector<int> permuted_dof_indexes(num_dofs, -1);
  int permuted_dof_index = 0;
  for (int dof = 0; dof < num_dofs; ++dof) {
    if (participating_dofs.contains(dof))
      permuted_dof_indexes[dof] = permuted_dof_index++;
  }
  PartialPermutation permutation(std::move(permuted_dof_indexes));
  permutation.ExtendToFullPermutation();

  std::vector<Eigen::Triplet<T>> A_triplets;
  std::vector<Eigen::Triplet<T>> Bt_triplets;
  std::vector<Eigen::Triplet<T>> D_triplets;

  for (int k = 0; k < mat.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it) {
      const int i = it.row();
      const int j = it.col();
      const int permuted_i = permutation.permuted_index(i);
      const int permuted_j = permutation.permuted_index(j);
      if (permuted_i < num_participating_dofs &&
          permuted_j < num_participating_dofs) {
        A_triplets.emplace_back(permuted_i, permuted_j, it.value());
      } else if (permuted_i >= num_participating_dofs &&
                 permuted_j >= num_participating_dofs) {
        D_triplets.emplace_back(permuted_i - num_participating_dofs,
                                permuted_j - num_participating_dofs,
                                it.value());
      } else if (permuted_i >= num_participating_dofs &&
                 permuted_j < num_participating_dofs) {
        Bt_triplets.emplace_back(permuted_i - num_participating_dofs,
                                 permuted_j, it.value());
      }
    }
  }

  Eigen::SparseMatrix<T> A(num_participating_dofs, num_participating_dofs);
  A.setFromTriplets(A_triplets.begin(), A_triplets.end());
  Eigen::SparseMatrix<T> Bt(num_dofs - num_participating_dofs,
                            num_participating_dofs);
  Bt.setFromTriplets(Bt_triplets.begin(), Bt_triplets.end());
  Eigen::SparseMatrix<T> D(num_dofs - num_participating_dofs,
                           num_dofs - num_participating_dofs);
  D.setFromTriplets(D_triplets.begin(), D_triplets.end());
  return SchurComplement<T>(A, Bt, D);
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS((&Convert<T>));

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&ComputeSchurComplement<T>));

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
