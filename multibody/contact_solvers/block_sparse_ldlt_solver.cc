#include "drake/multibody/contact_solvers/block_sparse_ldlt_solver.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "drake/multibody/contact_solvers/minimum_degree_ordering.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

template <typename BlockType>
BlockSparseLdltSolver<BlockType>::BlockSparseLdltSolver()
    : A_sparsity_pattern_({}, {}) {}

template <typename BlockType>
BlockSparseLdltSolver<BlockType>::~BlockSparseLdltSolver() = default;

template <typename BlockType>
void BlockSparseLdltSolver<BlockType>::SetMatrix(const SymmetricMatrix& A) {
  const BlockSparsityPattern& A_block_pattern = A.sparsity_pattern();
  /* Compute the elimination ordering using Minimum Degree algorithm. */
  const std::vector<int> elimination_ordering =
      ComputeMinimumDegreeOrdering(A_block_pattern);
  BlockSparsityPattern L_block_pattern =
      SymbolicFactor(A, elimination_ordering);
  SetMatrixImpl(A, elimination_ordering, std::move(L_block_pattern));
}

template <typename BlockType>
void BlockSparseLdltSolver<BlockType>::UpdateMatrix(const SymmetricMatrix& A) {
  PermuteAndCopyToL(A);
  solver_mode_ = SolverMode::kAnalyzed;
}

template <typename BlockType>
bool BlockSparseLdltSolver<BlockType>::Factor() {
  DRAKE_THROW_UNLESS(solver_mode_ == SolverMode::kAnalyzed);
  const bool success = ComputeFactorization();
  solver_mode_ = success ? SolverMode::kFactored : SolverMode::kEmpty;
  return success;
}

template <typename BlockType>
void BlockSparseLdltSolver<BlockType>::FactorForcePositiveDefinite(
    Scalar regularized_D_min_eigenvalue) {
  DRAKE_THROW_UNLESS(solver_mode_ == SolverMode::kAnalyzed);
  DRAKE_THROW_UNLESS(regularized_D_min_eigenvalue > 0);
  const bool success = ComputeFactorization(regularized_D_min_eigenvalue);
  DRAKE_DEMAND(success);
  solver_mode_ = SolverMode::kFactored;
}

template <typename BlockType>
VectorX<typename BlockSparseLdltSolver<BlockType>::Scalar>
BlockSparseLdltSolver<BlockType>::Solve(
    const Eigen::Ref<const VectorX<Scalar>>& b) const {
  DRAKE_THROW_UNLESS(solver_mode() == SolverMode::kFactored);
  DRAKE_THROW_UNLESS(b.size() == L_->cols());
  VectorX<Scalar> permuted_b(b.size());
  scalar_permutation_.Apply(b, &permuted_b);

  const BlockSparsityPattern& block_sparsity_pattern = L_->sparsity_pattern();
  const std::vector<int>& block_sizes = block_sparsity_pattern.block_sizes();
  const std::vector<int>& starting_cols = L_->starting_cols();

  /* Solve Lz = b in place. */
  for (int j = 0; j < L_->block_cols(); ++j) {
    /* The jᵗʰ block entry. */
    const VectorX<Scalar> zj =
        permuted_b.segment(starting_cols[j], block_sizes[j]);
    /* Eliminate for the jᵗʰ block entry from the system. */
    const auto& blocks_in_col_j = L_->block_row_indices(j);
    for (int flat = 1; flat < ssize(blocks_in_col_j); ++flat) {
      const int i = blocks_in_col_j[flat];
      permuted_b.segment(starting_cols[i], block_sizes[i]).noalias() -=
          L_->block_flat(flat, j) * zj;
    }
  }
  VectorX<Scalar>& permuted_z = permuted_b;

  /* Solve Dw = z in place. */
  for (int j = 0; j < L_->block_cols(); ++j) {
    permuted_b.segment(starting_cols[j], block_sizes[j]) =
        Dinv_[j] * permuted_b.segment(starting_cols[j], block_sizes[j]);
  }
  VectorX<Scalar>& permuted_w = permuted_z;

  /* Solve Lᵀx = w in place. */
  for (int j = L_->block_cols() - 1; j >= 0; --j) {
    /* Eliminate all solved variables. */
    const auto& blocks_in_col_j = L_->block_row_indices(j);
    for (int flat = 1; flat < ssize(blocks_in_col_j); ++flat) {
      const int i = blocks_in_col_j[flat];
      permuted_w.segment(starting_cols[j], block_sizes[j]).noalias() -=
          L_->block_flat(flat, j).transpose() *
          permuted_w.segment(starting_cols[i], block_sizes[i]);
    }
  }
  VectorX<Scalar>& permuted_x = permuted_w;

  VectorX<Scalar> x(permuted_x.size());
  scalar_permutation_.ApplyInverse(permuted_x, &x);
  return x;
}

template <typename BlockType>
typename BlockSparseLdltSolver<BlockType>::SymmetricMatrix
BlockSparseLdltSolver<BlockType>::A() const {
  DRAKE_THROW_UNLESS(solver_mode() == SolverMode::kFactored);

  /* Compute A = PᵀLP ⋅ PᵀDP ⋅ (PᵀL⋅P)ᵀ
               = ∑ₖ (PᵀLP)ᵢₖ (PᵀDP)ₖₖ ((PᵀL⋅P)ᵀ)ₖⱼ
               = ∑ₖ (PᵀLP)ᵢₖ (PᵀDP)ₖₖ ((PᵀL⋅P)ⱼₖ)ᵀ . */
  SymmetricMatrix A(A_sparsity_pattern_);
  for (int j = 0; j < A.block_cols(); ++j) {
    const int pj = block_permutation_.permuted_index(j);
    for (int i : A.block_row_indices(j)) {
      const int pi = block_permutation_.permuted_index(i);
      for (int pk = 0; pk <= std::min(pi, pj); ++pk) {
        if (!(L_->HasBlock(pi, pk) && L_->HasBlock(pj, pk))) continue;
        A.AddToBlock(
            i, j, L_->block(pi, pk) * D_[pk] * L_->block(pj, pk).transpose());
      }
    }
  }
  return A;
}

template <typename BlockType>
typename BlockSparseLdltSolver<BlockType>::LowerTriangularMatrix
BlockSparseLdltSolver<BlockType>::L() const {
  DRAKE_THROW_UNLESS(solver_mode() == SolverMode::kFactored);
  return *L_;
}

template <typename BlockType>
typename BlockSparseLdltSolver<BlockType>::SymmetricMatrix
BlockSparseLdltSolver<BlockType>::D() const {
  DRAKE_THROW_UNLESS(solver_mode() == SolverMode::kFactored);
  const std::vector<int>& block_sizes = L_->sparsity_pattern().block_sizes();
  std::vector<std::vector<int>> neighbors(block_sizes.size());
  for (int j = 0; j < ssize(block_sizes); ++j) neighbors[j].push_back(j);
  SymmetricMatrix result(
      BlockSparsityPattern(block_sizes, std::move(neighbors)));
  for (int j = 0; j < result.block_cols(); ++j) {
    result.SetBlock(j, j, D_[j]);
  }
  return result;
}

template <typename BlockType>
Eigen::PermutationMatrix<Eigen::Dynamic> BlockSparseLdltSolver<BlockType>::P()
    const {
  DRAKE_THROW_UNLESS(solver_mode() != SolverMode::kEmpty);
  const std::vector<int>& p = scalar_permutation_.permutation();
  return Eigen::PermutationMatrix<Eigen::Dynamic>(
      Eigen::Map<const VectorX<int>>(p.data(), p.size()));
}

template <typename BlockType>
void BlockSparseLdltSolver<BlockType>::SetMatrixImpl(
    const SymmetricMatrix& A, const std::vector<int>& elimination_ordering,
    BlockSparsityPattern&& L_pattern) {
  /* First documented responsibility: set `A_sparsity_pattern_`. */
  A_sparsity_pattern_ = A.sparsity_pattern();
  /* Second documented responsibility: set `block_permutation_`. */
  /* Construct the inverse of the elimination ordering, which permutes the
   original indices to new indices. */
  std::vector<int> permutation(elimination_ordering.size());
  for (int i = 0; i < ssize(permutation); ++i) {
    permutation[elimination_ordering[i]] = i;
  }
  block_permutation_ = PartialPermutation(std::move(permutation));
  /* Third documented responsibility: set `scalar_permutation_`. */
  SetScalarPermutation(A, elimination_ordering);
  /* Fourth documented responsibility: allocate `L_`, `D_` and `Dinv_`. */
  L_ = std::make_unique<LowerTriangularMatrix>(std::move(L_pattern));
  D_.resize(A.block_cols());
  Dinv_.resize(A.block_cols());
  /* Fifth documented responsibility: UpdateMatrix. */
  UpdateMatrix(A);
}

template <typename BlockType>
void BlockSparseLdltSolver<BlockType>::SetScalarPermutation(
    const SymmetricMatrix& A, const std::vector<int>& elimination_ordering) {
  /* It's easier to build the scalar elimination ordering first from block
   elimination ordering and then convert it to the scalar permutation (the
   inverse of the scalar elimination ordering) that induces the permutation P
   such that L⋅Lᵀ = P⋅A⋅Pᵀ.
   More specificially, Pᵢⱼ = 1 for j = scalar_elimination_ordering[i] (or
   equivalently i = scalar_permutation_[j]) and Pᵢⱼ = 0 otherwise. */
  std::vector<int> scalar_elimination_ordering(A.cols());
  {
    const BlockSparsityPattern& A_block_pattern = A.sparsity_pattern();
    const std::vector<int>& A_block_sizes = A_block_pattern.block_sizes();
    const std::vector<int>& starting_indices = A.starting_cols();
    int i_permuted = 0;
    for (int block_permuted = 0; block_permuted < ssize(elimination_ordering);
         ++block_permuted) {
      const int block = elimination_ordering[block_permuted];
      const int start = starting_indices[block];
      const int size = A_block_sizes[block];
      for (int i = start; i < start + size; ++i) {
        scalar_elimination_ordering[i_permuted++] = i;
      }
    }
  }
  /* Invert the elimination ordering to get the permutation. */
  std::vector<int> scalar_permutation(scalar_elimination_ordering.size());
  for (int i_permuted = 0; i_permuted < ssize(scalar_permutation);
       ++i_permuted) {
    scalar_permutation[scalar_elimination_ordering[i_permuted]] = i_permuted;
  }
  scalar_permutation_ = PartialPermutation(std::move(scalar_permutation));
}

template <typename BlockType>
BlockSparsityPattern BlockSparseLdltSolver<BlockType>::SymbolicFactor(
    const SymmetricMatrix& A, const std::vector<int>& elimination_ordering) {
  /* 1. Compute the block permutation as well as the scalar permutation. */
  const int n = elimination_ordering.size();
  /* Construct the inverse of the elimination ordering, which permutes the
   original indices to new indices. */
  std::vector<int> permutation(n);
  for (int i = 0; i < n; ++i) {
    permutation[elimination_ordering[i]] = i;
  }
  const PartialPermutation block_permutation(std::move(permutation));

  /* Find the sparsity pattern of the permuted A (under the permutation induced
   by the elimination ordering). */
  const BlockSparsityPattern& A_block_pattern = A.sparsity_pattern();
  const std::vector<int>& A_block_sizes = A_block_pattern.block_sizes();
  const std::vector<std::vector<int>>& sparsity_pattern =
      A_block_pattern.neighbors();
  std::vector<std::vector<int>> permuted_sparsity_pattern(
      sparsity_pattern.size());
  for (int i = 0; i < ssize(sparsity_pattern); ++i) {
    const int pi = block_permutation.permuted_index(i);
    for (int j : sparsity_pattern[i]) {
      const int pj = block_permutation.permuted_index(j);
      permuted_sparsity_pattern[std::min(pi, pj)].emplace_back(
          std::max(pi, pj));
    }
  }
  std::vector<int> permuted_block_sizes(A.block_cols());
  block_permutation.Apply(A_block_sizes, &permuted_block_sizes);

  /* Compute the sparsity pattern of L given the sparsity pattern of A in the
   new ordering. */
  return contact_solvers::internal::SymbolicCholeskyFactor(
      BlockSparsityPattern(permuted_block_sizes, permuted_sparsity_pattern));
}

template <typename BlockType>
bool BlockSparseLdltSolver<BlockType>::ComputeFactorization(
    std::optional<Scalar> regularized_D_min_eigenvalue) {
  DRAKE_THROW_UNLESS(solver_mode() == SolverMode::kAnalyzed);

  Eigen::SelfAdjointEigenSolver<BlockType> D_decomp;
  Eigen::Vector<Scalar, BlockType::RowsAtCompileTime> D_lambda;

  for (int j = 0; j < L_->block_cols(); ++j) {
    /* The bottom right square sub-matrix starting from column j looks like:
     | a₁₁ A₂₁ᵀ | = |  1  0 | × | d₁₁  0  | × | 1  L₂₁ᵀ |
     | A₂₁ A₂₂  |   | L₂₁ I |   |  0  D₂₂ |   | 0   I   |
                  = | d₁₁        d₁₁L₂₁ᵀ     |
                    | L₂₁d₁₁  L₂₁d₁₁L₂₁ᵀ+D₂₂ |. */
    /* Update d₁₁ according to d₁₁ = a₁₁, and regularize if specified. */
    const BlockType& Ajj = L_->diagonal_block(j);
    D_decomp.compute(Ajj);
    if (D_decomp.info() != Eigen::Success) {
      throw std::runtime_error(
          fmt::format("Eigen-decomposition of the matrix\n{}\nfailed. Probably "
                      "due to Nan and Infinite in the matrix.",
                      fmt_eigen(Ajj)));
    }
    D_lambda = D_decomp.eigenvalues();
    if (regularized_D_min_eigenvalue.has_value()) {
      D_lambda = D_lambda.cwiseMax(*regularized_D_min_eigenvalue);
    } else if ((D_lambda.array().abs() <
                8 * std::numeric_limits<double>::epsilon())
                   .any()) {
      return false;
    }
    const BlockType& D_V = D_decomp.eigenvectors();
    D_[j] = D_V * D_lambda.asDiagonal() * D_V.transpose();
    Dinv_[j] = D_V * D_lambda.cwiseInverse().asDiagonal() * D_V.transpose();
    /* Set the jᵗʰ diagonal block of L to identity. */
    if constexpr (BlockType::RowsAtCompileTime != Eigen::Dynamic &&
                  BlockType::ColsAtCompileTime != Eigen::Dynamic) {
      L_->SetBlockFlat(0, j, BlockType::Identity());
    } else {
      const int size = L_->diagonal_block(j).cols();
      L_->SetBlockFlat(0, j, BlockType::Identity(size, size));
    }
    /* Update L₂₁ column according to d₁₁L₂₁ᵀ = A₂₁ᵀ. */
    const std::vector<int>& row_blocks = L_->block_row_indices(j);
    for (int flat = 1; flat < ssize(row_blocks); ++flat) {
      const BlockType& Aij = L_->block_flat(flat, j);
      BlockType Lij = (Dinv_[j] * Aij.transpose()).transpose();
      L_->SetBlockFlat(flat, j, std::move(Lij));
    }
    /* Update D₂₂ according to D₂₂ = A₂₂ - L₂₁d₁₁L₂₁ᵀ. */
    RightLookingSymmetricRank1Update(j);
  }
  return true;
}

template <typename BlockType>
void BlockSparseLdltSolver<BlockType>::RightLookingSymmetricRank1Update(int j) {
  const std::vector<int>& blocks_in_col_j = L_->block_row_indices(j);
  const int n = blocks_in_col_j.size();
  /* We start from k = 1 here to skip the j,j diagonal entry. */
  for (int k = 1; k < n; ++k) {
    const int col = blocks_in_col_j[k];
    const BlockType& B = L_->block_flat(k, j);
    for (int l = k; l < n; ++l) {
      const int row = blocks_in_col_j[l];
      const BlockType& A = L_->block_flat(l, j);
      L_->AddToBlock(row, col, -A * D_[j] * B.transpose());
    }
  }
}

template <typename BlockType>
void BlockSparseLdltSolver<BlockType>::PermuteAndCopyToL(
    const SymmetricMatrix& A) {
  const int n = A.block_cols();
  DRAKE_DEMAND(n == block_permutation_.domain_size());
  DRAKE_DEMAND(n == block_permutation_.permuted_domain_size());
  L_->SetZero();
  for (int j = 0; j < n; ++j) {
    const std::vector<int>& row_indices = A.block_row_indices(j);
    for (int i : row_indices) {
      const BlockType& block = A.block(i, j);
      const int pi = block_permutation_.permuted_index(i);
      const int pj = block_permutation_.permuted_index(j);
      if (pi >= pj) {
        L_->SetBlock(pi, pj, block);
      } else {
        L_->SetBlock(pj, pi, block.transpose());
      }
    }
  }
}

template class BlockSparseLdltSolver<MatrixX<double>>;
template class BlockSparseLdltSolver<Matrix4<double>>;

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
