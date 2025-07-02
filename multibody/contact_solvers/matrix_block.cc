#include "drake/multibody/contact_solvers/matrix_block.h"

#include "drake/common/overloaded.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

template <class T>
MatrixBlock<T>::MatrixBlock(Block3x3SparseMatrix<T> data)
    : data_(std::move(data)) {}

template <class T>
MatrixBlock<T>::MatrixBlock(Block3x1SparseMatrix<T> data)
    : data_(std::move(data)) {}

template <class T>
MatrixBlock<T>::MatrixBlock(MatrixX<T> data) : data_(std::move(data)) {}

template <class T>
int MatrixBlock<T>::rows() const {
  return std::visit(
      [](auto&& arg) {
        /* We need the static_cast here because Eigen's rows() is `long`. */
        return static_cast<int>(arg.rows());
      },
      data_);
}

template <class T>
int MatrixBlock<T>::cols() const {
  return std::visit(
      [](auto&& arg) {
        /* We need the static_cast here because Eigen's cols() is `long`. */
        return static_cast<int>(arg.cols());
      },
      data_);
}

template <class T>
void MatrixBlock<T>::MultiplyAndAddTo(const Eigen::Ref<const MatrixX<T>>& A,
                                      EigenPtr<MatrixX<T>> y) const {
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(cols() == A.rows());
  DRAKE_DEMAND(rows() == y->rows());
  DRAKE_DEMAND(A.cols() == y->cols());
  std::visit(overloaded{[&](const MatrixX<T>& M) {
                          *y += M * A;
                        },
                        [&](const auto& M) {
                          M.MultiplyAndAddTo(A, y);
                        }},
             data_);
}

template <class T>
void MatrixBlock<T>::TransposeAndMultiplyAndAddTo(
    const Eigen::Ref<const MatrixX<T>>& A, EigenPtr<MatrixX<T>> y) const {
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(cols() == y->rows());
  DRAKE_DEMAND(rows() == A.rows());
  DRAKE_DEMAND(A.cols() == y->cols());
  std::visit(overloaded{[&](const MatrixX<T>& M) {
                          *y += M.transpose() * A;
                        },
                        [&](const auto& M) {
                          M.TransposeAndMultiplyAndAddTo(A, y);
                        }},
             data_);
}

// TODO(xuchenhan-tri): consider a double dispatch strategy where each block
// type provides an API to operate on every other block type.
template <class T>
void MatrixBlock<T>::TransposeAndMultiplyAndAddTo(
    const MatrixBlock<T>& A, EigenPtr<MatrixX<T>> y) const {
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(cols() == y->rows());
  DRAKE_DEMAND(rows() == A.rows());
  DRAKE_DEMAND(A.cols() == y->cols());
  std::visit(overloaded{[&](const MatrixX<T>& M) {
                          std::visit(overloaded{[&](const MatrixX<T>& A_mat) {
                                                  *y += M.transpose() * A_mat;
                                                },
                                                [&](const auto& A_mat) {
                                                  A_mat.LeftMultiplyAndAddTo(
                                                      M.transpose(), y);
                                                }},
                                     A.data_);
                        },
                        [&](const auto& M) {
                          std::visit(
                              [&](const auto& A_mat) {
                                M.TransposeAndMultiplyAndAddTo(A_mat, y);
                              },
                              A.data_);
                        }},
             data_);
}

// TODO(xuchenhan-tri): consider a double dispatch strategy where each block
// type provides an API to operate on every other block type.
template <class T>
MatrixBlock<T> MatrixBlock<T>::LeftMultiplyByBlockDiagonal(
    const std::vector<MatrixX<T>>& Gs, int start, int end) const {
  DRAKE_DEMAND(start >= 0);
  DRAKE_DEMAND(end >= start);
  DRAKE_DEMAND(static_cast<int>(Gs.size()) > end);
  /* Verify that the sizes of G and M is compatible. */
  std::vector<int>
      row_starts;  // starting row index for each diagonal block in G
  row_starts.reserve(end - start + 1);
  int row = 0;
  for (int i = start; i <= end; ++i) {
    DRAKE_DEMAND(Gs[i].rows() == Gs[i].cols());
    row_starts.emplace_back(row);
    row += Gs[i].rows();
    if (!is_dense()) {
      DRAKE_DEMAND(Gs[i].rows() % 3 == 0);
    }
  }
  DRAKE_DEMAND(row == rows());

  return std::visit(
      overloaded{[&](const MatrixX<T>& M) {
                   MatrixX<T> GM(rows(), cols());
                   for (int i = start; i <= end; ++i) {
                     const int rows = Gs[i].rows();
                     const int row_start = row_starts[i - start];
                     GM.middleRows(row_start, rows).noalias() =
                         Gs[i] * M.middleRows(row_start, rows);
                   }
                   return MatrixBlock<T>(std::move(GM));
                 },
                 [&](const auto& M) {
                   return MatrixBlock<T>(
                       M.LeftMultiplyByBlockDiagonal(Gs, start, end));
                 }},
      data_);
}

template <class T>
void MatrixBlock<T>::MultiplyWithScaledTransposeAndAddTo(
    const VectorX<T>& scale, EigenPtr<MatrixX<T>> y) const {
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(cols() == scale.size());
  DRAKE_DEMAND(rows() == y->rows());
  DRAKE_DEMAND(rows() == y->cols());
  std::visit(overloaded{[&](const MatrixX<T>& M) {
                          *y += M * scale.asDiagonal() * M.transpose();
                        },
                        [&](const auto& M) {
                          M.MultiplyWithScaledTransposeAndAddTo(scale, y);
                        }},
             data_);
}

template <class T>
MatrixBlock<T> MatrixBlock<T>::operator+(const MatrixBlock<T>& other) const {
  DRAKE_THROW_UNLESS(this->rows() == other.rows());
  DRAKE_THROW_UNLESS(this->cols() == other.cols());
  if (this->is_dense() || other.is_dense()) {
    return MatrixBlock<T>(this->MakeDenseMatrix() + other.MakeDenseMatrix());
  }

  std::vector<typename Block3x1SparseMatrix<T>::Triplet> triplets;
  auto extract_triplets = overloaded{
      [&](const Block3x1SparseMatrix<T>& entry) {
        for (const auto& row_data : entry.get_triplets()) {
          for (const auto& t : row_data) {
            triplets.push_back(t);
          }
        }
      },
      [&](const Block3x3SparseMatrix<T>& entry) {
        for (const auto& row_data : entry.get_triplets()) {
          for (const auto& t : row_data) {
            const int block_row = std::get<0>(t);
            const int block_col = std::get<1>(t) * 3;
            const Matrix3<T>& m = std::get<2>(t);
            triplets.emplace_back(block_row, block_col + 0, m.col(0));
            triplets.emplace_back(block_row, block_col + 1, m.col(1));
            triplets.emplace_back(block_row, block_col + 2, m.col(2));
          }
        }
      },
      [&](const MatrixX<T>&) {
        DRAKE_UNREACHABLE();
      },
  };
  std::visit(extract_triplets, this->data_);
  std::visit(extract_triplets, other.data_);
  DRAKE_DEMAND(rows() % 3 == 0);
  Block3x1SparseMatrix<T> result(rows() / 3, cols());
  result.SetFromTriplets(std::move(triplets));
  return MatrixBlock<T>(std::move(result));
}

template <class T>
MatrixX<T> MatrixBlock<T>::MakeDenseMatrix() const {
  return std::visit(overloaded{[&](const MatrixX<T>& M) {
                                 return M;
                               },
                               [&](const auto& M) {
                                 return M.MakeDenseMatrix();
                               }},
                    data_);
}

template <typename T>
MatrixBlock<T> StackMatrixBlocks(const std::vector<MatrixBlock<T>>& blocks) {
  if (blocks.empty()) {
    return {};
  }

  const bool is_dense = blocks[0].is_dense();
  bool is_3x3sparse = true;
  const int cols = blocks[0].cols();
  int rows = 0;
  for (const auto& b : blocks) {
    DRAKE_DEMAND(is_dense == b.is_dense());
    if (!std::holds_alternative<Block3x3SparseMatrix<T>>(b.data_))
      is_3x3sparse = false;
    DRAKE_DEMAND(cols == b.cols());
    rows += b.rows();
  }

  if (is_dense) {
    MatrixX<T> result(rows, cols);
    int row_offset = 0;
    for (const auto& b : blocks) {
      result.middleRows(row_offset, b.rows()) = std::get<MatrixX<T>>(b.data_);
      row_offset += b.rows();
    }
    return MatrixBlock<T>(std::move(result));
  }

  if (is_3x3sparse) {
    DRAKE_DEMAND(rows % 3 == 0);
    DRAKE_DEMAND(cols % 3 == 0);
    const int block_rows = rows / 3;
    const int block_cols = cols / 3;
    int block_row_offset = 0;
    Block3x3SparseMatrix<T> result(block_rows, block_cols);
    using Triplet = typename Block3x3SparseMatrix<T>::Triplet;
    std::vector<Triplet> result_triplets;
    int nonzero_blocks = 0;
    for (const auto& b : blocks) {
      const Block3x3SparseMatrix<T>& entry =
          std::get<Block3x3SparseMatrix<T>>(b.data_);
      nonzero_blocks += entry.num_blocks();
    }
    result_triplets.reserve(nonzero_blocks);

    for (const auto& b : blocks) {
      const Block3x3SparseMatrix<T>& entry =
          std::get<Block3x3SparseMatrix<T>>(b.data_);
      const std::vector<std::vector<Triplet>>& b_triplets =
          entry.get_triplets();
      for (const auto& row_data : b_triplets) {
        for (const Triplet& t : row_data) {
          const int block_row = std::get<0>(t) + block_row_offset;
          const int block_col = std::get<1>(t);
          const Matrix3<T>& m = std::get<2>(t);
          result_triplets.emplace_back(block_row, block_col, m);
        }
      }
      block_row_offset += entry.block_rows();
    }
    result.SetFromTriplets(result_triplets);
    return MatrixBlock<T>(std::move(result));
  }

  DRAKE_DEMAND(rows % 3 == 0);
  const int block_rows = rows / 3;
  int block_row_offset = 0;
  Block3x1SparseMatrix<T> result(block_rows, cols);
  std::vector<typename Block3x1SparseMatrix<T>::Triplet> result_triplets;
  int nonzero_blocks = 0;
  for (const auto& b : blocks) {
    std::visit(overloaded{[&](const Block3x1SparseMatrix<T>& entry) {
                            nonzero_blocks += entry.num_blocks();
                          },
                          [&](const Block3x3SparseMatrix<T>& entry) {
                            nonzero_blocks += entry.num_blocks() * 3;
                          },
                          [&](const MatrixX<T>&) {
                            DRAKE_UNREACHABLE();
                          }},
               b.data_);
  }
  result_triplets.reserve(nonzero_blocks);

  for (const auto& b : blocks) {
    std::visit(
        overloaded{
            [&](const Block3x1SparseMatrix<T>& entry) {
              for (const auto& row_data : entry.get_triplets()) {
                for (const typename Block3x1SparseMatrix<T>::Triplet& t :
                     row_data) {
                  const int block_row = std::get<0>(t) + block_row_offset;
                  const int block_col = std::get<1>(t);
                  const Vector3<T>& m = std::get<2>(t);
                  result_triplets.emplace_back(block_row, block_col, m);
                }
              }
              block_row_offset += entry.block_rows();
            },
            [&](const Block3x3SparseMatrix<T>& entry) {
              for (const auto& row_data : entry.get_triplets()) {
                for (const typename Block3x3SparseMatrix<T>::Triplet& t :
                     row_data) {
                  const int block_row = std::get<0>(t) + block_row_offset;
                  const int block_col = std::get<1>(t) * 3;
                  const Matrix3<T>& m = std::get<2>(t);
                  result_triplets.emplace_back(block_row, block_col, m.col(0));
                  result_triplets.emplace_back(block_row, block_col + 1,
                                               m.col(1));
                  result_triplets.emplace_back(block_row, block_col + 2,
                                               m.col(2));
                }
              }
              block_row_offset += entry.block_rows();
            },
            [&](const MatrixX<T>&) {
              DRAKE_UNREACHABLE();
            }},
        b.data_);
  }
  result.SetFromTriplets(result_triplets);
  return MatrixBlock<T>(std::move(result));
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (&StackMatrixBlocks<T>));

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::contact_solvers::internal::MatrixBlock);
