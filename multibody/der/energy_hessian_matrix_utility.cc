#include "drake/multibody/der/energy_hessian_matrix_utility.h"

#include "drake/common/default_scalars.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

namespace {

/* Return true only if the number of rows of `mat` is equal to `num_dofs` or
 `num_dofs + 1`. For number of rows equal `num_dofs + 1`, return true only if
 the last row and last column of `mat` are all zero.  */
template <typename T>
[[nodiscard]] bool IsMatrixSizeOk(const Block4x4SparseSymmetricMatrix<T>& mat,
                                  int num_dofs) {
  DRAKE_DEMAND(num_dofs > 0);
  DRAKE_DEMAND(mat.rows() == mat.cols());
  if (mat.rows() == num_dofs) {
    return true;
  } else if (mat.rows() == num_dofs + 1) {
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
  return false;
}

}  // namespace

template <typename T>
Block4x4SparseSymmetricMatrixVectorProduct<T>::
    Block4x4SparseSymmetricMatrixVectorProduct(
        const Block4x4SparseSymmetricMatrix<T>* mat,
        const Eigen::VectorX<T>* vec, const T& scale)
    : mat_(mat), vec_(vec), scale_(scale) {
  DRAKE_ASSERT(IsMatrixSizeOk(*mat, vec->size()));
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
  DRAKE_ASSERT(IsMatrixSizeOk(*lhs, rhs.rows()));
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

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (static_cast<void (*)(Block4x4SparseSymmetricMatrix<T>*,
                          const Eigen::DiagonalMatrix<T, Eigen::Dynamic>&,
                          const T&)>(&AddScaledMatrix<T>),
     static_cast<void (*)(Block4x4SparseSymmetricMatrix<T>*,
                          const Block4x4SparseSymmetricMatrix<T>&,  //
                          const T&)>(&AddScaledMatrix<T>)));

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::
        Block4x4SparseSymmetricMatrixVectorProduct);
