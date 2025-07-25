#include "drake/multibody/der/schur_complement.h"

#include <utility>

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
SchurComplement<T>::SchurComplement(
    const Eigen::Ref<const Eigen::SparseMatrix<T>>& A,
    const Eigen::Ref<const Eigen::SparseMatrix<T>>& B_transpose,
    const Eigen::Ref<const Eigen::SparseMatrix<T>>& D)
    : p_(A.rows()), q_(D.rows()) {
  DRAKE_THROW_UNLESS(A.cols() == A.rows());
  DRAKE_THROW_UNLESS(D.cols() == D.rows());
  DRAKE_THROW_UNLESS(B_transpose.rows() == q_);
  DRAKE_THROW_UNLESS(B_transpose.cols() == p_);
  /* Special treatment for M = A is needed because the linear solver throws
   exception if the matrix under decomposition is empty. */
  if (q_ == 0) {
    neg_Dinv_B_transpose_.resize(0, p_);
    D_complement_ = A;
  } else if (p_ == 0) {
    neg_Dinv_B_transpose_.resize(q_, 0);
    // D_complement is empty, same as default.
  } else {
    /* The D matrix is mostly banded, so natural ordering is good enough. */
    const Eigen::SimplicialLLT<Eigen::SparseMatrix<T>, Eigen::Lower,
                               Eigen::NaturalOrdering<int>>
        D_factorization(D);
    if (D_factorization.info() != Eigen::Success) {
      throw std::runtime_error(
          "Matrix factorization failed because it is not positive definite");
    }
    /* A column major rhs is required for the SimplicialLLT solve, so we take
     `B_transpose` instead of `B.transpose()`. */
    neg_Dinv_B_transpose_ = D_factorization.solve(-B_transpose);
    D_complement_ = A + B_transpose.transpose() * neg_Dinv_B_transpose_;
  }
}

template <typename T>
VectorX<T> SchurComplement<T>::SolveForY(
    const Eigen::Ref<const VectorX<T>>& x) const {
  /* If M = D, then the system reduces to Dy = 0. */
  if (p_ == 0) {
    return VectorX<T>::Zero(q_);
  }
  DRAKE_THROW_UNLESS(x.size() == p_);
  return neg_Dinv_B_transpose_ * x;
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::der::internal::SchurComplement);
