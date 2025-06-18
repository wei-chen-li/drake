#pragma once

#include <array>
#include <optional>
#include <tuple>
#include <utility>

#include "drake/common/eigen_types.h"

namespace drake {
namespace math {

template <typename Derived, int nq>
using AutoDiffAutoDiffMatrixType = MatrixLikewise<
    Eigen::AutoDiffScalar<Vector<
        Eigen::AutoDiffScalar<Vector<typename Derived::Scalar, nq>>, nq>>,
    Derived>;

template <typename Derived, typename DerivedAutoDiffAutoDiff>
void InitializeAutoDiffAutoDiff(
    const Eigen::MatrixBase<Derived>& matrix,
    std::optional<int> num_derivatives, std::optional<int> deriv_num_start,
    Eigen::MatrixBase<DerivedAutoDiffAutoDiff>* auto_diff_auto_diff_matrix) {
  // Any fixed-size dimension of auto_diff_matrix must match the
  // corresponding fixed-size dimension of value. Any dynamic-size
  // dimension must be dynamic in both matrices.
  static_assert(
      static_cast<int>(Derived::RowsAtCompileTime) ==
          static_cast<int>(DerivedAutoDiffAutoDiff::RowsAtCompileTime),
      "auto diff matrix has wrong number of rows at compile time");
  static_assert(
      static_cast<int>(Derived::ColsAtCompileTime) ==
          static_cast<int>(DerivedAutoDiffAutoDiff::ColsAtCompileTime),
      "auto diff matrix has wrong number of columns at compile time");
  static_assert(static_cast<int>(Derived::IsRowMajor) ==
                    static_cast<int>(DerivedAutoDiffAutoDiff::IsRowMajor),
                "auto diff matrix has wrong storage order at compile time");

  DRAKE_DEMAND(auto_diff_auto_diff_matrix != nullptr);
  if (!num_derivatives.has_value()) num_derivatives = matrix.size();

  using ADADScalar = typename DerivedAutoDiffAutoDiff::Scalar;
  auto_diff_auto_diff_matrix->resize(matrix.rows(), matrix.cols());
  const int deriv_offset = deriv_num_start.value_or(0);
  for (int i = 0; i < matrix.size(); ++i) {
    ADADScalar outer;
    outer.value().value() = matrix(i);

    outer.value().derivatives() = Eigen::VectorXd::Zero(*num_derivatives);
    outer.value().derivatives()[deriv_offset + i] = 1.0;

    outer.derivatives() =
        Eigen::VectorX<typename ADADScalar::Scalar>(*num_derivatives);
    for (int j = 0; j < *num_derivatives; ++j)
      outer.derivatives()[j].value() = (j == deriv_offset + i) ? 1.0 : 0.0;

    (*auto_diff_auto_diff_matrix)(i) = std::move(outer);
  }
}

template <int nq = Eigen::Dynamic, typename Derived>
AutoDiffAutoDiffMatrixType<Derived, nq> InitializeAutoDiffAutoDiff(
    const Eigen::MatrixBase<Derived>& matrix,
    std::optional<int> num_derivatives = {},
    std::optional<int> deriv_num_start = {}) {
  AutoDiffAutoDiffMatrixType<Derived, nq> auto_diff_auto_diff_matrix(
      matrix.rows(), matrix.cols());
  InitializeAutoDiffAutoDiff(matrix, num_derivatives.value_or(matrix.size()),
                             deriv_num_start.value_or(0),
                             &auto_diff_auto_diff_matrix);
  return auto_diff_auto_diff_matrix;
}

template <typename... Deriveds>
auto InitializeAutoDiffAutoDiffTuple(
    const Eigen::MatrixBase<Deriveds>&... args) {
  // Compute the total compile-time size of all args (or Dynamic, if unknown).
  // Refer to https://en.cppreference.com/w/cpp/language/fold for the syntax.
  constexpr int nq = ((Deriveds::SizeAtCompileTime != Eigen::Dynamic) && ...)
                         ? (static_cast<int>(Deriveds::SizeAtCompileTime) + ...)
                         : Eigen::Dynamic;

  // Compute each deriv_num_start value and then the total runtime size.
  constexpr size_t N = sizeof...(args);
  const std::array<int, N> sizes{static_cast<int>(args.size())...};
  std::array<int, N + 1> deriv_num_starts = {0};
  for (size_t i = 1; i <= N; ++i) {
    deriv_num_starts[i] = deriv_num_starts[i - 1] + sizes[i - 1];
  }
  const int num_derivatives = deriv_num_starts.back();

  // Allocate the result.
  std::tuple<AutoDiffAutoDiffMatrixType<Deriveds, nq>...> result(
      AutoDiffAutoDiffMatrixType<Deriveds, nq>(args.rows(), args.cols())...);

  // Set the values and gradients of the result using InitializeAutoDiff from
  // each Matrix in 'args...'. This is a "constexpr for" loop for 0 <= I < N.
  auto args_tuple = std::forward_as_tuple(args...);
  [&]<size_t... I>(std::integer_sequence<size_t, I...>&&) {
    (InitializeAutoDiffAutoDiff(std::get<I>(args_tuple), num_derivatives,
                                std::get<I>(deriv_num_starts),
                                &std::get<I>(result)),
     ...);
  }(std::make_index_sequence<N>{});

  return result;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar::Scalar,
              Derived::DerType::RowsAtCompileTime,
              Derived::DerType::RowsAtCompileTime>
ExtractHessian(const Derived& value, std::optional<int> num_derivatives = {}) {
  int num_derivatives_from_matrix = value.derivatives().size();
  for (int i = 0; i < value.derivatives().size(); ++i) {
    const int entry_num_derivs = value.derivatives()[i].derivatives().size();
    if (entry_num_derivs != 0 &&
        entry_num_derivs != num_derivatives_from_matrix) {
      throw std::logic_error(fmt::format(
          "ExtractHessian(): Input value has elements with inconsistent,"
          " non-zero numbers of derivatives ({} and {}).",
          num_derivatives_from_matrix, *num_derivatives));
    }
  }

  if (!num_derivatives.has_value()) {
    num_derivatives = num_derivatives_from_matrix;
  } else if (num_derivatives_from_matrix != 0 &&
             num_derivatives_from_matrix != *num_derivatives) {
    throw std::logic_error(fmt::format(
        "ExtractHessian(): Input value has {} derivatives, but"
        " num_derivatives was specified as {}. Either the input value should"
        " have zero derivatives, or the number should match num_derivatives.",
        num_derivatives_from_matrix, *num_derivatives));
  }

  Eigen::Matrix<typename Derived::Scalar::Scalar,
                Derived::DerType::RowsAtCompileTime,
                Derived::DerType::RowsAtCompileTime>
      hessian = Eigen::MatrixX<typename Derived::Scalar::Scalar>::Zero(
          *num_derivatives, *num_derivatives);
  if (value.derivatives().size()) {
    for (int i = 0; i < *num_derivatives; ++i) {
      if (value.derivatives()[i].derivatives().size()) {
        hessian.col(i) = value.derivatives()[i].derivatives();
      }
    }
  }
  return hessian;
}

}  // namespace math
}  // namespace drake
