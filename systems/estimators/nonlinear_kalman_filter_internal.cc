#include "drake/systems/estimators/nonlinear_kalman_filter_internal.h"

namespace drake {
namespace systems {
namespace estimators {
namespace internal {

Eigen::VectorXd ConcatenateVectorAndSquareMatrix(
    const Eigen::Ref<const Eigen::VectorXd>& vector,
    const Eigen::Ref<const Eigen::MatrixXd>& square_matrix) {
  DRAKE_ASSERT(square_matrix.rows() == vector.size());
  DRAKE_ASSERT(square_matrix.cols() == vector.size());
  Eigen::VectorXd concatenated(vector.size() + square_matrix.size());
  concatenated << vector, Eigen::Map<const Eigen::VectorXd>(
                              square_matrix.data(), square_matrix.size());
  return concatenated;
}

Eigen::VectorXd ConcatenateVectorAndLowerTriMatrix(
    const Eigen::Ref<const Eigen::VectorXd>& vector,
    const Eigen::Ref<const Eigen::MatrixXd>& lower_tri_matrix) {
  const int size = vector.size();
  DRAKE_ASSERT(lower_tri_matrix.rows() == size);
  DRAKE_ASSERT(lower_tri_matrix.cols() == size);
  Eigen::VectorXd concatenated(size + size * (size + 1) / 2);
  concatenated.head(size) = vector;

  int idx = size;
  for (int j = 0; j < size; ++j) {
    for (int i = j; i < size; ++i) {
      concatenated(idx++) = lower_tri_matrix(i, j);
    }
  }
  return concatenated;
}

Eigen::VectorXd ConcatenateVectorAndLowerTriMatrix(
    const Eigen::Ref<const Eigen::VectorXd>& vector,
    const Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Lower>&
        lower_tri_matrix) {
  return ConcatenateVectorAndLowerTriMatrix(vector,
                                            Eigen::MatrixXd(lower_tri_matrix));
}

void ExtractSquareMatrix(const Eigen::Ref<const Eigen::VectorXd>& concatenated,
                         Eigen::Ref<Eigen::MatrixXd> square_matrix) {
  const int size = square_matrix.rows();
  DRAKE_ASSERT(square_matrix.rows() == square_matrix.cols());
  DRAKE_ASSERT(concatenated.size() == size + size * size);
  square_matrix =
      Eigen::Map<const Eigen::MatrixXd>(concatenated.data() + size, size, size);
}

void ExtractLowerTriMatrix(
    const Eigen::Ref<const Eigen::VectorXd>& concatenated,
    Eigen::Ref<Eigen::MatrixXd> lower_tri_matrix) {
  const int size = lower_tri_matrix.rows();
  DRAKE_ASSERT(lower_tri_matrix.rows() == lower_tri_matrix.cols());
  DRAKE_ASSERT(concatenated.size() == size + size * (size + 1) / 2);
  lower_tri_matrix.setZero();
  int idx = size;
  for (int j = 0; j < size; ++j) {
    for (int i = j; i < size; ++i) {
      lower_tri_matrix(i, j) = concatenated(idx++);
    }
  }
}

Eigen::MatrixXd MatrixHypot(const Eigen::Ref<const Eigen::MatrixXd>& M1,
                            const Eigen::Ref<const Eigen::MatrixXd>& M2) {
  const int size = M1.rows();
  DRAKE_ASSERT(M1.rows() == M2.rows());
  Eigen::MatrixXd M(M1.cols() + M2.cols(), size);
  M << M1.transpose(), M2.transpose();

  Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
  Eigen::MatrixXd R =
      qr.matrixQR().topRows(size).triangularView<Eigen::Upper>();
  return R.transpose();
}

}  // namespace internal
}  // namespace estimators
}  // namespace systems
}  // namespace drake
