#include "drake/multibody/der/transport.h"

#include "drake/common/autodiff.h"
#include "drake/common/default_scalars.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/* Tolerance for ‖unit_vector‖ is 1e-14 (≈ 5.5 bits) of 1.0.
   Tolerance for v1.dot(v2) if v1 ⊥ v2 is 1e-14 of 0.0.
 Note: 1e-14 ≈ 2^5.5 * std::numeric_limits<double>::epsilon(); */
constexpr double kTol = 1e-14;

template <typename T>
void ComputeTransport(const Eigen::Ref<const Eigen::Vector3<T>>& t_0,
                      const Eigen::Ref<const Eigen::Vector3<T>>& d1_0,
                      const Eigen::Ref<const Eigen::Vector3<T>>& t_1,
                      Eigen::Ref<Eigen::Vector3<T>> d1_1) {
  DRAKE_ASSERT(std::abs(ExtractDoubleOrThrow(t_0.norm()) - 1.0) < kTol);
  DRAKE_ASSERT(std::abs(ExtractDoubleOrThrow(d1_0.norm()) - 1.0) < kTol);
  DRAKE_ASSERT(std::abs(ExtractDoubleOrThrow(t_1.norm()) - 1.0) < kTol);
  DRAKE_ASSERT(std::abs(ExtractDoubleOrThrow(t_0.dot(d1_0))) < kTol);
  Eigen::Vector3<T> b = t_0.cross(t_1);
  if (b.norm() == 0.0) {
    d1_1 = d1_0;
  } else {
    b /= b.norm();

    Eigen::Vector3<T> n_0 = t_0.cross(b);
    Eigen::Vector3<T> n_1 = t_1.cross(b);
    d1_1 = d1_0.dot(t_0) * t_1 + d1_0.dot(n_0) * n_1 + d1_0.dot(b) * b;
    d1_1 -= d1_1.dot(t_1) * t_1;
    d1_1 /= d1_1.norm();
  }
}

template <typename T>
void ComputeSpaceParallelTransport(
    const Eigen::Ref<const Eigen::Matrix<T, 3, Eigen::Dynamic>>& t,
    const std::optional<Eigen::Vector3<T>>& d1_0_in,
    EigenPtr<Eigen::Matrix<T, 3, Eigen::Dynamic>> d1) {
  DRAKE_THROW_UNLESS(d1 != nullptr);
  DRAKE_THROW_UNLESS(d1->cols() == t.cols());

  auto t_0 = t.col(0);
  auto d1_0 = d1->col(0);
  if (d1_0_in) {
    DRAKE_THROW_UNLESS(std::abs(ExtractDoubleOrThrow(d1_0_in->norm()) - 1.0) <
                       kTol);
    DRAKE_THROW_UNLESS(std::abs(ExtractDoubleOrThrow(d1_0_in->dot(t_0))) <
                       kTol);
    d1_0 = *d1_0_in;
  } else {
    d1_0 = Eigen::Vector3<T>(-t_0[1], t_0[0], 0);
    if (d1_0.norm() >= kTol) {
      d1_0 /= d1_0.norm();
    } else {
      d1_0 = Eigen::Vector3<T>(0, -t_0[2], t_0[1]);
      d1_0 /= d1_0.norm();
    }
  }
  d1_0 -= d1_0.dot(t_0) * t_0;
  d1_0 /= d1_0.norm();

  for (int i = 0; i < t.cols() - 1; ++i) {
    ComputeTransport<T>(t.col(i), d1->col(i), t.col(i + 1), d1->col(i + 1));
  }
}

template <typename T>
void ComputeTimeParallelTransport(
    const Eigen::Ref<const Eigen::Matrix<T, 3, Eigen::Dynamic>>& t,
    const Eigen::Ref<const Eigen::Matrix<T, 3, Eigen::Dynamic>>& d1,
    const Eigen::Ref<const Eigen::Matrix<T, 3, Eigen::Dynamic>>& t_next,
    EigenPtr<Eigen::Matrix<T, 3, Eigen::Dynamic>> d1_next) {
  DRAKE_THROW_UNLESS(d1_next != nullptr);
  DRAKE_THROW_UNLESS(t.cols() == d1.cols() && t.cols() == t_next.cols() &&
                     t.cols() == d1_next->cols());
  for (int i = 0; i < t.cols(); ++i) {
    ComputeTransport<T>(t.col(i), d1.col(i), t_next.col(i), d1_next->col(i));
  }
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&ComputeTransport<T>, &ComputeSpaceParallelTransport<T>,
     &ComputeTimeParallelTransport<T>));

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
