#include "drake/math/frame_transport.h"

#include "drake/common/autodiff.h"
#include "drake/common/default_scalars.h"
#include "drake/math/unit_vector.h"

namespace drake {
namespace math {

template <typename T>
void FrameTransport(const Eigen::Ref<const Eigen::Vector3<T>>& t_0,
                    const Eigen::Ref<const Eigen::Vector3<T>>& d1_0,
                    const Eigen::Ref<const Eigen::Vector3<T>>& t_1,
                    Eigen::Ref<Eigen::Vector3<T>> d1_1) {
  internal::ThrowIfNotOrthonormal<T>(t_0, d1_0, __func__);
  internal::ThrowIfNotUnitVector<T>(t_0, __func__);
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
void SpaceParallelFrameTransport(
    const Eigen::Ref<const Eigen::Matrix3X<T>>& t,
    const std::optional<Eigen::Vector3<T>>& d1_0_in,
    EigenPtr<Eigen::Matrix3X<T>> d1) {
  DRAKE_THROW_UNLESS(d1 != nullptr);
  DRAKE_THROW_UNLESS(d1->cols() == t.cols());

  auto t_0 = t.col(0);
  auto d1_0 = d1->col(0);
  if (d1_0_in) {
    internal::ThrowIfNotOrthonormal<T>(t_0, *d1_0_in, __func__);
    d1_0 = *d1_0_in;
  } else {
    d1_0 = Eigen::Vector3<T>(-t_0[1], t_0[0], 0);
    if (d1_0.norm() >= 1e-10) {
      d1_0 /= d1_0.norm();
    } else {
      d1_0 = Eigen::Vector3<T>(0, -t_0[2], t_0[1]);
      d1_0 /= d1_0.norm();
    }
  }
  d1_0 -= d1_0.dot(t_0) * t_0;
  d1_0 /= d1_0.norm();

  for (int i = 0; i < t.cols() - 1; ++i) {
    FrameTransport<T>(t.col(i), d1->col(i), t.col(i + 1), d1->col(i + 1));
  }
}

template <typename T>
void TimeParallelFrameTransport(
    const Eigen::Ref<const Eigen::Matrix3X<T>>& t,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& d1,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& t_next,
    EigenPtr<Eigen::Matrix3X<T>> d1_next) {
  DRAKE_THROW_UNLESS(d1_next != nullptr);
  DRAKE_THROW_UNLESS(t.cols() == d1.cols() && t.cols() == t_next.cols() &&
                     t.cols() == d1_next->cols());
  for (int i = 0; i < t.cols(); ++i) {
    FrameTransport<T>(t.col(i), d1.col(i), t_next.col(i), d1_next->col(i));
  }
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (&FrameTransport<T>, &SpaceParallelFrameTransport<T>,
     &TimeParallelFrameTransport<T>));

}  // namespace math
}  // namespace drake
