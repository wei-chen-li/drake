#include "drake/math/axis_angle.h"

#include "drake/common/default_scalars.h"
#include "drake/math/unit_vector.h"

namespace drake {
namespace math {

template <typename T>
Eigen::Vector3<T> RotateAxisAngle(
    const Eigen::Ref<const Eigen::Vector3<T>>& vec,
    const Eigen::Ref<const Eigen::Vector3<T>>& axis, const T& angle) {
  internal::ThrowIfNotUnitVector<T>(axis, __func__);
  T cos_ang = cos(angle);
  T sin_ang = sin(angle);
  return cos_ang * vec + sin_ang * axis.cross(vec) +
         axis.dot(vec) * (1.0 - cos_ang) * axis;
}

template <typename T>
T SignedAngleAroundAxis(const Eigen::Ref<const Eigen::Vector3<T>>& vec1,
                        const Eigen::Ref<const Eigen::Vector3<T>>& vec2,
                        const Eigen::Ref<const Eigen::Vector3<T>>& axis) {
  internal::ThrowIfNotOrthonormal<T>(vec1, axis, __func__);
  internal::ThrowIfNotOrthonormal<T>(vec2, axis, __func__);
  T angle = atan2(vec1.cross(vec2).dot(axis), vec1.dot(vec2));
  return angle;
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (&RotateAxisAngle<T>, &SignedAngleAroundAxis<T>));

}  // namespace math
}  // namespace drake
