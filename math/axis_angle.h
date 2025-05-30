#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace math {

/** Returns the rotation of `vec` around the specified `axis` by `angle`.
 @pre `‖axis‖ ≈ 1`. */
template <typename T>
Eigen::Vector3<T> RotateAxisAngle(const Eigen::Vector3<T>& vec,
                                  const Eigen::Vector3<T>& axis,
                                  const T& angle);

/** Returns the signed angle of rotation from `vec1` to `vec2` around a given
 `axis`. The signed angle is in the range (−π, π]. The angle is positive if the
 rotation follows the right-hand rule with `axis` as the thumb direction.
 @pre `‖vec1‖ ≈ 1`.
 @pre `‖vec2‖ ≈ 1`.
 @pre `‖axis‖ ≈ 1`.
 @pre `vec1.dot(axis) ≈ 0`.
 @pre `vec2.dot(axis) ≈ 0`. */
template <typename T>
T SignedAngleAroundAxis(const Eigen::Vector3<T>& vec1,
                        const Eigen::Vector3<T>& vec2,
                        const Eigen::Vector3<T>& axis);

}  // namespace math
}  // namespace drake
