#pragma once

#include "drake/multibody/der/energy_hessian_matrix_utility.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

using AutoDiffXAutoDiffXd = Eigen::AutoDiffScalar<VectorX<AutoDiffXd>>;

/* Computes the shortest distance between the line segment x₁⎯x₂ and the line
 segment x₃⎯x₄.
 @tparam The scalar type, which must be one of the default scalars or
         AutoDiffXAutoDiffXd. */
template <typename T>
T ComputeDistanceBetweenLineSegments(const Eigen::Ref<const Vector3<T>>& x1,
                                     const Eigen::Ref<const Vector3<T>>& x2,
                                     const Eigen::Ref<const Vector3<T>>& x3,
                                     const Eigen::Ref<const Vector3<T>>& x4);

/* Computes the jacobian of the distance value with respect to the node
 positions x₁, x₂, x₃, and x₄.
 @tparam_double_only */
template <typename T>
Eigen::Vector<T, 12> ComputeLineSegmentsDistanceJacobian(
    const Eigen::Ref<const Vector3<T>>& x1,
    const Eigen::Ref<const Vector3<T>>& x2,
    const Eigen::Ref<const Vector3<T>>& x3,
    const Eigen::Ref<const Vector3<T>>& x4);

/* Computes the hessian of the distance value with respect to the node
 positions x₁, x₂, x₃, and x₄.
 @tparam_double_only */
template <typename T>
Eigen::Matrix<T, 12, 12> ComputeLineSegmentsDistanceHessian(
    const Eigen::Ref<const Vector3<T>>& x1,
    const Eigen::Ref<const Vector3<T>>& x2,
    const Eigen::Ref<const Vector3<T>>& x3,
    const Eigen::Ref<const Vector3<T>>& x4);

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
