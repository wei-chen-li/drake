#pragma once

#include <optional>

#include "drake/common/eigen_types.h"

namespace drake {
namespace math {

/** The transport operator maps tᵏ to tᵏ⁺¹ by rotating it φ degrees about the
 axis tᵏ × tᵏ⁺¹. Compute d₁ᵏ⁺¹ as the vector rotating d₁ᵏ by φ degrees about the
 same axis tᵏ × tᵏ⁺¹.

 @param[in] t_0 The tangent director in frame k.
 @param[in] d1_0 Another director in frame k.
 @param[in] t_1 The tangent director in frame k+1.
 @param[out] d1_1 Mapped director in frame k+1 as described above.

 @pre `‖t_0‖ ≈ 1`.
 @pre `‖d1_0‖ ≈ 1`.
 @pre `‖t_1‖ ≈ 1`.
 @pre `t_0.dot(d1_0) ≈ 0`.
 @tparam_default_scalar */
template <typename T>
void FrameTransport(const Eigen::Ref<const Eigen::Vector3<T>>& t_0,
                    const Eigen::Ref<const Eigen::Vector3<T>>& d1_0,
                    const Eigen::Ref<const Eigen::Vector3<T>>& t_1,
                    Eigen::Ref<Eigen::Vector3<T>> d1_1);

/** For k = 0,1,..., transform d₁ᵏ to d₁ᵏ⁺¹ using the transport operator that
 maps tᵏ to tᵏ⁺¹.

 @param[in] t The tangent directors in all frames.
 @param[in] d1_0 The director d₁⁰ in frame 0. If not specified, chooses an
                 arbitrary director that is perpendicular to t⁰.
 @param[out] d1 Mapped directors in all frames as described above.

 @pre `d1 != nullptr`.
 @pre `t.cols() == d1->cols()`.
 @pre `!d1_0 || d1_0.dot(t.col(0)) ≈ 0`.
 @tparam_default_scalar */
template <typename T>
void SpaceParallelFrameTransport(const Eigen::Ref<const Eigen::Matrix3X<T>>& t,
                                 const std::optional<Eigen::Vector3<T>>& d1_0,
                                 EigenPtr<Eigen::Matrix3X<T>> d1);

/** For k = 0,1,..., transform d₁ᵏ(𝑡) to d₁ᵏ(𝑡+𝛥𝑡) using the transport operator
 that maps tᵏ(𝑡) to tᵏ(𝑡+𝛥𝑡).

 @param[in] t The tangent directors at time 𝑡.
 @param[in] d1 The d₁ directors at time 𝑡.
 @param[in] t_next The tangent directors at time 𝑡+𝛥𝑡.
 @param[out] d1_next The d₁ directors at time 𝑡+𝛥𝑡.

 @pre `d1_next != nullptr`.
 @pre Number of columns of `t`, `d1`, `t_next`, `d1_next` are the same.
 @tparam_default_scalar */
template <typename T>
void TimeParallelFrameTransport(
    const Eigen::Ref<const Eigen::Matrix3X<T>>& t,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& d1,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& t_next,
    EigenPtr<Eigen::Matrix3X<T>> d1_next);

}  // namespace math
}  // namespace drake
