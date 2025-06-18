#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "drake/geometry/proximity/filament_self_contact_filter.h"
#include "drake/multibody/der/der_indexes.h"
#include "drake/multibody/der/der_state.h"
#include "drake/multibody/der/der_undeformed_state.h"
#include "drake/multibody/der/energy_hessian_matrix_utility.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

using AutoDiffXAutoDiffXd = Eigen::AutoDiffScalar<VectorX<AutoDiffXd>>;

/* The artificial contact energy between a pair of edges with distance D is
 defined as:
   Ec = (C - D)²                       if 0     < D ≤ C - δ,
   Ec = (1/K ln(1 + exp(K (C - D))))²  if C - δ < D < C + δ,
   Ec = 0                              if C + δ ≤ D,
 where C is the contact distance at which contact would occur, δ is the contact
 distance tolerance and is set to 0.01C, K is the stiffness parameter and is set
 to 15/δ.
 @tparam_default_scalar */
template <typename T>
class ContactEnergy {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ContactEnergy);

  /* Creates a ContactEnergy class from the contact distance `C` and the
   undeformed state `undeformed`. The parameter `scale` controls the scaling of
   the contact energy.
   @pre `C > 0`. */
  ContactEnergy(double C, const DerUndeformedState<T>& undeformed);

  /* Computes Ec given the `state`.
   @pre `state` is compatible with `undeformed` supplied at construction. */
  const T& ComputeEnergy(const DerState<T>& state);

  /* Computes ∂Ec/∂q given the `state`.
   @pre `state` is compatible with `undeformed` supplied at construction. */
  template <typename T1 = T>
  std::enable_if_t<std::is_same_v<T1, double>, const Eigen::VectorX<T>&>
  ComputeEnergyJacobian(const DerState<T>& state);

  /* Computes ∂²Ec/∂q² given the `state`.
   @pre `state` is compatible with `undeformed` supplied at construction. */
  template <typename T1 = T>
  std::enable_if_t<std::is_same_v<T1, double>,
                   const Block4x4SparseSymmetricMatrix<T>&>
  ComputeEnergyHessian(const DerState<T>& state);

 private:
  struct Cache {
    Eigen::Matrix3X<T> node_positions;
    std::vector<std::tuple<DerEdgeIndex, DerEdgeIndex, T>> contacts;
    std::optional<T> energy;
    std::optional<Eigen::VectorX<T>> jacobian;
    std::optional<Block4x4SparseSymmetricMatrix<T>> hessian;
  };

  Cache& EvalCache(const DerState<T>& state);

  struct EnergyModel {
    explicit EnergyModel(double C);
    T Ec(const T& D) const;
    T dEc_dD(const T& D) const;
    T d2Ec_dD2(const T& D) const;

    double C;
    double delta;
    double K;
  };

  EnergyModel energy_model_;
  bool has_closed_ends_;
  int num_nodes_;
  geometry::internal::filament::FilamentSelfContactFilter filter_;
  std::tuple<const void*, int64_t, Cache> state_to_cache_;
};

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
