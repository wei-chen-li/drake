#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/sap/sap_holonomic_constraint.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

/* Struct to store the kinematics of the filament fixed constraint in its
 current configuration, when it gets constructed. */
template <typename T>
struct FilamentConstraintKinematics {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FilamentConstraintKinematics);

  /* Constructor for fixed constraints where only a single body has degrees of
     freedom.
     @param[in] objectA
                Index of the physical object A on which dof P attaches.
                Must be non-negative.
     @param[in] q_PQs_W
                Difference from dof Ps to dof Qs.
     @param[in] J_ApBq_W
                Jacobian for the relative velocity v_ApBq_W.
     @pre `object_A >= 0`.
     @pre `q_PQs_W.size() == J_ApBq_W.rows()`. */
  FilamentConstraintKinematics(int objectA_in, VectorX<T> q_PQs_W,
                               SapConstraintJacobian<T> J_ApBq_W)
      : objectA(objectA_in),
        q_PQs(std::move(q_PQs_W)),
        J(std::move(J_ApBq_W)),
        num_constraints(q_PQs.size()) {
    DRAKE_THROW_UNLESS(objectA >= 0);
    DRAKE_THROW_UNLESS(q_PQs.size() == num_constraints);
    DRAKE_THROW_UNLESS(J.rows() == num_constraints);
  }

  /* Constructor for fixed constraints where both bodies have degrees of
     freedom.
     @param[in] objectA
                Index of the physical object A on which dof P attaches.
                Must be non-negative.
     @param[in] objectB
                Index of the physical object B on which dof Q attaches.
                Must be non-negative.
     @param[in] q_PQs_W
                Difference from dof Ps to dof Qs.
     @param[in] J_ApBq_W
                Jacobian for the relative velocity v_ApBq_W.
     @pre `object_A >= 0 && object_B >= 0`.
     @pre `q_PQs_W.size() == J_ApBq_W.rows()`. */
  FilamentConstraintKinematics(int objectA_in, int objectB_in,
                               VectorX<T> q_PQs_W,
                               SapConstraintJacobian<T> J_ApBq_W)
      : objectA(objectA_in),
        objectB(objectB_in),
        q_PQs(std::move(q_PQs_W)),
        J(std::move(J_ApBq_W)),
        num_constraints(q_PQs.size()) {
    DRAKE_THROW_UNLESS(objectA >= 0);
    DRAKE_THROW_UNLESS(objectB >= 0);
    DRAKE_THROW_UNLESS(q_PQs.size() == num_constraints);
    DRAKE_THROW_UNLESS(J.rows() == num_constraints);
  }

  int objectA{};
  std::optional<int> objectB;
  VectorX<T> q_PQs;
  SapConstraintJacobian<T> J;
  int num_constraints;
};

/* Implements a SAP fixed constraint between two objects.

 Given two objects A and B with at least one of which being a filament, consider
 n scalar constraines: dof Pᵢ on object A (assumed to be filament) and
 coincident dof Qᵢ on object B, for i = 0, 1, ..., n-1. The dofs may be linear
 position or angular position. Working in a common frame W, this constraint
 penalizes non-coincident Pᵢ and Qᵢ, for each i = 0, 1, ..., n-1, with the
 constraint functions:

   gᵢ = p_WQᵢ - p_WPᵢ = 0

 with corresponding constraint velocity:

   ġᵢ = vcᵢ(v) = v_W_PᵢQᵢ

 @tparam_nonsymbolic_scalar */
template <typename T>
class SapFilamentConstraint final : public SapHolonomicConstraint<T> {
 public:
  /* We do not allow copy, move, or assignment generally to avoid slicing. */
  //@{
  SapFilamentConstraint& operator=(const SapFilamentConstraint&) = delete;
  SapFilamentConstraint(SapFilamentConstraint&&) = delete;
  SapFilamentConstraint& operator=(SapFilamentConstraint&&) = delete;
  //@}

  /* Constructs a fixed constraint given its kinematics in a particular
   configuration. */
  explicit SapFilamentConstraint(FilamentConstraintKinematics<T> kinematics);

 private:
  /* Private copy construction is enabled to use in the implementation of
   DoClone(). */
  SapFilamentConstraint(const SapFilamentConstraint&) = default;

  void DoAccumulateGeneralizedImpulses(int, const Eigen::Ref<const VectorX<T>>&,
                                       EigenPtr<VectorX<T>>) const final {
    // TODO(wei-chen): Implement DoAccumulateGeneralizedImpulses() for
    // SapFilamentConstraint.
  }

  void DoAccumulateSpatialImpulses(int i,
                                   const Eigen::Ref<const VectorX<T>>& gamma,
                                   SpatialForce<T>* F) const final {
    // TODO(wei-chen): Implement DoAccumulateSpatialImpulses() for
    // SapFilamentConstraint.
  }

  /* Helper used at construction. This method makes the parameters needed by the
   base class SapHolonomicConstraint. */
  static typename SapHolonomicConstraint<T>::Parameters
  MakeSapHolonomicConstraintParameters(int num_constraint_equations);

  /* Helper used at construction. This method makes the object indices needed by
   the base class SapHolonomicConstraint. */
  static std::vector<int> MakeObjectIndices(
      const FilamentConstraintKinematics<T>& kinematics);

  std::unique_ptr<SapConstraint<T>> DoClone() const final {
    return std::unique_ptr<SapFilamentConstraint<T>>(
        new SapFilamentConstraint<T>(*this));
  }

  // We do not yet support scalar conversion for constraints used for
  // deformables.
  std::unique_ptr<SapConstraint<double>> DoToDouble() const final {
    throw std::runtime_error(
        "SapFilamentConstraint: Scalar conversion to double not supported.");
  }
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
