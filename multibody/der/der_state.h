#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/multibody/der/der_state_system.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/*
 @p DerState holds q, q̇, and q̈ of a discrete elastic rod. Furthermore, it has
 methods for evaluating the quantities associated with edges and internal nodes.

 has_closed_ends = false
                   ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
      node index   0   1   2   3   4   5   6   7
        edge index   0   1   2   3   4   5   6
 internal node index   0   1   2   3   4   5


 has_closed_ends = true
           ┌────── 5 ─ 4 ──────────┐
           │  ┌───── 5 ─ 4 ─────┐  │
           │  │  ┌──── 5 ─ 4 ┐  │  │
           │  │  6 ●━━━●━━━● │  │  3
           │  6  │ ┃       ┃ │  3  │
           6  │  7 ●       ● 3  │  2
           │  7    ┃       ┃ │  2  │
           7       ●━━━●━━━● 2  │  │
      node index ─ 0 ─ 1 ────┘  │  │
        edge index ─ 0 ─ 1 ─────┘  │
 internal node index ─ 0 ─ 1 ──────┘

 @tparam_default_scalar
 */
template <typename T>
class DerState {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DerState);

  /* Constructs a DerState referencing an externally owned DerStateSystem. This
   DerState must not outlive the DerStateSystem. */
  explicit DerState(const DerStateSystem<T>* der_state_system);

  /*
   @name Properties of the DER
   @{
   These methods provide the properties of the DerModel that created this state.
   */
  bool has_closed_ends() const { return der_state_system_->has_closed_ends(); }
  int num_nodes() const { return der_state_system_->num_nodes(); }
  int num_edges() const { return der_state_system_->num_edges(); }
  int num_internal_nodes() const {
    return der_state_system_->num_internal_nodes();
  }
  int num_dofs() const { return der_state_system_->num_dofs(); }
  // @}

  /*
   @name Getting the state vectors
   @{
   These methods get the state vectors: position q, velocity q̇, and acceleration
   q̈. The position and velocity are actual state variables, while the
   acceleration is the saved result of previous calculations required by certain
   integrators such as NewmarkScheme.
   */
  const Eigen::VectorX<T>& get_position() const {
    return der_state_system_->get_position(*context_);
  }

  const Eigen::VectorX<T>& get_velocity() const {
    return der_state_system_->get_velocity(*context_);
  }

  const Eigen::VectorX<T>& get_acceleration() const {
    return der_state_system_->get_acceleration(*context_);
  }
  // @}

  /*
   @name Setting the state vectors
   @{
   These methods set the state vectors: position q, velocity q̇, and acceleration
   q̈.

   The calculation of some quantities depend on both the current and previous
   time step positions. To handle this, there are two distinct methods for
   setting the position:

   - AdjustPositionWithinStep(): Updates the position vector *within* the
     current time step, without affecting the previous step data.

   - AdvancePositionToNextStep(): Finalizes the current position and advances
     the system to the next time step. The current position becomes the
     "previous" one of which future calculations depend on.

   The typical usage would be to first call AdvancePositionToNextStep() with an
   initial guess for the new position vector, then refine that guess by calling
   AdjustPositionWithinStep() as needed.
   */
  /* Advances the position to its value at the next time step (`next_q`). */
  template <typename Derived>
  void AdvancePositionToNextStep(const Eigen::MatrixBase<Derived>& next_q) {
    der_state_system_->AdvancePositionToNextStep(context_.get_mutable(),
                                                 next_q);
  }

  /* Adjusts the position within the current time step. */
  template <typename Derived>
  void AdjustPositionWithinStep(const Eigen::MatrixBase<Derived>& q) {
    der_state_system_->AdjustPositionWithinStep(context_.get_mutable(), q);
  }

  template <typename Derived>
  void SetVelocity(const Eigen::MatrixBase<Derived>& qdot) {
    der_state_system_->SetVelocity(context_.get_mutable(), qdot);
  }

  template <typename Derived>
  void SetAcceleration(const Eigen::MatrixBase<Derived>& qddot) {
    der_state_system_->SetAcceleration(context_.get_mutable(), qddot);
  }
  // @}

  /*
   @name Dangerous methods for changing the state vectors
   @{
   These methods return mutable references into the state vectors: position q,
   velocity q̇, and acceleration q̈. Although they do issue out-of-date
   notifications when invoked, so you can safely write to the reference once,
   there is no way to issue notifications if you make subsequent changes. So you
   must not hold these references for writing.
   */
  Eigen::VectorBlock<Eigen::VectorX<T>> get_mutable_position_within_step() {
    return der_state_system_->get_mutable_position_within_step(
        context_.get_mutable());
  }

  Eigen::VectorBlock<Eigen::VectorX<T>> get_mutable_velocity() {
    return der_state_system_->get_mutable_velocity(context_.get_mutable());
  }

  Eigen::VectorBlock<Eigen::VectorX<T>> get_mutable_acceleration() {
    return der_state_system_->get_mutable_acceleration(context_.get_mutable());
  }
  // @}

  /*
   @name Quantities associated with edges
   @{
   These methods evaluate (compute or retrieve from cache) quantities associated
   with edges.
   */
  const Eigen::Matrix<T, 1, Eigen::Dynamic>& get_edge_length() const {
    return der_state_system_->get_edge_length(*context_);
  }

  const Eigen::Matrix<T, 3, Eigen::Dynamic>& get_tangent() const {
    return der_state_system_->get_tangent(*context_);
  }

  const Eigen::Matrix<T, 3, Eigen::Dynamic>& get_reference_frame_d1() const {
    return der_state_system_->get_reference_frame_d1(*context_);
  }

  const Eigen::Matrix<T, 3, Eigen::Dynamic>& get_reference_frame_d2() const {
    return der_state_system_->get_reference_frame_d2(*context_);
  }

  const Eigen::Matrix<T, 3, Eigen::Dynamic>& get_material_frame_m1() const {
    return der_state_system_->get_material_frame_m1(*context_);
  }

  const Eigen::Matrix<T, 3, Eigen::Dynamic>& get_material_frame_m2() const {
    return der_state_system_->get_material_frame_m2(*context_);
  }
  // @}

  /*
   @name Quantities associated with internal nodes
   @{
   These methods evaluate (compute or retrieve from cache) quantities associated
   with internal nodes.
   */
  const Eigen::Matrix<T, 3, Eigen::Dynamic>& get_discrete_integrated_curvature()
      const {
    return der_state_system_->get_discrete_integrated_curvature(*context_);
  }

  const Eigen::Matrix<T, 1, Eigen::Dynamic>& get_curvature_kappa1() const {
    return der_state_system_->get_curvature_kappa1(*context_);
  }

  const Eigen::Matrix<T, 1, Eigen::Dynamic>& get_curvature_kappa2() const {
    return der_state_system_->get_curvature_kappa2(*context_);
  }

  const Eigen::Matrix<T, 1, Eigen::Dynamic>& get_twist() const {
    return der_state_system_->get_twist(*context_);
  }
  // @}

  /* Makes `this` DerState an exact copy of the given `other` DerState.
   @pre Both DerState are created using the same DerStateSystem. */
  void CopyFrom(const DerState<T>& other);

  /* Returns an identical copy of this DerState. */
  std::unique_ptr<DerState<T>> Clone() const;

  /* Serializes `this` DerState into an Eigen::VectorX. The resulting vector
   will contain q, q̇, q̈ (in that order), and data at the previous time step. */
  Eigen::VectorX<T> Serialize() const {
    return der_state_system_->Serialize(*context_);
  }

  /* Deserializes the Serialize() returned `serialized` vector into `this`.
   @pre `serialized` has the correct size. */
  void Deserialize(const Eigen::Ref<const Eigen::VectorX<T>>& serialized) {
    der_state_system_->Deserialize(context_.get_mutable(), serialized);
  }

  /* Returns the serial number. This counts up every time a setter method is
   called or whenever mutable access is granted. */
  int64_t serial_number() const {
    return der_state_system_->serial_number(*context_);
  }

  /* (Advanced.) Make the derivatives of other quantities with respect to d1 and
   d2 be zero when performing automatic differentiation. */
  template <typename U = T,
            std::enable_if_t<std::is_same_v<U, AutoDiffXd>, bool> = true>
  void FixReferenceFrameDuringAutoDiff() {
    der_state_system_->FixReferenceFrameDuringAutoDiff(context_.get_mutable());
  }

  /* Returns true if this DerState is constructed from the given
   `der_state_system`. */
  bool is_created_from_system(const DerStateSystem<T>& der_state_system) const {
    return &der_state_system == der_state_system_;
  }

 private:
  DerState(const DerStateSystem<T>* der_state_system,
           std::unique_ptr<Context<T>> context);

  const DerStateSystem<T>* const der_state_system_;
  copyable_unique_ptr<Context<T>> context_;
};

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DerState);
