#pragma once

#include <optional>
#include <vector>

#include "drake/math/rigid_transform.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/framework/scalar_conversion_traits.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/* Forward decleration. */
template <typename T>
struct PrevStep;

/*
 @p DerStateSystem has discrete states representing q, q̇, and q̈ of a discrete
 elastic rod. Furthermore, it contains logic to compute other quantities (e.g.,
 compute the tangent vectors from q), and utilizes the cache mechanism of
 %System to cache the computation results.

 @tparam_default_scalar
 */
template <typename T>
class DerStateSystem final : public systems::LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DerStateSystem);

  /* Constructs a DerStateSystem (open-ends or closed-ends) from the initial
   node positions xᵢ and the edge angles γⁱ. The edge angle is the angle between
   the reference frame and material frame of each edge. d₁⁰ is the reference
   frame director on the first edge, it must be perpendicular to x₁-x₀. */
  DerStateSystem(bool has_closed_ends,
                 std::vector<Eigen::Vector3<T>> initial_node_positions,
                 std::vector<T> initial_edge_angles,
                 std::optional<Eigen::Vector3<T>> initial_d1_0);

  /* Scalar-converting copy constructor.  See @ref system_scalar_conversion. */
  template <typename U>
  explicit DerStateSystem(const DerStateSystem<U>& other);

  ~DerStateSystem() override;

  bool has_closed_ends() const { return has_closed_ends_; }
  int num_nodes() const { return ssize(initial_node_positions_); }
  int num_edges() const {
    return has_closed_ends_ ? num_nodes() : num_nodes() - 1;
  }
  int num_internal_nodes() const {
    return has_closed_ends_ ? num_nodes() : num_nodes() - 2;
  }
  int num_dofs() const { return num_nodes() * 3 + num_edges(); }

  /*
   @name Getting the state vectors
   @{
   */
  const Eigen::VectorX<T>& get_position(
      const systems::Context<T>& context) const {
    return get_discrete_state_vector(context, q_index_);
  }

  const Eigen::VectorX<T>& get_velocity(
      const systems::Context<T>& context) const {
    return get_discrete_state_vector(context, qdot_index_);
  }

  const Eigen::VectorX<T>& get_acceleration(
      const systems::Context<T>& context) const {
    return get_discrete_state_vector(context, qddot_index_);
  }
  // @}

  /*
   @name Setting the state vectors
   @{
   */
  /* Advances the position to its value at the next time step (`next_q`). */
  template <typename Derived>
  void AdvancePositionToNextStep(
      systems::Context<T>* context,
      const Eigen::MatrixBase<Derived>& next_q) const {
    this->ValidateContext(context);
    static_assert(Derived::ColsAtCompileTime == 1);
    DRAKE_THROW_UNLESS(next_q.size() == num_dofs());
    StorePrevStep(context);
    context->get_mutable_discrete_state(q_index_).get_mutable_value() = next_q;
    increment_serial_number(context);
  }

  /* Adjusts the position within the current time step. */
  template <typename Derived>
  void AdjustPositionWithinStep(systems::Context<T>* context,
                                const Eigen::MatrixBase<Derived>& q) const {
    this->ValidateContext(context);
    static_assert(Derived::ColsAtCompileTime == 1);
    DRAKE_THROW_UNLESS(q.size() == num_dofs());
    context->get_mutable_discrete_state(q_index_).get_mutable_value() = q;
    increment_serial_number(context);
  }

  template <typename Derived>
  void SetVelocity(systems::Context<T>* context,
                   const Eigen::MatrixBase<Derived>& qdot) const {
    this->ValidateContext(context);
    static_assert(Derived::ColsAtCompileTime == 1);
    DRAKE_THROW_UNLESS(qdot.size() == num_dofs());
    context->get_mutable_discrete_state(qdot_index_).get_mutable_value() = qdot;
    increment_serial_number(context);
  }

  template <typename Derived>
  void SetAcceleration(systems::Context<T>* context,
                       const Eigen::MatrixBase<Derived>& qddot) const {
    this->ValidateContext(context);
    static_assert(Derived::ColsAtCompileTime == 1);
    DRAKE_THROW_UNLESS(qddot.size() == num_dofs());
    context->get_mutable_discrete_state(qddot_index_).get_mutable_value() =
        qddot;
    increment_serial_number(context);
  }
  // @}

  /*
   @name Dangerous methods for changing the state vectors
   @{
   */
  Eigen::VectorBlock<Eigen::VectorX<T>> get_mutable_position_within_step(
      systems::Context<T>* context) const;

  Eigen::VectorBlock<Eigen::VectorX<T>> get_mutable_velocity(
      systems::Context<T>* context) const;

  Eigen::VectorBlock<Eigen::VectorX<T>> get_mutable_acceleration(
      systems::Context<T>* context) const;
  // @}

  /*
   @name Quantities associated with edges
   @{
   */
  decltype(auto) get_edge_vector(const systems::Context<T>& context) const {
    return get_cache_matrix<3>(context, edge_vector_index_);
  }

  decltype(auto) get_edge_length(const systems::Context<T>& context) const {
    return get_cache_matrix<1>(context, edge_length_index_);
  }

  decltype(auto) get_tangent(const systems::Context<T>& context) const {
    return get_cache_matrix<3>(context, tangent_index_);
  }

  decltype(auto) get_reference_frame_d1(
      const systems::Context<T>& context) const {
    return get_cache_matrix<3>(context, reference_frame_d1_index_);
  }

  decltype(auto) get_reference_frame_d2(
      const systems::Context<T>& context) const {
    return get_cache_matrix<3>(context, reference_frame_d2_index_);
  }

  decltype(auto) get_material_frame_m1(
      const systems::Context<T>& context) const {
    return get_cache_matrix<3>(context, material_frame_m1_index_);
  }

  decltype(auto) get_material_frame_m2(
      const systems::Context<T>& context) const {
    return get_cache_matrix<3>(context, material_frame_m2_index_);
  }
  // @}

  /*
   @name Quantities associated with internal nodes
   @{
   */
  decltype(auto) get_discrete_integrated_curvature(
      const systems::Context<T>& context) const {
    return get_cache_matrix<3>(context, discrete_integrated_curvature_index_);
  }

  decltype(auto) get_curvature_kappa1(
      const systems::Context<T>& context) const {
    return get_cache_matrix<1>(context, curvature_kappa1_index_);
  }

  decltype(auto) get_curvature_kappa2(
      const systems::Context<T>& context) const {
    return get_cache_matrix<1>(context, curvature_kappa2_index_);
  }

  decltype(auto) get_reference_twist(const systems::Context<T>& context) const {
    return get_cache_matrix<1>(context, reference_twist_index_);
  }

  decltype(auto) get_twist(const systems::Context<T>& context) const {
    return get_cache_matrix<1>(context, twist_index_);
  }
  // @}

  /* Performs deep copying from `from_context` into `to_context` including the
   serial number.
   @pre `to_context != nullptr`.
   @pre Both context are created from this system. */
  void CopyContext(const systems::Context<T>& from_context,
                   systems::Context<T>* to_context) const;

  /* Serializes the states in `context` into an Eigen::VectorX. The resulting
   vector will contain q, q̇, q̈ (in that order), and data at the previous time
   step. */
  Eigen::VectorX<T> Serialize(const systems::Context<T>& context) const;

  /* Deserializes the Serialize() returned `serialized` vector into `context`.
   @pre `context != nullptr`.
   @pre `serialized` has the correct size. */
  void Deserialize(systems::Context<T>* context,
                   const Eigen::Ref<const Eigen::VectorX<T>>& serialized) const;

  /* Transforms the states in `context` by `X`.
   @pre `context != nullptr`. */
  void Transform(systems::Context<T>* context,
                 const math::RigidTransform<T>& X) const;

  /* Returns the serial number. The serial number is incremented every time the
   `context` is modified by DerStateStstem. */
  int64_t serial_number(const systems::Context<T>& context) const;

  const std::vector<Eigen::Vector3<T>>& initial_node_positions() const {
    return initial_node_positions_;
  }
  const std::vector<T>& initial_edge_angles() const {
    return initial_edge_angles_;
  }

  /* (Advanced) Makes the derivatives of other quantities with respect to d1 and
   d2 be zero during performing automatic differentiation.
   @pre `std::is_same_v<T, AutoDiffXd>` */
  void FixReferenceFrameDuringAutoDiff(systems::Context<T>* context) const;

 private:
  /* All DerStateSystem of different template type can access other's data. */
  template <typename U>
  friend class DerStateSystem;

  /* Friend class to facilitate testing. */
  friend class DerStateSystemTester;

  void CalcEdgeVector(const systems::Context<T>& context,
                      Eigen::Matrix<T, 3, Eigen::Dynamic>* edge_vector) const;

  void CalcEdgeLength(const systems::Context<T>& context,
                      Eigen::Matrix<T, 1, Eigen::Dynamic>* edge_length) const;

  void CalcTangent(const systems::Context<T>& context,
                   Eigen::Matrix<T, 3, Eigen::Dynamic>* tangent) const;

  void CalcReferenceFrameD1(const systems::Context<T>& context,
                            Eigen::Matrix<T, 3, Eigen::Dynamic>* d1) const;

  void CalcReferenceFrameD2(const systems::Context<T>& context,
                            Eigen::Matrix<T, 3, Eigen::Dynamic>* d2) const;

  void CalcMaterialFrameM1(const systems::Context<T>& context,
                           Eigen::Matrix<T, 3, Eigen::Dynamic>* m1) const;

  void CalcMaterialFrameM2(const systems::Context<T>& context,
                           Eigen::Matrix<T, 3, Eigen::Dynamic>* m2) const;

  void CalcDiscreteIntegratedCurvature(
      const systems::Context<T>& context,
      Eigen::Matrix<T, 3, Eigen::Dynamic>* curvature) const;

  void CalcCurvatureKappa1(const systems::Context<T>& context,
                           Eigen::Matrix<T, 1, Eigen::Dynamic>* kappa1) const;

  void CalcCurvatureKappa2(const systems::Context<T>& context,
                           Eigen::Matrix<T, 1, Eigen::Dynamic>* kappa2) const;

  void CalcReferenceTwist(const systems::Context<T>& context,
                          Eigen::Matrix<T, 1, Eigen::Dynamic>* ref_twist) const;

  void CalcTwist(const systems::Context<T>& context,
                 Eigen::Matrix<T, 1, Eigen::Dynamic>* twist) const;

  bool get_fix_reference_frame_during_autodiff_flag(
      const systems::Context<T>& context) const;

  const PrevStep<T>& get_prev_step(const systems::Context<T>& context) const;

  void StorePrevStep(systems::Context<T>* context) const;

  const Eigen::VectorX<T>& get_discrete_state_vector(
      const systems::Context<T>& context,
      systems::DiscreteStateIndex index) const;

  template <int num_rows>
  const Eigen::Matrix<T, num_rows, Eigen::Dynamic>& get_cache_matrix(
      const systems::Context<T>& context, systems::CacheIndex index) const {
    this->ValidateContext(context);
    return this->get_cache_entry(index)
        .template Eval<Eigen::Matrix<T, num_rows, Eigen::Dynamic>>(context);
  }

  void increment_serial_number(systems::Context<T>* context) const;

  const bool has_closed_ends_;
  const std::vector<Eigen::Vector3<T>> initial_node_positions_;
  const std::vector<T> initial_edge_angles_;
  const std::optional<Eigen::Vector3<T>> initial_d1_0_;

  systems::DiscreteStateIndex q_index_{};
  systems::DiscreteStateIndex qdot_index_{};
  systems::DiscreteStateIndex qddot_index_{};
  systems::AbstractStateIndex prev_step_index_{};
  systems::AbstractParameterIndex fix_ref_frame_flag_index_{};
  systems::AbstractParameterIndex serial_number_index_{};
  systems::CacheIndex edge_vector_index_{};
  systems::CacheIndex edge_length_index_{};
  systems::CacheIndex tangent_index_{};
  systems::CacheIndex reference_frame_d1_index_{};
  systems::CacheIndex reference_frame_d2_index_{};
  systems::CacheIndex material_frame_m1_index_{};
  systems::CacheIndex material_frame_m2_index_{};
  systems::CacheIndex discrete_integrated_curvature_index_{};
  systems::CacheIndex curvature_kappa1_index_{};
  systems::CacheIndex curvature_kappa2_index_{};
  systems::CacheIndex reference_twist_index_{};
  systems::CacheIndex twist_index_{};
};

/* Struct storing previous step data. */
template <typename T>
struct PrevStep {
  Eigen::Matrix<T, 3, Eigen::Dynamic> tangent;
  Eigen::Matrix<T, 3, Eigen::Dynamic> reference_frame_d1;
  Eigen::Matrix<T, 1, Eigen::Dynamic> reference_twist;
};

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DerStateSystem);
