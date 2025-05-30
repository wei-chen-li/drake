#include "drake/multibody/der/der_state_system.h"

#include <fmt/format.h>

#include "drake/math/axis_angle.h"
#include "drake/math/frame_transport.h"
#include "drake/math/unit_vector.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

/* Returns a zero matrix with size `num_rows` × `num_cols`.  */
template <typename T, int num_rows>
Eigen::Matrix<T, num_rows, Eigen::Dynamic> Zero(int num_cols) {
  static_assert(num_rows > 0);
  DRAKE_DEMAND(num_cols > 0);
  return Eigen::Matrix<T, num_rows, Eigen::Dynamic>::Zero(num_rows, num_cols);
}

/* Assembles the configuration vector `q` by interleaving the columns of
 `initial_node_positions_` and `initial_edge_angles_`. */
template <typename T>
Eigen::VectorX<T> AssembleQVector(
    const std::vector<Eigen::Vector3<T>>& initial_node_positions_,
    const std::vector<T>& initial_edge_angles_) {
  Eigen::VectorX<T> q(ssize(initial_node_positions_) * 3 +
                      ssize(initial_edge_angles_));
  for (int i = 0; i < ssize(initial_node_positions_); ++i) {
    q.template segment<3>(i * 4) = initial_node_positions_[i];
  }
  for (int i = 0; i < ssize(initial_edge_angles_); ++i) {
    q(i * 4 + 3) = initial_edge_angles_[i];
  }
  return q;
}

/* Computes tangent vectors from `initial_node_positions_`. Throws an error if
 any consecutive pair of nodes are too close. */
template <typename T>
Eigen::Matrix<T, 3, Eigen::Dynamic> ComputeUnitTangents(
    bool has_closed_ends,
    const std::vector<Eigen::Vector3<T>>& initial_node_positions_) {
  const int num_nodes = initial_node_positions_.size();
  const int num_edges = has_closed_ends ? num_nodes : num_nodes - 1;
  Eigen::Matrix3X<T> tangent(3, num_edges);
  for (int i = 0; i < num_edges; ++i) {
    const int ip1 = (i + 1) % num_nodes;
    tangent.col(i) = math::internal::NormalizeOrThrow<T>(
        initial_node_positions_[ip1] - initial_node_positions_[i], __func__);
  }
  return tangent;
}

/* Casts objects from type U to type T. */
template <typename T, typename U>
std::vector<Eigen::Vector3<T>> cast(
    const std::vector<Eigen::Vector3<U>>& from) {
  std::vector<Eigen::Vector3<T>> to(from.size());
  for (int i = 0; i < ssize(from); ++i) to[i] = ExtractDoubleOrThrow(from[i]);
  return to;
}
template <typename T, typename U>
std::vector<T> cast(const std::vector<U>& from) {
  std::vector<T> to(from.size());
  for (int i = 0; i < ssize(from); ++i) to[i] = ExtractDoubleOrThrow(from[i]);
  return to;
}
template <typename T, typename U>
std::optional<Eigen::Vector3<T>> cast(
    const std::optional<Eigen::Vector3<U>>& from) {
  std::optional<Eigen::Vector3<T>> to;
  if (from) to = ExtractDoubleOrThrow(*from);
  return to;
}

/* Computes `d2` such that each set of columns in (`d1`, `d2`, `t`) form a
 right-handed orthonormal frame. */
template <typename T>
void CompleteFrames(
    const Eigen::Ref<const Eigen::Matrix<T, 3, Eigen::Dynamic>>& t,
    const Eigen::Ref<const Eigen::Matrix<T, 3, Eigen::Dynamic>>& d1,
    EigenPtr<Eigen::Matrix<T, 3, Eigen::Dynamic>> d2) {
  DRAKE_THROW_UNLESS(d2 != nullptr);
  DRAKE_THROW_UNLESS(t.cols() == d1.cols() && d1.cols() == d2->cols());
  for (int i = 0; i < t.cols(); ++i) {
    math::internal::ThrowIfNotOrthonormal<T>(t.col(i), d1.col(i), __func__);
    d2->col(i) = t.col(i).cross(d1.col(i));
    d2->col(i) /= d2->col(i).norm();
  }
}

/* Remove the derivatives for all entries in the AutoDiffXd matrix. */
void RemoveDerivatives(EigenPtr<Eigen::MatrixX<AutoDiffXd>> mat) {
  DRAKE_THROW_UNLESS(mat != nullptr);
  for (int j = 0; j < mat->cols(); ++j) {
    for (int i = 0; i < mat->rows(); ++i) {
      (*mat)(i, j).derivatives().setZero();
    }
  }
}

}  // namespace

template <typename T>
DerStateSystem<T>::DerStateSystem(
    bool has_closed_ends,  //
    std::vector<Eigen::Vector3<T>> initial_node_positions,
    std::vector<T> initial_edge_angles,
    std::optional<Eigen::Vector3<T>> initial_d1_0)
    : systems::LeafSystem<T>(systems::SystemTypeTag<DerStateSystem>{}),
      has_closed_ends_(has_closed_ends),
      initial_node_positions_(std::move(initial_node_positions)),
      initial_edge_angles_(std::move(initial_edge_angles)),
      initial_d1_0_(std::move(initial_d1_0)) {
  DRAKE_THROW_UNLESS(ssize(initial_node_positions_) == num_nodes());
  DRAKE_THROW_UNLESS(ssize(initial_edge_angles_) == num_edges());
  DRAKE_THROW_UNLESS(num_edges() ==
                     (has_closed_ends_ ? num_nodes() : num_nodes() - 1));
  DRAKE_THROW_UNLESS(num_edges() >= 2);

  auto q = AssembleQVector(initial_node_positions_, initial_edge_angles_);
  DRAKE_DEMAND(q.size() == num_dofs());
  q_index_ = this->DeclareDiscreteState(q);
  qdot_index_ = this->DeclareDiscreteState(num_dofs());
  qddot_index_ = this->DeclareDiscreteState(num_dofs());

  PrevStep<T> prev_step;
  prev_step.tangent =
      ComputeUnitTangents(has_closed_ends, initial_node_positions_);
  prev_step.reference_frame_d1.resize(3, num_edges());
  math::SpaceParallelFrameTransport<T>(prev_step.tangent, initial_d1_0_,
                                       &prev_step.reference_frame_d1);
  prev_step.reference_twist =
      Eigen::Matrix<T, 1, Eigen::Dynamic>::Zero(num_internal_nodes());
  prev_step_index_ = this->DeclareAbstractState(Value(prev_step));

  fix_ref_frame_flag_index_ = systems::AbstractParameterIndex{
      this->DeclareAbstractParameter(Value(false))};

  serial_number_index_ = systems::AbstractParameterIndex{
      this->DeclareAbstractParameter(Value(int64_t(0)))};

  edge_vector_index_ =
      this->DeclareCacheEntry("edge vector", Zero<T, 3>(num_edges()),
                              &DerStateSystem<T>::CalcEdgeVector,
                              {this->discrete_state_ticket(q_index_)})
          .cache_index();
  edge_length_index_ =
      this->DeclareCacheEntry("edge length", Zero<T, 1>(num_edges()),
                              &DerStateSystem<T>::CalcEdgeLength,
                              {this->cache_entry_ticket(edge_vector_index_)})
          .cache_index();
  tangent_index_ =
      this->DeclareCacheEntry("tangent", Zero<T, 3>(num_edges()),
                              &DerStateSystem<T>::CalcTangent,
                              {this->cache_entry_ticket(edge_vector_index_),
                               this->cache_entry_ticket(edge_length_index_)})
          .cache_index();
  reference_frame_d1_index_ =
      this->DeclareCacheEntry("reference frame d1", Zero<T, 3>(num_edges()),
                              &DerStateSystem<T>::CalcReferenceFrameD1,
                              {this->cache_entry_ticket(tangent_index_)})
          .cache_index();
  reference_frame_d2_index_ =
      this->DeclareCacheEntry(
              "reference frame d2", Zero<T, 3>(num_edges()),
              &DerStateSystem<T>::CalcReferenceFrameD2,
              {this->cache_entry_ticket(tangent_index_),
               this->cache_entry_ticket(reference_frame_d1_index_)})
          .cache_index();
  material_frame_m1_index_ =
      this->DeclareCacheEntry(
              "material frame m1", Zero<T, 3>(num_edges()),
              &DerStateSystem<T>::CalcMaterialFrameM1,
              {this->cache_entry_ticket(tangent_index_),
               this->cache_entry_ticket(reference_frame_d1_index_),
               this->discrete_state_ticket(q_index_),
               this->abstract_parameter_ticket(fix_ref_frame_flag_index_)})
          .cache_index();
  material_frame_m2_index_ =
      this->DeclareCacheEntry(
              "material frame m2", Zero<T, 3>(num_edges()),
              &DerStateSystem<T>::CalcMaterialFrameM2,
              {this->cache_entry_ticket(tangent_index_),
               this->cache_entry_ticket(material_frame_m1_index_),
               this->abstract_parameter_ticket(fix_ref_frame_flag_index_)})
          .cache_index();
  discrete_integrated_curvature_index_ =
      this->DeclareCacheEntry(
              "discrete integrated curvature", Zero<T, 3>(num_internal_nodes()),
              &DerStateSystem<T>::CalcDiscreteIntegratedCurvature,
              {this->cache_entry_ticket(tangent_index_)})
          .cache_index();
  curvature_kappa1_index_ =
      this->DeclareCacheEntry(
              "curvature kappa1", Zero<T, 1>(num_internal_nodes()),
              &DerStateSystem<T>::CalcCurvatureKappa1,
              {this->cache_entry_ticket(discrete_integrated_curvature_index_),
               this->cache_entry_ticket(material_frame_m2_index_)})
          .cache_index();
  curvature_kappa2_index_ =
      this->DeclareCacheEntry(
              "curvature kappa2", Zero<T, 1>(num_internal_nodes()),
              &DerStateSystem<T>::CalcCurvatureKappa2,
              {this->cache_entry_ticket(discrete_integrated_curvature_index_),
               this->cache_entry_ticket(material_frame_m1_index_)})
          .cache_index();
  reference_twist_index_ =
      this->DeclareCacheEntry(
              "reference twist", Zero<T, 1>(num_internal_nodes()),
              &DerStateSystem<T>::CalcReferenceTwist,
              {this->cache_entry_ticket(tangent_index_),
               this->cache_entry_ticket(reference_frame_d1_index_),
               this->abstract_parameter_ticket(fix_ref_frame_flag_index_)})
          .cache_index();
  twist_index_ =  //
      this->DeclareCacheEntry("twist", Zero<T, 1>(num_internal_nodes()),
                              &DerStateSystem<T>::CalcTwist,
                              {this->cache_entry_ticket(reference_twist_index_),
                               this->discrete_state_ticket(q_index_)})
          .cache_index();
}

template <typename T>
template <typename U>
DerStateSystem<T>::DerStateSystem(const DerStateSystem<U>& other)
    : DerStateSystem(other.has_closed_ends_,
                     cast<T, U>(other.initial_node_positions_),
                     cast<T, U>(other.initial_edge_angles_),
                     cast<T, U>(other.initial_d1_0_)) {}

template <typename T>
DerStateSystem<T>::~DerStateSystem() = default;

template <typename T>
Eigen::VectorBlock<Eigen::VectorX<T>>
DerStateSystem<T>::get_mutable_position_within_step(
    systems::Context<T>* context) const {
  this->ValidateContext(context);
  increment_serial_number(context);
  return context->get_mutable_discrete_state(q_index_).get_mutable_value();
}

template <typename T>
Eigen::VectorBlock<Eigen::VectorX<T>> DerStateSystem<T>::get_mutable_velocity(
    systems::Context<T>* context) const {
  this->ValidateContext(context);
  increment_serial_number(context);
  return context->get_mutable_discrete_state(qdot_index_).get_mutable_value();
}

template <typename T>
Eigen::VectorBlock<Eigen::VectorX<T>>
DerStateSystem<T>::get_mutable_acceleration(
    systems::Context<T>* context) const {
  this->ValidateContext(context);
  increment_serial_number(context);
  return context->get_mutable_discrete_state(qddot_index_).get_mutable_value();
}

template <typename T>
void DerStateSystem<T>::CalcEdgeVector(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 3, Eigen::Dynamic>* edge_vector) const {
  DRAKE_DEMAND(edge_vector->cols() == num_edges());
  const Eigen::VectorX<T>& q = get_position(context);
  for (int i = 0; i < num_edges(); ++i) {
    const int ip1 = (i + 1) % num_nodes();
    edge_vector->col(i) =
        q.template segment<3>(4 * ip1) - q.template segment<3>(4 * i);
  }
}

template <typename T>
void DerStateSystem<T>::CalcEdgeLength(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 1, Eigen::Dynamic>* edge_length) const {
  DRAKE_DEMAND(edge_length->cols() == num_edges());
  const auto& edge_vector = get_edge_vector(context);
  for (int i = 0; i < edge_length->cols(); ++i) {
    (*edge_length)[i] = edge_vector.col(i).norm();
  }
}

template <typename T>
void DerStateSystem<T>::CalcTangent(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 3, Eigen::Dynamic>* tangent) const {
  DRAKE_DEMAND(tangent->cols() == num_edges());
  const auto& edge_vector = get_edge_vector(context);
  const auto& edge_length = get_edge_length(context);
  const auto& prev_tangent = get_prev_step(context).tangent;
  for (int i = 0; i < num_edges(); ++i) {
    if (edge_length[i] > std::numeric_limits<double>::epsilon()) {
      tangent->col(i) = edge_vector.col(i) / edge_length[i];
    } else {
      // Fallback to the previous step tangent if the denominator is too small.
      tangent->col(i) = prev_tangent.col(i);
    }
  }
}

template <typename T>
void DerStateSystem<T>::CalcReferenceFrameD1(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 3, Eigen::Dynamic>* d1) const {
  DRAKE_DEMAND(d1->cols() == num_edges());
  const auto& prev_tangent = get_prev_step(context).tangent;
  const auto& prev_d1 = get_prev_step(context).reference_frame_d1;
  const auto& tangent = get_tangent(context);
  math::TimeParallelFrameTransport<T>(prev_tangent, prev_d1, tangent, d1);
}

template <typename T>
void DerStateSystem<T>::CalcReferenceFrameD2(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 3, Eigen::Dynamic>* d2) const {
  DRAKE_DEMAND(d2->cols() == num_edges());
  const auto& t = get_tangent(context);
  const auto& d1 = get_reference_frame_d1(context);
  CompleteFrames<T>(t, d1, d2);
}

template <typename T>
void DerStateSystem<T>::CalcMaterialFrameM1(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 3, Eigen::Dynamic>* m1) const {
  DRAKE_DEMAND(m1->cols() == num_edges());
  if constexpr (std::is_same_v<T, AutoDiffXd>) {
    if (get_fix_reference_frame_during_autodiff_flag(context)) {
      FixReferenceFrame_CalcMaterialFrameM1(this, context, m1);
      return;
    }
  }
  const auto& t = get_tangent(context);
  const auto& d1 = get_reference_frame_d1(context);
  const auto& q = get_position(context);
  for (int i = 0; i < num_edges(); ++i) {
    m1->col(i) = math::RotateAxisAngle<T>(d1.col(i), t.col(i), q(4 * i + 3));
  }
}

static void FixReferenceFrame_CalcMaterialFrameM1(
    const DerStateSystem<AutoDiffXd>* self,
    const systems::Context<AutoDiffXd>& context,
    Eigen::Matrix<AutoDiffXd, 3, Eigen::Dynamic>* m1) {
  auto t = self->get_tangent(context);
  RemoveDerivatives(&t);
  auto d1 = self->get_reference_frame_d1(context);
  RemoveDerivatives(&d1);
  const auto& q = self->get_position(context);
  for (int i = 0; i < self->num_edges(); ++i) {
    m1->col(i) =
        math::RotateAxisAngle<AutoDiffXd>(d1.col(i), t.col(i), q(4 * i + 3));
  }
}

template <typename T>
void DerStateSystem<T>::CalcMaterialFrameM2(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 3, Eigen::Dynamic>* m2) const {
  DRAKE_DEMAND(m2->cols() == num_edges());
  if constexpr (std::is_same_v<T, AutoDiffXd>) {
    if (get_fix_reference_frame_during_autodiff_flag(context)) {
      FixReferenceFrame_CalcMaterialFrameM2(this, context, m2);
      return;
    }
  }
  const auto& t = get_tangent(context);
  const auto& m1 = get_material_frame_m1(context);
  CompleteFrames<T>(t, m1, m2);
}

static void FixReferenceFrame_CalcMaterialFrameM2(
    const DerStateSystem<AutoDiffXd>* self,
    const systems::Context<AutoDiffXd>& context,
    Eigen::Matrix<AutoDiffXd, 3, Eigen::Dynamic>* m2) {
  auto t = self->get_tangent(context);
  RemoveDerivatives(&t);
  const auto& m1 = self->get_material_frame_m1(context);
  CompleteFrames<AutoDiffXd>(t, m1, m2);
}

template <typename T>
void DerStateSystem<T>::CalcDiscreteIntegratedCurvature(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 3, Eigen::Dynamic>* curvature) const {
  DRAKE_DEMAND(curvature->cols() == num_internal_nodes());
  const auto& t = get_tangent(context);
  for (int i = 0; i < num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % num_edges();
    curvature->col(i) = 2 * t.col(i).cross(t.col(ip1));
    curvature->col(i) /= 1.0 + t.col(i).dot(t.col(ip1));
  }
}

template <typename T>
void DerStateSystem<T>::CalcCurvatureKappa1(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 1, Eigen::Dynamic>* kappa1) const {
  DRAKE_DEMAND(kappa1->cols() == num_internal_nodes());
  const auto& curvature = get_discrete_integrated_curvature(context);
  const auto& m2 = get_material_frame_m2(context);
  for (int i = 0; i < num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % num_edges();
    (*kappa1)[i] = 0.5 * (m2.col(i) + m2.col(ip1)).dot(curvature.col(i));
  }
}

template <typename T>
void DerStateSystem<T>::CalcCurvatureKappa2(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 1, Eigen::Dynamic>* kappa2) const {
  DRAKE_DEMAND(kappa2->cols() == num_internal_nodes());
  const auto& curvature = get_discrete_integrated_curvature(context);
  const auto& m1 = get_material_frame_m1(context);
  for (int i = 0; i < num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % num_edges();
    (*kappa2)[i] = -0.5 * (m1.col(i) + m1.col(ip1)).dot(curvature.col(i));
  }
}

template <typename T>
void DerStateSystem<T>::CalcReferenceTwist(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 1, Eigen::Dynamic>* ref_twist) const {
  DRAKE_DEMAND(ref_twist->cols() == num_internal_nodes());
  if constexpr (std::is_same_v<T, AutoDiffXd>) {
    if (get_fix_reference_frame_during_autodiff_flag(context)) {
      FixReferenceFrame_CalcReferenceTwist(
          this, context, get_prev_step(context).reference_twist, ref_twist);
      return;
    }
  }
  const auto& prev_ref_twist = get_prev_step(context).reference_twist;
  const auto& t = get_tangent(context);
  const auto& d1 = get_reference_frame_d1(context);
  Eigen::Vector3<T> vec;
  for (int i = 0; i < num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % num_edges();
    /* We transform d₁ⁱ using the transport operator that maps tⁱ to tⁱ⁺¹ and
     write the result to `vec`. */
    math::FrameTransport<T>(t.col(i), d1.col(i), t.col(ip1), vec);
    /* The angle between `vec` and d₁ⁱ⁺¹ is the reference twist. However,
     instead of directly calculating the angle, we first rotate `vec` by the
     previous reference twist. Then we calculate the delta angle and add it
     to the previous reference twist. This avoids incorrect angle wrapping. */
    vec = math::RotateAxisAngle<T>(vec, t.col(ip1), prev_ref_twist[i]);
    (*ref_twist)[i] = prev_ref_twist[i] + math::SignedAngleAroundAxis<T>(
                                              vec, d1.col(ip1), t.col(ip1));
  }
}

static void FixReferenceFrame_CalcReferenceTwist(
    const DerStateSystem<AutoDiffXd>* self,
    const systems::Context<AutoDiffXd>& context,
    const Eigen::Matrix<AutoDiffXd, 1, Eigen::Dynamic>& prev_ref_twist,
    Eigen::Matrix<AutoDiffXd, 1, Eigen::Dynamic>* ref_twist) {
  using T = AutoDiffXd;
  const auto& t = self->get_tangent(context);
  auto d1 = self->get_reference_frame_d1(context);
  RemoveDerivatives(&d1);
  Eigen::Vector3<T> vec;
  for (int i = 0; i < self->num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % self->num_edges();
    math::FrameTransport<T>(t.col(i), d1.col(i), t.col(ip1), vec);
    vec = math::RotateAxisAngle<T>(vec, t.col(ip1), prev_ref_twist[i]);
    (*ref_twist)[i] = prev_ref_twist[i] + math::SignedAngleAroundAxis<T>(
                                              vec, d1.col(ip1), t.col(ip1));
  }
}

template <typename T>
void DerStateSystem<T>::CalcTwist(
    const systems::Context<T>& context,
    Eigen::Matrix<T, 1, Eigen::Dynamic>* twist) const {
  DRAKE_DEMAND(twist->cols() == num_internal_nodes());
  const auto& ref_twist = get_reference_twist(context);
  const auto& q = get_position(context);
  for (int i = 0; i < num_internal_nodes(); ++i) {
    const int edge_i = 4 * i + 3;
    const int edge_ip1 = 4 * ((i + 1) % num_edges()) + 3;
    (*twist)[i] = q[edge_ip1] - q[edge_i] + ref_twist[i];
  }
}

template <typename T>
void DerStateSystem<T>::CopyContext(const systems::Context<T>& from_context,
                                    systems::Context<T>* to_context) const {
  this->ValidateContext(from_context);
  this->ValidateContext(to_context);
  if (&from_context == to_context) return;

  /* Copy the discrete state vectors representing the position, velocity, and
   acceleration, as well as the abstract state holding previous step data. */
  to_context->SetTimeStateAndParametersFrom(from_context);

  /* Copy the cache entry values that are up-to-date to avoid recalculation when
   the `to_context` is later used. */
  for (systems::CacheIndex i(0); i < this->num_cache_entries(); ++i) {
    const systems::CacheEntryValue& from_cache_entry_value =
        this->get_cache_entry(i).get_cache_entry_value(from_context);
    if (from_cache_entry_value.is_out_of_date()) continue;

    systems::CacheEntryValue& to_cache_entry_value =
        this->get_cache_entry(i).get_mutable_cache_entry_value(*to_context);
    to_cache_entry_value.mark_out_of_date();
    to_cache_entry_value.GetMutableAbstractValueOrThrow().SetFrom(
        from_cache_entry_value.GetAbstractValueOrThrow());
    to_cache_entry_value.mark_up_to_date();
  }
}

template <typename T>
Eigen::VectorX<T> DerStateSystem<T>::Serialize(
    const systems::Context<T>& context) const {
  this->ValidateContext(context);
  const int size = num_dofs() * 3 + num_edges() * 6 + num_internal_nodes();
  Eigen::VectorX<T> serialized(size);
  const PrevStep<T>& prev_step = get_prev_step(context);
  DRAKE_DEMAND(prev_step.tangent.cols() == num_edges());
  DRAKE_DEMAND(prev_step.reference_frame_d1.cols() == num_edges());
  DRAKE_DEMAND(prev_step.reference_twist.cols() == num_internal_nodes());
  serialized << get_position(context), get_velocity(context),
      get_acceleration(context), prev_step.tangent.reshaped(),
      prev_step.reference_frame_d1.reshaped(),
      prev_step.reference_twist.reshaped();
  return serialized;
}

template <typename T>
void DerStateSystem<T>::Deserialize(
    systems::Context<T>* context,
    const Eigen::Ref<const Eigen::VectorX<T>>& serialized) const {
  this->ValidateContext(context);
  if (serialized.size() !=
      num_dofs() * 3 + num_edges() * 6 + num_internal_nodes()) {
    throw std::logic_error(
        "The serialized vector has size incompatible with this DerState and "
        "therefore cannot be deserialized.");
  }
  int offset = 0;
  context->SetDiscreteState(q_index_, serialized.segment(offset, num_dofs()));
  offset += num_dofs();
  context->SetDiscreteState(qdot_index_,
                            serialized.segment(offset, num_dofs()));
  offset += num_dofs();
  context->SetDiscreteState(qddot_index_,
                            serialized.segment(offset, num_dofs()));
  offset += num_dofs();

  PrevStep<T>& prev_step =
      context->template get_mutable_abstract_state<PrevStep<T>>(
          prev_step_index_);
  prev_step.tangent =
      serialized.segment(offset, 3 * num_edges()).reshaped(3, num_edges());
  offset += 3 * num_edges();
  prev_step.reference_frame_d1 =
      serialized.segment(offset, 3 * num_edges()).reshaped(3, num_edges());
  offset += 3 * num_edges();
  prev_step.reference_twist = serialized.segment(offset, num_internal_nodes())
                                  .reshaped(1, num_internal_nodes());

  increment_serial_number(context);
}

template <typename T>
int64_t DerStateSystem<T>::serial_number(
    const systems::Context<T>& context) const {
  this->ValidateContext(context);
  return context.get_abstract_parameter(serial_number_index_)
      .template get_value<int64_t>();
}

template <typename T>
void DerStateSystem<T>::increment_serial_number(
    systems::Context<T>* context) const {
  this->ValidateContext(context);
  context->get_mutable_abstract_parameter(serial_number_index_)
      .set_value(serial_number(*context) + 1);
}

template <typename T>
void DerStateSystem<T>::FixReferenceFrameDuringAutoDiff(
    systems::Context<T>* context) const {
  this->ValidateContext(context);
  DRAKE_THROW_UNLESS((std::is_same_v<T, AutoDiffXd>));
  if (!get_fix_reference_frame_during_autodiff_flag(*context)) {
    context->get_mutable_abstract_parameter(fix_ref_frame_flag_index_)
        .set_value(true);
    increment_serial_number(context);
  }
}

template <typename T>
bool DerStateSystem<T>::get_fix_reference_frame_during_autodiff_flag(
    const systems::Context<T>& context) const {
  this->ValidateContext(context);
  return context.get_abstract_parameter(fix_ref_frame_flag_index_)
      .template get_value<bool>();
}

template <typename T>
const PrevStep<T>& DerStateSystem<T>::get_prev_step(
    const systems::Context<T>& context) const {
  this->ValidateContext(context);
  return context.template get_abstract_state<PrevStep<T>>(prev_step_index_);
}

template <typename T>
void DerStateSystem<T>::StorePrevStep(systems::Context<T>* context) const {
  this->ValidateContext(context);
  /* Force the calculation of tangent, reference_frame_d1, and reference_twist
   (if not already calculated and cached). */
  this->get_reference_twist(*context);
  /* Store the previous step tangent, reference_frame_d1, and reference_twist.
   Use GetKnownUpToDate() to ensure that the quantities aren't recalculated.
   Do not add `abstract_state_ticket(prev_step_index_)` to the prerequisites
   of any of the cache entries, otherwise GetKnownUpToDate() will throw. */
  PrevStep<T>& prev_step =
      context->template get_mutable_abstract_state<PrevStep<T>>(
          prev_step_index_);
  prev_step.tangent =
      this->get_cache_entry(tangent_index_)
          .template GetKnownUpToDate<Eigen::Matrix<T, 3, Eigen::Dynamic>>(
              *context);
  prev_step.reference_frame_d1 =
      this->get_cache_entry(reference_frame_d1_index_)
          .template GetKnownUpToDate<Eigen::Matrix<T, 3, Eigen::Dynamic>>(
              *context);
  prev_step.reference_twist =
      this->get_cache_entry(reference_twist_index_)
          .template GetKnownUpToDate<Eigen::Matrix<T, 1, Eigen::Dynamic>>(
              *context);
}

template <typename T>
const Eigen::VectorX<T>& DerStateSystem<T>::get_discrete_state_vector(
    const systems::Context<T>& context,
    systems::DiscreteStateIndex index) const {
  this->ValidateContext(context);
  return context.get_discrete_state(index).value();
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DerStateSystem);
