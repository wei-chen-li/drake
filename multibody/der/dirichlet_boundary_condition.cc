#include "drake/multibody/der/dirichlet_boundary_condition.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
void DirichletBoundaryCondition<T>::AddBoundaryCondition(
    DerNodeIndex index, const NodeState<T>& boundary_state) {
  node_to_boundary_state_[index] = boundary_state;
  dofs_.insert(4 * index);
  dofs_.insert(4 * index + 1);
  dofs_.insert(4 * index + 2);
}

template <typename T>
const NodeState<T>* DirichletBoundaryCondition<T>::GetBoundaryCondition(
    DerNodeIndex index) const {
  auto iter = node_to_boundary_state_.find(index);
  if (iter == node_to_boundary_state_.end())
    return nullptr;
  else
    return &iter->second;
}

template <typename T>
void DirichletBoundaryCondition<T>::AddBoundaryCondition(
    DerEdgeIndex index, const EdgeState<T>& boundary_state) {
  edge_to_boundary_state_[index] = boundary_state;
  dofs_.insert(4 * index + 3);
}

template <typename T>
const EdgeState<T>* DirichletBoundaryCondition<T>::GetBoundaryCondition(
    DerEdgeIndex index) const {
  auto iter = edge_to_boundary_state_.find(index);
  if (iter == edge_to_boundary_state_.end())
    return nullptr;
  else
    return &iter->second;
}

template <typename T>
void DirichletBoundaryCondition<T>::ApplyBoundaryConditionToState(
    DerState<T>* der_state) const {
  DRAKE_THROW_UNLESS(der_state != nullptr);
  this->VerifyIndices(der_state->num_dofs());
  if (node_to_boundary_state_.empty() && edge_to_boundary_state_.empty())
    return;

  Eigen::VectorBlock<Eigen::VectorX<T>> q =
      der_state->get_mutable_position_within_step();
  Eigen::VectorBlock<Eigen::VectorX<T>> v = der_state->get_mutable_velocity();
  Eigen::VectorBlock<Eigen::VectorX<T>> a =
      der_state->get_mutable_acceleration();
  for (const auto& [node_index, boundary_state] : node_to_boundary_state_) {
    q.template segment<3>(4 * node_index) = boundary_state.x;
    v.template segment<3>(4 * node_index) = boundary_state.x_dot;
    a.template segment<3>(4 * node_index) = boundary_state.x_ddot;
  }
  for (const auto& [edge_index, boundary_state] : edge_to_boundary_state_) {
    q(4 * edge_index + 3) = boundary_state.gamma;
    v(4 * edge_index + 3) = boundary_state.gamma_dot;
    a(4 * edge_index + 3) = boundary_state.gamma_ddot;
  }
}

template <typename T>
void DirichletBoundaryCondition<T>::ApplyHomogeneousBoundaryCondition(
    EigenPtr<VectorX<T>> v) const {
  DRAKE_THROW_UNLESS(v != nullptr);
  this->VerifyIndices(v->size());
  if (node_to_boundary_state_.empty() && edge_to_boundary_state_.empty())
    return;

  for (const int dof : dofs_) {
    v->coeffRef(dof) = 0.0;
  }
}

template <typename T>
void DirichletBoundaryCondition<T>::ApplyBoundaryConditionToTangentMatrix(
    Eigen::SparseMatrix<T>* tangent_matrix) const {
  DRAKE_THROW_UNLESS(tangent_matrix != nullptr);
  DRAKE_THROW_UNLESS(tangent_matrix->rows() == tangent_matrix->cols());
  this->VerifyIndices(tangent_matrix->rows());
  if (node_to_boundary_state_.empty() && edge_to_boundary_state_.empty())
    return;

  for (int k = 0; k < tangent_matrix->outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(*tangent_matrix, k);
         it; ++it) {
      const int i = it.row();
      const int j = it.col();
      if (dofs_.contains(i) || dofs_.contains(j)) {
        if (i != j)
          it.valueRef() = 0.0;
        else
          it.valueRef() = 1.0;
      }
    }
  }
}

template <typename T>
void DirichletBoundaryCondition<T>::VerifyIndices(int num_dofs) const {
  const int largest_node_index = node_to_boundary_state_.empty()
                                     ? -1
                                     : node_to_boundary_state_.crbegin()->first;
  if (4 * largest_node_index + 2 >= num_dofs) {
    throw std::out_of_range(
        "A node index of the Dirichlet boundary condition is out of range.");
  }
  const int largest_edge_index = edge_to_boundary_state_.empty()
                                     ? -1
                                     : edge_to_boundary_state_.crbegin()->first;
  if (4 * largest_edge_index + 3 >= num_dofs) {
    throw std::out_of_range(
        "An edge index of the Dirichlet boundary condition is out of range.");
  }
}

template <typename T>
template <typename U>
DirichletBoundaryCondition<U> DirichletBoundaryCondition<T>::ToScalarType()
    const {
  static_assert(!std::is_same_v<T, U>);
  DirichletBoundaryCondition<U> to;
  NodeState<U> to_node_state;
  for (const auto& [node_index, from_node_state] : node_to_boundary_state_) {
    to_node_state.x = ExtractDoubleOrThrow(from_node_state.x);
    to_node_state.x_dot = ExtractDoubleOrThrow(from_node_state.x_dot);
    to_node_state.x_ddot = ExtractDoubleOrThrow(from_node_state.x_ddot);
    to.AddBoundaryCondition(node_index, to_node_state);
  }
  EdgeState<U> to_edge_state;
  for (const auto& [edge_index, from_edge_state] : edge_to_boundary_state_) {
    to_edge_state.gamma = ExtractDoubleOrThrow(from_edge_state.gamma);
    to_edge_state.gamma_dot = ExtractDoubleOrThrow(from_edge_state.gamma_dot);
    to_edge_state.gamma_ddot = ExtractDoubleOrThrow(from_edge_state.gamma_ddot);
    to.AddBoundaryCondition(edge_index, to_edge_state);
  }
  return to;
}

using symbolic::Expression;
template DirichletBoundaryCondition<AutoDiffXd>
DirichletBoundaryCondition<double>::ToScalarType<AutoDiffXd>() const;
template DirichletBoundaryCondition<Expression>
DirichletBoundaryCondition<double>::ToScalarType<Expression>() const;
template DirichletBoundaryCondition<double>
DirichletBoundaryCondition<AutoDiffXd>::ToScalarType<double>() const;
template DirichletBoundaryCondition<Expression>
DirichletBoundaryCondition<AutoDiffXd>::ToScalarType<Expression>() const;
template DirichletBoundaryCondition<double>
DirichletBoundaryCondition<Expression>::ToScalarType<double>() const;
template DirichletBoundaryCondition<AutoDiffXd>
DirichletBoundaryCondition<Expression>::ToScalarType<AutoDiffXd>() const;

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DirichletBoundaryCondition);
