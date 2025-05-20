#include "drake/multibody/der/dirichlet_boundary_condition.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
void DirichletBoundaryCondition<T>::AddBoundaryCondition(
    DerNodeIndex index, const NodeState<T>& boundary_state) {
  node_to_boundary_state_[index] = boundary_state;
}

template <typename T>
void DirichletBoundaryCondition<T>::AddBoundaryCondition(
    DerEdgeIndex index, const EdgeState<T>& boundary_state) {
  edge_to_boundary_state_[index] = boundary_state;
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
  DRAKE_DEMAND(v != nullptr);
  this->VerifyIndices(v->size());
  if (node_to_boundary_state_.empty() && edge_to_boundary_state_.empty())
    return;

  for (const auto& pair : node_to_boundary_state_) {
    const DerNodeIndex node_index = pair.first;
    v->template segment<3>(4 * node_index).setZero();
  }
  for (const auto& pair : edge_to_boundary_state_) {
    const DerEdgeIndex edge_index = pair.first;
    v->coeffRef(4 * edge_index + 3) = 0.0;
  }
}

template <typename T>
void DirichletBoundaryCondition<T>::ApplyBoundaryConditionToTangentMatrix(
    Block4x4SparseSymmetricMatrix<T>* tangent_matrix) const {
  DRAKE_DEMAND(tangent_matrix != nullptr);
  this->VerifyIndices(tangent_matrix->rows());
  if (node_to_boundary_state_.empty() && edge_to_boundary_state_.empty())
    return;

  const auto& sparsity_pattern_neighbors =
      tangent_matrix->sparsity_pattern().neighbors();
  for (const auto& pair : node_to_boundary_state_) {
    const DerNodeIndex node_index = pair.first;
    for (int j = 0; j < tangent_matrix->block_cols(); ++j) {
      for (int i : sparsity_pattern_neighbors[j]) {  // i ≥ j
        if (!(i == node_index || j == node_index)) continue;
        Eigen::Matrix4<T> block = tangent_matrix->block(i, j);
        if (i == node_index) block.template topRows<3>().setZero();
        if (j == node_index) block.template leftCols<3>().setZero();
        if (i == j) block.template topLeftCorner<3, 3>().setIdentity();
        tangent_matrix->SetBlock(i, j, block);
      }
    }
  }
  for (const auto& pair : edge_to_boundary_state_) {
    const DerEdgeIndex edge_index = pair.first;
    for (int j = 0; j < tangent_matrix->block_cols(); ++j) {
      for (int i : sparsity_pattern_neighbors[j]) {  // i ≥ j
        if (!(i == edge_index || j == edge_index)) continue;
        Eigen::Matrix4<T> block = tangent_matrix->block(i, j);
        if (i == edge_index) block.template bottomRows<1>().setZero();
        if (j == edge_index) block.template rightCols<1>().setZero();
        if (i == j) block(3, 3) = 1.0;
        tangent_matrix->SetBlock(i, j, block);
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
