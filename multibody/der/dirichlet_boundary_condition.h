#pragma once

#include <map>
#include <unordered_set>

#include "drake/multibody/der/der_indexes.h"
#include "drake/multibody/der/der_state.h"
#include "drake/multibody/der/energy_hessian_matrix_utility.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/* The position, velocity, and acceleration of a DER node.
 @tparam_nonsymbolic_scalar */
template <typename T>
struct NodeState {
  Eigen::Vector3<T> x;
  Eigen::Vector3<T> x_dot;
  Eigen::Vector3<T> x_ddot;
};

/* The angular position, velocity, and acceleration of a DER edge.
 @tparam_nonsymbolic_scalar */
template <typename T>
struct EdgeState {
  T gamma;
  T gamma_dot;
  T gamma_ddot;
};

/* DirichletBoundaryCondition provides functionalities related to Dirichlet
 boundary conditions (BC) on DER models. In particular, it provides the
 following functionalities:
 1. storing the information necessary to apply the BC;
 2. modifying a given state to comply with the stored BC;
 3. modifying a given residual/tangent matrix that arises from the DER model
 without BC and transform it into the residual/tangent matrix for the same
 model under the stored BC.

 For the node degrees of freedom (DoFs), the BC must be jointly specified. That
 means that either all DoFs of a node are subject to BCs or none of the DoFs of
 the node is subject to BCs. For example, we don't allow specifying BCs for
 just the x-components of the states while leaving the y and z components free.

 @tparam_default_scalar */
template <typename T>
class DirichletBoundaryCondition {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(DirichletBoundaryCondition);

  /* Constructs an empty boundary condition. */
  DirichletBoundaryCondition() {}

  /* Sets the node with the given `index` to be subject to the prescribed
   `boundary_state`. If multiple boundary conditions are specified for the same
    node, the last one will be used.
   @pre `index` is valid and `boundary_state` values are finite. */
  void AddBoundaryCondition(DerNodeIndex index,
                            const NodeState<T>& boundary_state);

  /* Retrieves the boundary condition associated with the given `index`. Returns
   nullptr is none exists.
   @pre `index` is valid. */
  const NodeState<T>* GetBoundaryCondition(DerNodeIndex index) const;

  /* Sets the edge with the given `index` to be subject to the prescribed
   `boundary_state`. If multiple boundary conditions are specified for the same
    edge, the last one will be used.
   @pre `index` is valid and `boundary_state` values are finite. */
  void AddBoundaryCondition(DerEdgeIndex index,
                            const EdgeState<T>& boundary_state);

  /* Retrieves the boundary condition associated with the given `index`. Returns
   nullptr is none exists.
   @pre `index` is valid. */
  const EdgeState<T>* GetBoundaryCondition(DerEdgeIndex index) const;

  /* Modifies the given `der_state` to comply with this boundary condition.
   @pre `der_state != nullptr`.
   @throw std::exception if the index of any DoF subject to `this` BC is greater
          than or equal to `der_state->num_dofs()`. */
  void ApplyBoundaryConditionToState(DerState<T>* der_state) const;

  /* Modifies the given vector `v` (e.g, the residual of the system or the
   velocities/positions) that arises from an DER model without BC into the a
   vector for the same model subject to `this` BC. More specifically, the
   entries corresponding to nodes under the BC will be zeroed out.
   @pre `v != nullptr`.
   @throw std::exception if the index of any DoF subject to `this` BC is greater
          than or equal to `v->size()`. */
  void ApplyHomogeneousBoundaryCondition(EigenPtr<VectorX<T>> v) const;

  /* Modifies the given tangent matrix that arises from an DER model into the
   tangent matrix for the same model subject to this BC. More specifically,
   the rows and columns corresponding to DoFs under this BC will be zeroed out
   with the exception of the diagonal entries for those DoFs which will be set
   to one.
   @pre `tangent_matrix != nullptr`.
   @pre `tangent_matrix->rows() == tangent_matrix->cols()`.
   @throw std::exception if the index of any DoF subject to `this` BC is greater
          than or equal to `tangent_matrix->rows()`. */
  void ApplyBoundaryConditionToTangentMatrix(
      Eigen::SparseMatrix<T>* tangent_matrix) const;

  template <typename U>
  DirichletBoundaryCondition<U> ToScalarType() const;

 private:
  /* Throws an exception if any DoF subject to `this` BC is greater
   than or equal to `num_dofs`. */
  void VerifyIndices(int num_dofs) const;

  std::map<DerNodeIndex, NodeState<T>> node_to_boundary_state_{};
  std::map<DerEdgeIndex, EdgeState<T>> edge_to_boundary_state_{};
  std::unordered_set<int> dofs_;
};

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DirichletBoundaryCondition);
