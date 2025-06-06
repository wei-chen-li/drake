#pragma once

#include <array>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "drake/common/identifier.h"
#include "drake/multibody/der/damping_model.h"
#include "drake/multibody/der/der_indexes.h"
#include "drake/multibody/der/der_state.h"
#include "drake/multibody/der/der_structural_property.h"
#include "drake/multibody/der/der_undeformed_state.h"
#include "drake/multibody/der/dirichlet_boundary_condition.h"
#include "drake/multibody/der/energy_hessian_matrix_utility.h"
#include "drake/multibody/der/external_force_field.h"

namespace drake {
namespace multibody {
namespace der {

/**
 Discrete Elastic Rod (DER) captures the mechanics of a deformable filament
 discretized along its length. A DER is composed of nodes connected sequentially
 by edges and is subject to internal elastic forces arising from stretching,
 bending, and twisting. A DER can have either open ends (e.g., to model a rope)
 or closed ends (e.g., to model a pulley belt).

 For each pair of consecutive nodes xᵢ and xᵢ₊₁, an edge is defined as
 eⁱ = xᵢ₊₁ - xᵢ, with an associated tangent vector tⁱ = eⁱ / ‖eⁱ‖. Subscripts
 denote quantities associated with nodes, while superscripts refer to quantities
 associated with edges. Each edge has an associated reference frame with
 directors (d₁ⁱ, d₂ⁱ, tⁱ), and a material frame with directors (m₁ⁱ, m₂ⁱ, tⁱ).
 The directors m₁ⁱ and m₂ⁱ align with the two principal axes of the rod's
 cross-section, meaning the material frame rotates with the rod. The angle of
 rotation from the reference frame to the material frame is denoted by γⁱ.

 A DER with n nodes and open ends has a configuration vector
 q = [x₀ᵀ γ⁰ x₁ᵀ ... xₙ₋₁ᵀ]ᵀ
 with a total of 4n-1 degrees of freedom.

 A DER with n nodes and closed ends has a configuration vector
 q = [x₀ᵀ γ⁰ x₁ᵀ ... xₙ₋₁ᵀ γⁿ⁻¹]ᵀ
 with a total of 4n degrees of freedom.

 The governing equation for a DER is

  M q̈ = Fᵢₙₜ(q, q̇) + Fₑₓₜ,

 where M is the lumped mass matrix. The internal force Fᵢₙₜ includes elastic
 force, given by −∂E(q)/∂q, where E(q) is the elastic energy, as well as
 internal damping forces. The external force Fₑₓₜ includes effects such as
 gravity.

 @p DerModel provides a method to evaluate the residual

  R(q, q̇, q̈) = M q̈ - Fᵢₙₜ(q, q̇) - Fₑₓₜ.

 It also provides a method to compute a weighted sum of the matrices ∂R/∂q,
 ∂R/∂q̇, and ∂R/∂q̈.

 @tparam_default_scalar
 */
template <typename T>
class DerModel {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DerModel);

  /** @ref DerModel::Builder is a factory class for @ref DerModel.
   It is single use: after calling Build(), this builder should be discarded. */
  class Builder {
   public:
    Builder() = default;

    /** @{
     @name Setting the Rod Configuration
     These methods are used to define the node positions and edge angles of a
     DER. */

    /** Adds the first edge from node x₀ to node x₁. Also specifies the
     reference frame director d₁⁰. If d₁⁰ is not specified, it is choosen as an
     arbitrary director that is perpendicular to x₁ - x₀. The material frame
     director m₁⁰ is the rotation of d₁⁰ around the tangent by γ⁰.
     @return A tuple of indexes of (node, edge, node).
     @pre d1_0 is a unit vector.
     @pre d1_0 is perpendicular to x_1 - x_0. */
    std::tuple<DerNodeIndex, DerEdgeIndex, DerNodeIndex> AddFirstEdge(
        const Eigen::Ref<const Eigen::Vector3<T>>& x_0, const T& gamma_0,
        const Eigen::Ref<const Eigen::Vector3<T>>& x_1,
        const std::optional<Eigen::Vector3<T>>& d1_0 = std::nullopt);

    /** Connects the current end of the rod to the next node xᵢ₊₁. The material
     frame is rotated around the new edge’s tangent by γⁱ relative to the
     reference frame.
     The rod configuration is finalized either when the new node xᵢ₊₁ coincides
     with the first node x₀ (resulting in a closed-ends DER), or when any
     methods under [Setting the Undeformed State](\ref set_undeformed_state) is
     called.
     @return A tuple of indexes of (edge, node).
     @note Must be called after AddFirstEdge(). */
    std::tuple<DerEdgeIndex, DerNodeIndex> AddEdge(
        const T& gamma_i, const Eigen::Ref<const Eigen::Vector3<T>>& x_ip1);
    // @}

    /** @{
     @name Setting the Undeformed State
     @anchor set_undeformed_state
     These methods are used to specify the undeformed (rest) state of the DER,
     including edge lengths, curvatures, and twists. */

    /** Sets the undeformed curvatures and twists to zero; the undeformed edge
     lengths are computed from the current node positions. */
    void SetZeroUndeformedCurvatureAndTwist();

    /** Sets the undeformed edge lengths, curvatures, and twists based on the
     currently specified node positions and edge angles. Effectively, the
     undeformed state matches the initial state. */
    void SetUndeformedStateToInitialState();
    // @}

    /** @{
     @name Setting the Material Property
     This method is used to specify the Young's modulus, shear modulus, and
     mass density of the DER. */

    /** Sets the Young's modulus, shear modulus, and mass density. */
    void SetMaterialProperties(const T& youngs_modulus, const T& shear_modulus,
                               const T& mass_density);
    // @}

    /** @{
     @name Setting the Cross Section
     These methods are used to specify the cross section of the DER. */

    /** Sets the cross section to be a circle. */
    void SetCircularCrossSection(const T& radius);

    /** Sets the cross section to be a rectangle. The `width` is aligned with
     the material frame m₁ director, and the `height` is aligned with the
     material frame m₂ director. */
    void SetRectangularCrossSection(const T& width, const T& height);

    /** Sets the cross section to be a rectangle. One semi-axis with length `a`
     is aligned with the material frame m₁ director; another semi-axis with
     length `b` is aligned with the material frame m₂ director. */
    void SetEllipticalCrossSection(const T& a, const T& b);
    // @}

    /** @{
     @name Setting the Damping Coefficients
     This method is used to specify the damping coefficients of the DER. */

    /** Sets the mass coefficient α and the stiffness coefficient β. */
    void SetDampingCoefficients(const T& mass_coeff_alpha,
                                const T& stiffness_coeff_beta);
    // @}

    /** @{
     @name Building the DER Model
     This method is used to build the DER model. */

    /** Builds the @ref DerModel from the configurations and settings sepcified
     above. */
    std::unique_ptr<DerModel<T>> Build();
    // @}

    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Builder);

   private:
    enum CrossSectionType { kCircular, kRectangular, kElliptical };

    bool is_first_edge_added_{false};
    bool is_configuration_finalized_{false};
    bool is_built_{false};

    bool has_closed_ends_{false};
    std::vector<Eigen::Vector3<T>> node_positions_;
    std::vector<T> edge_angles_;
    std::optional<Eigen::Vector3<T>> d1_0_;

    std::optional<std::tuple<T, T, T>> material_property_;
    std::optional<std::pair<CrossSectionType, std::vector<T>>> cross_section_;
    std::optional<internal::DerUndeformedState<T>> der_undeformed_state_;
    std::optional<internal::DampingModel<T>> damping_model_;
    internal::DirichletBoundaryCondition<T> boundary_condition_;
  };  // class Builder

  /** Returns true if this DER has closed ends. */
  bool has_closed_ends() const { return der_state_system_->has_closed_ends(); }

  /** Returns the number of nodes in this DER. */
  int num_nodes() const { return der_state_system_->num_nodes(); }

  /** Returns the number of edges in this DER. */
  int num_edges() const { return der_state_system_->num_edges(); }

  /** Returns the number of degrees of freedom in this DER. */
  int num_dofs() const { return der_state_system_->num_dofs(); }

  /** Fixes the node position or the edge angle indexed by `index`.
   @pre `index` is within the node index or edge index range. */
  void FixPositionOrAngle(std::variant<DerNodeIndex, DerEdgeIndex> index);

  /** Creates a default DerState compatible with this DER model. */
  std::unique_ptr<internal::DerState<T>> CreateDerState() const;

  /** Struct to hold preallocated memory. */
  struct Scratch;
  struct ScratchDeleter {
    void operator()(Scratch* ptr) const;
  };

  /** Creates a stratch that can be used by ComputeResidual() and
   ComputeTangentMatrix(). */
  std::unique_ptr<Scratch, ScratchDeleter> MakeScratch() const;

  /** Computes the residual R(q, q̇, q̈) (see class doc) evaluated at the given
   `state` and under the `external_force_field`.

   The residual for degrees of freedom under the Dirichlet boundary condition is
   set to zero. Therefore their residual should not be used as a metric for the
   error on the boundary condition.

   @param[in] state The @p DerState at which the residual is evaluated.
   @param[in] external_force_field The external force field to evaluate under.
   @param[in] scratch The scratch allocated using MakeScratch().
   @returns The residual vector.

   @pre `state` is allocated using CreateDerState() of the same DerModel.
   @pre `scratch != nullptr`.
   @pre `scratch` is allocated using MakeScratch() of the same DerModel. */
  const Eigen::VectorX<T>& ComputeResidual(
      const internal::DerState<T>& state,
      const internal::ExternalForceField<T>& external_force_field,
      Scratch* scratch) const;

  /** Computes an approximated tangent matrix evaluated at the given `state`.

   The tangent matrix is given by a weighted sum of the stiffness matrix
   (∂R/∂q), damping matrix (∂R/∂q̇), and mass matrix (∂R/∂q̈).

   The rows and columns in the tangent matrix corresponding to the degrees of
   freedom under the Dirichlet boundary condition is set to zero with the
   exception of the diagonal entries which is set to one.

   If not `has_closed_ends()` meaning `result.rows() == num_dofs() + 1`,
   the last row and last column of the returned `result` is zero with the
   exception of the last diagonal entry which is set to one.

   @param[in] state The @p DerState at which the tangent matrix is evaluated.
   @param[in] weights The weights used to combine stiffness, damping, and mass
                      matrices (in that order) into the tangent matrix.
   @param[in] scratch The scratch allocated using MakeScratch().
   @return `result` The tangent matrix represented by a
   Block4x4SparseSymmetricMatrix.

   @pre `state` is allocated using CreateDerState() of the same DerModel.
   @pre `scratch != nullptr`.
   @pre `scratch` is allocated using MakeScratch() of the same DerModel. */
  const internal::Block4x4SparseSymmetricMatrix<T>& ComputeTangentMatrix(
      const internal::DerState<T>& state, const std::array<T, 3>& weights,
      Scratch* scratch) const;

  /** Applies the boundary condition for this DerModel to the `state`.
   @pre `state != nullptr`.
   @pre `state` is allocated using CreateDerState() of the same DerModel. */
  void ApplyBoundaryCondition(internal::DerState<T>* state) const;

  /** Creates a deep copy of this DerModel. Even though the cloned model is
   functionally identical, any DerState and Scratch created for this model are
   not compatible with the cloned model, and vice versa. */
  std::unique_ptr<DerModel<T>> Clone() const;

  /** Creates a deep copy of this DerModel, transmogrified to use the scalar
   type selected by a template parameter.
   @throws std::exception if this model does not support the destination type.
   @tparam U The destination scalar type, which must be one of the default
             nonsymbolic scalars.  */
  template <typename U>
  std::unique_ptr<DerModel<U>> ToScalarType() const;

  /** Returns the structural property of this model. */
  const internal::DerStructuralProperty<T>& structural_property() const {
    return der_structural_property_;
  }

  /** Checks whether the given `state` is created from `this` DerModel. */
  void ValidateDerState(const internal::DerState<T>& state) const;

 private:
  /* All DerModel of different template type can see other's private fields. */
  template <typename U>
  friend class DerModel;

  /* Friend class to facilitate testing. */
  friend class DerModelTester;

  /* Private constructor. */
  DerModel(std::unique_ptr<const internal::DerStateSystem<T>> der_state_system,
           internal::DerStructuralProperty<T> der_structural_property,
           internal::DerUndeformedState<T> der_undeformed_state,
           internal::DampingModel<T> damping_model,
           internal::DirichletBoundaryCondition<T> boundary_condition);

  /** Checks whether `scratch` is nonnull and created from `this` DerModel. */
  void ValidateScratch(const Scratch* scratch) const;

  const std::unique_ptr<const internal::DerStateSystem<T>> der_state_system_;
  const internal::DerStructuralProperty<T> der_structural_property_;
  const internal::DerUndeformedState<T> der_undeformed_state_;
  const internal::DampingModel<T> damping_model_;
  internal::DirichletBoundaryCondition<T> boundary_condition_;
};

}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::DerModel);
