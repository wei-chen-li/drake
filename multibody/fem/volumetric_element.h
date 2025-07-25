#pragma once

#include <array>
#include <type_traits>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/fem_element.h"
#include "drake/multibody/fem/isoparametric_element.h"
#include "drake/multibody/fem/quadrature.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* The data struct that stores per element data for VolumetricElement. See
 FemElement for the requirement. We define it here instead of nesting it in the
 traits class below due to #17109. */
template <typename ConstitutiveModelType, int num_dofs,
          int num_quadrature_points>
struct VolumetricElementData {
  using T = typename ConstitutiveModelType::T;
  /* The states evaluated at nodes of the element. */
  Vector<T, num_dofs> element_q;
  Vector<T, num_dofs> element_q0;
  Vector<T, num_dofs> element_v;
  Vector<T, num_dofs> element_a;
  /* The current locations of the quadrature points in the world frame. */
  std::array<Vector<T, 3>, num_quadrature_points> quadrature_positions;
  /* The current velocities of the quadrature points in the world frame. */
  std::array<Vector<T, 3>, num_quadrature_points> quadrature_velocities;

  using DeformationGradientData = typename ConstitutiveModelType::Data;
  std::array<DeformationGradientData, num_quadrature_points>
      deformation_gradient_data;
  /* The elastic energy density evaluated at quadrature points. Note that this
   is energy per unit of "reference" volume. */
  std::array<T, num_quadrature_points> Psi;
  /* The first Piola stress evaluated at quadrature points. */
  std::array<Matrix3<T>, num_quadrature_points> P;
  /* The derivative of first Piola stress with respect to the deformation
   gradient evaluated at quadrature points. */
  std::array<math::internal::FourthOrderTensor<T>, num_quadrature_points> dPdF;
  /* Inverse dynamics: Ma-fₑ(x)-fᵥ(x, v). */
  Vector<T, num_dofs> inverse_dynamics;
  /* The stiffness matrix K. */
  Eigen::Matrix<T, num_dofs, num_dofs> stiffness_matrix;
  /* The damping matrix D = aM + bK where M is the mass matrix. */
  Eigen::Matrix<T, num_dofs, num_dofs> damping_matrix;
  /* The tangent matrix matrix = w0*K + w1*D + w2*D. */
  Eigen::Matrix<T, num_dofs, num_dofs> tangent_matrix;
};

/* Forward declaration needed for defining the traits below. */
template <class IsoparametricElementType, class QuadratureType,
          class ConstitutiveModelType>
class VolumetricElement;

/* The traits class for volumetric elasticity FEM element. */
template <class IsoparametricElementType, class QuadratureType,
          class ConstitutiveModelType>
struct FemElementTraits<VolumetricElement<
    IsoparametricElementType, QuadratureType, ConstitutiveModelType>> {
  /* Check that template parameters are of the correct types. */
  static_assert(
      is_isoparametric_element<IsoparametricElementType>::value,
      "The IsoparametricElementType template parameter must be a derived "
      "class of IsoparametricElement");
  static_assert(
      is_quadrature<QuadratureType>::value,
      "The QuadratureType template parameter must be a derived class of "
      "Quadrature<T, natural_dimension, num_quadrature_points>, where "
      "`natural_dimension` can be 1, 2 or 3.");
  static_assert(
      is_constitutive_model<ConstitutiveModelType>::value,
      "The ConstitutiveModelType template parameter must be a derived "
      "class of ConstitutiveModel");
  /* Check that the scalar types are compatible. */
  static_assert(std::is_same_v<typename IsoparametricElementType::T,
                               typename ConstitutiveModelType::T>,
                "The scalar type of the isoparametric element and the "
                "constitutive model must be the same.");
  /* Check that the number of quadrature points are compatible. */
  static_assert(
      QuadratureType::num_quadrature_points ==
          IsoparametricElementType::num_sample_locations,
      "The number of quadrature points of the quadrature rule must be the same "
      "as the number of evaluation locations in the isoparametric element.");
  /* Check that the natural dimensions are compatible. */
  static_assert(IsoparametricElementType::natural_dimension ==
                    QuadratureType::natural_dimension,
                "The natural dimension of the isoparametric element and the "
                "quadrature rule must be the same.");
  /* Only 3D elasticity is supported. */
  static_assert(IsoparametricElementType::spatial_dimension == 3,
                "The spatial dimension of the isoparametric element must be 3 "
                "for volumetric FEM elements.");
  /* Codimensional objects are not fully supported yet. */
  static_assert(IsoparametricElementType::natural_dimension == 3,
                "The natural dimension of the isoparametric element must be 3 "
                "for volumetric FEM elements. Codimensional objects are not "
                "yet supported.");

  using T = typename ConstitutiveModelType::T;
  using ConstitutiveModel = ConstitutiveModelType;

  static constexpr int num_nodes = IsoparametricElementType::num_nodes;
  static constexpr int num_quadrature_points =
      QuadratureType::num_quadrature_points;
  static constexpr int natural_dimension = QuadratureType::natural_dimension;
  /* The number of degrees of freedom is equal to the spatial dimension (which
   gives the number of degrees of freedom for a single node) times the number of
   nodes. */
  static constexpr int num_dofs = 3 * num_nodes;

  using Data = VolumetricElementData<ConstitutiveModelType, num_dofs,
                                     num_quadrature_points>;
};

/* This class models a single 3D elasticity FEM element in which the
 displacement of the material can be interpolated from that of the element nodes
 using the isoparametric shape function.
 @tparam IsoparametricElementType  The type of isoparametric element used in
                                   this VolumetricElement.
                                   IsoparametricElementType must derived
                                   from IsoparametricElement.
 @tparam QuadratureType  The type of quadrature rule used in this
                         VolumetricElement. QuadratureType must be derived from
                         Quadrature.
 @tparam ConstitutiveModelType  The type of constitutive model used in this
                                VolumetricElement. ConstitutiveModelType must be
                                derived from ConstitutiveModel. */
template <class IsoparametricElementType, class QuadratureType,
          class ConstitutiveModelType>
class VolumetricElement
    : public FemElement<VolumetricElement<
          IsoparametricElementType, QuadratureType, ConstitutiveModelType>> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(VolumetricElement);

  using ElementType = VolumetricElement<IsoparametricElementType,
                                        QuadratureType, ConstitutiveModelType>;
  using IsoparametricElement = IsoparametricElementType;
  using Quadrature = QuadratureType;

  using Traits = FemElementTraits<ElementType>;
  using Data = typename Traits::Data;
  using T = typename Traits::T;
  static constexpr int natural_dimension = Traits::natural_dimension;
  static constexpr int kSpatialDimension = 3;
  static constexpr int num_quadrature_points = Traits::num_quadrature_points;
  static constexpr int num_dofs = Traits::num_dofs;
  static constexpr int num_nodes = Traits::num_nodes;

  /* Constructs a new VolumetricElement. In that process, precomputes the mass
   matrix and the gravity force acting on the element.
   @param[in] node_indices         The node indices of the nodes of this
                                   element.
   @param[in] constitutive_model   The ConstitutiveModel to be used for this
                                   element.
   @param[in] reference_positions  The positions (in world frame) of the nodes
                                   of this element in the reference
                                   configuration. The positions must be such
                                   that the element is not degenerate or
                                   inverted.
   @param[in] density              The mass density of the element with unit
                                   kg/m³.
   @param[in] damping_model        The DampingModel to be used for this element.
   @pre element_index and node_indices are valid.
   @pre density > 0. */
  VolumetricElement(const std::array<FemNodeIndex, num_nodes>& node_indices,
                    ConstitutiveModelType constitutive_model,
                    const Eigen::Ref<const Eigen::Matrix<T, 3, num_nodes>>&
                        reference_positions,
                    T density, DampingModel<T> damping_model)
      : FemElement<ElementType>(node_indices, std::move(constitutive_model),
                                std::move(damping_model)),
        density_(std::move(density)) {
    DRAKE_DEMAND(density_ > 0);
    /* Computes the Jacobian of the change of variable function X(ξ). */
    const std::array<Eigen::Matrix<T, 3, natural_dimension>,
                     num_quadrature_points>
        dXdxi = isoparametric_element_.CalcJacobian(reference_positions);
    /* Record the quadrature point volume in reference configuration for each
     quadrature location. */
    for (int q = 0; q < num_quadrature_points; ++q) {
      // TODO(xuchenhan-tri): The volume scale calculation should be in
      // IsoparametricElement.
      /* The scale to transform quadrature weight in parent coordinates to
       reference coordinates. */
      T volume_scale;
      volume_scale = dXdxi[q].determinant();
      /* Degenerate element in the initial configuration is not allowed. */
      DRAKE_DEMAND(volume_scale > 0);
      reference_volume_[q] = volume_scale * quadrature_.get_weight(q);
      quadrature_point_mass_[q] = density_ * reference_volume_[q];
    }

    /* Record the inverse Jacobian at the reference configuration which is used
     in the calculation of deformation gradient. */
    dxidX_ = isoparametric_element_.CalcJacobianPseudoinverse(dXdxi);
    /* Record the gradient of the shape functions w.r.t. the reference
     positions, which is used in calculating the residual. */
    const auto dSdX = isoparametric_element_.CalcGradientInSpatialCoordinates(
        reference_positions);
    for (int q = 0; q < num_quadrature_points; ++q) {
      dSdX_transpose_[q] = dSdX[q].transpose();
    }
    T total_volume = 0.0;
    for (int q = 0; q < num_quadrature_points; ++q) {
      total_volume += reference_volume_[q];
    }
    this->set_mass(density_ * total_volume);

    mass_matrix_ = PrecomputeMassMatrix();
  }

  /* Calculates the elastic potential energy (in joules) stored in this element.
   */
  T CalcElasticEnergy(const std::array<T, num_quadrature_points>& Psi) const {
    T elastic_energy = 0;
    for (int q = 0; q < num_quadrature_points; ++q) {
      elastic_energy += reference_volume_[q] * Psi[q];
    }
    return elastic_energy;
  }

  /* Returns this element's the first moment of mass (mass * position) about
   World origin, expressed in World.
   @param[in] data The FEM data containing quadrature point positions. */
  Vector3<T> CalcMassTimesPositionForQuadraturePoints(const Data& data) const {
    Vector3<T> element_moment = Vector3<T>::Zero();
    for (int q = 0; q < num_quadrature_points; ++q) {
      element_moment +=
          quadrature_point_mass_[q] * data.quadrature_positions[q];
    }
    return element_moment;
  }

  /* Returns the linear momentum for the quadrature points in this element.
   @param[in] data The FEM data containing quadrature point velocities.
   @returns The linear momentum for this element. */
  Vector3<T> CalcTranslationalMomentumForQuadraturePoints(
      const Data& data) const {
    Vector3<T> element_translational_momentum = Vector3<T>::Zero();
    for (int q = 0; q < num_quadrature_points; ++q) {
      element_translational_momentum +=
          quadrature_point_mass_[q] * data.quadrature_velocities[q];
    }
    return element_translational_momentum;
  }

  /* Returns this element S's angular momentum about world origin, measured
   and expressed in World.
   @param[in] data The FEM data containing quadrature point positions and
                   velocities. */
  Vector3<T> CalcAngularMomentumAboutWorldOrigin(const Data& data) const {
    Vector3<T> H_SWWo_W = Vector3<T>::Zero();
    for (int q = 0; q < num_quadrature_points; ++q) {
      const Vector3<T>& p_WQ = data.quadrature_positions[q];
      const Vector3<T>& v_WQ = data.quadrature_velocities[q];

      H_SWWo_W += p_WQ.cross(v_WQ * quadrature_point_mass_[q]);
    }
    return H_SWWo_W;
  }

  /* Returns this element's rotational inertia about world origin, expressed
   in World.
   @param[in] data The FEM data containing quadrature point positions. */
  Matrix3<T> CalcRotationalInertiaAboutWorldOrigin(const Data& data) const {
    /* Rotational inertia for this element S, about Wo, expressed in W. */
    Matrix3<T> I_SWo_W = Matrix3<T>::Zero();
    for (int q = 0; q < num_quadrature_points; ++q) {
      const Vector3<T>& p_WQ = data.quadrature_positions[q];

      I_SWo_W +=
          quadrature_point_mass_[q] *
          (p_WQ.dot(p_WQ) * Matrix3<T>::Identity() - p_WQ * p_WQ.transpose());
    }
    return I_SWo_W;
  }

 private:
  /* Friend the base class so that FemElement::DoFoo() can reach its
   implementation. */
  friend FemElement<ElementType>;
  friend class VolumetricElementTest;

  /* Adds the negative elastic force on the nodes of this element into the
   given force vector. The negative elastic force is the derivative of the
   elastic energy (see CalcElasticEnergy()) with respect to the generalized
   positions of the nodes.
   @param[in] P               The first Piola-Kirchhoff stress evaluated at the
                              quadrature points of this element.
   @param[in, out] neg_force  The negative force vector to be added to.
   @pre neg_force != nullptr. */
  void AddNegativeElasticForce(
      const std::array<Matrix3<T>, num_quadrature_points>& P,
      EigenPtr<Vector<T, num_dofs>> neg_force) const {
    DRAKE_ASSERT(neg_force != nullptr);
    auto neg_force_matrix = Eigen::Map<Eigen::Matrix<T, 3, num_nodes>>(
        neg_force->data(), 3, num_nodes);
    for (int q = 0; q < num_quadrature_points; ++q) {
      /* Negative force is the gradient of energy.
       -f = ∫dΨ/dx = ∫dΨ/dF : dF/dx dX.
       Notice that Fᵢⱼ = xᵃᵢdSᵃ/dXⱼ, so dFᵢⱼ/dxᵇₖ = δᵃᵇδᵢₖdSᵃ/dXⱼ,
       and dΨ/dFᵢⱼ = Pᵢⱼ, so the integrand becomes
       PᵢⱼδᵃᵇδᵢₖdSᵃ/dXⱼ = PₖⱼdSᵇ/dXⱼ = P * dSdX.transpose() */
      neg_force_matrix += reference_volume_[q] * P[q] * dSdX_transpose_[q];
    }
  }

  /* Adds the negative damping force on the nodes of this element into the given
   `negative_damping_force`. The negative damping force is given by the product
   of the damping matrix with the velocity of the nodes.
   @param[in] damping_matrix  The damping matrix of this element.
   @param[in] element_v       The velocities of the vertex nodes of this
                              element.
   @param[in, out] neg_force  The negative force vector to be added to.
   @pre neg_force != nullptr. */
  void AddNegativeDampingForce(
      const Eigen::Matrix<T, num_dofs, num_dofs>& damping_matrix,
      const Vector<T, num_dofs>& element_v,
      EigenPtr<Vector<T, num_dofs>> neg_force) const {
    DRAKE_ASSERT(neg_force != nullptr);
    /* Note that the damping force fᵥ = -D * v, where D is the damping matrix.
     As we are accumulating the negative damping force here, the `+=` sign
     should be used. */
    *neg_force += damping_matrix * element_v;
  }

  /* The matrix calculated here is the same as the stiffness matrix
   calculated in [Bonet, 2016] equation (9.50b) without the external force
   component.
   Without the external force component, (9.50b) reads Kₐᵦ = Kₐᵦ,c + Kₐᵦ,σ.
   Kₐᵦ,c is given by ∫dSᵃ/dxₖ cᵢₖⱼₗ dSᵇ/dxₗ dx (9.35), and
   Kₐᵦ,σ is given by ∫dSᵃ/dxₖ σₖₗ dSᵇ/dxₗ dx (9.44c). Notice that we use S to
   denote shape functions whereas [Bonet, 2016] uses N.
   The negative force derivative we calculate here is given by ∫ dF/dxᵇ :
   dP/dF : dF/dxᵃ dX. The calculation uses a different conjugate pair, but is
   analytically equal to Kₐᵦ,c + Kₐᵦ,σ. See
   multibody/fem/doc/stiffness_matrix.tex for the derivation that
   shows the equivalence.

   Reference: [Bonet, 2016] Bonet, Javier, Antonio J.Gil, and
   Richard D. Wood. Nonlinear solid mechanics for finite element analysis:
   statics. Cambridge University Press, 2016. */

  /* TODO(xuchenhan-tri): Consider performing the calculation in current
   coordinates. A few trade-offs:
    1. The shape function derivatives needs to be recalculated every time.
    2. There will be two terms instead of one.
    3. The c matrix has symmetries that can be exploited and can be represented
   by a symmetric 6x6 matrix, whereas dP/dF is an asymmetric 9x9 matrix. The
   two stress-strain pairs need to be carefully profiled against each other as
   this operation might be (one of) the bottleneck(s). */

  /* Adds a scaled derivative of the elastic force on the nodes of this
   element into the given matrix.
   @param[in] dPdF    The PK1 stress derivative w.r.t. the deformation gradient
                      evaluated at the quadrature points of this element.
   @param[in] scale   The scaling factor applied to the derivative.
   @param[in, out] K  The scaled force derivative matrix to be added to.
   @pre K != nullptr. */
  void AddScaledElasticForceDerivative(
      const std::array<math::internal::FourthOrderTensor<T>,
                       num_quadrature_points>& dPdF,
      const T& scale, EigenPtr<Eigen::Matrix<T, num_dofs, num_dofs>> K) const {
    DRAKE_ASSERT(K != nullptr);
    // clang-format off
    /* Let e be the elastic energy, then the elastic force f is given by

     fᵃᵢ = -de/dxᵃᵢ = -∫dψ/dxᵃᵢ dX = -∫dψ/dF:dF/dxᵃᵢ dX.

     and the ab-th block of the stiffness matrix K is given by:
     Kᵃᵇᵢⱼ = dfᵃᵢ/dxᵇⱼ
           = d²e/dxᵃᵢdxᵇⱼ = ∫dF/dxᵇⱼ:d²ψ/dF²:dF/dxᵃᵢ + dψ/dF:d²F/dxᵃᵢdxᵇⱼ dX.

     The second term vanishes because Fₖₗ = xᵃₖdSᵃ/dXₗ is linear in x.
     We calculate the first term:
     dF/dxᵇⱼ : d²ψ/dF² : dF/dxᵃᵢ = dFₘₙ/dxᵃᵢ dPₘₙ/dFₖₗ dFₖₗ/dxᵇⱼ.  */
    // clang-format on
    // The ab-th 3-by-3 block of K.
    Matrix3<T> K_ab;
    for (int q = 0; q < num_quadrature_points; ++q) {
      /* Notice that Fₖₗ = xᵃₖdSᵃ/dXₗ, so dFₖₗ/dxᵇⱼ = δᵃᵇδₖⱼdSᵃ/dXₗ, and thus
       Kᵃᵇᵢⱼ = dFₘₙ/dxᵃᵢ dPₘₙ/dFₖₗ dFₖₗ/dxᵇⱼ =  dSᵃ/dXₙ dPᵢₙ/dFⱼₗ dSᵇ/dXₗ. */
      for (int a = 0; a < num_nodes; ++a) {
        for (int b = 0; b < num_nodes; ++b) {
          /* Note that the scale is negated here because the tensor contraction
           gives the second derivative of energy, which is the opposite of the
           force derivative. */
          dPdF[q].ContractWithVectors(
              dSdX_transpose_[q].col(a),
              dSdX_transpose_[q].col(b) * reference_volume_[q] * -scale, &K_ab);
          AccumulateMatrixBlock(K_ab, a, b, K);
        }
      }
    }
  }

  /* Calculates the tangent matrix for the element by combining the stiffness
   matrix, damping matrix, and the mass matrix according to the given `weights`.
   In particular, given a weight of (w₀, w₁, w₂), the tangent matrix is equal to
   w₀⋅K + w₁⋅D + w₂⋅M, where K, D, and M are stiffness, damping, and mass matrix
   respectively.
   @param[in]  weights           The weights to be used for the stiffness,
                                 damping, and mass matrices.
   @param[in]  stiffness_matrix  The stiffness matrix of this element.
   @param[out] tangent_matrix    The tangent matrix of this element. All entries
                                 are reset by this function. */
  void CalcTangentMatrix(
      const Vector3<T>& weights,
      const Eigen::Matrix<T, num_dofs, num_dofs>& stiffness_matrix,
      EigenPtr<Eigen::Matrix<T, num_dofs, num_dofs>> tangent_matrix) const {
    DRAKE_ASSERT(tangent_matrix != nullptr);
    const T stiffness_weight =
        weights(0) + weights(1) * this->damping_model().stiffness_coeff_beta();
    const T mass_weight =
        weights(2) + weights(1) * this->damping_model().mass_coeff_alpha();
    tangent_matrix->noalias() = stiffness_weight * stiffness_matrix;
    tangent_matrix->noalias() += mass_weight * mass_matrix_;
  }

  /* Calculates the force required to induce the acceleration `a` given the
   configuration `x` and velocities `v`, with `x`, `v`, and `a` stored in
   `data`. The required force equals is ID(a, x, v) = Ma-fₑ(x)-fᵥ(x, v), where M
   is the mass matrix, fₑ(x) is the elastic force, fᵥ(x, v) is the damping
   force. The residual is then given by ID(a, x, v) - fₑₓₜ. Notice that the
   result is "discrete" in space and "continuous" in time.
   @param[in] damping_matrix  The damping matrix of this element.
   @param[in] P               The first Piola-Kirchhoff stress evaluated at the
                              quadrature points of this element.
   @param[in] element_v       The velocities of the vertex nodes of this
                              element.
   @param[in] element_a       The accelerations of the vertex nodes of this
                              element.
   @param[out] result         The resulting inverse dynamics force. All values
                              in `result` will be overwritten.
   @pre result != nullptr */
  void CalcInverseDynamics(
      const Eigen::Matrix<T, num_dofs, num_dofs>& damping_matrix,
      const std::array<Matrix3<T>, num_quadrature_points>& P,
      const Vector<T, num_dofs>& element_v,
      const Vector<T, num_dofs>& element_a,
      EigenPtr<Vector<T, num_dofs>> result) const {
    /* residual = Ma-fₑ(x)-fᵥ(x, v), where M is the mass matrix, fₑ(x) is
     the elastic force, and fᵥ(x, v) is the damping force. */
    *result = mass_matrix_ * element_a;
    this->AddNegativeElasticForce(P, result);
    AddNegativeDampingForce(damping_matrix, element_v, result);
  }

  /* Implements FemElement::ComputeData(). */
  Data DoComputeData(const FemState<T>& state,
                     const Vector3<T>& weights) const {
    Data data;
    data.element_q = this->ExtractElementDofs(state.GetPositions());
    data.element_q0 =
        this->ExtractElementDofs(state.GetPreviousStepPositions());
    data.element_v = this->ExtractElementDofs(state.GetVelocities());
    data.element_a = this->ExtractElementDofs(state.GetAccelerations());
    const auto& element_q_reshaped =
        Eigen::Map<const Eigen::Matrix<T, 3, num_nodes>>(data.element_q.data(),
                                                         3, num_nodes);
    const auto& element_v_reshaped =
        Eigen::Map<const Eigen::Matrix<T, 3, num_nodes>>(data.element_v.data(),
                                                         3, num_nodes);
    data.quadrature_positions =
        isoparametric_element_.template InterpolateNodalValues<3>(
            element_q_reshaped);
    data.quadrature_velocities =
        isoparametric_element_.template InterpolateNodalValues<3>(
            element_v_reshaped);

    std::array<Matrix3<T>, num_quadrature_points> F =
        CalcDeformationGradient(data.element_q);
    std::array<Matrix3<T>, num_quadrature_points> F0 =
        CalcDeformationGradient(data.element_q0);

    for (int q = 0; q < num_quadrature_points; ++q) {
      data.deformation_gradient_data[q].UpdateData(F[q], F0[q]);
      this->constitutive_model().CalcElasticEnergyDensity(
          data.deformation_gradient_data[q], &(data.Psi[q]));
      this->constitutive_model().CalcFirstPiolaStress(
          data.deformation_gradient_data[q], &(data.P[q]));
      this->constitutive_model().CalcFilteredHessian(
          data.deformation_gradient_data[q], &(data.dPdF[q]));
    }
    data.stiffness_matrix.setZero();
    AddScaledElasticForceDerivative(data.dPdF, -1.0, &data.stiffness_matrix);
    data.damping_matrix =
        this->damping_model().stiffness_coeff_beta() * data.stiffness_matrix +
        this->damping_model().mass_coeff_alpha() * mass_matrix_;
    CalcInverseDynamics(data.damping_matrix, data.P, data.element_v,
                        data.element_a, &data.inverse_dynamics);
    CalcTangentMatrix(weights, data.stiffness_matrix, &data.tangent_matrix);
    return data;
  }

  void DoAddScaledExternalForces(const Data& data,
                                 const FemPlantData<T>& plant_data,
                                 const T& scale,
                                 EigenPtr<Vector<T, num_dofs>> result) const {
    const std::array<Vector<T, 3>, num_quadrature_points>&
        quadrature_positions = data.quadrature_positions;
    const std::array<Vector<T, num_nodes>, num_quadrature_points>& S =
        isoparametric_element_.GetShapeFunctions();
    for (int q = 0; q < num_quadrature_points; ++q) {
      const Matrix3<T>& deformation_gradient =
          data.deformation_gradient_data[q].deformation_gradient();
      Vector3<T> scaled_force = Vector3<T>::Zero();
      for (const multibody::ForceDensityFieldBase<T>* force_density :
           plant_data.force_density_fields) {
        DRAKE_ASSERT(force_density != nullptr);
        const T change_of_volume =
            force_density->density_type() ==
                    multibody::ForceDensityType::kPerReferenceVolume
                ? 1.0
                : deformation_gradient.determinant();
        scaled_force += scale *
                        force_density->EvaluateAt(plant_data.plant_context,
                                                  quadrature_positions[q]) *
                        reference_volume_[q] * change_of_volume;
      }
      for (int n = 0; n < num_nodes; ++n) {
        result->template segment<3>(3 * n) += scaled_force * S[q](n);
      }
    }
  }

  /* Calculates the deformation gradient at all quadrature points in this
   element.
   @param[in] element_q  The positions of the nodes of the element in a flat
   vector. */
  std::array<Matrix3<T>, num_quadrature_points> CalcDeformationGradient(
      const Eigen::Ref<const VectorX<T>>& element_q) const {
    std::array<Matrix3<T>, num_quadrature_points> F;
    const auto& element_q_reshaped =
        Eigen::Map<const Eigen::Matrix<T, 3, num_nodes>>(element_q.data(), 3,
                                                         num_nodes);
    const std::array<typename IsoparametricElementType::JacobianMatrix,
                     num_quadrature_points>
        dxdxi = isoparametric_element_.CalcJacobian(element_q_reshaped);
    for (int quad = 0; quad < num_quadrature_points; ++quad) {
      F[quad] = dxdxi[quad] * dxidX_[quad];
    }
    return F;
  }

  /* Helper function that adds a 3x3 matrix into the 3x3 block in a bigger
   matrix `matrix` with starting row index 3*node_a and starting column index
   3*node_b. Note that this function assumes the pointer `matrix` is not null.
   It also does not check the index it tries to write in `matrix` is valid and
   does not clear any stale data that might exist in `matrix`. */
  static void AccumulateMatrixBlock(
      const Eigen::Ref<const Matrix3<T>>& block, int node_a, int node_b,
      EigenPtr<Eigen::Matrix<T, num_dofs, num_dofs>> matrix) {
    matrix->template block<3, 3>(3 * node_a, 3 * node_b) += block;
  }

  Eigen::Matrix<T, num_dofs, num_dofs> PrecomputeMassMatrix() const {
    Eigen::Matrix<T, num_dofs, num_dofs> mass_matrix =
        Eigen::Matrix<T, num_dofs, num_dofs>::Zero();
    const std::array<Vector<T, num_nodes>, num_quadrature_points>& S =
        isoparametric_element_.GetShapeFunctions();
    /* S_mat is the matrix representation of S. */
    Eigen::Matrix<T, num_nodes, num_quadrature_points> S_mat;
    for (int q = 0; q < num_quadrature_points; ++q) {
      S_mat.col(q) = S[q];
    }
    /* weighted_S stores the shape function weighted by the reference
     volume of the quadrature point. */
    Eigen::Matrix<T, num_nodes, num_quadrature_points> weighted_S(S_mat);
    for (int q = 0; q < num_quadrature_points; ++q) {
      weighted_S.col(q) *= reference_volume_[q];
    }
    /* weighted_SST = weighted_S * Sᵀ. The ij-th entry approximates the integral
     ∫SᵢSⱼ dX */
    const Eigen::Matrix<T, num_nodes, num_nodes> weighted_SST =
        weighted_S * S_mat.transpose();
    constexpr int kDim = 3;
    for (int a = 0; a < num_nodes; ++a) {
      for (int b = 0; b < num_nodes; ++b) {
        mass_matrix.template block<kDim, kDim>(kDim * a, kDim * b) =
            Eigen::Matrix<T, kDim, kDim>::Identity() * weighted_SST(a, b) *
            density_;
      }
    }
    return mass_matrix;
  }

  // TODO(xuchenhan-tri): Consider bumping this up into FemElement when new
  //  FemElement types are added.
  /* The quadrature rule used for this element. */
  QuadratureType quadrature_{};
  /* The isoparametric element used for this element. */
  IsoparametricElementType isoparametric_element_{quadrature_.get_points()};
  /* The inverse element Jacobian evaluated at reference configuration at
   the quadrature points in this element. */
  std::array<Eigen::Matrix<T, natural_dimension, 3>, num_quadrature_points>
      dxidX_;
  /* The transpose of the derivatives of the shape functions with respect to the
   reference positions evaluated at the quadrature points in this element. */
  std::array<Eigen::Matrix<T, 3, num_nodes>, num_quadrature_points>
      dSdX_transpose_;
  // TODO(xuchenhan-tri): Consider exposing this through an accessor if it turns
  // out to be useful.
  /* The volume evaluated at reference configuration occupied by the
   quadrature points in this element. To integrate a function f over the
   reference domain, sum f(q)*reference_volume_[q] over all the quadrature
   points q in the element. */
  std::array<T, num_quadrature_points> reference_volume_;
  /* Mass of the quadrature points, computed as density_ * reference_volume_. */
  std::array<T, num_quadrature_points> quadrature_point_mass_;
  /* The uniform mass density of the element in the reference configuration with
   unit kg/m³. */
  T density_;
  /* Precomputed mass matrix. */
  Eigen::Matrix<T, num_dofs, num_dofs> mass_matrix_;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
