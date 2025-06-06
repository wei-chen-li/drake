#include "drake/multibody/der/der_model.h"

#include <exception>
#include <tuple>

#include "drake/common/pointer_cast.h"
#include "drake/math/unit_vector.h"
#include "drake/multibody/der/elastic_energy.h"
#include "drake/multibody/der/energy_hessian_matrix_utility.h"

namespace drake {
namespace multibody {
namespace der {

// Import from "energy_hessian_matrix_utility.h".
using internal::operator*;
using internal::AddScaledMatrix;

template <typename T>
std::tuple<DerNodeIndex, DerEdgeIndex, DerNodeIndex>
DerModel<T>::Builder::AddFirstEdge(
    const Eigen::Ref<const Eigen::Vector3<T>>& x_0, const T& gamma_0,
    const Eigen::Ref<const Eigen::Vector3<T>>& x_1,
    const std::optional<Eigen::Vector3<T>>& d1_0) {
  if (is_built_) throw std::logic_error("The DER model is already build.");
  if (is_first_edge_added_)
    throw std::logic_error("AddFirstEdge() can only be called once.");
  if (d1_0) {
    Eigen::Vector3<T> t_0 =
        math::internal::NormalizeOrThrow<T>(x_1 - x_0, __func__);
    math::internal::ThrowIfNotOrthonormal<T>(*d1_0, t_0, __func__);
  }
  node_positions_.push_back(x_0);
  edge_angles_.push_back(gamma_0);
  node_positions_.push_back(x_1);
  d1_0_ = d1_0;
  is_first_edge_added_ = true;
  DRAKE_DEMAND(node_positions_.size() == 2);
  DRAKE_DEMAND(edge_angles_.size() == 1);
  return {DerNodeIndex(0), DerEdgeIndex(0), DerNodeIndex(1)};
}

template <typename T>
std::tuple<DerEdgeIndex, DerNodeIndex> DerModel<T>::Builder::AddEdge(
    const T& gamma_i, const Eigen::Ref<const Eigen::Vector3<T>>& x_ip1) {
  if (is_built_) throw std::logic_error("The DER model is already build.");
  if (!is_first_edge_added_)
    throw std::logic_error("AddFirstEdge() not called yet.");
  if (is_configuration_finalized_)
    throw std::logic_error("The rod configuration is already finalized.");

  edge_angles_.push_back(gamma_i);
  node_positions_.push_back(x_ip1);
  if ((node_positions_.back() - node_positions_.front()).norm() < 1e-14) {
    node_positions_.pop_back();
    is_configuration_finalized_ = true;
    has_closed_ends_ = true;
    return {DerEdgeIndex(edge_angles_.size() - 1), DerNodeIndex(0)};
  }
  return {DerEdgeIndex(edge_angles_.size() - 1),
          DerNodeIndex(node_positions_.size() - 1)};
}

template <typename T>
void DerModel<T>::Builder::SetZeroUndeformedCurvatureAndTwist() {
  if (is_built_) throw std::logic_error("The DER model is already build.");
  if (der_undeformed_state_)
    throw std::logic_error("The undeformed state is already set.");

  is_configuration_finalized_ = true;
  const int num_nodes = node_positions_.size();
  const int num_edges = has_closed_ends_ ? num_nodes : num_nodes - 1;
  std::vector<T> edge_length(num_edges);
  for (int i = 0; i < num_edges; ++i) {
    const int ip1 = (i + 1) % num_nodes;
    edge_length[i] = (node_positions_[ip1] - node_positions_[i]).norm();
  }
  der_undeformed_state_ =
      internal::DerUndeformedState<T>::ZeroCurvatureAndTwist(
          has_closed_ends_, std::move(edge_length));
}

template <typename T>
void DerModel<T>::Builder::SetUndeformedStateToInitialState() {
  if (is_built_) throw std::logic_error("The DER model is already build.");
  if (der_undeformed_state_)
    throw std::logic_error("The undeformed state is already set.");

  is_configuration_finalized_ = true;
  internal::DerStateSystem<T> der_state_system(
      has_closed_ends_, node_positions_, edge_angles_, d1_0_);
  internal::DerState<T> der_state(&der_state_system);

  der_undeformed_state_ =
      internal::DerUndeformedState<T>::FromCurrentDerState(der_state);
}

template <typename T>
void DerModel<T>::Builder::SetMaterialProperties(const T& youngs_modulus,
                                                 const T& shear_modulus,
                                                 const T& mass_density) {
  if (is_built_) throw std::logic_error("The DER model is already build.");
  if (material_property_)
    throw std::logic_error("The material properties are already set.");

  material_property_ =
      std::make_tuple(youngs_modulus, shear_modulus, mass_density);
}

template <typename T>
void DerModel<T>::Builder::SetCircularCrossSection(const T& radius) {
  if (is_built_) throw std::logic_error("The DER model is already build.");
  if (cross_section_)
    throw std::logic_error("The cross section is already set.");

  cross_section_ = std::make_pair(kCircular, std::vector<T>{radius});
}

template <typename T>
void DerModel<T>::Builder::SetRectangularCrossSection(const T& width,
                                                      const T& height) {
  if (is_built_) throw std::logic_error("The DER model is already build.");
  if (cross_section_)
    throw std::logic_error("The cross section is already set.");

  cross_section_ = std::make_pair(kRectangular, std::vector<T>{width, height});
}

template <typename T>
void DerModel<T>::Builder::SetEllipticalCrossSection(const T& a, const T& b) {
  if (is_built_) throw std::logic_error("The DER model is already build.");
  if (cross_section_)
    throw std::logic_error("The cross section is already set.");

  cross_section_ = std::make_pair(kElliptical, std::vector<T>{a, b});
}

template <typename T>
void DerModel<T>::Builder::SetDampingCoefficients(
    const T& mass_coeff_alpha, const T& stiffness_coeff_beta) {
  if (is_built_) throw std::logic_error("The DER model is already build.");
  if (damping_model_)
    throw std::logic_error("The damping coefficients are already set.");

  damping_model_ =
      internal::DampingModel<T>(mass_coeff_alpha, stiffness_coeff_beta);
}

template <typename T>
std::unique_ptr<DerModel<T>> DerModel<T>::Builder::Build() {
  using internal::DerStructuralProperty;

  if (is_built_) throw std::logic_error("The DER model is already build.");
  if (!is_configuration_finalized_)
    throw std::logic_error("The rod configuration is not finalized yet.");
  if (!der_undeformed_state_)
    throw std::logic_error("The undeformed state have not been set.");
  if (!material_property_)
    throw std::logic_error("The material properties have not been set.");
  if (!cross_section_)
    throw std::logic_error("The cross section have not been set.");
  if (!damping_model_)
    throw std::logic_error("The damping coefficients have not been set.");

  auto der_state_system = std::make_unique<internal::DerStateSystem<T>>(
      has_closed_ends_, std::move(node_positions_), std::move(edge_angles_),
      std::move(d1_0_));

  const auto& [E, G, rho] = *material_property_;
  const std::vector<T>& par = cross_section_->second;
  std::optional<DerStructuralProperty<T>> der_structural_property;
  if (cross_section_->first == kCircular) {
    der_structural_property =
        DerStructuralProperty<T>::FromCircularCrossSection(par[0], E, G, rho);
  } else if (cross_section_->first == kRectangular) {
    der_structural_property =
        DerStructuralProperty<T>::FromRectangularCrossSection(par[0], par[1], E,
                                                              G, rho);
  } else if (cross_section_->first == kElliptical) {
    der_structural_property =
        DerStructuralProperty<T>::FromEllipticalCrossSection(par[0], par[1], E,
                                                             G, rho);
  }
  DRAKE_DEMAND(der_structural_property.has_value());

  is_built_ = true;
  return std::unique_ptr<DerModel<T>>(new DerModel<T>(
      std::move(der_state_system), std::move(*der_structural_property),
      std::move(*der_undeformed_state_), std::move(*damping_model_),
      std::move(boundary_condition_)));
}

template <typename T>
DerModel<T>::DerModel(
    std::unique_ptr<const internal::DerStateSystem<T>> der_state_system,
    internal::DerStructuralProperty<T> der_structural_property,
    internal::DerUndeformedState<T> der_undeformed_state,
    internal::DampingModel<T> damping_model,
    internal::DirichletBoundaryCondition<T> boundary_condition)
    : der_state_system_(std::move(der_state_system)),
      der_structural_property_(std::move(der_structural_property)),
      der_undeformed_state_(std::move(der_undeformed_state)),
      damping_model_(std::move(damping_model)),
      boundary_condition_(std::move(boundary_condition)) {
  DRAKE_THROW_UNLESS(der_state_system_ != nullptr);
  DRAKE_THROW_UNLESS(der_state_system_->has_closed_ends() ==
                     der_undeformed_state_.has_closed_ends());
  DRAKE_THROW_UNLESS(der_state_system_->num_nodes() ==
                     der_undeformed_state_.num_nodes());
}

template <typename T>
void DerModel<T>::FixPosition(std::variant<DerNodeIndex, DerEdgeIndex> index) {
  if (std::holds_alternative<DerNodeIndex>(index)) {
    DerNodeIndex node_index = std::get<DerNodeIndex>(index);
    DRAKE_THROW_UNLESS(0 <= node_index && node_index < num_nodes());
    boundary_condition_.AddBoundaryCondition(
        node_index,
        internal::NodeState<T>{
            .x = der_state_system_->initial_node_positions()[node_index],
            .x_dot = Eigen::Vector3<T>::Zero(),
            .x_ddot = Eigen::Vector3<T>::Zero()});
  } else {
    DerEdgeIndex edge_index = std::get<DerEdgeIndex>(index);
    DRAKE_THROW_UNLESS(0 <= edge_index && edge_index < num_edges());
    boundary_condition_.AddBoundaryCondition(
        edge_index,
        internal::EdgeState<T>{
            .gamma = der_state_system_->initial_edge_angles()[edge_index],
            .gamma_dot = 0.0,
            .gamma_ddot = 0.0});
  }
}

template <typename T>
bool DerModel<T>::IsPositionFixed(
    std::variant<DerNodeIndex, DerEdgeIndex> index) const {
  if (std::holds_alternative<DerNodeIndex>(index)) {
    DerNodeIndex node_index = std::get<DerNodeIndex>(index);
    DRAKE_THROW_UNLESS(0 <= node_index && node_index < num_nodes());
    const internal::NodeState<T>* node_state =
        boundary_condition_.GetBoundaryCondition(node_index);
    if (node_state == nullptr) return false;
    return ExtractDoubleOrThrow(node_state->x_dot).isZero() &&
           ExtractDoubleOrThrow(node_state->x_ddot).isZero();
  } else {
    DerEdgeIndex edge_index = std::get<DerEdgeIndex>(index);
    DRAKE_THROW_UNLESS(0 <= edge_index && edge_index < num_edges());
    const internal::EdgeState<T>* edge_state =
        boundary_condition_.GetBoundaryCondition(edge_index);
    if (edge_state == nullptr) return false;
    return ExtractDoubleOrThrow(edge_state->gamma_dot) == 0.0 &&
           ExtractDoubleOrThrow(edge_state->gamma_ddot) == 0.0;
  }
}

template <typename T>
std::unique_ptr<internal::DerState<T>> DerModel<T>::CreateDerState() const {
  return std::make_unique<internal::DerState<T>>(der_state_system_.get());
}

template <typename T>
struct DerModel<T>::Scratch {
  const DerModel<T>* der_model;
  const Eigen::DiagonalMatrix<T, Eigen::Dynamic> M;
  Eigen::VectorX<T> residual;
  Eigen::VectorX<T> dEdq;
  internal::Block4x4SparseSymmetricMatrix<T> tangent_matrix;
  std::tuple<const void*, int64_t, internal::Block4x4SparseSymmetricMatrix<T>>
      state_to_d2Edq2;

  Scratch(const DerModel<T>& model)
      : der_model(&model),
        M(internal::ComputeMassMatrix(model.der_structural_property_,
                                      model.der_undeformed_state_)),
        residual(model.num_dofs()),
        dEdq(model.num_dofs()),
        tangent_matrix(  //
            internal::MakeEnergyHessianMatrix<T>(
                model.has_closed_ends(), model.num_nodes(), model.num_edges())),
        state_to_d2Edq2({0, 0,
                         internal::MakeEnergyHessianMatrix<T>(
                             model.has_closed_ends(), model.num_nodes(),
                             model.num_edges())}) {}
};

template <typename T>
void DerModel<T>::ScratchDeleter::operator()(
    typename DerModel<T>::Scratch* ptr) const {
  // Operator delete needs to see the full definition of DerModel<T>::Scratch.
  delete ptr;
}

template <typename T>
std::unique_ptr<typename DerModel<T>::Scratch,
                typename DerModel<T>::ScratchDeleter>
DerModel<T>::MakeScratch() const {
  return std::unique_ptr<Scratch, ScratchDeleter>(new Scratch(*this));
}

template <typename T>
static decltype(auto) operator*(
    const Eigen::DiagonalMatrix<T, Eigen::Dynamic>& mat,
    const Eigen::VectorX<T>& vec) {
  /* The default Eigen implementation of a diagonal matrix times a vector
   allocates heap, so we redefine it here. */
  return mat.diagonal().cwiseProduct(vec);
}

template <typename T>
const Eigen::VectorX<T>& DerModel<T>::ComputeResidual(
    const internal::DerState<T>& state,
    const internal::ExternalForceField<T>& external_force_field,
    typename DerModel<T>::Scratch* scratch) const {
  this->ValidateDerState(state);
  this->ValidateScratch(scratch);

  const Eigen::VectorX<T>& qdot = state.get_velocity();
  const Eigen::VectorX<T>& qddot = state.get_acceleration();
  const Eigen::DiagonalMatrix<T, Eigen::Dynamic>& M = scratch->M;
  const T& alpha = damping_model_.mass_coeff_alpha();
  const T& beta = damping_model_.stiffness_coeff_beta();

  Eigen::VectorX<T>& dEdq = scratch->dEdq;
  internal::ComputeElasticEnergyJacobian<T>(
      der_structural_property_, der_undeformed_state_, state, &dEdq);

  /* R(q, q̇, q̈) = M q̈ - Fᵢₙₜ(q, q̇) - Fₑₓₜ
                = M q̈ + ∂E(q)/∂q + (αM + βK) q̇ - Fₑₓₜ,
   where K = ∂²E(q)/∂q². */
  Eigen::VectorX<T>& residual = scratch->residual;
  residual.setZero();
  residual += M * qddot;
  residual += dEdq;
  if (alpha != 0) residual += alpha * (M * qdot);
  if (beta != 0) {
    auto& cache = scratch->state_to_d2Edq2;
    std::get<0>(cache) = static_cast<const void*>(&state);
    std::get<1>(cache) = state.serial_number();
    internal::Block4x4SparseSymmetricMatrix<T>& d2Edq2 = std::get<2>(cache);
    internal::ComputeElasticEnergyHessian<T>(
        der_structural_property_, der_undeformed_state_, state, &d2Edq2);

    const internal::Block4x4SparseSymmetricMatrix<T>& K = d2Edq2;
    residual += beta * (K * qdot);
  }
  residual -= external_force_field(der_structural_property_,
                                   der_undeformed_state_, state);

  /* Set boundary condition DoFs corresponding residual entries to zero. */
  boundary_condition_.ApplyHomogeneousBoundaryCondition(&residual);

  return residual;
}

template <typename T>
const internal::Block4x4SparseSymmetricMatrix<T>&
DerModel<T>::ComputeTangentMatrix(const internal::DerState<T>& state,
                                  const std::array<T, 3>& weights,
                                  Scratch* scratch) const {
  this->ValidateDerState(state);
  this->ValidateScratch(scratch);

  /* If the state is the same one passed to ComputeResidual(), no need to
   recalculate `d2Edq2`. */
  auto& cache = scratch->state_to_d2Edq2;
  internal::Block4x4SparseSymmetricMatrix<T>& d2Edq2 = std::get<2>(cache);
  if (!(&state == std::get<0>(cache) &&
        state.serial_number() == std::get<1>(cache))) {
    internal::ComputeElasticEnergyHessian<T>(
        der_structural_property_, der_undeformed_state_, state, &d2Edq2);
  }

  const internal::Block4x4SparseSymmetricMatrix<T>& K = d2Edq2;
  const Eigen::DiagonalMatrix<T, Eigen::Dynamic>& M = scratch->M;
  const T& alpha = damping_model_.mass_coeff_alpha();
  const T& beta = damping_model_.stiffness_coeff_beta();

  /* R(q, q̇, q̈) = M q̈ + ∂E(q)/∂q + (αM + βK) q̇ - Fₑₓₜ.
   ∂R/∂q = ∂²E(q)/∂q² = K, ∂R/∂q̇ = αM + βK, ∂R/∂q̈ = M. */
  internal::Block4x4SparseSymmetricMatrix<T>& tangent_matrix =
      scratch->tangent_matrix;
  tangent_matrix.SetZero();
  AddScaledMatrix(&tangent_matrix, K, weights[0] + beta * weights[1]);
  AddScaledMatrix(&tangent_matrix, M, alpha * weights[1] + weights[2]);

  /* Set boundary condition DoFs corresponding tangent matrix rows and columns
   to zero, and corresponding diagonal entries to one. */
  boundary_condition_.ApplyBoundaryConditionToTangentMatrix(&tangent_matrix);

  /* If tangent_matrix is larger than num_dofs() by one, set the last diagonal
   entry to one. */
  DRAKE_DEMAND(tangent_matrix.rows() ==
               (has_closed_ends() ? num_dofs() : num_dofs() + 1));
  if (tangent_matrix.rows() == num_dofs() + 1) {
    const int i = tangent_matrix.block_rows() - 1;
    Eigen::Matrix4<T> block = tangent_matrix.block(i, i);
    DRAKE_ASSERT(ExtractDoubleOrThrow(block).template rightCols<1>().isZero());
    DRAKE_ASSERT(ExtractDoubleOrThrow(block).template bottomRows<1>().isZero());
    block(3, 3) = 1.0;
    tangent_matrix.SetBlock(i, i, block);
  }

  return tangent_matrix;
}

template <typename T>
void DerModel<T>::ApplyBoundaryCondition(internal::DerState<T>* state) const {
  DRAKE_THROW_UNLESS(state != nullptr);
  this->ValidateDerState(*state);
  boundary_condition_.ApplyBoundaryConditionToState(state);
}

template <typename T>
std::unique_ptr<DerModel<T>> DerModel<T>::Clone() const {
  return std::unique_ptr<DerModel<T>>(
      new DerModel<T>(dynamic_pointer_cast<internal::DerStateSystem<T>>(
                          der_state_system_->Clone()),
                      der_structural_property_, der_undeformed_state_,
                      damping_model_, boundary_condition_));
}

template <typename T>
template <typename U>
std::unique_ptr<DerModel<U>> DerModel<T>::ToScalarType() const {
  static_assert(!std::is_same_v<T, U>);
  return std::unique_ptr<DerModel<U>>(
      new DerModel<U>(dynamic_pointer_cast<internal::DerStateSystem<U>>(
                          der_state_system_->template ToScalarType<U>()),
                      der_structural_property_.template ToScalarType<U>(),
                      der_undeformed_state_.template ToScalarType<U>(),
                      damping_model_.template ToScalarType<U>(),
                      boundary_condition_.template ToScalarType<U>()));
}

using symbolic::Expression;
template std::unique_ptr<DerModel<AutoDiffXd>>
DerModel<double>::ToScalarType<AutoDiffXd>() const;
template std::unique_ptr<DerModel<Expression>>
DerModel<double>::ToScalarType<Expression>() const;
template std::unique_ptr<DerModel<double>>
DerModel<AutoDiffXd>::ToScalarType<double>() const;
template std::unique_ptr<DerModel<Expression>>
DerModel<AutoDiffXd>::ToScalarType<Expression>() const;
template std::unique_ptr<DerModel<double>>
DerModel<Expression>::ToScalarType<double>() const;
template std::unique_ptr<DerModel<AutoDiffXd>>
DerModel<Expression>::ToScalarType<AutoDiffXd>() const;

template <typename T>
void DerModel<T>::ValidateDerState(const internal::DerState<T>& state) const {
  if (!state.is_created_from_system(*der_state_system_))
    throw std::invalid_argument(
        "The passed DerState is not created from this DerModel.");
}

template <typename T>
void DerModel<T>::ValidateScratch(
    const typename DerModel<T>::Scratch* scratch) const {
  DRAKE_THROW_UNLESS(scratch != nullptr);
  if (scratch->der_model != this)
    throw std::invalid_argument(
        "The passed DerModel::Scratch is not created from this DerModel.");
}

}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::DerModel);
