#include "drake/multibody/der/elastic_energy.h"

#include <vector>

#include "drake/math/cross_product.h"
#include "drake/multibody/der/der_indexes.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

using Eigen::Matrix3;
using Eigen::Vector3;

namespace {

template <typename T>
auto outer(const Eigen::Ref<const Vector3<T>>& vec1,
           const Eigen::Ref<const Vector3<T>>& vec2) {
  return vec1 * vec2.transpose();
}

template <typename T>
auto symm(const Eigen::Ref<const Matrix3<T>>& mat) {
  return 0.5 * (mat + mat.transpose());
}

template <typename T>
auto skew(const Eigen::Ref<const Vector3<T>>& vec) {
  return math::VectorToSkewSymmetric(vec);
}

template <typename T>
auto eye() {
  return Matrix3<T>::Identity();
}

/* Add mat = ∂²E/∂(xᵢ)(xⱼ) to the hessian.  */
template <typename T>
void AddTo(Block4x4SparseSymmetricMatrix<T>* hessian, DerNodeIndex i,
           DerNodeIndex j, const Eigen::Ref<const Matrix3<T>>& mat) {
  Eigen::Matrix4<T> filler = Eigen::Matrix4<T>::Zero();
  filler.template topLeftCorner<3, 3>() = mat;
  if (int{i} >= int{j})
    hessian->AddToBlock(i, j, filler);
  else
    hessian->AddToBlock(j, i, filler.transpose());
}

/* Add vec = ∂²E/∂(xᵢ)(γʲ) to the hessian.  */
template <typename T>
void AddTo(Block4x4SparseSymmetricMatrix<T>* hessian, DerNodeIndex i,
           DerEdgeIndex j, const Eigen::Ref<const Vector3<T>>& vec) {
  Eigen::Matrix4<T> filler = Eigen::Matrix4<T>::Zero();
  filler.template topRightCorner<3, 1>() = vec;
  if (int{i} == int{j}) {
    hessian->AddToBlock(i, j, filler + filler.transpose());
  } else {
    if (int{i} > int{j})
      hessian->AddToBlock(i, j, filler);
    else
      hessian->AddToBlock(j, i, filler.transpose());
  }
}

/* Add val = ∂²E/∂(γⁱ)(γʲ) to the hessian.  */
template <typename T>
void AddTo(Block4x4SparseSymmetricMatrix<T>* hessian, DerEdgeIndex i,
           DerEdgeIndex j, const T& val) {
  Eigen::Matrix4<T> filler = Eigen::Matrix4<T>::Zero();
  filler(3, 3) = val;
  if (int{i} >= int{j})
    hessian->AddToBlock(i, j, filler);
  else
    hessian->AddToBlock(j, i, filler.transpose());
}

/* Usage: ASSERT_NUM_COLS(matrix1, matrix2, ...,matrixN, number). Assert all
 matrices have the `number` of columns. */
template <typename First>
void ASSERT_NUM_COLS(const First&) {
  static_assert(std::is_integral_v<First>);  // No more matrices to check.
}
template <typename First, typename Second, typename... Other>
void ASSERT_NUM_COLS(const First& first, const Second& second,
                     const Other&... other) {
  static_assert(First::RowsAtCompileTime != Eigen::Dynamic);
  if constexpr (std::is_integral_v<Second>) {
    DRAKE_ASSERT(first.cols() == second);
  } else {
    DRAKE_ASSERT(first.cols() == second.cols());
  }
  ASSERT_NUM_COLS(second, other...);
}

}  // namespace

template <typename T>
T ComputeElasticEnergy(const DerStructuralProperty<T>& prop,
                       const DerUndeformedState<T>& undeformed,
                       const DerState<T>& state) {
  DRAKE_THROW_UNLESS(undeformed.has_closed_ends() == state.has_closed_ends());
  DRAKE_THROW_UNLESS(undeformed.num_nodes() == state.num_nodes());
  return ComputeStretchingEnergy(prop, undeformed, state) +
         ComputeTwistingEnergy(prop, undeformed, state) +
         ComputeBendingEnergy(prop, undeformed, state);
}

template <typename T>
void ComputeElasticEnergyJacobian(const DerStructuralProperty<T>& prop,
                                  const DerUndeformedState<T>& undeformed,
                                  const DerState<T>& state,
                                  EigenPtr<Eigen::VectorX<T>> jacobian) {
  DRAKE_THROW_UNLESS(undeformed.has_closed_ends() == state.has_closed_ends());
  DRAKE_THROW_UNLESS(undeformed.num_nodes() == state.num_nodes());
  DRAKE_THROW_UNLESS(jacobian != nullptr);
  DRAKE_THROW_UNLESS(jacobian->size() == state.num_dofs());
  jacobian->setZero();
  AddStretchingEnergyJacobian(prop, undeformed, state, jacobian);
  AddTwistingEnergyJacobian(prop, undeformed, state, jacobian);
  AddBendingEnergyJacobian(prop, undeformed, state, jacobian);
}

template <typename T>
void ComputeElasticEnergyHessian(const DerStructuralProperty<T>& prop,
                                 const DerUndeformedState<T>& undeformed,
                                 const DerState<T>& state,
                                 Block4x4SparseSymmetricMatrix<T>* hessian) {
  DRAKE_THROW_UNLESS(undeformed.has_closed_ends() == state.has_closed_ends());
  DRAKE_THROW_UNLESS(undeformed.num_nodes() == state.num_nodes());
  const int hessian_rows =
      state.has_closed_ends() ? state.num_dofs() : state.num_dofs() + 1;
  DRAKE_THROW_UNLESS(hessian != nullptr);
  DRAKE_THROW_UNLESS(hessian->rows() == hessian_rows &&
                     hessian->cols() == hessian_rows);
  hessian->SetZero();
  AddStretchingEnergyHessian(prop, undeformed, state, hessian);
  AddTwistingEnergyHessian(prop, undeformed, state, hessian);
  AddBendingEnergyHessian(prop, undeformed, state, hessian);
}

template <typename T>
Block4x4SparseSymmetricMatrix<T> MakeEnergyHessianMatrix(bool has_closed_ends,
                                                         int num_nodes) {
  std::vector<int> block_sizes(num_nodes, 4);

  std::vector<std::vector<int>> neighbors(num_nodes);
  if (!has_closed_ends) {
    /*
     ⎡ ██ ██ ██             ⎤
     ⎢ ██ ██ ██ ██          ⎥
     ⎢ ██ ██ ██ ██ ██       ⎥
     ⎢    ██ ██ ██ ██ ██    ⎥
     ⎢       ██ ██ ██ ██ ██ ⎥
     ⎢          ██ ██ ██ ██ ⎥
     ⎣             ██ ██ ██ ⎦
    */
    for (int i = 0; i < num_nodes; ++i) {
      neighbors[i].push_back(i);
      if (i + 1 < num_nodes) neighbors[i].push_back(i + 1);
      if (i + 2 < num_nodes) neighbors[i].push_back(i + 2);
    }
  } else {
    /*
     ⎡ ██ ██ ██       ██ ██ ⎤
     ⎢ ██ ██ ██ ██       ██ ⎥
     ⎢ ██ ██ ██ ██ ██       ⎥
     ⎢    ██ ██ ██ ██ ██    ⎥
     ⎢       ██ ██ ██ ██ ██ ⎥
     ⎢ ██       ██ ██ ██ ██ ⎥
     ⎣ ██ ██       ██ ██ ██ ⎦
    */
    for (int i = 0; i < num_nodes; ++i) {
      neighbors[i].push_back(i);
      if (i + 1 < num_nodes)
        neighbors[i].push_back(i + 1);
      else
        neighbors[(i + 1) % num_nodes].push_back(i);
      if (i + 2 < num_nodes)
        neighbors[i].push_back(i + 2);
      else
        neighbors[(i + 2) % num_nodes].push_back(i);
    }
  }

  contact_solvers::internal::BlockSparsityPattern block_sparsity_pattern(
      std::move(block_sizes), std::move(neighbors));
  return Block4x4SparseSymmetricMatrix<T>(std::move(block_sparsity_pattern));
}

template <typename T>
T ComputeStretchingEnergy(const DerStructuralProperty<T>& prop,
                          const DerUndeformedState<T>& undeformed,
                          const DerState<T>& state) {
  auto& l_bar = undeformed.get_edge_length();
  auto& l = state.get_edge_length();
  ASSERT_NUM_COLS(l_bar, l, state.num_edges());

  T energy = 0;  // Eₛ
  for (int i = 0; i < state.num_edges(); ++i) {
    T strain = l[i] / l_bar[i] - 1.0;
    energy += 0.5 * prop.EA() * strain * strain * l_bar[i];
  }
  return energy;
}

template <typename T>
void AddStretchingEnergyJacobian(const DerStructuralProperty<T>& prop,
                                 const DerUndeformedState<T>& undeformed,
                                 const DerState<T>& state,
                                 EigenPtr<Eigen::VectorX<T>> jacobian) {
  DRAKE_THROW_UNLESS(jacobian != nullptr);
  DRAKE_THROW_UNLESS(jacobian->size() == state.num_dofs());

  auto& l_bar = undeformed.get_edge_length();
  auto& l = state.get_edge_length();
  auto& tangent = state.get_tangent();
  ASSERT_NUM_COLS(l_bar, l, tangent, state.num_edges());

  for (int i = 0; i < state.num_edges(); ++i) {
    // ∂Eₛ/∂eⁱ
    Vector3<T> grad_E_ei = (l[i] / l_bar[i] - 1.0) * tangent.col(i) * prop.EA();

    const int node_i = 4 * i;
    const int node_ip1 = 4 * ((i + 1) % state.num_nodes());
    jacobian->template segment<3>(node_i) -= grad_E_ei;
    jacobian->template segment<3>(node_ip1) += grad_E_ei;
  }
}

template <typename T>
void AddStretchingEnergyHessian(const DerStructuralProperty<T>& prop,
                                const DerUndeformedState<T>& undeformed,
                                const DerState<T>& state,
                                Block4x4SparseSymmetricMatrix<T>* hessian) {
  DRAKE_THROW_UNLESS(hessian != nullptr);

  auto& l_bar = undeformed.get_edge_length();
  auto& l = state.get_edge_length();
  auto& tangent = state.get_tangent();
  ASSERT_NUM_COLS(l_bar, l, tangent, state.num_edges());

  for (int i = 0; i < state.num_edges(); ++i) {
    // ∂²Eₛ/∂(eⁱ)²
    Matrix3<T> grad2_E_ei_ei =
        ((1 / l_bar[i] - 1 / l[i]) * eye<T>() +
         1 / l[i] * outer<T>(tangent.col(i), tangent.col(i))) *
        prop.EA();

    const DerNodeIndex node_i(i);
    const DerNodeIndex node_ip1((i + 1) % state.num_nodes());
    AddTo<T>(hessian, node_i, node_i, grad2_E_ei_ei);
    AddTo<T>(hessian, node_ip1, node_ip1, grad2_E_ei_ei);
    AddTo<T>(hessian, node_i, node_ip1, -grad2_E_ei_ei);
  }
}

template <typename T>
T ComputeTwistingEnergy(const DerStructuralProperty<T>& prop,
                        const DerUndeformedState<T>& undeformed,
                        const DerState<T>& state) {
  auto& V_bar = undeformed.get_voronoi_length();
  auto& twist_bar = undeformed.get_twist();
  auto& twist = state.get_twist();
  ASSERT_NUM_COLS(V_bar, twist_bar, twist, state.num_internal_nodes());

  T energy = 0;  // Eₜ
  for (int i = 0; i < state.num_internal_nodes(); ++i) {
    T delta = twist[i] - twist_bar[i];
    energy += 0.5 * prop.GJ() * delta * delta / V_bar[i];
  }
  return energy;
}

template <typename T>
void AddTwistingEnergyJacobian(const DerStructuralProperty<T>& prop,
                               const DerUndeformedState<T>& undeformed,
                               const DerState<T>& state,
                               EigenPtr<Eigen::VectorX<T>> jacobian) {
  DRAKE_THROW_UNLESS(jacobian != nullptr);
  DRAKE_THROW_UNLESS(jacobian->size() == state.num_dofs());

  auto& V_bar = undeformed.get_voronoi_length();
  auto& twist_bar = undeformed.get_twist();
  auto& twist = state.get_twist();
  auto& curvature = state.get_discrete_integrated_curvature();
  ASSERT_NUM_COLS(V_bar, twist_bar, twist, curvature,
                  state.num_internal_nodes());

  auto& l = state.get_edge_length();
  ASSERT_NUM_COLS(l, state.num_edges());

  for (int i = 0; i < state.num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % state.num_edges();

    // ∂Eₜ/∂τᵢ
    T grad_E_twisti = (twist[i] - twist_bar[i]) * prop.GJ() / V_bar[i];

    const int edge_i = 4 * i + 3;
    const int edge_ip1 = 4 * ip1 + 3;
    jacobian->coeffRef(edge_i) -= grad_E_twisti;
    jacobian->coeffRef(edge_ip1) += grad_E_twisti;

    // ∂τᵢ/∂eⁱ
    Vector3<T> grad_twisti_ei = 0.5 / l[i] * curvature.col(i);
    // ∂τᵢ/∂eⁱ
    Vector3<T> grad_twisti_eip1 = 0.5 / l[ip1] * curvature.col(i);

    // ∂Eₜ/∂eⁱ
    Vector3<T> grad_E_ei = grad_E_twisti * grad_twisti_ei;
    // ∂Eₜ/∂eⁱ⁺¹
    Vector3<T> grad_E_eip1 = grad_E_twisti * grad_twisti_eip1;

    const int node_i = 4 * i;
    const int node_ip1 = 4 * ((i + 1) % state.num_nodes());
    const int node_ip2 = 4 * ((i + 2) % state.num_nodes());
    jacobian->template segment<3>(node_i) -= grad_E_ei;
    jacobian->template segment<3>(node_ip1) += grad_E_ei;
    jacobian->template segment<3>(node_ip1) -= grad_E_eip1;
    jacobian->template segment<3>(node_ip2) += grad_E_eip1;
  }
}

template <typename T>
void AddTwistingEnergyHessian(const DerStructuralProperty<T>& prop,
                              const DerUndeformedState<T>& undeformed,
                              const DerState<T>& state,
                              Block4x4SparseSymmetricMatrix<T>* hessian) {
  DRAKE_THROW_UNLESS(hessian != nullptr);

  auto& V_bar = undeformed.get_voronoi_length();
  auto& twist_bar = undeformed.get_twist();
  auto& twist = state.get_twist();
  auto& curvature = state.get_discrete_integrated_curvature();
  ASSERT_NUM_COLS(V_bar, twist_bar, twist, curvature,
                  state.num_internal_nodes());

  auto& l = state.get_edge_length();
  auto& t = state.get_tangent();
  ASSERT_NUM_COLS(l, t, state.num_edges());

  for (int i = 0; i < state.num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % state.num_edges();

    // ∂Eₜ/∂τᵢ
    T grad_E_twisti = (twist[i] - twist_bar[i]) * prop.GJ() / V_bar[i];
    // ∂²Eₜ/∂(τᵢ)²
    T grad2_E_twisti_twisti = prop.GJ() / V_bar[i];

    const DerEdgeIndex edge_i(i);
    const DerEdgeIndex edge_ip1((i + 1) % state.num_edges());
    AddTo<T>(hessian, edge_i, edge_i, grad2_E_twisti_twisti);
    AddTo<T>(hessian, edge_ip1, edge_ip1, grad2_E_twisti_twisti);
    AddTo<T>(hessian, edge_i, edge_ip1, -grad2_E_twisti_twisti);

    // ∂τᵢ/∂eⁱ
    Vector3<T> grad_twisti_ei = 0.5 / l[i] * curvature.col(i);
    // ∂τᵢ/∂eⁱ⁺¹
    Vector3<T> grad_twisti_eip1 = 0.5 / l[ip1] * curvature.col(i);

    // t̃
    Vector3<T> tilde_t =
        (t.col(i) + t.col(ip1)) / (1 + t.col(ip1).dot(t.col(i)));

    // ∂²τᵢ/∂(eⁱ)²
    Matrix3<T> grad2_twisti_ei_ei =
        -0.5 / (l[i] * l[i]) *
        symm<T>(outer<T>(curvature.col(i), t.col(i) + tilde_t));
    // ∂²τᵢ/∂(eⁱ⁺¹)²
    Matrix3<T> grad2_twisti_eip1_eip1 =
        -0.5 / (l[ip1] * l[ip1]) *
        symm<T>(outer<T>(curvature.col(i), t.col(ip1) + tilde_t));
    // ∂²τᵢ/∂(eⁱ)(eⁱ⁺¹)
    Matrix3<T> grad2_twisti_ei_eip1 =
        (skew<T>(t.col(i)) -
         outer<T>(curvature.col(i), 0.5 * (t.col(i) + t.col(ip1)))) /
        (l[i] * l[ip1] * (1 + t.col(ip1).dot(t.col(i))));

    // ∂²Eₜ/∂(eⁱ)²
    Matrix3<T> grad2_E_ei_ei =
        grad2_E_twisti_twisti * outer<T>(grad_twisti_ei, grad_twisti_ei) +
        grad_E_twisti * grad2_twisti_ei_ei;
    // ∂²Eₜ/∂(eⁱ⁺¹)²
    Matrix3<T> grad2_E_eip1_eip1 =
        grad2_E_twisti_twisti * outer<T>(grad_twisti_eip1, grad_twisti_eip1) +
        grad_E_twisti * grad2_twisti_eip1_eip1;
    // ∂²Eₜ/∂(eⁱ)(eⁱ⁺¹)
    Matrix3<T> grad2_E_ei_eip1 =
        grad2_E_twisti_twisti * outer<T>(grad_twisti_ei, grad_twisti_eip1) +
        grad_E_twisti * grad2_twisti_ei_eip1;

    const DerNodeIndex node_i(i);
    const DerNodeIndex node_ip1((i + 1) % state.num_nodes());
    const DerNodeIndex node_ip2((i + 2) % state.num_nodes());
    AddTo<T>(hessian, node_i, node_i, grad2_E_ei_ei);
    AddTo<T>(hessian, node_ip1, node_ip1, grad2_E_ei_ei);
    AddTo<T>(hessian, node_i, node_ip1, -grad2_E_ei_ei);

    AddTo<T>(hessian, node_ip1, node_ip1, grad2_E_eip1_eip1);
    AddTo<T>(hessian, node_ip2, node_ip2, grad2_E_eip1_eip1);
    AddTo<T>(hessian, node_ip1, node_ip2, -grad2_E_eip1_eip1);

    AddTo<T>(hessian, node_i, node_ip1, grad2_E_ei_eip1);
    AddTo<T>(hessian, node_ip1, node_ip2, grad2_E_ei_eip1);
    AddTo<T>(hessian, node_i, node_ip2, -grad2_E_ei_eip1);
    AddTo<T>(hessian, node_ip1, node_ip1,
             -grad2_E_ei_eip1 - grad2_E_ei_eip1.transpose());

    // ∂²Eₜ/∂(eⁱ)(γⁱ)
    Vector3<T> grad2_E_ei_gammai = -grad2_E_twisti_twisti * grad_twisti_ei;
    // ∂²Eₜ/∂(eⁱ)(γⁱ⁺¹)
    Vector3<T> grad2_E_ei_gammaip1 = grad2_E_twisti_twisti * grad_twisti_ei;
    // ∂²Eₜ/∂(eⁱ⁺¹)(γⁱ)
    Vector3<T> grad2_E_eip1_gammai = -grad2_E_twisti_twisti * grad_twisti_eip1;
    // ∂²Eₜ/∂(eⁱ⁺¹)(γⁱ⁺¹)
    Vector3<T> grad2_E_eip1_gammaip1 = grad2_E_twisti_twisti * grad_twisti_eip1;

    AddTo<T>(hessian, node_i, edge_i, -grad2_E_ei_gammai);
    AddTo<T>(hessian, node_ip1, edge_i, grad2_E_ei_gammai);

    AddTo<T>(hessian, node_i, edge_ip1, -grad2_E_ei_gammaip1);
    AddTo<T>(hessian, node_ip1, edge_ip1, grad2_E_ei_gammaip1);

    AddTo<T>(hessian, node_ip1, edge_i, -grad2_E_eip1_gammai);
    AddTo<T>(hessian, node_ip2, edge_i, grad2_E_eip1_gammai);

    AddTo<T>(hessian, node_ip1, edge_ip1, -grad2_E_eip1_gammaip1);
    AddTo<T>(hessian, node_ip2, edge_ip1, grad2_E_eip1_gammaip1);
  }
}

template <typename T>
T ComputeBendingEnergy(const DerStructuralProperty<T>& prop,
                       const DerUndeformedState<T>& undeformed,
                       const DerState<T>& state) {
  auto& V_bar = undeformed.get_voronoi_length();
  auto& kappa1_bar = undeformed.get_curvature_kappa1();
  auto& kappa2_bar = undeformed.get_curvature_kappa2();
  auto& kappa1 = state.get_curvature_kappa1();
  auto& kappa2 = state.get_curvature_kappa2();
  ASSERT_NUM_COLS(V_bar, kappa1_bar, kappa2_bar, kappa1, kappa2,
                  state.num_internal_nodes());

  T energy = 0;  // Eₙ
  for (int i = 0; i < state.num_internal_nodes(); ++i) {
    T delta = kappa1[i] - kappa1_bar[i];
    energy += 0.5 * prop.EI1() * delta * delta / V_bar[i];
    delta = kappa2[i] - kappa2_bar[i];
    energy += 0.5 * prop.EI2() * delta * delta / V_bar[i];
  }
  return energy;
}

template <typename T>
void AddBendingEnergyJacobian(const DerStructuralProperty<T>& prop,
                              const DerUndeformedState<T>& undeformed,
                              const DerState<T>& state,
                              EigenPtr<Eigen::VectorX<T>> jacobian) {
  DRAKE_THROW_UNLESS(jacobian != nullptr);
  DRAKE_THROW_UNLESS(jacobian->size() == state.num_dofs());

  auto& V_bar = undeformed.get_voronoi_length();
  auto& kappa1_bar = undeformed.get_curvature_kappa1();
  auto& kappa2_bar = undeformed.get_curvature_kappa2();
  auto& kappa1 = state.get_curvature_kappa1();
  auto& kappa2 = state.get_curvature_kappa2();
  auto& curvature = state.get_discrete_integrated_curvature();
  ASSERT_NUM_COLS(V_bar, kappa1_bar, kappa2_bar, kappa1, kappa2, curvature,
                  state.num_internal_nodes());

  auto& l = state.get_edge_length();
  auto& t = state.get_tangent();
  auto& m1 = state.get_material_frame_m1();
  auto& m2 = state.get_material_frame_m2();
  ASSERT_NUM_COLS(l, t, m1, m2, state.num_edges());

  for (int i = 0; i < state.num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % state.num_edges();

    // ∂Eₙ/∂κ₁ᵢ
    T grad_E_kappa1i = (kappa1[i] - kappa1_bar[i]) * prop.EI1() / V_bar[i];
    // ∂Eₙ/∂κ₂ᵢ
    T grad_E_kappa2i = (kappa2[i] - kappa2_bar[i]) * prop.EI2() / V_bar[i];

    // 1/χ
    T chi_inv = 1.0 / (1.0 + t.col(ip1).dot(t.col(i)));
    // t̃
    Vector3<T> tilde_t = (t.col(i) + t.col(ip1)) * chi_inv;
    // m̃₁
    Vector3<T> tilde_m1 = (m1.col(i) + m1.col(ip1)) * chi_inv;
    // m̃₂
    Vector3<T> tilde_m2 = (m2.col(i) + m2.col(ip1)) * chi_inv;

    // ∂κ₁ᵢ/∂eⁱ
    Vector3<T> grad_kappa1i_ei =
        (-kappa1[i] * tilde_t + t.col(ip1).cross(tilde_m2)) / l[i];
    // ∂κ₂ᵢ/∂eⁱ
    Vector3<T> grad_kappa2i_ei =
        (-kappa2[i] * tilde_t - t.col(ip1).cross(tilde_m1)) / l[i];
    // ∂κ₁ᵢ/∂eⁱ⁺¹
    Vector3<T> grad_kappa1i_eip1 =
        (-kappa1[i] * tilde_t - t.col(i).cross(tilde_m2)) / l[ip1];
    // ∂κ₂ᵢ/∂eⁱ⁺¹
    Vector3<T> grad_kappa2i_eip1 =
        (-kappa2[i] * tilde_t + t.col(i).cross(tilde_m1)) / l[ip1];

    // ∂Eₙ/∂eⁱ
    Vector3<T> grad_E_ei =
        grad_E_kappa1i * grad_kappa1i_ei + grad_E_kappa2i * grad_kappa2i_ei;
    // ∂Eₙ/∂eⁱ⁺¹
    Vector3<T> grad_E_eip1 =
        grad_E_kappa1i * grad_kappa1i_eip1 + grad_E_kappa2i * grad_kappa2i_eip1;

    const int node_i = 4 * i;
    const int node_ip1 = 4 * ((i + 1) % state.num_nodes());
    const int node_ip2 = 4 * ((i + 2) % state.num_nodes());
    jacobian->template segment<3>(node_i) -= grad_E_ei;
    jacobian->template segment<3>(node_ip1) += grad_E_ei;
    jacobian->template segment<3>(node_ip1) -= grad_E_eip1;
    jacobian->template segment<3>(node_ip2) += grad_E_eip1;

    // ∂κ₁ᵢ/∂γⁱ
    T grad_kappa1i_gammai = -0.5 * m1.col(i).dot(curvature.col(i));
    // ∂κ₂ᵢ/∂γⁱ
    T grad_kappa2i_gammai = -0.5 * m2.col(i).dot(curvature.col(i));
    // ∂κ₁ᵢ/∂γⁱ⁺¹
    T grad_kappa1i_gammaip1 = -0.5 * m1.col(ip1).dot(curvature.col(i));
    // ∂κ₂ᵢ/∂γⁱ⁺¹
    T grad_kappa2i_gammaip1 = -0.5 * m2.col(ip1).dot(curvature.col(i));

    // ∂Eₙ/∂γⁱ
    T grad_E_gammai = grad_E_kappa1i * grad_kappa1i_gammai +
                      grad_E_kappa2i * grad_kappa2i_gammai;
    // ∂Eₙ/∂γⁱ⁺¹
    T grad_E_gammaip1 = grad_E_kappa1i * grad_kappa1i_gammaip1 +
                        grad_E_kappa2i * grad_kappa2i_gammaip1;

    const int edge_i = 4 * i + 3;
    const int edge_ip1 = 4 * ((i + 1) % state.num_edges()) + 3;
    jacobian->coeffRef(edge_i) += grad_E_gammai;
    jacobian->coeffRef(edge_ip1) += grad_E_gammaip1;
  }
}

template <typename T>
void AddBendingEnergyHessian(const DerStructuralProperty<T>& prop,
                             const DerUndeformedState<T>& undeformed,
                             const DerState<T>& state,
                             Block4x4SparseSymmetricMatrix<T>* hessian) {
  DRAKE_THROW_UNLESS(hessian != nullptr);

  auto& V_bar = undeformed.get_voronoi_length();
  auto& kappa1_bar = undeformed.get_curvature_kappa1();
  auto& kappa2_bar = undeformed.get_curvature_kappa2();
  auto& kappa1 = state.get_curvature_kappa1();
  auto& kappa2 = state.get_curvature_kappa2();
  auto& curvature = state.get_discrete_integrated_curvature();
  ASSERT_NUM_COLS(V_bar, kappa1_bar, kappa2_bar, kappa1, kappa2, curvature,
                  state.num_internal_nodes());

  auto& l = state.get_edge_length();
  auto& t = state.get_tangent();
  auto& m1 = state.get_material_frame_m1();
  auto& m2 = state.get_material_frame_m2();
  ASSERT_NUM_COLS(l, t, m1, m2, state.num_edges());

  for (int i = 0; i < state.num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % state.num_edges();

    // ∂Eₙ/∂κ₁ᵢ
    T grad_E_kappa1i = (kappa1[i] - kappa1_bar[i]) * prop.EI1() / V_bar[i];
    // ∂Eₙ/∂κ₂ᵢ
    T grad_E_kappa2i = (kappa2[i] - kappa2_bar[i]) * prop.EI2() / V_bar[i];
    // ∂²Eₙ/∂(κ₁ᵢ)²
    T grad2_E_kappa1i_kappa1i = prop.EI1() / V_bar[i];
    // ∂²Eₙ/∂(κ₂ᵢ)²
    T grad2_E_kappa2i_kappa2i = prop.EI2() / V_bar[i];

    // 1/χ
    T chi_inv = 1.0 / (1.0 + t.col(ip1).dot(t.col(i)));
    // t̃
    Vector3<T> tilde_t = (t.col(i) + t.col(ip1)) * chi_inv;
    // m̃₁
    Vector3<T> tilde_m1 = (m1.col(i) + m1.col(ip1)) * chi_inv;
    // m̃₂
    Vector3<T> tilde_m2 = (m2.col(i) + m2.col(ip1)) * chi_inv;

    // ∂χ/∂eⁱ
    Vector3<T> grad_chi_ei =
        (eye<T>() - outer<T>(t.col(i), t.col(i))) * t.col(ip1) / l[i];
    // ∂χ/∂eⁱ⁺¹
    Vector3<T> grad_chi_eip1 =
        (eye<T>() - outer<T>(t.col(ip1), t.col(ip1))) * t.col(i) / l[ip1];

    // ∂t̃/∂eⁱ
    Matrix3<T> grad_tildet_ei =
        ((eye<T>() - outer<T>(t.col(i), t.col(i))) -
         outer<T>(tilde_t,
                  (eye<T>() - outer<T>(t.col(i), t.col(i))) * t.col(ip1))) *
        chi_inv / l[i];
    // ∂t̃/∂eⁱ⁺¹
    Matrix3<T> grad_tildet_eip1 =
        ((eye<T>() - outer<T>(t.col(ip1), t.col(ip1))) -
         outer<T>(tilde_t,
                  (eye<T>() - outer<T>(t.col(ip1), t.col(ip1))) * t.col(i))) *
        chi_inv / l[ip1];

    // ∂κ₁ᵢ/∂eⁱ
    Vector3<T> grad_kappa1i_ei =
        (-kappa1[i] * tilde_t + t.col(ip1).cross(tilde_m2)) / l[i];
    // ∂κ₂ᵢ/∂eⁱ
    Vector3<T> grad_kappa2i_ei =
        (-kappa2[i] * tilde_t - t.col(ip1).cross(tilde_m1)) / l[i];
    // ∂κ₁ᵢ/∂eⁱ⁺¹
    Vector3<T> grad_kappa1i_eip1 =
        (-kappa1[i] * tilde_t - t.col(i).cross(tilde_m2)) / l[ip1];
    // ∂κ₂ᵢ/∂eⁱ⁺¹
    Vector3<T> grad_kappa2i_eip1 =
        (-kappa2[i] * tilde_t + t.col(i).cross(tilde_m1)) / l[ip1];

    // ∂²κ₁ᵢ/∂(eⁱ)²
    Matrix3<T> grad2_kappa1i_ei_ei =
        -1 / (l[i] * l[i]) *
            outer<T>(-kappa1[i] * tilde_t + t.col(ip1).cross(tilde_m2),
                     t.col(i)) +
        -1 / l[i] *
            (outer<T>(tilde_t, grad_kappa1i_ei) + kappa1[i] * grad_tildet_ei +
             chi_inv * outer<T>(t.col(ip1).cross(tilde_m2), grad_chi_ei));
    // ∂²κ₂ᵢ/∂(eⁱ)²
    Matrix3<T> grad2_kappa2i_ei_ei =
        -1 / (l[i] * l[i]) *
            outer<T>(-kappa2[i] * tilde_t - t.col(ip1).cross(tilde_m1),
                     t.col(i)) +
        -1 / l[i] *
            (outer<T>(tilde_t, grad_kappa2i_ei) + kappa2[i] * grad_tildet_ei -
             chi_inv * outer<T>(t.col(ip1).cross(tilde_m1), grad_chi_ei));

    // ∂²κ₁ᵢ/∂(eⁱ⁺¹)²
    Matrix3<T> grad2_kappa1i_eip1_eip1 =
        -1 / (l[ip1] * l[ip1]) *
            outer<T>(-kappa1[i] * tilde_t - t.col(i).cross(tilde_m2),
                     t.col(ip1)) +
        -1 / l[ip1] *
            (outer<T>(tilde_t, grad_kappa1i_eip1) +
             kappa1[i] * grad_tildet_eip1 -
             chi_inv * outer<T>(t.col(i).cross(tilde_m2), grad_chi_eip1));
    // ∂²κ₂ᵢ/∂(eⁱ⁺¹)²
    Matrix3<T> grad2_kappa2i_eip1_eip1 =
        -1 / (l[ip1] * l[ip1]) *
            outer<T>(-kappa2[i] * tilde_t + t.col(i).cross(tilde_m1),
                     t.col(ip1)) +
        -1 / l[ip1] *
            (outer<T>(tilde_t, grad_kappa2i_eip1) +
             kappa2[i] * grad_tildet_eip1 +
             chi_inv * outer<T>(t.col(i).cross(tilde_m1), grad_chi_eip1));

    // ∂²κ₁ᵢ/∂(eⁱ)(eⁱ⁺¹)
    Matrix3<T> grad2_kappa1i_ei_eip1 =
        -1 / l[i] *
        (outer<T>(tilde_t, grad_kappa1i_eip1) + kappa1[i] * grad_tildet_eip1 +
         chi_inv * outer<T>(t.col(ip1).cross(tilde_m2), grad_chi_eip1) +
         1 / l[ip1] *
             (outer<T>(t.col(ip1).cross(tilde_m2), t.col(ip1)) +
              skew<T>(tilde_m2)));
    // ∂²κ₂ᵢ/∂(eⁱ)(eⁱ⁺¹)
    Matrix3<T> grad2_kappa2i_ei_eip1 =
        -1 / l[i] *
        (outer<T>(tilde_t, grad_kappa2i_eip1) + kappa2[i] * grad_tildet_eip1 -
         chi_inv * outer<T>(t.col(ip1).cross(tilde_m1), grad_chi_eip1) -
         1 / l[ip1] *
             (outer<T>(t.col(ip1).cross(tilde_m1), t.col(ip1)) +
              skew<T>(tilde_m1)));

    // ∂²Eₙ/∂(eⁱ)²
    Matrix3<T> grad2_E_ei_ei =
        grad2_E_kappa1i_kappa1i * outer<T>(grad_kappa1i_ei, grad_kappa1i_ei) +
        grad_E_kappa1i * grad2_kappa1i_ei_ei +
        grad2_E_kappa2i_kappa2i * outer<T>(grad_kappa2i_ei, grad_kappa2i_ei) +
        grad_E_kappa2i * grad2_kappa2i_ei_ei;
    // ∂²Eₙ/∂(eⁱ⁺¹)²
    Matrix3<T> grad2_E_eip1_eip1 =
        grad2_E_kappa1i_kappa1i *
            outer<T>(grad_kappa1i_eip1, grad_kappa1i_eip1) +
        grad_E_kappa1i * grad2_kappa1i_eip1_eip1 +
        grad2_E_kappa2i_kappa2i *
            outer<T>(grad_kappa2i_eip1, grad_kappa2i_eip1) +
        grad_E_kappa2i * grad2_kappa2i_eip1_eip1;
    // ∂²Eₙ/∂(eⁱ)(eⁱ⁺¹)
    Matrix3<T> grad2_E_ei_eip1 =
        grad2_E_kappa1i_kappa1i * outer<T>(grad_kappa1i_ei, grad_kappa1i_eip1) +
        grad_E_kappa1i * grad2_kappa1i_ei_eip1 +
        grad2_E_kappa2i_kappa2i * outer<T>(grad_kappa2i_ei, grad_kappa2i_eip1) +
        grad_E_kappa2i * grad2_kappa2i_ei_eip1;

    const DerNodeIndex node_i(i);
    const DerNodeIndex node_ip1((i + 1) % state.num_nodes());
    const DerNodeIndex node_ip2((i + 2) % state.num_nodes());
    AddTo<T>(hessian, node_i, node_i, grad2_E_ei_ei);
    AddTo<T>(hessian, node_ip1, node_ip1, grad2_E_ei_ei);
    AddTo<T>(hessian, node_i, node_ip1, -grad2_E_ei_ei);

    AddTo<T>(hessian, node_ip1, node_ip1, grad2_E_eip1_eip1);
    AddTo<T>(hessian, node_ip2, node_ip2, grad2_E_eip1_eip1);
    AddTo<T>(hessian, node_ip1, node_ip2, -grad2_E_eip1_eip1);

    AddTo<T>(hessian, node_i, node_ip1, grad2_E_ei_eip1);
    AddTo<T>(hessian, node_ip1, node_ip2, grad2_E_ei_eip1);
    AddTo<T>(hessian, node_i, node_ip2, -grad2_E_ei_eip1);
    AddTo<T>(hessian, node_ip1, node_ip1,
             -grad2_E_ei_eip1 - grad2_E_ei_eip1.transpose());

    // ∂κ₁ᵢ/∂γⁱ
    T grad_kappa1i_gammai = -0.5 * m1.col(i).dot(curvature.col(i));
    // ∂κ₂ᵢ/∂γⁱ
    T grad_kappa2i_gammai = -0.5 * m2.col(i).dot(curvature.col(i));
    // ∂κ₁ᵢ/∂γⁱ⁺¹
    T grad_kappa1i_gammaip1 = -0.5 * m1.col(ip1).dot(curvature.col(i));
    // ∂κ₂ᵢ/∂γⁱ⁺¹
    T grad_kappa2i_gammaip1 = -0.5 * m2.col(ip1).dot(curvature.col(i));

    // ∂²κ₁ᵢ/∂(γⁱ)²
    T grad2_kappa1i_gammai_gammai = -0.5 * m2.col(i).dot(curvature.col(i));
    // ∂²κ₂ᵢ/∂(γⁱ)²
    T grad2_kappa2i_gammai_gammai = +0.5 * m1.col(i).dot(curvature.col(i));
    // ∂²κ₁ᵢ/∂(γⁱ⁺¹)²
    T grad2_kappa1i_gammaip1_gammaip1 =
        -0.5 * m2.col(ip1).dot(curvature.col(i));
    // ∂²κ₂ᵢ/∂(γⁱ⁺¹)²
    T grad2_kappa2i_gammaip1_gammaip1 =
        +0.5 * m1.col(ip1).dot(curvature.col(i));

    // ∂²Eₙ/∂(γⁱ)²
    T grad2_E_gammai_gammai =
        grad2_E_kappa1i_kappa1i * grad_kappa1i_gammai * grad_kappa1i_gammai +
        grad_E_kappa1i * grad2_kappa1i_gammai_gammai +
        grad2_E_kappa2i_kappa2i * grad_kappa2i_gammai * grad_kappa2i_gammai +
        grad_E_kappa2i * grad2_kappa2i_gammai_gammai;
    // ∂²Eₙ/∂(γⁱ⁺¹)²
    T grad2_E_gammaip1_gammaip1 =
        grad2_E_kappa1i_kappa1i * grad_kappa1i_gammaip1 *
            grad_kappa1i_gammaip1 +
        grad_E_kappa1i * grad2_kappa1i_gammaip1_gammaip1 +
        grad2_E_kappa2i_kappa2i * grad_kappa2i_gammaip1 *
            grad_kappa2i_gammaip1 +
        grad_E_kappa2i * grad2_kappa2i_gammaip1_gammaip1;
    // ∂²Eₙ/∂(γⁱ)(γⁱ⁺¹)
    T grad2_E_gammai_gammaip1 =
        grad2_E_kappa1i_kappa1i * grad_kappa1i_gammai * grad_kappa1i_gammaip1 +
        grad2_E_kappa2i_kappa2i * grad_kappa2i_gammai * grad_kappa2i_gammaip1;

    const DerEdgeIndex edge_i(i);
    const DerEdgeIndex edge_ip1((i + 1) % state.num_edges());
    AddTo<T>(hessian, edge_i, edge_i, grad2_E_gammai_gammai);
    AddTo<T>(hessian, edge_ip1, edge_ip1, grad2_E_gammaip1_gammaip1);
    AddTo<T>(hessian, edge_i, edge_ip1, grad2_E_gammai_gammaip1);

    // ∂²κ₁ᵢ/∂(eⁱ)(γⁱ)
    Vector3<T> grad2_kappa1i_ei_gammai =
        (-grad_kappa1i_gammai * tilde_t +
         chi_inv * t.col(ip1).cross(-m1.col(i))) /
        l[i];
    // ∂²κ₂ᵢ/∂(eⁱ)(γⁱ)
    Vector3<T> grad2_kappa2i_ei_gammai =
        (-grad_kappa2i_gammai * tilde_t -
         chi_inv * t.col(ip1).cross(m2.col(i))) /
        l[i];
    // ∂²κ₁ᵢ/∂(eⁱ)(γⁱ⁺¹)
    Vector3<T> grad2_kappa1i_ei_gammaip1 =
        (-grad_kappa1i_gammaip1 * tilde_t +
         chi_inv * t.col(ip1).cross(-m1.col(ip1))) /
        l[i];
    // ∂²κ₂ᵢ/∂(eⁱ)(γⁱ⁺¹)
    Vector3<T> grad2_kappa2i_ei_gammaip1 =
        (-grad_kappa2i_gammaip1 * tilde_t -
         chi_inv * t.col(ip1).cross(m2.col(ip1))) /
        l[i];
    // ∂²κ₁ᵢ/∂(eⁱ⁺¹)(γⁱ)
    Vector3<T> grad2_kappa1i_eip1_gammai =
        (-grad_kappa1i_gammai * tilde_t -
         chi_inv * t.col(i).cross(-m1.col(i))) /
        l[ip1];
    // ∂²κ₂ᵢ/∂(eⁱ⁺¹)(γⁱ)
    Vector3<T> grad2_kappa2i_eip1_gammai =
        (-grad_kappa2i_gammai * tilde_t +  //
         chi_inv * t.col(i).cross(m2.col(i))) /
        l[ip1];
    // ∂²κ₁ᵢ/∂(eⁱ⁺¹)(γⁱ⁺¹)
    Vector3<T> grad2_kappa1i_eip1_gammaip1 =
        (-grad_kappa1i_gammaip1 * tilde_t -
         chi_inv * t.col(i).cross(-m1.col(ip1))) /
        l[ip1];
    // ∂²κ₂ᵢ/∂(eⁱ⁺¹)(γⁱ⁺¹)
    Vector3<T> grad2_kappa2i_eip1_gammaip1 =
        (-grad_kappa2i_gammaip1 * tilde_t +
         chi_inv * t.col(i).cross(m2.col(ip1))) /
        l[ip1];

    // ∂²Eₙ/∂(eⁱ)(γⁱ)
    Vector3<T> grad2_E_ei_gammai =
        grad2_E_kappa1i_kappa1i * grad_kappa1i_ei * grad_kappa1i_gammai +
        grad_E_kappa1i * grad2_kappa1i_ei_gammai +
        grad2_E_kappa2i_kappa2i * grad_kappa2i_ei * grad_kappa2i_gammai +
        grad_E_kappa2i * grad2_kappa2i_ei_gammai;
    // ∂²Eₙ/∂(eⁱ)(γⁱ⁺¹)
    Vector3<T> grad2_E_ei_gammaip1 =
        grad2_E_kappa1i_kappa1i * grad_kappa1i_ei * grad_kappa1i_gammaip1 +
        grad_E_kappa1i * grad2_kappa1i_ei_gammaip1 +
        grad2_E_kappa2i_kappa2i * grad_kappa2i_ei * grad_kappa2i_gammaip1 +
        grad_E_kappa2i * grad2_kappa2i_ei_gammaip1;
    // ∂²Eₙ/∂(eⁱ⁺¹)(γⁱ)
    Vector3<T> grad2_E_eip1_gammai =
        grad2_E_kappa1i_kappa1i * grad_kappa1i_eip1 * grad_kappa1i_gammai +
        grad_E_kappa1i * grad2_kappa1i_eip1_gammai +
        grad2_E_kappa2i_kappa2i * grad_kappa2i_eip1 * grad_kappa2i_gammai +
        grad_E_kappa2i * grad2_kappa2i_eip1_gammai;
    // ∂²Eₙ/∂(eⁱ⁺¹)(γⁱ⁺¹)
    Vector3<T> grad2_E_eip1_gammaip1 =
        grad2_E_kappa1i_kappa1i * grad_kappa1i_eip1 * grad_kappa1i_gammaip1 +
        grad_E_kappa1i * grad2_kappa1i_eip1_gammaip1 +
        grad2_E_kappa2i_kappa2i * grad_kappa2i_eip1 * grad_kappa2i_gammaip1 +
        grad_E_kappa2i * grad2_kappa2i_eip1_gammaip1;

    AddTo<T>(hessian, node_i, edge_i, -grad2_E_ei_gammai);
    AddTo<T>(hessian, node_ip1, edge_i, grad2_E_ei_gammai);

    AddTo<T>(hessian, node_i, edge_ip1, -grad2_E_ei_gammaip1);
    AddTo<T>(hessian, node_ip1, edge_ip1, grad2_E_ei_gammaip1);

    AddTo<T>(hessian, node_ip1, edge_i, -grad2_E_eip1_gammai);
    AddTo<T>(hessian, node_ip2, edge_i, grad2_E_eip1_gammai);

    AddTo<T>(hessian, node_ip1, edge_ip1, -grad2_E_eip1_gammaip1);
    AddTo<T>(hessian, node_ip2, edge_ip1, grad2_E_eip1_gammaip1);
  }
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (&ComputeElasticEnergy<T>, &ComputeElasticEnergyJacobian<T>,
     &ComputeElasticEnergyHessian<T>, &MakeEnergyHessianMatrix<T>));

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
