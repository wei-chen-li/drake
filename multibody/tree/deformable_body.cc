#include "drake/multibody/tree/deformable_body.h"

#include "drake/common/overloaded.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/axis_angle.h"
#include "drake/math/frame_transport.h"
#include "drake/multibody/fem/corotated_model.h"
#include "drake/multibody/fem/dirichlet_boundary_condition.h"
#include "drake/multibody/fem/linear_constitutive_model.h"
#include "drake/multibody/fem/linear_corotated_model.h"
#include "drake/multibody/fem/linear_simplex_element.h"
#include "drake/multibody/fem/neohookean_model.h"
#include "drake/multibody/fem/simplex_gaussian_quadrature.h"
#include "drake/multibody/fem/velocity_newmark_scheme.h"
#include "drake/multibody/fem/volumetric_model.h"
#include "drake/multibody/tree/force_density_field.h"
#include "drake/multibody/tree/multibody_tree.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"

namespace drake {
namespace multibody {

using fem::MaterialModel;
using geometry::GeometryId;
using geometry::GeometryInstance;
using geometry::VolumeMesh;

template <typename T>
ScopedName DeformableBody<T>::scoped_name() const {
  return ScopedName(
      this->get_parent_tree().GetModelInstanceName(this->model_instance()),
      name_);
}

template <typename T>
void DeformableBody<T>::SetWallBoundaryCondition(const Vector3<T>& p_WQ,
                                                 const Vector3<T>& n_W) const {
  DRAKE_THROW_UNLESS(n_W.norm() > 1e-10);
  const Vector3<T> nhat_W = n_W.normalized();

  constexpr int kDim = 3;
  auto is_inside_wall = [&p_WQ, &nhat_W](const Vector3<T>& p_WV) {
    const T distance_to_wall = (p_WV - p_WQ).dot(nhat_W);
    return distance_to_wall < 0;
  };

  if (fem_model_) {
    const int num_nodes = fem_model_->num_nodes();
    fem::internal::DirichletBoundaryCondition<T> bc;
    for (int n = 0; n < num_nodes; ++n) {
      const int dof_index = kDim * n;
      const auto p_WV = reference_positions_.template segment<kDim>(dof_index);
      if (is_inside_wall(p_WV)) {
        /* Set this node to be subject to zero Dirichlet BC. */
        bc.AddBoundaryCondition(fem::FemNodeIndex(n),
                                {p_WV, Vector3<T>::Zero(), Vector3<T>::Zero()});
      }
    }
    fem_model_->SetDirichletBoundaryCondition(bc);
  } else if (der_model_) {
    const int num_nodes = der_model_->num_nodes();
    /* Fix the nodes that are inside the wall. */
    std::vector<int> fixed_node_indexes;
    for (int i = 0; i < num_nodes; ++i) {
      const auto p_WV = reference_positions_.template segment<kDim>(kDim * i);
      if (is_inside_wall(p_WV)) {
        der_model_->FixPosition(der::DerNodeIndex(i));
        fixed_node_indexes.push_back(i);
      }
    }
    /* If node i and node i+1 are fixed, also fix edge i. */
    for (int k = 0; k < ssize(fixed_node_indexes); ++k) {
      const int i = fixed_node_indexes[k];
      const int ip1 =
          der_model_->has_closed_ends() ? (i + 1) % num_nodes : (i + 1);
      if (ip1 == fixed_node_indexes[(k + 1) % ssize(fixed_node_indexes)]) {
        der_model_->FixPosition(der::DerEdgeIndex(i));
      }
    }
  } else {
    DRAKE_UNREACHABLE();
  }
}

template <typename T>
MultibodyConstraintId DeformableBody<T>::AddFixedConstraint(
    const RigidBody<T>& body_B, const math::RigidTransform<double>& X_BA,
    const geometry::Shape& shape_G, const math::RigidTransform<double>& X_BG) {
  if (&this->get_parent_tree().get_body(body_B.index()) != &body_B) {
    throw std::logic_error(fmt::format(
        "AddFixedConstraint(): The rigid body with name {} is not registered "
        "with the MultibodyPlant owning the deformable model.",
        body_B.name()));
  }
  /* X_WG is the pose of this body's reference mesh in the world frame. In the
   scope of this function, we call that the A frame and G is used to denote
   the rigid body's geometry, so we rename X_WG_ to X_WA here to avoid
   confusion. */
  const math::RigidTransformd& X_WA = X_WG_;
  const MultibodyConstraintId constraint_id =
      MultibodyConstraintId::get_new_id();
  geometry::SceneGraph<double> scene_graph;
  geometry::SourceId source_id = scene_graph.RegisterSource("deformable_model");
  /* Register the geometry in deformable reference geometry A frame. */
  const math::RigidTransform<double> X_AG = X_BA.InvertAndCompose(X_BG);
  auto instance =
      std::make_unique<GeometryInstance>(X_AG, shape_G.Clone(), "rigid shape");
  GeometryId geometry_id =
      scene_graph.RegisterAnchoredGeometry(source_id, std::move(instance));
  scene_graph.AssignRole(source_id, geometry_id,
                         geometry::ProximityProperties());
  auto context = scene_graph.CreateDefaultContext();
  auto query =
      scene_graph.get_query_output_port().Eval<geometry::QueryObject<double>>(
          *context);
  if (fem_model_) {
    /* Create an empty spec first. We will add to it. */
    internal::DeformableRigidFixedConstraintSpec spec{
        id_, body_B.index(), {}, {}, constraint_id};
    const VectorX<double>& p_WPi = reference_positions_;
    for (int vertex_index = 0; vertex_index < fem_model_->num_nodes();
         ++vertex_index) {
      /* The vertex position in the deformable body's geometry frame. */
      const Vector3<double>& p_APi =
          X_WA.inverse() * p_WPi.template segment<3>(vertex_index * 3);
      /* Note that `shape` is also registered in the A frame in the throw-away
       scene graph. */
      const std::vector<geometry::SignedDistanceToPoint<double>>
          signed_distances = query.ComputeSignedDistanceToPoint(p_APi);
      DRAKE_DEMAND(ssize(signed_distances) == 1);
      const double signed_distance = signed_distances[0].distance;
      if (signed_distance <= 0.0) {
        spec.vertices.push_back(vertex_index);
        /* Qi is conincident with Pi. */
        spec.p_BQs.emplace_back(X_BA * p_APi);
      }
    }
    // TODO(xuchenhan-tri): consider adding an option to allow empty constraint.
    if (spec.vertices.size() == 0) {
      throw std::runtime_error(fmt::format(
          "AddFixedConstraint(): No constraint has been added between "
          "deformable body with id {} and rigid body with name {}. Remove the "
          "call to AddFixedConstraint() if this is intended.",
          id_, body_B.name()));
    }
    fixed_constraint_specs_.push_back(std::move(spec));
  } else if (der_model_) {
    /* Create an empty spec first. We will add to it. */
    internal::FilamentRigidFixedConstraintSpec spec{
        id_, body_B.index(), {}, {}, {}, {}, constraint_id};
    const int num_nodes = der_model_->num_nodes();
    for (int i = 0; i < num_nodes; ++i) {
      /* The node position in the filament body's geometry frame. */
      const Vector3<double> p_APi = filament_G_->node_pos().col(i);
      /* Note that `shape` is also registered in the A frame in the throw-away
       scene graph. */
      const std::vector<geometry::SignedDistanceToPoint<double>>
          signed_distances = query.ComputeSignedDistanceToPoint(p_APi);
      DRAKE_DEMAND(ssize(signed_distances) == 1);
      const double signed_distance = signed_distances[0].distance;
      if (signed_distance <= 0.0) {
        spec.nodes.push_back(i);
        spec.p_BQs.emplace_back(X_BA * p_APi);
      }
    }
    /* If node i and node i+1 are fixed, also fix edge i. */
    for (int k = 0; k < ssize(spec.nodes); ++k) {
      const int i = spec.nodes[k];
      const int ip1 =
          der_model_->has_closed_ends() ? (i + 1) % num_nodes : (i + 1);
      /* The m₁ director in the filament body's geometry frame. */
      const Vector3<double> m1_Ai = filament_G_->edge_m1().col(i);
      if (ip1 == spec.nodes[(k + 1) % ssize(spec.nodes)]) {
        spec.edges.push_back(i);
        spec.m1_Bs.emplace_back(X_BA * m1_Ai);
      }
    }
    /* Throw if no constraints are added. */
    if (spec.nodes.size() == 0) {
      throw std::runtime_error(fmt::format(
          "AddFixedConstraint(): No constraint has been added between "
          "deformable body with id {} and rigid body with name {}. Remove the "
          "call to AddFixedConstraint() if this is intended.",
          id_, body_B.name()));
    }
  } else {
    DRAKE_UNREACHABLE();
  }
  return constraint_id;
}

template <typename T>
void DeformableBody<T>::SetPositions(
    systems::Context<T>* context,
    const Eigen::Ref<const Matrix3X<T>>& q) const {
  DRAKE_THROW_UNLESS(context != nullptr);
  this->GetParentTreeSystem().ValidateContext(*context);
  if (fem_model_) {
    const int num_nodes = fem_model_->num_nodes();
    DRAKE_THROW_UNLESS(q.cols() == num_nodes);
    auto all_finite = [](const Matrix3X<T>& positions) {
      return positions.array().isFinite().all();
    };
    DRAKE_THROW_UNLESS(all_finite(q));

    context->get_mutable_discrete_state(discrete_state_index_)
        .get_mutable_value()
        .head(num_nodes * 3) = Eigen::Map<const VectorX<T>>(q.data(), q.size());
  } else if (der_model_) {
    const int num_nodes = der_model_->num_nodes();
    DRAKE_THROW_UNLESS(q.cols() == num_nodes);
    Eigen::VectorBlock<VectorX<T>> discrete_state_vector =
        context->get_mutable_discrete_state(discrete_state_index_)
            .get_mutable_value();
    for (int i = 0; i < num_nodes; ++i)
      discrete_state_vector.template segment<3>(4 * i) = q.col(i);
  } else {
    DRAKE_UNREACHABLE();
  }
}

template <typename T>
Matrix3X<T> DeformableBody<T>::GetPositions(
    const systems::Context<T>& context) const {
  this->GetParentTreeSystem().ValidateContext(context);

  if (fem_model_) {
    const int num_nodes = fem_model_->num_nodes();
    const VectorX<T>& q = context.get_discrete_state(discrete_state_index_)
                              .get_value()
                              .head(num_nodes * 3);
    return Eigen::Map<const Matrix3X<T>>(q.data(), 3, num_nodes);
  } else if (der_model_) {
    const VectorX<T>& discrete_state_vector =
        context.get_discrete_state(discrete_state_index_).get_value();
    const int num_nodes = der_model_->num_nodes();
    Matrix3X<T> node_positions(3, num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      node_positions.col(i) = discrete_state_vector.template segment<3>(4 * i);
    }
    return node_positions;
  } else {
    DRAKE_UNREACHABLE();
  }
}

template <typename T>
bool DeformableBody<T>::is_enabled(const systems::Context<T>& context) const {
  this->GetParentTreeSystem().ValidateContext(context);
  return context.get_parameters().template get_abstract_parameter<bool>(
      is_enabled_parameter_index_);
}

template <typename T>
void DeformableBody<T>::Disable(systems::Context<T>* context) const {
  DRAKE_THROW_UNLESS(context != nullptr);
  this->GetParentTreeSystem().ValidateContext(*context);
  context->get_mutable_abstract_parameter(is_enabled_parameter_index_)
      .template set_value<bool>(false);
  /* Set both the accelerations and the velocities to zero, noting that the
   dofs are stored in the order of q, v, and then a. */
  context->get_mutable_discrete_state(discrete_state_index_)
      .get_mutable_value()
      .segment(num_dofs(), 2 * num_dofs())
      .setZero();
}

template <typename T>
void DeformableBody<T>::Enable(systems::Context<T>* context) const {
  DRAKE_THROW_UNLESS(context != nullptr);
  this->GetParentTreeSystem().ValidateContext(*context);
  context->get_mutable_abstract_parameter(is_enabled_parameter_index_)
      .set_value(true);
}

template <typename T>
Vector3<T> DeformableBody<T>::CalcCenterOfMassPositionInWorld(
    const systems::Context<T>& context) const {
  DRAKE_DEMAND(fem_model_ != nullptr);
  const fem::FemState<T>& fem_state =
      this->GetParentTreeSystem()
          .get_cache_entry(fem_state_cache_index_)
          .template Eval<fem::FemState<T>>(context);
  return fem_model_->CalcCenterOfMassPositionInWorld(fem_state);
}

template <typename T>
Vector3<T> DeformableBody<T>::CalcCenterOfMassTranslationalVelocityInWorld(
    const systems::Context<T>& context) const {
  DRAKE_DEMAND(fem_model_ != nullptr);
  const fem::FemState<T>& fem_state =
      this->GetParentTreeSystem()
          .get_cache_entry(fem_state_cache_index_)
          .template Eval<fem::FemState<T>>(context);
  return fem_model_->CalcCenterOfMassTranslationalVelocityInWorld(fem_state);
}

template <typename T>
Vector3<T> DeformableBody<T>::CalcEffectiveAngularVelocity(
    const systems::Context<T>& context) const {
  DRAKE_DEMAND(fem_model_ != nullptr);
  const fem::FemState<T>& fem_state =
      this->GetParentTreeSystem()
          .get_cache_entry(fem_state_cache_index_)
          .template Eval<fem::FemState<T>>(context);
  return fem_model_->CalcEffectiveAngularVelocity(fem_state);
}

template <typename T>
DeformableBody<T>::DeformableBody(
    DeformableBodyIndex index, DeformableBodyId id, std::string name,
    GeometryId geometry_id, ModelInstanceIndex model_instance,
    const VolumeMesh<double>& mesh_G, const math::RigidTransform<double>& X_WG,
    const fem::DeformableBodyConfig<T>& config, const Vector3<double>& weights)
    : MultibodyElement<T>(model_instance, index),
      id_(id),
      name_(std::move(name)),
      geometry_id_(geometry_id),
      mesh_G_(mesh_G),
      X_WG_(X_WG),
      X_WD_(X_WG),
      config_(config) {
  if constexpr (std::is_same_v<T, double>) {
    geometry::VolumeMesh<double> mesh_W = mesh_G;
    mesh_W.TransformVertices(X_WG);
    BuildLinearVolumetricModel(mesh_W, config, weights);
    reference_positions_.resize(3 * mesh_W.num_vertices());
    for (int v = 0; v < mesh_W.num_vertices(); ++v) {
      reference_positions_.template segment<3>(3 * v) = mesh_W.vertex(v);
    }
  } else {
    throw std::runtime_error(
        "DeformableBody<T>::DeformableBody(): T must be double.");
  }
}

template <typename T>
DeformableBody<T>::DeformableBody(DeformableBodyIndex index,
                                  DeformableBodyId id, std::string name,
                                  geometry::GeometryId geometry_id,
                                  ModelInstanceIndex model_instance,
                                  const geometry::Filament& filament_G,
                                  const math::RigidTransform<double>& X_WG,
                                  const fem::DeformableBodyConfig<T>& config)
    : MultibodyElement<T>(model_instance, index),
      id_(id),
      name_(std::move(name)),
      geometry_id_(geometry_id),
      filament_G_(filament_G),
      X_WG_(X_WG),
      X_WD_(X_WG),
      config_(config) {
  if constexpr (std::is_same_v<T, double>) {
    BuildFilamentDerModel(X_WG, filament_G, config);
    reference_positions_ = (X_WG * filament_G.node_pos()).reshaped();
  } else {
    throw std::runtime_error(
        "DeformableBody<T>::DeformableBody(): T must be double.");
  }
}

template <typename T>
std::unique_ptr<DeformableBody<double>> DeformableBody<T>::CloneToDouble()
    const {
  if constexpr (!std::is_same_v<T, double>) {
    /* A non-double body shouldn't exist in the first place. */
    DRAKE_UNREACHABLE();
  } else {
    std::unique_ptr<DeformableBody<double>> clone;
    if (mesh_G_.has_value()) {
      clone =
          std::unique_ptr<DeformableBody<double>>(new DeformableBody<double>(
              this->index(), id_, name_, geometry_id_, this->model_instance(),
              *mesh_G_, X_WG_, config_, fem_model_->tangent_matrix_weights()));
    } else if (filament_G_.has_value()) {
      clone =
          std::unique_ptr<DeformableBody<double>>(new DeformableBody<double>(
              this->index(), id_, name_, geometry_id_, this->model_instance(),
              *filament_G_, X_WG_, config_));
    } else {
      DRAKE_UNREACHABLE();
    }
    /* We go through all data member one by one in order, and either copy them
     over or explain why they don't need to be copied. */
    /* id_ is copied in the constructor above. */
    /* name_ is copied in the constructor above. */
    /* geometry_id_ is copied in the constructor above. */
    /* mesh_G_ is copied in the constructor above. */
    /* filament_G_ is copied in the constructor above. */
    /* X_WG_ is copied in the constructor above. */
    /* Copy over X_WD_. */
    clone->X_WD_ = X_WD_;
    /* config_ is copied in the constructor above. */
    /* Copy over reference_positions_. */
    clone->reference_positions_ = reference_positions_;
    /* fem_model_ is constructed in the constructor above. */
    /* discrete_state_index_ is set in DoDeclareDiscreteState() when the
     owning DeformableModel declares system resources. */
    /* is_enabled_parameter_index_ is set in DoDeclareParameters() when the
     owning DeformableModel declares system resources. */
    /* Copy over fixed_constraint_specs_. */
    clone->fixed_constraint_specs_ = fixed_constraint_specs_;
    /* gravity_forces_ and external_forces_ are set when the owning
     DeformableModel declares system resources. */
    return clone;
  }
}

template <typename T>
void DeformableBody<T>::SetExternalForces(
    const std::vector<std::unique_ptr<ForceDensityFieldBase<T>>>&
        external_forces,
    const Vector3<T>& gravity) {
  const T& density = config_.mass_density();
  gravity_force_ = std::make_unique<GravityForceField<T>>(gravity, density);
  external_forces_.clear();
  external_forces_.push_back(gravity_force_.get());
  for (const auto& force : external_forces) {
    external_forces_.push_back(force.get());
  }
}

template <typename T>
template <typename T1>
typename std::enable_if_t<std::is_same_v<T1, double>, void>
DeformableBody<T>::BuildLinearVolumetricModel(
    const VolumeMesh<double>& mesh, const fem::DeformableBodyConfig<T>& config,
    const Vector3<double>& weights) {
  switch (config.material_model()) {
    case MaterialModel::kLinear:
      BuildLinearVolumetricModelHelper<fem::internal::LinearConstitutiveModel>(
          mesh, config, weights);
      break;
    case MaterialModel::kCorotated:
      BuildLinearVolumetricModelHelper<fem::internal::CorotatedModel>(
          mesh, config, weights);
      break;
    case MaterialModel::kNeoHookean:
      BuildLinearVolumetricModelHelper<fem::internal::NeoHookeanModel>(
          mesh, config, weights);
      break;
    case MaterialModel::kLinearCorotated:
      BuildLinearVolumetricModelHelper<fem::internal::LinearCorotatedModel>(
          mesh, config, weights);
      break;
  }
}

template <typename T>
template <template <typename> class Model, typename T1>
typename std::enable_if_t<std::is_same_v<T1, double>, void>
DeformableBody<T>::BuildLinearVolumetricModelHelper(
    const VolumeMesh<double>& mesh, const fem::DeformableBodyConfig<T>& config,
    const Vector3<double>& weights) {
  constexpr int kNaturalDimension = 3;
  constexpr int kSpatialDimension = 3;
  constexpr int kQuadratureOrder = 1;
  using QuadratureType =
      fem::internal::SimplexGaussianQuadrature<kNaturalDimension,
                                               kQuadratureOrder>;
  constexpr int kNumQuads = QuadratureType::num_quadrature_points;
  using IsoparametricElementType =
      fem::internal::LinearSimplexElement<T, kNaturalDimension,
                                          kSpatialDimension, kNumQuads>;
  using ConstitutiveModelType = Model<T>;
  static_assert(
      std::is_base_of_v<
          fem::internal::ConstitutiveModel<
              ConstitutiveModelType, typename ConstitutiveModelType::Traits>,
          ConstitutiveModelType>,
      "The template parameter 'Model' must be derived from "
      "ConstitutiveModel.");
  using FemElementType =
      fem::internal::VolumetricElement<IsoparametricElementType, QuadratureType,
                                       ConstitutiveModelType>;
  using FemModelType = fem::internal::VolumetricModel<FemElementType>;

  const fem::DampingModel<T> damping_model(
      config.mass_damping_coefficient(),
      config.stiffness_damping_coefficient());

  auto concrete_fem_model = std::make_unique<FemModelType>(weights);
  ConstitutiveModelType constitutive_model(config.youngs_modulus(),
                                           config.poissons_ratio());
  typename FemModelType::VolumetricBuilder builder(concrete_fem_model.get());
  builder.AddLinearTetrahedralElements(mesh, constitutive_model,
                                       config.mass_density(), damping_model);
  builder.Build();
  fem_model_ = std::move(concrete_fem_model);
}

template <typename T>
template <typename T1>
typename std::enable_if_t<std::is_same_v<T1, double>, void>
DeformableBody<T>::BuildFilamentDerModel(
    const math::RigidTransform<T>& X_WG, const geometry::Filament& filament_G,
    const fem::DeformableBodyConfig<T>& config) {
  const Eigen::Matrix3X<T> node_pos = X_WG * filament_G.node_pos();
  const Eigen::Matrix3X<T> edge_m1 = X_WG.rotation() * filament_G.edge_m1();
  const int num_nodes = node_pos.cols();
  const int num_edges = edge_m1.cols();
  DRAKE_THROW_UNLESS(num_nodes >= 2);
  DRAKE_THROW_UNLESS(num_edges ==
                     (filament_G.closed() ? num_nodes : num_nodes - 1));
  Eigen::Matrix3X<T> edge_t(3, num_edges);
  for (int i = 0; i < num_edges; ++i) {
    const int ip1 = (i + 1) % num_nodes;
    edge_t.col(i) = math::internal::NormalizeOrThrow<T>(
        node_pos.col(ip1) - node_pos.col(i), __func__);
  }
  Eigen::Matrix3X<T> edge_d1(3, num_edges);
  math::SpaceParallelFrameTransport<T>(edge_t, edge_m1.col(0), &edge_d1);

  typename der::DerModel<T>::Builder builder;
  builder.AddFirstEdge(node_pos.col(0), 0.0, node_pos.col(1), edge_m1.col(0));
  for (int i = 1; i < num_edges; ++i) {
    const T gamma_i = math::SignedAngleAroundAxis<T>(
        edge_d1.col(i), edge_m1.col(i), edge_t.col(i));
    const int ip1 = (i + 1) % num_nodes;
    builder.AddEdge(gamma_i, node_pos.col(ip1));
  }

  builder.SetUndeformedStateToInitialState();

  const T E = config.youngs_modulus();
  const T G = E / (2 * (1 + config.poissons_ratio()));
  const T rho = config.mass_density();
  builder.SetMaterialProperties(E, G, rho);
  builder.SetDampingCoefficients(config.mass_damping_coefficient(),
                                 config.stiffness_damping_coefficient());

  std::visit(
      overloaded{
          [&builder](const geometry::Filament::CircularCrossSection& cs) {
            builder.SetCircularCrossSection(cs.diameter / 2);
          },
          [&builder](const geometry::Filament::RectangularCrossSection& cs) {
            builder.SetRectangularCrossSection(cs.width, cs.height);
          }},
      filament_G.cross_section());

  der_model_ = builder.Build();
}

template <typename T>
void DeformableBody<T>::DoDeclareDiscreteState(
    internal::MultibodyTreeSystem<T>* tree_system) {
  if (fem_model_) {
    std::unique_ptr<fem::FemState<T>> default_fem_state =
        fem_model_->MakeFemState();
    const int num_dofs = default_fem_state->num_dofs();
    VectorX<T> model_state(num_dofs * 3 /* q, v, and a */);
    model_state.head(num_dofs) = default_fem_state->GetPositions();
    model_state.segment(num_dofs, num_dofs) =
        default_fem_state->GetVelocities();
    model_state.tail(num_dofs) = default_fem_state->GetAccelerations();
    discrete_state_index_ =
        this->DeclareDiscreteState(tree_system, model_state);
  } else if (der_model_) {
    std::unique_ptr<der::internal::DerState<T>> default_der_state =
        der_model_->CreateDerState();
    const VectorX<T> model_state = default_der_state->Serialize();
    discrete_state_index_ =
        this->DeclareDiscreteState(tree_system, model_state);
  } else {
    DRAKE_UNREACHABLE();
  }
}

template <typename T>
void DeformableBody<T>::SetDefaultState(const systems::Context<T>&,
                                        systems::State<T>* state) const {
  if (fem_model_) {
    state->get_mutable_discrete_state(discrete_state_index_)
        .get_mutable_value()
        .head(fem_model_->num_dofs()) = CalcDefaultPositions();
  } else if (der_model_) {
    std::unique_ptr<der::internal::DerState<T>> der_state =
        der_model_->CreateDerState();
    der_state->Transform((X_WD_ * X_WG_.inverse()).cast<T>());
    state->get_mutable_discrete_state(discrete_state_index_)
        .get_mutable_value() = der_state->Serialize();
  } else {
    DRAKE_UNREACHABLE();
  }
}

template <typename T>
VectorX<T> DeformableBody<T>::CalcDefaultPositions() const {
  const VectorX<double>& p_WVg = reference_positions_;
  const int num_dofs = this->num_dofs();
  VectorX<T> p_WVd = VectorX<T>::Zero(num_dofs);
  /* `reference_positions_` stores the list of p_WVg for all vertices V
   in the mesh and we have p_WVg = X_WG * p_GV, where p_GV is the position
   of vertex V in the geometry frame G.
   Now we want the default state to store the poisitions p_WVd, the positions
   of the vertices of the deformable body with the default pose D, which is
   given by p_WVd = X_WD * p_DV. By noting p_DV = p_GV, we get
   p_WVd = X_WD * p_GV = X_WD * X_WG⁻¹ * p_WVd. */
  const math::RigidTransform<double> X_DG = X_WD_ * X_WG_.inverse();
  for (int i = 0; i < num_dofs / 3; ++i) {
    p_WVd.template segment<3>(3 * i) = X_DG * p_WVg.template segment<3>(3 * i);
  }
  return p_WVd;
}

template <typename T>
void DeformableBody<T>::DoDeclareParameters(
    internal::MultibodyTreeSystem<T>* tree_system) {
  is_enabled_parameter_index_ =
      this->DeclareAbstractParameter(tree_system, Value<bool>(true));
}

template <typename T>
void DeformableBody<T>::DoDeclareCacheEntries(
    internal::MultibodyTreeSystem<T>* tree_system) {
  if (fem_model_) {
    /* Declare cache entry for FemState. */
    DRAKE_DEMAND(fem_model_ != nullptr);
    std::unique_ptr<fem::FemState<T>> model_state = fem_model_->MakeFemState();
    const auto& fem_state_cache_entry = this->DeclareCacheEntry(
        tree_system, fmt::format("fem_state_for_body_{}", id_.get_value()),
        systems::ValueProducer(
            *model_state,
            std::function<void(const systems::Context<T>&, fem::FemState<T>*)>(
                [this](const systems::Context<T>& context,
                       fem::FemState<T>* state) {
                  this->CalcFemStateFromDiscreteValues(context, state);
                })),
        {tree_system->xd_ticket()});
    fem_state_cache_index_ = fem_state_cache_entry.cache_index();
  } else if (der_model_) {
    std::unique_ptr<der::internal::DerState<T>> model_state =
        der_model_->CreateDerState();
    const auto& der_state_cache_entry = this->DeclareCacheEntry(
        tree_system, fmt::format("der_state_for_body_{}", id_.get_value()),
        systems::ValueProducer(*model_state,
                               std::function<void(const systems::Context<T>&,
                                                  der::internal::DerState<T>*)>(
                                   [this](const systems::Context<T>& context,
                                          der::internal::DerState<T>* state) {
                                     this->CalcDerStateFromDiscreteValues(
                                         context, state);
                                   })),
        {tree_system->xd_ticket()});
    der_state_cache_index_ = der_state_cache_entry.cache_index();
  } else {
    DRAKE_UNREACHABLE();
  }
}

template <typename T>
void DeformableBody<T>::CalcFemStateFromDiscreteValues(
    const systems::Context<T>& context, fem::FemState<T>* fem_state) const {
  DRAKE_DEMAND(fem_state != nullptr);
  const systems::BasicVector<T>& discrete_vector =
      context.get_discrete_state().get_vector(discrete_state_index_);
  const VectorX<T>& discrete_value = discrete_vector.value();
  DRAKE_DEMAND(discrete_value.size() % 3 == 0);
  const int num_dofs = discrete_value.size() / 3;
  DRAKE_DEMAND(num_dofs == fem_model_->num_dofs());

  fem_state->SetPositions(discrete_value.head(num_dofs));
  fem_state->SetVelocities(discrete_value.segment(num_dofs, num_dofs));
  fem_state->SetAccelerations(discrete_value.tail(num_dofs));
}

template <typename T>
void DeformableBody<T>::CalcDerStateFromDiscreteValues(
    const systems::Context<T>& context,
    der::internal::DerState<T>* der_state) const {
  DRAKE_DEMAND(der_state != nullptr);
  const systems::BasicVector<T>& discrete_vector =
      context.get_discrete_state().get_vector(discrete_state_index_);
  const VectorX<T>& discrete_value = discrete_vector.value();
  der_state->Deserialize(discrete_value);
}

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::DeformableBody);

}  // namespace multibody
}  // namespace drake
