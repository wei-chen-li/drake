#include "drake/geometry/proximity/filament_contact_internal.h"

#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <fcl/fcl.h>

#include "drake/common/overloaded.h"
#include "drake/common/type_safe_index.h"
#include "drake/geometry/proximity/filament_meshed_geometry.h"
#include "drake/geometry/proximity/filament_self_contact_filter.h"
#include "drake/geometry/proximity/hydroelastic_calculator.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/proximity_utilities.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/math/unit_vector.h"

namespace drake {
namespace geometry {
namespace internal {

namespace filament {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using math::RigidTransformd;
using math::RotationMatrixd;

namespace {

/* Helper function that creates a *deep* copy of the given collision object.
 */
std::unique_ptr<fcl::CollisionObjectd> CloneFclObject(
    const fcl::CollisionObjectd& object_source) {
  const auto& shape_source = *object_source.collisionGeometry();

  std::shared_ptr<fcl::CollisionGeometryd> shape_clone;
  /* Filaments consist of only cylinder or box shapes. */
  if (shape_source.getNodeType() == fcl::GEOM_CYLINDER) {
    const auto& cylinder = dynamic_cast<const fcl::Cylinderd&>(shape_source);
    shape_clone =
        std::make_shared<fcl::Cylinderd>(cylinder.radius, cylinder.lz);

  } else if (shape_source.getNodeType() == fcl::GEOM_BOX) {
    const auto& box = dynamic_cast<const fcl::Boxd&>(shape_source);
    shape_clone = std::make_shared<fcl::Boxd>(box.side);

  } else {
    DRAKE_UNREACHABLE();
  }

  /* A copy of the geometry is passed to FCL, but CollisionObject's
   constructor resets that copy's local bounding box to fit the _instantiated_
   shape. So we retain a pointer to the shape copy long enough after handing
   it off to FCL to fix it back up to its original AABB. */
  auto object_clone = std::make_unique<fcl::CollisionObjectd>(shape_clone);

  /* The source's local AABB may have been inflated if the underlying object
   is associated with a compliant hydroelastic shape with a non-zero margin;
   therefore the AABB that fits the shape may not be what we want. We can't
   tell simply by looking at the fcl object if this is the case, so, we'll
   simply copy the source's local AABB verbatim to preserve the effect. */
  shape_clone->aabb_local.min_ = shape_source.aabb_local.min_;
  shape_clone->aabb_local.max_ = shape_source.aabb_local.max_;
  shape_clone->aabb_radius = shape_source.aabb_radius;

  object_clone->setUserData(object_source.getUserData());
  object_clone->setTransform(object_source.getTransform());
  object_clone->computeAABB();

  return object_clone;
}

/* Helper functions to facilitate exercising FCL's broadphase code. FCL has
 inconsistent usage of `const`. As such, even though the broadphase structures
 do not change during collision and distance queries, they are nevertheless
 declared non-const, requiring Drake to do some const casting in what would
 otherwise be a const context. */
template <typename T>
void FclCollide(const fcl::DynamicAABBTreeCollisionManager<T>& tree1,
                const fcl::DynamicAABBTreeCollisionManager<T>& tree2,
                void* cdata, fcl::CollisionCallBack<T> callback) {
  DRAKE_THROW_UNLESS(&tree1 != &tree2);
  tree1.collide(const_cast<fcl::DynamicAABBTreeCollisionManager<T>*>(&tree2),
                cdata, callback);
}

/* Each filament is represented by a FilamentData struct, which contains the
 collision objects and the AABB tree. Because Drake compiles FCL using hidden
 symbol visibility, keep this struct inside anonymous namespace to avoid
 visibility complaints from the compiler. */
struct FilamentData {
  FilamentData(bool closed_in, int num_nodes_in, int num_edges_in)
      : closed(closed_in), num_nodes(num_nodes_in), num_edges(num_edges_in) {
    DRAKE_THROW_UNLESS(num_nodes >= 2);
    DRAKE_THROW_UNLESS(num_edges == (closed ? num_nodes : num_nodes - 1));
    tree.clear();
    objects.reserve(num_edges);
    object_pointers.reserve(num_edges);
  }

  const bool closed{};
  const int num_nodes{};
  const int num_edges{};
  Eigen::Matrix3Xd node_positions;
  Eigen::Vector3d margins;
  fcl::DynamicAABBTreeCollisionManagerd tree;
  std::vector<std::unique_ptr<fcl::CollisionObjectd>> objects;
  std::unordered_set<const fcl::CollisionObjectd*> object_pointers;
  std::optional<FilamentSelfContactFilter> self_contact_filter;
  std::optional<FilamentMeshedGeometry> soft_geometry;
};

/* Each added filament to Geometries is indexed by a FilamentIndex. */
using FilamentIndex = TypeSafeIndex<class FilamentTag>;

/* A struct holding metadata of a filament edge. */
class FilamentEdgeData {
 public:
  FilamentEdgeData(FilamentIndex filament_index, int edge_index)
      : filament_index_(filament_index), edge_index_(edge_index) {
    DRAKE_THROW_UNLESS(edge_index >= 0);
  }

  void write_to(fcl::CollisionObjectd* object) const {
    static_assert(sizeof(FilamentIndex) + sizeof(int) == sizeof(intptr_t));
    const intptr_t data = (intptr_t{filament_index_} << (sizeof(int) * 8)) |
                          intptr_t{edge_index_};
    static_assert(sizeof(intptr_t) == sizeof(void*));
    object->setUserData(reinterpret_cast<void*>(data));
  }

  static FilamentEdgeData read_from(const fcl::CollisionObjectd& object) {
    const intptr_t data = reinterpret_cast<intptr_t>(object.getUserData());
    FilamentIndex filament_index(data >> (sizeof(int) * 8));
    constexpr intptr_t kEdgeIndexMask = (intptr_t{1} << (sizeof(int) * 8)) - 1;
    int edge_index = static_cast<int>(data & kEdgeIndexMask);
    return FilamentEdgeData(filament_index, edge_index);
  }

  FilamentIndex filament_index() const { return filament_index_; }
  int edge_index() const { return edge_index_; }

  void swap(FilamentEdgeData& other) {
    std::swap(filament_index_, other.filament_index_);
    std::swap(edge_index_, other.edge_index_);
  }

 private:
  FilamentIndex filament_index_;
  int edge_index_;
};

template <typename T>
struct FilamentFilamentCollisionCallbackData {
  FilamentFilamentCollisionCallbackData(
      const std::vector<GeometryId>* filament_index_to_id_in,
      const FilamentSelfContactFilter* self_contact_filter_in,
      const FilamentMeshedGeometry* filament_meshed_geometry_A_in,
      const FilamentMeshedGeometry* filament_meshed_geometry_B_in)
      : filament_index_to_id(filament_index_to_id_in),
        self_contact_filter(self_contact_filter_in),
        filament_meshed_geometry_A(filament_meshed_geometry_A_in),
        filament_meshed_geometry_B(filament_meshed_geometry_B_in) {
    DRAKE_THROW_UNLESS(filament_index_to_id != nullptr);
    request.num_max_contacts = 1;
    request.enable_contact = true;
    request.gjk_tolerance = 2e-12;
    request.gjk_solver_type = fcl::GJKSolverType::GST_LIBCCD;
  }

  /* Request passed to fcl::collide(). */
  fcl::CollisionRequestd request;
  /* Mapping from FilamentIndex to GeometryId. */
  const std::vector<GeometryId>* const filament_index_to_id;
  /* Self-contact filter to detemine if two edges of the same filament should
   detect collision. Is nullptr if two edges belong to different filaments. */
  const FilamentSelfContactFilter* const self_contact_filter;
  /* Filament soft geometry for generating meshes and hydroelastic pressures for
   filament segments. */
  const FilamentMeshedGeometry* const filament_meshed_geometry_A;
  const FilamentMeshedGeometry* const filament_meshed_geometry_B;

  /* Write the contact result to this field. */
  struct ContactResult {
    std::vector<Vector3<T>> p_WCs;
    std::vector<Vector3<T>> nhats_BA_W;
    std::vector<T> signed_distances;
    std::vector<int> contact_edge_indexes_A;
    std::vector<int> contact_edge_indexes_B;
    std::vector<std::unique_ptr<ContactSurface<T>>> contact_surfaces;
  } contact_result;
};

template <typename T>
bool FilamentFilamentCollisionCallback(fcl::CollisionObjectd* object_A,
                                       fcl::CollisionObjectd* object_B,
                                       void* callback_data_in) {
  auto callback_data =
      static_cast<FilamentFilamentCollisionCallbackData<T>*>(callback_data_in);
  const std::vector<GeometryId>& filament_index_to_id =
      *callback_data->filament_index_to_id;

  auto data_A = FilamentEdgeData::read_from(*object_A);
  auto data_B = FilamentEdgeData::read_from(*object_B);
  GeometryId id_A = filament_index_to_id.at(data_A.filament_index());
  GeometryId id_B = filament_index_to_id.at(data_B.filament_index());
  if ((id_A.get_value() > id_B.get_value()) ||
      (id_A == id_B && data_A.edge_index() > data_B.edge_index())) {
    std::swap(object_A, object_B);
    std::swap(data_A, data_B);
    std::swap(id_A, id_B);
  }

  if (id_A == id_B) {
    DRAKE_DEMAND(callback_data->self_contact_filter != nullptr);
    if (!callback_data->self_contact_filter->ShouldCollide(
            data_A.edge_index(), data_B.edge_index())) {
      /* NOTE: Here and below, false is returned regardless of whether collision
       is detected or not because true tells the broadphase manager to
       terminate. Since we want *all* collisions, we return false. */
      return false;
    }
  }

  const bool use_patch_contact =
      callback_data->filament_meshed_geometry_A != nullptr &&
      callback_data->filament_meshed_geometry_B != nullptr;

  /* If use patch contact, we only need the binary isCollision() result. */
  if (use_patch_contact) callback_data->request.enable_contact = false;
  fcl::CollisionResultd result;
  fcl::collide(object_A, object_B, callback_data->request, result);
  if (use_patch_contact) callback_data->request.enable_contact = true;
  if (!result.isCollision()) return false;

  /* Fill in point contact result. */
  if (!use_patch_contact) {
    const fcl::Contactd& contact = result.getContact(0);
    auto& contact_result = callback_data->contact_result;
    contact_result.p_WCs.emplace_back(contact.pos);
    contact_result.nhats_BA_W.emplace_back(-contact.normal);
    contact_result.signed_distances.emplace_back(-contact.penetration_depth);
    contact_result.contact_edge_indexes_A.push_back(data_A.edge_index());
    contact_result.contact_edge_indexes_B.push_back(data_B.edge_index());
    return false;
  }

  /* Compute patch contact. */
  hydroelastic::SoftGeometry soft_geometry_A =
      callback_data->filament_meshed_geometry_A->MakeSoftGeometryForEdge(
          data_A.edge_index());
  hydroelastic::SoftGeometry soft_geometry_B =
      callback_data->filament_meshed_geometry_B->MakeSoftGeometryForEdge(
          data_B.edge_index());
  std::unique_ptr<ContactSurface<T>> contact_surface =
      hydroelastic::CalcCompliantCompliant(
          soft_geometry_A, math::RigidTransformd::Identity(), id_A,
          soft_geometry_B, math::RigidTransformd::Identity(), id_B,
          HydroelasticContactRepresentation::kPolygon);
  if (contact_surface == nullptr) return false;
  auto& contact_result = callback_data->contact_result;
  contact_result.contact_edge_indexes_A.push_back(data_A.edge_index());
  contact_result.contact_edge_indexes_B.push_back(data_B.edge_index());
  contact_result.contact_surfaces.emplace_back(std::move(contact_surface));
  return false;
}

template <typename T>
struct FilamentRigidCollisionCallbackData {
  FilamentRigidCollisionCallbackData(
      const CollisionFilter* collision_filter_in,
      const std::vector<GeometryId>* filament_index_to_id_in,
      const std::unordered_set<const fcl::CollisionObjectd*>*
          filament_objects_in,
      const FilamentMeshedGeometry* const filament_meshed_geometry_in,
      const hydroelastic::Geometries* hydroelastic_geometries_in)
      : collision_filter(collision_filter_in),
        filament_index_to_id(filament_index_to_id_in),
        filament_objects(filament_objects_in),
        filament_meshed_geometry(filament_meshed_geometry_in),
        hydroelastic_geometries(hydroelastic_geometries_in) {
    DRAKE_THROW_UNLESS(collision_filter != nullptr);
    DRAKE_THROW_UNLESS(filament_index_to_id != nullptr);
    DRAKE_THROW_UNLESS(filament_objects != nullptr);
    DRAKE_THROW_UNLESS(hydroelastic_geometries != nullptr);
    request.num_max_contacts = 1;
    request.enable_contact = true;
    request.gjk_tolerance = 2e-12;
    request.gjk_solver_type = fcl::GJKSolverType::GST_LIBCCD;
  }

  /* Request passed to fcl::collide(). */
  fcl::CollisionRequestd request;
  /* Collision filter to determine if a filament can collide with a rigid
   geometry. */
  const CollisionFilter* const collision_filter;
  /* Mapping from FilamentIndex to GeometryId. */
  const std::vector<GeometryId>* const filament_index_to_id;
  /* All collision objects in the filament. */
  const std::unordered_set<const fcl::CollisionObjectd*>* const
      filament_objects;
  /* Filament soft geometry for generating meshes and hydroelastic pressures for
   filament segments. */
  const FilamentMeshedGeometry* const filament_meshed_geometry;
  /* Hydroelastic representations of regid geometries. */
  const hydroelastic::Geometries* const hydroelastic_geometries;

  struct ContactResults {
    std::vector<Vector3<T>> p_WCs;
    std::vector<Vector3<T>> nhats_BA_W;
    std::vector<T> signed_distances;
    std::vector<int> contact_edge_indexes_A;
    std::vector<std::unique_ptr<ContactSurface<T>>> contact_surfaces;
  };
  /* Write the contact result to this field. */
  std::unordered_map<GeometryId, ContactResults> id_B_to_contact_result;
};

template <typename T>
bool FilamentRigidCollisionCallback(fcl::CollisionObjectd* object_A,
                                    fcl::CollisionObjectd* object_B,
                                    void* callback_data_in) {
  auto callback_data =
      static_cast<FilamentRigidCollisionCallbackData<T>*>(callback_data_in);
  const CollisionFilter& collision_filter = *callback_data->collision_filter;
  const std::vector<GeometryId>& filament_index_to_id =
      *callback_data->filament_index_to_id;
  const std::unordered_set<const fcl::CollisionObjectd*>& filament_objects =
      *callback_data->filament_objects;
  const hydroelastic::Geometries& hydroelastic_geometries =
      *callback_data->hydroelastic_geometries;

  DRAKE_ASSERT(filament_objects.contains(object_A) ^
               filament_objects.contains(object_B));
  if (filament_objects.contains(object_B)) {
    std::swap(object_A, object_B);
  }
  const auto data_A = FilamentEdgeData::read_from(*object_A);
  const GeometryId id_A = filament_index_to_id.at(data_A.filament_index());
  const auto data_B = EncodedData(*object_B);
  const GeometryId id_B = data_B.id();

  if (!collision_filter.CanCollideWith(id_A, id_B)) {
    /* NOTE: Here and below, false is returned regardless of whether collision
     is detected or not because true tells the broadphase manager to terminate.
     Since we want *all* collisions, we return false. */
    return false;
  }

  const bool use_patch_contact =
      callback_data->filament_meshed_geometry != nullptr &&
      hydroelastic_geometries.hydroelastic_type(id_B) !=
          HydroelasticType::kUndefined;

  /* If use patch contact, we only need the binary isCollision() result. */
  if (use_patch_contact) callback_data->request.enable_contact = false;
  fcl::CollisionResultd result;
  fcl::collide(object_A, object_B, callback_data->request, result);
  if (use_patch_contact) callback_data->request.enable_contact = true;
  if (!result.isCollision()) return false;

  /* Fill in point contact result. */
  if (!use_patch_contact) {
    const fcl::Contactd& contact = result.getContact(0);
    auto& contact_result = callback_data->id_B_to_contact_result[id_B];
    contact_result.p_WCs.emplace_back(contact.pos);
    contact_result.nhats_BA_W.emplace_back(-contact.normal);
    contact_result.signed_distances.emplace_back(-contact.penetration_depth);
    contact_result.contact_edge_indexes_A.push_back(data_A.edge_index());
    return false;
  }

  /* Compute patch contact. */
  hydroelastic::SoftGeometry soft_geometry_A =
      callback_data->filament_meshed_geometry->MakeSoftGeometryForEdge(
          data_A.edge_index());
  std::unique_ptr<ContactSurface<T>> contact_surface = [&]() {
    math::RigidTransformd X_WB(object_B->getTransform());
    if (hydroelastic_geometries.hydroelastic_type(id_B) ==
        HydroelasticType::kSoft) {
      const hydroelastic::SoftGeometry& soft_geometry_B =
          hydroelastic_geometries.soft_geometry(id_B);
      return hydroelastic::CalcCompliantCompliant(
          soft_geometry_A, math::RigidTransformd::Identity(), id_A,
          soft_geometry_B, X_WB, id_B,
          HydroelasticContactRepresentation::kPolygon);
    } else {
      const hydroelastic::RigidGeometry& rigid_geometry_B =
          hydroelastic_geometries.rigid_geometry(id_B);
      return hydroelastic::CalcRigidCompliant(
          soft_geometry_A, math::RigidTransformd::Identity(), id_A,
          rigid_geometry_B, X_WB, id_B,
          HydroelasticContactRepresentation::kPolygon);
    }
  }();

  if (contact_surface == nullptr) return false;
  auto& contact_result = callback_data->id_B_to_contact_result[id_B];
  contact_result.contact_edge_indexes_A.push_back(data_A.edge_index());
  contact_result.contact_surfaces.emplace_back(std::move(contact_surface));
  return false;
}

}  // namespace

class Geometries::Impl {
 public:
  Impl() = default;

  void AddFilamentGeometry(GeometryId id, const Filament& filament,
                           const ProximityProperties& props) {
    DRAKE_THROW_UNLESS(!is_filament(id));
    const Eigen::Matrix3Xd& node_pos = filament.node_pos();
    const Eigen::Matrix3Xd& edge_m1 = filament.edge_m1();
    const int num_nodes = node_pos.cols();
    const int num_edges = edge_m1.cols();

    std::optional<FilamentHydroelasticParameters> hydroelastic_params =
        FilamentHydroelasticParameters::Parse(props);

    FilamentData& filament_data =
        id_to_filament_data_
            .emplace(id, FilamentData(filament.closed(), num_nodes, num_edges))
            .first->second;
    const Vector3d margins = hydroelastic_params
                                 ? Vector3d(hydroelastic_params->margin,
                                            hydroelastic_params->margin, 0)
                                 : Vector3d::Zero();

    filament_data.node_positions = node_pos;
    filament_data.margins = margins;
    filament_data.objects.resize(num_edges);

    filament_index_to_id_.push_back(id);
    FilamentIndex filament_index(ssize(filament_index_to_id_) - 1);

    Eigen::RowVectorXd edge_lengths(num_edges);
    for (int i = 0; i < num_edges; ++i) {
      const int ip1 = (i + 1) % num_nodes;
      const Vector3d node_i = node_pos.col(i);
      const Vector3d node_ip1 = node_pos.col(ip1);
      const double l = (node_ip1 - node_i).norm();
      edge_lengths[i] = l;
      const Vector3d t = (node_ip1 - node_i) / l;
      const Vector3d& m1 = edge_m1.col(i);
      math::internal::ThrowIfNotOrthonormal(t, m1, __func__);
      /* Pose of the material frame in the world frame. */
      const RigidTransformd X_WM(
          RotationMatrixd::MakeFromOrthonormalColumns(m1, t.cross(m1), t),
          (node_i + node_ip1) / 2);

      std::shared_ptr<fcl::CollisionGeometryd> shape;
      const auto& cs = filament.cross_section();
      if (std::holds_alternative<Filament::CircularCrossSection>(cs)) {
        const auto& circ_cs = std::get<Filament::CircularCrossSection>(cs);
        shape = std::make_unique<fcl::Cylinderd>(circ_cs.diameter / 2, l);
      } else {
        const auto& rect_cs = std::get<Filament::RectangularCrossSection>(cs);
        shape = std::make_unique<fcl::Boxd>(rect_cs.width, rect_cs.height, l);
      }
      filament_data.objects[i] =
          std::make_unique<fcl::CollisionObjectd>(shape, X_WM.GetAsIsometry3());

      /* CollisionObject constructors resets bounding box of CollisionGeometry.
       Therefore, modifying the local bounding box must be done after that. */
      shape->computeLocalAABB();
      shape->aabb_local.max_ += margins;
      shape->aabb_local.min_ -= margins;
      shape->aabb_radius = (shape->aabb_local.min_ - shape->aabb_center).norm();
      filament_data.objects[i]->computeAABB();

      FilamentEdgeData filament_edge_data(filament_index, i);
      filament_edge_data.write_to(filament_data.objects[i].get());

      filament_data.object_pointers.insert(filament_data.objects[i].get());
      filament_data.tree.registerObject(filament_data.objects[i].get());
    }
    filament_data.tree.setup();

    const bool enable_self_contact =
        props.GetPropertyOrDefault("collision", "self_contact", true);
    if (enable_self_contact) {
      filament_data.self_contact_filter = FilamentSelfContactFilter(
          filament.closed(), edge_lengths,
          std::visit(
              overloaded{[](const Filament::CircularCrossSection& cs) {
                           return cs.diameter;
                         },
                         [](const Filament::RectangularCrossSection& cs) {
                           return std::hypot(cs.width, cs.height);
                         }},
              filament.cross_section()));
    }

    if (hydroelastic_params) {
      filament_data.soft_geometry =
          FilamentMeshedGeometry(filament, *hydroelastic_params);
    }
  }

  void RemoveGeometry(GeometryId id) { id_to_filament_data_.erase(id); }

  void UpdateFilamentConfigurationVector(
      GeometryId id, const Eigen::Ref<const Eigen::VectorXd>& q_WG) {
    DRAKE_THROW_UNLESS(is_filament(id));
    FilamentData& filament_data = id_to_filament_data_.at(id);
    const int num_nodes = filament_data.num_nodes;
    const int num_edges = filament_data.num_edges;
    /* `q_WG` should hold the node positions and edge m1 directors. */
    DRAKE_DEMAND(q_WG.size() == num_nodes * 3 + num_edges * 3);
    filament_data.node_positions =
        Eigen::Map<const Eigen::Matrix3Xd>(q_WG.data(), 3, num_nodes);
    if (filament_data.soft_geometry) {
      filament_data.soft_geometry->UpdateConfigurationVector(q_WG);
    }
    const Vector3d& margins = filament_data.margins;

    for (int i = 0; i < num_edges; ++i) {
      const int ip1 = (i + 1) % num_nodes;
      const Vector3d node_i = q_WG.template segment<3>(3 * i);
      const Vector3d node_ip1 = q_WG.template segment<3>(3 * ip1);
      const double l = (node_ip1 - node_i).norm();
      const Vector3d t = (node_ip1 - node_i) / l;
      const Vector3d m1 = q_WG.template segment<3>(num_nodes * 3 + 3 * i);
      math::internal::ThrowIfNotOrthonormal(t, m1, __func__);
      /* Pose of the material frame in the world frame. */
      const RigidTransformd X_WM(
          RotationMatrixd::MakeFromOrthonormalColumns(m1, t.cross(m1), t),
          (node_i + node_ip1) / 2);

      fcl::CollisionObjectd& object = *filament_data.objects[i];
      /* Casting away constness should be fine as long as we remember to call
       computeLocalAABB() after modifying the shape. */
      fcl::CollisionGeometryd* shape = const_cast<fcl::CollisionGeometryd*>(
          object.collisionGeometry().get());
      /* For performance considerations, we use static_cast here and guard with
       dynamic_cast assertions. */
      if (object.getNodeType() == fcl::GEOM_CYLINDER) {
        DRAKE_ASSERT(dynamic_cast<fcl::Cylinderd*>(shape) != nullptr);
        auto& cylinder = static_cast<fcl::Cylinderd&>(*shape);
        cylinder.lz = l;
      } else if (object.getNodeType() == fcl::GEOM_BOX) {
        DRAKE_ASSERT(dynamic_cast<fcl::Boxd*>(shape) != nullptr);
        auto& box = static_cast<fcl::Boxd&>(*shape);
        box.side[2] = l;
      } else {
        DRAKE_UNREACHABLE();
      }
      shape->computeLocalAABB();
      shape->aabb_local.max_ += margins;
      shape->aabb_local.min_ -= margins;
      shape->aabb_radius = (shape->aabb_local.min_ - shape->aabb_center).norm();
      object.setTransform(X_WM.GetAsIsometry3());
      object.computeAABB();
    }
    filament_data.tree.update();
  }

  FilamentContact<double> ComputeFilamentContact(
      const CollisionFilter& collision_filter,
      const std::vector<const void*>& rigid_body_trees_in,
      const hydroelastic::Geometries* hydroelastic_geometries) const {
    std::vector<const fcl::DynamicAABBTreeCollisionManagerd*> rigid_body_trees;
    for (const void* rigid_body_tree : rigid_body_trees_in) {
      DRAKE_THROW_UNLESS(rigid_body_tree != nullptr);
      rigid_body_trees.push_back(
          static_cast<const fcl::DynamicAABBTreeCollisionManagerd*>(
              rigid_body_tree));
    }
    DRAKE_THROW_UNLESS(hydroelastic_geometries != nullptr);
    FilamentContact<double> filament_contact;

    const auto cbegin = id_to_filament_data_.cbegin();
    const auto cend = id_to_filament_data_.cend();
    for (auto iter_A = cbegin; iter_A != cend; ++iter_A) {
      for (auto iter_B = iter_A; iter_B != cend; ++iter_B) {
        GeometryId id_A = iter_A->first;
        GeometryId id_B = iter_B->first;
        if (id_A == id_B || collision_filter.CanCollideWith(id_A, id_B))
          AddFilamentFilamentContactGeometryPair(id_A, id_B, &filament_contact);
      }
    }
    for (auto iter = cbegin; iter != cend; ++iter) {
      GeometryId id_A = iter->first;
      AddFilamentRigidContactGeometryPairs(
          id_A, collision_filter, rigid_body_trees, hydroelastic_geometries,
          &filament_contact);
    }
    return filament_contact;
  }

  bool is_filament(GeometryId id) const {
    return id_to_filament_data_.contains(id);
  }

  std::unique_ptr<Impl, ImplDeleter> Clone() const {
    auto clone = std::unique_ptr<Impl, ImplDeleter>(new Impl());
    for (const auto& pair : this->id_to_filament_data_) {
      GeometryId id = pair.first;
      const FilamentData& filament_data_source = pair.second;
      const bool closed = filament_data_source.closed;
      const int num_nodes = filament_data_source.num_nodes;
      const int num_edges = filament_data_source.num_edges;

      FilamentData& filament_data_clone =
          clone->id_to_filament_data_
              .emplace(id, FilamentData(closed, num_nodes, num_edges))
              .first->second;

      filament_data_clone.node_positions = filament_data_source.node_positions;
      filament_data_clone.margins = filament_data_source.margins;
      filament_data_clone.objects.resize(num_edges);

      for (int i = 0; i < num_edges; ++i) {
        filament_data_clone.objects[i] =
            CloneFclObject(*filament_data_source.objects[i]);
        filament_data_clone.object_pointers.insert(
            filament_data_clone.objects[i].get());
        filament_data_clone.tree.registerObject(
            filament_data_clone.objects[i].get());
      }
      filament_data_clone.tree.setup();
      filament_data_clone.self_contact_filter =
          filament_data_source.self_contact_filter;
      filament_data_clone.soft_geometry = filament_data_source.soft_geometry;
    }
    clone->filament_index_to_id_ = this->filament_index_to_id_;
    return clone;
  }

 private:
  void AddFilamentFilamentContactGeometryPair(
      GeometryId id_A, GeometryId id_B,
      FilamentContact<double>* filament_contact) const {
    DRAKE_THROW_UNLESS(filament_contact != nullptr);
    if (id_A.get_value() > id_B.get_value()) std::swap(id_A, id_B);
    const FilamentData& filament_data_A = id_to_filament_data_.at(id_A);
    const FilamentData& filament_data_B = id_to_filament_data_.at(id_B);
    /* Premature return if self contact is disabled. */
    if (id_A == id_B && !filament_data_A.self_contact_filter) return;

    FilamentFilamentCollisionCallbackData<double> callback_data(
        &filament_index_to_id_,
        (id_A == id_B) ? &filament_data_A.self_contact_filter.value() : nullptr,
        filament_data_A.soft_geometry ? &filament_data_A.soft_geometry.value()
                                      : nullptr,
        filament_data_B.soft_geometry ? &filament_data_B.soft_geometry.value()
                                      : nullptr);
    if (id_A == id_B) {
      filament_data_A.tree.collide(&callback_data,
                                   &FilamentFilamentCollisionCallback<double>);
    } else {
      FclCollide(filament_data_A.tree, filament_data_B.tree, &callback_data,
                 &FilamentFilamentCollisionCallback<double>);
    }
    auto& contact_result = callback_data.contact_result;
    if (contact_result.contact_edge_indexes_A.empty()) return;
    if (contact_result.contact_surfaces.empty()) {
      filament_contact->AddFilamentFilamentContactGeometryPair(
          id_A, id_B, std::move(contact_result.p_WCs),
          std::move(contact_result.nhats_BA_W),
          std::move(contact_result.signed_distances),
          std::move(contact_result.contact_edge_indexes_A),
          std::move(contact_result.contact_edge_indexes_B),
          filament_data_A.node_positions, filament_data_B.node_positions);
    } else {
      filament_contact->AddFilamentFilamentContactGeometryPair(
          id_A, id_B, contact_result.contact_surfaces,
          contact_result.contact_edge_indexes_A,
          contact_result.contact_edge_indexes_B, filament_data_A.node_positions,
          filament_data_B.node_positions);
    }
  }

  void AddFilamentRigidContactGeometryPairs(
      GeometryId id_A, const CollisionFilter& collision_filter,
      const std::vector<const fcl::DynamicAABBTreeCollisionManagerd*>&
          rigid_body_trees,
      const hydroelastic::Geometries* hydroelastic_geometries,
      FilamentContact<double>* filament_contact) const {
    const FilamentData& filament_data = id_to_filament_data_.at(id_A);
    FilamentRigidCollisionCallbackData<double> callback_data(
        &collision_filter, &filament_index_to_id_,
        &filament_data.object_pointers,
        filament_data.soft_geometry ? &filament_data.soft_geometry.value()
                                    : nullptr,
        hydroelastic_geometries);
    for (const auto rigid_body_tree : rigid_body_trees) {
      FclCollide(filament_data.tree, *rigid_body_tree, &callback_data,
                 &FilamentRigidCollisionCallback<double>);
    }
    for (const auto& [id_B, contact_result] :
         callback_data.id_B_to_contact_result) {
      if (contact_result.contact_edge_indexes_A.empty()) continue;
      if (contact_result.contact_surfaces.empty()) {
        filament_contact->AddFilamentRigidContactGeometryPair(
            id_A, id_B, std::move(contact_result.p_WCs),
            std::move(contact_result.nhats_BA_W),
            std::move(contact_result.signed_distances),
            std::move(contact_result.contact_edge_indexes_A),
            filament_data.node_positions);
      } else {
        filament_contact->AddFilamentRigidContactGeometryPair(
            id_A, id_B, contact_result.contact_surfaces,
            contact_result.contact_edge_indexes_A,
            filament_data.node_positions);
      }
    }
  }

  std::unordered_map<GeometryId, FilamentData> id_to_filament_data_;
  std::vector<GeometryId> filament_index_to_id_;
};

void Geometries::ImplDeleter::operator()(Geometries::Impl* ptr) {
  delete ptr;
}

Geometries::Geometries() {
  impl_ = std::unique_ptr<Impl, ImplDeleter>(new Impl());
}

Geometries::Geometries(const Geometries& other) : impl_(other.impl_->Clone()) {}

Geometries& Geometries::operator=(const Geometries& other) {
  this->impl_ = other.impl_->Clone();
  return *this;
}

Geometries::Geometries(Geometries&& other) = default;

Geometries& Geometries::operator=(Geometries&& other) = default;

Geometries::~Geometries() = default;

void Geometries::AddFilamentGeometry(GeometryId id, const Filament& filament,
                                     const ProximityProperties& props) {
  DRAKE_DEMAND(impl_ != nullptr);
  impl_->AddFilamentGeometry(id, filament, props);
}

void Geometries::RemoveGeometry(GeometryId id) {
  DRAKE_DEMAND(impl_ != nullptr);
  impl_->RemoveGeometry(id);
}

void Geometries::UpdateFilamentConfigurationVector(
    GeometryId id, const Eigen::Ref<const Eigen::VectorXd>& q_WG) {
  DRAKE_DEMAND(impl_ != nullptr);
  impl_->UpdateFilamentConfigurationVector(id, q_WG);
}

FilamentContact<double> Geometries::ComputeFilamentContact(
    const CollisionFilter& collision_filter,
    const std::vector<const void*>& rigid_body_trees,
    const hydroelastic::Geometries* hydroelastic_geometries) const {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->ComputeFilamentContact(collision_filter, rigid_body_trees,
                                       hydroelastic_geometries);
}

bool Geometries::is_filament(GeometryId id) const {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->is_filament(id);
}

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
