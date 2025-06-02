#include "drake/geometry/proximity/filament_contact_internal.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <fcl/fcl.h>

#include "drake/common/overloaded.h"
#include "drake/common/type_safe_index.h"
#include "drake/geometry/proximity/filament_self_contact_filter.h"
#include "drake/math/unit_vector.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

using Eigen::Matrix3d;
using Eigen::Vector3d;

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

/* Each filament is represented by a FilamentData struct, which contains the
 collision objects and the AABB tree. Because Drake compiles FCL using hidden
 symbol visibility, keep this struct inside anonymous namespace to avoid
 visibility complaints from the compiler. */
struct FilamentData {
  FilamentData(bool closed_in, int num_nodes_in, int num_edges_in)
      : closed(closed_in), num_nodes(num_nodes_in), num_edges(num_edges_in) {
    DRAKE_THROW_UNLESS(num_nodes >= 2);
    DRAKE_THROW_UNLESS(num_edges == (closed ? num_nodes : num_nodes - 1));
    objects.reserve(num_edges);
    object_pointers.reserve(num_edges);
  }

  const bool closed{};
  const int num_nodes{};
  const int num_edges{};
  fcl::DynamicAABBTreeCollisionManagerd tree;
  std::vector<std::unique_ptr<fcl::CollisionObjectd>> objects;
  std::unordered_set<const fcl::CollisionObjectd*> object_pointers;
  std::optional<FilamentSelfContactFilter> self_contact_filter;
};

/* Each added filament to Geometries is indexed by a FilamentIndex. */
using FilamentIndex = TypeSafeIndex<class FilamentTag>;

/* A struct holding metadata of a filament edge. */
struct FilamentEdgeData {
  FilamentEdgeData(FilamentIndex filament_index_in, int edge_index_in)
      : filament_index(filament_index_in), edge_index(edge_index_in) {
    DRAKE_THROW_UNLESS(edge_index >= 0);
  }

  void write_to(fcl::CollisionObjectd* object) {
    static_assert(sizeof(FilamentIndex) + sizeof(int) == sizeof(intptr_t));
    const intptr_t data =
        (intptr_t{filament_index} << (sizeof(int) * 8)) | intptr_t{edge_index};
    static_assert(sizeof(intptr_t) == sizeof(void*));
    object->setUserData(reinterpret_cast<void*>(data));
  }

  static FilamentEdgeData read_from(const fcl::CollisionObjectd& object) {
    // A mask 0x00000000FFFFFFFF.
    constexpr intptr_t kEdgeIndexMask = (intptr_t{1} << (sizeof(int) * 8)) - 1;
    const intptr_t data = reinterpret_cast<intptr_t>(object.getUserData());
    const FilamentIndex filament_index(data >> (sizeof(int) * 8));
    const int edge_index = static_cast<int>(data & kEdgeIndexMask);
    return FilamentEdgeData(filament_index, edge_index);
  }

  const FilamentIndex filament_index{};
  const int edge_index{};
};

}  // namespace

/* Geometries::Impl class. */
class Geometries::Impl {
 public:
  Impl() = default;

  void AddFilamentGeometry(GeometryId id, const Filament& filament) {
    DRAKE_THROW_UNLESS(!is_filament(id));
    const Eigen::Matrix3Xd& node_pos = filament.node_pos();
    const Eigen::Matrix3Xd& edge_m1 = filament.edge_m1();
    const int num_nodes = node_pos.cols();
    const int num_edges = edge_m1.cols();

    FilamentData& filament_data =
        id_to_filament_data_
            .emplace(id, FilamentData(filament.closed(), num_nodes, num_edges))
            .first->second;
    filament_data.objects.resize(num_edges);

    filament_index_to_id_.push_back(id);
    FilamentIndex filament_index(ssize(filament_index_to_id_) - 1);

    for (int i = 0; i < num_edges; ++i) {
      const int ip1 = (i + 1) % num_nodes;
      const Vector3d node_i = node_pos.col(i);
      const Vector3d node_ip1 = node_pos.col(ip1);
      const double l = (node_ip1 - node_i).norm();
      const Vector3d t = (node_ip1 - node_i) / l;
      const Vector3d& m1 = edge_m1.col(i);
      math::internal::ThrowIfNotOrthonormal(t, m1, __func__);
      /* Rotation of the material frame. */
      Matrix3d R_WM;
      R_WM.col(0) = m1;
      R_WM.col(1) = t.cross(m1);
      R_WM.col(2) = t;
      /* Position of the material frame origin expressed in world frame. */
      const Vector3d p_WM = (node_i + node_ip1) / 2;

      std::unique_ptr<fcl::CollisionGeometryd> shape;
      const auto& cs = filament.cross_section();
      if (std::holds_alternative<Filament::CircularCrossSection>(cs)) {
        const auto& circ_cs = std::get<Filament::CircularCrossSection>(cs);
        shape = std::make_unique<fcl::Cylinderd>(circ_cs.diameter / 2, l);
      } else {
        const auto& rect_cs = std::get<Filament::RectangularCrossSection>(cs);
        shape = std::make_unique<fcl::Boxd>(rect_cs.width, rect_cs.height, l);
      }

      filament_data.objects[i] =
          std::make_unique<fcl::CollisionObjectd>(std::move(shape), R_WM, p_WM);

      FilamentEdgeData filament_edge_data(filament_index, i);
      filament_edge_data.write_to(filament_data.objects[i].get());

      filament_data.object_pointers.insert(filament_data.objects[i].get());
      filament_data.tree.registerObject(filament_data.objects[i].get());
    }
    filament_data.tree.setup();
    filament_data.self_contact_filter = FilamentSelfContactFilter(filament);
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
    for (int i = 0; i < num_edges; ++i) {
      const int ip1 = (i + 1) % num_nodes;
      const Vector3d node_i = q_WG.template segment<3>(3 * i);
      const Vector3d node_ip1 = q_WG.template segment<3>(3 * ip1);
      const double l = (node_ip1 - node_i).norm();
      const Vector3d t = (node_ip1 - node_i) / l;
      const Vector3d m1 = q_WG.template segment<3>(num_nodes * 3 + 3 * i);
      math::internal::ThrowIfNotOrthonormal(t, m1, __func__);
      /* Rotation of the material frame. */
      Matrix3d R_WM;
      R_WM.col(0) = m1;
      R_WM.col(1) = t.cross(m1);
      R_WM.col(2) = t;
      /* Position of the material frame origin expressed in world frame. */
      const Vector3d p_WM = (node_i + node_ip1) / 2;

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
        cylinder.computeLocalAABB();
      } else if (object.getNodeType() == fcl::GEOM_BOX) {
        DRAKE_ASSERT(dynamic_cast<fcl::Boxd*>(shape) != nullptr);
        auto& box = static_cast<fcl::Boxd&>(*shape);
        box.side[2] = l;
        box.computeLocalAABB();
      } else {
        DRAKE_UNREACHABLE();
      }
      object.setTransform(R_WM, p_WM);
      object.computeAABB();
    }
    filament_data.tree.update();
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
    }
    clone->filament_index_to_id_ = this->filament_index_to_id_;
    return clone;
  }

 private:
  std::unordered_map<GeometryId, FilamentData> id_to_filament_data_;
  std::vector<GeometryId> filament_index_to_id_;
};

/* Deleter for Geometries::Impl class. */
void Geometries::ImplDeleter::operator()(Geometries::Impl* ptr) {
  delete ptr;
}

/* Geometries class functions delegate to corresponding Impl functions. */
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

void Geometries::AddFilamentGeometry(GeometryId id, const Filament& filament) {
  DRAKE_DEMAND(impl_ != nullptr);
  impl_->AddFilamentGeometry(id, filament);
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

bool Geometries::is_filament(GeometryId id) const {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->is_filament(id);
}

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
