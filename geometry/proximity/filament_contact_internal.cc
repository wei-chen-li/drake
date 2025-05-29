#include "drake/geometry/proximity/filament_contact_internal.h"

#include <unordered_map>
#include <unordered_set>

#include <fcl/fcl.h>

#include "drake/common/overloaded.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace {

/* Helper function that creates a *deep* copy of the given collision object. */
std::unique_ptr<fcl::CollisionObjectd> CloneFclObject(
    const fcl::CollisionObjectd& object_source) {
  const auto& shape_source = *object_source.collisionGeometry();

  std::shared_ptr<fcl::CollisionGeometryd> shape_clone;
  /* Filaments consist of only cylinder or box shapes. */
  switch (shape_source.getNodeType()) {
    case fcl::GEOM_CYLINDER: {
      const auto& cylinder = dynamic_cast<const fcl::Cylinderd&>(shape_source);
      shape_clone =
          std::make_shared<fcl::Cylinderd>(cylinder.radius, cylinder.lz);
      break;
    }
    case fcl::GEOM_BOX: {
      const auto& box = dynamic_cast<const fcl::Boxd&>(shape_source);
      shape_clone = std::make_shared<fcl::Boxd>(box.side);
      break;
    }
    default:
      DRAKE_UNREACHABLE();
  }

  /* A copy of the geometry is passed to FCL, but CollisionObject's constructor
   resets that copy's local bounding box to fit the _instantiated_ shape. So we
   retain a pointer to the shape copy long enough after handing it off to FCL to
   fix it back up to its original AABB. */
  auto object_clone = std::make_unique<fcl::CollisionObjectd>(shape_clone);

  /* The source's local AABB may have been inflated if the underlying object is
   associated with a compliant hydroelastic shape with a non-zero margin;
   therefore the AABB that fits the shape may not be what we want. We can't tell
   simply by looking at the fcl object if this is the case, so, we'll simply
   copy the source's local AABB verbatim to preserve the effect. */
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

    for (int i = 0; i < num_edges; ++i) {
      const int ip1 = (i + 1) % num_nodes;
      const double l = (node_pos.col(ip1) - node_pos.col(i)).norm();
      const Vector3d t = (node_pos.col(ip1) - node_pos.col(i)) / l;
      const Vector3d& m1 = edge_m1.col(i);
      // Rotation of material frame.
      Matrix3d R_WM;
      R_WM.col(0) = m1;
      R_WM.col(1) = t.cross(m1);
      R_WM.col(2) = t;
      // Position of the material frame origin expressed in world frame.
      const Vector3d p_WM = (node_pos.col(ip1) - node_pos.col(i)) / 2;

      std::unique_ptr<fcl::CollisionGeometryd> geom;
      const auto& cs = filament.cross_section();
      if (std::holds_alternative<Filament::CircularCrossSection>(cs)) {
        const auto& circ_cs = std::get<Filament::CircularCrossSection>(cs);
        geom = std::make_unique<fcl::Cylinderd>(circ_cs.diameter / 2, l);
      } else {
        const auto& rect_cs = std::get<Filament::RectangularCrossSection>(cs);
        geom = std::make_unique<fcl::Boxd>(rect_cs.width, rect_cs.height, l);
      }

      filament_data.objects[i] =
          std::make_unique<fcl::CollisionObjectd>(std::move(geom), R_WM, p_WM);
      fcl::CollisionObjectd* object_pointer = filament_data.objects[i].get();

      filament_data.object_pointers.insert(object_pointer);
      filament_data.tree.registerObject(object_pointer);
    }
    filament_data.tree.setup();
  }

  void RemoveGeometry(GeometryId id) { id_to_filament_data_.erase(id); }

  bool is_filament(GeometryId id) const {
    return id_to_filament_data_.find(id) != id_to_filament_data_.end();
  }

  std::unique_ptr<Impl, ImplDeleter> Clone() const {
    auto clone = std::unique_ptr<Impl, ImplDeleter>(new Impl());
    for (const auto& pair : id_to_filament_data_) {
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
        fcl::CollisionObjectd* cloned_object_pointer =
            filament_data_clone.objects[i].get();

        filament_data_clone.object_pointers.insert(cloned_object_pointer);
        filament_data_clone.tree.registerObject(cloned_object_pointer);
      }
      filament_data_clone.tree.setup();
    }
    return clone;
  }

 private:
  std::unordered_map<GeometryId, FilamentData> id_to_filament_data_;
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

bool Geometries::is_filament(GeometryId id) const {
  DRAKE_DEMAND(impl_ != nullptr);
  return impl_->is_filament(id);
}

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
