#include "drake/geometry/proximity/filament_contact_internal.h"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "drake/common/overloaded.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace {

/* Drake compiles FCL using hidden symbol visibility. To avoid visibility
 complaints from the compiler, we need to use hidden subclasses for any FCL data
 types used as member fields of Geometries::Impl. Note that FCL Objects on
 the stack are fine without worrying about hidden; it's only Impl member fields
 that cause trouble. */
class MapGeometryIdToDynamicAABBTreeCollisionManager
    : public std::unordered_map<GeometryId,
                                fcl::DynamicAABBTreeCollisionManagerd> {};
class MapGeometryIdToCollisionObjects
    : public std::unordered_map<
          GeometryId, std::vector<std::unique_ptr<fcl::CollisionObjectd>>> {};

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
    fcl::DynamicAABBTreeCollisionManagerd& tree =
        filament_trees.emplace(id, fcl::DynamicAABBTreeCollisionManagerd{})
            .first->second;
    std::vector<std::unique_ptr<fcl::CollisionObjectd>>& objects =
        filament_objects.emplace(id, num_edges).first->second;
    unused(num_nodes, tree, objects);

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

      objects[i] =
          std::make_unique<fcl::CollisionObjectd>(std::move(geom), R_WM, p_WM);
      tree.registerObject(objects[i].get());
    }
    tree.setup();
  }

  void RemoveGeometry(GeometryId id) {
    filament_trees.erase(id);
    filament_objects.erase(id);
  }

  bool is_filament(GeometryId id) const {
    return filament_trees.find(id) != filament_trees.end();
  };

 private:
  MapGeometryIdToDynamicAABBTreeCollisionManager filament_trees;
  MapGeometryIdToCollisionObjects filament_objects;
};

/* Deleter for Geometries::Impl class. */
void Geometries::ImplDeleter::operator()(Geometries::Impl* ptr) {
  delete ptr;
}

/* Geometries class functions delegate to corresponding Impl class function. */
Geometries::Geometries() {
  impl_ = std::unique_ptr<Impl, ImplDeleter>(new Impl());
}

Geometries& Geometries::operator=(Geometries&& other) {
  this->impl_ = std::move(other.impl_);
  return *this;
}

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
