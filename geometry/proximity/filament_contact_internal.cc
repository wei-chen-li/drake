#include "drake/geometry/proximity/filament_contact_internal.h"

#include <algorithm>

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

namespace {

/* Drake compiles FCL using hidden symbol visibility. To avoid visibility
 complaints from the compiler, we need to use hidden subclasses for any FCL data
 types used as member fields of Geometries::Impl. Note that FCL Objects on
 the stack are fine without worrying about hidden; it's only Impl member fields
 that cause trouble. */
class VectorOfPointersToFclDynamicAABBTreeCollisionManager
    : public std::vector<const fcl::DynamicAABBTreeCollisionManager<double>*> {
 public:
  VectorOfPointersToFclDynamicAABBTreeCollisionManager(
      std::vector<const fcl::DynamicAABBTreeCollisionManager<double>*> trees)
      : std::vector<const fcl::DynamicAABBTreeCollisionManager<double>*>(
            std::move(trees)) {}
};

}  // namespace

/* Geometries::Impl class. */
class Geometries::Impl {
 public:
  Impl(std::vector<const fcl::DynamicAABBTreeCollisionManager<double>*>
           rigid_body_trees)
      : rigid_body_trees_(rigid_body_trees) {};

  void AddFilamentGeometry(GeometryId id, const Filament& filament) {
    unused(id, filament);
  }

 private:
  const VectorOfPointersToFclDynamicAABBTreeCollisionManager rigid_body_trees_;
};

/* Deleter for Impl class. */
void Geometries::ImplDeleter::operator()(Geometries::Impl* ptr) {
  delete ptr;
}

/* Delegate calls to the Geometries class to the Impl class. */
Geometries::Geometries(
    std::vector<const fcl::DynamicAABBTreeCollisionManager<double>*>
        rigid_body_trees) {
  impl_ =
      std::unique_ptr<Impl, ImplDeleter>(new Impl(std::move(rigid_body_trees)));
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

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
