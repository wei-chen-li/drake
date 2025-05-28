#pragma once

#include <memory>
#include <vector>

#include <fcl/fcl.h>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

class Geometries {
 public:
  Geometries() = default;

  Geometries(std::vector<const fcl::DynamicAABBTreeCollisionManager<double>*>
                 rigid_body_trees);

  Geometries(const Geometries& other) = delete;
  Geometries& operator=(const Geometries& other) = delete;
  Geometries(Geometries&& other) = delete;

  Geometries& operator=(Geometries&& other);

  ~Geometries();

  void AddFilamentGeometry(GeometryId id, const Filament& filament);

 private:
  /* Unique pointer to Impl. */
  class Impl;
  struct ImplDeleter {
    void operator()(Impl*);
  };
  std::unique_ptr<Impl, ImplDeleter> impl_;
};

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
