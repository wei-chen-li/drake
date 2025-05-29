#pragma once

#include <memory>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

class Geometries {
 public:
  Geometries();

  Geometries(const Geometries& other);
  Geometries& operator=(const Geometries& other);
  Geometries(Geometries&& other);
  Geometries& operator=(Geometries&& other);

  ~Geometries();

  void AddFilamentGeometry(GeometryId id, const Filament& filament);

  void RemoveGeometry(GeometryId id);

  bool is_filament(GeometryId id) const;

 private:
  /* Pointer to Impl. */
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
