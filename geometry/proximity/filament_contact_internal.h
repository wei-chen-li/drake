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

  /* Adds a filament geometry
   @param id        The unique identifier for the geometry.
   @param filament  The filament with information on number of nodes and edges,
                    and cross-section.
   @pre There is no previous geometry associated with `id`. */
  void AddFilamentGeometry(GeometryId id, const Filament& filament);

  /* Removes the geometry specified by `id`. No-op if no geometry exist with the
   provided `id`. */
  void RemoveGeometry(GeometryId id);

  /* Updates the node positions and edge m1 directors of the filament geometry
   associated with `id`.
   @param id    The identifier for the filament geometry.
   @param q_WG  The vector holding the node positions and the edge m1
                directors. Hence, the vector should have size
                (3 * num_nodes + 3 * num_edges).
   @pre The `id` is associated with a filament geometry.
   @pre `q_WG` has the correct size. */
  void UpdateFilamentConfigurationVector(
      GeometryId id, const Eigen::Ref<const Eigen::VectorXd>& q_WG);

  /* Returns true if a filament geometry with the given `id` exists. */
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
