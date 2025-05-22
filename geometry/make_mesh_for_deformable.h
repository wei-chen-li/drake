#pragma once

#include <memory>

#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {

/* Returns a volume mesh suitable for deformable body simulation that
 discretizes the shape with the resolution provided at construction.
 @pre resolution_hint > 0.
 @throws std::exception if `shape` doesn't support a mesh discretization. */
std::unique_ptr<VolumeMesh<double>> MakeMeshForDeformable(
    const Shape& shape, double resolution_hint);

/* Returns a finer spatially discretized filament from the input filament.
 @pre resolution_hint > 0. */
std::unique_ptr<Filament> MakeFinerFilament(const Filament& filament,
                                            double resolution_hint);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
