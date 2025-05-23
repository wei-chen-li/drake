#pragma once

#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {

/* Creates a surface mesh for the given `filament`; the level of
 tessellation is guided by the `resolution_hint`.

 @param filament         The filament for which a surface mesh is created.
 @param resolution_hint  The size of the mesh in guaranteed to be smaller than
                         resolution_hint.
 @return The triangulated surface mesh for the given filament.
 @tparam T double or float.
*/
template <typename T>
TriangleSurfaceMesh<T> MakeFilamentSurfaceMesh(const Filament& filament,
                                               double resolution_hint);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
