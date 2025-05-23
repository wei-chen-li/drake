#pragma once

#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {

/* Creates a surface mesh for the given `filament`; the level of
 tessellation is guided by the `resolution_hint`.

 @param filament The filament for which a surface mesh is created.
 @return The triangulated surface mesh for the given filament.
 @tparam_double_only
*/
template <typename T>
TriangleSurfaceMesh<T> MakeFilamentSurfaceMesh(const Filament& filament);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
