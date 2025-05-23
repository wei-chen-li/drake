#include "drake/geometry/proximity/make_filament_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

template <typename T>
TriangleSurfaceMesh<T> MakeFilamentSurfaceMesh(const Filament& filament,
                                               double resolution_hint) {
  std::vector<Vector3<T>> vertices;
  std::vector<SurfaceTriangle> triangles;
  vertices.push_back(Vector3<T>(0, 0, 0));
  vertices.push_back(Vector3<T>(1, 0, 0));
  vertices.push_back(Vector3<T>(0, 1, 0));
  triangles.push_back(SurfaceTriangle(0, 1, 2));
  return TriangleSurfaceMesh<T>(std::move(triangles), std::move(vertices));
}

template TriangleSurfaceMesh<double> MakeFilamentSurfaceMesh<double>(
    const Filament&, double);
template TriangleSurfaceMesh<float> MakeFilamentSurfaceMesh<float>(
    const Filament&, double);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
