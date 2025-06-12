#include <iostream>
#include <vector>

#include "drake/geometry/meshcat.h"
#include "drake/geometry/proximity/filament_soft_geometry.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace examples {
namespace {

using drake::geometry::Filament;
using drake::geometry::Meshcat;
using drake::geometry::MeshcatParams;
using drake::geometry::Rgba;
using drake::geometry::Sphere;
using drake::geometry::TriangleSurfaceMesh;
using drake::geometry::VolumeElement;
using drake::geometry::VolumeMesh;
using drake::geometry::internal::filament::FilamentSoftGeometry;
using Eigen::Vector3d;

void DrawMesh(Meshcat* meshcat, std::string_view path,
              const VolumeMesh<double>& mesh,
              const Rgba& rgba = Rgba(0.1, 0.1, 0.1, 1.0)) {
  DRAKE_THROW_UNLESS(meshcat != nullptr);
  std::vector<std::pair<Vector3d, Vector3d>> lines;
  const std::vector<Vector3d>& vertices = mesh.vertices();
  const std::vector<VolumeElement>& elements = mesh.tetrahedra();
  for (int k = 0; k < ssize(elements); ++k) {
    const VolumeElement& elem = elements[k];
    const Vector3d& vert0 = vertices.at(elem.vertex(0));
    const Vector3d& vert1 = vertices.at(elem.vertex(1));
    const Vector3d& vert2 = vertices.at(elem.vertex(2));
    const Vector3d& vert3 = vertices.at(elem.vertex(3));
    lines.emplace_back(vert0, vert1);
    lines.emplace_back(vert0, vert2);
    lines.emplace_back(vert0, vert3);
    lines.emplace_back(vert1, vert2);
    lines.emplace_back(vert1, vert3);
    lines.emplace_back(vert2, vert3);
  }
  Eigen::Matrix3Xd start(3, ssize(lines));
  Eigen::Matrix3Xd end(3, ssize(lines));
  for (int i = 0; i < ssize(lines); ++i) {
    start.col(i) = lines[i].first;
    end.col(i) = lines[i].second;
  }
  meshcat->SetLineSegments(path, start, end, 1.0, rgba);
}

void do_main() {
  Meshcat meshcat;

  const bool closed = true;
  const int kN = 10;
  Eigen::Matrix3Xd node_pos(3, kN);
  auto theta = Eigen::VectorXd::LinSpaced(kN + 1, 0, 2 * M_PI).head(kN).array();
  node_pos.row(0) = 0.05 * cos(theta);
  node_pos.row(1) = 0.05 * sin(theta);
  node_pos.row(2).setZero();

  Filament filament(closed, node_pos,
                    Filament::CircularCrossSection{.diameter = 0.01});
  const int num_edges = filament.edge_m1().cols();

  const double hydroelastic_pressure = 1e5;
  const double resolution_hint = 0.003;
  const double hydroelastic_margin = 0.005;

  FilamentSoftGeometry filament_soft_geometry(
      filament, hydroelastic_pressure, resolution_hint, hydroelastic_margin);

  for (int i = 0; i < num_edges; ++i) {
    auto soft_geometry = filament_soft_geometry.MakeSoftGeometryForEdge(i);
    auto soft_mesh = soft_geometry.soft_mesh();
    const std::string path = fmt::format("filament/edge_{}", i);
    const Rgba color = (i % 2 == 0) ? Rgba(0, 1, 1, 1) : Rgba(1, 0, 1, 1);
    DrawMesh(&meshcat, path, soft_mesh.mesh(), color);
  }

  std::cout << "Press enter to continue...";
  std::cin.get();
}

}  // namespace
}  // namespace examples
}  // namespace drake

int main() {
  drake::examples::do_main();
  return 0;
}
