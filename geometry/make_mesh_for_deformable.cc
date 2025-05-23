#include "drake/geometry/make_mesh_for_deformable.h"

#include "drake/common/drake_assert.h"
#include "drake/common/overloaded.h"
#include "drake/geometry/proximity/make_mesh_from_vtk.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

std::unique_ptr<VolumeMesh<double>> MakeMeshForDeformable(
    const Shape& shape, double resolution_hint) {
  DRAKE_DEMAND(resolution_hint > 0.0);
  return shape.Visit(overloaded{
      [](const Mesh& mesh) {
        return std::make_unique<VolumeMesh<double>>(
            MakeVolumeMeshFromVtk<double>(mesh));
      },
      [resolution_hint](const Sphere& sphere) {
        return std::make_unique<VolumeMesh<double>>(
            MakeSphereVolumeMesh<double>(
                sphere, resolution_hint,
                TessellationStrategy::kDenseInteriorVertices));
      },
      // TODO(xuchenhan-tri): As other shapes get supported, include their
      //  specific overrides here.
      [](const auto& unsupported) -> std::unique_ptr<VolumeMesh<double>> {
        throw std::logic_error(fmt::format(
            "MakeMeshForDeformable: We don't yet generate deformable meshes "
            "for {}.",
            unsupported));
      }});
}

std::unique_ptr<Filament> MakeFinerFilament(const Filament& filament,
                                            double resolution_hint) {
  DRAKE_THROW_UNLESS(resolution_hint > 0.0);
  const Eigen::Matrix3Xd& node_pos = filament.node_pos();
  const Eigen::Matrix3Xd& edge_m1 = filament.edge_m1();
  const int num_nodes = node_pos.cols();
  const int num_edges = edge_m1.cols();
  DRAKE_THROW_UNLESS(num_nodes >= 2);
  DRAKE_THROW_UNLESS(num_edges ==
                     (filament.closed() ? num_nodes : num_nodes - 1));

  std::vector<Eigen::Vector3d> finer_node_pos = {node_pos.col(0)};
  std::vector<Eigen::Vector3d> finer_edge_m1 = {};
  for (int i = 0; i < num_edges; ++i) {
    const int ip1 = (i + 1) % num_nodes;
    const Eigen::Vector3d edge_vector = node_pos.col(ip1) - node_pos.col(i);
    const double edge_length = edge_vector.norm();
    const int divisions = std::max(
        1, static_cast<int>(std::round(edge_length / resolution_hint)));
    for (int div = 1; div <= divisions; ++div) {
      finer_node_pos.emplace_back(node_pos.col(i) +
                                  edge_vector * div / divisions);
      finer_edge_m1.emplace_back(edge_m1.col(i));
    }
  }
  if (filament.closed()) finer_node_pos.pop_back();
  DRAKE_DEMAND(
      ssize(finer_edge_m1) ==
      (filament.closed() ? ssize(finer_node_pos) : ssize(finer_node_pos) - 1));

  Eigen::Matrix3Xd node_pos_new(3, finer_node_pos.size());
  for (int i = 0; i < ssize(finer_node_pos); ++i) {
    node_pos_new.col(i) = finer_node_pos[i];
  }
  Eigen::Matrix3Xd edge_m1_new(3, finer_edge_m1.size());
  for (int i = 0; i < ssize(finer_edge_m1); ++i) {
    edge_m1_new.col(i) = finer_edge_m1[i];
  }
  return std::make_unique<Filament>(filament.closed(), std::move(node_pos_new),
                                    std::move(edge_m1_new),
                                    filament.cross_section());
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
