#pragma once

#include <memory>
#include <variant>

#include "drake/common/drake_copyable.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

class FilamentSoftGeometrySegment {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FilamentSoftGeometrySegment)

  FilamentSoftGeometrySegment() = default;

  FilamentSoftGeometrySegment(
      const std::variant<Filament::CircularCrossSection,
                         Filament::RectangularCrossSection>& cross_section,
      double segment_length, double hydroelastic_margin, double resolution_hint,
      double hydroelastic_pressure);

  hydroelastic::SoftGeometry MakeSoftGeometry(
      const Eigen::Ref<const Eigen::Vector3d>& node_0,
      const Eigen::Ref<const Eigen::Vector3d>& node_1,
      const Eigen::Ref<const Eigen::Vector3d>& t_0,
      const Eigen::Ref<const Eigen::Vector3d>& t_1,
      const Eigen::Ref<const Eigen::Vector3d>& m1_0,
      const Eigen::Ref<const Eigen::Vector3d>& m1_1) const;

 private:
  std::unique_ptr<VolumeMesh<double>> MakeVolumeMesh(
      const Eigen::Ref<const Eigen::Vector3d>& node_0,
      const Eigen::Ref<const Eigen::Vector3d>& node_1,
      const Eigen::Ref<const Eigen::Vector3d>& t_0,
      const Eigen::Ref<const Eigen::Vector3d>& t_1,
      const Eigen::Ref<const Eigen::Vector3d>& m1_0,
      const Eigen::Ref<const Eigen::Vector3d>& m1_1) const;

  Eigen::Matrix3Xd p_CVs_;
  int num_cross_sections_{};
  std::vector<VolumeElement> elements_;
  std::vector<double> pressures_;
};

class FilamentSoftGeometry {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FilamentSoftGeometry)

  FilamentSoftGeometry(const Filament& filament, double hydroelastic_pressure,
                       double resolution_hint, double hydroelastic_margin = 0);

  /* Updates the node positions and edge m1 directors of the filament geometry.
   @param q_WG  The vector holding the node positions and the edge m1
                directors. Hence, the vector should have size
                (3 * num_nodes + 3 * num_edges).
   @pre `q_WG` has the correct size. */
  void UpdateConfigurationVector(const Eigen::Ref<const Eigen::VectorXd>& q_WG);

  /* Makes the hydroelastic soft geometry fro the edge with index `edge_index`
   whose configuration was prescribed in UpdateConfigurationVector(). */
  hydroelastic::SoftGeometry MakeSoftGeometryForEdge(int edge_index) const;

 private:
  std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
  find_node_pos_t_m1(int node_index) const;

  std::tuple<Eigen::Vector3d, Eigen::Vector3d> find_edge_t_m1(
      int edge_index) const;

  bool closed_;
  int num_nodes_;
  int num_edges_;
  FilamentSoftGeometrySegment segment_;

  Eigen::Matrix3Xd node_pos_;
  Eigen::Matrix3Xd edge_m1_;
};

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
