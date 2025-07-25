#pragma once

#include <memory>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

/* A struct holding hydroelsatic parameters of a filament.  */
struct FilamentHydroelasticParameters {
  double hydroelastic_modulus{};
  double circumferential_resolution_hint{};
  double longitudinal_resolution_hint{};
  double margin{};

  /* Below lists the properties parsed, all under the "hydroelastic" group name:
   "compliance_type",
   "hydroelastic_modulus",
   ("circumferential_resolution_hint" and "longitudinal_resolution_hint") xor
   "resolution_hint",
   "margin". */
  static std::optional<FilamentHydroelasticParameters> Parse(
      const ProximityProperties& props);
};

class FilamentSegmentMeshedGeometry {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FilamentSegmentMeshedGeometry)

  FilamentSegmentMeshedGeometry() = default;

  /* Constructs a FilamentSegmentMeshedGeometry that can be used to generate
   volume mesh and pressure field for a filament segment.
   @param cross_section   Cross-section of the filament.
   @param segment_length  Nominal segment length of the filament. The mesh is
                          generated from this length and the later deformed
                          based on the actual segment length.
   @param params          Hydroelastic parameters for the filament.
   @pre `segment_length > 0`. */
  FilamentSegmentMeshedGeometry(
      const std::variant<Filament::CircularCrossSection,
                         Filament::RectangularCrossSection>& cross_section,
      double segment_length, const FilamentHydroelasticParameters& params);

  /* Makes volume mesh of a filament segment based on the two end node
   positions, tangent vectors, and m₁ directors. */
  std::unique_ptr<VolumeMesh<double>> MakeVolumeMesh(
      const Eigen::Ref<const Eigen::Vector3d>& node_0,
      const Eigen::Ref<const Eigen::Vector3d>& node_1,
      const Eigen::Ref<const Eigen::Vector3d>& t_0,
      const Eigen::Ref<const Eigen::Vector3d>& t_1,
      const Eigen::Ref<const Eigen::Vector3d>& m1_0,
      const Eigen::Ref<const Eigen::Vector3d>& m1_1) const;

  /* Makes soft geometry of a filament segment based on the two end node
   positions, tangent vectors, and m₁ directors.
   @pre `hydroelastic_modulus.has_value()` at construction. */
  hydroelastic::SoftGeometry MakeSoftGeometry(
      const Eigen::Ref<const Eigen::Vector3d>& node_0,
      const Eigen::Ref<const Eigen::Vector3d>& node_1,
      const Eigen::Ref<const Eigen::Vector3d>& t_0,
      const Eigen::Ref<const Eigen::Vector3d>& t_1,
      const Eigen::Ref<const Eigen::Vector3d>& m1_0,
      const Eigen::Ref<const Eigen::Vector3d>& m1_1) const;

 private:
  Eigen::Matrix3Xd p_CVs_;
  int num_cross_sections_{};
  std::vector<VolumeElement> elements_;
  std::optional<std::vector<double>> pressures_;
  std::optional<Bvh<Obb, VolumeMesh<double>>> bvh_;
};

class FilamentMeshedGeometry {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FilamentMeshedGeometry)

  /* Constructs a FilamentMeshedGeometry that can be used to generate volume
   mesh and pressure field for any filament segment.
   @param filament  The reference filament with cross-section and average edge
                    length information.
   @param params    Hydroelastic parameters for the filament.
   @pre `params.hydroelastic_modulus > 0`.
   @pre `params.circumferential_resolution_hint > 0`.
   @pre `params.longitudinal_resolution_hint > 0`.
   @pre `params.margin >= 0`. */
  FilamentMeshedGeometry(const Filament& filament,
                         const FilamentHydroelasticParameters& params);

  /* Updates the node positions and edge m1 directors of the filament geometry.
   @param q_WG  The vector holding the node positions and the edge m1
                directors. Hence, the vector should have size
                (3 * num_nodes + 3 * num_edges).
   @pre `q_WG` has the correct size. */
  void UpdateConfigurationVector(const Eigen::Ref<const Eigen::VectorXd>& q_WG);

  /* Makes the hydroelastic soft geometry for the edge with index `edge_index`
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
  FilamentSegmentMeshedGeometry segment_;

  Eigen::Matrix3Xd node_pos_;
  Eigen::Matrix3Xd edge_m1_;
};

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
