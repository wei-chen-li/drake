#include "drake/geometry/proximity/filament_meshed_geometry.h"

#include <tuple>

#include "drake/common/overloaded.h"
#include "drake/geometry/proximity/bvh_updater.h"
#include "drake/geometry/proximity/meshing_utilities.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/math/frame_transport.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

using Eigen::Vector3d;
using Eigen::Vector3i;

namespace {

/* Meshes the circular cross-section into vertices and triangles.
 @returns A tuple (p_CVs, triangles).
          p_CVs      The vertices of a cross-section in the cross-section frame.
          triangles  The connectivity of the vertices into triangles. */
std::tuple<Eigen::Matrix3Xd, Eigen::Matrix3Xi> MakeCrossSectionMesh(
    const Filament::CircularCrossSection& cross_section,
    const FilamentHydroelasticParameters& params) {
  DRAKE_THROW_UNLESS(cross_section.diameter > 0);
  const double d = cross_section.diameter + 2 * params.margin;

  const int num_thetas = std::max(
      3, static_cast<int>(
             std::round(M_PI * d / params.circumferential_resolution_hint)));

  const int num_verts_per_cross_section = 1 + num_thetas;
  Eigen::Matrix3Xd p_CVs(3, num_verts_per_cross_section);
  p_CVs.col(0).setZero();
  auto theta = Eigen::VectorXd::LinSpaced(num_thetas + 1, 0, 2 * M_PI)
                   .head(num_thetas)
                   .array();
  p_CVs.rightCols(num_thetas).row(0) = 0.5 * d * cos(theta);
  p_CVs.rightCols(num_thetas).row(1) = 0.5 * d * sin(theta);
  p_CVs.rightCols(num_thetas).row(2).setZero();

  Eigen::Matrix3Xi triangles(3, num_thetas);
  for (int i = 0; i < num_thetas; ++i) {
    const int ip1 = (i + 1) % num_thetas;
    triangles.col(i) = Vector3<int>(0, 1 + i, 1 + ip1);
  }

  return {std::move(p_CVs), std::move(triangles)};
}

/* Meshes the rectangular cross-section into vertices and triangles.
 @returns A tuple (p_CVs, triangles).
          p_CVs      The vertices of a cross-section in the cross-section frame.
          triangles  The connectivity of the vertices into triangles. */
std::tuple<Eigen::Matrix3Xd, Eigen::Matrix3Xi> MakeCrossSectionMesh(
    const Filament::RectangularCrossSection& cross_section,
    const FilamentHydroelasticParameters& params) {
  DRAKE_THROW_UNLESS(cross_section.width > 0 && cross_section.height > 0);
  const double w = cross_section.width + 2 * params.margin;
  const double h = cross_section.height + 2 * params.margin;
  if (w / h < 2.0 / 3.0) {
    /* A thin-tall rectangle. */
    const double ma = h - w;
    const int num_w_divs = std::max(
        1, static_cast<int>(
               std::round(w / params.circumferential_resolution_hint)));
    const int num_h_divs = std::max(
        2, static_cast<int>(
               std::round(h / params.circumferential_resolution_hint)));
    const int num_ma_divs = num_h_divs - 1;

    Eigen::Matrix3Xd p_CVs(3,
                           (num_ma_divs + 1) + (num_w_divs + num_h_divs) * 2);
    int j = -1;
    for (int k = 0; k < num_ma_divs + 1; ++k)
      p_CVs.col(++j) = Vector3d(0, -ma / 2 + ma * k / num_ma_divs, 0);
    for (int k = 0; k < num_h_divs; ++k)
      p_CVs.col(++j) = Vector3d(w / 2, -h / 2 + h * k / num_h_divs, 0);
    for (int k = 0; k < num_w_divs; ++k)
      p_CVs.col(++j) = Vector3d(w / 2 - w * k / num_w_divs, h / 2, 0);
    for (int k = 0; k < num_h_divs; ++k)
      p_CVs.col(++j) = Vector3d(-w / 2, h / 2 - h * k / num_h_divs, 0);
    for (int k = 0; k < num_w_divs; ++k)
      p_CVs.col(++j) = Vector3d(-w / 2 + w * k / num_w_divs, -h / 2, 0);

    const int num_triangles = (num_w_divs + num_ma_divs + num_h_divs) * 2;
    Eigen::Matrix3Xi triangles(3, num_triangles);
    j = -1;
    for (int k = 0; k < num_h_divs; ++k) {
      triangles.col(++j) = Vector3i(k, k + num_h_divs, k + num_h_divs + 1);
      if (k == num_h_divs - 1) continue;
      triangles.col(++j) = Vector3i(k, k + num_h_divs + 1, k + 1);
    }
    int offset = num_ma_divs + num_h_divs + 1;
    for (int k = 0; k < num_w_divs; ++k) {
      triangles.col(++j) = Vector3i(num_ma_divs, offset + k, offset + k + 1);
    }
    offset = num_ma_divs + num_h_divs + num_w_divs + 1;
    for (int k = num_h_divs - 1; k >= 0; --k) {
      triangles.col(++j) =
          Vector3i(k, offset + num_ma_divs - k, offset + num_ma_divs - k + 1);
      if (k == 0) continue;
      triangles.col(++j) = Vector3i(k, offset + num_ma_divs - k + 1, k - 1);
    }
    offset = num_ma_divs + num_h_divs + num_w_divs + num_h_divs + 1;
    for (int k = 0; k < num_w_divs; ++k) {
      if (k < num_w_divs - 1)
        triangles.col(++j) = Vector3i(0, offset + k, offset + k + 1);
      else
        triangles.col(++j) = Vector3i(0, offset + k, num_h_divs);
    }

    return {std::move(p_CVs), std::move(triangles)};
  } else if (w / h <= 3.0 / 2.0) {
    /* A rectangle close to a square. */
    const int num_w_divs = std::max(
        1, static_cast<int>(
               std::round(w / params.circumferential_resolution_hint)));
    const int num_h_divs = std::max(
        1, static_cast<int>(
               std::round(h / params.circumferential_resolution_hint)));
    Eigen::Matrix3Xd p_CVs(3, 1 + (num_w_divs + num_h_divs) * 2);
    p_CVs.col(0).setZero();
    int j = 0;
    for (int k = 0; k < num_h_divs; ++k)
      p_CVs.col(++j) = Vector3d(w / 2, -h / 2 + h * k / num_h_divs, 0);
    for (int k = 0; k < num_w_divs; ++k)
      p_CVs.col(++j) = Vector3d(w / 2 - w * k / num_w_divs, h / 2, 0);
    for (int k = 0; k < num_h_divs; ++k)
      p_CVs.col(++j) = Vector3d(-w / 2, h / 2 - h * k / num_h_divs, 0);
    for (int k = 0; k < num_w_divs; ++k)
      p_CVs.col(++j) = Vector3d(-w / 2 + w * k / num_w_divs, -h / 2, 0);

    const int num_triangles = (num_w_divs + num_h_divs) * 2;
    Eigen::Matrix3Xi triangles(3, num_triangles);
    for (int i = 0; i < num_triangles; ++i) {
      const int ip1 = (i + 1) % num_triangles;
      triangles.col(i) = Vector3i(0, 1 + i, 1 + ip1);
    }

    return {std::move(p_CVs), std::move(triangles)};
  } else {
    /* A fat-short rectangle. */
    Filament::RectangularCrossSection cs = cross_section;
    std::swap(cs.width, cs.height);
    auto [p_CVs, triangles] = MakeCrossSectionMesh(cs, params);
    p_CVs.row(0).swap(p_CVs.row(1));
    triangles.row(1).swap(triangles.row(2));
    return {std::move(p_CVs), std::move(triangles)};
  }
}

/* Makes the cross-section vertices and the elements representing the mesh of
 the filament segment.
 @returns A tuple (p_CVs, num_cross_sections, elements).
          p_CVs               The vertices of a cross-section in the
                              cross-section frame
          num_cross_sections  The number of cross section that make
                              up the mesh.
          elements            The connectivity of the mesh. */
std::tuple<Eigen::Matrix3Xd, int, std::vector<VolumeElement>>
MakeFilamentSegmentMesh(
    const std::variant<Filament::CircularCrossSection,
                       Filament::RectangularCrossSection>& cross_section,
    double segment_length, const FilamentHydroelasticParameters& params) {
  const auto [p_CVs, triangles] = std::visit(
      [&](auto&& cs) {
        return MakeCrossSectionMesh(cs, params);
      },
      cross_section);
  const int num_verts_per_cross_section = p_CVs.cols();

  int num_cross_sections = std::max(
      2, static_cast<int>(
             std::round(segment_length / params.longitudinal_resolution_hint)) +
             1);

  std::vector<VolumeElement> elements;
  for (int k = 1; k < num_cross_sections; ++k) {
    const int offset0 = num_verts_per_cross_section * (k - 1);
    const int offset1 = num_verts_per_cross_section * k;
    for (int j = 0; j < triangles.cols(); ++j) {
      const Vector3i tri = triangles.col(j);
      Append(SplitTriangularPrismToTetrahedra(
                 offset0 + tri[0], offset0 + tri[1], offset0 + tri[2],
                 offset1 + tri[0], offset1 + tri[1], offset1 + tri[2]),
             &elements);
    }
  }
  return {std::move(p_CVs), num_cross_sections, std::move(elements)};
}

/* Makes the hydroelastic pressures for the mesh vertices of a filament with
 circular cross section. */
std::vector<double> MakeFilamentSegmentPressureField(
    const Filament::CircularCrossSection& cross_section,
    const FilamentHydroelasticParameters& params, const Eigen::Matrix3Xd& p_CVs,
    int num_cross_sections) {
  DRAKE_THROW_UNLESS(cross_section.diameter > 0);
  DRAKE_THROW_UNLESS(num_cross_sections > 0);
  std::vector<double> pressures;
  pressures.reserve(p_CVs.cols() * num_cross_sections);
  const double pressure_max = params.hydroelastic_modulus;
  const double pressure_min = -params.hydroelastic_modulus /
                              (cross_section.diameter / 2) * params.margin;
  for (int k = 0; k < num_cross_sections; ++k) {
    /* The first vertex for each cross section is on the center and has maximum
     pressure. */
    pressures.push_back(pressure_max);
    /* All other vertices are on the perimeter and has minimum pressure. */
    for (int i = 1; i < p_CVs.cols(); ++i) {
      pressures.push_back(pressure_min);
    }
  }
  return pressures;
}

/* Makes the hydroelastic pressures for the mesh vertices of a filament with
 rectangular cross section. */
std::vector<double> MakeFilamentSegmentPressureField(
    const Filament::RectangularCrossSection& cross_section,
    const FilamentHydroelasticParameters& params, const Eigen::Matrix3Xd& p_CVs,
    int num_cross_sections) {
  DRAKE_THROW_UNLESS(cross_section.width > 0 && cross_section.height > 0);
  DRAKE_THROW_UNLESS(num_cross_sections > 0);
  std::vector<double> pressures;
  pressures.reserve(p_CVs.cols() * num_cross_sections);
  const double pressure_max = params.hydroelastic_modulus;
  const double pressure_min =
      -params.hydroelastic_modulus /
      (std::min(cross_section.width, cross_section.height) / 2) * params.margin;
  /* A flag to indicate if the medial axis is vertical or horizontal. */
  const bool vertical_ma = (p_CVs.col(0)[0] == 0.0);
  DRAKE_DEMAND((vertical_ma && p_CVs.col(0)[0] == 0.0) ||
               (!vertical_ma && p_CVs.col(0)[1] == 0.0));
  for (int k = 0; k < num_cross_sections; ++k) {
    /* The first few vertices for each cross section on the medial axis has
     maximum pressure. Remaining vertices are on the perimeter and have minimum
     pressure. */
    bool on_ma = true;
    for (int i = 0; i < p_CVs.cols(); ++i) {
      if (on_ma) {
        if (!((vertical_ma && p_CVs.col(i)[0] == 0.0) ||
              (!vertical_ma && p_CVs.col(i)[1] == 0.0))) {
          on_ma = false;
        }
      }
      if (on_ma)
        pressures.push_back(pressure_max);
      else
        pressures.push_back(pressure_min);
    }
  }
  return pressures;
}

Vector3d LinearInterpolate(const Eigen::Ref<const Vector3d>& vec_0,
                           const Eigen::Ref<const Vector3d>& vec_1,
                           double alpha) {
  DRAKE_THROW_UNLESS(0 <= alpha && alpha <= 1);
  return vec_0 * (1 - alpha) + vec_1 * alpha;
}

Vector3d SphericalInterpolate(const Eigen::Ref<const Vector3d>& vec_0,
                              const Eigen::Ref<const Vector3d>& vec_1,
                              double alpha) {
  DRAKE_ASSERT(std::abs(vec_0.norm() - 1) <= 1e-14);
  DRAKE_ASSERT(std::abs(vec_1.norm() - 1) <= 1e-14);
  DRAKE_THROW_UNLESS(0 <= alpha && alpha <= 1);
  const double cos_theta = vec_0.dot(vec_1);
  DRAKE_THROW_UNLESS(cos_theta != -1.0);
  if (cos_theta >= 0.9995) {
    return LinearInterpolate(vec_0, vec_1, alpha).normalized();
  }
  const double theta = acos(cos_theta);
  const double sin_theta = sin(theta);
  return sin((1 - alpha) * theta) / sin_theta * vec_0 +
         sin(alpha * theta) / sin_theta * vec_1;
}

/* Appends the columns of `new_vertices` into `mesh_vertices`. */
template <typename T>
void Append(const Eigen::Matrix3X<T>& new_vertices,
            std::vector<Vector3<T>>* mesh_vertices) {
  DRAKE_THROW_UNLESS(mesh_vertices != nullptr);
  for (int i = 0; i < new_vertices.cols(); ++i)
    mesh_vertices->emplace_back(new_vertices.col(i));
}

}  // namespace

std::optional<FilamentHydroelasticParameters>
FilamentHydroelasticParameters::Parse(const ProximityProperties& props) {
  if (!props.HasGroup(kHydroGroup)) return std::nullopt;

  const HydroelasticType compliance_type = props.GetPropertyOrDefault(
      kHydroGroup, kComplianceType, HydroelasticType::kCompliant);
  if (compliance_type != HydroelasticType::kCompliant) {
    throw std::invalid_argument(
        fmt::format("Filament only supports HydroelasticType::kCompliant for "
                    "the ('{}','{}') property",
                    kHydroGroup, kComplianceType));
  }

  FilamentHydroelasticParameters params;
  constexpr const char* kCircRezHint = "circumferential_resolution_hint";
  constexpr const char* kLongRezHint = "longitudinal_resolution_hint";

  std::string full_property_name =
      fmt::format("('{}', '{}')", kHydroGroup, kElastic);
  if (!props.HasProperty(kHydroGroup, kElastic)) {
    throw std::invalid_argument(
        fmt::format("Cannot create compliant filament; missing the {} property",
                    full_property_name));
  }
  params.hydroelastic_modulus =
      props.GetProperty<double>(kHydroGroup, kElastic);
  if (params.hydroelastic_modulus <= 0) {
    throw std::invalid_argument(
        fmt::format("The {} property must be positive", full_property_name));
  }

  full_property_name = fmt::format("('{}', '{}')", kHydroGroup, kCircRezHint);
  if (!props.HasProperty(kHydroGroup, kCircRezHint) &&
      !props.HasProperty(kHydroGroup, kRezHint)) {
    throw std::invalid_argument(
        fmt::format("Cannot create compliant filament; missing the {} property",
                    full_property_name));
  }
  params.circumferential_resolution_hint =
      props.HasProperty(kHydroGroup, kCircRezHint)
          ? props.GetProperty<double>(kHydroGroup, kCircRezHint)
          : props.GetProperty<double>(kHydroGroup, kRezHint);
  if (params.circumferential_resolution_hint <= 0) {
    throw std::invalid_argument(
        fmt::format("The {} property must be positive", full_property_name));
  }

  full_property_name = fmt::format("('{}', '{}')", kHydroGroup, kLongRezHint);
  if (!props.HasProperty(kHydroGroup, kLongRezHint) &&
      !props.HasProperty(kHydroGroup, kRezHint)) {
    throw std::invalid_argument(
        fmt::format("Cannot create compliant filament; missing the {} property",
                    full_property_name));
  }
  params.longitudinal_resolution_hint =
      props.HasProperty(kHydroGroup, kLongRezHint)
          ? props.GetProperty<double>(kHydroGroup, kLongRezHint)
          : props.GetProperty<double>(kHydroGroup, kRezHint);
  if (params.longitudinal_resolution_hint <= 0) {
    throw std::invalid_argument(
        fmt::format("The {} property must be positive", full_property_name));
  }

  full_property_name = fmt::format("('{}', '{}')", kHydroGroup, kMargin);
  params.margin = props.GetPropertyOrDefault(kHydroGroup, kMargin, 0.0);
  if (params.margin < 0) {
    throw std::invalid_argument(fmt::format(
        "The {} property must be non-negative", full_property_name));
  }
  return params;
}

FilamentSegmentMeshedGeometry::FilamentSegmentMeshedGeometry(
    const std::variant<Filament::CircularCrossSection,
                       Filament::RectangularCrossSection>& cross_section,
    double segment_length, const FilamentHydroelasticParameters& params) {
  DRAKE_THROW_UNLESS(segment_length > 0);
  DRAKE_THROW_UNLESS(params.hydroelastic_modulus > 0);
  DRAKE_THROW_UNLESS(params.circumferential_resolution_hint > 0);
  DRAKE_THROW_UNLESS(params.longitudinal_resolution_hint > 0);
  DRAKE_THROW_UNLESS(params.margin >= 0);
  /* Compute the mesh for a cross section and the connectivity of the
   tetrahedron mesh. */
  std::tie(p_CVs_, num_cross_sections_, elements_) =
      MakeFilamentSegmentMesh(cross_section, segment_length, params);
  /* Compute the pressure for the vertices. */
  std::visit(
      [&](auto&& cs) {
        pressures_ = MakeFilamentSegmentPressureField(cs, params, p_CVs_,
                                                      num_cross_sections_);
      },
      cross_section);
  /* Computes the reference bounding volume hierarchy for the volume mesh. The
   BVH will later be updated when volume mesh deforms. */
  std::unique_ptr<VolumeMesh<double>> mesh = MakeVolumeMesh(
      /* node_0 = */ Vector3d(0, 0, 0),
      /* node_1 = */ Vector3d(0, 0, segment_length),
      /* t_0 = */ Vector3d(0, 0, 1), /* t_1 = */ Vector3d(0, 0, 1),
      /* m1_0 = */ Vector3d(1, 0, 0), /* m1_1 = */ Vector3d(1, 0, 0));
  bvh_ = Bvh<Obb, VolumeMesh<double>>(*mesh);
}

std::unique_ptr<VolumeMesh<double>>
FilamentSegmentMeshedGeometry::MakeVolumeMesh(
    const Eigen::Ref<const Vector3d>& node_0,
    const Eigen::Ref<const Vector3d>& node_1,
    const Eigen::Ref<const Vector3d>& t_0,
    const Eigen::Ref<const Vector3d>& t_1,
    const Eigen::Ref<const Vector3d>& m1_0,
    const Eigen::Ref<const Vector3d>& m1_1) const {
  std::vector<Vector3d> vertices;
  for (int i = 0; i < num_cross_sections_; ++i) {
    const double alpha = static_cast<double>(i) / (num_cross_sections_ - 1);
    Vector3d pos = LinearInterpolate(node_0, node_1, alpha);
    Vector3d t = SphericalInterpolate(t_0, t_1, alpha);
    Vector3d m1 = SphericalInterpolate(m1_0, m1_1, alpha);
    m1 = (m1 - m1.dot(t) * t).normalized();
    /* Pose of cross-section frame in world frame. */
    math::RigidTransformd X_WC(
        math::RotationMatrixd::MakeFromOrthonormalColumns(m1, t.cross(m1), t),
        pos);
    Append(X_WC * p_CVs_, &vertices);
  }
  std::vector<VolumeElement> elements = elements_;
  return std::make_unique<VolumeMesh<double>>(std::move(elements),
                                              std::move(vertices));
}

hydroelastic::SoftGeometry FilamentSegmentMeshedGeometry::MakeSoftGeometry(
    const Eigen::Ref<const Eigen::Vector3d>& node_0,
    const Eigen::Ref<const Eigen::Vector3d>& node_1,
    const Eigen::Ref<const Eigen::Vector3d>& t_0,
    const Eigen::Ref<const Eigen::Vector3d>& t_1,
    const Eigen::Ref<const Eigen::Vector3d>& m1_0,
    const Eigen::Ref<const Eigen::Vector3d>& m1_1) const {
  DRAKE_THROW_UNLESS(pressures_.has_value());
  DRAKE_DEMAND(bvh_.has_value());

  /* Make volume mesh. */
  std::unique_ptr<VolumeMesh<double>> mesh =
      MakeVolumeMesh(node_0, node_1, t_0, t_1, m1_0, m1_1);

  /* Make pressure field. */
  std::vector<double> pressures = *pressures_;
  // TODO(wei-chen): Consider precomputing pressure gradient.
  auto pressure_field = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      std::move(pressures), mesh.get());

  /* Make BVH for volume mesh. We restrict the OBBs to have a common orientation
   `R_WB` for faster refitting. */
  auto bvh = std::make_unique<Bvh<Obb, VolumeMesh<double>>>(*bvh_);
  Eigen::Vector3d t = (node_1 - node_0).normalized();
  Eigen::Vector3d m1 = (m1_0 + m1_1).normalized();
  m1 = (m1 - m1.dot(t) * t).normalized();
  const math::RotationMatrixd R_WB =
      math::RotationMatrixd::MakeFromOrthonormalColumns(t, m1, t.cross(m1));
  BvhUpdater<VolumeMesh<double>>(mesh.get(), bvh.get()).Update(R_WB);

  hydroelastic::SoftMesh soft_mesh(std::move(mesh), std::move(pressure_field),
                                   std::move(bvh));
  return hydroelastic::SoftGeometry(std::move(soft_mesh));
}

FilamentMeshedGeometry::FilamentMeshedGeometry(
    const Filament& filament, const FilamentHydroelasticParameters& params)
    : closed_(filament.closed()),
      num_nodes_(filament.node_pos().cols()),
      num_edges_(filament.edge_m1().cols()),
      node_pos_(filament.node_pos()),
      edge_m1_(filament.edge_m1()) {
  double length = 0;
  for (int i = 0; i < num_edges_; ++i) {
    const int ip1 = (i + 1) % num_nodes_;
    length +=
        (filament.node_pos().col(ip1) - filament.node_pos().col(i)).norm();
  }
  segment_ = FilamentSegmentMeshedGeometry(filament.cross_section(),
                                           length / num_edges_, params);
}

void FilamentMeshedGeometry::UpdateConfigurationVector(
    const Eigen::Ref<const Eigen::VectorXd>& q_WG) {
  DRAKE_THROW_UNLESS(q_WG.size() == 3 * num_nodes_ + 3 * num_edges_);
  node_pos_ = Eigen::Map<const Eigen::Matrix3Xd>(q_WG.data(), 3, num_nodes_);
  const int offset = 3 * num_nodes_;
  edge_m1_ =
      Eigen::Map<const Eigen::Matrix3Xd>(q_WG.data() + offset, 3, num_edges_);
}

hydroelastic::SoftGeometry FilamentMeshedGeometry::MakeSoftGeometryForEdge(
    int edge_index) const {
  DRAKE_THROW_UNLESS(0 <= edge_index && edge_index < num_edges_);
  const int node0_index = edge_index;
  const int node1_index = (edge_index + 1) % num_nodes_;
  const auto [node_0, t_0, m1_0] = find_node_pos_t_m1(node0_index);
  const auto [node_1, t_1, m1_1] = find_node_pos_t_m1(node1_index);
  return segment_.MakeSoftGeometry(node_0, node_1, t_0, t_1, m1_0, m1_1);
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
FilamentMeshedGeometry::find_node_pos_t_m1(int node_index) const {
  DRAKE_THROW_UNLESS(0 <= node_index && node_index < num_nodes_);
  const Vector3d pos = node_pos_.col(node_index);
  Vector3d t;
  Vector3d m1;
  if (!closed_ && (node_index == 0 || node_index == num_nodes_ - 1)) {
    if (node_index == 0) {
      const int edge_index = node_index;
      std::tie(t, m1) = find_edge_t_m1(edge_index);
    } else {
      const int edge_index = node_index - 1;
      std::tie(t, m1) = find_edge_t_m1(edge_index);
    }
  } else {
    const int edge0_index = (node_index - 1 + num_edges_) % num_edges_;
    const int edge1_index = node_index;
    const auto [t_0, m1_0] = find_edge_t_m1(edge0_index);
    const auto [t_1, m1_1] = find_edge_t_m1(edge1_index);
    t = ((t_0 + t_1) / 2).normalized();
    Vector3d m1_c0, m1_c1;
    math::FrameTransport<double>(t_0, m1_0, t, m1_c0);
    math::FrameTransport<double>(t_1, m1_1, t, m1_c1);
    m1 = (m1_c0 + m1_c1) / 2;
    m1 = (m1 - m1.dot(t) * t).normalized();
  }
  return {pos, t, m1};
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d>
FilamentMeshedGeometry::find_edge_t_m1(int edge_index) const {
  DRAKE_THROW_UNLESS(0 <= edge_index && edge_index < num_edges_);
  const Vector3d m1 = edge_m1_.col(edge_index);

  const int node0_index = edge_index;
  const int node1_index = (edge_index + 1) % num_nodes_;
  const Vector3d t =
      (node_pos_.col(node1_index) - node_pos_.col(node0_index)).normalized();
  return {t, m1};
}

}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
