#include "drake/geometry/proximity/filament_meshed_geometry.h"

#include <tuple>

#include "drake/common/overloaded.h"
#include "drake/geometry/proximity/meshing_utilities.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/math/axis_angle.h"
#include "drake/math/frame_transport.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/math/unit_vector.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {

using Eigen::Vector3d;

namespace {

/* Meshes the circular cross-section into vertices and triangles.
 @returns A tuple (p_CVs, triangles).
          p_CVs      The vertices of a cross-section in the cross-section frame.
          triangles  The connectivity of the vertices into triangles. */
std::tuple<Eigen::Matrix3Xd, Eigen::Matrix3Xi> MakeCrossSectionMesh(
    const Filament::CircularCrossSection& cross_section, double resolution_hint,
    double hydroelastic_margin) {
  DRAKE_THROW_UNLESS(cross_section.diameter > 0);
  DRAKE_THROW_UNLESS(resolution_hint > 0);
  DRAKE_THROW_UNLESS(hydroelastic_margin >= 0);
  const double d = cross_section.diameter + 2 * hydroelastic_margin;

  int num_thetas = M_PI * d / resolution_hint;
  num_thetas = std::max(num_thetas, 8);

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
    double resolution_hint, double hydroelastic_margin) {
  DRAKE_THROW_UNLESS(cross_section.width > 0 && cross_section.height > 0);
  DRAKE_THROW_UNLESS(resolution_hint > 0);
  DRAKE_THROW_UNLESS(hydroelastic_margin >= 0);
  const double w = cross_section.width + 2 * hydroelastic_margin;
  const double h = cross_section.height + 2 * hydroelastic_margin;
  if (w / h < 2.0 / 3.0) {
    // A thin-tall rectangle.
    Eigen::Matrix3Xd p_CVs(3, 8);
    const double ma = h - w;
    // clang-format off
    p_CVs <<     0,    0,  w/2, w/2, w/2, -w/2, -w/2, -w/2,
             -ma/2, ma/2, -h/2,   0, h/2,  h/2,    0, -h/2,
                 0,    0,    0,   0,   0,    0,    0,    0;
    // clang-format on
    Eigen::Matrix3Xi triangles(3, 8);
    // clang-format off
    triangles << 0, 0, 1, 1, 1, 0, 0, 0,
                 2, 3, 3, 4, 5, 1, 6, 7,
                 3, 1, 4, 5, 6, 6, 7, 2;
    // clang-format on
    return {std::move(p_CVs), std::move(triangles)};
  } else if (w / h <= 3.0 / 2.0) {
    // A rectangle close to a square.
    Eigen::Matrix3Xd p_CVs(3, 5);
    // clang-format off
    p_CVs << 0,  w/2, w/2, -w/2, -w/2,
             0, -h/2, h/2,  h/2, -h/2,
             0,    0,   0,    0,    0;
    // clang-format on
    Eigen::Matrix3Xi triangles(3, 4);
    // clang-format off
    triangles << 0, 0, 0, 0,
                 1, 2, 3, 4,
                 2, 3, 4, 1;
    // clang-format on
    return {std::move(p_CVs), std::move(triangles)};
  } else {
    // A fat-short rectangle.
    Filament::RectangularCrossSection cs = cross_section;
    std::swap(cs.width, cs.height);
    auto [p_CVs, triangles] =
        MakeCrossSectionMesh(cs, resolution_hint, hydroelastic_margin);
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
    double segment_length, double resolution_hint, double hydroelastic_margin) {
  const auto [p_CVs, triangles] = std::visit(
      [&](auto&& cs) {
        return MakeCrossSectionMesh(cs, resolution_hint, hydroelastic_margin);
      },
      cross_section);
  const int num_verts_per_cross_section = p_CVs.cols();

  int num_cross_sections = std::round(segment_length / resolution_hint) + 1;
  num_cross_sections = std::max(num_cross_sections, 2);

  std::vector<VolumeElement> elements;
  for (int k = 1; k < num_cross_sections; ++k) {
    const int offset0 = num_verts_per_cross_section * (k - 1);
    const int offset1 = num_verts_per_cross_section * k;
    for (int j = 0; j < triangles.cols(); ++j) {
      const Vector3<int> tri = triangles.col(j);
      Append(SplitTriangularPrismToTetrahedra(
                 offset0 + tri[0], offset0 + tri[1], offset0 + tri[2],
                 offset1 + tri[0], offset1 + tri[1], offset1 + tri[2]),
             &elements);
    }
  }
  return {std::move(p_CVs), num_cross_sections, std::move(elements)};
}

/* Makes the hydroelastic pressures for the mesh vertices. */
std::vector<double> MakeFilamentSegmentPressureField(
    const Filament::CircularCrossSection& cross_section,
    double hydroelastic_margin, double hydroelastic_modulus,
    const Eigen::Matrix3Xd& p_CVs, int num_cross_sections) {
  DRAKE_THROW_UNLESS(cross_section.diameter > 0);
  DRAKE_THROW_UNLESS(hydroelastic_margin >= 0);
  DRAKE_THROW_UNLESS(hydroelastic_modulus > 0);
  DRAKE_THROW_UNLESS(num_cross_sections > 0);
  std::vector<double> pressures;
  pressures.reserve(p_CVs.cols() * num_cross_sections);
  const double pressure_max = hydroelastic_modulus;
  const double pressure_min = -hydroelastic_modulus /
                              (cross_section.diameter / 2) *
                              hydroelastic_margin;
  for (int k = 0; k < num_cross_sections; ++k) {
    pressures.push_back(pressure_max);
    for (int i = 1; i < p_CVs.cols(); ++i) {
      pressures.push_back(pressure_min);
    }
  }
  return pressures;
}

/* Makes the hydroelastic pressures for the mesh vertices. */
std::vector<double> MakeFilamentSegmentPressureField(
    const Filament::RectangularCrossSection& cross_section,
    double hydroelastic_margin, double hydroelastic_modulus,
    const Eigen::Matrix3Xd& p_CVs, int num_cross_sections) {
  DRAKE_THROW_UNLESS(cross_section.width > 0 && cross_section.height > 0);
  DRAKE_THROW_UNLESS(hydroelastic_margin >= 0);
  DRAKE_THROW_UNLESS(hydroelastic_modulus > 0);
  DRAKE_THROW_UNLESS(num_cross_sections > 0);
  std::vector<double> pressures;
  pressures.reserve(p_CVs.cols() * num_cross_sections);
  const double pressure_max = hydroelastic_modulus;
  const double pressure_min =
      -hydroelastic_modulus /
      (std::min(cross_section.width, cross_section.height) / 2) *
      hydroelastic_margin;
  for (int k = 0; k < num_cross_sections; ++k) {
    for (int i = 0; i < p_CVs.cols(); ++i) {
      if (i <= 1 && (p_CVs.col(i)[0] == 0 || p_CVs.col(i)[1] == 0))
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
  DRAKE_THROW_UNLESS(0 <= alpha && alpha <= 1);
  Vector3d axis = vec_0.cross(vec_1);
  const double axis_norm = axis.norm();
  if (axis_norm < 1e-14)
    return LinearInterpolate(vec_0, vec_1, alpha).normalized();
  axis /= axis_norm;
  axis -= axis.dot(vec_0) * vec_0;
  axis -= axis.dot(vec_1) * vec_1;
  axis.normalize();
  const double angle = math::SignedAngleAroundAxis<double>(vec_0, vec_1, axis);
  return math::RotateAxisAngle<double>(vec_0, axis, alpha * angle);
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

FilamentSegmentMeshedGeometry::FilamentSegmentMeshedGeometry(
    const std::variant<Filament::CircularCrossSection,
                       Filament::RectangularCrossSection>& cross_section,
    double segment_length, double resolution_hint, double hydroelastic_margin,
    std::optional<double> hydroelastic_modulus) {
  DRAKE_THROW_UNLESS(segment_length > 0);
  DRAKE_THROW_UNLESS(resolution_hint > 0);
  DRAKE_THROW_UNLESS(hydroelastic_margin >= 0);
  DRAKE_THROW_UNLESS(!hydroelastic_modulus || hydroelastic_modulus > 0);
  std::tie(p_CVs_, num_cross_sections_, elements_) = MakeFilamentSegmentMesh(
      cross_section, segment_length, resolution_hint, hydroelastic_margin);
  std::visit(
      [&](auto&& cs) {
        if (hydroelastic_modulus.has_value()) {
          pressures_ = MakeFilamentSegmentPressureField(
              cs, hydroelastic_margin, *hydroelastic_modulus, p_CVs_,
              num_cross_sections_);
        }
      },
      cross_section);
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
  std::unique_ptr<VolumeMesh<double>> mesh =
      MakeVolumeMesh(node_0, node_1, t_0, t_1, m1_0, m1_1);
  std::vector<double> pressures = *pressures_;
  // TODO(wei-chen): Try to reduce the need to compute pressure gradient, and
  // other SoftMesh quantities every time this function is called.
  auto pressure_field = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      std::move(pressures), mesh.get());
  hydroelastic::SoftMesh soft_mesh(std::move(mesh), std::move(pressure_field));
  return hydroelastic::SoftGeometry(std::move(soft_mesh));
}

FilamentMeshedGeometry::FilamentMeshedGeometry(
    const Filament& filament, double resolution_hint,
    double hydroelastic_margin, std::optional<double> hydroelastic_modulus)
    : closed_(filament.closed()),
      num_nodes_(filament.node_pos().cols()),
      num_edges_(filament.edge_m1().cols()),
      node_pos_(filament.node_pos()),
      edge_m1_(filament.edge_m1()) {
  DRAKE_THROW_UNLESS(resolution_hint > 0);
  DRAKE_THROW_UNLESS(hydroelastic_margin >= 0);
  DRAKE_THROW_UNLESS(!hydroelastic_modulus || *hydroelastic_modulus > 0);
  double length = 0;
  for (int i = 0; i < num_edges_; ++i) {
    const int ip1 = (i + 1) % num_nodes_;
    length +=
        (filament.node_pos().col(ip1) - filament.node_pos().col(i)).norm();
  }
  segment_ = FilamentSegmentMeshedGeometry(
      filament.cross_section(), length / num_edges_, resolution_hint,
      hydroelastic_margin, hydroelastic_modulus);
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
