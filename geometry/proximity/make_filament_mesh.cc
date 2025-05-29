#include "drake/geometry/proximity/make_filament_mesh.h"

#include <vector>

#include "drake/common/overloaded.h"
#include "drake/math/frame_transport.h"

namespace drake {
namespace geometry {
namespace internal {

namespace {

template <typename T>
void AddCrossSection(std::vector<SurfaceTriangle>* triangles,
                     std::vector<Vector3<T>>* vertices_W,
                     const Eigen::Matrix3X<T>& cs_vertices_C,
                     const Eigen::Ref<const Vector3<T>>& pos,
                     const Eigen::Ref<const Vector3<T>>& t,
                     const Eigen::Ref<const Vector3<T>>& m1) {
  DRAKE_DEMAND(triangles != nullptr);
  DRAKE_DEMAND(vertices_W != nullptr);
  const int num_segments = cs_vertices_C.cols();
  DRAKE_DEMAND(ssize(*vertices_W) % num_segments == 0);
  Matrix3<T> R_WC;
  R_WC.col(0) = m1;
  R_WC.col(1) = t.cross(m1);
  R_WC.col(2) = t;
  for (int i = 0; i < num_segments; ++i) {
    vertices_W->emplace_back(R_WC * cs_vertices_C.col(i) + pos);
  }
  const int offset2 = ssize(*vertices_W) - num_segments;
  const int offset1 = offset2 - num_segments;
  if (offset1 < 0) return;
  for (int i = 0; i < num_segments; ++i) {
    const int ip1 = (i + 1) % num_segments;
    triangles->push_back(
        SurfaceTriangle(offset1 + i, offset1 + ip1, offset2 + i));
    triangles->push_back(
        SurfaceTriangle(offset1 + ip1, offset2 + ip1, offset2 + i));
  }
}

template <typename T>
void AddCaps(std::vector<SurfaceTriangle>* triangles,
             std::vector<Vector3<T>>* vertices, int num_segments,
             const Eigen::Ref<const Vector3<T>>& first_pos,
             const Eigen::Ref<const Vector3<T>>& last_pos) {
  DRAKE_DEMAND(triangles != nullptr);
  DRAKE_DEMAND(vertices != nullptr);
  DRAKE_DEMAND(ssize(*vertices) % num_segments == 0);
  vertices->push_back(first_pos);
  for (int i = 0; i < num_segments; ++i) {
    const int ip1 = (i + 1) % num_segments;
    triangles->push_back(SurfaceTriangle(ssize(*vertices) - 1, ip1, i));
  }
  vertices->push_back(last_pos);
  const int offset = ssize(*vertices) - 2 - num_segments;
  for (int i = 0; i < num_segments; ++i) {
    const int ip1 = (i + 1) % num_segments;
    triangles->push_back(
        SurfaceTriangle(ssize(*vertices) - 1, offset + i, offset + ip1));
  }
}

template <typename T>
void ConnectEnds(std::vector<SurfaceTriangle>* triangles,
                 const std::vector<Vector3<T>>& vertices, int num_segments) {
  DRAKE_DEMAND(triangles != nullptr);
  DRAKE_DEMAND(ssize(vertices) % num_segments == 0);
  const int offset1 = ssize(vertices) - num_segments;
  const int offset2 = 0;
  for (int i = 0; i < num_segments; ++i) {
    const int ip1 = (i + 1) % num_segments;
    triangles->push_back(
        SurfaceTriangle(offset1 + i, offset1 + ip1, offset2 + i));
    triangles->push_back(
        SurfaceTriangle(offset1 + ip1, offset2 + ip1, offset2 + i));
  }
}

}  // namespace

template <typename T>
TriangleSurfaceMesh<T> MakeFilamentSurfaceMesh(const Filament& filament) {
  DRAKE_THROW_UNLESS(filament.edge_m1().cols() ==
                     (filament.closed() ? filament.node_pos().cols()
                                        : filament.node_pos().cols() - 1));

  /* Find the cross-section vertices in the cross-section frame.  */
  const Eigen::Matrix3X<T> cs_vertices = std::visit(
      overloaded{
          [](const Filament::RectangularCrossSection& cs) {
            Eigen::Matrix3X<T> vertices(3, 4);
            const T w = cs.width;
            const T h = cs.height;
            // clang-format off
            vertices << -w / 2.0,  w / 2.0, w / 2.0, -w / 2.0,
                        -h / 2.0, -h / 2.0, h / 2.0,  h / 2.0,
                             0.0,      0.0,     0.0,      0.0;
            // clang-format on
            return vertices;
          },
          [](const Filament::CircularCrossSection& cs) {
            const int kN =
                20;  // TODO(wei-chen): Choose this number dynamically.
            Eigen::Matrix3X<T> vertices(3, kN);
            auto theta =
                VectorX<T>::LinSpaced(kN + 1, 0, 2 * M_PI).head(kN).array();
            vertices.row(0) = 0.5 * cs.diameter * cos(theta);
            vertices.row(1) = 0.5 * cs.diameter * sin(theta);
            vertices.row(2) = VectorX<T>::Zero(kN);
            return vertices;
          }},
      filament.cross_section());

  /* Find the tangent and m1 vector of the edge frames. */
  std::vector<Vector3<T>> edge_t;
  std::vector<Vector3<T>> edge_m1;
  std::vector<Vector3<T>> node_pos;
  node_pos.emplace_back(filament.node_pos().col(0).cast<T>());
  for (int i = 0; i < filament.edge_m1().cols(); ++i) {
    const int ip1 = (i + 1) % filament.node_pos().cols();
    const Vector3<T> edge =
        filament.node_pos().col(ip1).cast<T>() - node_pos.back();
    if (edge.norm() < 1e-10) continue;
    const Vector3<T> t = edge.normalized();
    edge_t.emplace_back(t);
    edge_m1.emplace_back(filament.edge_m1().col(i).cast<T>());
    edge_m1.back() -= edge_m1.back().dot(t) * t;
    edge_m1.back().normalize();
    if (ip1 != 0) node_pos.emplace_back(filament.node_pos().col(ip1).cast<T>());
  }
  DRAKE_ASSERT(ssize(edge_t) == ssize(edge_m1));
  DRAKE_ASSERT(ssize(edge_t) ==
               (filament.closed() ? ssize(node_pos) : ssize(node_pos) - 1));

  std::vector<SurfaceTriangle> triangles;
  std::vector<Vector3<T>> vertices;
  for (int i = 0; i < ssize(node_pos); ++i) {
    Vector3<T> t, m1;
    if (!filament.closed() && (i == 0 || i == ssize(node_pos) - 1)) {
      t = edge_t[i == 0 ? i : i - 1];
      m1 = edge_m1[i == 0 ? i : i - 1];
    } else {
      const int im1 = (i - 1 + ssize(edge_t)) % ssize(edge_t);
      t = (edge_t[im1] + edge_t[i]) / 2;
      Vector3<T> m1a, m1b;
      math::FrameTransport<T>(edge_t[im1], edge_m1[im1], t, m1a);
      math::FrameTransport<T>(edge_t[i], edge_m1[i], t, m1b);
      m1 = (m1a + m1b) / 2;
    }
    AddCrossSection<T>(&triangles, &vertices, cs_vertices, node_pos[i], t, m1);
  }
  if (!filament.closed()) {
    AddCaps<T>(&triangles, &vertices, cs_vertices.cols(), node_pos.front(),
               node_pos.back());
  } else {
    ConnectEnds<T>(&triangles, vertices, cs_vertices.cols());
  }
  return TriangleSurfaceMesh<T>(std::move(triangles), std::move(vertices));
}

template TriangleSurfaceMesh<double> MakeFilamentSurfaceMesh<double>(
    const Filament&);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
