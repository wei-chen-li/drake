#include "drake/geometry/proximity/filament_contact_internal.h"

#include <gtest/gtest.h>

#include "drake/geometry/proximity_engine.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {
namespace {

using Eigen::Vector3d;
using Eigen::VectorXd;

GTEST_TEST(FilamentContactInternalTest, FilamentSelfContact) {
  ProximityEngine<double> engine;

  Eigen::Matrix3Xd node_positions(3, 4);
  node_positions.col(0) = Vector3d(-0.1, 0, -1e-3);
  node_positions.col(1) = Vector3d(+0.1, 0, -1e-3);
  node_positions.col(2) = Vector3d(0, +0.1, +1e-3);
  node_positions.col(3) = Vector3d(0, -0.1, +1e-3);

  /* Edge 0 and edge 2 has a diatance of 2e-3, with a circular cross-section
   diameter of 1e-3, there should be no contact. */
  Filament filament(/* closed = */ true, node_positions,
                    Filament::CircularCrossSection{.diameter = 1e-3});
  GeometryId id = GeometryId::get_new_id();
  engine.AddFilamentGeometry(filament, id);

  FilamentContact<double> filament_contact;
  engine.ComputeFilamentContact(&filament_contact);
  EXPECT_TRUE(filament_contact.contact_geometry_pairs().empty());

  node_positions.col(0)[2] = -0.4e-3;
  node_positions.col(1)[2] = -0.4e-3;
  node_positions.col(2)[2] = +0.4e-3;
  node_positions.col(3)[2] = +0.4e-3;
  Eigen::Matrix3Xd edge_m1_directors(3, 4);
  edge_m1_directors.col(0) = Vector3d(0, 1, 0);
  edge_m1_directors.col(1) = Vector3d(1, 1, 0).normalized();
  edge_m1_directors.col(2) = Vector3d(1, 0, 0);
  edge_m1_directors.col(3) = Vector3d(1, 1, 0).normalized();
  VectorXd q_WG(node_positions.size() + edge_m1_directors.size());
  q_WG << node_positions.reshaped(), edge_m1_directors.reshaped();
  std::unordered_map<GeometryId, VectorXd> q_WGs = {{id, q_WG}};
  /* Edge 0 and edge 2 now has a diatance of 0.8e-3, with a circular
   cross-section diameter of 1e-3, there should be a contact. */
  engine.UpdateFilamentConfigurationVector(q_WGs);
  engine.ComputeFilamentContact(&filament_contact);
  EXPECT_EQ(filament_contact.contact_geometry_pairs().size(), 1);
}

}  // namespace
}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
