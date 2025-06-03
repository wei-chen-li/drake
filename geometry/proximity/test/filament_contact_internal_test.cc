#include "drake/geometry/proximity/filament_contact_internal.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
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

  const bool closed = true;
  const Filament::CircularCrossSection cross_section{.diameter = 1e-3};
  Filament filament(closed, node_positions, cross_section);
  GeometryId id = GeometryId::get_new_id();
  engine.AddFilamentGeometry(filament, id);

  /* Edge 0 and edge 2 have a centerline distance of 2e-3. With a circular
   cross-section diameter of 1e-3, there should be no contact. */
  FilamentContact<double> filament_contact;
  engine.ComputeFilamentContact(&filament_contact);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 0);

  /* Update the node positions. */
  node_positions.col(0) = Vector3d(-0.1, 0, -0.4e-3);
  node_positions.col(1) = Vector3d(+0.1, 0, -0.4e-3);
  node_positions.col(2) = Vector3d(0, +0.1, +0.4e-3);
  node_positions.col(3) = Vector3d(0, -0.1, +0.4e-3);
  filament = Filament(closed, node_positions, cross_section);
  VectorXd q_WG(filament.node_pos().size() + filament.edge_m1().size());
  q_WG << filament.node_pos().reshaped(), filament.edge_m1().reshaped();
  std::unordered_map<GeometryId, VectorXd> q_WGs = {{id, q_WG}};
  engine.UpdateFilamentConfigurationVector(q_WGs);

  /* Edge 0 and edge 2 now have a centerline distance of 0.8e-3. With a circular
   cross-section diameter of 1e-3, there should be a contact with 0.2e-3
   penetration depth. */
  engine.ComputeFilamentContact(&filament_contact);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);

  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A(), id);
  EXPECT_EQ(pair.id_B(), id);
  ASSERT_EQ(pair.num_contacts(), 1);
  EXPECT_EQ(pair.contact_edge_indexes_A()[0], 0);
  EXPECT_EQ(pair.contact_edge_indexes_B()[0], 2);
  constexpr double kTol = 1e-10;
  EXPECT_TRUE(CompareMatrices(pair.p_WCs()[0], Vector3d(0, 0, 0), kTol));
  EXPECT_TRUE(CompareMatrices(pair.nhats_BA_W()[0], Vector3d(0, 0, -1), kTol));
  EXPECT_NEAR(pair.signed_distances()[0], -0.2e-3, kTol);
}

GTEST_TEST(FilamentContactInternalTest, FilamentFilamentContact) {
  ProximityEngine<double> engine;

  const bool closed = false;
  const Filament::CircularCrossSection cross_section{.diameter = 1e-3};

  Eigen::Matrix3Xd node_positions_A(3, 3);
  node_positions_A.col(0) = Vector3d(-0.3, 0, -1e-3);
  node_positions_A.col(1) = Vector3d(-0.1, 0, -1e-3);
  node_positions_A.col(2) = Vector3d(+0.1, 0, -1e-3);
  Filament filament_A(closed, node_positions_A, cross_section);
  GeometryId id_A = GeometryId::get_new_id();
  engine.AddFilamentGeometry(filament_A, id_A);

  Eigen::Matrix3Xd node_positions_B(3, 3);
  node_positions_B.col(0) = Vector3d(0, -0.1, +1e-3);
  node_positions_B.col(1) = Vector3d(0, +0.1, +1e-3);
  node_positions_B.col(2) = Vector3d(0, +0.3, +1e-3);
  Filament filament_B(closed, node_positions_B, cross_section);
  GeometryId id_B = GeometryId::get_new_id();
  engine.AddFilamentGeometry(filament_B, id_B);

  /* The two filaments have a centerline distance of 2e-3. With a circular
   * cross-section diameter of 1e-3, there should be no contact. */
  FilamentContact<double> filament_contact;
  engine.ComputeFilamentContact(&filament_contact);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 0);

  /* Update the node positions. */
  node_positions_A.col(1) = Vector3d(-0.1, 0, -0.4e-3);
  node_positions_A.col(2) = Vector3d(+0.1, 0, -0.4e-3);
  filament_A = Filament(closed, node_positions_A, cross_section);
  VectorXd q_WA(filament_A.node_pos().size() + filament_A.edge_m1().size());
  q_WA << filament_A.node_pos().reshaped(), filament_A.edge_m1().reshaped();

  node_positions_B.col(0) = Vector3d(0, -0.1, +0.4e-3);
  node_positions_B.col(1) = Vector3d(0, +0.1, +0.4e-3);
  filament_B = Filament(closed, node_positions_B, cross_section);
  VectorXd q_WB(filament_B.node_pos().size() + filament_B.edge_m1().size());
  q_WB << filament_B.node_pos().reshaped(), filament_B.edge_m1().reshaped();

  std::unordered_map<GeometryId, VectorXd> q_WGs = {{id_A, q_WA}, {id_B, q_WB}};
  engine.UpdateFilamentConfigurationVector(q_WGs);

  /* The two filaments now have a centerline distance of 0.8e-3. With a circular
   cross-section diameter of 1e-3, there should be a contact with 0.2e-3
   penetration depth. */
  engine.ComputeFilamentContact(&filament_contact);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);

  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A(), id_A);
  EXPECT_EQ(pair.id_B(), id_B);
  ASSERT_EQ(pair.num_contacts(), 1);
  EXPECT_EQ(pair.contact_edge_indexes_A()[0], 1);
  EXPECT_EQ(pair.contact_edge_indexes_B()[0], 0);
  constexpr double kTol = 1e-10;
  EXPECT_TRUE(CompareMatrices(pair.p_WCs()[0], Vector3d(0, 0, 0), kTol));
  EXPECT_TRUE(CompareMatrices(pair.nhats_BA_W()[0], Vector3d(0, 0, -1), kTol));
  EXPECT_NEAR(pair.signed_distances()[0], -0.2e-3, kTol);
}

}  // namespace
}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
