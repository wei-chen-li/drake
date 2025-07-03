#include "drake/geometry/proximity/filament_contact_internal.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity_engine.h"
#include "drake/geometry/proximity_properties.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {
namespace {

using Eigen::Vector3d;
using Eigen::VectorXd;
using testing::Each;
using testing::UnorderedElementsAre;

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
  engine.UpdateFilamentConfigurationVector({{id, q_WG}});

  /* Edge 0 and edge 2 now have a centerline distance of 0.8e-3. With a circular
   cross-section diameter of 1e-3, there should be a contact with 0.2e-3
   penetration depth. */
  engine.ComputeFilamentContact(&filament_contact);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);

  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A(), id);
  EXPECT_EQ(pair.id_B(), id);
  ASSERT_EQ(pair.num_contact_points(), 1);
  EXPECT_EQ(pair.contact_edge_indexes_A()[0], 0);
  EXPECT_EQ(pair.contact_edge_indexes_B()[0], 2);
  constexpr double kTol = 1e-10;
  EXPECT_TRUE(CompareMatrices(pair.p_WCs()[0], Vector3d(0, 0, 0), kTol));
  EXPECT_TRUE(CompareMatrices(pair.nhats_BA_W()[0], Vector3d(0, 0, -1), kTol));
  EXPECT_NEAR(pair.signed_distances()[0], -0.2e-3, kTol);

  EXPECT_THAT(filament_contact.contact_edges(id), UnorderedElementsAre(0, 2));
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

  engine.UpdateFilamentConfigurationVector({{id_A, q_WA}, {id_B, q_WB}});

  /* The two filaments now have a centerline distance of 0.8e-3. With a circular
   cross-section diameter of 1e-3, there should be a contact with 0.2e-3
   penetration depth. */
  engine.ComputeFilamentContact(&filament_contact);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);

  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A(), id_A);
  EXPECT_EQ(pair.id_B(), id_B);
  ASSERT_EQ(pair.num_contact_points(), 1);
  EXPECT_EQ(pair.contact_edge_indexes_A()[0], 1);
  EXPECT_EQ(pair.contact_edge_indexes_B()[0], 0);
  constexpr double kTol = 1e-10;
  EXPECT_TRUE(CompareMatrices(pair.p_WCs()[0], Vector3d(0, 0, 0), kTol));
  EXPECT_TRUE(CompareMatrices(pair.nhats_BA_W()[0], Vector3d(0, 0, -1), kTol));
  EXPECT_NEAR(pair.signed_distances()[0], -0.2e-3, kTol);

  EXPECT_THAT(filament_contact.contact_edges(id_A), UnorderedElementsAre(1));
  EXPECT_THAT(filament_contact.contact_edges(id_B), UnorderedElementsAre(0));
}

GTEST_TEST(FilamentContactInternalTest, FilamentRigidContact) {
  ProximityEngine<double> engine;

  math::RigidTransformd X_WG1(math::RotationMatrixd(), Vector3d(-0.6, 0, 0.3));
  GeometryId id_G1 = GeometryId::get_new_id();
  engine.AddDynamicGeometry(Sphere(0.3), X_WG1, id_G1);

  math::RigidTransformd X_WG2(math::RotationMatrixd(), Vector3d(0.6, 0, 0.3));
  GeometryId id_G2 = GeometryId::get_new_id();
  engine.AddAnchoredGeometry(Sphere(0.3), X_WG2, id_G2);

  const bool closed = false;
  const double d = 1e-3;
  const Filament::CircularCrossSection cross_section{.diameter = d};
  Eigen::Matrix3Xd node_positions(3, 3);
  node_positions.col(0) = Vector3d(-0.2, 0, 0);
  node_positions.col(1) = Vector3d(0.0, 0, 0);
  node_positions.col(2) = Vector3d(0.2, 0, 0);
  Filament filament(closed, node_positions, cross_section);
  GeometryId id_A = GeometryId::get_new_id();
  engine.AddFilamentGeometry(filament, id_A);

  /* The two shperes have no contact with the filament. */
  FilamentContact<double> filament_contact;
  engine.ComputeFilamentContact(&filament_contact);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 0);

  /* Extend the filament to the left to touch sphere 1. */
  node_positions.col(0) = Vector3d(-0.61, 0, 0);
  filament = Filament(closed, node_positions, cross_section);
  VectorXd q_WA(filament.node_pos().size() + filament.edge_m1().size());
  q_WA << filament.node_pos().reshaped(), filament.edge_m1().reshaped();
  engine.UpdateFilamentConfigurationVector({{id_A, q_WA}});

  engine.ComputeFilamentContact(&filament_contact);
  EXPECT_THAT(filament_contact.contact_edges(id_A), UnorderedElementsAre(0));
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);

  const FilamentContactGeometryPair<double>* pair =
      &filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair->id_A(), id_A);
  EXPECT_EQ(pair->id_B(), id_G1);
  ASSERT_EQ(pair->num_contact_points(), 1);
  EXPECT_EQ(pair->contact_edge_indexes_A()[0], 0);
  constexpr double kTol = 1e-10;
  EXPECT_TRUE(
      CompareMatrices(pair->p_WCs()[0], Vector3d(-0.6, 0, d / 4), kTol));
  EXPECT_TRUE(CompareMatrices(pair->nhats_BA_W()[0], Vector3d(0, 0, -1), kTol));
  EXPECT_NEAR(pair->signed_distances()[0], -d / 2, kTol);

  /* Extend the filament to the right to touch sphere 2. */
  node_positions.col(0) = Vector3d(-0.2, 0, 0);
  node_positions.col(2) = Vector3d(0.61, 0, 0);
  filament = Filament(closed, node_positions, cross_section);
  q_WA << filament.node_pos().reshaped(), filament.edge_m1().reshaped();
  engine.UpdateFilamentConfigurationVector({{id_A, q_WA}});

  engine.ComputeFilamentContact(&filament_contact);
  EXPECT_THAT(filament_contact.contact_edges(id_A), UnorderedElementsAre(1));
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);

  pair = &filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair->id_A(), id_A);
  EXPECT_EQ(pair->id_B(), id_G2);
  ASSERT_EQ(pair->num_contact_points(), 1);
  EXPECT_EQ(pair->contact_edge_indexes_A()[0], 1);
  EXPECT_TRUE(CompareMatrices(pair->p_WCs()[0], Vector3d(0.6, 0, d / 4), kTol));
  EXPECT_TRUE(CompareMatrices(pair->nhats_BA_W()[0], Vector3d(0, 0, -1), kTol));
  EXPECT_NEAR(pair->signed_distances()[0], -d / 2, kTol);
}

GTEST_TEST(FilamentContactInternalTest, FilamentHydroelasticSelfContact) {
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

  ProximityProperties props;
  const double resolution_hint = 0.5e-3;
  const double hydroelastic_modulus = 1e6;
  AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,
                                     &props);
  engine.AddFilamentGeometry(filament, id, props);

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
  engine.UpdateFilamentConfigurationVector({{id, q_WG}});

  /* Edge 0 and edge 2 now have a centerline distance of 0.8e-3. With a circular
   cross-section diameter of 1e-3, contact should occur. */
  engine.ComputeFilamentContact(&filament_contact);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);

  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A(), id);
  EXPECT_EQ(pair.id_B(), id);
  EXPECT_TRUE(pair.is_patch_contact());
  ASSERT_GE(pair.num_contact_points(), 1);
  EXPECT_THAT(pair.contact_edge_indexes_A(), Each(0));
  EXPECT_THAT(pair.contact_edge_indexes_B(), Each(2));

  EXPECT_THAT(filament_contact.contact_edges(id), UnorderedElementsAre(0, 2));
}

GTEST_TEST(FilamentContactInternalTest, FilamentFilamentHydroelasticContact) {
  ProximityEngine<double> engine;

  const bool closed = false;
  const Filament::CircularCrossSection cross_section{.diameter = 1e-3};

  Eigen::Matrix3Xd node_positions_A(3, 3);
  node_positions_A.col(0) = Vector3d(-0.3, 0, -1e-3);
  node_positions_A.col(1) = Vector3d(-0.1, 0, -1e-3);
  node_positions_A.col(2) = Vector3d(+0.1, 0, -1e-3);
  Filament filament_A(closed, node_positions_A, cross_section);
  GeometryId id_A = GeometryId::get_new_id();

  ProximityProperties props;
  const double resolution_hint = 0.5e-3;
  const double hydroelastic_modulus = 1e6;
  AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,
                                     &props);
  engine.AddFilamentGeometry(filament_A, id_A, props);

  Eigen::Matrix3Xd node_positions_B(3, 3);
  node_positions_B.col(0) = Vector3d(0, -0.1, +1e-3);
  node_positions_B.col(1) = Vector3d(0, +0.1, +1e-3);
  node_positions_B.col(2) = Vector3d(0, +0.3, +1e-3);
  Filament filament_B(closed, node_positions_B, cross_section);
  GeometryId id_B = GeometryId::get_new_id();
  engine.AddFilamentGeometry(filament_B, id_B, props);

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

  engine.UpdateFilamentConfigurationVector({{id_A, q_WA}, {id_B, q_WB}});

  /* The two filaments now have a centerline distance of 0.8e-3. With a circular
   cross-section diameter of 1e-3, contact should occur. */
  engine.ComputeFilamentContact(&filament_contact);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);

  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A(), id_A);
  EXPECT_EQ(pair.id_B(), id_B);
  EXPECT_TRUE(pair.is_patch_contact());
  ASSERT_GE(pair.num_contact_points(), 1);
  EXPECT_THAT(pair.contact_edge_indexes_A(), Each(1));
  EXPECT_THAT(pair.contact_edge_indexes_B(), Each(0));

  EXPECT_THAT(filament_contact.contact_edges(id_A), UnorderedElementsAre(1));
  EXPECT_THAT(filament_contact.contact_edges(id_B), UnorderedElementsAre(0));
}

GTEST_TEST(FilamentContactInternalTest, FilamentRigidHydroelasticContact) {
  ProximityEngine<double> engine;
  ProximityProperties props;

  math::RigidTransformd X_WG1(math::RotationMatrixd(), Vector3d(-0.6, 0, 0.3));
  GeometryId id_G1 = GeometryId::get_new_id();
  double resolution_hint = 0.15;
  props = ProximityProperties();
  AddRigidHydroelasticProperties(resolution_hint, &props);
  engine.AddDynamicGeometry(Sphere(0.3), X_WG1, id_G1, props);

  math::RigidTransformd X_WG2(math::RotationMatrixd(), Vector3d(0.6, 0, 0.3));
  GeometryId id_G2 = GeometryId::get_new_id();
  double hydroelastic_modulus = 1e8;
  props = ProximityProperties();
  AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,
                                     &props);
  engine.AddAnchoredGeometry(Sphere(0.3), X_WG2, id_G2, props);

  const bool closed = false;
  const double d = 1e-3;
  const Filament::CircularCrossSection cross_section{.diameter = d};
  Eigen::Matrix3Xd node_positions(3, 3);
  node_positions.col(0) = Vector3d(-0.2, 0, 0);
  node_positions.col(1) = Vector3d(0.0, 0, 0);
  node_positions.col(2) = Vector3d(0.2, 0, 0);
  Filament filament(closed, node_positions, cross_section);
  GeometryId id_A = GeometryId::get_new_id();

  resolution_hint = 0.5e-3;
  hydroelastic_modulus = 1e6;
  props = ProximityProperties();
  AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,
                                     &props);
  engine.AddFilamentGeometry(filament, id_A, props);

  /* The two shperes have no contact with the filament. */
  FilamentContact<double> filament_contact;
  engine.ComputeFilamentContact(&filament_contact);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 0);

  /* Extend the filament to the left to touch sphere 1. */
  node_positions.col(0) = Vector3d(-0.61, 0, 0);
  filament = Filament(closed, node_positions, cross_section);
  VectorXd q_WA(filament.node_pos().size() + filament.edge_m1().size());
  q_WA << filament.node_pos().reshaped(), filament.edge_m1().reshaped();
  engine.UpdateFilamentConfigurationVector({{id_A, q_WA}});

  engine.ComputeFilamentContact(&filament_contact);
  EXPECT_THAT(filament_contact.contact_edges(id_A), UnorderedElementsAre(0));
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);

  const FilamentContactGeometryPair<double>* pair =
      &filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair->id_A(), id_A);
  EXPECT_EQ(pair->id_B(), id_G1);
  ASSERT_GE(pair->num_contact_points(), 1);
  EXPECT_THAT(pair->contact_edge_indexes_A(), Each(0));
  EXPECT_TRUE(std::all_of(
      pair->p_WCs().begin(), pair->p_WCs().end(), [d](const Vector3d& p_WC) {
        return (p_WC - Vector3d(-0.6, 0, d / 4)).norm() < 3 * d;
      }));

  /* Extend the filament to the right to touch sphere 2. */
  node_positions.col(0) = Vector3d(-0.2, 0, 0);
  node_positions.col(2) = Vector3d(0.61, 0, 0);
  filament = Filament(closed, node_positions, cross_section);
  q_WA << filament.node_pos().reshaped(), filament.edge_m1().reshaped();
  engine.UpdateFilamentConfigurationVector({{id_A, q_WA}});

  engine.ComputeFilamentContact(&filament_contact);
  EXPECT_THAT(filament_contact.contact_edges(id_A), UnorderedElementsAre(1));
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);

  pair = &filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair->id_A(), id_A);
  EXPECT_EQ(pair->id_B(), id_G2);
  ASSERT_GE(pair->num_contact_points(), 1);
  EXPECT_THAT(pair->contact_edge_indexes_A(), Each(1));
  EXPECT_TRUE(std::all_of(
      pair->p_WCs().begin(), pair->p_WCs().end(), [d](const Vector3d& p_WC) {
        return (p_WC - Vector3d(0.6, 0, d / 4)).norm() < 3 * d;
      }));
}

}  // namespace
}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
