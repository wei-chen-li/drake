#include "drake/geometry/query_results/filament_contact.h"

#include <gtest/gtest.h>

#include "drake/common/ssize.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;

const GeometryId kIdA = GeometryId::get_new_id();
const GeometryId kIdB = GeometryId::get_new_id();

GTEST_TEST(FilamentContactGeometryPairTest, EmptyFilamentFilamentContact) {
  FilamentContactGeometryPair<double> pair(kIdA, kIdB, {}, {}, {}, {}, {}, {},
                                           {});
  EXPECT_EQ(pair.id_A().get_value(), kIdA.get_value());
  EXPECT_EQ(pair.id_B().get_value(), kIdB.get_value());
  EXPECT_TRUE(pair.is_B_filament());
  EXPECT_EQ(pair.num_contacts(), 0);
}

GTEST_TEST(FilamentContactTest, AddFilamentFilamentContact) {
  const std::vector<Vector3d> p_WCs = {Vector3d(0, 0, 0)};
  const std::vector<Vector3d> nhats_BA_W = {Vector3d(0, 0, 1)};
  const std::vector<double> signed_distances = {-1e-3};
  const std::vector<int> contact_edge_indexes_A = {0};
  const std::vector<int> contact_edge_indexes_B = {0};
  Eigen::Matrix3Xd node_positions_A(3, 2);
  node_positions_A.col(0) = Vector3d(-2, 0, -0.01);
  node_positions_A.col(1) = Vector3d(1, 0, -0.01);
  Eigen::Matrix3Xd node_positions_B(3, 2);
  node_positions_B.col(0) = Vector3d(0, -3, 0.01);
  node_positions_B.col(1) = Vector3d(0, 1, 0.01);

  FilamentContact<double> filament_contact;
  filament_contact.AddFilamentFilamentContact(
      kIdA, kIdB, p_WCs, nhats_BA_W, signed_distances, contact_edge_indexes_A,
      contact_edge_indexes_B, node_positions_A, node_positions_B);
  EXPECT_EQ(filament_contact.contact_geometry_pairs().size(), 1);
  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A().get_value(), kIdA.get_value());
  EXPECT_EQ(pair.id_B().get_value(), kIdB.get_value());
  EXPECT_TRUE(pair.is_B_filament());
  EXPECT_EQ(pair.num_contacts(), 1);

  constexpr double kTol = 1e-16;
  EXPECT_NEAR(std::get<0>(pair.kinematic_coordinates_A()[0]), 1 / 3.0, kTol);
  EXPECT_NEAR(std::get<2>(pair.kinematic_coordinates_A()[0]), 2 / 3.0, kTol);
  EXPECT_TRUE(CompareMatrices(std::get<1>(pair.kinematic_coordinates_A()[0]),
                              Vector3d(0, -0.01, 0), kTol));
  EXPECT_NEAR(std::get<0>(pair.kinematic_coordinates_B()[0]), 1 / 4.0, kTol);
  EXPECT_NEAR(std::get<2>(pair.kinematic_coordinates_B()[0]), 3 / 4.0, kTol);
  EXPECT_TRUE(CompareMatrices(std::get<1>(pair.kinematic_coordinates_B()[0]),
                              Vector3d(-0.01, 0, 0), kTol));
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
