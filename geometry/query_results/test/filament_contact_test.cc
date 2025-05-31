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

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
