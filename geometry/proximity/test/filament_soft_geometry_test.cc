#include "drake/geometry/proximity/filament_soft_geometry.h"

#include <optional>

#include <gtest/gtest.h>

namespace drake {
namespace geometry {
namespace internal {
namespace filament {
namespace {

using Eigen::VectorXd;

class FilamentSoftGeometryTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    bool closed = GetParam();
    Eigen::Matrix3Xd node_pos;
    if (closed) {
      const int kN = 10;
      node_pos.resize(3, kN);
      auto theta = VectorXd::LinSpaced(kN + 1, 0, 2 * M_PI).head(kN).array();
      node_pos.row(0) = 0.05 * cos(theta);
      node_pos.row(1) = 0.05 * sin(theta);
      node_pos.row(2).setZero();
    } else {
      const int kN = 10;
      node_pos.resize(3, kN);
      node_pos.row(0) = VectorXd::LinSpaced(kN, 0, 0.3);
      node_pos.row(1).setZero();
      node_pos.row(2).setZero();
    }
    Filament filament(closed, node_pos,
                      Filament::CircularCrossSection{.diameter = 0.01});
    num_edges_ = filament.edge_m1().cols();

    const double hydroelastic_pressure = 1e5;
    const double resolution_hint = 0.003;
    const double hydroelastic_margin = 0.005;

    filament_soft_geometry_ = FilamentSoftGeometry(
        filament, hydroelastic_pressure, resolution_hint, hydroelastic_margin);
  }

  int num_edges_{};
  std::optional<FilamentSoftGeometry> filament_soft_geometry_;
};

INSTANTIATE_TEST_SUITE_P(Closed, FilamentSoftGeometryTest,
                         ::testing::Values(false, true));

TEST_P(FilamentSoftGeometryTest, MakeSoftGeometryForEdge) {
  for (int i = 0; i < num_edges_; ++i) {
    EXPECT_NO_THROW(filament_soft_geometry_->MakeSoftGeometryForEdge(i));
  }
}

}  // namespace
}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
