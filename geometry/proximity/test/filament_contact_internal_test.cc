#include "drake/geometry/proximity/filament_contact_internal.h"

#include <gtest/gtest.h>

#include "drake/geometry/proximity_engine.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {
namespace {

GTEST_TEST(hello, world) {
  ProximityEngine<double> engine;
  FilamentContact<double> filament_contact;
  engine.ComputeFilamentContact(&filament_contact);
}

}  // namespace
}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
