#include "drake/geometry/proximity/filament_contact_internal.h"

#include <fcl/fcl.h>
#include <gtest/gtest.h>

#include "drake/geometry/proximity_engine.h"

namespace drake {
namespace geometry {
namespace internal {
namespace filament {
namespace {

GTEST_TEST(hello, world) {
  ProximityEngine<double> engine;
  unused(engine);
}

}  // namespace
}  // namespace filament
}  // namespace internal
}  // namespace geometry
}  // namespace drake
