#include "drake/math/frame_transport.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace math {
namespace {

using Eigen::Vector3d;

constexpr double kTol = 1e-15;

GTEST_TEST(FrameTransportTest, test) {
  auto [sin30, cos30] = std::make_pair(1.0 / 2, sqrt(3) / 2);
  Vector3d t_0, t_1, d1_0, d1_1;
  t_0 << 1.0, 0.0, 0.0;
  t_1 << cos30, sin30, 0.0;
  d1_0 << 0, 4.0 / 5, 3.0 / 5;
  FrameTransport<double>(t_0, d1_0, t_1, d1_1);

  Vector3d d1_1_expected;
  d1_1_expected << 4.0 / 5 * -sin30, 4.0 / 5 * cos30, 3.0 / 5;
  EXPECT_TRUE(CompareMatrices(d1_1, d1_1_expected, kTol));
}

GTEST_TEST(SpaceParallelFrameTransportTest, DidntSpecifyFirstDirector) {
  auto [sin30, cos30] = std::make_pair(1.0 / 2, sqrt(3) / 2);
  Eigen::Matrix3Xd t(3, 2);
  t.col(0) = Vector3d(1, 0, 0);
  t.col(1) = Vector3d(cos30, sin30, 0);

  Eigen::Matrix3Xd d1(3, 2);
  SpaceParallelFrameTransport<double>(t, std::nullopt, &d1);

  EXPECT_LT(abs(t.col(0).dot(d1.col(0))), kTol);
  EXPECT_LT(abs(d1.col(0).norm() - 1.0), kTol);
  EXPECT_LT(abs(t.col(1).dot(d1.col(1))), kTol);
  EXPECT_LT(abs(d1.col(1).norm() - 1.0), kTol);

  Vector3d d1_1_expected;
  FrameTransport<double>(t.col(0), d1.col(0), t.col(1), d1_1_expected);
  EXPECT_TRUE(CompareMatrices(d1.col(1), d1_1_expected, kTol));
}

GTEST_TEST(SpaceParallelFrameTransportTest, SpecifyFirstDirector) {
  const double a = sqrt(2) / 2;
  auto [sin30, cos30] = std::make_pair(1.0 / 2, sqrt(3) / 2);

  Eigen::Matrix3Xd t(3, 2);
  t.col(0) = Vector3d(1, 0, 0);
  t.col(1) = Vector3d(cos30, sin30, 0);

  Vector3d d1_0(0, a, a);
  Eigen::Matrix3Xd d1(3, 2);
  SpaceParallelFrameTransport<double>(t, d1_0, &d1);

  EXPECT_TRUE(CompareMatrices(d1.col(0), d1_0, kTol));
  EXPECT_TRUE(
      CompareMatrices(d1.col(1), Vector3d(a * -sin30, a * cos30, a), kTol));
}

GTEST_TEST(ComputTimeParallelTransport, test) {
  const double a = sqrt(2) / 2;
  auto [sin30, cos30] = std::make_pair(1.0 / 2, sqrt(3) / 2);

  Eigen::Matrix3Xd t(3, 2), d1(3, 2);
  t.col(0) = Vector3d(1, 0, 0);
  t.col(1) = Vector3d(cos30, sin30, 0);
  d1.col(0) = Vector3d(0, a, a);
  SpaceParallelFrameTransport<double>(t, d1.col(0), &d1);

  Eigen::Matrix3Xd t_next(3, 2);
  t_next.col(0) = Vector3d(cos30, 0, sin30);
  t_next.col(1) = Vector3d(cos30 * cos30, sin30 * cos30, sin30);

  Eigen::Matrix3Xd d1_next(3, 2);
  TimeParallelFrameTransport<double>(t, d1, t_next, &d1_next);

  Eigen::Matrix3Xd d1_next_expected(3, 2);
  d1_next_expected.col(0) = Vector3d(a * -sin30, a, a * cos30);
  FrameTransport<double>(t.col(1), Vector3d(a * -sin30, a * cos30, a),
                         t_next.col(1), d1_next_expected.col(1));
  EXPECT_TRUE(CompareMatrices(d1_next, d1_next_expected, kTol));
}

}  // namespace
}  // namespace math
}  // namespace drake
