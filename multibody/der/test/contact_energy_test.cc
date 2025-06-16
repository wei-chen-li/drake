#include "drake/multibody/der/contact_energy.h"

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/text_logging.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using Eigen::Vector3d;

class LineSegmentsDistanceTest : public ::testing::TestWithParam<int> {
 private:
  void SetUp() override {
    const std::vector<std::pair<std::array<Vector3d, 4>, double>> test_cases = {
        {{Vector3d(1, 0, 0), Vector3d(2, 0, 0), Vector3d(0, 1, 0),
          Vector3d(0, 2, 0)},
         sqrt(2)},
        {{Vector3d(1, 0, 0), Vector3d(2, 0, 0), Vector3d(0, -0.5, 1),
          Vector3d(0, 0.5, 1)},
         sqrt(2)},
        {{Vector3d(-0.5, 0, 0), Vector3d(0.5, 0, 0), Vector3d(0, -0.5, 0.8),
          Vector3d(0, 0.5, 0.8)},
         0.8},
        {{Vector3d(0, 0, 0), Vector3d(3, 0, 0), Vector3d(1, 0.8, 0),
          Vector3d(2, 0.8, 0)},
         0.8},
    };

    for (const auto& [xs, distance] : test_cases) {
      using Quad = std::array<Vector3d, 4>;
      test_cases_.emplace_back(Quad{xs[0], xs[1], xs[2], xs[3]}, distance);
      test_cases_.emplace_back(Quad{xs[1], xs[0], xs[2], xs[3]}, distance);
      test_cases_.emplace_back(Quad{xs[0], xs[1], xs[3], xs[2]}, distance);
      test_cases_.emplace_back(Quad{xs[1], xs[0], xs[3], xs[2]}, distance);
      test_cases_.emplace_back(Quad{xs[2], xs[3], xs[0], xs[1]}, distance);
      test_cases_.emplace_back(Quad{xs[2], xs[3], xs[1], xs[0]}, distance);
      test_cases_.emplace_back(Quad{xs[3], xs[2], xs[0], xs[1]}, distance);
      test_cases_.emplace_back(Quad{xs[3], xs[2], xs[1], xs[0]}, distance);
    }
  }

 protected:
  std::vector<std::pair<std::array<Vector3d, 4>, double>> test_cases_;
};

/*  There are 4 * 8 test cases. */
INSTANTIATE_TEST_SUITE_P(TestCases, LineSegmentsDistanceTest,
                         ::testing::Range(0, 32));

TEST_P(LineSegmentsDistanceTest, ComputeDistanceBetweenLineSegments) {
  const auto& [xs, expected_distance] = test_cases_.at(GetParam());
  const auto& [x1, x2, x3, x4] = xs;
  EXPECT_EQ(ComputeDistanceBetweenLineSegments<double>(x1, x2, x3, x4),
            expected_distance);
}

TEST_P(LineSegmentsDistanceTest, ComputeLineSegmentsDistanceJacobian) {
  const auto& [xs, _] = test_cases_.at(GetParam());
  const auto& [x1, x2, x3, x4] = xs;
  Eigen::Vector<double, 12> jacobian =
      ComputeLineSegmentsDistanceJacobian<double>(x1, x2, x3, x4);

  const auto xs_ad = math::InitializeAutoDiffTuple(x1, x2, x3, x4);
  const Vector3<AutoDiffXd> x1_ad = std::get<0>(xs_ad).cast<AutoDiffXd>();
  const Vector3<AutoDiffXd> x2_ad = std::get<1>(xs_ad).cast<AutoDiffXd>();
  const Vector3<AutoDiffXd> x3_ad = std::get<2>(xs_ad).cast<AutoDiffXd>();
  const Vector3<AutoDiffXd> x4_ad = std::get<3>(xs_ad).cast<AutoDiffXd>();
  const AutoDiffXd distance_ad = ComputeDistanceBetweenLineSegments<AutoDiffXd>(
      x1_ad, x2_ad, x3_ad, x4_ad);
  Eigen::Vector<double, 12> expected_jacobian = distance_ad.derivatives();

  EXPECT_TRUE(CompareMatrices(jacobian, expected_jacobian));
}

static std::array<Vector3<AutoDiffXAutoDiffXd>, 4>
InitializeAutoDiffAutoDiffTuple(const Vector3d& x1, const Vector3d& x2,
                                const Vector3d& x3, const Vector3d& x4) {
  constexpr int kNumVars = 12;
  Eigen::Vector<double, 12> values;
  values << x1, x2, x3, x4;

  std::vector<AutoDiffXAutoDiffXd> vars(kNumVars);
  for (int i = 0; i < kNumVars; ++i) {
    vars[i].value().value() = values[i];
    vars[i].value().derivatives() = Eigen::VectorXd::Zero(kNumVars);
    vars[i].value().derivatives()[i] = 1.0;
    vars[i].derivatives() = Eigen::VectorX<AutoDiffXd>(kNumVars);
    for (int j = 0; j < kNumVars; ++j)
      vars[i].derivatives()[j].value() = (i == j) ? 1.0 : 0.0;
  }

  Vector3<AutoDiffXAutoDiffXd>  //
      x1_ad(vars[0], vars[1], vars[2]), x2_ad(vars[3], vars[4], vars[5]),
      x3_ad(vars[6], vars[7], vars[8]), x4_ad(vars[9], vars[10], vars[11]);
  return {std::move(x1_ad), std::move(x2_ad), std::move(x3_ad),
          std::move(x4_ad)};
}

TEST_P(LineSegmentsDistanceTest, ComputeLineSegmentsDistanceHessian) {
  if (GetParam() >= 28) return;
  const auto& [xs, _] = test_cases_.at(GetParam());
  const auto& [x1, x2, x3, x4] = xs;
  Eigen::Matrix<double, 12, 12> hessian =
      ComputeLineSegmentsDistanceHessian<double>(x1, x2, x3, x4);

  const auto [x1_ad, x2_ad, x3_ad, x4_ad] =
      InitializeAutoDiffAutoDiffTuple(x1, x2, x3, x4);
  const AutoDiffXAutoDiffXd distance_ad =
      ComputeDistanceBetweenLineSegments<AutoDiffXAutoDiffXd>(x1_ad, x2_ad,
                                                              x3_ad, x4_ad);
  Eigen::Matrix<double, 12, 12> expected_hessian;
  for (int i = 0; i < 12; ++i)
    expected_hessian.col(i) = distance_ad.derivatives()[i].derivatives();

  EXPECT_TRUE(CompareMatrices(hessian, expected_hessian, 1e-15));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
