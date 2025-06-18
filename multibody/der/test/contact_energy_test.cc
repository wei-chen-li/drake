#include "drake/multibody/der/contact_energy.h"

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/der/test_utilities/autodiff_autodiff.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

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

  const double distance =
      ComputeDistanceBetweenLineSegments<double>(x1, x2, x3, x4);

  EXPECT_EQ(distance, expected_distance);
}

TEST_P(LineSegmentsDistanceTest, ComputeLineSegmentsDistanceJacobian) {
  const auto& [xs, _] = test_cases_.at(GetParam());
  const auto& [x1, x2, x3, x4] = xs;
  const Eigen::Vector<double, 12> jacobian =
      ComputeLineSegmentsDistanceJacobian<double>(x1, x2, x3, x4);

  const auto [x1_ad, x2_ad, x3_ad, x4_ad] =
      math::InitializeAutoDiffTuple(VectorXd(x1), x2, x3, x4);
  const AutoDiffXd distance_ad = ComputeDistanceBetweenLineSegments<AutoDiffXd>(
      x1_ad, x2_ad, x3_ad, x4_ad);
  const Eigen::Vector<double, 12> expected_jacobian = distance_ad.derivatives();

  EXPECT_TRUE(CompareMatrices(jacobian, expected_jacobian));
}

TEST_P(LineSegmentsDistanceTest, ComputeLineSegmentsDistanceHessian) {
  if (GetParam() >= 28) return;
  const auto& [xs, _] = test_cases_.at(GetParam());
  const auto& [x1, x2, x3, x4] = xs;
  const Eigen::Matrix<double, 12, 12> hessian =
      ComputeLineSegmentsDistanceHessian<double>(x1, x2, x3, x4);

  const auto [x1_adad, x2_adad, x3_adad, x4_adad] =
      math::InitializeAutoDiffAutoDiffTuple(VectorXd(x1), x2, x3, x4);
  const AutoDiffXAutoDiffXd distance_adad =
      ComputeDistanceBetweenLineSegments<AutoDiffXAutoDiffXd>(x1_adad, x2_adad,
                                                              x3_adad, x4_adad);
  const Eigen::Matrix<double, 12, 12> expected_hessian =
      math::ExtractHessian(distance_adad);

  EXPECT_TRUE(CompareMatrices(hessian, expected_hessian, 1e-15));
}

class ContactEnergyTest
    : public ::testing::TestWithParam<std::tuple<bool, double>> {
 private:
  void SetUp() override {
    const bool has_closed_ends = std::get<0>(GetParam());
    std::vector<Vector3d> node_positions;
    std::vector<double> edge_angles;
    if (!has_closed_ends) {
      node_positions = {Vector3d(-1, 0, 0), Vector3d(1, 0, 0),
                        Vector3d(1, 1, 0.15), Vector3d(0, 1, 0.3),
                        Vector3d(0, -1, 0.3)};
      edge_angles = {0, 0, 0, 0};
    } else {
      node_positions = {Vector3d(-1, 0, 0),   Vector3d(1, 0, 0),
                        Vector3d(1, 1, 0.15), Vector3d(0, 1, 0.3),
                        Vector3d(0, -1, 0.3), Vector3d(-1, -1, 0.15)};
      edge_angles = {0, 0, 0, 0, 0, 0};
    }
    der_state_system_ = std::make_unique<DerStateSystem<double>>(
        has_closed_ends, node_positions, edge_angles, std::nullopt);

    der_state_ = std::make_unique<DerState<double>>(der_state_system_.get());

    undeformed_ = DerUndeformedState<double>::FromCurrentDerState(*der_state_);

    const double C = std::get<1>(GetParam());
    contact_energy_ = ContactEnergy<double>(C, *undeformed_);
  }

  static double ExtractDoubleOrThrow(const AutoDiffXd& in) {
    return in.value();
  }
  static double ExtractDoubleOrThrow(const AutoDiffXAutoDiffXd& in) {
    return in.value().value();
  }

 protected:
  template <typename T>
  T ComputeContactEnergy(const VectorX<T>& q) {
    const double C = std::get<1>(GetParam());
    const double delta = 0.01 * C;
    const double K = 15 / delta;

    const geometry::internal::filament::FilamentSelfContactFilter filter(
        undeformed_->has_closed_ends(), undeformed_->get_edge_length(), C);
    const int num_nodes = undeformed_->num_nodes();
    const int num_edges = undeformed_->num_edges();

    T energy = 0.0 * q[0];
    for (int i = 0; i < num_edges; ++i) {
      for (int j = i; j < num_edges; ++j) {
        if (!filter.ShouldCollide(i, j)) continue;
        const int ip1 = (i + 1) % num_nodes;
        const int jp1 = (j + 1) % num_nodes;
        const T D = ComputeDistanceBetweenLineSegments<T>(
            q.template segment<3>(4 * i), q.template segment<3>(4 * ip1),
            q.template segment<3>(4 * j), q.template segment<3>(4 * jp1));
        if (ExtractDoubleOrThrow(D) <= C - delta) {
          const T val = C - D;
          energy += val * val;
        } else if (ExtractDoubleOrThrow(D) < C + delta) {
          const T val = 1 / K * log(1 + exp(K * (C - D)));
          energy += val * val;
        }
      }
    }
    return energy;
  }

  std::unique_ptr<DerStateSystem<double>> der_state_system_;
  std::unique_ptr<DerState<double>> der_state_;
  std::optional<DerUndeformedState<double>> undeformed_;
  std::optional<ContactEnergy<double>> contact_energy_;
};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds_ContactDistances, ContactEnergyTest,
                         ::testing::Combine(::testing::Values(false, true),
                                            ::testing::Values(0.29, 0.3,
                                                              0.31)));

TEST_P(ContactEnergyTest, ComputeEnergy) {
  const double energy = contact_energy_->ComputeEnergy(*der_state_);

  const VectorXd q = der_state_->get_position();
  const double expected_energy = ComputeContactEnergy(q);

  EXPECT_EQ(energy, expected_energy);
}

TEST_P(ContactEnergyTest, ComputeEnergyJacobian) {
  const VectorXd& jacobian =
      contact_energy_->ComputeEnergyJacobian(*der_state_);

  const VectorXd q = der_state_->get_position();
  const auto q_ad = math::InitializeAutoDiff(q);
  const AutoDiffXd energy_ad = ComputeContactEnergy(q_ad);
  const VectorXd expected_jacobian = energy_ad.derivatives();

  EXPECT_TRUE(CompareMatrices(jacobian, expected_jacobian));
}

TEST_P(ContactEnergyTest, ComputeEnergyHessian) {
  const Block4x4SparseSymmetricMatrix<double>& hessian =
      contact_energy_->ComputeEnergyHessian(*der_state_);

  const VectorXd q = der_state_->get_position();
  const auto q_adad = math::InitializeAutoDiffAutoDiff(q);
  const AutoDiffXAutoDiffXd energy_adad = ComputeContactEnergy(q_adad);
  const MatrixXd expected_hessian = math::ExtractHessian(energy_adad);

  const int num_dofs = der_state_->num_dofs();
  EXPECT_TRUE(CompareMatrices(
      hessian.MakeDenseMatrix().topLeftCorner(num_dofs, num_dofs),
      expected_hessian, 1e-15));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
