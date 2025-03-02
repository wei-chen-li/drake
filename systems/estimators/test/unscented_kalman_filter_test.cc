#include "drake/systems/estimators/unscented_kalman_filter.h"

#include <gtest/gtest.h>

#include "drake/systems/estimators/test/nonlinear_kalman_filter_test.h"

namespace drake {
namespace systems {
namespace estimators {
namespace {
class DiscreteTimeUKFTest : public DiscreteTimeNonlinearKalmanFilterTest<
                                UnscentedKalmanFilterOptions> {
 private:
  std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      const UnscentedKalmanFilterOptions& options) const override {
    DRAKE_THROW_UNLESS(!options.discrete_measurement_time_period.has_value());
    return UnscentedKalmanFilter(observed_system,
                                 *observed_system.CreateDefaultContext(), W, V,
                                 options);
  }
};

TEST_F(DiscreteTimeUKFTest, Contruction) {
  this->TestConstruction();
}
TEST_F(DiscreteTimeUKFTest, VectorInputDynamics) {
  this->TestVectorInputDynamics();
}
TEST_F(DiscreteTimeUKFTest, NoInputDynamics) {
  this->TestNoInputDynamics();
}
TEST_F(DiscreteTimeUKFTest, AbstractInputDynamics) {
  this->TestAbstractInputDynamics();
}
TEST_F(DiscreteTimeUKFTest, ProcessNoiseInputDynamics) {
  this->TestProcessNoiseInputDynamics();
}
TEST_F(DiscreteTimeUKFTest, MeasurementNoiseInputDynamics) {
  this->TestMeasurementNoiseInputDynamics();
}
TEST_F(DiscreteTimeUKFTest, ProcessAndMeasurementNoiseInputDynamics) {
  this->TestProcessAndMeasurementNoiseInputDynamics();
}
TEST_F(DiscreteTimeUKFTest, SteadyState) {
  this->TestSteadyState();
}

class ContinuousDiscreteUKFTest
    : public ContinuousDiscreteNonlinearKalmanFilterTest<
          UnscentedKalmanFilterOptions> {
 private:
  std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      const UnscentedKalmanFilterOptions& options) const override {
    DRAKE_THROW_UNLESS(options.discrete_measurement_time_period.has_value());
    return UnscentedKalmanFilter(observed_system,
                                 *observed_system.CreateDefaultContext(), W, V,
                                 options);
  }
};

TEST_F(ContinuousDiscreteUKFTest, Contruction) {
  this->TestConstruction();
}
TEST_F(ContinuousDiscreteUKFTest, VectorInputDynamics) {
  this->TestVectorInputDynamics();
}
TEST_F(ContinuousDiscreteUKFTest, NoInputDynamics) {
  this->TestNoInputDynamics();
}
TEST_F(ContinuousDiscreteUKFTest, AbstractInputDynamics) {
  this->TestAbstractInputDynamics();
}
TEST_F(ContinuousDiscreteUKFTest, ProcessNoiseInputDynamics) {
  this->TestProcessNoiseInputDynamics();
}
TEST_F(ContinuousDiscreteUKFTest, MeasurementNoiseInputDynamics) {
  this->TestMeasurementNoiseInputDynamics();
}
TEST_F(ContinuousDiscreteUKFTest, ProcessAndMeasurementNoiseInputDynamics) {
  this->TestProcessAndMeasurementNoiseInputDynamics();
}
TEST_F(ContinuousDiscreteUKFTest, Simulation) {
  this->TestSimulation();
}

class ContinuousTimeUKFTest : public ContinuousTimeNonlinearKalmanFilterTest<
                                  UnscentedKalmanFilterOptions> {
 private:
  std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      const UnscentedKalmanFilterOptions& options) const override {
    DRAKE_THROW_UNLESS(!options.discrete_measurement_time_period.has_value());
    return UnscentedKalmanFilter(observed_system,
                                 *observed_system.CreateDefaultContext(), W, V,
                                 options);
  }
};

TEST_F(ContinuousTimeUKFTest, Contruction) {
  this->TestConstruction();
}
TEST_F(ContinuousTimeUKFTest, VectorInputDynamics) {
  this->TestVectorInputDynamics();
}
TEST_F(ContinuousTimeUKFTest, NoInputDynamics) {
  this->TestNoInputDynamics();
}
TEST_F(ContinuousTimeUKFTest, AbstractInputDynamics) {
  this->TestAbstractInputDynamics();
}
TEST_F(ContinuousTimeUKFTest, ProcessNoiseInputDynamics) {
  this->TestProcessNoiseInputDynamics();
}
TEST_F(ContinuousTimeUKFTest, MeasurementNoiseInputDynamics) {
  this->TestMeasurementNoiseInputDynamics();
}
TEST_F(ContinuousTimeUKFTest, ProcessAndMeasurementNoiseInputDynamics) {
  this->TestProcessAndMeasurementNoiseInputDynamics();
}
TEST_F(ContinuousTimeUKFTest, SteadyState) {
  this->TestSteadyState();
}

}  // namespace

namespace internal {
namespace {

GTEST_TEST(UnscentedTransformTest, test) {
  const int n = 2;
  Eigen::Vector2d mean;
  Eigen::Matrix2d covar;
  mean << 12.3, 7.6;
  // clang-format off
  covar << 1.44, 0,
           0, 2.89;
  // clang-format on

  const double alpha = 1, beta = 2, kappa = 0;
  const double lambda = alpha * alpha * (n + kappa) - n;
  auto [X, wm, Wc] = UnscentedTransform(
      std::make_pair(mean, covar),
      UnscentedKalmanFilterOptions::UnscentedTransformParameters(alpha, beta,
                                                                 kappa));

  Eigen::MatrixXd X_expected = mean.replicate(1, 5);
  EXPECT_EQ(n + lambda, 2.0);
  X_expected.col(1) += sqrt(2.0) * Eigen::Vector2d(1.2, 0);
  X_expected.col(2) += sqrt(2.0) * Eigen::Vector2d(0, 1.7);
  X_expected.col(3) -= sqrt(2.0) * Eigen::Vector2d(1.2, 0);
  X_expected.col(4) -= sqrt(2.0) * Eigen::Vector2d(0, 1.7);
  EXPECT_TRUE(CompareMatrices(X, X_expected));

  Eigen::VectorXd wm_expected =
      Eigen::VectorXd::Constant(5, 1 / (2 * (n + lambda)));
  wm_expected(0) = lambda / (n + lambda);
  EXPECT_TRUE(CompareMatrices(wm, wm_expected));

  // wₘ is a vector containing the mean weights, allowing us to compute the mean
  // using X * wₘ.
  EXPECT_TRUE(CompareMatrices(X * wm, mean));

  // Wc is a matrix, allowing us to compute the covariance using
  // X * Wc * X.transpose().
  EXPECT_TRUE(CompareMatrices(X * Wc * X.transpose(), covar, 1e-13));
}

GTEST_TEST(JointGaussianTest, TwoGaussians) {
  Eigen::VectorXd mean1(1), mean2(1);
  Eigen::MatrixXd covar1(1, 1), covar2(1, 1);

  mean1 << 1;
  covar1 << 2;
  mean2 << 3;
  covar2 << 4;
  auto [joint_mean, joint_covar] = JointGaussian(mean1, covar1, mean2, covar2);

  Eigen::VectorXd expected_mean(2);
  Eigen::MatrixXd expected_covar(2, 2);
  expected_mean << 1, 3;
  // clang-format off
  expected_covar << 2, 0,
                    0, 4;
  // clang-format on
  EXPECT_TRUE(CompareMatrices(joint_mean, expected_mean));
  EXPECT_TRUE(CompareMatrices(joint_covar, expected_covar));
}

GTEST_TEST(JointGaussianTest, ThreeGaussians) {
  Eigen::VectorXd mean1(1), mean2(1), mean3(1);
  Eigen::MatrixXd covar1(1, 1), covar2(1, 1), covar3(1, 1);

  mean1 << 1;
  covar1 << 2;
  mean2 << 3;
  covar2 << 4;
  mean3 << 5;
  covar3 << 6;
  auto [joint_mean, joint_covar] =
      JointGaussian(mean1, covar1, mean2, covar2, mean3, covar3);

  Eigen::VectorXd expected_mean(3);
  Eigen::MatrixXd expected_covar(3, 3);
  expected_mean << 1, 3, 5;
  // clang-format off
  expected_covar << 2, 0, 0,
                    0, 4, 0,
                    0, 0, 6;
  // clang-format on
  EXPECT_TRUE(CompareMatrices(joint_mean, expected_mean));
  EXPECT_TRUE(CompareMatrices(joint_covar, expected_covar));
}

}  // namespace
}  // namespace internal

}  // namespace estimators
}  // namespace systems
}  // namespace drake
