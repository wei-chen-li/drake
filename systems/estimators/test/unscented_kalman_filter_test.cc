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
    return UnscentedKalmanFilter(observed_system,
                                 *observed_system.CreateDefaultContext(), W, V,
                                 options);
  }
};

TEST_F(DiscreteTimeUKFTest, Contruction) {
  this->TestConstruction(false);
}
TEST_F(DiscreteTimeUKFTest, ContructionSqrt) {
  this->TestConstruction(true);
}
TEST_F(DiscreteTimeUKFTest, VectorInputErrorDynamics) {
  this->TestVectorInputErrorDynamics(false);
}
TEST_F(DiscreteTimeUKFTest, ExplicitNoiseErrorDynamics) {
  this->TestExplicitNoiseErrorDynamics(false);
}
TEST_F(DiscreteTimeUKFTest, SteadyState) {
  this->TestSteadyState(false);
}

class ContinuousDiscreteUKFTest
    : public ContinuousDiscreteNonlinearKalmanFilterTest<
          UnscentedKalmanFilterOptions> {
 private:
  std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      double discrete_measurement_time_period,
      const UnscentedKalmanFilterOptions& options_temp) const override {
    auto options = options_temp;
    options.discrete_measurement_time_period = discrete_measurement_time_period;
    return UnscentedKalmanFilter(observed_system,
                                 *observed_system.CreateDefaultContext(), W, V,
                                 options);
  }
};

TEST_F(ContinuousDiscreteUKFTest, Contruction) {
  this->TestConstruction(false);
}
TEST_F(ContinuousDiscreteUKFTest, VectorInputErrorDynamics) {
  this->TestVectorInputErrorDynamics(false);
}
TEST_F(ContinuousDiscreteUKFTest, ExplicitNoiseErrorDynamics) {
  this->TestExplicitNoiseErrorDynamics(false);
}
TEST_F(ContinuousDiscreteUKFTest, Simulation) {
  this->TestSimulation(false);
}

class ContinuousTimeUKFTest : public ContinuousTimeNonlinearKalmanFilterTest<
                                  UnscentedKalmanFilterOptions> {
 private:
  std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      const UnscentedKalmanFilterOptions& options) const override {
    return UnscentedKalmanFilter(observed_system,
                                 *observed_system.CreateDefaultContext(), W, V,
                                 options);
  }
};

TEST_F(ContinuousTimeUKFTest, Contruction) {
  this->TestConstruction(false);
}
TEST_F(ContinuousTimeUKFTest, VectorInputErrorDynamics) {
  this->TestVectorInputErrorDynamics(false);
}
TEST_F(ContinuousTimeUKFTest, ExplicitNoiseErrorDynamics) {
  this->TestExplicitNoiseErrorDynamics(false);
}
TEST_F(ContinuousTimeUKFTest, SteadyState) {
  this->TestSteadyState(false);
}

}  // namespace
}  // namespace estimators
}  // namespace systems
}  // namespace drake
