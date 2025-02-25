#include "drake/systems/estimators/extended_kalman_filter.h"

#include <gtest/gtest.h>

#include "drake/systems/estimators/test/nonlinear_kalman_filter_test.h"

namespace drake {
namespace systems {
namespace estimators {
namespace {

class DiscreteTimeEKFTest : public DiscreteTimeNonlinearKalmanFilterTest<
                                ExtendedKalmanFilterOptions> {
 private:
  std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      const ExtendedKalmanFilterOptions& options) const override {
    return ExtendedKalmanFilter(observed_system,
                                *observed_system.CreateDefaultContext(), W, V,
                                options);
  }
};

TEST_F(DiscreteTimeEKFTest, Contruction) {
  this->TestConstruction(false);
}
TEST_F(DiscreteTimeEKFTest, ContructionSqrt) {
  this->TestConstruction(true);
}
TEST_F(DiscreteTimeEKFTest, ErrorDynamics) {
  this->TestErrorDynamics(false);
}
TEST_F(DiscreteTimeEKFTest, ErrorDynamicsSqrt) {
  this->TestErrorDynamics(true);
}
TEST_F(DiscreteTimeEKFTest, SteadyState) {
  this->TestSteadyState(false);
}
TEST_F(DiscreteTimeEKFTest, SteadyStateSqrt) {
  this->TestSteadyState(true);
}
TEST_F(DiscreteTimeEKFTest, ExplicitNoiseErrorDynamics) {
  this->TestExplicitNoiseErrorDynamics(false);
}
TEST_F(DiscreteTimeEKFTest, ExplicitNoiseErrorDynamicsSqrt) {
  this->TestExplicitNoiseErrorDynamics(true);
}

class ContinuousDiscreteEKFTest
    : public ContinuousDiscreteNonlinearKalmanFilterTest<
          ExtendedKalmanFilterOptions> {
 private:
  std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      double discrete_measurement_time_period,
      const ExtendedKalmanFilterOptions& options_temp) const override {
    auto options = options_temp;
    options.discrete_measurement_time_period = discrete_measurement_time_period;
    return ExtendedKalmanFilter(observed_system,
                                *observed_system.CreateDefaultContext(), W, V,
                                options);
  }
};

TEST_F(ContinuousDiscreteEKFTest, Contruction) {
  this->TestConstruction(false);
}
TEST_F(ContinuousDiscreteEKFTest, ContructionSqrt) {
  this->TestConstruction(true);
}
TEST_F(ContinuousDiscreteEKFTest, ErrorDynamics) {
  this->TestErrorDynamics(false);
}
TEST_F(ContinuousDiscreteEKFTest, ErrorDynamicsSqrt) {
  this->TestErrorDynamics(true);
}
TEST_F(ContinuousDiscreteEKFTest, Simulation) {
  this->TestSimulation(false);
}
TEST_F(ContinuousDiscreteEKFTest, SimulationSqrt) {
  this->TestSimulation(true);
}
TEST_F(ContinuousDiscreteEKFTest, ExplicitNoiseErrorDynamic) {
  this->TestExplicitNoiseErrorDynamics(false);
}
TEST_F(ContinuousDiscreteEKFTest, ExplicitNoiseErrorDynamicSqrt) {
  this->TestExplicitNoiseErrorDynamics(true);
}

class ContinuousTimeEKFTest : public ContinuousTimeNonlinearKalmanFilterTest<
                                  ExtendedKalmanFilterOptions> {
 private:
  std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      const ExtendedKalmanFilterOptions& options) const override {
    return ExtendedKalmanFilter(observed_system,
                                *observed_system.CreateDefaultContext(), W, V,
                                options);
  }
};

TEST_F(ContinuousTimeEKFTest, Contruction) {
  this->TestConstruction(false);
}
TEST_F(ContinuousTimeEKFTest, ContructionSqrt) {
  this->TestConstruction(true);
}
TEST_F(ContinuousTimeEKFTest, ErrorDynamics) {
  this->TestErrorDynamics(false);
}
TEST_F(ContinuousTimeEKFTest, ErrorDynamicsSqrt) {
  this->TestErrorDynamics(true);
}
TEST_F(ContinuousTimeEKFTest, SteadyState) {
  this->TestSteadyState(false);
}
TEST_F(ContinuousTimeEKFTest, SteadyStateSqrt) {
  this->TestSteadyState(true);
}
TEST_F(ContinuousTimeEKFTest, ExplicitNoiseErrorDynamics) {
  this->TestExplicitNoiseErrorDynamics(false);
}
TEST_F(ContinuousTimeEKFTest, ExplicitNoiseErrorDynamicsSqrt) {
  this->TestExplicitNoiseErrorDynamics(true);
}

}  // namespace
}  // namespace estimators
}  // namespace systems
}  // namespace drake
