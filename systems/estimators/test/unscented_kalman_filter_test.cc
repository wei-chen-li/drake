#include "drake/systems/estimators/unscented_kalman_filter.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/estimators/kalman_filter.h"
#include "drake/systems/estimators/test_utilities/stochastic_linear_system.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/linear_system.h"

namespace drake {
namespace systems {
namespace estimators {
namespace {

template <typename T>
constexpr void TestInputOutputPorts(const GaussianStateObserver<T>& observer) {
  EXPECT_EQ(observer.GetInputPort("observed_system_input").get_index(),
            observer.get_observed_system_input_input_port().get_index());
  EXPECT_EQ(observer.GetInputPort("observed_system_output").get_index(),
            observer.get_observed_system_output_input_port().get_index());
  EXPECT_EQ(observer.GetOutputPort("estimated_state").get_index(),
            observer.get_estimated_state_output_port().get_index());
}

class UKFDiscreteInputDiscretePlantTest : public ::testing::Test {
 public:
  void SetUp() override {
    Eigen::Matrix2d A;
    Eigen::Vector2d B;
    Eigen::Matrix<double, 1, 2> C;
    Eigen::Matrix<double, 1, 1> D;
    A << 1, h_, 0, 1;
    B << 0.5 * h_ * h_, h_;
    C << 1, 0;
    D << 0;
    plant_ = std::make_unique<LinearSystem<double>>(A, B, C, D, h_);
  }

 protected:
  std::unique_ptr<LinearSystem<double>> plant_;
  const double h_ = 0.01;
  const Eigen::Matrix<double, 2, 2> W_ = Eigen::MatrixXd::Identity(2, 2) * h_;
  const Eigen::Matrix<double, 1, 1> V_ = Eigen::MatrixXd::Identity(1, 1) / h_;
};

TEST_F(UKFDiscreteInputDiscretePlantTest, Construction) {
  auto observer =
      UnscentedKalmanFilter(*plant_, *plant_->CreateDefaultContext(), W_, V_);
  EXPECT_TRUE(observer->IsDifferenceEquationSystem());
  TestInputOutputPorts(*observer);
}

// TEST_F(UKFDiscreteInputDiscretePlantTest, ErrorDynamics) {
//   Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
//   Eigen::MatrixXd Phat = Eigen::Matrix2d::Identity() * h_;
//   GaussianStateObserverOptions options;
//   options.initial_state_estimate = xhat;
//   options.initial_state_covariance = Phat;
//   auto observer = UnscentedKalmanFilter(
//       *plant_, *plant_->CreateDefaultContext(), W_, V_, options);
//   auto context = observer->CreateDefaultContext();

//   EXPECT_TRUE(CompareMatrices(observer->GetStateEstimate(*context), xhat));
//   EXPECT_TRUE(CompareMatrices(observer->GetStateCovariance(*context), Phat));
//   EXPECT_TRUE(CompareMatrices(
//       observer->get_estimated_state_output_port().Eval(*context), xhat));

//   Eigen::VectorXd u(1);
//   u << 1.2;
//   Eigen::VectorXd y(1);
//   y << 3.4;
//   observer->get_observed_system_input_input_port().FixValue(context.get(),
//   u);
//   observer->get_observed_system_output_input_port().FixValue(context.get(),
//   y);

//   // System matrices.
//   auto& A = plant_->A();
//   auto& B = plant_->B();
//   auto& C = plant_->C();
//   auto& D = plant_->D();
//   // Measurement update.
//   // K = P̂C'(CP̂C' + V)⁻¹
//   // P̂[n|n] = (I - KC)P̂[n|n-1]
//   // x̂[n|n] = x̂[n|n-1] + K(y - ŷ)
//   Eigen::MatrixXd K =
//       Phat * C.transpose() * (C * Phat * C.transpose() + V_).inverse();
//   Phat = (Eigen::Matrix2d::Identity() - K * C) * Phat;
//   xhat = xhat + K * (y - C * xhat - D * u);
//   // Prediction update.
//   // P̂[n+1|n] = AP̂[n|n-1]A' + W
//   // x̂[n+1|n] = Ax̂[n|n] + Bu[n]
//   Phat = A * Phat * A.transpose() + W_;
//   xhat = A * xhat + B * u;

//   const DiscreteValues<double>& updated =
//       observer->EvalUniquePeriodicDiscreteUpdate(*context);
//   context->SetDiscreteState(updated);
//   EXPECT_TRUE(CompareMatrices(observer->GetStateEstimate(*context), xhat));
//   EXPECT_TRUE(CompareMatrices(observer->GetStateCovariance(*context), Phat));
//   EXPECT_TRUE(CompareMatrices(
//       observer->get_estimated_state_output_port().Eval(*context), xhat));
// }

// TEST_F(UKFDiscreteInputDiscretePlantTest, SteadyState) {
//   std::shared_ptr observer =
//       UnscentedKalmanFilter(*plant_, *plant_->CreateDefaultContext(), W_,
//       V_);

//   DiagramBuilder<double> builder;
//   auto source =
//       builder.AddSystem<ConstantVectorSource>(Eigen::VectorXd::Ones(1));
//   auto plant = builder.AddSystem(std::move(plant_));
//   builder.AddSystem(observer);
//   builder.Connect(source->get_output_port(), plant->get_input_port());
//   builder.Connect(source->get_output_port(),
//                   observer->get_observed_system_input_input_port());
//   builder.Connect(plant->get_output_port(),
//                   observer->get_observed_system_output_input_port());
//   auto diagram = builder.Build();

//   Simulator<double> simulator(*diagram);
//   simulator.AdvanceTo(10);

//   // Steady state observer gain.
//   // x̂[n+1|n] = Ax̂[n|n-1] + Bu[n] + L(y - ŷ)
//   auto& A = plant->A();
//   auto& C = plant->C();
//   Eigen::MatrixXd L1 = DiscreteTimeSteadyStateKalmanFilter(A, C, W_, V_);

//   // Discrete-time observer dynamics.
//   // K = P̂C'(CP̂C' + V)⁻¹
//   // x̂[n|n] = x̂[n|n-1] + K(y - ŷ)
//   // x̂[n+1|n] = Ax̂[n|n] + Bu[n]
//   auto& observer_context =
//       dynamic_cast<const DiagramContext<double>&>(simulator.get_context())
//           .GetSubsystemContext(diagram->GetSystemIndexOrAbort(observer.get()));
//   Eigen::MatrixXd Phat = observer->GetStateCovariance(observer_context);
//   Eigen::MatrixXd L2 =
//       A * Phat * C.transpose() * (C * Phat * C.transpose() + V_).inverse();

//   EXPECT_TRUE(CompareMatrices(L1, L2, 1e-5));
// }

// TEST_F(UKFDiscreteInputDiscretePlantTest, ExplicitNoiseErrorDynamics) {
//   Eigen::MatrixXd G(2, 2), H(1, 2);
//   G << 1.0 * h_, 2.0 * h_, 3.0 * h_, 4.0 * h_;
//   H << 5.0 * h_, 6.0 * h_;
//   estimators_test::StochasticLinearSystem plant(*plant_, G, H);

//   Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
//   Eigen::MatrixXd Phat = Eigen::Matrix2d::Identity() * h_;
//   const Eigen::Matrix2d W = Eigen::Matrix2d::Identity() * h_;
//   const Eigen::Matrix2d V = Eigen::Matrix2d::Identity() / h_;
//   GaussianStateObserverOptions options;
//   options.initial_state_estimate = xhat;
//   options.initial_state_covariance = Phat;
//   options.process_noise_input_port_index =
//   plant.get_w_input_port().get_index();
//   options.measurement_noise_input_port_index =
//       plant.get_v_input_port().get_index();
//   auto observer = UnscentedKalmanFilter(plant, *plant.CreateDefaultContext(),
//   W,
//                                         V, options);
//   auto context = observer->CreateDefaultContext();

//   EXPECT_TRUE(CompareMatrices(observer->GetStateEstimate(*context), xhat));
//   EXPECT_TRUE(CompareMatrices(observer->GetStateCovariance(*context), Phat));
//   EXPECT_TRUE(CompareMatrices(
//       observer->get_estimated_state_output_port().Eval(*context), xhat));

//   Eigen::VectorXd u(1);
//   u << 1.2;
//   Eigen::VectorXd y(1);
//   y << 3.4;
//   observer->get_observed_system_input_input_port().FixValue(context.get(),
//   u);
//   observer->get_observed_system_output_input_port().FixValue(context.get(),
//   y);

//   // System matrices.
//   auto& A = plant.A();
//   auto& B = plant.B();
//   auto& C = plant.C();
//   auto& D = plant.D();
//   // Measurement update.
//   // K = P̂C'(CP̂C' + HVH')⁻¹
//   // P̂[n|n] = (I - KC)P̂[n|n-1]
//   // x̂[n|n] = x̂[n|n-1] + K(y - ŷ)
//   Eigen::MatrixXd K =
//       Phat * C.transpose() *
//       (C * Phat * C.transpose() + H * V * H.transpose()).inverse();
//   Phat = (Eigen::Matrix2d::Identity() - K * C) * Phat;
//   xhat = xhat + K * (y - C * xhat - D * u);
//   // Prediction update.
//   // P̂[n+1|n] = AP̂[n|n-1]A' + GWG'
//   // x̂[n+1|n] = Ax̂[n|n] + Bu[n]
//   Phat = A * Phat * A.transpose() + G * W * G.transpose();
//   xhat = A * xhat + B * u;

//   const DiscreteValues<double>& updated =
//       observer->EvalUniquePeriodicDiscreteUpdate(*context);
//   context->SetDiscreteState(updated);
//   EXPECT_TRUE(CompareMatrices(observer->GetStateEstimate(*context), xhat));
//   EXPECT_TRUE(CompareMatrices(observer->GetStateCovariance(*context), Phat));
//   EXPECT_TRUE(CompareMatrices(
//       observer->get_estimated_state_output_port().Eval(*context), xhat));
// }

}  // namespace
}  // namespace estimators
}  // namespace systems
}  // namespace drake
