#pragma once

#include <limits>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/estimators/gaussian_state_observer.h"
#include "drake/systems/estimators/kalman_filter.h"
#include "drake/systems/estimators/nonlinear_kalman_filter_internal.h"
#include "drake/systems/estimators/test_utilities/stochastic_linear_system.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/linear_system.h"

namespace drake {
namespace systems {
namespace estimators {

template <typename T>
constexpr void TestInputOutputPorts(const GaussianStateObserver<T>& observer) {
  EXPECT_EQ(observer.GetInputPort("observed_system_input").get_index(),
            observer.get_observed_system_input_input_port().get_index());
  EXPECT_EQ(observer.GetInputPort("observed_system_output").get_index(),
            observer.get_observed_system_output_input_port().get_index());
  EXPECT_EQ(observer.GetOutputPort("estimated_state").get_index(),
            observer.get_estimated_state_output_port().get_index());
}

template <typename OptionsType>
class DiscreteTimeNonlinearKalmanFilterTest : public ::testing::Test {
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

 private:
  std::unique_ptr<LinearSystem<double>> plant_;
  const double h_ = 0.01;
  const Eigen::Matrix<double, 2, 2> W_ = Eigen::MatrixXd::Identity(2, 2) * h_;
  const Eigen::Matrix<double, 1, 1> V_ = Eigen::MatrixXd::Identity(1, 1) / h_;

  virtual std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      const OptionsType& options) const = 0;

 protected:
  void TestConstruction(bool use_square_root_method) {
    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    auto observer = MakeObserver(*plant_, W_, V_, options);
    EXPECT_TRUE(observer->IsDifferenceEquationSystem());

    TestInputOutputPorts(*observer);

    auto context = observer->CreateDefaultContext();
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateEstimate(*context), xhat, 1e-14));
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateCovariance(*context), Phat, 1e-14));
    EXPECT_TRUE(CompareMatrices(
        observer->get_estimated_state_output_port().Eval(*context), xhat,
        1e-14));
  }

  void TestErrorDynamics(bool use_square_root_method) {
    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    auto observer = MakeObserver(*plant_, W_, V_, options);
    auto context = observer->CreateDefaultContext();

    Eigen::VectorXd u(1);
    u << 1.2;
    Eigen::VectorXd y(1);
    y << 3.4;
    observer->get_observed_system_input_input_port().FixValue(context.get(), u);
    observer->get_observed_system_output_input_port().FixValue(context.get(),
                                                               y);

    auto& A = plant_->A();
    auto& B = plant_->B();
    auto& C = plant_->C();
    auto& D = plant_->D();

    // Measurement update.
    // K = P̂C'(CP̂C' + V)⁻¹
    // P̂[n|n] = (I - KC)P̂[n|n-1]
    // x̂[n|n] = x̂[n|n-1] + K(y - ŷ)
    Eigen::MatrixXd K =
        Phat * C.transpose() * (C * Phat * C.transpose() + V_).inverse();
    Phat = (Eigen::Matrix2d::Identity() - K * C) * Phat;
    xhat = xhat + K * (y - C * xhat - D * u);
    // Prediction update.
    // P̂[n+1|n] = AP̂[n|n-1]A' + W
    // x̂[n+1|n] = Ax̂[n|n] + Bu[n]
    Phat = A * Phat * A.transpose() + W_;
    xhat = A * xhat + B * u;

    const DiscreteValues<double>& updated =
        observer->EvalUniquePeriodicDiscreteUpdate(*context);
    context->SetDiscreteState(updated);
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateEstimate(*context), xhat, 1e-12));
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateCovariance(*context), Phat, 1e-12));
    EXPECT_TRUE(CompareMatrices(
        observer->get_estimated_state_output_port().Eval(*context), xhat,
        1e-12));
  }

  void TestSteadyState(bool use_square_root_method) {
    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    std::shared_ptr observer = MakeObserver(*plant_, W_, V_, options);

    DiagramBuilder<double> builder;
    auto source =
        builder.AddSystem<ConstantVectorSource>(Eigen::VectorXd::Ones(1));
    auto plant = builder.AddSystem(std::move(plant_));
    builder.AddSystem(observer);
    builder.Connect(source->get_output_port(), plant->get_input_port());
    builder.Connect(source->get_output_port(),
                    observer->get_observed_system_input_input_port());
    builder.Connect(plant->get_output_port(),
                    observer->get_observed_system_output_input_port());
    auto diagram = builder.Build();

    Simulator<double> simulator(*diagram);
    simulator.AdvanceTo(10);

    // Steady state observer gain.
    // x̂[n+1|n] = Ax̂[n|n-1] + Bu[n] + L(y - ŷ)
    auto& A = plant->A();
    auto& C = plant->C();
    Eigen::MatrixXd L1 = DiscreteTimeSteadyStateKalmanFilter(A, C, W_, V_);

    // Discrete-time observer dynamics.
    // K = P̂C'(CP̂C' + V)⁻¹
    // x̂[n|n] = x̂[n|n-1] + K(y - ŷ)
    // x̂[n+1|n] = Ax̂[n|n] + Bu[n]
    auto& observer_context =
        dynamic_cast<const DiagramContext<double>&>(simulator.get_context())
            .GetSubsystemContext(
                diagram->GetSystemIndexOrAbort(observer.get()));
    Phat = observer->GetStateCovariance(observer_context);
    Eigen::MatrixXd L2 =
        A * Phat * C.transpose() * (C * Phat * C.transpose() + V_).inverse();

    EXPECT_TRUE(CompareMatrices(L1, L2, 1e-5));
  }

  void TestExplicitNoiseErrorDynamics(bool use_square_root_method) {
    Eigen::MatrixXd G(2, 2), H(1, 2);
    G << 1.0 * h_, 2.0 * h_, 3.0 * h_, 4.0 * h_;
    H << 5.0 * h_, 6.0 * h_;
    estimators_test::StochasticLinearSystem plant(*plant_, G, H);

    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    const Eigen::Matrix2d W = Eigen::Matrix2d::Identity() * h_;
    const Eigen::Matrix2d V = Eigen::Matrix2d::Identity() / h_;

    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    options.process_noise_input_port_index =
        plant.get_w_input_port().get_index();
    options.measurement_noise_input_port_index =
        plant.get_v_input_port().get_index();

    auto observer = MakeObserver(plant, W, V, options);
    auto context = observer->CreateDefaultContext();

    Eigen::VectorXd u(1);
    u << 1.2;
    Eigen::VectorXd y(1);
    y << 3.4;
    observer->get_observed_system_input_input_port().FixValue(context.get(), u);
    observer->get_observed_system_output_input_port().FixValue(context.get(),
                                                               y);

    // System matrices.
    auto& A = plant.A();
    auto& B = plant.B();
    auto& C = plant.C();
    auto& D = plant.D();
    // Measurement update.
    // K = P̂C'(CP̂C' + HVH')⁻¹
    // P̂[n|n] = (I - KC)P̂[n|n-1]
    // x̂[n|n] = x̂[n|n-1] + K(y - ŷ)
    Eigen::MatrixXd K =
        Phat * C.transpose() *
        (C * Phat * C.transpose() + H * V * H.transpose()).inverse();
    Phat = (Eigen::Matrix2d::Identity() - K * C) * Phat;
    xhat = xhat + K * (y - C * xhat - D * u);
    // Prediction update.
    // P̂[n+1|n] = AP̂[n|n-1]A' + GWG'
    // x̂[n+1|n] = Ax̂[n|n] + Bu[n]
    Phat = A * Phat * A.transpose() + G * W * G.transpose();
    xhat = A * xhat + B * u;

    const DiscreteValues<double>& updated =
        observer->EvalUniquePeriodicDiscreteUpdate(*context);
    context->SetDiscreteState(updated);
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateEstimate(*context), xhat, 1e-12));
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateCovariance(*context), Phat, 1e-12));
    EXPECT_TRUE(CompareMatrices(
        observer->get_estimated_state_output_port().Eval(*context), xhat,
        1e-12));
  }
};

template <typename OptionsType>
class ContinuousDiscreteNonlinearKalmanFilterTest : public ::testing::Test {
 public:
  void SetUp() override {
    Eigen::Matrix2d A;
    Eigen::Vector2d B;
    Eigen::Matrix<double, 1, 2> C;
    Eigen::Matrix<double, 1, 1> D;
    A << 0, 1, 0, 0;
    B << 0, 1;
    C << 1, 0;
    D << 0;
    plant_ = std::make_unique<LinearSystem<double>>(A, B, C, D);
  }

 private:
  std::unique_ptr<LinearSystem<double>> plant_;
  const Eigen::Matrix<double, 2, 2> W_ = Eigen::MatrixXd::Identity(2, 2);
  const Eigen::Matrix<double, 1, 1> V_ = Eigen::MatrixXd::Identity(1, 1);
  const double measurement_time_period_ = 0.01;

  virtual std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      double discrete_measurement_time_period,
      const OptionsType& options) const = 0;

 protected:
  void TestConstruction(bool use_square_root_method) {
    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    auto observer =
        MakeObserver(*plant_, W_, V_, measurement_time_period_, options);

    TestInputOutputPorts(*observer);

    auto context = observer->CreateDefaultContext();
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateEstimate(*context), xhat, 1e-14));
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateCovariance(*context), Phat, 1e-14));
    EXPECT_TRUE(CompareMatrices(
        observer->get_estimated_state_output_port().Eval(*context), xhat,
        1e-14));
  }

  void TestErrorDynamics(bool use_square_root_method) {
    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    auto observer =
        MakeObserver(*plant_, W_, V_, measurement_time_period_, options);
    auto context = observer->CreateDefaultContext();

    Eigen::VectorXd u(1);
    u << 1.2;
    Eigen::VectorXd y(1);
    y << 3.4;
    observer->get_observed_system_input_input_port().FixValue(context.get(), u);
    observer->get_observed_system_output_input_port().FixValue(context.get(),
                                                               y);

    auto& A = plant_->A();
    auto& B = plant_->B();
    auto& C = plant_->C();
    auto& D = plant_->D();

    // Continuous-time process update.
    // dx̂/dt = Ax̂ + Bu
    // dP̂/dt = AP̂ + P̂A' + W
    Eigen::VectorXd xhat_dot = A * xhat + B * u;
    Eigen::MatrixXd Phat_dot = A * Phat + Phat * A.transpose() + W_;

    const Eigen::VectorXd derivatives =
        observer->EvalTimeDerivatives(*context).CopyToVector();
    const int num_states = A.rows();
    EXPECT_TRUE(CompareMatrices(derivatives.head(num_states), xhat_dot, 1e-12));

    if (!use_square_root_method) {
      Eigen::MatrixXd Phat_dot_sim(num_states, num_states);
      internal::ExtractSquareMatrix(derivatives, Phat_dot_sim);
      EXPECT_TRUE(CompareMatrices(Phat_dot_sim, Phat_dot, 1e-12));
    } else {
      Eigen::MatrixXd Shat_sim(num_states, num_states);
      internal::ExtractSquareMatrix(
          context->get_continuous_state().CopyToVector(), Shat_sim);
      Eigen::MatrixXd Shat_dot_sim(num_states, num_states);
      internal::ExtractSquareMatrix(derivatives, Shat_dot_sim);
      Eigen::MatrixXd Phat_dot_sim = Shat_dot_sim * Shat_sim.transpose() +
                                     Shat_sim * Shat_dot_sim.transpose();
      EXPECT_TRUE(CompareMatrices(Phat_dot_sim, Phat_dot, 1e-12));
    }

    // Discrete-time measurement update.
    // K = P̂C'(CP̂C' + V)⁻¹
    // P̂ ← (I - KC)P̂
    // x̂ ← x̂ + K(y - ŷ)
    Eigen::MatrixXd K =
        Phat * C.transpose() * (C * Phat * C.transpose() + V_).inverse();
    Phat = (Eigen::Matrix2d::Identity() - K * C) * Phat;
    xhat = xhat + K * (y - C * xhat - D * u);

    auto event_collection = observer->AllocateCompositeEventCollection();
    observer->GetPeriodicEvents(*context, event_collection.get());
    EXPECT_TRUE(event_collection->get_unrestricted_update_events().HasEvents());
    EXPECT_TRUE(observer
                    ->CalcUnrestrictedUpdate(
                        *context,
                        event_collection->get_unrestricted_update_events(),
                        &context->get_mutable_state())
                    .succeeded());
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateEstimate(*context), xhat, 1e-12));
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateCovariance(*context), Phat, 1e-12));
    EXPECT_TRUE(CompareMatrices(
        observer->get_estimated_state_output_port().Eval(*context), xhat,
        1e-12));
  }

  void TestSimulation(bool use_square_root_method) {
    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    std::shared_ptr observer =
        MakeObserver(*plant_, W_, V_, measurement_time_period_, options);

    Eigen::VectorXd u(1);
    u << 1.2;
    Eigen::VectorXd y(1);
    y << 3.4;

    DiagramBuilder<double> builder;
    auto u_source = builder.AddSystem<ConstantVectorSource>(u);
    auto y_source = builder.AddSystem<ConstantVectorSource>(y);
    builder.AddSystem(observer);
    builder.Connect(u_source->get_output_port(),
                    observer->get_observed_system_input_input_port());
    builder.Connect(y_source->get_output_port(),
                    observer->get_observed_system_output_input_port());
    auto diagram = builder.Build();

    Simulator<double> simulator(*diagram);
    auto& observer_context =
        dynamic_cast<const DiagramContext<double>&>(simulator.get_context())
            .GetSubsystemContext(
                diagram->GetSystemIndexOrAbort(observer.get()));

    // System matrices.
    auto& C = plant_->C();
    auto& D = plant_->D();

    // Discrete-time measurement update.
    // K = P̂C'(CP̂C' + V)⁻¹
    // P̂ ← (I - KC)P̂
    // x̂ ← x̂ + K(y - ŷ)
    Eigen::MatrixXd K =
        Phat * C.transpose() * (C * Phat * C.transpose() + V_).inverse();
    Phat = (Eigen::Matrix2d::Identity() - K * C) * Phat;
    xhat = xhat + K * (y - C * xhat - D * u);

    simulator.AdvanceTo(0);
    EXPECT_TRUE(CompareMatrices(observer->GetStateEstimate(observer_context),
                                xhat, 1e-12));
    EXPECT_TRUE(CompareMatrices(observer->GetStateCovariance(observer_context),
                                Phat, 1e-12));
    EXPECT_TRUE(CompareMatrices(
        observer->get_estimated_state_output_port().Eval(observer_context),
        xhat, 1e-12));

    // Hardcoded solution for integrating the continuous-time process update:
    // dx̂/dt = Ax̂ + Bu,  dP̂/dt = AP̂ + P̂A' + W.
    xhat << 2.216060000000000, 1.612000000000000;
    // clang-format off
    Phat << 0.515087833333333, 0.258800000000000,
            0.258800000000000, 0.885000000000000;
    // clang-format on
    simulator.AdvanceTo(measurement_time_period_ -
                        std::numeric_limits<double>::epsilon());
    EXPECT_TRUE(CompareMatrices(observer->GetStateEstimate(observer_context),
                                xhat, 1e-12));
    EXPECT_TRUE(CompareMatrices(observer->GetStateCovariance(observer_context),
                                Phat, !use_square_root_method ? 1e-12 : 1e-9));
    EXPECT_TRUE(CompareMatrices(
        observer->get_estimated_state_output_port().Eval(observer_context),
        xhat, 1e-12));
  }

  void TestExplicitNoiseErrorDynamics(bool use_square_root_method) {
    Eigen::MatrixXd G(2, 2), H(1, 2);
    G << 1.0, 2.0, 3.0, 4.0;
    H << 5.0, 6.0;
    estimators_test::StochasticLinearSystem plant(*plant_, G, H);

    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    const Eigen::Matrix2d W = Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d V = Eigen::Matrix2d::Identity();

    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    options.process_noise_input_port_index =
        plant.get_w_input_port().get_index();
    options.measurement_noise_input_port_index =
        plant.get_v_input_port().get_index();

    auto observer =
        MakeObserver(plant, W, V, measurement_time_period_, options);
    auto context = observer->CreateDefaultContext();

    Eigen::VectorXd u(1);
    u << 1.2;
    Eigen::VectorXd y(1);
    y << 3.4;
    observer->get_observed_system_input_input_port().FixValue(context.get(), u);
    observer->get_observed_system_output_input_port().FixValue(context.get(),
                                                               y);

    auto& A = plant.A();
    auto& B = plant.B();
    auto& C = plant.C();
    auto& D = plant.D();

    // Continuous-time process update.
    // dx̂/dt = Ax̂ + Bu
    // dP̂/dt = AP̂ + P̂A' + GWG'
    Eigen::VectorXd xhat_dot = A * xhat + B * u;
    Eigen::MatrixXd Phat_dot =
        A * Phat + Phat * A.transpose() + G * W * G.transpose();

    const Eigen::VectorXd derivatives =
        observer->EvalTimeDerivatives(*context).CopyToVector();
    const int num_states = A.rows();
    EXPECT_TRUE(CompareMatrices(derivatives.head(num_states), xhat_dot, 1e-12));

    if (!use_square_root_method) {
      Eigen::MatrixXd Phat_dot_sim(num_states, num_states);
      internal::ExtractSquareMatrix(derivatives, Phat_dot_sim);
      EXPECT_TRUE(CompareMatrices(Phat_dot_sim, Phat_dot, 1e-12));
    } else {
      Eigen::MatrixXd Shat_sim(num_states, num_states);
      internal::ExtractSquareMatrix(
          context->get_continuous_state().CopyToVector(), Shat_sim);
      Eigen::MatrixXd Shat_dot_sim(num_states, num_states);
      internal::ExtractSquareMatrix(derivatives, Shat_dot_sim);
      Eigen::MatrixXd Phat_dot_sim = Shat_dot_sim * Shat_sim.transpose() +
                                     Shat_sim * Shat_dot_sim.transpose();
      EXPECT_TRUE(CompareMatrices(Phat_dot_sim, Phat_dot, 1e-12));
    }

    // Discrete-time measurement update.
    // K = P̂C'(CP̂C' + HVH')⁻¹
    // P̂ ← (I - KC)P̂
    // x̂ ← x̂ + K(y - ŷ)
    Eigen::MatrixXd K =
        Phat * C.transpose() *
        (C * Phat * C.transpose() + H * V * H.transpose()).inverse();
    Phat = (Eigen::Matrix2d::Identity() - K * C) * Phat;
    xhat = xhat + K * (y - C * xhat - D * u);

    auto event_collection = observer->AllocateCompositeEventCollection();
    observer->GetPeriodicEvents(*context, event_collection.get());
    EXPECT_TRUE(event_collection->get_unrestricted_update_events().HasEvents());
    EXPECT_TRUE(observer
                    ->CalcUnrestrictedUpdate(
                        *context,
                        event_collection->get_unrestricted_update_events(),
                        &context->get_mutable_state())
                    .succeeded());
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateEstimate(*context), xhat, 1e-12));
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateCovariance(*context), Phat, 1e-12));
    EXPECT_TRUE(CompareMatrices(
        observer->get_estimated_state_output_port().Eval(*context), xhat,
        1e-12));
  }
};

template <typename OptionsType>
class ContinuousTimeNonlinearKalmanFilterTest : public ::testing::Test {
 public:
  void SetUp() override {
    Eigen::Matrix2d A;
    Eigen::Vector2d B;
    Eigen::Matrix<double, 1, 2> C;
    Eigen::Matrix<double, 1, 1> D;
    A << 0, 1, 0, 0;
    B << 0, 1;
    C << 1, 0;
    D << 0;
    plant_ = std::make_unique<LinearSystem<double>>(A, B, C, D);
  }

 private:
  std::unique_ptr<LinearSystem<double>> plant_;
  const Eigen::Matrix<double, 2, 2> W_ = Eigen::MatrixXd::Identity(2, 2);
  const Eigen::Matrix<double, 1, 1> V_ = Eigen::MatrixXd::Identity(1, 1);

  virtual std::unique_ptr<GaussianStateObserver<double>> MakeObserver(
      const System<double>& observed_system,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      const OptionsType& options) const = 0;

 protected:
  void TestConstruction(bool use_square_root_method) {
    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    auto observer = MakeObserver(*plant_, W_, V_, options);
    EXPECT_TRUE(observer->IsDifferentialEquationSystem());

    TestInputOutputPorts(*observer);

    auto context = observer->CreateDefaultContext();
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateEstimate(*context), xhat, 1e-14));
    EXPECT_TRUE(
        CompareMatrices(observer->GetStateCovariance(*context), Phat, 1e-14));
    EXPECT_TRUE(CompareMatrices(
        observer->get_estimated_state_output_port().Eval(*context), xhat,
        1e-14));
  }

  void TestErrorDynamics(bool use_square_root_method) {
    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    auto observer = MakeObserver(*plant_, W_, V_, options);
    auto context = observer->CreateDefaultContext();

    Eigen::VectorXd u(1);
    u << 1.2;
    Eigen::VectorXd y(1);
    y << 3.4;
    observer->get_observed_system_input_input_port().FixValue(context.get(), u);
    observer->get_observed_system_output_input_port().FixValue(context.get(),
                                                               y);

    auto& A = plant_->A();
    auto& B = plant_->B();
    auto& C = plant_->C();
    auto& D = plant_->D();

    // dx̂/dt = Ax̂ + Bu + P̂C'V⁻¹(y - Cx̂ - Du)
    // dP̂/dt = AP̂ + P̂A' + W - P̂C'V⁻¹CP̂
    Eigen::VectorXd xhat_dot =
        A * xhat + B * u +
        Phat * C.transpose() * V_.inverse() * (y - C * xhat - D * u);
    Eigen::MatrixXd Phat_dot = A * Phat + Phat * A.transpose() + W_ -
                               Phat * C.transpose() * V_.inverse() * C * Phat;

    const Eigen::VectorXd derivatives =
        observer->EvalTimeDerivatives(*context).CopyToVector();
    const int num_states = A.rows();
    EXPECT_TRUE(CompareMatrices(derivatives.head(num_states), xhat_dot, 1e-12));

    if (!use_square_root_method) {
      Eigen::MatrixXd Phat_dot_sim(num_states, num_states);
      internal::ExtractSquareMatrix(derivatives, Phat_dot_sim);
      EXPECT_TRUE(CompareMatrices(Phat_dot_sim, Phat_dot, 1e-12));
    } else {
      Eigen::MatrixXd Shat_sim(num_states, num_states);
      internal::ExtractSquareMatrix(
          context->get_continuous_state().CopyToVector(), Shat_sim);
      Eigen::MatrixXd Shat_dot_sim(num_states, num_states);
      internal::ExtractSquareMatrix(derivatives, Shat_dot_sim);
      Eigen::MatrixXd Phat_dot_sim = Shat_dot_sim * Shat_sim.transpose() +
                                     Shat_sim * Shat_dot_sim.transpose();
      EXPECT_TRUE(CompareMatrices(Phat_dot_sim, Phat_dot, 1e-12));
    }
  }

  void TestSteadyState(bool use_square_root_method) {
    Eigen::VectorXd xhat = Eigen::Vector2d::Zero();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    std::shared_ptr observer = MakeObserver(*plant_, W_, V_, options);

    DiagramBuilder<double> builder;
    auto source =
        builder.AddSystem<ConstantVectorSource>(Eigen::VectorXd::Ones(1));
    auto plant = builder.AddSystem(std::move(plant_));
    builder.AddSystem(observer);
    builder.Connect(source->get_output_port(), plant->get_input_port());
    builder.Connect(source->get_output_port(),
                    observer->get_observed_system_input_input_port());
    builder.Connect(plant->get_output_port(),
                    observer->get_observed_system_output_input_port());
    auto diagram = builder.Build();

    Simulator<double> simulator(*diagram);
    simulator.AdvanceTo(10);

    // Steady state observer gain.
    // dx̂/dt = Ax̂ + Bu + L(y - ŷ)
    auto& A = plant->A();
    auto& C = plant->C();
    Eigen::MatrixXd L1 = SteadyStateKalmanFilter(A, C, W_, V_);

    // Continuous-time observer dynamics.
    // dx̂/dt = Ax̂ + Bu + P̂C'V⁻¹(y - ŷ)
    auto& observer_context =
        dynamic_cast<const DiagramContext<double>&>(simulator.get_context())
            .GetSubsystemContext(
                diagram->GetSystemIndexOrAbort(observer.get()));
    Phat = observer->GetStateCovariance(observer_context);
    Eigen::MatrixXd L2 = Phat * C.transpose() * V_.inverse();

    EXPECT_TRUE(CompareMatrices(L1, L2, !use_square_root_method ? 1e-5 : 1e-4));
  }

  void TestExplicitNoiseErrorDynamics(bool use_square_root_method) {
    Eigen::MatrixXd G(2, 2), H(1, 2);
    G << 1.0, 2.0, 3.0, 4.0;
    H << 5.0, 6.0;
    estimators_test::StochasticLinearSystem plant(*plant_, G, H);

    Eigen::VectorXd xhat = Eigen::Vector2d::Ones();
    Eigen::MatrixXd Phat(2, 2);
    Phat << 1, 0.5, 0.5, 1;
    const Eigen::Matrix2d W = Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d V = Eigen::Matrix2d::Identity();
    auto options = OptionsType();
    options.use_square_root_method = use_square_root_method;
    options.initial_state_estimate = xhat;
    options.initial_state_covariance = Phat;
    options.process_noise_input_port_index =
        plant.get_w_input_port().get_index();
    options.measurement_noise_input_port_index =
        plant.get_v_input_port().get_index();

    auto observer = MakeObserver(plant, W, V, options);
    auto context = observer->CreateDefaultContext();

    Eigen::VectorXd u(1);
    u << 1.2;
    Eigen::VectorXd y(1);
    y << 3.4;
    observer->get_observed_system_input_input_port().FixValue(context.get(), u);
    observer->get_observed_system_output_input_port().FixValue(context.get(),
                                                               y);

    auto& A = plant.A();
    auto& B = plant.B();
    auto& C = plant.C();
    auto& D = plant.D();

    // dx̂/dt = Ax̂ + Bu + P̂C'(HVH')⁻¹(y - Cx̂ - Du)
    // dP̂/dt = AP̂ + P̂A' + GWG' - P̂C'(HVH')⁻¹CP̂
    Eigen::VectorXd xhat_dot = A * xhat + B * u +
                               Phat * C.transpose() *
                                   (H * V * H.transpose()).inverse() *
                                   (y - C * xhat - D * u);
    Eigen::MatrixXd Phat_dot =
        A * Phat + Phat * A.transpose() + G * W * G.transpose() -
        Phat * C.transpose() * (H * V * H.transpose()).inverse() * C * Phat;

    const Eigen::VectorXd derivatives =
        observer->EvalTimeDerivatives(*context).CopyToVector();
    const int num_states = A.rows();
    EXPECT_TRUE(CompareMatrices(derivatives.head(num_states), xhat_dot, 1e-12));

    if (!use_square_root_method) {
      Eigen::MatrixXd Phat_dot_sim(num_states, num_states);
      internal::ExtractSquareMatrix(derivatives, Phat_dot_sim);
      EXPECT_TRUE(CompareMatrices(Phat_dot_sim, Phat_dot, 1e-12));
    } else {
      Eigen::MatrixXd Shat_sim(num_states, num_states);
      internal::ExtractSquareMatrix(
          context->get_continuous_state().CopyToVector(), Shat_sim);
      Eigen::MatrixXd Shat_dot_sim(num_states, num_states);
      internal::ExtractSquareMatrix(derivatives, Shat_dot_sim);
      Eigen::MatrixXd Phat_dot_sim = Shat_dot_sim * Shat_sim.transpose() +
                                     Shat_sim * Shat_dot_sim.transpose();
      EXPECT_TRUE(CompareMatrices(Phat_dot_sim, Phat_dot, 1e-12));
    }
  }
};

}  // namespace estimators
}  // namespace systems
}  // namespace drake
