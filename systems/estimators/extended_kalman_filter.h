#pragma once

#include <memory>
#include <optional>
#include <variant>

#include "drake/systems/estimators/gaussian_state_observer.h"

namespace drake {
namespace systems {
namespace estimators {

/**
 * A structure for passing optional parameters to Gaussian state observer
 * synthesis functions.
 */
struct ExtendedKalmanFilterOptions {
  ExtendedKalmanFilterOptions() = default;

  /**
   *  The initial state estimate. Defaults to zero initial state estimate.
   */
  std::optional<Eigen::VectorXd> initial_state_estimate{std::nullopt};

  /**
   * The initial state covariance. Defaults to zero initial state covariance.
   */
  std::optional<Eigen::MatrixXd> initial_state_covariance{std::nullopt};

  /**
   * The actuation input port of the observed system. Defaults to using the
   * first input port if it exists.
   */
  std::variant<systems::InputPortSelection, InputPortIndex>
      actuation_input_port_index{
          systems::InputPortSelection::kUseFirstInputIfItExists};

  /**
   * The measurement output port of the observed system. Defaults to using the
   * first output port.
   */
  std::variant<systems::OutputPortSelection, OutputPortIndex>
      measurement_output_port_index{
          systems::OutputPortSelection::kUseFirstOutputIfItExists};

  /**
   * The process noise input port of the observed system, which must have
   * kGaussian random type. If not specified, process noise is additive to the
   * observed system dynamics.
   */
  std::optional<InputPortIndex> process_noise_input_port_index{std::nullopt};

  /**
   * The measurement noise input port of the observed system, which must have
   * kGaussian random type. If not specified, noise is directly added to the
   * measurement output.
   */
  std::optional<InputPortIndex> measurement_noise_input_port_index{
      std::nullopt};

  /**
   * Enables the "square-root" method for computation.
   */
  bool use_square_root_method{false};

  /**
   * For observed system with continuous-time dynamics while discrete-time
   * measurements are made periodically for state estimation. Use this parameter
   * to specify the discrete measurement time period.
   */
  std::optional<double> discrete_measurement_time_period{std::nullopt};

  /**
   * Used in conjunction with @p discrete_measurement_time_period. Use this
   * parameter to specify the discrete measurement time offset.
   */
  double discrete_measurement_time_offset{0.0};
};

/**
 * Constructs an extended Kalman filter for the given @p observed_system. The
 * filter can be synthesized for either discrete-time or continuous-time
 * dynamics.
 *
 * **Discrete-time dynamics**
 * The observed system dynamics can be written in one of two forms:
 * x[n+1] = f(x[n],u[n]) + w[n] or x[n+1] = f(x[n],u[n],w[n]).
 * In the latter case, specify the process noise input port using
 * @p options.process_noise_input_port_index.
 * The measurement model can also be written in one of two forms:
 * y[n] = g(x[n],u[n]) + v[n] or y[n] = g(x[n],u[n],v[n]).
 * In the latter case, specify the measurement noise input port using
 * @p options.measurement_noise_input_port_index.
 *
 * **Continuous-time dynamics**
 * The observed system dynamics can be written in one of two forms:
 * ẋ = f(x,u) + w or ẋ = f(x,u,w).
 * In the latter case, specify the process noise input port using
 * @p options.process_noise_input_port_index.
 * The measurement model can also be written in one of two forms:
 * y = g(x,u) + v or y = g(x,u,v).
 * In the latter case, specify the measurement noise input port using
 * @p options.measurement_noise_input_port_index.
 * Additionally, if @p options.discrete_measurement_time_period is specified,
 * the synthesized filter will perform continuous-time process updates combined
 * with discrete-time measurement updates. Otherwise, a pure continuous-time
 * extended Kalman filter is synthesized.
 *
 * @param observed_system  The forward model for the observer.
 * @param observed_system_context Required because it may contain parameters
 * required to evaluate the observed system.
 * @param W The process noise covariance matrix, E[ww'].
 * @param V The measurement noise covariance matrix, E[vv'].
 * @param options Optional @ref ExtendedKalmanFilterOptions.
 *
 * @pre @p observed_system must convertible to System<AutoDiffXd>.
 * @returns The synthesized extended Kalman filter.
 *
 * @throws std::exception if W is not positive semi-definite or if V is not
 * positive definite.
 *
 * @ingroup estimator_systems
 */
std::unique_ptr<GaussianStateObserver<double>> ExtendedKalmanFilter(
    const System<double>& observed_system,
    const Context<double>& observed_system_context,
    const Eigen::Ref<const Eigen::MatrixXd>& W,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const ExtendedKalmanFilterOptions& options = ExtendedKalmanFilterOptions());

}  // namespace estimators
}  // namespace systems
}  // namespace drake
