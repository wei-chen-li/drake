#pragma once

#include <memory>

#include "drake/systems/estimators/gaussian_state_observer.h"

namespace drake {
namespace systems {
namespace estimators {

/**
 * Constructs an unscented Kalman filter for the given @p observed_system. The
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
 * unscented Kalman filter is synthesized.
 *
 * @param observed_system  The forward model for the observer.
 * @param observed_system_context Required because it may contain parameters
 * required to evaluate the observed system.
 * @param W The process noise covariance matrix, E[ww'].
 * @param V The measurement noise covariance matrix, E[vv'].
 * @param options Optional @ref GaussianStateObserverOptions.
 *
 * @pre @p observed_system must be autodiff convertable.
 * @returns The synthesized extended Kalman filter.
 *
 * @throws std::exception if W is not positive semi-definite or if V is not
 * positive definite.
 *
 * @ingroup estimator_systems
 */
std::unique_ptr<GaussianStateObserver<double>> UnscentedKalmanFilter(
    std::shared_ptr<const System<double>> observed_system,
    const Context<double>& observed_system_context,
    const Eigen::Ref<const Eigen::MatrixXd>& W,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const GaussianStateObserverOptions& options =
        GaussianStateObserverOptions());

/**
 * Constructs the unscented Kalman filter, without claiming ownership of
 * @p observed_system. Note: The @p observed_system reference must remain valid
 * for the lifetime of the returned system.
 *
 * @exclude_from_pydrake_mkdoc{This constructor is not bound.}
 */
std::unique_ptr<GaussianStateObserver<double>> UnscentedKalmanFilter(
    const System<double>& observed_system,
    const Context<double>& observed_system_context,
    const Eigen::Ref<const Eigen::MatrixXd>& W,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const GaussianStateObserverOptions& options =
        GaussianStateObserverOptions());

}  // namespace estimators
}  // namespace systems
}  // namespace drake
