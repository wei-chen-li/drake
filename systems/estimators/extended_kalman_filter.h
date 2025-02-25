#pragma once

#include <memory>

#include "drake/systems/estimators/gaussian_state_observer.h"

namespace drake {
namespace systems {
namespace estimators {

/// @param observed_system  The forward model for the observer.  Currently,
/// this system must have a maximum of one input port and exactly one output
/// port.
/// @param observed_system_context Required because it may contain parameters
/// which we need to evaluate the system.
/// @param W The process noise covariance matrix, E[ww'].
/// @param V The measurement noise covariance matrix, E[vv'].
/// @param options The optional @ref GaussianStateObserverOptions.

/// @pre @p observed_system must be autodiff convertable.
///
/// @ingroup estimator_systems
std::unique_ptr<GaussianStateObserver<double>> ExtendedKalmanFilter(
    const System<double>& observed_system,
    const Context<double>& observed_system_context,
    const Eigen::Ref<const Eigen::MatrixXd>& W,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const GaussianStateObserverOptions& options =
        GaussianStateObserverOptions());

}  // namespace estimators
}  // namespace systems
}  // namespace drake
