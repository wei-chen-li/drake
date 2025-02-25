#pragma once

#include <optional>
#include <variant>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace estimators {

/**
 * A Gaussian state observer estimates the state of an observed system (plant)
 * using its input and output. All probability distributions are approximated as
 * Gaussian.
 *
 * @system
 * name: GaussianStateObserver
 * input_ports:
 * - observed_system_input
 * - observed_system_output
 * output_ports:
 * - estimated_state
 * @endsystem
 *
 * @tparam_default_scalar
/// @ingroup estimator_systems
 */
template <typename T>
class GaussianStateObserver : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GaussianStateObserver);

  /// Returns the input port that receives the observed system's input.
  virtual const InputPort<T>& get_observed_system_input_input_port() const = 0;

  /// Returns the input port that receives the observed system's output.
  virtual const InputPort<T>& get_observed_system_output_input_port() const = 0;

  /// Returns the output port that provides the estimated state.
  virtual const OutputPort<T>& get_estimated_state_output_port() const = 0;

  /// Sets the state estimate and covariance in the given @p context.
  virtual void SetStateEstimateAndCovariance(
      Context<T>* context,
      const Eigen::Ref<const Eigen::VectorX<T>>& state_estimate,
      const Eigen::Ref<const Eigen::MatrixX<T>>& state_covariance) const = 0;

  /// Gets the state estimate from the given @p context.
  virtual Eigen::VectorX<T> GetStateEstimate(
      const Context<T>& context) const = 0;

  /// Gets the state covariance from the given @p context.
  virtual Eigen::MatrixX<T> GetStateCovariance(
      const Context<T>& context) const = 0;

 protected:
  GaussianStateObserver();

  explicit GaussianStateObserver(SystemScalarConverter converter);
};

/**
 * A structure for passing optional parameters to Gaussian state observer
 * synthesis functions.
 */
struct GaussianStateObserverOptions {
  GaussianStateObserverOptions() = default;

  /// The initial state estimate. Defaults to zero initial state estimate.
  std::optional<Eigen::VectorXd> initial_state_estimate{std::nullopt};

  /// The initial state covariance. Defaults to zero initial state covariance.
  std::optional<Eigen::MatrixXd> initial_state_covariance{std::nullopt};

  /// Specifies the actuation input port of the observed system.
  /// Defaults to using the first input port.
  std::variant<systems::InputPortSelection, InputPortIndex>
      actuation_input_port_index{
          systems::InputPortSelection::kUseFirstInputIfItExists};

  /// Specifies the measurement output port of the observed system.
  /// Defaults to using the first output port.
  std::variant<systems::OutputPortSelection, OutputPortIndex>
      measurement_output_port_index{
          systems::OutputPortSelection::kUseFirstOutputIfItExists};

  /// Specifies the process noise input port, which must have kGaussian random
  /// type. Defaults to additive process noise.
  std::optional<InputPortIndex> process_noise_input_port_index{std::nullopt};

  /// Specifies the measurement noise input port, which must have kGaussian
  /// random type. Defaults to additive process noise.
  std::optional<InputPortIndex> measurement_noise_input_port_index{
      std::nullopt};

  /// Specifies the discrete measurement time period for observed system with
  /// continuous-time dynamics while discrete-time measurements are made
  /// periodically for state estimation.
  std::optional<double> discrete_measurement_time_period{std::nullopt};

  /// Specifies the discrete measurement time offset, used in conjunction
  /// with @p discrete_measurement_time_period.
  double discrete_measurement_time_offset{0.0};
};

}  // namespace estimators
}  // namespace systems
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class drake::systems::estimators::GaussianStateObserver);
