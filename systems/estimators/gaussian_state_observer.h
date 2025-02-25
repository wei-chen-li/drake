#pragma once

#include <optional>
#include <variant>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace estimators {

/// A Gaussian state observer takes the input and output of a plant and
/// estimates the plant's state. All probability distributions are approximated
/// as Gaussian.
/// @system
/// name: StateObserver
/// input_ports:
/// - observed_system_input
/// - observed_system_output
/// output_ports:
/// - estimated_state
/// @endsystem
///
/// @tparam_default_scalar
template <typename T>
class GaussianStateObserver : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GaussianStateObserver);

  virtual const InputPort<T>& get_observed_system_input_input_port() const = 0;

  virtual const InputPort<T>& get_observed_system_output_input_port() const = 0;

  virtual const OutputPort<T>& get_estimated_state_output_port() const = 0;

  virtual void SetStateEstimateAndCovariance(
      Context<T>* context,
      const Eigen::Ref<const Eigen::VectorX<T>>& state_estimate,
      const Eigen::Ref<const Eigen::MatrixX<T>>& state_covariance) const = 0;

  virtual Eigen::VectorX<T> GetStateEstimate(
      const Context<T>& context) const = 0;

  virtual Eigen::MatrixX<T> GetStateCovariance(
      const Context<T>& context) const = 0;

 protected:
  GaussianStateObserver();

  explicit GaussianStateObserver(SystemScalarConverter converter);
};

/// A structure to facilitate passing the myriad of optional arguments to state
/// observer synthesis functions.
struct GaussianStateObserverOptions {
  GaussianStateObserverOptions() = default;

  /// For systems with multiple input ports, we must specify the actuation input
  /// port.
  std::variant<systems::InputPortSelection, InputPortIndex>
      actuation_input_port_index{
          systems::InputPortSelection::kUseFirstInputIfItExists};

  /// The process noise input port, the port must have kGaussian random type.
  /// Defaults to additive process noise.
  std::optional<InputPortIndex> process_noise_input_port_index{std::nullopt};

  /// The measurement noise input port, the port must have kGaussian random
  /// type. Defaults to additive measurement noise.
  std::optional<InputPortIndex> measurement_noise_input_port_index{
      std::nullopt};

  /// For systems with multiple output ports, we must specify the measurement
  /// output port.
  std::variant<systems::OutputPortSelection, OutputPortIndex>
      measurement_output_port_index{
          systems::OutputPortSelection::kUseFirstOutputIfItExists};

  /// The initial state estimate.
  std::optional<Eigen::VectorXd> initial_state_estimate;

  /// The initial state covariance.
  std::optional<Eigen::MatrixXd> initial_state_covariance;

  // For observed system represented as continuous-time models while
  // discrete-time measurements are made periodically for state estimation via a
  // digital processor, specify the discrete time period here.
  std::optional<double> discrete_measurement_time_period{std::nullopt};

  // In conjunction with the above, sepcify the discrete time offset here.
  double discrete_measurement_time_offset{0.0};
};

}  // namespace estimators
}  // namespace systems
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class drake::systems::estimators::GaussianStateObserver);
