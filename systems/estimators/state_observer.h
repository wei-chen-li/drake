#pragma once

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace estimators {

/// A state observer takes the input and output of a plant and estimates the
/// plant's state.
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
class StateObserver : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(StateObserver);

  // Returns the input port that expects the input passed to the observed
  // system.
  virtual const InputPort<T>& get_observed_system_input_input_port() const = 0;

  // Returns the input port that expects the outputs of the observed system.
  virtual const InputPort<T>& get_observed_system_output_input_port() const = 0;

  // Returns the output port that provides the estimated state.
  virtual const OutputPort<T>& get_estimated_state_output_port() const = 0;

 protected:
  StateObserver();

  explicit StateObserver(SystemScalarConverter converter);
};

}  // namespace estimators
}  // namespace systems
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class drake::systems::estimators::StateObserver);
