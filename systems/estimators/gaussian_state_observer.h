#pragma once

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace estimators {

/**
 A Gaussian state observer estimates the state of an observed system using its
 input and output. All probability distributions are approximated as Gaussian.

 @system
 name: GaussianStateObserver
 input_ports:
 - observed_system_input
 - observed_system_output
 output_ports:
 - estimated_state
 @endsystem

 @see ExtendedKalmanFilter()
 @see UnscentedKalmanFilter()

 @tparam_double_only
 @ingroup estimator_systems
 */
template <typename T>
class GaussianStateObserver : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GaussianStateObserver);

  /** Returns the input port that receives the observed system's input. */
  const InputPort<T>& get_observed_system_input_input_port() const;

  /** Returns the input port that receives the observed system's output. */
  const InputPort<T>& get_observed_system_output_input_port() const;

  /** Returns the output port that provides the estimated state. */
  const OutputPort<T>& get_estimated_state_output_port() const;

  /** Sets the state estimate and covariance in the given @p context. */
  void SetStateEstimateAndCovariance(
      Context<T>* context,
      const Eigen::Ref<const Eigen::VectorX<T>>& state_estimate,
      const Eigen::Ref<const Eigen::MatrixX<T>>& state_covariance) const;

  /** Gets the state estimate from the given @p context. */
  Eigen::VectorX<T> GetStateEstimate(const Context<T>& context) const;

  /** Gets the state covariance from the given @p context. */
  Eigen::MatrixX<T> GetStateCovariance(const Context<T>& context) const;

  ~GaussianStateObserver() override;

 protected:
  GaussianStateObserver();

  explicit GaussianStateObserver(SystemScalarConverter converter);

  /** Derived classes must override this method to implement the NVI
   get_observed_system_input_input_port(). */
  virtual const InputPort<T>& do_get_observed_system_input_input_port()
      const = 0;

  /** Derived classes must override this method to implement the NVI
   get_observed_system_output_input_port(). */
  virtual const InputPort<T>& do_get_observed_system_output_input_port()
      const = 0;

  /** Derived classes must override this method to implement the NVI
   get_estimated_state_output_port(). */
  virtual const OutputPort<T>& do_get_estimated_state_output_port() const = 0;

  /** Derived classes must override this method to implement the NVI
   SetStateEstimateAndCovariance(). */
  virtual void DoSetStateEstimateAndCovariance(
      Context<T>* context,
      const Eigen::Ref<const Eigen::VectorX<T>>& state_estimate,
      const Eigen::Ref<const Eigen::MatrixX<T>>& state_covariance) const = 0;

  /** Derived classes must override this method to implement the NVI
   GetStateEstimate(). */
  virtual Eigen::VectorX<T> DoGetStateEstimate(
      const Context<T>& context) const = 0;

  /** Derived classes must override this method to implement the NVI
   GetStateCovariance(). */
  virtual Eigen::MatrixX<T> DoGetStateCovariance(
      const Context<T>& context) const = 0;
};

}  // namespace estimators
}  // namespace systems
}  // namespace drake
