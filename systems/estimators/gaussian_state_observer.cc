#include "drake/systems/estimators/gaussian_state_observer.h"

namespace drake {
namespace systems {
namespace estimators {

template <typename T>
GaussianStateObserver<T>::GaussianStateObserver() {}

template <typename T>
GaussianStateObserver<T>::GaussianStateObserver(SystemScalarConverter converter)
    : LeafSystem<T>(converter) {}

template <typename T>
GaussianStateObserver<T>::~GaussianStateObserver() = default;

template <typename T>
const InputPort<T>&
GaussianStateObserver<T>::get_observed_system_input_input_port() const {
  return this->do_get_observed_system_input_input_port();
}

template <typename T>
const InputPort<T>&
GaussianStateObserver<T>::get_observed_system_output_input_port() const {
  return this->do_get_observed_system_output_input_port();
}

template <typename T>
const OutputPort<T>& GaussianStateObserver<T>::get_estimated_state_output_port()
    const {
  return this->do_get_estimated_state_output_port();
}

template <typename T>
void GaussianStateObserver<T>::SetStateEstimateAndCovariance(
    Context<T>* context,
    const Eigen::Ref<const Eigen::VectorX<T>>& state_estimate,
    const Eigen::Ref<const Eigen::MatrixX<T>>& state_covariance) const {
  this->ValidateContext(context);
  DRAKE_THROW_UNLESS(state_covariance.rows() == state_estimate.size() &&
                     state_covariance.cols() == state_estimate.size());
  this->DoSetStateEstimateAndCovariance(context, state_estimate,
                                        state_covariance);
}

template <typename T>
Eigen::VectorX<T> GaussianStateObserver<T>::GetStateEstimate(
    const Context<T>& context) const {
  this->ValidateContext(context);
  return this->DoGetStateEstimate(context);
}

template <typename T>
Eigen::MatrixX<T> GaussianStateObserver<T>::GetStateCovariance(
    const Context<T>& context) const {
  this->ValidateContext(context);
  return this->DoGetStateCovariance(context);
}

}  // namespace estimators
}  // namespace systems
}  // namespace drake

template class drake::systems::estimators::GaussianStateObserver<double>;
