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

}  // namespace estimators
}  // namespace systems
}  // namespace drake

template class drake::systems::estimators::GaussianStateObserver<double>;
