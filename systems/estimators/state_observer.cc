#include "drake/systems/estimators/state_observer.h"

namespace drake {
namespace systems {
namespace estimators {

template <typename T>
StateObserver<T>::StateObserver() {}

template <typename T>
StateObserver<T>::StateObserver(SystemScalarConverter converter)
    : LeafSystem<T>(converter) {}

}  // namespace estimators
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::estimators::StateObserver);
