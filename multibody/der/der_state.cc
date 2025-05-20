#include "drake/multibody/der/der_state.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
DerState<T>::DerState(const DerStateSystem<T>* der_state_system)
    : DerState(der_state_system, der_state_system
                                     ? der_state_system->CreateDefaultContext()
                                     : nullptr) {}

template <typename T>
DerState<T>::DerState(const DerStateSystem<T>* der_state_system,
                      std::unique_ptr<Context<T>> context)
    : der_state_system_(der_state_system), context_(std::move(context)) {
  DRAKE_THROW_UNLESS(der_state_system_ != nullptr);
  DRAKE_THROW_UNLESS(context_ != nullptr);
  der_state_system_->ValidateContext(*context_);
}

template <typename T>
void DerState<T>::CopyFrom(const DerState<T>& other) {
  DRAKE_THROW_UNLESS(this->der_state_system_ == other.der_state_system_);
  der_state_system_->CopyContext(*other.context_, this->context_.get_mutable());
}

template <typename T>
std::unique_ptr<DerState<T>> DerState<T>::Clone() const {
  return std::unique_ptr<DerState<T>>(
      new DerState<T>(der_state_system_, context_->Clone()));
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DerState);
