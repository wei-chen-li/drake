#include "drake/multibody/der/damping_model.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
DampingModel<T>::DampingModel(const T& mass_coeff_alpha,
                              const T& stiffness_coeff_beta)
    : mass_coeff_alpha_(mass_coeff_alpha),
      stiffness_coeff_beta_(stiffness_coeff_beta) {
  DRAKE_THROW_UNLESS(mass_coeff_alpha >= 0.0);
  DRAKE_THROW_UNLESS(stiffness_coeff_beta >= 0.0);
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::der::internal::DampingModel);
