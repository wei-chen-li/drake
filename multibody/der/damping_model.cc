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

template <typename T>
template <typename U>
DampingModel<U> DampingModel<T>::ToScalarType() const {
  static_assert(!std::is_same_v<T, U>);
  return DampingModel<U>(ExtractDoubleOrThrow(mass_coeff_alpha_),
                         ExtractDoubleOrThrow(stiffness_coeff_beta_));
}

using symbolic::Expression;
template DampingModel<AutoDiffXd>
DampingModel<double>::ToScalarType<AutoDiffXd>() const;
template DampingModel<Expression>
DampingModel<double>::ToScalarType<Expression>() const;
template DampingModel<double>  //
DampingModel<AutoDiffXd>::ToScalarType<double>() const;
template DampingModel<Expression>
DampingModel<AutoDiffXd>::ToScalarType<Expression>() const;
template DampingModel<double>  //
DampingModel<Expression>::ToScalarType<double>() const;
template DampingModel<AutoDiffXd>
DampingModel<Expression>::ToScalarType<AutoDiffXd>() const;

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DampingModel);
