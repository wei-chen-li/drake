#include "drake/multibody/der/der_structural_property.h"

#include <math.h>

#include "drake/common/drake_throw.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
DerStructuralProperty<T> DerStructuralProperty<T>::FromRectangularCrossSection(
    const T& width, const T& height, const T& youngs_modulus,
    const T& shear_modulus, const T& mass_density) {
  const T A = width * height;
  const T I1 = A * width * width / 12;
  const T I2 = A * height * height / 12;
  return DerStructuralProperty<T>(youngs_modulus, shear_modulus, mass_density,
                                  A, I1, I2);
}

template <typename T>
DerStructuralProperty<T> DerStructuralProperty<T>::FromEllipticalCrossSection(
    const T& a, const T& b, const T& youngs_modulus, const T& shear_modulus,
    const T& mass_density) {
  const T A = M_PI * a * b;
  const T I1 = A * a * a / 4;
  const T I2 = A * b * b / 4;
  return DerStructuralProperty<T>(youngs_modulus, shear_modulus, mass_density,
                                  A, I1, I2);
}

template <typename T>
DerStructuralProperty<T> DerStructuralProperty<T>::FromCircularCrossSection(
    const T& r, const T& youngs_modulus, const T& shear_modulus,
    const T& mass_density) {
  return DerStructuralProperty<T>::FromEllipticalCrossSection(
      r, r, youngs_modulus, shear_modulus, mass_density);
}

template <typename T>
DerStructuralProperty<T>::DerStructuralProperty(const T& E, const T& G,
                                                const T& rho, const T& A,
                                                const T& I1, const T& I2)
    : E_(E), G_(G), rho_(rho), A_(A), I1_(I1), I2_(I2), J_(I1 + I2) {
  DRAKE_THROW_UNLESS(E > 0);
  DRAKE_THROW_UNLESS(G > 0);
  DRAKE_THROW_UNLESS(rho > 0);
  DRAKE_THROW_UNLESS(A > 0);
  DRAKE_THROW_UNLESS(I1 > 0);
  DRAKE_THROW_UNLESS(I2 > 0);
}

template <typename T>
template <typename U>
DerStructuralProperty<U> DerStructuralProperty<T>::ToScalarType() const {
  static_assert(!std::is_same_v<T, U>);
  return DerStructuralProperty<U>(
      ExtractDoubleOrThrow(E_), ExtractDoubleOrThrow(G_),
      ExtractDoubleOrThrow(rho_), ExtractDoubleOrThrow(A_),
      ExtractDoubleOrThrow(I1_), ExtractDoubleOrThrow(I2_));
}

using symbolic::Expression;
template DerStructuralProperty<AutoDiffXd>
DerStructuralProperty<double>::ToScalarType<AutoDiffXd>() const;
template DerStructuralProperty<Expression>
DerStructuralProperty<double>::ToScalarType<Expression>() const;
template DerStructuralProperty<double>
DerStructuralProperty<AutoDiffXd>::ToScalarType<double>() const;
template DerStructuralProperty<Expression>
DerStructuralProperty<AutoDiffXd>::ToScalarType<Expression>() const;
template DerStructuralProperty<double>
DerStructuralProperty<Expression>::ToScalarType<double>() const;
template DerStructuralProperty<AutoDiffXd>
DerStructuralProperty<Expression>::ToScalarType<AutoDiffXd>() const;

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DerStructuralProperty);
