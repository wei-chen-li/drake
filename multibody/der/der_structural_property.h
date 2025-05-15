#pragma once

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/*
 @p DerStructuralProperty holds properties regarding the Young's modulus, shear
 modulus, mass density, and cross section of a discrete elastic rod.

 @tparam_default_scalar
 */
template <typename T>
class DerStructuralProperty {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(DerStructuralProperty);

  /* Create a @p DerStructuralProperty corresponding to a rectangular cross
   section shown in the following figure.
   @verbatim
                  m₂
                  ↑
           +------+------+
           |      |      |
    height |      +------+-→ m₁
           |             |
           +-------------+
               width
   @endverbatim
   m₁ and m₂ are the orthonormal material frame directors perpendicular to the
   tangent. */
  static DerStructuralProperty<T> FromRectangularCrossSection(
      const T& width, const T& height, const T& youngs_modulus,
      const T& shear_modulus, const T& mass_density);

  /* Create a @p DerStructuralProperty corresponding to an elliptical cross
   section shown in the following figure.
   @verbatim
                  m₂
                  ↑
             +----+----+
            /     b     \
           |      +--a---+-→ m₁
            \           /
             +---------+
   @endverbatim
   m₁ and m₂ are the orthonormal material frame directors perpendicular to the
   tangent. */
  static DerStructuralProperty<T> FromEllipticalCrossSection(
      const T& a, const T& b, const T& youngs_modulus, const T& shear_modulus,
      const T& mass_density);

  /* Create a @p DerStructuralProperty corresponding to a circular cross
   section with radius @p r. */
  static DerStructuralProperty<T> FromCircularCrossSection(
      const T& r, const T& youngs_modulus, const T& shear_modulus,
      const T& mass_density);

  /* ∫dA. */
  const T& A() const { return A_; }
  /* Young's modulus times ∫dA. */
  T EA() const { return E_ * A_; }
  /* Young's modulus times ∫(p⋅m₁)²dA. */
  T EI1() const { return E_ * I1_; }
  /* Young's modulus times ∫(p⋅m₂)²dA. */
  T EI2() const { return E_ * I2_; }
  /* Shear modulus times ∫((p⋅m₁)²+(p⋅m₂)²)dA. */
  T GJ() const { return G_ * J_; }
  /* Mass density times ∫dA. */
  T rhoA() const { return rho_ * A_; }
  /* Mass density times ∫√((p⋅m₁)²+(p⋅m₂)²)dA. */
  T rhoJ() const { return rho_ * J_; }

  template <typename U>
  DerStructuralProperty<U> ToScalarType() const;

 private:
  template <typename U>
  friend class DerStructuralProperty;

  DerStructuralProperty(const T& E, const T& G, const T& rho, const T& A,
                        const T& I1, const T& I2);

  T E_{};
  T G_{};
  T rho_{};
  T A_{};
  T I1_{};
  T I2_{};
  T J_{};
};

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::DerStructuralProperty);
