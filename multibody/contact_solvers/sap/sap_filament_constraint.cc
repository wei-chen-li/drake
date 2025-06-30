#include "drake/multibody/contact_solvers/sap/sap_filament_constraint.h"

#include <limits>
#include <utility>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

template <typename T>
SapFilamentConstraint<T>::SapFilamentConstraint(
    FilamentConstraintKinematics<T> kinematics)
    : SapHolonomicConstraint<T>(
          typename SapHolonomicConstraint<T>::Kinematics(
              /* constraint function value */
              std::move(kinematics.q_PQs),
              /* Jacobian */ std::move(kinematics.J),
              /* Bias term */
              VectorX<T>::Zero(kinematics.num_constraints)),
          MakeSapHolonomicConstraintParameters(kinematics.num_constraints),
          MakeObjectIndices(kinematics)) {}

template <typename T>
typename SapHolonomicConstraint<T>::Parameters
SapFilamentConstraint<T>::MakeSapHolonomicConstraintParameters(
    int num_constraints) {
  /* "Near-rigid" regime parameter, see [Castro et al., 2022]. */
  // TODO(amcastro-tri): consider exposing this parameter.
  constexpr double kBeta = 0.1;

  /* Fixed constraints do not have impulse limits, they are bi-lateral
   constraints. */
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  VectorX<T> gamma_lower = VectorX<T>::Constant(num_constraints, -kInfinity);
  VectorX<T> gamma_upper = VectorX<T>::Constant(num_constraints, kInfinity);

  VectorX<T> stiffness = VectorX<T>::Constant(num_constraints, kInfinity);
  VectorX<T> relaxation_time = VectorX<T>::Zero(num_constraints);

  return typename SapHolonomicConstraint<T>::Parameters{
      std::move(gamma_lower), std::move(gamma_upper), std::move(stiffness),
      std::move(relaxation_time), kBeta};
}

template <typename T>
std::vector<int> SapFilamentConstraint<T>::MakeObjectIndices(
    const FilamentConstraintKinematics<T>& kinematics) {
  if (kinematics.objectB.has_value()) {
    return {kinematics.objectA, kinematics.objectB.value()};
  }
  return {kinematics.objectA};
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::SapFilamentConstraint);
