#pragma once

#include <vector>

#include "drake/multibody/der/der_state.h"
#include "drake/multibody/der/der_structural_property.h"
#include "drake/multibody/der/der_undeformed_state.h"
#include "drake/multibody/fem/force_density_field_base.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/* Forward declaration. */
template <typename T>
class ExternalForceVector;

/* ExternalForceField stores pointers to the owning MultibodyPlant's resources
 for a DerModel used to compute the external force on a DER model.
 ExternalForceField should not be persisted. It is advisable to acquire it for
 DER computation that depends on the owning MultibodyPlant's resources in a
 limited scope and then discard it. Constructing and populating an
 ExternalForceField is usually cheap.

 @tparam_default_scalar */
template <typename T>
class ExternalForceField {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ExternalForceField);

  ExternalForceField(
      const systems::Context<T>* plant_context,
      std::vector<const ForceDensityFieldBase<T>*> force_density_fields);

  /* Returns an object representing the external generalized force evaluated at
   the given `state`.
   @note The actual computation not performed until the returned
         ExternalForceVector is assigned to another Eigen::VectorX.
   @note The returned ExternalForceVector should not outlive this
         ExternalForceField.
   @pre `undeformed` and `state` are compatible. */
  ExternalForceVector<T> operator()(const DerStructuralProperty<T>& prop,
                                    const DerUndeformedState<T>& undeformed,
                                    const DerState<T>& state) const;

 private:
  const systems::Context<T>* plant_context_;
  std::vector<const ForceDensityFieldBase<T>*> force_density_fields_;
};

/* Class returned by ExternalForceField::ComputeForce(). */
template <typename T>
class ExternalForceVector {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ExternalForceVector);

  /* Computes and adds this external force vector to another Eigen::VectorX.
   @pre `other.size() == self.state_->num_dofs()`. */
  friend Eigen::Ref<VectorX<T>> operator+=(Eigen::Ref<VectorX<T>> other,
                                           const ExternalForceVector<T>& self) {
    return self.ScaleAndAddToVector(+1.0, &other);
  }

  /* Computes and subtracts this external force vector from another
   Eigen::VectorX.
   @pre `other.size() == self.state_->num_dofs()`. */
  friend Eigen::Ref<VectorX<T>> operator-=(Eigen::Ref<VectorX<T>> other,
                                           const ExternalForceVector<T>& self) {
    return self.ScaleAndAddToVector(-1.0, &other);
  }

  /* Computes and returns this external force vector as a Eigen::VectorX. */
  Eigen::VectorX<T> eval() const;

 private:
  friend class ExternalForceField<T>;

  ExternalForceVector(
      const systems::Context<T>* plant_context,
      const std::vector<const ForceDensityFieldBase<T>*>* force_density_fields,
      const DerStructuralProperty<T>* prop,
      const DerUndeformedState<T>* undeformed, const DerState<T>* state);

  /* Computes of the generalized force represented by this ExternalForceVector,
   and adds the `scale`d result to `other` Eigen::VectorX.
   @return `*other`.
   @pre `other != nullptr`.
   @pre `other.size() == state_->num_dofs()`. */
  Eigen::Ref<Eigen::VectorX<T>> ScaleAndAddToVector(
      const T& scale, EigenPtr<Eigen::VectorX<T>> other) const;

  const systems::Context<T>* plant_context_;
  const std::vector<const ForceDensityFieldBase<T>*>* force_density_fields_;
  const DerStructuralProperty<T>* prop_;
  const DerUndeformedState<T>* undeformed_;
  const DerState<T>* state_;
};

/* Computes the generalized mass matrix.
 @tparam_default_scalar */
template <typename T>
Eigen::DiagonalMatrix<T, Eigen::Dynamic> ComputeMassMatrix(
    const DerStructuralProperty<T>& prop,
    const DerUndeformedState<T>& undeformed);

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::ExternalForceField);
