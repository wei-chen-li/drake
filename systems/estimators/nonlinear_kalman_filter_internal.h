#pragma once

#include <exception>
#include <tuple>
#include <utility>

#include "drake/systems/framework/system.h"

namespace drake {
namespace systems {
namespace estimators {
namespace internal {

/**
 * Concatenate @p vector with the vectorized @p square_matrix.
 */
Eigen::VectorXd ConcatenateVectorAndSquareMatrix(
    const Eigen::Ref<const Eigen::VectorXd>& vector,
    const Eigen::Ref<const Eigen::MatrixXd>& square_matrix);

/**
 * Concatenate @p vector with the vectorized @p lower_tri_matrix.
 */
Eigen::VectorXd ConcatenateVectorAndLowerTriMatrix(
    const Eigen::Ref<const Eigen::VectorXd>& vector,
    const Eigen::Ref<const Eigen::MatrixXd>& lower_tri_matrix);

/**
 * Concatenate @p vector with the vectorized @p lower_tri_matrix.
 */
Eigen::VectorXd ConcatenateVectorAndLowerTriMatrix(
    const Eigen::Ref<const Eigen::VectorXd>& vector,
    const Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Lower>&
        lower_tri_matrix);

/**
 * Extract the @p square_matrix from the @p concatenated vector returned by
 * @ref ConcatenateVectorAndSquareMatrix.
 */
void ExtractSquareMatrix(const Eigen::Ref<const Eigen::VectorXd>& concatenated,
                         Eigen::Ref<Eigen::MatrixXd> square_matrix);

/**
 * Extract the @p lower_tri_matrix from the @p concatenated vector returned by
 * @ref ConcatenateVectorAndLowerTriMatrix.
 */
void ExtractLowerTriMatrix(
    const Eigen::Ref<const Eigen::VectorXd>& concatenated,
    Eigen::Ref<Eigen::MatrixXd> lower_tri_matrix);

/**
 * Returns a lower triangular matrix L, where LL' = M₁M₁' + M₂M₂'.
 */
Eigen::MatrixXd MatrixHypot(const Eigen::Ref<const Eigen::MatrixXd>& M1,
                            const Eigen::Ref<const Eigen::MatrixXd>& M2);

/**
 * Check the @p observed_system for input and output ports according to
 * @p options.actuation_input_port_index,
 * @p options.measurement_output_port_index,
 * @p options.process_noise_input_port_index, and
 * @p options.measurement_noise_input_port_index. Check that the ports do not
 * repeat and have the correct types.
 *
 * @param observed_system [in] The observed system.
 * @param options [in] For port specification.
 * @param observed_system_actuation_input_port [out] Pointer to the actuation
 * input port (may be nullptr indicating no actuation input).
 * @param observed_system_measurement_output_port [out] Pointer to the
 * measurement output port (will not be nullptr).
 * @param observed_system_process_noise_input_port [out] Pointer to the
 * process noise output port (may be nullptr indicating additive noise).
 * @param observed_system_measurement_noise_input_port [out] Pointer to the
 * measurement noise output port (may be nullptr indicating additive noise).
 */
template <typename T, typename OptionsType>
void CheckObservedSystemInputOutputPorts(
    const System<T>& observed_system, const OptionsType& options,
    const InputPort<T>** observed_system_actuation_input_port,
    const OutputPort<T>** observed_system_measurement_output_port,
    const InputPort<T>** observed_system_process_noise_input_port,
    const InputPort<T>** observed_system_measurement_noise_input_port) {
  // Check observed system actuation input port.
  const InputPort<T>* actuation_input_port =
      observed_system.get_input_port_selection(
          options.actuation_input_port_index);
  if (observed_system_actuation_input_port) {
    *observed_system_actuation_input_port = actuation_input_port;
  }

  // Check observed system measurement output port.
  const OutputPort<T>* measurement_output_port =
      observed_system.get_output_port_selection(
          options.measurement_output_port_index);
  DRAKE_THROW_UNLESS(measurement_output_port != nullptr);
  if (measurement_output_port->get_data_type() == kAbstractValued) {
    throw std::logic_error(
        "The specified output port is abstract-valued, but Kalman filter only "
        "supports vector-valued output ports.  Did you perhaps forget to pass "
        "a non-default `measurement_output_port_index` argument?");
  }
  DRAKE_THROW_UNLESS(measurement_output_port->size() > 0);
  if (observed_system_measurement_output_port) {
    *observed_system_measurement_output_port = measurement_output_port;
  }

  // Check observed system process noise input port.
  if (options.process_noise_input_port_index.has_value()) {
    const InputPort<T>& process_noise_input_port =
        observed_system.get_input_port(*options.process_noise_input_port_index);
    DRAKE_THROW_UNLESS(process_noise_input_port.get_index() !=
                       actuation_input_port->get_index());
    DRAKE_THROW_UNLESS(process_noise_input_port.get_data_type() ==
                       kVectorValued);
    DRAKE_THROW_UNLESS(process_noise_input_port.size() > 0);
    DRAKE_THROW_UNLESS(process_noise_input_port.is_random());
    DRAKE_THROW_UNLESS(process_noise_input_port.get_random_type().value() ==
                       RandomDistribution::kGaussian);
    DRAKE_THROW_UNLESS(!observed_system.HasDirectFeedthrough(
        process_noise_input_port.get_index(),
        measurement_output_port->get_index()));
    if (observed_system_process_noise_input_port) {
      *observed_system_process_noise_input_port = &process_noise_input_port;
    }
  } else {
    if (observed_system_process_noise_input_port) {
      *observed_system_process_noise_input_port = nullptr;
    }
  }

  // Check observed system measurement noise input port.
  if (options.measurement_noise_input_port_index.has_value()) {
    const InputPort<T>& measurement_noise_input_port =
        observed_system.get_input_port(
            *options.measurement_noise_input_port_index);
    DRAKE_THROW_UNLESS(measurement_noise_input_port.get_index() !=
                       actuation_input_port->get_index());
    if (options.process_noise_input_port_index.has_value()) {
      DRAKE_THROW_UNLESS(measurement_noise_input_port.get_index() !=
                         *options.process_noise_input_port_index);
    }
    DRAKE_THROW_UNLESS(measurement_noise_input_port.get_data_type() ==
                       kVectorValued);
    DRAKE_THROW_UNLESS(measurement_noise_input_port.size() > 0);
    DRAKE_THROW_UNLESS(measurement_noise_input_port.is_random());
    DRAKE_THROW_UNLESS(measurement_noise_input_port.get_random_type().value() ==
                       RandomDistribution::kGaussian);
    DRAKE_THROW_UNLESS(observed_system.HasDirectFeedthrough(
        measurement_noise_input_port.get_index(),
        measurement_output_port->get_index()));
    if (observed_system_measurement_noise_input_port) {
      *observed_system_measurement_noise_input_port =
          &measurement_noise_input_port;
    }
  } else {
    if (observed_system_measurement_noise_input_port) {
      *observed_system_measurement_noise_input_port = nullptr;
    }
  }
}

}  // namespace internal
}  // namespace estimators
}  // namespace systems
}  // namespace drake
