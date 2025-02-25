#pragma once

#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/linear_system.h"

namespace drake {
namespace systems {
namespace estimators_test {

/**
 * A system that sums the entries of the matrix on the input port.
 *
 * @system
 * name: StochasticLinearSystem
 * input_ports:
 * - u0
 * output_ports:
 * - y0
 * @endsystem
 *
 * @tparam_default_scalar
 */
template <typename T>
class SumMatrixEntriesSystem : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SumMatrixEntriesSystem);

  SumMatrixEntriesSystem(int rows, int cols)
      : LeafSystem<T>(SystemTypeTag<SumMatrixEntriesSystem>{}),
        rows_(rows),
        cols_(cols) {
    Eigen::MatrixXd model_value = Eigen::MatrixXd::Zero(rows, cols);
    this->DeclareAbstractInputPort(kUseDefaultName, Value(model_value));
    this->DeclareVectorOutputPort(
        kUseDefaultName, 1,
        [this](const Context<T>& context, BasicVector<T>* out) {
          out->get_mutable_value()[0] =
              this->get_input_port()
                  .template Eval<Eigen::MatrixXd>(context)
                  .sum();
        });
  }

  // Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  SumMatrixEntriesSystem(const SumMatrixEntriesSystem<U>& other)
      : SumMatrixEntriesSystem(other.rows_, other.cols_) {}

 private:
  template <typename U>
  friend class SumMatrixEntriesSystem;

  const int rows_;
  const int cols_;
};

}  // namespace estimators_test
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::estimators_test::SumMatrixEntriesSystem);
