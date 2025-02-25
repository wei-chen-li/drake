#include "drake/systems/estimators/extended_kalman_filter.h"

namespace drake {
namespace systems {
namespace estimators {

std::unique_ptr<StateObserver<double>> ExtendedKalmanFilter(
    std::shared_ptr<const System<double>> system,
    const Context<double>& context, const Eigen::Ref<const Eigen::MatrixXd>& W,
    const Eigen::Ref<const Eigen::MatrixXd>& V) {
  return std::unique_ptr<StateObserver<double>>();
}

}  // namespace estimators
}  // namespace systems
}  // namespace drake
