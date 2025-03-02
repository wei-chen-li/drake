#include "drake/systems/estimators/unscented_kalman_filter.h"

#include <limits>
#include <optional>
#include <tuple>
#include <utility>

#include "drake/common/autodiff.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/matrix_util.h"

namespace drake {
namespace systems {
namespace estimators {

namespace {

// LeafSystem<double>                 [Inheritance map]
//           ↓
// GaussianStateObserver<double>
//           ↓
// UnscentedKalmanFilterBase ------→ UnscentedKalmanFilterCont
//           ↓                        ↓                     ↓
// UnscentedKalmanFilterDD  UnscentedKalmanFilterCC  UnscentedKalmanFilterDC

// Base class for all unscented Kalman filters.
class UnscentedKalmanFilterBase : public GaussianStateObserver<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(UnscentedKalmanFilterBase);

  UnscentedKalmanFilterBase(
      std::shared_ptr<const System<double>> observed_system,
      const Context<double>& observed_system_context,
      const Eigen::Ref<const Eigen::MatrixXd>& W,
      const Eigen::Ref<const Eigen::MatrixXd>& V,
      const GaussianStateObserverOptions& options)
      : observed_system_(std::move(observed_system)),
        W_(W),
        V_(V),
        num_states_(observed_system_context.num_total_states()) {
    observed_system_->ValidateContext(observed_system_context);
    DRAKE_THROW_UNLESS(num_states_ > 0);  // Or else we don't need an observer.
    DRAKE_THROW_UNLESS(
        observed_system_context.has_only_continuous_state() ||
        (observed_system_context.has_only_discrete_state() &&
         observed_system_context.num_discrete_state_groups() == 1));

    // Copy the observed system context into a cache entry where we can safely
    // modify it without runtime reallocation or a (non-thread-safe) mutable
    // member.
    observed_system_context_cache_entry_ = &this->DeclareCacheEntry(
        "observed system context",
        ValueProducer(observed_system_context, &ValueProducer::NoopCalc),
        {SystemBase::nothing_ticket()});

    // Check the observed system input/output ports and set the
    // observed_system_(.*)put_port_ member variables.
    this->CheckObservedSystemInputOutputPorts(*observed_system_, options);
    DRAKE_ASSERT(observed_system_actuation_input_port_ != nullptr);
    DRAKE_ASSERT(observed_system_measurement_output_port_ != nullptr);

    // First input port is the output of the observed system.
    const int y_size = observed_system_measurement_output_port_->size();
    this->DeclareVectorInputPort("observed_system_output", y_size);

    // Second input port is the input to the observed system.
    const int u_size = observed_system_actuation_input_port_->size();
    this->DeclareVectorInputPort("observed_system_input", u_size);

    // Check W, V, x̂₀, and P̂₀.
    const double kSymmetryTolerance = 1e-8;
    const int w_size = observed_system_process_noise_input_port_ != nullptr
                           ? observed_system_process_noise_input_port_->size()
                           : num_states_;
    DRAKE_THROW_UNLESS(W.rows() == w_size && W.cols() == w_size);
    DRAKE_THROW_UNLESS(math::IsPositiveDefinite(W, 0.0, kSymmetryTolerance));
    const int v_size =
        observed_system_measurement_noise_input_port_ != nullptr
            ? observed_system_measurement_noise_input_port_->size()
            : y_size;
    DRAKE_THROW_UNLESS(V.rows() == v_size && V.cols() == v_size);
    DRAKE_THROW_UNLESS(math::IsPositiveDefinite(
        V, std::numeric_limits<double>::epsilon(), kSymmetryTolerance));

    if (options.initial_state_estimate.has_value()) {
      DRAKE_THROW_UNLESS(options.initial_state_estimate->size() == num_states_);
    }
    if (options.initial_state_covariance.has_value()) {
      DRAKE_THROW_UNLESS(
          options.initial_state_covariance->rows() == num_states_ &&
          options.initial_state_covariance->cols() == num_states_);
      DRAKE_THROW_UNLESS(math::IsPositiveDefinite(
          *options.initial_state_covariance, 0.0, kSymmetryTolerance));
    }
  }

  const InputPort<double>& get_observed_system_input_input_port()
      const override {
    return this->get_input_port(1);
  }

  const InputPort<double>& get_observed_system_output_input_port()
      const override {
    return this->get_input_port(0);
  }

  const OutputPort<double>& get_estimated_state_output_port() const override {
    return this->get_output_port(0);
  }

 private:
  void CheckObservedSystemInputOutputPorts(
      const System<double>& system,
      const GaussianStateObserverOptions& options) {
    // Check observed system actuation input port.
    const InputPort<double>* actuation_input_port =
        system.get_input_port_selection(options.actuation_input_port_index);
    DRAKE_THROW_UNLESS(actuation_input_port != nullptr);
    DRAKE_THROW_UNLESS(actuation_input_port->get_data_type() == kVectorValued);
    DRAKE_THROW_UNLESS(actuation_input_port->size() > 0);
    observed_system_actuation_input_port_ = actuation_input_port;

    // Check observed system measurement output port.
    const OutputPort<double>* measurement_output_port =
        system.get_output_port_selection(options.measurement_output_port_index);
    DRAKE_THROW_UNLESS(measurement_output_port != nullptr);
    DRAKE_THROW_UNLESS(measurement_output_port->get_data_type() ==
                       kVectorValued);
    DRAKE_THROW_UNLESS(measurement_output_port->size() > 0);
    observed_system_measurement_output_port_ = measurement_output_port;

    // Check observed system process noise input port.
    if (options.process_noise_input_port_index.has_value()) {
      const InputPort<double>& process_noise_input_port =
          system.get_input_port(*options.process_noise_input_port_index);
      DRAKE_THROW_UNLESS(process_noise_input_port.get_index() !=
                         actuation_input_port->get_index());
      DRAKE_THROW_UNLESS(process_noise_input_port.get_data_type() ==
                         kVectorValued);
      DRAKE_THROW_UNLESS(process_noise_input_port.size() > 0);
      DRAKE_THROW_UNLESS(process_noise_input_port.is_random());
      DRAKE_THROW_UNLESS(process_noise_input_port.get_random_type().value() ==
                         RandomDistribution::kGaussian);
      DRAKE_THROW_UNLESS(
          !system.HasDirectFeedthrough(process_noise_input_port.get_index(),
                                       measurement_output_port->get_index()));
      observed_system_process_noise_input_port_ = &process_noise_input_port;
    }

    // Check observed system measurement noise input port.
    if (options.measurement_noise_input_port_index.has_value()) {
      const InputPort<double>& measurement_noise_input_port =
          system.get_input_port(*options.measurement_noise_input_port_index);
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
      DRAKE_THROW_UNLESS(
          measurement_noise_input_port.get_random_type().value() ==
          RandomDistribution::kGaussian);
      DRAKE_THROW_UNLESS(
          system.HasDirectFeedthrough(measurement_noise_input_port.get_index(),
                                      measurement_output_port->get_index()));
      observed_system_measurement_noise_input_port_ =
          &measurement_noise_input_port;
    }
  }

  // Cache entry for storing the context for the observed system.
  const CacheEntry* observed_system_context_cache_entry_{};
  // Input and output port of the observed system (will not be nullptr).
  const InputPort<double>* observed_system_actuation_input_port_{};
  const OutputPort<double>* observed_system_measurement_output_port_{};
  // Noise input ports of the observed system (may be nullptr indicating
  // additive noise).
  const InputPort<double>* observed_system_process_noise_input_port_{};
  const InputPort<double>* observed_system_measurement_noise_input_port_{};

 protected:
  // Returns the observed system context stored in the cache in the context.
  Context<double>& get_mutable_observed_system_context(
      const Context<double>& context) const {
    CacheEntryValue& cache_entry_value =
        observed_system_context_cache_entry_->get_mutable_cache_entry_value(
            context);
    Context<double>& observed_system_context =
        cache_entry_value.GetMutableValueOrThrow<Context<double>>();
    observed_system_context.SetTime(context.get_time());
    return observed_system_context;
  }

  // Evaluates the dynamics of the observed system:
  // ẋ = f(x,u,w) or x[n+1] = f(x[n],u,w).
  Eigen::VectorXd EvaluateDynamics(
      Context<double>* observed_system_context,
      const Eigen::Ref<const Eigen::VectorXd>& x,
      const Eigen::Ref<const Eigen::VectorXd>& u,
      const Eigen::Ref<const Eigen::VectorXd>& w) const {
    const bool is_continuous =
        observed_system_context->has_only_continuous_state();

    const bool use_additive_w =
        observed_system_process_noise_input_port_ == nullptr;

    // dyn = f(x,u,w)
    if (is_continuous) {
      observed_system_context->SetContinuousState(x);
    } else {
      observed_system_context->SetDiscreteState(x);
    }

    observed_system_actuation_input_port_->FixValue(observed_system_context, u);

    if (!use_additive_w) {
      observed_system_process_noise_input_port_->FixValue(
          observed_system_context, u);
    }

    Eigen::VectorXd dyn;
    if (is_continuous) {
      dyn = observed_system_->EvalTimeDerivatives(*observed_system_context)
                .CopyToVector();
    } else {
      dyn = observed_system_
                ->EvalUniquePeriodicDiscreteUpdate(*observed_system_context)
                .value();
    }
    if (use_additive_w) {
      dyn += w;
    }

    return dyn;
  }

  // Evaluates the measurement output of the observed system: y = g(x,u,v).
  Eigen::VectorXd EvaluateMeasurement(
      Context<double>* observed_system_context,
      const Eigen::Ref<const Eigen::VectorXd>& x,
      const Eigen::Ref<const Eigen::VectorXd>& u,
      const Eigen::Ref<const Eigen::VectorXd>& v) const {
    const bool is_continuous =
        observed_system_context->has_only_continuous_state();

    const bool use_additive_v =
        observed_system_measurement_noise_input_port_ == nullptr;

    // y = g(x,u,v)
    if (is_continuous) {
      observed_system_context->SetContinuousState(x);
    } else {
      observed_system_context->SetDiscreteState(x);
    }

    observed_system_actuation_input_port_->FixValue(observed_system_context, u);

    if (!use_additive_v) {
      observed_system_measurement_noise_input_port_->FixValue(
          observed_system_context, v);
    }

    Eigen::VectorXd y = observed_system_measurement_output_port_->Eval(
        *observed_system_context);
    if (use_additive_v) {
      y += v;
    }

    return y;
  }

  // Helper method for concatenating state_estimate and state_covariance
  // vectorized.
  static Eigen::VectorXd ConcatenateStateEstimateAndCovariance(
      const Eigen::Ref<const Eigen::VectorXd>& state_estimate,
      const Eigen::Ref<const Eigen::MatrixXd>& state_covariance) {
    DRAKE_ASSERT(state_estimate.size() == state_covariance.rows());
    DRAKE_ASSERT(state_covariance.rows() == state_covariance.cols());
    Eigen::VectorXd concatenated(state_estimate.size() +
                                 state_covariance.size());
    concatenated << state_estimate,
        Eigen::VectorXd::Map(state_covariance.data(), state_covariance.size());
    return concatenated;
  }

  // Helper method for spitting the concatenated vector into state_estimate and
  // state_covariance.
  static void ExtractStateEstimateAndCovariance(
      const Eigen::Ref<const Eigen::VectorXd>& storage,
      std::optional<Eigen::Ref<Eigen::VectorXd>> state_estimate,
      std::optional<Eigen::Ref<Eigen::MatrixXd>> state_covariance) {
    if (state_estimate) {
      DRAKE_ASSERT(storage.size() ==
                   state_estimate->size() * (state_estimate->size() + 1));
      *state_estimate = storage.head(state_estimate->size());
    }
    if (state_covariance) {
      const int num_states = state_covariance->rows();
      DRAKE_ASSERT(state_covariance->rows() == state_covariance->cols());
      DRAKE_ASSERT(storage.size() == num_states * (num_states + 1));
      *state_covariance = Eigen::MatrixXd::Map(storage.data() + num_states,
                                               num_states, num_states);
    }
  }

  const std::shared_ptr<const System<double>> observed_system_;
  const Eigen::MatrixXd W_;
  const Eigen::MatrixXd V_;
  const int num_states_;
};

// Unscented Kalman filter with discrete-time measurement and discrete-time
// observed system dynamics.
class UnscentedKalmanFilterDD final : public UnscentedKalmanFilterBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(UnscentedKalmanFilterDD);

  UnscentedKalmanFilterDD(std::shared_ptr<const System<double>> observed_system,
                          const Context<double>& observed_system_context,
                          const Eigen::Ref<const Eigen::MatrixXd>& W,
                          const Eigen::Ref<const Eigen::MatrixXd>& V,
                          const GaussianStateObserverOptions& options)
      : UnscentedKalmanFilterBase(std::move(observed_system),
                                  observed_system_context, W, V, options) {
    DRAKE_THROW_UNLESS(observed_system_->IsDifferenceEquationSystem());

    // Initial state estimate and covariance.
    Eigen::VectorXd initial_state_estimate =
        options.initial_state_estimate.has_value()
            ? options.initial_state_estimate.value()
            : Eigen::VectorXd::Zero(num_states_);
    Eigen::MatrixXd initial_state_covariance =
        options.initial_state_covariance.has_value()
            ? options.initial_state_covariance.value()
            : Eigen::MatrixXd::Zero(num_states_, num_states_);

    //  We declare only one discrete state containing both the estimated state
    //  and variance so that this observer will be IsDifferenceEquationSystem().
    this->DeclareDiscreteState(this->ConcatenateStateEstimateAndCovariance(
        initial_state_estimate, initial_state_covariance));

    // Declare estimated state output.
    this->DeclareVectorOutputPort("estimated_state", num_states_,
                                  &UnscentedKalmanFilterDD::CalcOutput,
                                  {SystemBase::all_state_ticket()});

    // Declare periodic update for the state estimate and covaraiance.
    DRAKE_THROW_UNLESS(
        observed_system_->GetUniquePeriodicDiscreteUpdateAttribute()
            .has_value());
    auto discrete_attr =
        observed_system_->GetUniquePeriodicDiscreteUpdateAttribute().value();
    this->DeclarePeriodicDiscreteUpdateEvent(
        discrete_attr.period_sec(), discrete_attr.offset_sec(),
        &UnscentedKalmanFilterDD::PeriodicDiscreteUpdate);

    if ((options.discrete_measurement_time_period.has_value() &&
         options.discrete_measurement_time_period.value() !=
             discrete_attr.period_sec()) ||
        (options.discrete_measurement_time_offset != 0.0 &&
         options.discrete_measurement_time_offset !=
             discrete_attr.offset_sec())) {
      throw std::logic_error(
          "Discrete-time unscented Kalman filter does not use the "
          "`discrete_measurement_time_period` and "
          "`discrete_measurement_time_offset` options.");
    }
  }

  void SetStateEstimateAndCovariance(
      Context<double>* context,
      const Eigen::Ref<const Eigen::VectorXd>& state_estimate,
      const Eigen::Ref<const Eigen::MatrixXd>& state_covariance)
      const override {
    this->ValidateContext(context);
    const double kSymmetryTolerance = 1e-8;
    DRAKE_THROW_UNLESS(state_estimate.size() == num_states_);
    DRAKE_THROW_UNLESS(state_covariance.rows() == num_states_ &&
                       state_covariance.cols() == num_states_);
    DRAKE_THROW_UNLESS(
        math::IsPositiveDefinite(state_covariance, 0.0, kSymmetryTolerance));
    context->SetDiscreteState(this->ConcatenateStateEstimateAndCovariance(
        state_estimate, state_covariance));
  }

  Eigen::VectorXd GetStateEstimate(
      const Context<double>& context) const override {
    this->ValidateContext(context);
    Eigen::VectorXd state_estimate(num_states_);
    this->ExtractStateEstimateAndCovariance(
        context.get_discrete_state_vector().value(), state_estimate,
        std::nullopt);
    return state_estimate;
  }

  Eigen::MatrixXd GetStateCovariance(
      const Context<double>& context) const override {
    this->ValidateContext(context);
    Eigen::MatrixXd state_covariance(num_states_, num_states_);
    this->ExtractStateEstimateAndCovariance(
        context.get_discrete_state_vector().value(), std::nullopt,
        state_covariance);
    return state_covariance;
  }

 private:
  void CalcOutput(const Context<double>& context,
                  BasicVector<double>* out) const {
    out->SetFromVector(this->GetStateEstimate(context));
  }

  void PeriodicDiscreteUpdate(const Context<double>& context,
                              DiscreteValues<double>* discrete_state) const {
    // Get the mutable observed system context from the cache.
    Context<double>* observed_system_context =
        &this->get_mutable_observed_system_context(context);
    observed_system_context->get_time();

    // Get the current state estimate and covariance.
    Eigen::VectorXd xhat = GetStateEstimate(context);
    Eigen::MatrixXd Phat = GetStateCovariance(context);

    // Measurement and process update.
    // std::tie(xhat, Phat) =
    //     MeasurementUpdate(context, observed_system_context, xhat, Phat);
    // std::tie(xhat, Phat) =
    //     ProcessUpdate(context, observed_system_context, xhat, Phat);

    discrete_state->set_value(
        this->ConcatenateStateEstimateAndCovariance(xhat, Phat));
  }
};

}  // namespace

std::unique_ptr<GaussianStateObserver<double>> UnscentedKalmanFilter(
    std::shared_ptr<const System<double>> observed_system,
    const Context<double>& observed_system_context,
    const Eigen::Ref<const Eigen::MatrixXd>& W,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const GaussianStateObserverOptions& options) {
  if (observed_system->IsDifferenceEquationSystem()) {
    return std::make_unique<UnscentedKalmanFilterDD>(
        std::move(observed_system), observed_system_context, W, V, options);
  } else {
    throw std::logic_error(
        "UnscentedKalmanFilter only supports observed_system where "
        "observed_system->IsDifferentialEquationSystem() or "
        "observed_system->IsDifferenceEquationSystem() is true");
  }
}

std::unique_ptr<GaussianStateObserver<double>> UnscentedKalmanFilter(
    const System<double>& observed_system,
    const Context<double>& observed_system_context,
    const Eigen::Ref<const Eigen::MatrixXd>& W,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const GaussianStateObserverOptions& options) {
  return UnscentedKalmanFilter(
      std::shared_ptr<const System<double>>(
          /* managed object = */ std::shared_ptr<void>{},
          /* stored pointer = */ &observed_system),
      observed_system_context, W, V, options);
}

}  // namespace estimators
}  // namespace systems
}  // namespace drake
