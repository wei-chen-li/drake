#include "drake/systems/estimators/extended_kalman_filter.h"

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
// ExtendedKalmanFilterBase -------→ ExtendedKalmanFilterCont
//           ↓                        ↓                    ↓
// ExtendedKalmanFilterDD  ExtendedKalmanFilterCC  ExtendedKalmanFilterDC

// Base class for all extended Kalman filters.
class ExtendedKalmanFilterBase : public GaussianStateObserver<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExtendedKalmanFilterBase);

  ExtendedKalmanFilterBase(const System<double>& observed_system,
                           const Context<double>& observed_system_context,
                           const Eigen::Ref<const Eigen::MatrixXd>& W,
                           const Eigen::Ref<const Eigen::MatrixXd>& V,
                           const GaussianStateObserverOptions& options)
      : observed_system_(observed_system.ToAutoDiffXd()),
        W_(W),
        V_(V),
        num_states_(observed_system_context.num_total_states()) {
    DRAKE_THROW_UNLESS(num_states_ > 0);  // Or else we don't need an observer.
    DRAKE_THROW_UNLESS(
        observed_system_context.has_only_continuous_state() ||
        (observed_system_context.has_only_discrete_state() &&
         observed_system_context.num_discrete_state_groups() == 1));

    // Create autodiff observed system context.
    auto observed_system_context_ad = observed_system_->CreateDefaultContext();
    observed_system_context_ad->SetTimeStateAndParametersFrom(
        observed_system_context);

    // Copy the autodiff observed system context into a cache entry where we can
    // safely modify it without runtime reallocation or a (non-thread-safe)
    // mutable member.
    observed_system_context_cache_entry_ = &this->DeclareCacheEntry(
        "observed system context",
        ValueProducer(*observed_system_context_ad, &ValueProducer::NoopCalc),
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
      const System<AutoDiffXd>& system,
      const GaussianStateObserverOptions& options) {
    // Check observed system actuation input port.
    const InputPort<AutoDiffXd>* actuation_input_port =
        system.get_input_port_selection(options.actuation_input_port_index);
    DRAKE_THROW_UNLESS(actuation_input_port != nullptr);
    DRAKE_THROW_UNLESS(actuation_input_port->get_data_type() == kVectorValued);
    DRAKE_THROW_UNLESS(actuation_input_port->size() > 0);
    observed_system_actuation_input_port_ = actuation_input_port;

    // Check observed system measurement output port.
    const OutputPort<AutoDiffXd>* measurement_output_port =
        system.get_output_port_selection(options.measurement_output_port_index);
    DRAKE_THROW_UNLESS(measurement_output_port != nullptr);
    DRAKE_THROW_UNLESS(measurement_output_port->get_data_type() ==
                       kVectorValued);
    DRAKE_THROW_UNLESS(measurement_output_port->size() > 0);
    observed_system_measurement_output_port_ = measurement_output_port;

    // Check observed system process noise input port.
    if (options.process_noise_input_port_index.has_value()) {
      const InputPort<AutoDiffXd>& process_noise_input_port =
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
      const InputPort<AutoDiffXd>& measurement_noise_input_port =
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
  const InputPort<AutoDiffXd>* observed_system_actuation_input_port_{};
  const OutputPort<AutoDiffXd>* observed_system_measurement_output_port_{};
  // Noise input ports of the observed system (may be nullptr indicating
  // additive noise).
  const InputPort<AutoDiffXd>* observed_system_process_noise_input_port_{};
  const InputPort<AutoDiffXd>* observed_system_measurement_noise_input_port_{};

 protected:
  // Returns the observed system context stored in the cache in the context.
  Context<AutoDiffXd>& get_mutable_observed_system_context(
      const Context<double>& context) const {
    CacheEntryValue& cache_entry_value =
        observed_system_context_cache_entry_->get_mutable_cache_entry_value(
            context);
    Context<AutoDiffXd>& observed_system_context =
        cache_entry_value.GetMutableValueOrThrow<Context<AutoDiffXd>>();
    observed_system_context.SetTime(context.get_time());
    return observed_system_context;
  }

  // Evaluates the dynamics of the observed system:
  // ẋ = f(x,u,w=0) or x[n+1] = f(x[n],u,w=0).
  // Also calculates the Jacobians: ∂f/∂x and ∂f/∂w.
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd>
  LinearizeDynamics(Context<AutoDiffXd>* observed_system_context,
                    const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u) const {
    const bool is_continuous =
        observed_system_context->has_only_continuous_state();

    const bool use_additive_w =
        observed_system_process_noise_input_port_ == nullptr;
    const int w_size = use_additive_w
                           ? num_states_
                           : observed_system_process_noise_input_port_->size();

    // dyn = f(x,u,w=0)
    Eigen::VectorXd w = Eigen::VectorXd::Zero(w_size);
    auto autodiff_args = math::InitializeAutoDiffTuple(x, u, w);

    if (is_continuous) {
      observed_system_context->SetContinuousState(std::get<0>(autodiff_args));
    } else {
      observed_system_context->SetDiscreteState(std::get<0>(autodiff_args));
    }

    observed_system_actuation_input_port_->FixValue(observed_system_context,
                                                    std::get<1>(autodiff_args));

    if (!use_additive_w) {
      observed_system_process_noise_input_port_->FixValue(
          observed_system_context, std::get<2>(autodiff_args));
    }
    Eigen::VectorX<AutoDiffXd> dyn_ad;
    if (is_continuous) {
      dyn_ad = observed_system_->EvalTimeDerivatives(*observed_system_context)
                   .CopyToVector();
    } else {
      dyn_ad = observed_system_
                   ->EvalUniquePeriodicDiscreteUpdate(*observed_system_context)
                   .value();
    }
    if (use_additive_w) {
      dyn_ad = dyn_ad + std::get<2>(autodiff_args);
    }
    Eigen::VectorXd dyn = math::ExtractValue(dyn_ad);

    // A = ∂f/∂x(x,u,w)
    Eigen::MatrixXd ABG = math::ExtractGradient(dyn_ad);
    Eigen::MatrixXd A = ABG.leftCols(num_states_);

    // G = ∂f/∂w(x,u,w)
    Eigen::MatrixXd G;
    if (use_additive_w) {
      G = Eigen::MatrixXd::Identity(w_size, w_size);
    } else {
      G = ABG.rightCols(w_size);
    }

    return std::make_tuple(dyn, A, G);
  }

  // Evaluates the measurement output of the observed system: y = g(x,u,v=0).
  // Also calculates the Jacobians: ∂g/∂x and ∂g/∂v.
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd>
  LinearizeMeasurement(Context<AutoDiffXd>* observed_system_context,
                       const Eigen::Ref<const Eigen::VectorXd>& x,
                       const Eigen::Ref<const Eigen::VectorXd>& u) const {
    const bool is_continuous =
        observed_system_context->has_only_continuous_state();

    const bool use_additive_v =
        observed_system_measurement_noise_input_port_ == nullptr;
    const int v_size =
        use_additive_v ? observed_system_measurement_output_port_->size()
                       : observed_system_measurement_noise_input_port_->size();

    // y = g(x,u,v)
    Eigen::VectorXd v = Eigen::VectorXd::Zero(v_size);
    auto autodiff_args = math::InitializeAutoDiffTuple(x, u, v);

    if (is_continuous) {
      observed_system_context->SetContinuousState(std::get<0>(autodiff_args));
    } else {
      observed_system_context->SetDiscreteState(std::get<0>(autodiff_args));
    }

    observed_system_actuation_input_port_->FixValue(observed_system_context,
                                                    std::get<1>(autodiff_args));

    if (!use_additive_v) {
      observed_system_measurement_noise_input_port_->FixValue(
          observed_system_context, std::get<2>(autodiff_args));
    }

    const Eigen::VectorX<AutoDiffXd>& y_ad =
        observed_system_measurement_output_port_->Eval(
            *observed_system_context);
    const Eigen::VectorXd y = math::ExtractValue(y_ad);

    // C = ∂g/∂x(x,u,v)
    Eigen::MatrixXd CDH = math::ExtractGradient(y_ad);
    Eigen::MatrixXd C = CDH.leftCols(num_states_);

    // H = ∂g/∂v(x,u,v)
    Eigen::MatrixXd H;
    if (use_additive_v) {
      H = Eigen::MatrixXd::Identity(v_size, v_size);
    } else {
      H = CDH.rightCols(v_size);
    }

    return std::make_tuple(y, C, H);
  }

  // The Kalman discrete-time measurement update.
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> MeasurementUpdate(
      const Context<double>& context,
      Context<AutoDiffXd>* observed_system_context,
      const Eigen::Ref<const Eigen::VectorXd>& xhat,
      const Eigen::Ref<const Eigen::MatrixXd>& Phat) const {
    const Eigen::VectorXd& u =
        this->get_observed_system_input_input_port().Eval(context);
    const Eigen::VectorXd& y =
        this->get_observed_system_output_input_port().Eval(context);

    // ŷ = g(x̂,u,v=0), C = ∂g/∂x, H = ∂g/∂v
    auto [yhat, C, H] = LinearizeMeasurement(observed_system_context, xhat, u);

    // K = P̂C'(CP̂C' + HVH')⁻¹
    Eigen::MatrixXd K =
        Phat * C.transpose() *
        (C * Phat * C.transpose() + H * V_ * H.transpose()).inverse();

    // x̂ ← x̂ + K (y − ŷ)
    Eigen::VectorXd xhat_new = xhat + K * (y - yhat);

    // P̂ ← (I - KC) P̂
    Eigen::MatrixXd Phat_new =
        (Eigen::MatrixXd::Identity(num_states_, num_states_) - K * C) * Phat;

    return std::make_pair(xhat_new, Phat_new);
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
      DRAKE_ASSERT(state_covariance->rows() == state_covariance->cols());
      DRAKE_ASSERT(storage.size() ==
                   state_covariance->rows() * (state_covariance->rows() + 1));
      *state_covariance = Eigen::MatrixXd::Map(
          storage.data() + state_covariance->rows(), state_covariance->rows(),
          state_covariance->cols());
    }
  }

  const std::unique_ptr<const System<AutoDiffXd>> observed_system_;
  const Eigen::MatrixXd W_;
  const Eigen::MatrixXd V_;
  const int num_states_;
};

// Base class for extended Kalman filter with continuous-time obsereved system
// dynamics.
class ExtendedKalmanFilterCont : public ExtendedKalmanFilterBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExtendedKalmanFilterCont);

  ExtendedKalmanFilterCont(const System<double>& observed_system,
                           const Context<double>& observed_system_context,
                           const Eigen::Ref<const Eigen::MatrixXd>& W,
                           const Eigen::Ref<const Eigen::MatrixXd>& V,
                           const GaussianStateObserverOptions& options)
      : ExtendedKalmanFilterBase(observed_system, observed_system_context, W, V,
                                 options) {
    DRAKE_THROW_UNLESS(observed_system.IsDifferentialEquationSystem());

    // Initial state estimate and covariance.
    Eigen::VectorXd initial_state_estimate =
        options.initial_state_estimate.has_value()
            ? options.initial_state_estimate.value()
            : Eigen::VectorXd::Zero(num_states_);
    Eigen::MatrixXd initial_state_covariance =
        options.initial_state_covariance.has_value()
            ? options.initial_state_covariance.value()
            : Eigen::MatrixXd::Zero(num_states_, num_states_);

    // Declare estimated state and variance.
    const auto& xc = observed_system_context.get_continuous_state();
    const int num_q = xc.get_generalized_position().size();
    const int num_v = xc.get_generalized_velocity().size();
    const int num_z = xc.get_misc_continuous_state().size();
    this->DeclareContinuousState(
        BasicVector<double>(this->ConcatenateStateEstimateAndCovariance(
            initial_state_estimate, initial_state_covariance)),
        num_q, num_v, num_z + num_states_ * num_states_);

    // Declare estimated state output.
    this->DeclareVectorOutputPort("estimated_state", num_states_,
                                  &ExtendedKalmanFilterCont::CalcOutput,
                                  {SystemBase::all_state_ticket()});
  }

  void SetStateEstimateAndCovariance(
      Context<double>* context,
      const Eigen::Ref<const Eigen::VectorXd>& state_estimate,
      const Eigen::Ref<const Eigen::MatrixXd>& state_covariance)
      const override {
    const double kSymmetryTolerance = 1e-8;
    DRAKE_THROW_UNLESS(state_estimate.size() == num_states_);
    DRAKE_THROW_UNLESS(state_covariance.rows() == num_states_ &&
                       state_covariance.cols() == num_states_);
    DRAKE_THROW_UNLESS(
        math::IsPositiveDefinite(state_covariance, 0.0, kSymmetryTolerance));
    context->SetContinuousState(this->ConcatenateStateEstimateAndCovariance(
        state_estimate, state_covariance));
  }

  Eigen::VectorXd GetStateEstimate(
      const Context<double>& context) const override {
    Eigen::VectorXd state_estimate(num_states_);
    this->ExtractStateEstimateAndCovariance(
        context.get_continuous_state_vector().CopyToVector(), state_estimate,
        std::nullopt);
    return state_estimate;
  }

  Eigen::MatrixXd GetStateCovariance(
      const Context<double>& context) const override {
    Eigen::MatrixXd state_covariance(num_states_, num_states_);
    this->ExtractStateEstimateAndCovariance(
        context.get_continuous_state_vector().CopyToVector(), std::nullopt,
        state_covariance);
    return state_covariance;
  }

 private:
  void CalcOutput(const Context<double>& context,
                  BasicVector<double>* out) const {
    out->SetFromVector(this->GetStateEstimate(context));
  }
};

// Extended Kalman filter with continuous-time measurement and continuous-time
// observed system dynamics.
class ExtendedKalmanFilterCC final : public ExtendedKalmanFilterCont {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExtendedKalmanFilterCC);

  ExtendedKalmanFilterCC(const System<double>& observed_system,
                         const Context<double>& observed_system_context,
                         const Eigen::Ref<const Eigen::MatrixXd>& W,
                         const Eigen::Ref<const Eigen::MatrixXd>& V,
                         const GaussianStateObserverOptions& options)
      : ExtendedKalmanFilterCont(observed_system, observed_system_context, W, V,
                                 options) {
    DRAKE_THROW_UNLESS(!options.discrete_measurement_time_period.has_value());
  }

  void DoCalcTimeDerivatives(
      const Context<double>& context,
      ContinuousState<double>* derivatives) const override {
    // Get the mutable observed system context from the cache.
    Context<AutoDiffXd>* observed_system_context =
        &this->get_mutable_observed_system_context(context);

    // Get the current state estimate and covariance.
    Eigen::VectorXd xhat = GetStateEstimate(context);
    Eigen::MatrixXd Phat = GetStateCovariance(context);

    // Get u and y.
    const Eigen::VectorXd& u =
        this->get_observed_system_input_input_port().Eval(context);
    const Eigen::VectorXd& y =
        this->get_observed_system_output_input_port().Eval(context);

    // x̂dot = f(x̂,u,w=0), A = ∂f/∂x, G = ∂f/∂w
    auto [xhatdot, A, G] = LinearizeDynamics(observed_system_context, xhat, u);
    // ŷ = g(x̂,u,v=0), C = ∂g/∂x, H = ∂g/∂v
    auto [yhat, C, H] = LinearizeMeasurement(observed_system_context, xhat, u);

    Eigen::MatrixXd HVH_inv = (H * V_ * H.transpose()).inverse();
    Eigen::MatrixXd PhatC = Phat * C.transpose();

    // dP̂/dt = AP̂ + P̂A' + GWG' - P̂C'(HVH')⁻¹CP̂
    Eigen::MatrixXd Phat_deriv = A * Phat + Phat * A.transpose() +
                                 G * W_ * G.transpose() -
                                 PhatC * HVH_inv * PhatC.transpose();
    // dx̂/dt = f(x̂,u) + P̂C'(HVH')⁻¹(y - g(x̂,u))
    Eigen::VectorXd xhat_deriv = xhatdot + PhatC * HVH_inv * (y - yhat);

    derivatives->SetFromVector(
        this->ConcatenateStateEstimateAndCovariance(xhat_deriv, Phat_deriv));
  }
};

// Extended Kalman filter with discrete-time measurement and continuous-time
// observed system dynamics.
class ExtendedKalmanFilterDC final : public ExtendedKalmanFilterCont {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExtendedKalmanFilterDC);

  ExtendedKalmanFilterDC(const System<double>& observed_system,
                         const Context<double>& observed_system_context,
                         const Eigen::Ref<const Eigen::MatrixXd>& W,
                         const Eigen::Ref<const Eigen::MatrixXd>& V,
                         const GaussianStateObserverOptions& options)
      : ExtendedKalmanFilterCont(observed_system, observed_system_context, W, V,
                                 options) {
    DRAKE_THROW_UNLESS(options.discrete_measurement_time_period.has_value());
    this->DeclarePeriodicUnrestrictedUpdateEvent(
        options.discrete_measurement_time_period.value(),
        options.discrete_measurement_time_offset,
        &ExtendedKalmanFilterDC::PeriodicDiscreteUpdate);
  }

  void DoCalcTimeDerivatives(
      const Context<double>& context,
      ContinuousState<double>* derivatives) const override {
    // Get the mutable observed system context from the cache.
    Context<AutoDiffXd>* observed_system_context =
        &this->get_mutable_observed_system_context(context);

    // Get the current state estimate and covariance.
    Eigen::VectorXd xhat = GetStateEstimate(context);
    Eigen::MatrixXd Phat = GetStateCovariance(context);

    // Get u.
    const Eigen::VectorXd& u =
        this->get_observed_system_input_input_port().Eval(context);

    // dx̂dt = f(x̂,u,w=0), A = ∂f/∂x, G = ∂f/∂w
    auto [xhat_deriv, A, G] =
        LinearizeDynamics(observed_system_context, xhat, u);

    // dP̂/dt = AP̂ + P̂A' + GWG'
    Eigen::MatrixXd Phat_deriv =
        A * Phat + Phat * A.transpose() + G * W_ * G.transpose();

    derivatives->SetFromVector(
        this->ConcatenateStateEstimateAndCovariance(xhat_deriv, Phat_deriv));
  }

 private:
  void PeriodicDiscreteUpdate(const Context<double>& context,
                              State<double>* state) const {
    // Get the mutable observed system context from the cache.
    Context<AutoDiffXd>* observed_system_context =
        &this->get_mutable_observed_system_context(context);

    // Get the current state estimate and covariance.
    Eigen::VectorXd xhat = GetStateEstimate(context);
    Eigen::MatrixXd Phat = GetStateCovariance(context);

    // Measurement update.
    std::tie(xhat, Phat) =
        MeasurementUpdate(context, observed_system_context, xhat, Phat);

    state->get_mutable_continuous_state().SetFromVector(
        this->ConcatenateStateEstimateAndCovariance(xhat, Phat));
  }
};

// Extended Kalman filter with discrete-time measurement and discrete-time
// observed system dynamics.
class ExtendedKalmanFilterDD final : public ExtendedKalmanFilterBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExtendedKalmanFilterDD);

  ExtendedKalmanFilterDD(const System<double>& observed_system,
                         const Context<double>& observed_system_context,
                         const Eigen::Ref<const Eigen::MatrixXd>& W,
                         const Eigen::Ref<const Eigen::MatrixXd>& V,
                         const GaussianStateObserverOptions& options)
      : ExtendedKalmanFilterBase(observed_system, observed_system_context, W, V,
                                 options) {
    DRAKE_THROW_UNLESS(observed_system.IsDifferenceEquationSystem());

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
                                  &ExtendedKalmanFilterDD::CalcOutput,
                                  {SystemBase::all_state_ticket()});

    // Declare periodic update for the state estimate and covaraiance.
    DRAKE_THROW_UNLESS(
        observed_system.GetUniquePeriodicDiscreteUpdateAttribute().has_value());
    auto discrete_attr =
        observed_system.GetUniquePeriodicDiscreteUpdateAttribute().value();
    this->DeclarePeriodicDiscreteUpdateEvent(
        discrete_attr.period_sec(), discrete_attr.offset_sec(),
        &ExtendedKalmanFilterDD::PeriodicDiscreteUpdate);

    if ((options.discrete_measurement_time_period.has_value() &&
         options.discrete_measurement_time_period.value() !=
             discrete_attr.period_sec()) ||
        (options.discrete_measurement_time_offset != 0.0 &&
         options.discrete_measurement_time_offset !=
             discrete_attr.offset_sec())) {
      throw std::logic_error(
          "Discrete-time extended Kalman filter does not use the "
          "`discrete_measurement_time_period` and "
          "`discrete_measurement_time_offset` options.");
    }
  }

  void SetStateEstimateAndCovariance(
      Context<double>* context,
      const Eigen::Ref<const Eigen::VectorXd>& state_estimate,
      const Eigen::Ref<const Eigen::MatrixXd>& state_covariance)
      const override {
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
    Eigen::VectorXd state_estimate(num_states_);
    this->ExtractStateEstimateAndCovariance(
        context.get_discrete_state_vector().value(), state_estimate,
        std::nullopt);
    return state_estimate;
  }

  Eigen::MatrixXd GetStateCovariance(
      const Context<double>& context) const override {
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
    Context<AutoDiffXd>* observed_system_context =
        &this->get_mutable_observed_system_context(context);

    // Get the current state estimate and covariance.
    Eigen::VectorXd xhat = GetStateEstimate(context);
    Eigen::MatrixXd Phat = GetStateCovariance(context);

    // Measurement and process update.
    std::tie(xhat, Phat) =
        MeasurementUpdate(context, observed_system_context, xhat, Phat);
    std::tie(xhat, Phat) =
        ProcessUpdate(context, observed_system_context, xhat, Phat);

    discrete_state->set_value(
        this->ConcatenateStateEstimateAndCovariance(xhat, Phat));
  }

  // The discrete-time Kalman process update.
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> ProcessUpdate(
      const Context<double>& context,
      Context<AutoDiffXd>* observed_system_context,
      const Eigen::Ref<const Eigen::VectorXd>& xhat,
      const Eigen::Ref<const Eigen::MatrixXd>& Phat) const {
    const Eigen::VectorXd& u =
        this->get_observed_system_input_input_port().Eval(context);

    // x̂[n+1] = f(x̂[n],u,w=0), A = ∂f/∂x, G = ∂f/∂w
    auto [xhat_next, A, G] =
        LinearizeDynamics(observed_system_context, xhat, u);

    // P̂[n+1] = AP̂A' + GWG'
    Eigen::MatrixXd Phat_next =
        A * Phat * A.transpose() + G * W_ * G.transpose();

    return std::make_pair(xhat_next, Phat_next);
  }
};

}  // namespace

std::unique_ptr<GaussianStateObserver<double>> ExtendedKalmanFilter(
    const System<double>& observed_system,
    const Context<double>& observed_system_context,
    const Eigen::Ref<const Eigen::MatrixXd>& W,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const GaussianStateObserverOptions& options) {
  if (observed_system.IsDifferentialEquationSystem()) {
    if (!options.discrete_measurement_time_period.has_value()) {
      return std::make_unique<ExtendedKalmanFilterCC>(
          observed_system, observed_system_context, W, V, options);
    } else {
      return std::make_unique<ExtendedKalmanFilterDC>(
          observed_system, observed_system_context, W, V, options);
    }
  } else if (observed_system.IsDifferenceEquationSystem()) {
    return std::make_unique<ExtendedKalmanFilterDD>(
        observed_system, observed_system_context, W, V, options);
  } else {
    throw std::logic_error(
        "ExtendedKalmanFilter only supports observed_system where "
        "observed_system->IsDifferentialEquationSystem() or "
        "observed_system->IsDifferenceEquationSystem() is true");
  }
}

}  // namespace estimators
}  // namespace systems
}  // namespace drake
