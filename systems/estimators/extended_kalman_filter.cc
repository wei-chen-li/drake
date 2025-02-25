#include "drake/systems/estimators/extended_kalman_filter.h"

#include <limits>
#include <optional>
#include <tuple>
#include <utility>

#include "drake/common/autodiff.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/matrix_util.h"
#include "drake/systems/estimators/nonlinear_kalman_filter_internal.h"

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
// ExtendedKalmanFilterDD  ExtendedKalmanFilterCD  ExtendedKalmanFilterCC

// Base class for all extended Kalman filters.
class ExtendedKalmanFilterBase : public GaussianStateObserver<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExtendedKalmanFilterBase);

  ExtendedKalmanFilterBase(const System<double>& observed_system,
                           const Context<double>& observed_system_context,
                           const Eigen::Ref<const Eigen::MatrixXd>& W,
                           const Eigen::Ref<const Eigen::MatrixXd>& V,
                           const ExtendedKalmanFilterOptions& options)
      : observed_system_(observed_system.ToAutoDiffXd()),
        num_states_(observed_system_context.num_total_states()),
        use_sqrt_method_(options.use_square_root_method),
        W_(W),
        V_(V),
        Wsqrt_(use_sqrt_method_ ? Eigen::MatrixXd(W.llt().matrixL())
                                : Eigen::MatrixXd()),
        Vsqrt_(use_sqrt_method_ ? Eigen::MatrixXd(V.llt().matrixL())
                                : Eigen::MatrixXd()) {
    observed_system.ValidateContext(observed_system_context);
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
    internal::CheckObservedSystemInputOutputPorts(
        *observed_system_, options, &observed_system_actuation_input_port_,
        &observed_system_measurement_output_port_,
        &observed_system_process_noise_input_port_,
        &observed_system_measurement_noise_input_port_);
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
    DRAKE_THROW_UNLESS(math::IsPositiveDefinite(
        W, !use_sqrt_method_ ? 0.0 : std::numeric_limits<double>::epsilon(),
        kSymmetryTolerance));
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
          *options.initial_state_covariance,
          !use_sqrt_method_ ? 0.0 : std::numeric_limits<double>::epsilon(),
          kSymmetryTolerance));
    } else if (options.use_square_root_method) {
      throw std::logic_error(
          "options.initial_state_covariance is required when "
          "options.use_square_root_method is set to 'true'.");
    }
  }

  ~ExtendedKalmanFilterBase() override = default;

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

  // Calculates the dynamics of the observed system:
  // ẋ = f(x,u,w=0) or x[n+1] = f(x[n],u,w=0).
  // Also calculates the Jacobians: ∂f/∂x and ∂f/∂w.
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd>
  CalcDynamicsAndLinearize(Context<AutoDiffXd>* observed_system_context,
                           const Eigen::Ref<const Eigen::VectorXd>& x,
                           const Eigen::Ref<const Eigen::VectorXd>& u) const {
    const bool is_continuous =
        observed_system_context->has_only_continuous_state();

    const bool use_additive_w =
        observed_system_process_noise_input_port_ == nullptr;
    const int w_size = W_.rows();

    // dyn = f(x,u,w=0)
    Eigen::VectorX<AutoDiffXd> x_ad, u_ad;
    if (use_additive_w) {
      x_ad = math::InitializeAutoDiff(x);
      // We don't need gradient with respect to u.
      const int num_derivs = x_ad.size();
      u_ad = math::InitializeAutoDiff(u, num_derivs, num_derivs);
    } else {
      Eigen::VectorX<AutoDiffXd> w_ad;
      std::tie(x_ad, w_ad) =
          math::InitializeAutoDiffTuple(x, Eigen::VectorXd::Zero(w_size));
      observed_system_process_noise_input_port_->FixValue(
          observed_system_context, w_ad);
      // We don't need gradient with respect to u.
      const int num_derivs = x_ad.size() + w_ad.size();
      u_ad = math::InitializeAutoDiff(u, num_derivs, num_derivs);
    }

    is_continuous ? observed_system_context->SetContinuousState(x_ad)
                  : observed_system_context->SetDiscreteState(x_ad);

    observed_system_actuation_input_port_->FixValue(observed_system_context,
                                                    u_ad);

    Eigen::VectorX<AutoDiffXd> dyn_ad =
        is_continuous
            ? observed_system_->EvalTimeDerivatives(*observed_system_context)
                  .CopyToVector()
            : observed_system_
                  ->EvalUniquePeriodicDiscreteUpdate(*observed_system_context)
                  .value();
    Eigen::VectorXd dyn = math::ExtractValue(dyn_ad);

    // A = ∂f/∂x(x,u,w)
    Eigen::MatrixXd jacobian = math::ExtractGradient(dyn_ad);
    Eigen::MatrixXd A = jacobian.leftCols(num_states_);

    // G = ∂f/∂w(x,u,w)
    Eigen::MatrixXd G =
        use_additive_w
            ? Eigen::MatrixXd(Eigen::MatrixXd::Identity(w_size, w_size))
            : jacobian.rightCols(w_size);

    return {dyn, A, G};
  }

  // Calculates the measurement output of the observed system: y = g(x,u,v=0).
  // Also calculates the Jacobians: ∂g/∂x and ∂g/∂v.
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd>
  CalcMeasurementAndLinearize(
      Context<AutoDiffXd>* observed_system_context,
      const Eigen::Ref<const Eigen::VectorXd>& x,
      const Eigen::Ref<const Eigen::VectorXd>& u) const {
    const bool is_continuous =
        observed_system_context->has_only_continuous_state();

    const bool use_additive_v =
        observed_system_measurement_noise_input_port_ == nullptr;
    const int v_size = V_.rows();

    // y = g(x,u,v)
    Eigen::VectorX<AutoDiffXd> x_ad, u_ad;
    if (use_additive_v) {
      x_ad = math::InitializeAutoDiff(x);
      // We don't need gradient with respect to u.
      const int num_derivs = x_ad.size();
      u_ad = math::InitializeAutoDiff(u, num_derivs, num_derivs);
    } else {
      Eigen::VectorX<AutoDiffXd> v_ad;
      std::tie(x_ad, v_ad) =
          math::InitializeAutoDiffTuple(x, Eigen::VectorXd::Zero(v_size));
      observed_system_measurement_noise_input_port_->FixValue(
          observed_system_context, v_ad);
      // We don't need gradient with respect to u.
      const int num_derivs = x_ad.size() + v_ad.size();
      u_ad = math::InitializeAutoDiff(u, num_derivs, num_derivs);
    }

    is_continuous ? observed_system_context->SetContinuousState(x_ad)
                  : observed_system_context->SetDiscreteState(x_ad);

    observed_system_actuation_input_port_->FixValue(observed_system_context,
                                                    u_ad);

    const Eigen::VectorX<AutoDiffXd>& y_ad =
        observed_system_measurement_output_port_->Eval(
            *observed_system_context);
    const Eigen::VectorXd y = math::ExtractValue(y_ad);

    // C = ∂g/∂x(x,u,v)
    Eigen::MatrixXd jacobian = math::ExtractGradient(y_ad);
    Eigen::MatrixXd C = jacobian.leftCols(num_states_);

    // H = ∂g/∂v(x,u,v)
    Eigen::MatrixXd H =
        use_additive_v
            ? Eigen::MatrixXd(Eigen::MatrixXd::Identity(v_size, v_size))
            : jacobian.rightCols(v_size);

    return {y, C, H};
  }

  // The Kalman discrete-time measurement update.
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> MeasurementUpdate(
      const Context<double>& context,
      Context<AutoDiffXd>* observed_system_context,
      const Eigen::Ref<const Eigen::VectorXd>& xhat,
      const Eigen::Ref<const Eigen::MatrixXd>& Phat) const {
    DRAKE_ASSERT(!use_sqrt_method_);
    const Eigen::VectorXd& u =
        this->get_observed_system_input_input_port().Eval(context);
    const Eigen::VectorXd& y =
        this->get_observed_system_output_input_port().Eval(context);

    // ŷ = g(x̂,u,v=0), C = ∂g/∂x, H = ∂g/∂v
    auto [yhat, C, H] =
        CalcMeasurementAndLinearize(observed_system_context, xhat, u);

    // K = P̂C'(CP̂C' + HVH')⁻¹
    Eigen::MatrixXd K =
        Phat * C.transpose() *
        (C * Phat * C.transpose() + H * V_ * H.transpose()).inverse();

    // x̂ ← x̂ + K (y − ŷ)
    Eigen::VectorXd xhat_new = xhat + K * (y - yhat);

    // P̂ ← (I - KC) P̂
    Eigen::MatrixXd Phat_new =
        (Eigen::MatrixXd::Identity(num_states_, num_states_) - K * C) * Phat;

    return {xhat_new, Phat_new};
  }

  // The Kalman discrete-time measurement update using square root method.
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> MeasurementUpdateSqrt(
      const Context<double>& context,
      Context<AutoDiffXd>* observed_system_context,
      const Eigen::Ref<const Eigen::VectorXd>& xhat,
      const Eigen::Ref<const Eigen::MatrixXd>& Shat) const {
    DRAKE_ASSERT(use_sqrt_method_);
    const Eigen::VectorXd& u =
        this->get_observed_system_input_input_port().Eval(context);
    const Eigen::VectorXd& y =
        this->get_observed_system_output_input_port().Eval(context);

    // ŷ = g(x̂,u,v=0), C = ∂g/∂x, H = ∂g/∂v
    auto [yhat, C, H] =
        CalcMeasurementAndLinearize(observed_system_context, xhat, u);

    const int ns = num_states_;
    const int no = H.rows();
    const int nv = H.cols();
    if (!(nv >= no)) {
      throw std::logic_error(
          "For the measurement update using the square root method, the size "
          "of the measurement noise input port must be equal or greater than "
          "the size of the measurement output port.");
    }

    Eigen::MatrixXd M(nv + ns, no + ns);
    M.topLeftCorner(nv, no) = Vsqrt_.transpose() * H.transpose();
    M.topRightCorner(nv, ns).setZero();
    M.bottomLeftCorner(ns, no) = Shat.transpose() * C.transpose();
    M.bottomRightCorner(ns, ns) = Shat.transpose();

    Eigen::MatrixXd R = Eigen::HouseholderQR<Eigen::MatrixXd>(M)
                            .matrixQR()
                            .triangularView<Eigen::Upper>();
    Eigen::MatrixXd R1 = R.topLeftCorner(no, no);
    Eigen::MatrixXd R2 = R.topRightCorner(no, ns);
    Eigen::MatrixXd R3 = R.block(no, no, ns, ns);

    // K = P̂C'(CP̂C' + HVH')⁻¹
    // x̂ ← x̂ + K (y − ŷ)
    // P̂ ← P̂ - P̂C'(CP̂C' + HVH')⁻¹CP̂
    Eigen::MatrixXd K = R1.triangularView<Eigen::Upper>().solve(R2).transpose();

    Eigen::VectorXd xhat_new = xhat + K * (y - yhat);

    Eigen::MatrixXd Shat_new = R3.transpose();

    return {xhat_new, Shat_new};
  }

  const std::unique_ptr<const System<AutoDiffXd>> observed_system_;
  const int num_states_;
  const bool use_sqrt_method_;
  const Eigen::MatrixXd W_;
  const Eigen::MatrixXd V_;
  const Eigen::MatrixXd Wsqrt_;
  const Eigen::MatrixXd Vsqrt_;
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
                         const ExtendedKalmanFilterOptions& options)
      : ExtendedKalmanFilterBase(observed_system, observed_system_context, W, V,
                                 options) {
    DRAKE_THROW_UNLESS(observed_system_context.has_only_discrete_state() &&
                       observed_system_context.num_discrete_state_groups() ==
                           1);

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
    //  and variance.
    if (!use_sqrt_method_) {
      this->DeclareDiscreteState(internal::ConcatenateVectorAndSquareMatrix(
          initial_state_estimate, initial_state_covariance));
    } else {
      // Check if options.initial_state_covariance is specified and positive
      // definite when use_sqrt_method_ is true is done in the base class
      // constructor.
      this->DeclareDiscreteState(internal::ConcatenateVectorAndLowerTriMatrix(
          initial_state_estimate, initial_state_covariance.llt().matrixL()));
    }

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

  ~ExtendedKalmanFilterDD() override = default;

  // Implements GaussianStateObserver interface.
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
    DRAKE_THROW_UNLESS(math::IsPositiveDefinite(
        state_covariance,
        !use_sqrt_method_ ? 0.0 : std::numeric_limits<double>::epsilon(),
        kSymmetryTolerance));
    if (!use_sqrt_method_) {
      context->SetDiscreteState(internal::ConcatenateVectorAndSquareMatrix(
          state_estimate, state_covariance));
    } else {
      context->SetDiscreteState(internal::ConcatenateVectorAndLowerTriMatrix(
          state_estimate, state_covariance.llt().matrixL()));
    }
  }

  // Implements GaussianStateObserver interface.
  Eigen::VectorXd GetStateEstimate(
      const Context<double>& context) const override {
    this->ValidateContext(context);
    return context.get_discrete_state_vector().value().head(num_states_);
  }

  // Implements GaussianStateObserver interface.
  Eigen::MatrixXd GetStateCovariance(
      const Context<double>& context) const override {
    this->ValidateContext(context);
    if (!use_sqrt_method_) {
      Eigen::MatrixXd state_covariance(num_states_, num_states_);
      internal::ExtractSquareMatrix(context.get_discrete_state_vector().value(),
                                    state_covariance);
      return state_covariance;
    } else {
      Eigen::MatrixXd state_covariance_sqrt(num_states_, num_states_);
      internal::ExtractLowerTriMatrix(
          context.get_discrete_state_vector().value(), state_covariance_sqrt);
      return state_covariance_sqrt * state_covariance_sqrt.transpose();
    }
  }

 private:
  // Callback for computing the output.
  void CalcOutput(const Context<double>& context,
                  BasicVector<double>* out) const {
    out->SetFromVector(this->GetStateEstimate(context));
  }

  // Callback for discrete update of the state estimate and covariance.
  void PeriodicDiscreteUpdate(const Context<double>& context,
                              DiscreteValues<double>* discrete_state) const {
    // Get the mutable observed system context from the cache.
    Context<AutoDiffXd>* observed_system_context =
        &this->get_mutable_observed_system_context(context);

    if (!use_sqrt_method_) {
      // Get the current state estimate and covariance.
      Eigen::VectorXd xhat = GetStateEstimate(context);
      Eigen::MatrixXd Phat = GetStateCovariance(context);

      // Measurement and process update.
      std::tie(xhat, Phat) =
          MeasurementUpdate(context, observed_system_context, xhat, Phat);
      std::tie(xhat, Phat) =
          ProcessUpdate(context, observed_system_context, xhat, Phat);

      discrete_state->set_value(
          internal::ConcatenateVectorAndSquareMatrix(xhat, Phat));
    } else {
      // Get the current state estimate and covariance sqrt.
      Eigen::VectorXd xhat = GetStateEstimate(context);
      Eigen::MatrixXd Shat(num_states_, num_states_);
      internal::ExtractLowerTriMatrix(
          context.get_discrete_state_vector().value(), Shat);

      // Measurement and process update.
      std::tie(xhat, Shat) =
          MeasurementUpdateSqrt(context, observed_system_context, xhat, Shat);
      std::tie(xhat, Shat) =
          ProcessUpdateSqrt(context, observed_system_context, xhat, Shat);

      discrete_state->set_value(
          internal::ConcatenateVectorAndLowerTriMatrix(xhat, Shat));
    }
  }

  // The discrete-time Kalman process update.
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> ProcessUpdate(
      const Context<double>& context,
      Context<AutoDiffXd>* observed_system_context,
      const Eigen::Ref<const Eigen::VectorXd>& xhat,
      const Eigen::Ref<const Eigen::MatrixXd>& Phat) const {
    DRAKE_ASSERT(!use_sqrt_method_);
    const Eigen::VectorXd& u =
        this->get_observed_system_input_input_port().Eval(context);

    // x̂[n+1] = f(x̂[n],u,w=0), A = ∂f/∂x, G = ∂f/∂w
    auto [xhat_next, A, G] =
        CalcDynamicsAndLinearize(observed_system_context, xhat, u);

    // P̂[n+1] = AP̂A' + GWG'
    Eigen::MatrixXd Phat_next =
        A * Phat * A.transpose() + G * W_ * G.transpose();

    return {xhat_next, Phat_next};
  }

  // The discrete-time Kalman process update using square root method.
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> ProcessUpdateSqrt(
      const Context<double>& context,
      Context<AutoDiffXd>* observed_system_context,
      const Eigen::Ref<const Eigen::VectorXd>& xhat,
      const Eigen::Ref<const Eigen::MatrixXd>& Shat) const {
    DRAKE_ASSERT(use_sqrt_method_);
    const Eigen::VectorXd& u =
        this->get_observed_system_input_input_port().Eval(context);

    // x̂[n+1] = f(x̂[n],u,w=0), A = ∂f/∂x, G = ∂f/∂w
    auto [xhat_next, A, G] =
        CalcDynamicsAndLinearize(observed_system_context, xhat, u);

    // P̂[n+1] = AP̂A' + GWG'
    Eigen::MatrixXd Shat_next = internal::MatrixHypot(A * Shat, G * Wsqrt_);

    return {xhat_next, Shat_next};
  }
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
                           const ExtendedKalmanFilterOptions& options)
      : ExtendedKalmanFilterBase(observed_system, observed_system_context, W, V,
                                 options) {
    DRAKE_THROW_UNLESS(observed_system_context.has_only_continuous_state());

    // Initial state estimate and covariance.
    Eigen::VectorXd initial_state_estimate =
        options.initial_state_estimate.has_value()
            ? options.initial_state_estimate.value()
            : Eigen::VectorXd::Zero(num_states_);
    Eigen::MatrixXd initial_state_covariance =
        options.initial_state_covariance.has_value()
            ? options.initial_state_covariance.value()
            : Eigen::MatrixXd::Zero(num_states_, num_states_);

    // Declare estimated state and covariance.
    const auto& xc = observed_system_context.get_continuous_state();
    const int num_q = xc.get_generalized_position().size();
    const int num_v = xc.get_generalized_velocity().size();
    const int num_z = xc.get_misc_continuous_state().size();
    if (!use_sqrt_method_) {
      this->DeclareContinuousState(
          BasicVector<double>(internal::ConcatenateVectorAndSquareMatrix(
              initial_state_estimate, initial_state_covariance)),
          num_q, num_v, num_z + num_states_ * num_states_);
    } else {
      // Check if options.initial_state_covariance is specified and positive
      // definite when use_sqrt_method_ is true is done in the base class
      // constructor.
      // For continuous-time, the covariance needs to be stored in a square
      // matrix because the covariance time derivative is not lower triangle.
      this->DeclareContinuousState(
          BasicVector<double>(internal::ConcatenateVectorAndSquareMatrix(
              initial_state_estimate,
              Eigen::MatrixXd(initial_state_covariance.llt().matrixL()))),
          num_q, num_v, num_z + num_states_ * num_states_);
    }

    // Declare estimated state output.
    this->DeclareVectorOutputPort("estimated_state", num_states_,
                                  &ExtendedKalmanFilterCont::CalcOutput,
                                  {SystemBase::all_state_ticket()});
  }

  // Implements GaussianStateObserver interface.
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
    DRAKE_THROW_UNLESS(math::IsPositiveDefinite(
        state_covariance,
        !use_sqrt_method_ ? 0.0 : std::numeric_limits<double>::epsilon(),
        kSymmetryTolerance));
    if (!use_sqrt_method_) {
      context->SetContinuousState(internal::ConcatenateVectorAndSquareMatrix(
          state_estimate, state_covariance));
    } else {
      context->SetContinuousState(internal::ConcatenateVectorAndSquareMatrix(
          state_estimate, Eigen::MatrixXd(state_covariance.llt().matrixL())));
    }
  }

  // Implements GaussianStateObserver interface.
  Eigen::VectorXd GetStateEstimate(
      const Context<double>& context) const override {
    this->ValidateContext(context);
    return context.get_continuous_state_vector().CopyToVector().head(
        num_states_);
  }

  // Implements GaussianStateObserver interface.
  Eigen::MatrixXd GetStateCovariance(
      const Context<double>& context) const override {
    this->ValidateContext(context);
    if (!use_sqrt_method_) {
      Eigen::MatrixXd state_covariance(num_states_, num_states_);
      internal::ExtractSquareMatrix(
          context.get_continuous_state_vector().CopyToVector(),
          state_covariance);
      return state_covariance;
    } else {
      Eigen::MatrixXd state_covariance_sqrt = GetStateCovarianceSqrt(context);
      return state_covariance_sqrt * state_covariance_sqrt.transpose();
    }
  }

  ~ExtendedKalmanFilterCont() override = default;

 protected:
  Eigen::MatrixXd GetStateCovarianceSqrt(const Context<double>& context) const {
    DRAKE_ASSERT(use_sqrt_method_);
    Eigen::MatrixXd state_covariance_sqrt(num_states_, num_states_);
    internal::ExtractSquareMatrix(
        context.get_continuous_state_vector().CopyToVector(),
        state_covariance_sqrt);
    return state_covariance_sqrt;
  }

 private:
  // Callback for computing the output.
  void CalcOutput(const Context<double>& context,
                  BasicVector<double>* out) const {
    out->SetFromVector(this->GetStateEstimate(context));
  }
};

// Extended Kalman filter with discrete-time measurement and continuous-time
// observed system dynamics.
class ExtendedKalmanFilterCD final : public ExtendedKalmanFilterCont {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExtendedKalmanFilterCD);

  ExtendedKalmanFilterCD(const System<double>& observed_system,
                         const Context<double>& observed_system_context,
                         const Eigen::Ref<const Eigen::MatrixXd>& W,
                         const Eigen::Ref<const Eigen::MatrixXd>& V,
                         const ExtendedKalmanFilterOptions& options)
      : ExtendedKalmanFilterCont(observed_system, observed_system_context, W, V,
                                 options) {
    DRAKE_THROW_UNLESS(options.discrete_measurement_time_period.has_value() &&
                       options.discrete_measurement_time_period.value() > 0);
    this->DeclarePeriodicUnrestrictedUpdateEvent(
        options.discrete_measurement_time_period.value(),
        options.discrete_measurement_time_offset,
        &ExtendedKalmanFilterCD::PeriodicDiscreteUpdate);
  }

  ~ExtendedKalmanFilterCD() override = default;

 private:
  // Callback for computing the continuous-time process update.
  void DoCalcTimeDerivatives(
      const Context<double>& context,
      ContinuousState<double>* derivatives) const override {
    // Get the mutable observed system context from the cache.
    Context<AutoDiffXd>* observed_system_context =
        &this->get_mutable_observed_system_context(context);

    // Get u.
    const Eigen::VectorXd& u =
        this->get_observed_system_input_input_port().Eval(context);

    // Get the current state estimate.
    Eigen::VectorXd xhat = GetStateEstimate(context);

    // dx̂/dt = f(x̂,u,w=0), A = ∂f/∂x, G = ∂f/∂w
    auto [xhat_dot, A, G] =
        CalcDynamicsAndLinearize(observed_system_context, xhat, u);

    if (!use_sqrt_method_) {
      // Get the current state covariance.
      Eigen::MatrixXd Phat = GetStateCovariance(context);

      // dP̂/dt = AP̂ + P̂A' + GWG'
      Eigen::MatrixXd Phat_dot =
          A * Phat + Phat * A.transpose() + G * W_ * G.transpose();

      derivatives->SetFromVector(
          internal::ConcatenateVectorAndSquareMatrix(xhat_dot, Phat_dot));
    } else {
      // Get the current state covariance sqrt.
      Eigen::MatrixXd Shat = GetStateCovarianceSqrt(context);

      // dŜ/dt = AŜ + GWG'Ŝ⁻ᵀ/2
      Eigen::MatrixXd Shat_dot =
          A * Shat + 0.5 * G * W_ * G.transpose() * Shat.transpose().inverse();

      derivatives->SetFromVector(
          internal::ConcatenateVectorAndSquareMatrix(xhat_dot, Shat_dot));
    }
  }

  // Callback for computing the discrete-time measurement update.
  void PeriodicDiscreteUpdate(const Context<double>& context,
                              State<double>* state) const {
    // Get the mutable observed system context from the cache.
    Context<AutoDiffXd>* observed_system_context =
        &this->get_mutable_observed_system_context(context);

    if (!use_sqrt_method_) {
      // Get the current state estimate and covariance.
      Eigen::VectorXd xhat = GetStateEstimate(context);
      Eigen::MatrixXd Phat = GetStateCovariance(context);

      // Measurement update.
      std::tie(xhat, Phat) =
          MeasurementUpdate(context, observed_system_context, xhat, Phat);

      state->get_mutable_continuous_state().SetFromVector(
          internal::ConcatenateVectorAndSquareMatrix(xhat, Phat));
    } else {
      // Get the current state estimate and covariance sqrt.
      Eigen::VectorXd xhat = GetStateEstimate(context);
      Eigen::MatrixXd Shat = GetStateCovarianceSqrt(context);

      // Measurement update.
      std::tie(xhat, Shat) =
          MeasurementUpdateSqrt(context, observed_system_context, xhat, Shat);

      state->get_mutable_continuous_state().SetFromVector(
          internal::ConcatenateVectorAndSquareMatrix(xhat, Shat));
    }
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
                         const ExtendedKalmanFilterOptions& options)
      : ExtendedKalmanFilterCont(observed_system, observed_system_context, W, V,
                                 options) {
    DRAKE_THROW_UNLESS(!options.discrete_measurement_time_period.has_value());
  }

  ~ExtendedKalmanFilterCC() override = default;

 private:
  void DoCalcTimeDerivatives(
      const Context<double>& context,
      ContinuousState<double>* derivatives) const override {
    // Get the mutable observed system context from the cache.
    Context<AutoDiffXd>* observed_system_context =
        &this->get_mutable_observed_system_context(context);

    // Get u and y.
    const Eigen::VectorXd& u =
        this->get_observed_system_input_input_port().Eval(context);
    const Eigen::VectorXd& y =
        this->get_observed_system_output_input_port().Eval(context);

    // Get the current state estimate.
    Eigen::VectorXd xhat = GetStateEstimate(context);

    // x̂dot = f(x̂,u,w=0), A = ∂f/∂x, G = ∂f/∂w
    auto [xhatdot, A, G] =
        CalcDynamicsAndLinearize(observed_system_context, xhat, u);
    // ŷ = g(x̂,u,v=0), C = ∂g/∂x, H = ∂g/∂v
    auto [yhat, C, H] =
        CalcMeasurementAndLinearize(observed_system_context, xhat, u);
    Eigen::MatrixXd HVH_inv = (H * V_ * H.transpose()).inverse();

    if (!use_sqrt_method_) {
      // Get the current state covariance.
      Eigen::MatrixXd Phat = GetStateCovariance(context);

      // dx̂/dt = f(x̂,u) + P̂C'(HVH')⁻¹(y - g(x̂,u))
      Eigen::MatrixXd PhatC = Phat * C.transpose();
      Eigen::VectorXd xhat_deriv = xhatdot + PhatC * HVH_inv * (y - yhat);

      // dP̂/dt = AP̂ + P̂A' + GWG' - P̂C'(HVH')⁻¹CP̂
      Eigen::MatrixXd Phat_deriv = A * Phat + Phat * A.transpose() +
                                   G * W_ * G.transpose() -
                                   PhatC * HVH_inv * PhatC.transpose();

      derivatives->SetFromVector(
          internal::ConcatenateVectorAndSquareMatrix(xhat_deriv, Phat_deriv));
    } else {
      // Get the current state covariance.
      Eigen::MatrixXd Shat = GetStateCovarianceSqrt(context);

      // dx̂/dt = f(x̂,u) + P̂C'(HVH')⁻¹(y - g(x̂,u))
      Eigen::MatrixXd PhatC_HVH_inv =
          Shat * Shat.transpose() * C.transpose() * HVH_inv;
      Eigen::VectorXd xhat_deriv = xhatdot + PhatC_HVH_inv * (y - yhat);

      // dŜ/dt = AŜ + GWG'Ŝ⁻ᵀ/2 - P̂C'(HVH')⁻¹CŜ/2
      Eigen::MatrixXd Shat_deriv =
          A * Shat + 0.5 * G * W_ * G.transpose() * Shat.transpose().inverse() -
          0.5 * PhatC_HVH_inv * C * Shat;

      derivatives->SetFromVector(
          internal::ConcatenateVectorAndSquareMatrix(xhat_deriv, Shat_deriv));
    }
  }
};

}  // namespace

std::unique_ptr<GaussianStateObserver<double>> ExtendedKalmanFilter(
    const System<double>& observed_system,
    const Context<double>& observed_system_context,
    const Eigen::Ref<const Eigen::MatrixXd>& W,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const ExtendedKalmanFilterOptions& options) {
  observed_system.ValidateContext(observed_system_context);
  if (observed_system_context.has_only_discrete_state() &&
      observed_system_context.num_discrete_state_groups() == 1) {
    return std::make_unique<ExtendedKalmanFilterDD>(
        observed_system, observed_system_context, W, V, options);
  } else if (observed_system_context.has_only_continuous_state()) {
    if (options.discrete_measurement_time_period.has_value()) {
      return std::make_unique<ExtendedKalmanFilterCD>(
          observed_system, observed_system_context, W, V, options);
    } else {
      return std::make_unique<ExtendedKalmanFilterCC>(
          observed_system, observed_system_context, W, V, options);
    }
  } else {
    throw std::logic_error(
        "ExtendedKalmanFilter only supports systems with either only "
        "continuous states or only discrete states");
  }
}

}  // namespace estimators
}  // namespace systems
}  // namespace drake
