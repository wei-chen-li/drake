#pragma once

#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/linear_system.h"

namespace drake {
namespace systems {
namespace estimators_test {

/// If time_period == 0
///   ẋ = Ax + Bu + Gw
///   y = Cx + Du + Hv
/// If time_period > 0
///   x[n+1] = Ax[n] + Bu[n] + Gw[n]
///   y = Cx + Du + Hv
template <typename T>
class StochasticLinearSystem : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(StochasticLinearSystem);

  StochasticLinearSystem(const Eigen::Ref<const Eigen::MatrixXd>& A,
                         const Eigen::Ref<const Eigen::MatrixXd>& B,
                         const Eigen::Ref<const Eigen::MatrixXd>& G,
                         const Eigen::Ref<const Eigen::MatrixXd>& C,
                         const Eigen::Ref<const Eigen::MatrixXd>& D,
                         const Eigen::Ref<const Eigen::MatrixXd>& H,
                         double time_period = 0.0)
      : LeafSystem<T>(SystemTypeTag<StochasticLinearSystem>{}),
        A_(A),
        B_(B),
        G_(G),
        C_(C),
        D_(D),
        H_(H),
        time_period_(time_period) {
    const int num_states = A.rows();
    const int num_inputs = B.cols();
    const int num_outputs = C.rows();
    DRAKE_THROW_UNLESS(A.rows() == num_states && A.cols() == num_states);
    DRAKE_THROW_UNLESS(B.rows() == num_states && B.cols() == num_inputs);
    DRAKE_THROW_UNLESS(G.rows() == num_states && G.cols() > 0);
    DRAKE_THROW_UNLESS(C.rows() == num_outputs && C.cols() == num_states);
    DRAKE_THROW_UNLESS(D.rows() == num_outputs && D.cols() == num_inputs);
    DRAKE_THROW_UNLESS(H.rows() == num_outputs && H.cols() > 0);
    DRAKE_THROW_UNLESS(time_period >= 0.0);

    u_input_port_ = &this->DeclareVectorInputPort("u", num_inputs);
    w_input_port_ = &this->DeclareVectorInputPort(
        "w", G.cols(), RandomDistribution::kGaussian);
    v_input_port_ = &this->DeclareVectorInputPort(
        "v", H.cols(), RandomDistribution::kGaussian);
    this->DeclareVectorOutputPort(
        "y", num_outputs, &StochasticLinearSystem::CalculateOutput,
        {u_input_port_->ticket(), v_input_port_->ticket(),
         this->all_state_ticket()});

    if (time_period == 0) {
      this->DeclareContinuousState(num_states);
    } else {
      this->DeclareDiscreteState(num_states);
      this->DeclarePeriodicDiscreteUpdateEvent(
          time_period, 0.0, &StochasticLinearSystem::DiscreteUpdate);
    }
  }

  StochasticLinearSystem(const LinearSystem<T>& sys,
                         const Eigen::Ref<const Eigen::MatrixXd>& G,
                         const Eigen::Ref<const Eigen::MatrixXd>& H)
      : StochasticLinearSystem(
            sys.A(), sys.B(), G, sys.C(), sys.D(), H,
            sys.GetUniquePeriodicDiscreteUpdateAttribute().has_value()
                ? sys.GetUniquePeriodicDiscreteUpdateAttribute()
                      .value()
                      .period_sec()
                : 0.0) {}

  // Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  StochasticLinearSystem(const StochasticLinearSystem<U>& other)
      : StochasticLinearSystem(other.A_, other.B_, other.G_, other.C_, other.D_,
                               other.H_, other.time_period_) {}

  const InputPort<T>& get_u_input_port() const { return *u_input_port_; }

  const InputPort<T>& get_w_input_port() const { return *w_input_port_; }

  const InputPort<T>& get_v_input_port() const { return *v_input_port_; }

  const Eigen::MatrixXd& A() const { return A_; }
  const Eigen::MatrixXd& B() const { return B_; }
  const Eigen::MatrixXd& G() const { return G_; }
  const Eigen::MatrixXd& C() const { return C_; }
  const Eigen::MatrixXd& D() const { return D_; }
  const Eigen::MatrixXd& H() const { return H_; }

 private:
  template <typename U>
  friend class StochasticLinearSystem;

  void DoCalcTimeDerivatives(const Context<T>& context,
                             ContinuousState<T>* derivatives) const override {
    Eigen::VectorX<T> x =
        context.has_only_continuous_state()
            ? context.get_continuous_state_vector().CopyToVector()
            : context.get_discrete_state_vector().CopyToVector();
    Eigen::VectorX<T> u = get_u_input_port().Eval(context);
    Eigen::VectorX<T> w = get_w_input_port().Eval(context);
    Eigen::VectorX<T> xdot = A_ * x + B_ * u + G_ * w;
    derivatives->SetFromVector(xdot);
  }

  void DiscreteUpdate(const Context<T>& context,
                      DiscreteValues<T>* update) const {
    Eigen::VectorX<T> x =
        context.has_only_continuous_state()
            ? context.get_continuous_state_vector().CopyToVector()
            : context.get_discrete_state_vector().CopyToVector();
    Eigen::VectorX<T> u = get_u_input_port().Eval(context);
    Eigen::VectorX<T> w = get_w_input_port().Eval(context);
    Eigen::VectorX<T> x_next = A_ * x + B_ * u + G_ * w;
    update->set_value(x_next);
  }

  void CalculateOutput(const Context<T>& context, BasicVector<T>* out) const {
    Eigen::VectorX<T> x =
        context.has_only_continuous_state()
            ? context.get_continuous_state_vector().CopyToVector()
            : context.get_discrete_state_vector().CopyToVector();
    Eigen::VectorX<T> u = get_u_input_port().Eval(context);
    Eigen::VectorX<T> v = get_v_input_port().Eval(context);
    Eigen::VectorX<T> y = C_ * x + D_ * u + H_ * v;
    out->SetFromVector(y);
  }

  const Eigen::MatrixXd A_;
  const Eigen::MatrixXd B_;
  const Eigen::MatrixXd G_;
  const Eigen::MatrixXd C_;
  const Eigen::MatrixXd D_;
  const Eigen::MatrixXd H_;
  const double time_period_;

  const InputPort<T>* u_input_port_;
  const InputPort<T>* w_input_port_;
  const InputPort<T>* v_input_port_;
};

}  // namespace estimators_test
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::estimators_test::StochasticLinearSystem);
