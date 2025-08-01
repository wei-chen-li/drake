#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/eigen_pybind.h"
#include "drake/bindings/pydrake/common/serialize_pybind.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/affine_system.h"
#include "drake/systems/primitives/barycentric_system.h"
#include "drake/systems/primitives/bus_creator.h"
#include "drake/systems/primitives/bus_selector.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/discrete_derivative.h"
#include "drake/systems/primitives/discrete_time_delay.h"
#include "drake/systems/primitives/discrete_time_integrator.h"
#include "drake/systems/primitives/first_order_low_pass_filter.h"
#include "drake/systems/primitives/gain.h"
#include "drake/systems/primitives/integrator.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/systems/primitives/linear_transform_density.h"
#include "drake/systems/primitives/matrix_gain.h"
#include "drake/systems/primitives/multilayer_perceptron.h"
#include "drake/systems/primitives/multiplexer.h"
#include "drake/systems/primitives/pass_through.h"
#include "drake/systems/primitives/port_switch.h"
#include "drake/systems/primitives/random_source.h"
#include "drake/systems/primitives/saturation.h"
#include "drake/systems/primitives/selector.h"
#include "drake/systems/primitives/shared_pointer_system.h"
#include "drake/systems/primitives/sine.h"
#include "drake/systems/primitives/sparse_matrix_gain.h"
#include "drake/systems/primitives/symbolic_vector_system.h"
#include "drake/systems/primitives/trajectory_affine_system.h"
#include "drake/systems/primitives/trajectory_linear_system.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/systems/primitives/vector_log_sink.h"
#include "drake/systems/primitives/wrap_to_system.h"
#include "drake/systems/primitives/zero_order_hold.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace drake {

using symbolic::Expression;
using symbolic::Variable;

namespace pydrake {

PYBIND11_MODULE(primitives, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::systems;

  m.doc() = "Bindings for the primitives portion of the Systems framework.";
  constexpr auto& doc = pydrake_doc.drake.systems;

  py::module::import("pydrake.systems.framework");
  py::module::import("pydrake.trajectories");

  py::enum_<PerceptronActivationType>(
      m, "PerceptronActivationType", doc.PerceptronActivationType.doc)
      .value("kIdentity", PerceptronActivationType::kIdentity,
          doc.PerceptronActivationType.kIdentity.doc)
      .value("kReLU", PerceptronActivationType::kReLU,
          doc.PerceptronActivationType.kReLU.doc)
      .value("kTanh", PerceptronActivationType::kTanh,
          doc.PerceptronActivationType.kTanh.doc);

  {
    using Class = SelectorParams;
    py::class_<Class> cls(m, "SelectorParams", doc.SelectorParams.doc);
    {
      using Nested = Class::InputPortParams;
      py::class_<Nested> nested(
          cls, "InputPortParams", doc.SelectorParams.InputPortParams.doc);
      nested.def(ParamInit<Nested>());
      DefAttributesUsingSerialize(&nested, doc.SelectorParams.InputPortParams);
      DefReprUsingSerialize(&nested);
      DefCopyAndDeepCopy(&nested);
    }
    {
      using Nested = Class::OutputSelection;
      py::class_<Nested> nested(
          cls, "OutputSelection", doc.SelectorParams.OutputSelection.doc);
      nested.def(ParamInit<Nested>());
      DefAttributesUsingSerialize(&nested, doc.SelectorParams.OutputSelection);
      DefReprUsingSerialize(&nested);
      DefCopyAndDeepCopy(&nested);
    }
    {
      using Nested = Class::OutputPortParams;
      py::class_<Nested> nested(
          cls, "OutputPortParams", doc.SelectorParams.OutputPortParams.doc);
      nested.def(ParamInit<Nested>());
      DefAttributesUsingSerialize(&nested, doc.SelectorParams.OutputPortParams);
      DefReprUsingSerialize(&nested);
      DefCopyAndDeepCopy(&nested);
    }
    cls.def(ParamInit<Class>());
    DefAttributesUsingSerialize(&cls, doc.SelectorParams);
    DefReprUsingSerialize(&cls);
    DefCopyAndDeepCopy(&cls);
  }

  // N.B. Capturing `&doc` should not be required; workaround per #9600.
  auto bind_common_scalar_types = [&m, &doc](auto dummy) {
    using T = decltype(dummy);

    DefineTemplateClassWithDefault<Adder<T>, LeafSystem<T>>(
        m, "Adder", GetPyParam<T>(), doc.Adder.doc)
        .def(py::init<int, int>(), py::arg("num_inputs"), py::arg("size"),
            doc.Adder.ctor.doc);

    DefineTemplateClassWithDefault<AffineSystem<T>, LeafSystem<T>>(
        m, "AffineSystem", GetPyParam<T>(), doc.AffineSystem.doc)
        .def(py::init<const Eigen::Ref<const MatrixXd>&,
                 const Eigen::Ref<const MatrixXd>&,
                 const Eigen::Ref<const VectorXd>&,
                 const Eigen::Ref<const MatrixXd>&,
                 const Eigen::Ref<const MatrixXd>&,
                 const Eigen::Ref<const VectorXd>&, double>(),
            py::arg("A") = Eigen::MatrixXd(), py::arg("B") = Eigen::MatrixXd(),
            py::arg("f0") = Eigen::VectorXd(), py::arg("C") = Eigen::MatrixXd(),
            py::arg("D") = Eigen::MatrixXd(), py::arg("y0") = Eigen::VectorXd(),
            py::arg("time_period") = 0.0, doc.AffineSystem.ctor.doc_7args)
        // TODO(eric.cousineau): Fix these to return references instead of
        // copies.
        .def("A", overload_cast_explicit<const MatrixXd&>(&AffineSystem<T>::A),
            doc.AffineSystem.A.doc_0args)
        .def("B", overload_cast_explicit<const MatrixXd&>(&AffineSystem<T>::B),
            doc.AffineSystem.B.doc)
        .def("f0",
            overload_cast_explicit<const VectorXd&>(&AffineSystem<T>::f0),
            doc.AffineSystem.f0.doc)
        .def("C", overload_cast_explicit<const MatrixXd&>(&AffineSystem<T>::C),
            doc.AffineSystem.C.doc)
        .def("D", overload_cast_explicit<const MatrixXd&>(&AffineSystem<T>::D),
            doc.AffineSystem.D.doc)
        .def("y0",
            overload_cast_explicit<const VectorXd&>(&AffineSystem<T>::y0),
            doc.AffineSystem.y0.doc)
        .def("UpdateCoefficients", &AffineSystem<T>::UpdateCoefficients,
            py::arg("A") = Eigen::MatrixXd(), py::arg("B") = Eigen::MatrixXd(),
            py::arg("f0") = Eigen::VectorXd(), py::arg("C") = Eigen::MatrixXd(),
            py::arg("D") = Eigen::MatrixXd(), py::arg("y0") = Eigen::VectorXd(),
            doc.AffineSystem.UpdateCoefficients.doc)
        // Wrap a few methods from the TimeVaryingAffineSystem parent class.
        // TODO(russt): Move to TimeVaryingAffineSystem if/when that class is
        // wrapped.
        .def("time_period", &AffineSystem<T>::time_period,
            doc.TimeVaryingAffineSystem.time_period.doc)
        .def("num_states", &TrajectoryAffineSystem<T>::num_states,
            doc.TimeVaryingAffineSystem.num_states.doc)
        .def("num_inputs", &TrajectoryAffineSystem<T>::num_inputs,
            doc.TimeVaryingAffineSystem.num_inputs.doc)
        .def("num_outputs", &TrajectoryAffineSystem<T>::num_outputs,
            doc.TimeVaryingAffineSystem.num_outputs.doc)
        .def("configure_default_state",
            &TimeVaryingAffineSystem<T>::configure_default_state, py::arg("x0"),
            doc.TimeVaryingAffineSystem.configure_default_state.doc)
        .def("configure_random_state",
            &TimeVaryingAffineSystem<T>::configure_random_state,
            py::arg("covariance"),
            doc.TimeVaryingAffineSystem.configure_random_state.doc);

    DefineTemplateClassWithDefault<BusCreator<T>, LeafSystem<T>>(
        m, "BusCreator", GetPyParam<T>(), doc.BusCreator.doc)
        .def(py::init<std::variant<std::string, UseDefaultName>>(),
            py::arg("output_port_name") = kUseDefaultName,
            doc.BusCreator.ctor.doc)
        .def("DeclareVectorInputPort", &BusCreator<T>::DeclareVectorInputPort,
            py::arg("name"), py::arg("size"), py_rvp::reference_internal,
            doc.BusCreator.DeclareVectorInputPort.doc)
        .def("DeclareAbstractInputPort",
            &BusCreator<T>::DeclareAbstractInputPort, py::arg("name"),
            py::arg("model_value"), py_rvp::reference_internal,
            doc.BusCreator.DeclareAbstractInputPort.doc);

    DefineTemplateClassWithDefault<BusSelector<T>, LeafSystem<T>>(
        m, "BusSelector", GetPyParam<T>(), doc.BusSelector.doc)
        .def(py::init<std::variant<std::string, UseDefaultName>>(),
            py::arg("input_port_name") = kUseDefaultName,
            doc.BusSelector.ctor.doc)
        .def("DeclareVectorOutputPort",
            &BusSelector<T>::DeclareVectorOutputPort, py::arg("name"),
            py::arg("size"), py_rvp::reference_internal,
            doc.BusSelector.DeclareVectorOutputPort.doc)
        .def("DeclareAbstractOutputPort",
            &BusSelector<T>::DeclareAbstractOutputPort, py::arg("name"),
            py::arg("model_value"), py_rvp::reference_internal,
            doc.BusSelector.DeclareAbstractOutputPort.doc);

    DefineTemplateClassWithDefault<ConstantValueSource<T>, LeafSystem<T>>(
        m, "ConstantValueSource", GetPyParam<T>(), doc.ConstantValueSource.doc)
        .def(py::init<const AbstractValue&>(), py::arg("value"),
            doc.ConstantValueSource.ctor.doc);

    DefineTemplateClassWithDefault<ConstantVectorSource<T>, LeafSystem<T>>(m,
        "ConstantVectorSource", GetPyParam<T>(), doc.ConstantVectorSource.doc)
        .def(py::init<VectorX<T>>(), py::arg("source_value"),
            doc.ConstantVectorSource.ctor.doc)
        .def("get_source_value", &ConstantVectorSource<T>::get_source_value,
            py::arg("context"), py_rvp::reference_internal,
            doc.ConstantVectorSource.get_source_value.doc)
        .def("get_mutable_source_value",
            &ConstantVectorSource<T>::get_mutable_source_value,
            py::arg("context"), py_rvp::reference_internal,
            doc.ConstantVectorSource.get_mutable_source_value.doc);

    DefineTemplateClassWithDefault<Demultiplexer<T>, LeafSystem<T>>(
        m, "Demultiplexer", GetPyParam<T>(), doc.Demultiplexer.doc)
        .def(py::init<int, int>(), py::arg("size"),
            py::arg("output_ports_size") = 1, doc.Demultiplexer.ctor.doc_2args)
        .def(py::init<const std::vector<int>&>(), py::arg("output_ports_sizes"),
            doc.Demultiplexer.ctor.doc_1args)
        .def("get_output_ports_sizes",
            &Demultiplexer<T>::get_output_ports_sizes,
            doc.Demultiplexer.get_output_ports_sizes.doc);

    DefineTemplateClassWithDefault<DiscreteTimeDelay<T>, LeafSystem<T>>(
        m, "DiscreteTimeDelay", GetPyParam<T>(), doc.DiscreteTimeDelay.doc)
        .def(py::init<double, int, int>(), py::arg("update_sec"),
            py::arg("delay_time_steps"), py::arg("vector_size"),
            doc.DiscreteTimeDelay.ctor
                .doc_3args_update_sec_delay_time_steps_vector_size)
        .def(py::init<double, int, const AbstractValue&>(),
            py::arg("update_sec"), py::arg("delay_time_steps"),
            py::arg("abstract_model_value"),
            doc.DiscreteTimeDelay.ctor
                .doc_3args_update_sec_delay_time_steps_abstract_model_value);

    DefineTemplateClassWithDefault<DiscreteTimeIntegrator<T>, LeafSystem<T>>(m,
        "DiscreteTimeIntegrator", GetPyParam<T>(),
        doc.DiscreteTimeIntegrator.doc)
        .def(py::init<int, double>(), py::arg("size"), py::arg("time_step"),
            doc.DiscreteTimeIntegrator.ctor.doc)
        .def("set_integral_value",
            &DiscreteTimeIntegrator<T>::set_integral_value, py::arg("context"),
            py::arg("value"), doc.DiscreteTimeIntegrator.set_integral_value.doc)
        .def("time_step", &DiscreteTimeIntegrator<T>::time_step,
            doc.DiscreteTimeIntegrator.time_step.doc);

    DefineTemplateClassWithDefault<DiscreteDerivative<T>, LeafSystem<T>>(
        m, "DiscreteDerivative", GetPyParam<T>(), doc.DiscreteDerivative.doc)
        .def(py::init<int, double, bool>(), py::arg("num_inputs"),
            py::arg("time_step"), py::arg("suppress_initial_transient") = true,
            doc.DiscreteDerivative.ctor.doc)
        .def("time_step", &DiscreteDerivative<T>::time_step,
            doc.DiscreteDerivative.time_step.doc)
        .def("suppress_initial_transient",
            &DiscreteDerivative<T>::suppress_initial_transient,
            doc.DiscreteDerivative.suppress_initial_transient.doc);

    DefineTemplateClassWithDefault<                  // BR
        FirstOrderLowPassFilter<T>, LeafSystem<T>>(  //
        m, "FirstOrderLowPassFilter", GetPyParam<T>(),
        doc.FirstOrderLowPassFilter.doc)
        .def(py::init<double, int>(), py::arg("time_constant"),
            py::arg("size") = 1, doc.FirstOrderLowPassFilter.ctor.doc_2args)
        .def(py::init<const VectorX<double>&>(), py::arg("time_constants"),
            doc.FirstOrderLowPassFilter.ctor.doc_1args)
        .def("get_time_constant",
            &FirstOrderLowPassFilter<T>::get_time_constant,
            doc.FirstOrderLowPassFilter.get_time_constant.doc)
        .def("get_time_constants_vector",
            &FirstOrderLowPassFilter<T>::get_time_constants_vector,
            doc.FirstOrderLowPassFilter.get_time_constants_vector.doc)
        .def("set_initial_output_value",
            &FirstOrderLowPassFilter<T>::set_initial_output_value,
            doc.FirstOrderLowPassFilter.set_initial_output_value.doc);

    DefineTemplateClassWithDefault<Gain<T>, LeafSystem<T>>(
        m, "Gain", GetPyParam<T>(), doc.Gain.doc)
        .def(py::init<double, int>(), py::arg("k"), py::arg("size"),
            doc.Gain.ctor.doc_2args)
        .def(py::init<const Eigen::Ref<const VectorXd>&>(), py::arg("k"),
            doc.Gain.ctor.doc_1args);

    DefineTemplateClassWithDefault<Selector<T>, LeafSystem<T>>(
        m, "Selector", GetPyParam<T>(), doc.Selector.doc)
        .def(py::init<SelectorParams>(), py::arg("params"),
            doc.Selector.ctor.doc);

    DefineTemplateClassWithDefault<Sine<T>, LeafSystem<T>>(
        m, "Sine", GetPyParam<T>(), doc.Sine.doc)
        .def(py::init<double, double, double, int, bool>(),
            py::arg("amplitude"), py::arg("frequency"), py::arg("phase"),
            py::arg("size"), py::arg("is_time_based") = true,
            doc.Sine.ctor.doc_5args)
        .def(py::init<const Eigen::Ref<const VectorXd>&,
                 const Eigen::Ref<const VectorXd>&,
                 const Eigen::Ref<const VectorXd>&, bool>(),
            py::arg("amplitudes"), py::arg("frequencies"), py::arg("phases"),
            py::arg("is_time_based") = true, doc.Sine.ctor.doc_4args);

    DefineTemplateClassWithDefault<Integrator<T>, LeafSystem<T>>(
        m, "Integrator", GetPyParam<T>(), doc.Integrator.doc)
        .def(py::init<int>(), py::arg("size"),
            doc.Integrator.ctor.doc_1args_size)
        .def(py::init<const VectorXd&>(), py::arg("initial_value"),
            doc.Integrator.ctor.doc_1args_initial_value)
        .def("set_default_integral_value",
            &Integrator<T>::set_default_integral_value,
            py::arg("initial_value"),
            doc.Integrator.set_default_integral_value.doc)
        .def("set_integral_value", &Integrator<T>::set_integral_value,
            py::arg("context"), py::arg("value"),
            doc.Integrator.set_integral_value.doc);

    DefineTemplateClassWithDefault<LinearSystem<T>, AffineSystem<T>>(
        m, "LinearSystem", GetPyParam<T>(), doc.LinearSystem.doc)
        .def(py::init<const Eigen::Ref<const MatrixXd>&,
                 const Eigen::Ref<const MatrixXd>&,
                 const Eigen::Ref<const MatrixXd>&,
                 const Eigen::Ref<const MatrixXd>&, double>(),
            py::arg("A") = Eigen::MatrixXd(), py::arg("B") = Eigen::MatrixXd(),
            py::arg("C") = Eigen::MatrixXd(), py::arg("D") = Eigen::MatrixXd(),
            py::arg("time_period") = 0.0, doc.LinearSystem.ctor.doc_5args);

    DefineTemplateClassWithDefault<MatrixGain<T>, LinearSystem<T>>(
        m, "MatrixGain", GetPyParam<T>(), doc.MatrixGain.doc)
        .def(py::init<const Eigen::Ref<const MatrixXd>&>(), py::arg("D"),
            doc.MatrixGain.ctor.doc_1args_D);

    DefineTemplateClassWithDefault<Multiplexer<T>, LeafSystem<T>>(
        m, "Multiplexer", GetPyParam<T>(), doc.Multiplexer.doc)
        .def(py::init<int>(), py::arg("num_scalar_inputs"),
            doc.Multiplexer.ctor.doc_1args_num_scalar_inputs)
        .def(py::init<std::vector<int>>(), py::arg("input_sizes"),
            doc.Multiplexer.ctor.doc_1args_input_sizes)
        .def(py::init<const BasicVector<T>&>(), py::arg("model_vector"),
            doc.Multiplexer.ctor.doc_1args_model_vector);

    DefineTemplateClassWithDefault<MultilayerPerceptron<T>, LeafSystem<T>>(m,
        "MultilayerPerceptron", GetPyParam<T>(), doc.MultilayerPerceptron.doc)
        .def(py::init<const std::vector<int>&, PerceptronActivationType>(),
            py::arg("layers"),
            py::arg("activation_type") = PerceptronActivationType::kTanh,
            doc.MultilayerPerceptron.ctor.doc_single_activation)
        .def(py::init<const std::vector<int>&,
                 const std::vector<PerceptronActivationType>&>(),
            py::arg("layers"), py::arg("activation_types"),
            doc.MultilayerPerceptron.ctor.doc_vector_activation)
        .def(py::init<const std::vector<bool>&, const std::vector<int>&,
                 const std::vector<PerceptronActivationType>&>(),
            py::arg("use_sin_cos_for_input"), py::arg("remaining_layers"),
            py::arg("activation_types"),
            doc.MultilayerPerceptron.ctor.doc_sin_cos_features)
        .def("num_parameters", &MultilayerPerceptron<T>::num_parameters,
            doc.MultilayerPerceptron.num_parameters.doc)
        .def("layers", &MultilayerPerceptron<T>::layers,
            doc.MultilayerPerceptron.layers.doc)
        .def("activation_type", &MultilayerPerceptron<T>::activation_type,
            py::arg("layer"), doc.MultilayerPerceptron.activation_type.doc)
        .def("GetParameters", &MultilayerPerceptron<T>::GetParameters,
            py::arg("context"),
            py::keep_alive<0, 2>() /* return keeps context alive */,
            py_rvp::reference, doc.MultilayerPerceptron.GetParameters.doc)
        .def(
            "GetMutableParameters",
            [](const MultilayerPerceptron<T>* self,
                Context<T>* context) -> Eigen::Ref<VectorX<T>> {
              return self->GetMutableParameters(context);
            },
            py_rvp::reference, py::arg("context"),
            // Keep alive, ownership: `return` keeps `context` alive.
            py::keep_alive<0, 2>(),
            doc.MultilayerPerceptron.GetMutableParameters.doc)
        .def("SetParameters", &MultilayerPerceptron<T>::SetParameters,
            py::arg("context"), py::arg("params"),
            doc.MultilayerPerceptron.SetParameters.doc)
        .def("GetWeights",
            overload_cast_explicit<Eigen::Map<const MatrixX<T>>,
                const Context<T>&, int>(&MultilayerPerceptron<T>::GetWeights),
            py::arg("context"), py::arg("layer"),
            py::keep_alive<0, 2>() /* return keeps context alive */,
            py_rvp::reference, doc.MultilayerPerceptron.GetWeights.doc_context)
        .def("GetBiases",
            overload_cast_explicit<Eigen::Map<const VectorX<T>>,
                const Context<T>&, int>(&MultilayerPerceptron<T>::GetBiases),
            py::arg("context"), py::arg("layer"),
            py::keep_alive<0, 2>() /* return keeps context alive */,
            py_rvp::reference, doc.MultilayerPerceptron.GetBiases.doc_context)
        .def("SetWeights",
            overload_cast_explicit<void, Context<T>*, int,
                const Eigen::Ref<const MatrixX<T>>&>(
                &MultilayerPerceptron<T>::SetWeights),
            py::arg("context"), py::arg("layer"), py::arg("W"),
            doc.MultilayerPerceptron.SetWeights.doc_context)
        .def("SetBiases",
            overload_cast_explicit<void, Context<T>*, int,
                const Eigen::Ref<const VectorX<T>>&>(
                &MultilayerPerceptron<T>::SetBiases),
            py::arg("context"), py::arg("layer"), py::arg("b"),
            doc.MultilayerPerceptron.SetBiases.doc_context)
        .def("GetWeights",
            overload_cast_explicit<Eigen::Map<const MatrixX<T>>,
                const Eigen::Ref<const VectorX<T>>&, int>(
                &MultilayerPerceptron<T>::GetWeights),
            py::arg("params"), py::arg("layer"),
            py::keep_alive<0, 2>() /* return keeps params alive */,
            py_rvp::reference, doc.MultilayerPerceptron.GetWeights.doc_vector)
        .def("GetBiases",
            overload_cast_explicit<Eigen::Map<const VectorX<T>>,
                const Eigen::Ref<const VectorX<T>>&, int>(
                &MultilayerPerceptron<T>::GetBiases),
            py::arg("params"), py::arg("layer"),
            py::keep_alive<0, 2>() /* return keeps params alive */,
            py_rvp::reference, doc.MultilayerPerceptron.GetBiases.doc_vector)
        .def("SetWeights",
            py::overload_cast<EigenPtr<VectorX<T>>, int,
                const Eigen::Ref<const MatrixX<T>>&>(
                &MultilayerPerceptron<T>::SetWeights, py::const_),
            py::arg("params"), py::arg("layer"), py::arg("W"),
            doc.MultilayerPerceptron.SetWeights.doc_vector)
        .def("SetBiases",
            py::overload_cast<EigenPtr<VectorX<T>>, int,
                const Eigen::Ref<const VectorX<T>>&>(
                &MultilayerPerceptron<T>::SetBiases, py::const_),
            py::arg("params"), py::arg("layer"), py::arg("b"),
            doc.MultilayerPerceptron.SetBiases.doc_vector)
        .def("Backpropagation",
            WrapCallbacks(&MultilayerPerceptron<T>::Backpropagation),
            py::arg("context"), py::arg("X"), py::arg("loss"),
            py::arg("dloss_dparams"),
            doc.MultilayerPerceptron.Backpropagation.doc)
        .def("BackpropagationMeanSquaredError",
            &MultilayerPerceptron<T>::BackpropagationMeanSquaredError,
            py::arg("context"), py::arg("X"), py::arg("Y_desired"),
            py::arg("dloss_dparams"),
            doc.MultilayerPerceptron.BackpropagationMeanSquaredError.doc)
        .def("BatchOutput", &MultilayerPerceptron<T>::BatchOutput,
            py::arg("context"), py::arg("X"), py::arg("Y"),
            py::arg("dYdX") = nullptr, doc.MultilayerPerceptron.BatchOutput.doc)
        .def(
            "BatchOutput",
            [](const MultilayerPerceptron<T>* self, const Context<T>& context,
                const Eigen::Ref<const MatrixX<T>>& X) {
              MatrixX<T> Y(self->get_output_port().size(), X.cols());
              self->BatchOutput(context, X, &Y);
              return Y;
            },
            py::arg("context"), py::arg("X"),
            "Evaluates the batch output for the MLP with a batch input vector. "
            "See BatchOutput(context, X, Y) for a version that can avoid "
            "dynamic memory allocations of Y (e.g. if this is used inside an "
            "optimization loop).");

    DefineTemplateClassWithDefault<PassThrough<T>, LeafSystem<T>>(
        m, "PassThrough", GetPyParam<T>(), doc.PassThrough.doc)
        .def(py::init<int>(), py::arg("vector_size"),
            doc.PassThrough.ctor.doc_1args_vector_size)
        .def(py::init<const Eigen::Ref<const VectorXd>&>(), py::arg("value"),
            doc.PassThrough.ctor.doc_1args_value)
        .def(py::init<const AbstractValue&>(), py::arg("abstract_model_value"),
            doc.PassThrough.ctor.doc_1args_abstract_model_value);

    DefineTemplateClassWithDefault<PortSwitch<T>, LeafSystem<T>>(
        m, "PortSwitch", GetPyParam<T>(), doc.PortSwitch.doc)
        .def(py::init<int>(), py::arg("vector_size"), doc.PortSwitch.ctor.doc)
        // TODO(russt): implement AbstractValue version of the constructor and
        // bind it here.
        .def("get_port_selector_input_port",
            &PortSwitch<T>::get_port_selector_input_port,
            py_rvp::reference_internal,
            doc.PortSwitch.get_port_selector_input_port.doc)
        .def("DeclareInputPort", &PortSwitch<T>::DeclareInputPort,
            py::arg("name"), py_rvp::reference_internal,
            doc.PortSwitch.DeclareInputPort.doc);

    DefineTemplateClassWithDefault<Saturation<T>, LeafSystem<T>>(
        m, "Saturation", GetPyParam<T>(), doc.Saturation.doc)
        .def(py::init<const VectorX<T>&, const VectorX<T>&>(),
            py::arg("min_value"), py::arg("max_value"),
            doc.Saturation.ctor.doc_2args);

    DefineTemplateClassWithDefault<SparseMatrixGain<T>, LeafSystem<T>>(
        m, "SparseMatrixGain", GetPyParam<T>(), doc.SparseMatrixGain.doc)
        .def(py::init([](const Eigen::SparseMatrix<double>& D) {
          // Our interactions with scipy don't work yet with (0,N) matrices.
          DRAKE_THROW_UNLESS(D.rows() > 0 || D.cols() == 0);
          return std::make_unique<SparseMatrixGain<T>>(D);
        }),
            py::arg("D"), doc.SparseMatrixGain.ctor.doc)
        .def("D", &SparseMatrixGain<T>::D, doc.SparseMatrixGain.D.doc)
        .def("set_D", &SparseMatrixGain<T>::set_D, py::arg("D"),
            doc.SparseMatrixGain.set_D.doc);

    DefineTemplateClassWithDefault<StateInterpolatorWithDiscreteDerivative<T>,
        Diagram<T>>(m, "StateInterpolatorWithDiscreteDerivative",
        GetPyParam<T>(), doc.StateInterpolatorWithDiscreteDerivative.doc)
        .def(py::init<int, double, bool>(), py::arg("num_positions"),
            py::arg("time_step"), py::arg("suppress_initial_transient") = true,
            doc.StateInterpolatorWithDiscreteDerivative.ctor.doc)
        .def("suppress_initial_transient",
            &StateInterpolatorWithDiscreteDerivative<
                T>::suppress_initial_transient,
            doc.StateInterpolatorWithDiscreteDerivative
                .suppress_initial_transient.doc)
        .def(
            "set_initial_position",
            [](const StateInterpolatorWithDiscreteDerivative<T>* self,
                Context<T>* context,
                const Eigen::Ref<const VectorX<T>>& position) {
              self->set_initial_position(context, position);
            },
            py::arg("context"), py::arg("position"),
            doc.StateInterpolatorWithDiscreteDerivative.set_initial_position
                .doc_2args_context_position)
        .def(
            "set_initial_position",
            [](const StateInterpolatorWithDiscreteDerivative<T>* self,
                State<T>* state, const Eigen::Ref<const VectorX<T>>& position) {
              self->set_initial_position(state, position);
            },
            py::arg("state"), py::arg("position"),
            doc.StateInterpolatorWithDiscreteDerivative.set_initial_position
                .doc_2args_state_position);

    DefineTemplateClassWithDefault<SharedPointerSystem<T>, LeafSystem<T>>(
        m, "SharedPointerSystem", GetPyParam<T>(), doc.SharedPointerSystem.doc)
        .def(py::init([](py::object value_to_hold) {
          auto wrapped = std::make_unique<py::object>(std::move(value_to_hold));
          return std::make_unique<SharedPointerSystem<T>>(std::move(wrapped));
        }),
            py::arg("value_to_hold"), doc.SharedPointerSystem.ctor.doc)
        .def_static(
            "AddToBuilder",
            [](DiagramBuilder<T>* builder, py::object value_to_hold) {
              auto wrapped =
                  std::make_unique<py::object>(std::move(value_to_hold));
              return SharedPointerSystem<T>::AddToBuilder(
                  builder, std::move(wrapped));
            },
            py::arg("builder"), py::arg("value_to_hold"),
            doc.SharedPointerSystem.AddToBuilder.doc)
        .def(
            "get",
            [](const SharedPointerSystem<T>& self) {
              py::object result = py::none();
              py::object* held = self.template get<py::object>();
              if (held != nullptr) {
                result = std::move(*held);
              }
              return result;
            },
            doc.SharedPointerSystem.get.doc);

    DefineTemplateClassWithDefault<SymbolicVectorSystem<T>, LeafSystem<T>>(m,
        "SymbolicVectorSystem", GetPyParam<T>(), doc.SymbolicVectorSystem.doc)
        .def(py::init<std::optional<Variable>, VectorX<Variable>,
                 VectorX<Variable>, VectorX<Expression>, VectorX<Expression>,
                 double>(),
            py::arg("time") = std::nullopt,
            py::arg("state") = Vector0<Variable>{},
            py::arg("input") = Vector0<Variable>{},
            py::arg("dynamics") = Vector0<Expression>{},
            py::arg("output") = Vector0<Expression>{},
            py::arg("time_period") = 0.0,
            doc.SymbolicVectorSystem.ctor.doc_6args)
        .def(py::init<std::optional<Variable>, VectorX<Variable>,
                 VectorX<Variable>, VectorX<Variable>, VectorX<Expression>,
                 VectorX<Expression>, double>(),
            py::arg("time") = std::nullopt,
            py::arg("state") = Vector0<Variable>{},
            py::arg("input") = Vector0<Variable>{},
            py::arg("parameter") = Vector0<Variable>{},
            py::arg("dynamics") = Vector0<Expression>{},
            py::arg("output") = Vector0<Expression>{},
            py::arg("time_period") = 0.0,
            doc.SymbolicVectorSystem.ctor.doc_7args)
        .def("dynamics_for_variable",
            &SymbolicVectorSystem<T>::dynamics_for_variable, py::arg("var"),
            doc.SymbolicVectorSystem.dynamics_for_variable.doc);

    DefineTemplateClassWithDefault<VectorLog<T>>(
        m, "VectorLog", GetPyParam<T>(), doc.VectorLog.doc)
        .def_property_readonly_static("kDefaultCapacity",
            [](py::object) { return VectorLog<T>::kDefaultCapacity; })
        .def(py::init<int>(), py::arg("input_size"), doc.VectorLog.ctor.doc)
        .def("num_samples", &VectorLog<T>::num_samples,
            doc.VectorLog.num_samples.doc)
        .def(
            "sample_times",
            [](const VectorLog<T>* self) {
              // Reference
              return CopyIfNotPodType(self->sample_times());
            },
            return_value_policy_for_scalar_type<T>(),
            doc.VectorLog.sample_times.doc)
        .def(
            "data",
            [](const VectorLog<T>* self) {
              // Reference.
              return CopyIfNotPodType(self->data());
            },
            return_value_policy_for_scalar_type<T>(), doc.VectorLog.data.doc)
        .def("Clear", &VectorLog<T>::Clear, doc.VectorLog.Clear.doc)
        .def("Reserve", &VectorLog<T>::Reserve, doc.VectorLog.Reserve.doc)
        .def("AddData", &VectorLog<T>::AddData, py::arg("time"),
            py::arg("sample"), doc.VectorLog.AddData.doc)
        .def("get_input_size", &VectorLog<T>::get_input_size,
            doc.VectorLog.get_input_size.doc);

    DefineTemplateClassWithDefault<VectorLogSink<T>, LeafSystem<T>>(
        m, "VectorLogSink", GetPyParam<T>(), doc.VectorLogSink.doc)
        .def(py::init<int, double>(), py::arg("input_size"),
            py::arg("publish_period") = 0.0, doc.VectorLogSink.ctor.doc_2args)
        .def(py::init<int, const TriggerTypeSet&, double>(),
            py::arg("input_size"), py::arg("publish_triggers"),
            py::arg("publish_period") = 0.0, doc.VectorLogSink.ctor.doc_3args)
        .def(
            "GetLog",
            [](const VectorLogSink<T>* self, const Context<T>& context)
                -> const VectorLog<T>& { return self->GetLog(context); },
            py::arg("context"),
            // Keep alive, ownership: `return` keeps `context` alive.
            py::keep_alive<0, 2>(), py_rvp::reference,
            doc.VectorLogSink.GetLog.doc)
        .def(
            "GetMutableLog",
            [](const VectorLogSink<T>* self, Context<T>* context)
                -> VectorLog<T>& { return self->GetMutableLog(context); },
            py::arg("context"),
            // Keep alive, ownership: `return` keeps `context` alive.
            py::keep_alive<0, 2>(), py_rvp::reference,
            doc.VectorLogSink.GetMutableLog.doc)
        .def(
            "FindLog",
            [](const VectorLogSink<T>* self, const Context<T>& context)
                -> const VectorLog<T>& { return self->FindLog(context); },
            py::arg("context"),
            // Keep alive, ownership: `return` keeps `context` alive.
            py::keep_alive<0, 2>(), py_rvp::reference,
            doc.VectorLogSink.FindLog.doc)
        .def(
            "FindMutableLog",
            [](const VectorLogSink<T>* self, Context<T>* context)
                -> VectorLog<T>& { return self->FindMutableLog(context); },
            py::arg("context"),
            // Keep alive, ownership: `return` keeps `context` alive.
            py::keep_alive<0, 2>(), py_rvp::reference,
            doc.VectorLogSink.FindMutableLog.doc);

    m.def("LogVectorOutput",
        py::overload_cast<const OutputPort<T>&, DiagramBuilder<T>*, double>(
            &LogVectorOutput<T>),
        py::arg("src"), py::arg("builder"), py::arg("publish_period") = 0.0,
        // Keep alive, ownership: `return` keeps `builder` alive.
        py::keep_alive<0, 2>(),
        // See #11531 for why `py_rvp::reference` is needed.
        py_rvp::reference, doc.LogVectorOutput.doc_3args);

    m.def("LogVectorOutput",
        py::overload_cast<const OutputPort<T>&, DiagramBuilder<T>*,
            const TriggerTypeSet&, double>(&LogVectorOutput<T>),
        py::arg("src"), py::arg("builder"), py::arg("publish_triggers"),
        py::arg("publish_period") = 0.0,
        // Keep alive, ownership: `return` keeps `builder` alive.
        py::keep_alive<0, 2>(),
        // See #11531 for why `py_rvp::reference` is needed.
        py_rvp::reference, doc.LogVectorOutput.doc_4args);

    DefineTemplateClassWithDefault<WrapToSystem<T>, LeafSystem<T>>(
        m, "WrapToSystem", GetPyParam<T>(), doc.WrapToSystem.doc)
        .def(py::init<int>(), doc.WrapToSystem.ctor.doc)
        .def("set_interval", &WrapToSystem<T>::set_interval,
            doc.WrapToSystem.set_interval.doc);

    DefineTemplateClassWithDefault<ZeroOrderHold<T>, LeafSystem<T>>(
        m, "ZeroOrderHold", GetPyParam<T>(), doc.ZeroOrderHold.doc)
        .def(py::init<double, int, double>(), py::arg("period_sec"),
            py::arg("vector_size"), py::arg("offset_sec") = 0.0,
            doc.ZeroOrderHold.ctor.doc_3args_period_sec_vector_size_offset_sec)
        .def(py::init<double, const AbstractValue&, double>(),
            py::arg("period_sec"), py::arg("abstract_model_value"),
            py::arg("offset_sec") = 0.0,
            doc.ZeroOrderHold.ctor
                .doc_3args_period_sec_abstract_model_value_offset_sec)
        .def("period", &ZeroOrderHold<T>::period, doc.ZeroOrderHold.period.doc)
        .def("offset", &ZeroOrderHold<T>::offset, doc.ZeroOrderHold.offset.doc)
        .def("SetVectorState", &ZeroOrderHold<T>::SetVectorState,
            doc.ZeroOrderHold.SetVectorState.doc);

    DefineTemplateClassWithDefault<TrajectorySource<T>, LeafSystem<T>>(
        m, "TrajectorySource", GetPyParam<T>(), doc.TrajectorySource.doc)
        .def(py::init<const trajectories::Trajectory<T>&, int, bool>(),
            py::arg("trajectory"), py::arg("output_derivative_order") = 0,
            py::arg("zero_derivatives_beyond_limits") = true,
            doc.TrajectorySource.ctor.doc)
        .def("UpdateTrajectory", &TrajectorySource<T>::UpdateTrajectory,
            py::arg("trajectory"), doc.TrajectorySource.UpdateTrajectory.doc);
  };
  type_visit(bind_common_scalar_types, CommonScalarPack{});

  // N.B. Capturing `&doc` should not be required; workaround per #9600.
  auto bind_non_symbolic_scalar_types = [m, &doc](auto dummy) {
    using T = decltype(dummy);

    DefineTemplateClassWithDefault<LinearTransformDensity<T>, LeafSystem<T>>(m,
        "LinearTransformDensity", GetPyParam<T>(),
        doc.LinearTransformDensity.doc)
        .def(py::init<RandomDistribution, int, int>(), py::arg("distribution"),
            py::arg("input_size"), py::arg("output_size"),
            doc.LinearTransformDensity.ctor.doc)
        .def("get_input_port_w_in",
            &LinearTransformDensity<T>::get_input_port_w_in,
            py_rvp::reference_internal,
            doc.LinearTransformDensity.get_input_port_w_in.doc)
        .def("get_input_port_A", &LinearTransformDensity<T>::get_input_port_A,
            py_rvp::reference_internal,
            doc.LinearTransformDensity.get_input_port_A.doc)
        .def("get_input_port_b", &LinearTransformDensity<T>::get_input_port_b,
            py_rvp::reference_internal,
            doc.LinearTransformDensity.get_input_port_b.doc)
        .def("get_output_port_w_out",
            &LinearTransformDensity<T>::get_output_port_w_out,
            py_rvp::reference_internal,
            doc.LinearTransformDensity.get_output_port_w_out.doc)
        .def("get_output_port_w_out_density",
            &LinearTransformDensity<T>::get_output_port_w_out_density,
            py_rvp::reference_internal,
            doc.LinearTransformDensity.get_output_port_w_out_density.doc)
        .def("get_distribution", &LinearTransformDensity<T>::get_distribution,
            doc.LinearTransformDensity.get_distribution.doc)
        .def("FixConstantA", &LinearTransformDensity<T>::FixConstantA,
            py::arg("context"), py::arg("A"), py_rvp::reference_internal,
            doc.LinearTransformDensity.FixConstantA.doc)
        .def("FixConstantB", &LinearTransformDensity<T>::FixConstantB,
            py::arg("context"), py::arg("b"), py_rvp::reference_internal,
            doc.LinearTransformDensity.FixConstantB.doc)
        .def("CalcDensity", &LinearTransformDensity<T>::CalcDensity,
            py::arg("context"), doc.LinearTransformDensity.CalcDensity.doc);

    DefineTemplateClassWithDefault<TrajectoryAffineSystem<T>, LeafSystem<T>>(m,
        "TrajectoryAffineSystem", GetPyParam<T>(),
        doc.TrajectoryAffineSystem.doc)
        .def(py::init<const trajectories::Trajectory<double>&,
                 const trajectories::Trajectory<double>&,
                 const trajectories::Trajectory<double>&,
                 const trajectories::Trajectory<double>&,
                 const trajectories::Trajectory<double>&,
                 const trajectories::Trajectory<double>&, double>(),
            py::arg("A"), py::arg("B"), py::arg("f0"), py::arg("C"),
            py::arg("D"), py::arg("y0"), py::arg("time_period") = 0.0,
            doc.TrajectoryAffineSystem.ctor.doc)
        .def("A",
            overload_cast_explicit<MatrixX<T>, const T&>(
                &TrajectoryAffineSystem<T>::A),
            doc.TrajectoryAffineSystem.A.doc)
        .def("B",
            overload_cast_explicit<MatrixX<T>, const T&>(
                &TrajectoryAffineSystem<T>::B),
            doc.TrajectoryAffineSystem.B.doc)
        .def("f0",
            overload_cast_explicit<VectorX<T>, const T&>(
                &TrajectoryAffineSystem<T>::f0),
            doc.TrajectoryAffineSystem.f0.doc)
        .def("C",
            overload_cast_explicit<MatrixX<T>, const T&>(
                &TrajectoryAffineSystem<T>::C),
            doc.TrajectoryAffineSystem.C.doc)
        .def("D",
            overload_cast_explicit<MatrixX<T>, const T&>(
                &TrajectoryAffineSystem<T>::D),
            doc.TrajectoryAffineSystem.D.doc)
        .def("y0",
            overload_cast_explicit<VectorX<T>, const T&>(
                &TrajectoryAffineSystem<T>::y0),
            doc.TrajectoryAffineSystem.y0.doc)
        // Wrap a few methods from the TimeVaryingAffineSystem parent class.
        // TODO(russt): Move to TimeVaryingAffineSystem if/when that class is
        // wrapped.
        .def("time_period", &TrajectoryAffineSystem<T>::time_period,
            doc.TimeVaryingAffineSystem.time_period.doc)
        .def("configure_default_state",
            &TimeVaryingAffineSystem<T>::configure_default_state, py::arg("x0"),
            doc.TimeVaryingAffineSystem.configure_default_state.doc)
        .def("configure_random_state",
            &TimeVaryingAffineSystem<T>::configure_random_state,
            py::arg("covariance"),
            doc.TimeVaryingAffineSystem.configure_random_state.doc);

    DefineTemplateClassWithDefault<TrajectoryLinearSystem<T>, LeafSystem<T>>(m,
        "TrajectoryLinearSystem", GetPyParam<T>(),
        doc.TrajectoryLinearSystem.doc)
        .def(py::init<const trajectories::Trajectory<double>&,
                 const trajectories::Trajectory<double>&,
                 const trajectories::Trajectory<double>&,
                 const trajectories::Trajectory<double>&, double>(),
            py::arg("A"), py::arg("B"), py::arg("C"), py::arg("D"),
            py::arg("time_period") = 0.0, doc.TrajectoryLinearSystem.ctor.doc)
        .def("A",
            overload_cast_explicit<MatrixX<T>, const T&>(
                &TrajectoryLinearSystem<T>::A),
            doc.TrajectoryLinearSystem.A.doc)
        .def("B",
            overload_cast_explicit<MatrixX<T>, const T&>(
                &TrajectoryLinearSystem<T>::B),
            doc.TrajectoryLinearSystem.B.doc)
        .def("C",
            overload_cast_explicit<MatrixX<T>, const T&>(
                &TrajectoryLinearSystem<T>::C),
            doc.TrajectoryLinearSystem.C.doc)
        .def("D",
            overload_cast_explicit<MatrixX<T>, const T&>(
                &TrajectoryLinearSystem<T>::D),
            doc.TrajectoryLinearSystem.D.doc)
        // Wrap a few methods from the TimeVaryingAffineSystem parent class.
        // TODO(russt): Move to TimeVaryingAffineSystem if/when that class is
        // wrapped.
        .def("time_period", &TrajectoryAffineSystem<T>::time_period,
            doc.TimeVaryingAffineSystem.time_period.doc)
        .def("configure_default_state",
            &TimeVaryingAffineSystem<T>::configure_default_state, py::arg("x0"),
            doc.TimeVaryingAffineSystem.configure_default_state.doc)
        .def("configure_random_state",
            &TimeVaryingAffineSystem<T>::configure_random_state,
            py::arg("covariance"),
            doc.TimeVaryingAffineSystem.configure_random_state.doc);
  };
  type_visit(bind_non_symbolic_scalar_types, NonSymbolicScalarPack{});

  py::class_<BarycentricMeshSystem<double>, LeafSystem<double>>(
      m, "BarycentricMeshSystem", doc.BarycentricMeshSystem.doc)
      .def(py::init<math::BarycentricMesh<double>,
               const Eigen::Ref<const MatrixX<double>>&>(),
          doc.BarycentricMeshSystem.ctor.doc)
      .def("get_mesh", &BarycentricMeshSystem<double>::get_mesh,
          doc.BarycentricMeshSystem.get_mesh.doc)
      .def("get_output_values",
          &BarycentricMeshSystem<double>::get_output_values,
          doc.BarycentricMeshSystem.get_output_values.doc);

  // TODO(jwnimmer-tri) Add more scalar types bindings for this class.
  py::class_<RandomSource<double>, LeafSystem<double>>(
      m, "RandomSource", doc.RandomSource.doc)
      .def(py::init<RandomDistribution, int, double>(), py::arg("distribution"),
          py::arg("num_outputs"), py::arg("sampling_interval_sec"),
          doc.RandomSource.ctor.doc);

  m.def("AddRandomInputs", &AddRandomInputs<double>,
       py::arg("sampling_interval_sec"), py::arg("builder"),
       doc.AddRandomInputs.doc)
      .def("AddRandomInputs", &AddRandomInputs<AutoDiffXd>,
          py::arg("sampling_interval_sec"), py::arg("builder"),
          doc.AddRandomInputs.doc);

  m.def("Linearize", &Linearize, py::arg("system"), py::arg("context"),
      py::arg("input_port_index") =
          systems::InputPortSelection::kUseFirstInputIfItExists,
      py::arg("output_port_index") =
          systems::OutputPortSelection::kUseFirstOutputIfItExists,
      py::arg("equilibrium_check_tolerance") = 1e-6, doc.Linearize.doc);

  m.def("FirstOrderTaylorApproximation", &FirstOrderTaylorApproximation,
      py::arg("system"), py::arg("context"),
      py::arg("input_port_index") =
          systems::InputPortSelection::kUseFirstInputIfItExists,
      py::arg("output_port_index") =
          systems::OutputPortSelection::kUseFirstOutputIfItExists,
      doc.FirstOrderTaylorApproximation.doc);

  m.def("ControllabilityMatrix", &ControllabilityMatrix,
      doc.ControllabilityMatrix.doc);

  m.def("IsControllable", &IsControllable, py::arg("sys"),
      py::arg("threshold") = std::nullopt, doc.IsControllable.doc);

  m.def(
      "ObservabilityMatrix", &ObservabilityMatrix, doc.ObservabilityMatrix.doc);

  m.def("IsObservable", &IsObservable, py::arg("sys"),
      py::arg("threshold") = std::nullopt, doc.IsObservable.doc);

  m.def("IsStabilizable", &IsStabilizable, py::arg("sys"),
      py::arg("threshold") = std::nullopt, doc.IsStabilizable.doc);

  m.def("IsDetectable", &IsDetectable, py::arg("sys"),
      py::arg("threshold") = std::nullopt, doc.IsDetectable.doc);
}  // NOLINT(readability/fn_size)

}  // namespace pydrake
}  // namespace drake
