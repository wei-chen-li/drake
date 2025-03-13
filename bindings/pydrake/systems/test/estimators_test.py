import unittest
import copy
import numpy as np

from pydrake.examples import PendulumPlant
from pydrake.systems.estimators import (
    DiscreteTimeSteadyStateKalmanFilter,
    ExtendedKalmanFilter,
    ExtendedKalmanFilterOptions,
    LuenbergerObserver,
    SteadyStateKalmanFilter,
    UnscentedKalmanFilter,
    UnscentedKalmanFilterOptions,
)
from pydrake.systems.framework import (
    InputPortSelection, OutputPortSelection,
    DiagramBuilder
)
from pydrake.systems.primitives import LinearSystem


class TestEstimators(unittest.TestCase):

    def test_luenberger_observer(self):
        plant = PendulumPlant()
        context = plant.CreateDefaultContext()
        L = np.eye(2)
        observer = LuenbergerObserver(
            observed_system=plant,
            observed_system_context=context,
            observer_gain=L)
        port = observer.get_observed_system_input_input_port()
        self.assertEqual(port.size(), 1)
        port = observer.get_observed_system_output_input_port()
        self.assertEqual(port.size(), 2)
        port = observer.get_estimated_state_output_port()
        self.assertEqual(port.size(), 2)
        np.testing.assert_array_equal(L, observer.observer_gain())
        np.testing.assert_array_equal(L, observer.L())

        observer.Clone()
        copy.copy(observer)
        copy.deepcopy(observer)

    def test_steady_state_kalman_filter(self):
        A = np.array([[0., 1.], [-10., -0.1]])
        C = np.eye(2)
        W = np.eye(2)
        V = np.eye(2)
        L = SteadyStateKalmanFilter(A=A, C=C, W=W, V=V)
        self.assertEqual(L.shape, (2, 2))

        L = DiscreteTimeSteadyStateKalmanFilter(A=A, C=C, W=W, V=V)
        self.assertEqual(L.shape, (2, 2))

        plant = LinearSystem(A=A, C=C)
        filter = SteadyStateKalmanFilter(system=plant, W=W, V=V)
        self.assertIsInstance(filter, LuenbergerObserver)

        plant = PendulumPlant()
        context = plant.CreateDefaultContext()
        plant.get_input_port().FixValue(context, [0.])
        filter = SteadyStateKalmanFilter(
            system=plant, context=context, W=W, V=V)
        self.assertIsInstance(filter, LuenbergerObserver)

    def test_extended_kalman_filter(self):
        options = ExtendedKalmanFilterOptions()
        self.assertIsNone(options.initial_state_estimate)
        self.assertIsNone(options.initial_state_covariance)
        self.assertEqual(options.actuation_input_port_index,
                         InputPortSelection.kUseFirstInputIfItExists)
        self.assertEqual(options.measurement_output_port_index,
                         OutputPortSelection.kUseFirstOutputIfItExists)
        self.assertIsNone(options.process_noise_input_port_index)
        self.assertIsNone(options.measurement_noise_input_port_index)
        self.assertEqual(options.use_square_root_method, False)
        self.assertIsNone(options.discrete_measurement_time_period)
        self.assertEqual(options.discrete_measurement_time_offset, 0.0)

        sys = LinearSystem(
            np.array([[0, 1], [0, 0]]), np.array([[0], [1]]),
            np.array([[1, 0]]), np.array([[0]]))
        W = np.eye(2)
        V = np.eye(1)
        ekf = ExtendedKalmanFilter(
            observed_system=sys,
            observed_system_context=sys.CreateDefaultContext(),
            W=W, V=V, options=options)

        self.assertTrue(ekf.get_observed_system_input_input_port().size(), 1)
        self.assertTrue(ekf.get_observed_system_output_input_port().size(), 1)
        self.assertTrue(ekf.get_estimated_state_output_port().size(), 2)

        context = ekf.CreateDefaultContext()
        xhat = np.array([1, 2])
        Phat = np.eye(2)
        ekf.SetStateEstimateAndCovariance(context, xhat, Phat)
        np.testing.assert_array_equal(ekf.GetStateEstimate(context), xhat)
        np.testing.assert_array_equal(ekf.GetStateCovariance(context), Phat)

    def test_unscented_kalman_filter(self):
        options = UnscentedKalmanFilterOptions()
        self.assertIsNone(options.initial_state_estimate)
        self.assertIsNone(options.initial_state_covariance)
        self.assertEqual(options.actuation_input_port_index,
                         InputPortSelection.kUseFirstInputIfItExists)
        self.assertEqual(options.measurement_output_port_index,
                         OutputPortSelection.kUseFirstOutputIfItExists)
        self.assertIsNone(options.process_noise_input_port_index)
        self.assertIsNone(options.measurement_noise_input_port_index)
        options.unscented_transform_parameters = \
            UnscentedKalmanFilterOptions.UnscentedTransformParameters(
                alpha=1, beta=2, kappa=lambda n: 3-n)
        self.assertEqual(options.unscented_transform_parameters.kappa(2), 1)
        self.assertEqual(options.use_square_root_method, False)
        self.assertIsNone(options.discrete_measurement_time_period)
        self.assertEqual(options.discrete_measurement_time_offset, 0.0)

        sys = LinearSystem(
            np.array([[0, 1], [0, 0]]), np.array([[0], [1]]),
            np.array([[1, 0]]), np.array([[0]]))
        W = np.eye(2)
        V = np.eye(1)
        ukf = UnscentedKalmanFilter(
            observed_system=sys,
            observed_system_context=sys.CreateDefaultContext(),
            W=W, V=V, options=options)

        self.assertTrue(ukf.get_observed_system_input_input_port().size(), 1)
        self.assertTrue(ukf.get_observed_system_output_input_port().size(), 1)
        self.assertTrue(ukf.get_estimated_state_output_port().size(), 2)

        context = ukf.CreateDefaultContext()
        xhat = np.array([1, 2])
        Phat = np.eye(2)
        ukf.SetStateEstimateAndCovariance(context, xhat, Phat)
        np.testing.assert_array_equal(ukf.GetStateEstimate(context), xhat)
        np.testing.assert_array_equal(ukf.GetStateCovariance(context), Phat)

    def test_binding(self):
        sys = LinearSystem(
            np.array([[0, 1], [0, 0]]), np.array([[0], [1]]),
            np.array([[1, 0]]), np.array([[0]]))
        W = np.eye(2)
        V = np.eye(1)
        kf = ExtendedKalmanFilter(
            observed_system=sys,
            observed_system_context=sys.CreateDefaultContext(),
            W=W, V=V)

        builder = DiagramBuilder()
        kf = builder.AddSystem(kf)
        builder.Build()
