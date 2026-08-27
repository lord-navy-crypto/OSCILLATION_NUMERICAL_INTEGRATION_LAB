import numpy as np
import pytest

from oscillation_lab import (
    Method,
    OscillatorParams,
    analytic_free_response,
    convergence_scan,
    linear_energy,
    nonlinear_period_scan,
    resonance_scan,
    simulate_dop853,
    simulate_fixed,
)


@pytest.mark.parametrize("gamma_factor", [0.2, 2.0, 4.0])
def test_analytic_response_preserves_initial_state(gamma_factor: float) -> None:
    omega0 = 2 * np.pi
    params = OscillatorParams(omega0=omega0, gamma=gamma_factor * omega0)
    initial = np.asarray([0.7, -0.2])
    state = analytic_free_response(np.asarray([0.0, 0.2]), initial, params)
    np.testing.assert_allclose(state[0], initial, rtol=0, atol=1e-14)


def test_rk4_agrees_with_exact_free_response() -> None:
    params = OscillatorParams()
    result = simulate_fixed(np.asarray([1.0, 0.0]), params, duration=1.0, dt=0.005)
    exact = analytic_free_response(np.asarray([1.0]), np.asarray([1.0, 0.0]), params)[0]
    assert np.linalg.norm(result["state"][-1] - exact) < 1e-6


def test_dop853_is_an_independent_high_accuracy_reference() -> None:
    params = OscillatorParams(gamma=0.4)
    result = simulate_dop853(np.asarray([1.0, 0.2]), params, duration=2.0, samples=101)
    exact = analytic_free_response(result["time"], np.asarray([1.0, 0.2]), params)
    assert np.max(np.linalg.norm(result["state"] - exact, axis=1)) < 1e-8


@pytest.mark.parametrize(
    ("method", "lower", "upper"),
    [
        (Method.EULER, 0.8, 1.3),
        (Method.SYMPLECTIC_EULER, 0.8, 1.3),
        (Method.RK2, 1.7, 2.3),
        (Method.RK4, 3.7, 4.3),
    ],
)
def test_observed_convergence_order(method: Method, lower: float, upper: float) -> None:
    result = convergence_scan(
        np.asarray([0.04, 0.02, 0.01, 0.005]),
        np.asarray([1.0, 0.0]),
        OscillatorParams(),
        duration=1.0,
        method=method,
    )
    assert lower < result["observed_order"] < upper


def test_symplectic_euler_bounds_energy_better_than_euler() -> None:
    params = OscillatorParams()
    initial = np.asarray([1.0, 0.0])
    euler = simulate_fixed(initial, params, duration=10.0, dt=0.02, method=Method.EULER)
    symplectic = simulate_fixed(initial, params, duration=10.0, dt=0.02, method=Method.SYMPLECTIC_EULER)
    euler_change = abs(linear_energy(euler["state"], params)[-1] / linear_energy(euler["state"], params)[0] - 1)
    symplectic_change = abs(linear_energy(symplectic["state"], params)[-1] / linear_energy(symplectic["state"], params)[0] - 1)
    assert symplectic_change < euler_change * 1e-3


def test_resonance_scan_matches_analytical_steady_amplitude() -> None:
    params = OscillatorParams(gamma=0.5, force_amplitude=0.6)
    result = resonance_scan(
        np.linspace(0.7 * params.omega0, 1.3 * params.omega0, 15),
        params,
        natural_periods=35,
        steps_per_natural_period=100,
    )
    assert np.max(result["relative_difference"]) < 0.04


def test_nonlinear_period_increases_with_amplitude() -> None:
    result = nonlinear_period_scan(np.asarray([0.2, 1.0, 2.0]), 2 * np.pi)
    assert np.all(np.diff(result["exact_nonlinear_period"]) > 0)
    assert np.max(np.abs(result["numerical_period"] - result["exact_nonlinear_period"])) < 0.002


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        simulate_fixed(np.zeros(2), OscillatorParams(mass=-1), duration=1, dt=.01)
    with pytest.raises(ValueError):
        nonlinear_period_scan(np.asarray([np.pi]), 1.0)


def test_forced_damped_energy_balance_closes() -> None:
    from oscillation_lab import energy_balance_diagnostic

    params = OscillatorParams(gamma=0.4, force_amplitude=0.6, force_frequency=5.8)
    result = simulate_dop853(
        np.asarray([1.0, 0.2]), params, duration=5.0, samples=2001
    )
    diagnostic = energy_balance_diagnostic(result["time"], result["state"], params)
    assert diagnostic["max_relative_balance_residual"] < 1e-4


def test_resonance_scan_reports_transient_estimate() -> None:
    params = OscillatorParams(gamma=0.5, force_amplitude=0.6)
    result = resonance_scan(
        np.linspace(0.9 * params.omega0, 1.1 * params.omega0, 5),
        params,
        natural_periods=20,
        steps_per_natural_period=80,
    )
    transient = result["estimated_remaining_free_transient"]
    assert transient.shape == result["drive_frequency"].shape
    assert np.all((transient >= 0) & (transient <= 1))


def test_interpolated_nonlinear_period_is_high_accuracy() -> None:
    result = nonlinear_period_scan(np.asarray([0.2, 1.0, 2.0]), 2 * np.pi)
    error = np.max(np.abs(result["numerical_period"] - result["exact_nonlinear_period"]))
    assert error < 1e-6


def test_convergence_does_not_call_physical_damping_energy_loss_numerical_drift() -> None:
    result = convergence_scan(
        np.asarray([0.02, 0.01]),
        np.asarray([1.0, 0.0]),
        OscillatorParams(gamma=0.3),
        duration=1.0,
        method=Method.RK4,
    )
    assert np.all(np.isnan(result["maximum_relative_energy_drift"]))
