"""Reusable physics and numerical-analysis core for oscillation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.special import ellipk


ProgressCallback = Callable[[int, int], None]


class Method(str, Enum):
    EULER = "euler"
    SYMPLECTIC_EULER = "symplectic_euler"
    RK2 = "rk2"
    RK4 = "rk4"


@dataclass(frozen=True)
class OscillatorParams:
    mass: float = 1.0
    omega0: float = 2.0 * np.pi
    gamma: float = 0.0
    force_amplitude: float = 0.0
    force_frequency: float = 0.0


def _positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def validate_params(params: OscillatorParams) -> None:
    _positive("mass", params.mass)
    _positive("omega0", params.omega0)
    if params.gamma < 0 or not np.isfinite(params.gamma):
        raise ValueError("gamma must be non-negative and finite")
    if not np.isfinite(params.force_amplitude):
        raise ValueError("force_amplitude must be finite")
    if params.force_frequency < 0 or not np.isfinite(params.force_frequency):
        raise ValueError("force_frequency must be non-negative and finite")


def linear_oscillator_ode(
    t: float,
    state: np.ndarray,
    params: OscillatorParams,
) -> np.ndarray:
    displacement, velocity = state
    acceleration = (
        -params.omega0**2 * displacement
        - params.gamma * velocity
        + params.force_amplitude / params.mass * np.cos(params.force_frequency * t)
    )
    return np.asarray([velocity, acceleration])


def nonlinear_pendulum_ode(
    t: float,
    state: np.ndarray,
    params: OscillatorParams,
) -> np.ndarray:
    theta, angular_velocity = state
    angular_acceleration = (
        -params.omega0**2 * np.sin(theta)
        - params.gamma * angular_velocity
        + params.force_amplitude / params.mass * np.cos(params.force_frequency * t)
    )
    return np.asarray([angular_velocity, angular_acceleration])


def _step(
    ode: Callable[[float, np.ndarray, OscillatorParams], np.ndarray],
    t: float,
    state: np.ndarray,
    dt: float,
    params: OscillatorParams,
    method: Method,
) -> np.ndarray:
    if method is Method.EULER:
        return state + dt * ode(t, state, params)
    if method is Method.SYMPLECTIC_EULER:
        acceleration = ode(t, state, params)[1]
        velocity = state[1] + dt * acceleration
        displacement = state[0] + dt * velocity
        return np.asarray([displacement, velocity])
    if method is Method.RK2:
        k1 = ode(t, state, params)
        k2 = ode(t + 0.5 * dt, state + 0.5 * dt * k1, params)
        return state + dt * k2
    if method is Method.RK4:
        k1 = ode(t, state, params)
        k2 = ode(t + 0.5 * dt, state + 0.5 * dt * k1, params)
        k3 = ode(t + 0.5 * dt, state + 0.5 * dt * k2, params)
        k4 = ode(t + dt, state + dt * k3, params)
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    raise ValueError(f"unsupported method: {method}")


def simulate_fixed(
    initial_state: np.ndarray,
    params: OscillatorParams,
    *,
    duration: float,
    dt: float,
    method: Method | str = Method.RK4,
    nonlinear: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray | float | str]:
    """Integrate to the exact requested endpoint using a uniform step <= dt."""

    validate_params(params)
    _positive("duration", duration)
    _positive("dt", dt)
    selected = Method(method)
    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must contain two finite values")
    steps = int(np.ceil(duration / dt))
    if steps > 2_000_000:
        raise ValueError("trajectory exceeds the 2,000,000-step safety limit")
    actual_dt = duration / steps
    times = np.linspace(0.0, duration, steps + 1)
    states = np.empty((steps + 1, 2), dtype=float)
    states[0] = initial
    ode = nonlinear_pendulum_ode if nonlinear else linear_oscillator_ode
    stride = max(1, steps // 100)
    for index in range(steps):
        states[index + 1] = _step(
            ode, times[index], states[index], actual_dt, params, selected
        )
        if not np.all(np.isfinite(states[index + 1])):
            raise FloatingPointError(f"non-finite state at step {index + 1}")
        if progress_callback and ((index + 1) == steps or (index + 1) % stride == 0):
            progress_callback(index + 1, steps)
    return {
        "time": times,
        "state": states,
        "actual_dt": actual_dt,
        "method": selected.value,
    }


def simulate_dop853(
    initial_state: np.ndarray,
    params: OscillatorParams,
    *,
    duration: float,
    samples: int = 1000,
    nonlinear: bool = False,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> dict[str, np.ndarray | str]:
    validate_params(params)
    _positive("duration", duration)
    if samples < 2 or rtol <= 0 or atol <= 0:
        raise ValueError("invalid adaptive solver settings")
    initial = np.asarray(initial_state, dtype=float)
    ode = nonlinear_pendulum_ode if nonlinear else linear_oscillator_ode
    times = np.linspace(0.0, duration, samples)
    result = solve_ivp(
        lambda t, state: ode(t, state, params),
        (0.0, duration),
        initial,
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return {"time": result.t, "state": result.y.T, "method": "dop853"}


def analytic_free_response(
    times: np.ndarray,
    initial_state: np.ndarray,
    params: OscillatorParams,
) -> np.ndarray:
    """Exact linear response for an unforced oscillator in all damping regimes."""

    validate_params(params)
    if params.force_amplitude != 0:
        raise ValueError("analytic_free_response requires zero external force")
    t = np.asarray(times, dtype=float)
    u0, v0 = np.asarray(initial_state, dtype=float)
    alpha = params.gamma / 2.0
    omega0 = params.omega0
    tolerance = 64.0 * np.finfo(float).eps * omega0
    if alpha < omega0 - tolerance:
        omega_d = np.sqrt(omega0**2 - alpha**2)
        c1 = u0
        c2 = (v0 + alpha * u0) / omega_d
        cosine = np.cos(omega_d * t)
        sine = np.sin(omega_d * t)
        decay = np.exp(-alpha * t)
        displacement = decay * (c1 * cosine + c2 * sine)
        velocity = decay * (
            -alpha * (c1 * cosine + c2 * sine)
            + omega_d * (-c1 * sine + c2 * cosine)
        )
    elif abs(alpha - omega0) <= tolerance:
        c1 = u0
        c2 = v0 + omega0 * u0
        decay = np.exp(-omega0 * t)
        displacement = decay * (c1 + c2 * t)
        velocity = decay * (c2 - omega0 * (c1 + c2 * t))
    else:
        root = np.sqrt(alpha**2 - omega0**2)
        r1, r2 = -alpha + root, -alpha - root
        c1 = (v0 - r2 * u0) / (r1 - r2)
        c2 = u0 - c1
        displacement = c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)
        velocity = r1 * c1 * np.exp(r1 * t) + r2 * c2 * np.exp(r2 * t)
    return np.column_stack((displacement, velocity))


def linear_energy(states: np.ndarray, params: OscillatorParams) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    return 0.5 * params.mass * (
        values[..., 1] ** 2 + params.omega0**2 * values[..., 0] ** 2
    )


def nonlinear_energy(states: np.ndarray, params: OscillatorParams) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    return params.mass * (
        0.5 * values[..., 1] ** 2
        + params.omega0**2 * (1.0 - np.cos(values[..., 0]))
    )


def energy_power_balance(
    times: np.ndarray,
    states: np.ndarray,
    params: OscillatorParams,
) -> dict[str, np.ndarray]:
    """Return energy and instantaneous damping/input powers."""

    values = np.asarray(states, dtype=float)
    velocity = values[:, 1]
    damping_power = -params.mass * params.gamma * velocity**2
    input_power = (
        params.force_amplitude * np.cos(params.force_frequency * np.asarray(times)) * velocity
    )
    return {
        "energy": linear_energy(values, params),
        "damping_power": damping_power,
        "input_power": input_power,
        "net_power": damping_power + input_power,
    }


def energy_balance_diagnostic(
    times: np.ndarray,
    states: np.ndarray,
    params: OscillatorParams,
) -> dict[str, np.ndarray | float]:
    """Check E(t)-E(0)=integral(P_damping+P_input) dt for the linear model."""

    t = np.asarray(times, dtype=float)
    values = np.asarray(states, dtype=float)
    if t.ndim != 1 or values.shape != (t.size, 2) or t.size < 2:
        raise ValueError("times and states must describe one two-state trajectory")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(values)) or np.any(np.diff(t) <= 0):
        raise ValueError("trajectory samples must be finite with strictly increasing time")
    balance = energy_power_balance(t, values, params)
    energy = np.asarray(balance["energy"])
    net_power = np.asarray(balance["net_power"])
    integrated_power = np.concatenate(([0.0], cumulative_trapezoid(net_power, t)))
    residual = energy - energy[0] - integrated_power
    scale = max(float(np.max(np.abs(energy))), abs(float(energy[0])), np.finfo(float).tiny)
    return {
        **balance,
        "integrated_net_work": integrated_power,
        "balance_residual": residual,
        "max_relative_balance_residual": float(np.max(np.abs(residual)) / scale),
    }


def method_comparison(
    initial_state: np.ndarray,
    params: OscillatorParams,
    *,
    duration: float,
    dt: float,
) -> dict[str, dict[str, np.ndarray | float | str]]:
    output = {
        method.value: simulate_fixed(
            initial_state, params, duration=duration, dt=dt, method=method
        )
        for method in Method
    }
    times = np.asarray(output[Method.RK4.value]["time"])
    if params.force_amplitude == 0:
        output["analytic"] = {
            "time": times,
            "state": analytic_free_response(times, initial_state, params),
            "method": "analytic",
        }
    output["dop853"] = simulate_dop853(
        initial_state, params, duration=duration, samples=times.size
    )
    return output


def convergence_scan(
    dt_values: np.ndarray,
    initial_state: np.ndarray,
    params: OscillatorParams,
    *,
    duration: float,
    method: Method | str,
) -> dict[str, np.ndarray | float]:
    values = np.asarray(dt_values, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(values <= 0):
        raise ValueError("dt_values must contain at least two positive values")
    selected = Method(method)
    final_errors, maximum_errors, drifts, actual_steps = [], [], [], []
    for dt in values:
        result = simulate_fixed(
            initial_state, params, duration=duration, dt=float(dt), method=selected
        )
        states = np.asarray(result["state"])
        reference = analytic_free_response(np.asarray(result["time"]), initial_state, params)
        trajectory_error = np.linalg.norm(states - reference, axis=1)
        final_errors.append(float(trajectory_error[-1]))
        maximum_errors.append(float(np.max(trajectory_error)))
        energy = linear_energy(states, params)
        relative_energy_change = float(
            np.max(np.abs(energy - energy[0])) / max(abs(energy[0]), np.finfo(float).tiny)
        )
        drifts.append(relative_energy_change if params.gamma == 0 and params.force_amplitude == 0 else np.nan)
        actual_steps.append(float(result["actual_dt"]))
    errors_array = np.asarray(maximum_errors)
    mask = np.isfinite(errors_array) & (errors_array > 0)
    slope = float(np.polyfit(np.log(np.asarray(actual_steps)[mask]), np.log(errors_array[mask]), 1)[0])
    return {
        "requested_dt": values,
        "actual_dt": np.asarray(actual_steps),
        "final_state_error": np.asarray(final_errors),
        "maximum_state_error": errors_array,
        "maximum_relative_energy_drift": np.asarray(drifts),
        "observed_order": slope,
    }


def steady_state_amplitude(
    drive_frequency: np.ndarray | float,
    params: OscillatorParams,
) -> np.ndarray:
    frequency = np.asarray(drive_frequency, dtype=float)
    denominator = np.sqrt(
        (params.omega0**2 - frequency**2) ** 2
        + (params.gamma * frequency) ** 2
    )
    return (abs(params.force_amplitude) / params.mass) / denominator


def resonance_scan(
    frequency_values: np.ndarray,
    params: OscillatorParams,
    *,
    initial_state: np.ndarray = np.asarray([0.0, 0.0]),
    natural_periods: int = 80,
    steps_per_natural_period: int = 160,
    sample_fraction: float = 0.2,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray]:
    """Numerically scan drive frequency; frequency is the explicit scan axis."""

    validate_params(params)
    frequencies = np.asarray(frequency_values, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(frequencies <= 0):
        raise ValueError("frequency_values must be a positive one-dimensional array")
    if params.gamma <= 0 or params.force_amplitude == 0:
        raise ValueError("resonance scan requires nonzero damping and forcing")
    if natural_periods < 10 or steps_per_natural_period < 40:
        raise ValueError("insufficient resonance integration duration or resolution")
    period = 2.0 * np.pi / params.omega0
    dt = period / steps_per_natural_period
    steps = natural_periods * steps_per_natural_period
    state = np.asarray(
        [np.full(frequencies.size, initial_state[0]), np.full(frequencies.size, initial_state[1])]
    )
    time = 0.0
    start_sample = int(steps * (1.0 - sample_fraction))
    minimum = np.full(frequencies.size, np.inf)
    maximum = np.full(frequencies.size, -np.inf)
    sum_cc = np.zeros(frequencies.size)
    sum_ss = np.zeros(frequencies.size)
    sum_cs = np.zeros(frequencies.size)
    sum_yc = np.zeros(frequencies.size)
    sum_ys = np.zeros(frequencies.size)
    sample_count = 0
    stride = max(1, steps // 100)

    def vector_ode(t: float, y: np.ndarray) -> np.ndarray:
        u, v = y
        return np.asarray(
            [
                v,
                -params.omega0**2 * u
                - params.gamma * v
                + params.force_amplitude / params.mass * np.cos(frequencies * t),
            ]
        )

    def vector_rk4(t: float, y: np.ndarray) -> np.ndarray:
        k1 = vector_ode(t, y)
        k2 = vector_ode(t + dt / 2, y + dt * k1 / 2)
        k3 = vector_ode(t + dt / 2, y + dt * k2 / 2)
        k4 = vector_ode(t + dt, y + dt * k3)
        return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    for step in range(steps):
        state = vector_rk4(time, state)
        time += dt
        if step >= start_sample:
            minimum = np.minimum(minimum, state[0])
            maximum = np.maximum(maximum, state[0])
            cosine = np.cos(frequencies * time)
            sine = np.sin(frequencies * time)
            sum_cc += cosine * cosine
            sum_ss += sine * sine
            sum_cs += cosine * sine
            sum_yc += state[0] * cosine
            sum_ys += state[0] * sine
            sample_count += 1
        if progress_callback and ((step + 1) == steps or (step + 1) % stride == 0):
            progress_callback(step + 1, steps)
    peak_to_peak_amplitude = 0.5 * (maximum - minimum)
    determinant = sum_cc * sum_ss - sum_cs**2
    cosine_coefficient = np.divide(
        sum_yc * sum_ss - sum_ys * sum_cs,
        determinant,
        out=np.full_like(determinant, np.nan),
        where=np.abs(determinant) > np.finfo(float).tiny,
    )
    sine_coefficient = np.divide(
        sum_ys * sum_cc - sum_yc * sum_cs,
        determinant,
        out=np.full_like(determinant, np.nan),
        where=np.abs(determinant) > np.finfo(float).tiny,
    )
    harmonic_amplitude = np.hypot(cosine_coefficient, sine_coefficient)
    analytic = steady_state_amplitude(frequencies, params)
    total_time = steps * dt
    transient_factor = float(np.exp(-0.5 * params.gamma * total_time))
    return {
        "drive_frequency": frequencies,
        "frequency_ratio": frequencies / params.omega0,
        "numerical_amplitude": harmonic_amplitude,
        "harmonic_fit_amplitude": harmonic_amplitude,
        "peak_to_peak_amplitude": peak_to_peak_amplitude,
        "analytic_amplitude": analytic,
        "relative_difference": np.abs(harmonic_amplitude - analytic) / np.maximum(analytic, np.finfo(float).tiny),
        "estimated_remaining_free_transient": np.full(frequencies.size, transient_factor),
        "fit_sample_count": np.full(frequencies.size, sample_count, dtype=int),
    }


def nonlinear_period_scan(
    amplitude_values: np.ndarray,
    omega0: float,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, np.ndarray]:
    """Compare numerical nonlinear periods with the exact elliptic-integral formula."""

    _positive("omega0", omega0)
    amplitudes = np.asarray(amplitude_values, dtype=float)
    if amplitudes.ndim != 1 or amplitudes.size == 0 or np.any(amplitudes <= 0) or np.any(amplitudes >= np.pi):
        raise ValueError("amplitudes must lie strictly between 0 and pi")
    small_period = 2.0 * np.pi / omega0
    exact_periods = 4.0 * ellipk(np.sin(amplitudes / 2.0) ** 2) / omega0
    numerical_periods = []
    params = OscillatorParams(omega0=omega0)
    for index, (amplitude, exact_period) in enumerate(zip(amplitudes, exact_periods)):
        result = simulate_fixed(
            np.asarray([amplitude, 0.0]),
            params,
            duration=float(1.4 * exact_period),
            dt=small_period / 1200.0,
            method=Method.RK4,
            nonlinear=True,
        )
        times = np.asarray(result["time"])
        velocity = np.asarray(result["state"])[:, 1]
        maxima = np.flatnonzero((velocity[:-1] > 0) & (velocity[1:] <= 0)) + 1
        if maxima.size:
            crossing = int(maxima[0])
            previous = crossing - 1
            denominator = velocity[previous] - velocity[crossing]
            fraction = 0.0 if abs(denominator) <= np.finfo(float).tiny else velocity[previous] / denominator
            period_estimate = times[previous] + fraction * (times[crossing] - times[previous])
            numerical_periods.append(float(period_estimate))
        else:
            numerical_periods.append(np.nan)
        if progress_callback:
            progress_callback(index + 1, amplitudes.size)
    return {
        "initial_amplitude": amplitudes,
        "small_angle_period": np.full(amplitudes.size, small_period),
        "exact_nonlinear_period": exact_periods,
        "numerical_period": np.asarray(numerical_periods),
        "period_ratio": exact_periods / small_period,
    }
