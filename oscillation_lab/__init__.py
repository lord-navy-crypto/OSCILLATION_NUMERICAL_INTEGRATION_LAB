"""Public API for the Oscillation & Numerical Integration Lab."""

from .core import (
    Method,
    OscillatorParams,
    analytic_free_response,
    convergence_scan,
    energy_balance_diagnostic,
    energy_power_balance,
    linear_energy,
    method_comparison,
    nonlinear_energy,
    nonlinear_period_scan,
    resonance_scan,
    simulate_dop853,
    simulate_fixed,
    steady_state_amplitude,
)

__all__ = [
    "Method",
    "OscillatorParams",
    "analytic_free_response",
    "convergence_scan",
    "energy_balance_diagnostic",
    "energy_power_balance",
    "linear_energy",
    "method_comparison",
    "nonlinear_energy",
    "nonlinear_period_scan",
    "resonance_scan",
    "simulate_dop853",
    "simulate_fixed",
    "steady_state_amplitude",
]
