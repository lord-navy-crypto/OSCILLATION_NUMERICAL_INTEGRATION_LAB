# Audit report — Oscillation & Numerical Integration Lab v1.1.0

## Scope

The v1.0.0 package was unpacked and its numerical core, analytical references, Streamlit interface, launchers, tests, configuration handling, and GitHub workflow were reviewed before modification.

## Important findings and changes

1. **Energy semantics under damping/forcing** — mechanical energy change is physical when damping or forcing is present; it should not be called numerical drift. v1.1.0 keeps conservative drift only for conservative cases and adds an energy-work balance diagnostic for driven/damped trajectories.
2. **Energy-work balance** — the new diagnostic compares `E(t)-E(0)` with the time integral of damping power plus input power and reports a normalized residual.
3. **Resonance amplitude extraction** — the original late-time amplitude used only extrema. v1.1.0 estimates the drive-frequency harmonic amplitude by a two-basis least-squares projection and retains the peak-to-peak result as a secondary diagnostic.
4. **Nonlinear period timing** — the original numerical turning point used the nearest stored timestep. v1.1.0 linearly interpolates the angular-velocity zero crossing, greatly reducing discretization error in the measured period.
5. **Resonance settling** — a late-time peak-to-peak amplitude can still contain a surviving free transient when damping is weak. v1.1.0 reports the analytical decay estimate `exp(-gamma*t/2)` for the integration horizon and warns when it remains above 1%.
6. **External velocity data** — the UI previously documented optional velocity input but did not use it. v1.1.0 aligns velocity to the DOP853 reference and reports velocity RMSE/residuals when supplied.
7. **Zero-energy normalization** — method-comparison energy plots now guard against division by zero when the initial mechanical energy is zero.
8. **Data presentation** — figures and trends come first, key numerical values second, and complete tables/CSV only on request.
9. **Launcher robustness** — both launchers now find a free localhost port in 8501–8520.
10. **Configuration import** — imported JSON receives stronger physical and scan-bound validation.
11. **GitHub upload compatibility** — the CI workflow is shipped as an inactive example rather than an active `.github/workflows` file, avoiding PAT workflow-scope push rejection.

## Physics semantics retained

- Exact free response covers underdamped, critical, and overdamped regimes.
- Critical damping for the implemented equation remains `gamma_c = 2*omega0`.
- Resonance uses drive frequency as the scan axis.
- DOP853 remains an independent numerical reference, not an analytical solution.
- The nonlinear pendulum benchmark remains checked against the complete elliptic-integral period.

## Remaining limitations

- A resonance numerical scan must be long enough for transients to decay; harmonic fitting reduces window bias, but the transient indicator still needs to be checked.
- The nonlinear forcing form is a generic angular-acceleration forcing model; the nonlinear period benchmark itself is unforced.
- DOP853 agreement validates integration, not the applicability of the oscillator model to a real experiment.
- Streamlit AppTest was not executed in the isolated audit container because Streamlit was unavailable there; app source compilation passed.
