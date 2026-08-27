# Oscillation & Numerical Integration Lab

**Version 1.1.0 — GitHub-ready computational-physics laboratory**

An interactive localhost platform rebuilt from `h (1).ipynb`. It studies linear, damped, driven, and nonlinear oscillators while explicitly testing the numerical methods used to calculate them.

The physics/numerical core is independent from Streamlit. Version 1.1.0 also uses a **trend-first** interface: plots first, a compact key-results table second, and complete numerical tables only on request.

## What changed in v1.1.0

- Added a physical **energy-work balance diagnostic** for the forced/damped linear oscillator:
  \[
  E(t)-E(0)=\int_0^t(P_{\mathrm{damping}}+P_{\mathrm{input}})\,dt.
  \]
- Conservative energy drift is no longer semantically conflated with physical energy decay under damping.
- Nonlinear pendulum periods now use interpolated turning-point times, substantially improving period measurement without changing the integrator.
- Resonance scans report an **estimated remaining free-transient amplitude**, warning when the numerical response may not yet be settled.
- Method, convergence, damping, resonance, beat, nonlinear, external-data, and validation pages use **figure → key data → on-demand full data**.
- Zero initial mechanical energy no longer causes a divide-by-zero in normalized method diagnostics.
- macOS and Windows launchers automatically choose a free port in **8501–8520**.
- Added GitHub issue templates, contribution guidance, security/scope notes, release checklist, editor settings, and an optional CI workflow template.
- The Actions workflow is shipped as an example instead of an active workflow so ordinary HTTPS PAT uploads do not require GitHub's special workflow permission.

## Implemented model

The linear driven oscillator is

\[
m\ddot u+m\gamma\dot u+m\omega_0^2u=F_0\cos(\Omega t).
\]

The nonlinear benchmark replaces the linear restoring term with the pendulum form

\[
\ddot\theta+\gamma\dot\theta+\omega_0^2\sin\theta=\text{forcing term}.
\]

The nonlinear period benchmark in this platform is run without external forcing and is compared with the exact elliptic-integral result.

## Numerical methods

- Forward Euler
- Semi-implicit Euler (symplectic Euler in the conservative separable case)
- RK2 midpoint
- RK4
- SciPy DOP853 adaptive reference
- exact linear free response for underdamped, critically damped, and overdamped cases

The convergence page measures whole-trajectory state error and observed log-log order. The exact endpoint is respected by adjusting the actual uniform timestep to be no larger than the requested `dt`.

## Experiments

### Method comparison

Fixed-step methods are compared against the analytical free response when forcing is zero; otherwise DOP853 supplies the numerical reference. Mechanical energy is shown, but conservation language is used only for conservative configurations.

### Convergence

A logarithmic timestep scan measures final error, maximum trajectory error, and observed convergence order. Mechanical-energy drift is a conservative-only diagnostic.

### Damping regimes

Underdamped, critical, and overdamped analytical responses are generated from identical initial conditions. For the implemented equation, critical damping is

\[
\gamma_c=2\omega_0.
\]

### Resonance and beats

The frequency response uses drive frequency as the explicit horizontal scan axis and compares a late-time harmonic-fit amplitude with the analytical steady-state amplitude. A peak-to-peak amplitude is retained as a secondary diagnostic. Version 1.1.0 also estimates how much free transient should remain after the chosen integration time.

### Nonlinear pendulum

The exact period is

\[
T(\theta_0)=\frac{4}{\omega_0}K\!\left(\sin^2\frac{\theta_0}{2}\right),
\]

where `K` is the complete elliptic integral of the first kind. RK4 turning points are interpolated between timesteps before period comparison.

### External data

CSV files with `time` and `displacement` can be aligned against the DOP853 model. If a `velocity` column is present, velocity RMSE and residuals are also computed. The complete aligned table is exposed only on request.

## Energy-work balance

For the linear oscillator,

\[
E=\frac12m\left(v^2+\omega_0^2u^2\right),
\]

\[
P_{\mathrm{damping}}=-m\gamma v^2,
\qquad
P_{\mathrm{input}}=F_0\cos(\Omega t)v.
\]

The validation suite checks the numerical residual of the integrated balance. This distinguishes real physical energy exchange from numerical drift.

## Quick start

### macOS

1. Extract the complete project folder.
2. Double-click `RUN_OSCILLATION_LAB.command`.
3. The launcher creates `.venv`, installs dependencies, chooses a free localhost port in `8501–8520`, and starts Streamlit.

If required once:

```bash
chmod +x RUN_OSCILLATION_LAB.command
./RUN_OSCILLATION_LAB.command
```

### Windows

Double-click `RUN_OSCILLATION_LAB.bat`.

### Manual

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`tests/test_app.py` requires Streamlit. Core and repository tests can be run independently.

## Data presentation

Large data sets are not rendered by default. The UI follows:

1. plot / trend;
2. key numerical quantities;
3. **More data / complete data** button;
4. full table and CSV download only on request.

## Repository layout

```text
.
├── app.py
├── oscillation_lab/
│   ├── __init__.py
│   └── core.py
├── presets/
├── tests/
├── tools/find_free_port.py
├── docs/
├── .github/ISSUE_TEMPLATE/
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── RELEASE_CHECKLIST.md
├── RUN_OSCILLATION_LAB.command
├── RUN_OSCILLATION_LAB.bat
├── requirements.txt
└── pyproject.toml
```

## Optional GitHub Actions

The CI file is stored at `docs/github-actions/tests.yml.example`. See `docs/GITHUB_ACTIONS_OPTIONAL.md`. It is intentionally inactive in the downloadable package to avoid PAT workflow-scope upload failures.

## Scope

This is educational/research numerical software, not safety-certified engineering software. Solver agreement and convergence checks support a calculation; they do not replace model validation for a real physical system.

No software license is added automatically in this package. Repository owners should choose and add a license deliberately before redistribution if desired.
