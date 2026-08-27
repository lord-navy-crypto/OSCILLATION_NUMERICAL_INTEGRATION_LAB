# Validation results — Oscillation & Numerical Integration Lab v1.1.0

Audit environment: Python 3.13.5.

## Automated checks

- Core + repository tests: **21 passed**
- Python compilation: **PASS**
- `pyproject.toml` parsing: **PASS**
- preset JSON parsing: **PASS**
- macOS launcher shell syntax: **PASS**
- Python wheel build from `pyproject.toml`: **PASS**
- dynamic port helper test: **PASS**

`tests/test_app.py` remains available for environments with Streamlit installed. Streamlit was not available in the isolated audit container.

## Numerical spot checks

DOP853 versus exact undamped linear solution over `2 s`:

- maximum state-vector error: `1.110651766880e-10`

Observed convergence order using whole-trajectory error:

- Euler: `1.1477225625`
- Symplectic Euler: `1.0774553043`
- RK2 midpoint: `1.9965615541`
- RK4: `3.9899471679`

Driven resonance check (`gamma=0.5`, `F0=0.6`, 15 frequencies, 35 natural periods):

- maximum numerical/analytical relative amplitude difference: `4.244845604927e-04`
- estimated remaining free-transient fraction: `1.584613251158e-04`

Nonlinear pendulum period check at initial amplitudes `0.2`, `1.0`, and `2.0 rad`:

- maximum RK4/interpolated versus elliptic-integral absolute period error: `1.242097535936e-10 s`

Forced/damped energy-work balance (`gamma=0.4`, `F0=0.6`, `Omega=5.8 rad/s`):

- maximum relative balance residual: `2.865310676623e-06`

Long conservative energy behavior (`duration=10 s`, `dt=0.02 s`):

- Euler final relative energy change: `2.523984832810e+03`
- Symplectic Euler final relative energy change: `5.195156495549e-03`

This large contrast is a useful demonstration that visual smoothness alone does not establish numerical suitability for oscillatory Hamiltonian motion.
