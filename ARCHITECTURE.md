# Architecture

`oscillation_lab/core.py` owns equations, exact solutions, fixed/adaptive integration, convergence, resonance, nonlinear-period, and energy-balance calculations. `app.py` owns Streamlit controls, plotting, progress, configuration, and exports. Core tests are independent of Streamlit.

Long-result rendering is deliberately lazy: plots and key values appear first, and complete DataFrames are built only after the user enables the complete-data panel.
