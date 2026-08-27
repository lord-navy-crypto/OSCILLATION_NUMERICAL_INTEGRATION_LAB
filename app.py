from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oscillation_lab import (
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
)


APP_VERSION = "1.1.0"
METHOD_LABELS = {
    "Euler": Method.EULER,
    "Semi-implicit Euler (symplectic when conservative)": Method.SYMPLECTIC_EULER,
    "RK2 midpoint": Method.RK2,
    "RK4": Method.RK4,
}

st.set_page_config(
    page_title="Oscillation & Numerical Integration Lab",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp{background:linear-gradient(145deg,#f7faff 0%,#fff 45%,#f7f4ff 100%)}
    .hero{padding:1.45rem 1.7rem;border-radius:20px;color:white;
      background:linear-gradient(120deg,#172a46 0%,#365fa0 50%,#6b3fa0 100%);
      box-shadow:0 14px 34px rgba(23,42,70,.2);margin-bottom:1rem}
    .hero h1{margin:0 0 .35rem;font-size:2.1rem}.hero p{margin:0;opacity:.93}
    .note{padding:.85rem 1rem;border-left:4px solid #365fa0;border-radius:8px;
      background:#edf4ff;margin:.5rem 0 1rem}
    div[data-testid="stMetric"]{background:white;border:1px solid #dce5ef;
      padding:.7rem;border-radius:14px;box-shadow:0 4px 14px rgba(23,42,70,.05)}
    </style>
    """,
    unsafe_allow_html=True,
)


def defaults() -> dict[str, object]:
    return {
        "mass": 1.0,
        "omega0": float(2 * np.pi),
        "gamma": 0.0,
        "force_amplitude": 0.0,
        "force_frequency": float(1.92 * np.pi),
        "u0": 1.0,
        "v0": 0.0,
        "duration": 10.0,
        "dt": 0.02,
        "convergence_method": "RK4",
        "convergence_dt_min": 0.0025,
        "convergence_dt_max": 0.08,
        "convergence_points": 7,
        "resonance_gamma": 0.5,
        "resonance_force": 0.6,
        "resonance_ratio_min": 0.5,
        "resonance_ratio_max": 1.5,
        "resonance_points": 100,
        "nonlinear_amp_max": 2.8,
        "nonlinear_points": 40,
        "single_theta0": float(np.pi / 2),
    }


RESULT_KEYS = (
    "comparison_result",
    "convergence_result",
    "damping_result",
    "resonance_result",
    "beat_result",
    "nonlinear_result",
    "external_result",
    "validation_result",
)


def initialize() -> None:
    for key, value in defaults().items():
        st.session_state.setdefault(key, value)
    for key in RESULT_KEYS:
        st.session_state.setdefault(key, None)


def params(**overrides: float) -> OscillatorParams:
    values = {
        "mass": float(st.session_state.mass),
        "omega0": float(st.session_state.omega0),
        "gamma": float(st.session_state.gamma),
        "force_amplitude": float(st.session_state.force_amplitude),
        "force_frequency": float(st.session_state.force_frequency),
    }
    values.update(overrides)
    return OscillatorParams(**values)


def initial_state() -> np.ndarray:
    return np.asarray([st.session_state.u0, st.session_state.v0], dtype=float)


def config() -> dict[str, object]:
    return {
        "schema": "oscillation-numerical-integration-lab-v1",
        "app_version": APP_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **{key: st.session_state[key] for key in defaults()},
    }


def load_config(uploaded) -> None:
    if uploaded is None:
        return
    signature = (uploaded.name, uploaded.size)
    if st.session_state.get("loaded_signature") == signature:
        return
    try:
        loaded = json.loads(uploaded.getvalue().decode("utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("configuration root must be a JSON object")
        candidate = defaults()
        for key, fallback in candidate.items():
            if key in loaded:
                candidate[key] = type(fallback)(loaded[key])
        positive_keys=(
            "mass", "omega0", "duration", "dt", "convergence_dt_min", "convergence_dt_max",
            "resonance_gamma", "resonance_force", "resonance_ratio_min", "resonance_ratio_max",
            "nonlinear_amp_max", "single_theta0",
        )
        if any(float(candidate[key]) <= 0 for key in positive_keys):
            raise ValueError("physical scales, durations, timesteps, and scan bounds must be positive")
        if float(candidate["gamma"]) < 0 or float(candidate["force_frequency"]) < 0:
            raise ValueError("damping and drive frequency must be non-negative")
        if candidate["convergence_dt_min"] >= candidate["convergence_dt_max"]:
            raise ValueError("convergence dt minimum must be smaller than maximum")
        if candidate["resonance_ratio_min"] >= candidate["resonance_ratio_max"]:
            raise ValueError("resonance scan minimum must be smaller than maximum")
        if not 0 < float(candidate["nonlinear_amp_max"]) < np.pi or not 0 < float(candidate["single_theta0"]) < np.pi:
            raise ValueError("nonlinear pendulum angles must lie strictly between 0 and pi")
        if candidate["convergence_method"] not in METHOD_LABELS:
            raise ValueError("unknown convergence method")
        for key, value in candidate.items():
            st.session_state[key] = value
        for key in RESULT_KEYS:
            st.session_state[key] = None
        st.session_state.loaded_signature = signature
        st.success("Configuration loaded. Run an experiment to refresh results.")
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        st.error(f"Could not load configuration: {exc}")


def progress(label: str):
    bar = st.progress(0, text=label)

    def update(done: int, total: int) -> None:
        bar.progress(done / total, text=f"{label}: {done}/{total}")

    return bar, update


def download_frame(label: str, frame: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def plot_lines(
    x: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    title: str,
    x_title: str,
    y_title: str,
    log_y: bool = False,
) -> go.Figure:
    colors = ["#1f5f99", "#d1495b", "#2a9d8f", "#7b2cbf", "#f4a261", "#111111"]
    figure = go.Figure()
    for index, (name, values) in enumerate(series.items()):
        figure.add_scatter(x=x, y=values, mode="lines", name=name, line={"width":2,"color":colors[index%len(colors)]})
    figure.update_layout(template="plotly_white",height=450,title=title,xaxis_title=x_title,yaxis_title=y_title,yaxis_type="log" if log_y else "linear",hovermode="x unified")
    return figure


def key_results(rows: list[tuple[str, object, str]]) -> None:
    frame = pd.DataFrame(rows, columns=["Key quantity", "Value", "Unit / meaning"])
    st.dataframe(frame, width="stretch", hide_index=True, height=min(360, 38 * (len(frame) + 1)))


def complete_data_panel(key: str, frame_factory, filename: str) -> None:
    state_key=f"show_complete_{key}"
    st.session_state.setdefault(state_key,False)
    label="Hide complete data" if st.session_state[state_key] else "More data / complete data"
    if st.button(label,key=f"toggle_{key}"):
        st.session_state[state_key]=not st.session_state[state_key]
    if st.session_state[state_key]:
        frame=frame_factory()
        st.dataframe(frame,width="stretch",hide_index=True,height=min(900,38*(len(frame)+1)))
        download_frame("Download complete CSV",frame,filename)


initialize()

st.markdown(
    """
    <section class="hero">
      <h1>Oscillation & Numerical Integration Lab</h1>
      <p>Analytical references, solver convergence, energy behavior, damping,
      resonance, beats, nonlinear periods, and external-data validation.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Physical configuration")
    load_config(st.file_uploader("Import configuration", type=["json"]))
    st.number_input("Mass m (kg)", min_value=0.001, max_value=1000.0, key="mass")
    st.number_input("Natural angular frequency ω₀ (rad/s)", min_value=0.001, max_value=1000.0, key="omega0")
    st.number_input("Damping coefficient γ (s⁻¹)", min_value=0.0, max_value=1000.0, key="gamma")
    st.number_input("Force amplitude F₀ (N)", min_value=0.0, max_value=1000.0, key="force_amplitude")
    st.number_input("Drive angular frequency Ω (rad/s)", min_value=0.0, max_value=1000.0, key="force_frequency")
    st.subheader("Initial state")
    c1,c2=st.columns(2)
    c1.number_input("u(0) (m)", key="u0")
    c2.number_input("v(0) (m/s)", key="v0")
    st.download_button("Download configuration",json.dumps(config(),indent=2).encode("utf-8"),file_name="oscillation_lab_config.json",mime="application/json",width="stretch")
    if st.button("Reset configuration",width="stretch"):
        for key,value in defaults().items(): st.session_state[key]=value
        for key in RESULT_KEYS: st.session_state[key]=None
        st.rerun()
    st.caption(f"Critical damping γc = {2*st.session_state.omega0:.6g} s⁻¹")
    st.caption(f"Natural period T₀ = {2*np.pi/st.session_state.omega0:.6g} s")
    st.caption(f"Platform version: {APP_VERSION}")

overview_tab, methods_tab, convergence_tab, damping_tab, resonance_tab, nonlinear_tab, external_tab, validation_tab = st.tabs(
    ["Overview","Method comparison","Convergence","Damping","Resonance & beats","Nonlinear pendulum","External data","Validation"]
)

with overview_tab:
    st.subheader("What this platform investigates")
    st.latex(r"m\ddot u+m\gamma\dot u+m\omega_0^2u=F_0\cos(\Omega t)")
    a,b,c,d=st.columns(4)
    a.metric("Exact reference","Free linear response")
    b.metric("Adaptive reference","DOP853")
    c.metric("Fixed-step methods","4")
    d.metric("Nonlinear benchmark","Elliptic integral")
    st.markdown("""
    <div class="note"><b>Interpretation rule:</b> a method is not reliable merely because its curve
    looks smooth. The platform separately examines state error, convergence order, energy drift,
    damping/forcing balance, and agreement with independent references.</div>
    """,unsafe_allow_html=True)
    st.write("The original notebook's fixed y-axis limits hid Euler divergence and clipped the nonlinear energy curve. This platform uses data-aware axes and normalized diagnostics instead.")

with methods_tab:
    st.subheader("Fixed-step methods versus exact and adaptive references")
    c1,c2=st.columns(2)
    c1.number_input("Duration (s)",min_value=0.01,max_value=1000.0,key="duration")
    c2.number_input("Requested timestep dt (s)",min_value=0.00001,max_value=1.0,step=0.005,format="%.5f",key="dt")
    if st.button("Run method comparison",type="primary"):
        st.session_state.comparison_result=method_comparison(initial_state(),params(),duration=st.session_state.duration,dt=st.session_state.dt)
    if st.session_state.comparison_result is not None:
        result=st.session_state.comparison_result
        reference_key="analytic" if "analytic" in result else "dop853"
        reference=np.asarray(result[reference_key]["state"])
        times=np.asarray(result[reference_key]["time"])
        displacement={name:np.asarray(data["state"])[:,0] for name,data in result.items()}
        st.plotly_chart(plot_lines(times,displacement,title="Displacement comparison without hidden clipping",x_title="Time t (s) — evolution variable",y_title="Displacement u (m)"),width="stretch")
        error_series={}
        energy_series={}
        metrics=[]
        current_params=params()
        initial_energy=float(linear_energy(initial_state()[None,:],current_params)[0])
        static_displacement=(current_params.force_amplitude/current_params.mass/current_params.omega0**2) if current_params.force_amplitude else 0.0
        forcing_energy_scale=0.5*current_params.mass*current_params.omega0**2*static_displacement**2
        common_energy_scale=max(abs(initial_energy),forcing_energy_scale,np.finfo(float).tiny)
        for name,data in result.items():
            states=np.asarray(data["state"])
            error=np.linalg.norm(states-reference,axis=1)
            error_series[name]=np.maximum(error,np.finfo(float).tiny)
            energy=linear_energy(states,current_params)
            energy_series[name]=energy/common_energy_scale
            metrics.append({"method":name,"final_state_error":error[-1],"max_state_error":error.max(),"max_relative_energy_change":np.max(np.abs((energy-energy[0])/common_energy_scale))})
        st.plotly_chart(plot_lines(times,error_series,title=f"State error relative to {reference_key}",x_title="Time t (s)",y_title="State-vector error",log_y=True),width="stretch")
        energy_title="Normalized mechanical energy" if st.session_state.gamma==0 and st.session_state.force_amplitude==0 else "Mechanical energy response (not a conservation test)"
        st.plotly_chart(plot_lines(times,energy_series,title=energy_title,x_title="Time t (s)",y_title="E(t) / shared energy scale"),width="stretch")
        metric_frame=pd.DataFrame(metrics)
        fixed_metrics=metric_frame[metric_frame.method.isin([m.value for m in Method])]
        best=fixed_metrics.loc[fixed_metrics.max_state_error.idxmin()]
        rk4_data=result[Method.RK4.value]
        balance=energy_balance_diagnostic(np.asarray(rk4_data["time"]),np.asarray(rk4_data["state"]),params())
        key_results([
            ("Best fixed-step method at this dt", str(best["method"]), "lowest maximum state error"),
            ("Best maximum state error", f"{float(best['max_state_error']):.10g}", "state-vector norm"),
            ("RK4 energy-balance residual", f"{float(balance['max_relative_balance_residual']):.10g}", "relative"),
            ("Reference", reference_key, "comparison trajectory"),
        ])
        complete_data_panel("methods",lambda: metric_frame,"method_comparison_metrics.csv")

with convergence_tab:
    st.subheader("Observed convergence order and timestep stability")
    c1,c2,c3,c4=st.columns(4)
    c1.selectbox("Method",list(METHOD_LABELS),key="convergence_method")
    c2.number_input("Minimum dt",min_value=1e-5,max_value=.5,format="%.5f",key="convergence_dt_min")
    c3.number_input("Maximum dt",min_value=2e-5,max_value=1.0,format="%.5f",key="convergence_dt_max")
    c4.slider("Scan points",4,12,key="convergence_points")
    invalid=st.session_state.convergence_dt_min>=st.session_state.convergence_dt_max or st.session_state.force_amplitude!=0
    if st.session_state.force_amplitude!=0: st.warning("Set force amplitude to zero because the convergence page uses the exact free-response reference.")
    if st.button("Run timestep scan",type="primary",disabled=invalid):
        dt_values=np.geomspace(st.session_state.convergence_dt_max,st.session_state.convergence_dt_min,st.session_state.convergence_points)
        st.session_state.convergence_result=convergence_scan(dt_values,initial_state(),params(),duration=st.session_state.duration,method=METHOD_LABELS[st.session_state.convergence_method])
    if st.session_state.convergence_result is not None:
        result=st.session_state.convergence_result
        fig=go.Figure()
        fig.add_scatter(x=result["actual_dt"],y=result["maximum_state_error"],mode="lines+markers",name="Maximum trajectory error")
        fig.add_scatter(x=result["actual_dt"],y=result["final_state_error"],mode="lines+markers",name="Final-state error",opacity=.55)
        fig.update_layout(template="plotly_white",height=470,title="Error versus actual timestep",xaxis_type="log",yaxis_type="log",xaxis_title="Actual dt (s) — scanned independent variable",yaxis_title="State-vector error")
        st.plotly_chart(fig,width="stretch")
        key_results([
            ("Observed convergence order", f"{float(result['observed_order']):.10g}", "slope on log-log error curve"),
            ("Finest actual dt", f"{float(np.min(result['actual_dt'])):.10g}", "s"),
            ("Smallest maximum state error", f"{float(np.min(result['maximum_state_error'])):.10g}", "state-vector norm"),
            ("Energy drift diagnostic", "available" if st.session_state.gamma==0 else "not interpreted under damping", "conservative-only"),
        ])
        complete_data_panel("convergence",lambda: pd.DataFrame(result),"timestep_convergence.csv")

with damping_tab:
    st.subheader("Underdamped, critical, and overdamped response")
    if st.button("Generate damping regimes",type="primary"):
        critical=2*st.session_state.omega0
        gamma_values={"Underdamped":.1*critical,"Critical":critical,"Overdamped":2.4*critical}
        times=np.linspace(0,max(st.session_state.duration,5*2*np.pi/st.session_state.omega0),1500)
        st.session_state.damping_result={name:{"gamma":gamma,"time":times,"state":analytic_free_response(times,initial_state(),params(gamma=gamma,force_amplitude=0.0))} for name,gamma in gamma_values.items()}
    if st.session_state.damping_result is not None:
        result=st.session_state.damping_result
        times=next(iter(result.values()))["time"]
        st.plotly_chart(plot_lines(times,{name:data["state"][:,0] for name,data in result.items()},title="Damping regimes with identical initial conditions",x_title="Time t (s)",y_title="Displacement u (m)"),width="stretch")
        key_results([(name,f"{float(data['gamma']):.10g}","γ (s⁻¹)") for name,data in result.items()])
        complete_data_panel("damping",lambda: pd.DataFrame({"time":times,**{name:data["state"][:,0] for name,data in result.items()}}),"damping_regimes.csv")

with resonance_tab:
    st.subheader("Driven resonance scan and beat phenomenon")
    resonance_panel,beat_panel=st.tabs(["Resonance scan","Single beat analysis"])
    with resonance_panel:
        c1,c2,c3=st.columns(3)
        c1.number_input("Resonance damping γ",min_value=.001,max_value=100.0,key="resonance_gamma")
        c2.number_input("Driving force F₀",min_value=.001,max_value=1000.0,key="resonance_force")
        c3.slider("Frequency scan points",20,250,key="resonance_points")
        c1,c2=st.columns(2)
        c1.number_input("Minimum Ω/ω₀",min_value=.05,max_value=3.0,key="resonance_ratio_min")
        c2.number_input("Maximum Ω/ω₀",min_value=.1,max_value=5.0,key="resonance_ratio_max")
        if st.button("Run resonance frequency scan",type="primary",disabled=st.session_state.resonance_ratio_min>=st.session_state.resonance_ratio_max):
            bar,update=progress("Resonance integration")
            frequencies=np.linspace(st.session_state.resonance_ratio_min*st.session_state.omega0,st.session_state.resonance_ratio_max*st.session_state.omega0,st.session_state.resonance_points)
            p=params(gamma=st.session_state.resonance_gamma,force_amplitude=st.session_state.resonance_force)
            st.session_state.resonance_result=resonance_scan(frequencies,p,progress_callback=update)
            bar.progress(1.0,text="Resonance scan complete")
        if st.session_state.resonance_result is not None:
            result=st.session_state.resonance_result
            fig=go.Figure()
            fig.add_scatter(x=result["frequency_ratio"],y=result["numerical_amplitude"],name="Numerical steady amplitude")
            fig.add_scatter(x=result["frequency_ratio"],y=result["analytic_amplitude"],name="Analytical steady amplitude",line={"dash":"dash"})
            fig.update_layout(template="plotly_white",height=480,title="Frequency response",xaxis_title="Drive-frequency ratio Ω/ω₀ — scanned independent variable",yaxis_title="Steady-state amplitude (m)")
            st.plotly_chart(fig,width="stretch")
            peak_index=int(np.argmax(result["numerical_amplitude"]))
            transient=float(result["estimated_remaining_free_transient"][0])
            if transient>0.01:
                st.warning("The estimated surviving free transient exceeds 1%; increase damping or integration duration before treating the numerical peak as fully settled.")
            key_results([
                ("Numerical peak Ω/ω₀", f"{float(result['frequency_ratio'][peak_index]):.10g}", "frequency ratio"),
                ("Numerical peak amplitude", f"{float(result['numerical_amplitude'][peak_index]):.10g}", "m"),
                ("Maximum numerical/analytic relative difference", f"{float(np.max(result['relative_difference'])):.10g}", "relative"),
                ("Estimated remaining free transient", f"{transient:.10g}", "amplitude fraction"),
            ])
            complete_data_panel("resonance",lambda: pd.DataFrame(result),"resonance_frequency_scan.csv")
    with beat_panel:
        st.write("For the clearest conservative beat pattern, use zero damping and a drive frequency close to ω₀.")
        if st.button("Run beat analysis",type="primary"):
            p=params(gamma=0.0,force_amplitude=max(st.session_state.resonance_force,.001))
            result=simulate_fixed(initial_state(),p,duration=max(st.session_state.duration,30.0),dt=min(st.session_state.dt,.01),method=Method.RK4)
            delta=abs(p.omega0-p.force_frequency)
            st.session_state.beat_result={**result,"beat_frequency":delta/(2*np.pi),"beat_period":np.inf if delta==0 else 2*np.pi/delta}
        if st.session_state.beat_result is not None:
            result=st.session_state.beat_result
            st.plotly_chart(plot_lines(result["time"],{"Driven response":result["state"][:,0]},title="Near-resonant beat response",x_title="Time t (s)",y_title="Displacement u (m)"),width="stretch")
            key_results([
                ("Beat frequency", f"{float(result['beat_frequency']):.10g}", "Hz"),
                ("Beat period", f"{float(result['beat_period']):.10g}", "s"),
                ("Drive frequency", f"{float(st.session_state.force_frequency):.10g}", "rad/s"),
            ])
            complete_data_panel("beat",lambda: pd.DataFrame({"time":result["time"],"displacement":result["state"][:,0],"velocity":result["state"][:,1]}),"beat_response.csv")

with nonlinear_tab:
    st.subheader("Nonlinear pendulum period and phase-space analysis")
    c1,c2=st.columns(2)
    c1.number_input("Maximum initial angle (rad)",min_value=.05,max_value=3.1,key="nonlinear_amp_max")
    c2.slider("Amplitude scan points",8,80,key="nonlinear_points")
    if st.button("Run nonlinear amplitude scan",type="primary"):
        bar,update=progress("Nonlinear period scan")
        amplitudes=np.linspace(.02,st.session_state.nonlinear_amp_max,st.session_state.nonlinear_points)
        st.session_state.nonlinear_result=nonlinear_period_scan(amplitudes,st.session_state.omega0,progress_callback=update)
        bar.progress(1.0,text="Nonlinear scan complete")
    if st.session_state.nonlinear_result is not None:
        result=st.session_state.nonlinear_result
        fig=go.Figure()
        fig.add_scatter(x=result["initial_amplitude"],y=result["exact_nonlinear_period"],name="Exact nonlinear period")
        fig.add_scatter(x=result["initial_amplitude"],y=result["numerical_period"],name="RK4 measured period",mode="markers")
        fig.add_scatter(x=result["initial_amplitude"],y=result["small_angle_period"],name="Small-angle period",line={"dash":"dash"})
        fig.update_layout(template="plotly_white",height=480,title="Period growth beyond the small-angle approximation",xaxis_title="Initial angle θ₀ (rad) — scanned independent variable",yaxis_title="Period (s)")
        st.plotly_chart(fig,width="stretch")
        period_error=np.abs(result["numerical_period"]-result["exact_nonlinear_period"])
        key_results([
            ("Amplitude points", len(result["initial_amplitude"]), "count"),
            ("Largest period ratio T/T₀", f"{float(np.max(result['period_ratio'])):.10g}", "dimensionless"),
            ("Maximum RK4 vs elliptic-integral period error", f"{float(np.nanmax(period_error)):.10g}", "s"),
        ])
        complete_data_panel("nonlinear",lambda: pd.DataFrame(result),"nonlinear_period_scan.csv")
    st.number_input("Single nonlinear θ₀ (rad)",min_value=.001,max_value=3.13,key="single_theta0")
    single=simulate_fixed(np.asarray([st.session_state.single_theta0,0.0]),params(gamma=0.0,force_amplitude=0.0),duration=3*2*np.pi/st.session_state.omega0,dt=min(st.session_state.dt,.002),method=Method.RK4,nonlinear=True)
    phase=go.Figure(go.Scatter(x=single["state"][:,0],y=single["state"][:,1],mode="lines",name="Nonlinear orbit"))
    phase.update_layout(template="plotly_white",height=430,title="Single-point nonlinear phase portrait",xaxis_title="Angle θ (rad)",yaxis_title="Angular velocity (rad/s)")
    st.plotly_chart(phase,width="stretch")

with external_tab:
    st.subheader("Compare an external measurement or simulator CSV")
    st.write("Upload columns named `time` and `displacement`; an optional `velocity` column enables full state comparison. Units must be seconds, metres, and metres per second.")
    uploaded=st.file_uploader("Upload external CSV",type=["csv"],key="external_csv")
    if uploaded is not None:
        try:
            measured=pd.read_csv(uploaded)
            required={"time","displacement"}
            if not required.issubset(measured.columns): raise ValueError("CSV must contain time and displacement columns")
            measured=measured.sort_values("time").dropna(subset=["time","displacement"])
            if len(measured)<3 or measured.time.iloc[0]<0 or not np.all(np.diff(measured.time)>0): raise ValueError("time must contain at least three strictly increasing non-negative values")
            reference=simulate_dop853(initial_state(),params(),duration=float(measured.time.iloc[-1]),samples=max(1000,len(measured)*4))
            predicted=np.interp(measured.time,reference["time"],reference["state"][:,0])
            residual=measured.displacement.to_numpy()-predicted
            comparison=measured.copy();comparison["model_displacement"]=predicted;comparison["displacement_residual"]=residual
            rmse=float(np.sqrt(np.mean(residual**2)))
            velocity_rmse=None
            if "velocity" in measured.columns:
                if measured["velocity"].isna().any():
                    raise ValueError("velocity contains missing values")
                model_velocity=np.interp(measured.time,reference["time"],reference["state"][:,1])
                velocity_residual=measured.velocity.to_numpy()-model_velocity
                comparison["model_velocity"]=model_velocity
                comparison["velocity_residual"]=velocity_residual
                velocity_rmse=float(np.sqrt(np.mean(velocity_residual**2)))
            fig=go.Figure()
            fig.add_scatter(x=comparison.time,y=comparison.displacement,name="External data",mode="markers")
            fig.add_scatter(x=comparison.time,y=comparison.model_displacement,name="DOP853 model")
            fig.update_layout(template="plotly_white",height=470,title="External data versus model",xaxis_title="Time (s)",yaxis_title="Displacement (m)")
            st.plotly_chart(fig,width="stretch")
            rows=[
                ("Displacement RMSE", f"{rmse:.10g}", "m"),
                ("External samples", len(comparison), "count"),
                ("Final external time", f"{float(comparison.time.iloc[-1]):.10g}", "s"),
            ]
            if velocity_rmse is not None:
                rows.insert(1,("Velocity RMSE",f"{velocity_rmse:.10g}","m/s"))
            key_results(rows)
            complete_data_panel("external",lambda: comparison,"external_model_comparison.csv")
        except (ValueError,pd.errors.ParserError) as exc:
            st.error(f"Could not interpret CSV: {exc}")

with validation_tab:
    st.subheader("Built-in physical and numerical compliance checks")
    st.markdown("""
    - exact underdamped, critical, and overdamped free responses are implemented separately;
    - DOP853 provides an independent adaptive numerical reference;
    - observed method order is measured instead of assumed;
    - conservative energy drift is reported without fixed plot clipping;
    - resonance plots place drive frequency on the horizontal scan axis;
    - nonlinear periods are checked against the elliptic-integral result;
    - all model parameters carry explicit physical units.
    """)
    if st.button("Run compliance suite",type="primary"):
        p=params(gamma=0.0,force_amplitude=0.0)
        state=initial_state()
        times=np.linspace(0,2*2*np.pi/p.omega0,1001)
        exact=analytic_free_response(times,state,p)
        adaptive=simulate_dop853(state,p,duration=float(times[-1]),samples=len(times))
        adaptive_error=float(np.max(np.linalg.norm(adaptive["state"]-exact,axis=1)))
        scan=convergence_scan(np.asarray([.04,.02,.01,.005])*(2*np.pi/p.omega0),state,p,duration=2*2*np.pi/p.omega0,method=Method.RK4)
        nonlinear=nonlinear_period_scan(np.asarray([.2,1.0]),p.omega0)
        nonlinear_error=float(np.max(np.abs(nonlinear["numerical_period"]-nonlinear["exact_nonlinear_period"])))
        driven_params=OscillatorParams(mass=p.mass,omega0=p.omega0,gamma=0.4,force_amplitude=0.6,force_frequency=0.92*p.omega0)
        driven=simulate_dop853(state,driven_params,duration=5.0,samples=2001)
        balance=energy_balance_diagnostic(driven["time"],driven["state"],driven_params)
        report={"DOP853 maximum state error":adaptive_error,"RK4 observed order":float(scan["observed_order"]),"nonlinear period maximum error":nonlinear_error,"forced/damped energy-balance residual":float(balance["max_relative_balance_residual"])}
        report["passed"]=bool(adaptive_error<1e-8 and 3.7<report["RK4 observed order"]<4.3 and nonlinear_error<2e-3 and report["forced/damped energy-balance residual"]<1e-4)
        st.session_state.validation_result=report
    if st.session_state.validation_result is not None:
        report=st.session_state.validation_result
        if report["passed"]: st.success("Compliance suite passed.")
        else: st.error("One or more compliance checks failed.")
        key_results([
            ("DOP853 maximum state error", f"{float(report['DOP853 maximum state error']):.10g}", "state-vector norm"),
            ("RK4 observed order", f"{float(report['RK4 observed order']):.10g}", "expected ≈ 4"),
            ("Nonlinear period maximum error", f"{float(report['nonlinear period maximum error']):.10g}", "s"),
            ("Forced/damped energy-balance residual", f"{float(report['forced/damped energy-balance residual']):.10g}", "relative"),
            ("Compliance status", "PASS" if report["passed"] else "FAIL", "built-in checks"),
        ])
        complete_data_panel("validation",lambda: pd.DataFrame([report]),"validation_results.csv")
