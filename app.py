"""
CDS Basket Correlation  |  run: streamlit run copula_app.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(
    page_title="Basket CDS — Correlation Impact",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Basket CDS — Correlation Impact")

# ── Reset ────────────────────────────────────────────────────────────────────
if "rc" not in st.session_state:
    st.session_state.rc = 0
rc = st.session_state.rc

if st.button("↻ Reset to Defaults", key="reset"):
    st.session_state.rc = rc + 1
    st.rerun()

# ── All parameters on ONE line ───────────────────────────────────────────────
st.markdown("#### Parameters")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
n_names  = c1.number_input("Names",         min_value=2,     max_value=10,   value=5,     step=1,     key=f"n_{rc}")
lam      = c2.number_input("Hazard Rate λ", min_value=0.001, max_value=0.50, value=0.030, step=0.005, format="%.4f", key=f"lam_{rc}")
recovery = c3.number_input("Recovery",      min_value=0.0,   max_value=0.80, value=0.40,  step=0.05,  format="%.2f", key=f"rec_{rc}")
maturity = c4.number_input("Maturity (y)",  min_value=1,     max_value=10,   value=5,     step=1,     key=f"mat_{rc}")
rho      = c5.number_input("ρ (live)",      min_value=0.0,   max_value=0.99, value=0.30,  step=0.05,  format="%.2f", key=f"rho_{rc}")
rho_step = c6.number_input("ρ step",        min_value=0.05,  max_value=0.20, value=0.05,  step=0.05,  format="%.2f", key=f"step_{rc}")
n_sims   = c7.selectbox("Simulations", [10_000, 50_000, 100_000], index=1,
                         format_func=lambda x: f"{x:,}", key=f"sims_{rc}")

st.markdown("---")

# ── Simulation ───────────────────────────────────────────────────────────────
PALETTE = [
    "#636EFA","#EF553B","#00CC96","#AB63FA",
    "#FFA15A","#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52",
]

@st.cache_data(show_spinner="Simulating…")
def run(n_sims_v, n_names_v, lam_v, recovery_v, maturity_v, rho_v, rho_step_v):
    rng  = np.random.default_rng(42)
    lam  = lam_v          # fundamental input: hazard rate
    lgd  = 1 - recovery_v  # recovery only affects severity, not frequency

    def _tau(rho_val):
        M   = rng.standard_normal((n_sims_v, 1))
        Z   = rng.standard_normal((n_sims_v, n_names_v))
        X   = np.sqrt(rho_val) * M + np.sqrt(max(1 - rho_val, 0)) * Z
        U   = norm.cdf(X)
        return -np.log(np.clip(U, 1e-10, 1 - 1e-10)) / lam   # (sims, names)

    # ── Sweep: Nth-to-default spreads vs rho ────────────────────────────────
    rho_grid = np.round(np.arange(0.0, 0.96, rho_step_v), 4)
    nth_spreads = {n: [] for n in range(1, n_names_v + 1)}

    for rv in rho_grid:
        tau       = _tau(rv)
        sorted_t  = np.sort(tau, axis=1)
        for nth in range(1, n_names_v + 1):
            tau_nth = sorted_t[:, nth - 1]
            hit     = tau_nth < maturity_v
            if hit.sum() == 0:
                nth_spreads[nth].append(0.0)
                continue
            prot_pv = (lgd * np.exp(-0.03 * tau_nth) * hit).mean()
            pv01    = maturity_v * (1 - hit.mean() * 0.5)
            nth_spreads[nth].append(prot_pv / pv01 * 10_000 if pv01 > 0 else 0.0)

    # ── Live distribution at current rho ────────────────────────────────────
    tau_live   = _tau(rho_v)
    n_defaults = (tau_live < maturity_v).sum(axis=1)

    return rho_grid, nth_spreads, n_defaults, lam


rho_grid, nth_spreads, n_defaults, lam_out = run(
    n_sims, n_names, lam, recovery, maturity, rho, rho_step
)

# ── Key metrics (one row) ────────────────────────────────────────────────────
pd_indiv    = 1 - np.exp(-lam * maturity)   # depends only on λ and T
spread_bps  = lam * (1 - recovery) * 10_000  # derived from λ and LGD

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Individual PD",      f"{pd_indiv:.2%}")
m2.metric("Fair Spread",        f"{spread_bps:.1f} bps")
m3.metric("Expected Defaults",  f"{n_defaults.mean():.2f}")
m4.metric("P(≥ 1 default)",     f"{(n_defaults >= 1).mean():.2%}")
m5.metric("P(all default)",     f"{(n_defaults == n_names).mean():.2%}")

st.markdown("---")

# ── Charts: big left + bar chart right ───────────────────────────────────────
st.markdown("#### Results")
left, right = st.columns([3, 1])

# Left: Nth-to-default spread vs ρ
with left:
    fig_main = go.Figure()
    for nth in range(1, n_names + 1):
        suffix = {1:"st", 2:"nd", 3:"rd"}.get(nth, "th")
        fig_main.add_trace(go.Scatter(
            x=rho_grid, y=nth_spreads[nth],
            mode="lines+markers",
            name=f"{nth}{suffix}-to-default",
            line=dict(color=PALETTE[nth - 1], width=2.5),
            marker=dict(size=5),
        ))
    fig_main.add_vline(
        x=rho, line_dash="dot", line_color="rgba(255,255,255,0.5)",
        annotation_text=f"ρ = {rho:.2f}",
        annotation_font_color="white",
        annotation_position="top right",
    )
    fig_main.update_layout(
        title="Nth-to-Default Fair Spread vs Correlation ρ",
        xaxis_title="Correlation ρ",
        yaxis_title="Fair Spread (bps)",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.22, x=0),
        height=480,
        margin=dict(t=50, b=80),
    )
    st.plotly_chart(fig_main, width="stretch")

# Right: Default count distribution
with right:
    max_k    = n_names
    all_k    = np.arange(0, max_k + 1)
    freq_map = {k: 0 for k in all_k}
    vals, cnts = np.unique(n_defaults, return_counts=True)
    for v, c in zip(vals, cnts):
        freq_map[int(v)] = c
    probs = np.array([freq_map[k] / n_sims for k in all_k])

    fig_bar = go.Figure(go.Bar(
        x=all_k,
        y=probs,
        marker_color=[PALETTE[k % len(PALETTE)] for k in all_k],
        text=[f"{p:.1%}" if p >= 0.005 else "" for p in probs],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig_bar.update_layout(
        title=f"Default Count at ρ={rho:.2f}",
        xaxis_title="# Defaults",
        yaxis_title="Probability",
        xaxis=dict(tickmode="array", tickvals=list(all_k)),
        yaxis=dict(tickformat=".0%", range=[0, max(probs) * 1.25 + 0.01]),
        template="plotly_dark",
        showlegend=False,
        height=480,
        margin=dict(t=50, b=80),
    )
    st.plotly_chart(fig_bar, width="stretch")
