"""Streamlit dashboard: Game-Theoretic Bargaining for Fair and Efficient
Allocation of Computational Resources in Cloud Computing.

Run with:  streamlit run app.py

This file contains ONLY user-interface code. All mathematics lives in src/:
    src/algorithms.py   Equal, Priority-Based, Nash Bargaining
    src/game_theory.py  utilities, disagreement point, Nash product optimization
    src/metrics.py      utilization, satisfaction, Jain's fairness index
    src/validation.py   input checks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import charts
from src.algorithms import run_all_algorithms
from src.experiments import (
    PRESETS,
    capacity_sweep,
    disagreement_sweep,
    preset_scenario,
    preset_users_frame,
    weight_sweep,
)
from src.export import scenario_frame, to_csv_bytes
from src.game_theory import bargaining_users, disagreement_point, user_utilities
from src.metrics import all_user_tables, comparison_table, user_table
from src.models import ALGORITHMS, NASH, USER_COLUMNS, Scenario, scenario_from_frame
from src.validation import validate_scenario

DEFAULT_PRESET = "Balanced baseline"
MAX_UI_USERS = 12

st.set_page_config(
    page_title="Nash Bargaining Cloud Allocator",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
    .hero {padding: 1.2rem 1.5rem; border-radius: 14px; margin-bottom: 1rem;
           background: linear-gradient(120deg, #312E81 0%, #4F46E5 55%, #0EA5E9 100%); color: #fff;}
    .hero h1 {font-size: 1.65rem; margin: 0 0 .35rem 0; color: #fff; line-height: 1.25;}
    .hero p {margin: 0; opacity: .92; font-size: .97rem;}
    .algo-card {border-radius: 12px; padding: .8rem 1rem .2rem 1rem; margin-bottom: .6rem;
                border: 1px solid rgba(127,127,127,.25); border-top: 5px solid var(--accent);}
    .algo-card h4 {margin: 0 0 .2rem 0; font-size: 1.02rem;}
    .algo-card .tag {font-size: .78rem; opacity: .75;}
    div[data-testid="stMetricValue"] {font-size: 1.45rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _apply_preset(name: str) -> None:
    preset = PRESETS[name]
    st.session_state.users_base = preset_users_frame(name)
    st.session_state.users_current = st.session_state.users_base.copy()
    st.session_state.cpu_cap = float(preset["cpu"])
    st.session_state.ram_cap = float(preset["ram"])
    st.session_state.delta = float(preset["delta"])
    st.session_state.n_users = len(st.session_state.users_base)
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.session_state.loaded_preset = name


if "users_base" not in st.session_state:
    _apply_preset(DEFAULT_PRESET)


def _resize_users() -> None:
    """Called when the 'Number of users' input changes; keeps existing rows."""
    n = int(st.session_state.n_users)
    current = st.session_state.users_current.reset_index(drop=True)
    if n < len(current):
        current = current.iloc[:n].copy()
    else:
        taken = set(current["User"].astype(str))
        extra, k = [], len(current) + 1
        while len(current) + len(extra) < n:
            name = f"User {k}"
            if name not in taken:
                extra.append({"User": name, USER_COLUMNS[1]: 4.0, USER_COLUMNS[2]: 16.0,
                              USER_COLUMNS[3]: 1.0, USER_COLUMNS[4]: 1.0})
            k += 1
        current = pd.concat([current, pd.DataFrame(extra)], ignore_index=True)
    st.session_state.users_base = current
    st.session_state.users_current = current.copy()
    st.session_state.editor_version += 1


def _load_preset_from_sidebar() -> None:
    _apply_preset(st.session_state.preset_choice)


def _load_preset_from_lab() -> None:
    _apply_preset(st.session_state.lab_choice)


# ---------------------------------------------------------------------------
# Sidebar: system configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System configuration")
    st.number_input("CPU capacity (cores)", min_value=0.0, max_value=100000.0, step=1.0,
                    key="cpu_cap", help="Total CPU cores available in the cluster.")
    st.number_input("RAM capacity (GB)", min_value=0.0, max_value=1000000.0, step=4.0,
                    key="ram_cap", help="Total memory available in the cluster.")
    st.number_input("Number of users", min_value=1, max_value=MAX_UI_USERS, step=1,
                    key="n_users", on_change=_resize_users,
                    help="Rows are added or removed in the user table.")
    st.slider("Disagreement fraction δ", min_value=0.0, max_value=0.9, step=0.05, key="delta",
              help="Fraction of each resource reserved as the fallback share if bargaining fails. "
                   "δ = 0 → disagreement point D = 0.")
    st.divider()
    st.subheader("📂 Scenario presets")
    st.selectbox("Preset", list(PRESETS), key="preset_choice",
                 index=list(PRESETS).index(DEFAULT_PRESET))
    st.caption(PRESETS[st.session_state.preset_choice]["description"])
    c1, c2 = st.columns(2)
    c1.button("Load preset", on_click=_load_preset_from_sidebar, type="primary", width="stretch")
    c2.button("Reset", on_click=_apply_preset, args=(DEFAULT_PRESET,), width="stretch",
              help="Restore the default 'Balanced baseline' scenario.")
    st.divider()
    st.caption("Algorithms: Equal · Priority-Based · Weighted Nash Bargaining (SciPy SLSQP / trust-constr)")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>☁️ Game-Theoretic Bargaining for Fair &amp; Efficient Cloud Resource Allocation</h1>
      <p>Compare Equal Allocation, Priority-Based Allocation and the weighted Nash Bargaining
      Solution for sharing CPU cores and RAM among competing cloud users.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_dash, tab_compare, tab_lab, tab_export, tab_learn = st.tabs(
    ["📊 Dashboard", "📈 Comparison", "🧪 Experiment Lab", "📥 Export", "📘 Viva Guide"]
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_pct(v) -> str:
    return f"{v * 100:.1f}%" if v is not None and np.isfinite(v) else "N/A"


def fmt_num(v, digits=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if v == -np.inf:
        return "−∞"
    return f"{v:.{digits}f}"


def styled_user_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=["Algorithm"]).copy()
    for col in out.columns:
        if "satisfaction" in col:
            out[col] = out[col].map(fmt_pct)
    for col in ["Utility", "Disagreement utility", "Gain over disagreement"]:
        out[col] = out[col].map(lambda v: fmt_num(v, 4))
    return out


def styled_comparison(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if "utilization" in col or "demand satisfaction" in col:
            out[col] = out[col].map(fmt_pct)
        elif col in ("Mean utility", "Min utility", "Max utility", "Total utility",
                     "Jain's fairness index", "Weighted log Nash product"):
            out[col] = out[col].map(lambda v: fmt_num(v, 4))
        elif col.endswith(" idle"):
            out[col] = out[col].map(lambda v: fmt_num(v, 2))
        elif col == "Pareto efficient":
            out[col] = out[col].map(lambda v: "N/A" if v is None else ("Yes" if v else "No"))
    return out


# ---------------------------------------------------------------------------
# Cached experiment runners (keyed by plain tuples so caching is reliable)
# ---------------------------------------------------------------------------
def scenario_key(s: Scenario) -> tuple:
    return (tuple(s.capacities), tuple(map(tuple, s.demands)), tuple(s.priorities),
            tuple(s.weights), tuple(s.user_names), s.disagreement_fraction)


def scenario_from_key(key: tuple) -> Scenario:
    caps, dem, pri, wts, names, delta = key
    return Scenario(capacities=list(caps), demands=[list(r) for r in dem], priorities=list(pri),
                    weights=list(wts), user_names=list(names), disagreement_fraction=delta)


@st.cache_data(show_spinner=False)
def cached_capacity_sweep(key: tuple, lo: float, hi: float, points: int) -> pd.DataFrame:
    return capacity_sweep(scenario_from_key(key), np.linspace(lo, hi, points))


@st.cache_data(show_spinner=False)
def cached_weight_sweep(key: tuple, user_index: int, hi: float, points: int) -> pd.DataFrame:
    return weight_sweep(scenario_from_key(key), user_index, np.linspace(0.25, hi, points))


@st.cache_data(show_spinner=False)
def cached_disagreement_sweep(key: tuple, points: int) -> pd.DataFrame:
    return disagreement_sweep(scenario_from_key(key), np.linspace(0.0, 0.9, points))


def show_results_overview(scenario: Scenario, results: dict, table: pd.DataFrame) -> None:
    """Three summary cards, one per algorithm."""
    cols = st.columns(3)
    for col, name in zip(cols, ALGORITHMS):
        res = results[name]
        row = table[table["Algorithm"] == name].iloc[0]
        color = charts.ALGORITHM_COLORS[name]
        with col:
            st.markdown(
                f'<div class="algo-card" style="--accent:{color}"><h4>{name}</h4>'
                f'<span class="tag">{"✅ computed" if res.success else "❌ FAILED"}</span></div>',
                unsafe_allow_html=True,
            )
            if not res.success:
                st.error(res.message)
                continue
            m1, m2 = st.columns(2)
            m1.metric("Jain's fairness", fmt_num(row["Jain's fairness index"], 3))
            m2.metric("Mean utility", fmt_num(row["Mean utility"], 3))
            m3, m4 = st.columns(2)
            m3.metric("CPU used", fmt_pct(row["CPU utilization"]))
            m4.metric("RAM used", fmt_pct(row["RAM utilization"]))
            m5, m6 = st.columns(2)
            m5.metric("Min utility", fmt_num(row["Min utility"], 3))
            m6.metric("Pareto efficient", "Yes" if row["Pareto efficient"] else "No")


# ---------------------------------------------------------------------------
# DASHBOARD TAB: inputs + headline results
# ---------------------------------------------------------------------------
with tab_dash:
    st.subheader("1 · Users and their demands")
    st.caption(
        "Edit any cell. **Priority** is used only by Priority-Based allocation (higher = served first). "
        "**Bargaining weight** is used only by Nash Bargaining (higher = more bargaining power)."
    )
    edited = st.data_editor(
        st.session_state.users_base,
        key=f"user_editor_{st.session_state.editor_version}",
        num_rows="fixed",
        hide_index=True,

        column_config={
            "User": st.column_config.TextColumn("User", required=True, max_chars=40),
            USER_COLUMNS[1]: st.column_config.NumberColumn(USER_COLUMNS[1], min_value=0.0, step=0.5, format="%.2f"),
            USER_COLUMNS[2]: st.column_config.NumberColumn(USER_COLUMNS[2], min_value=0.0, step=1.0, format="%.2f"),
            USER_COLUMNS[3]: st.column_config.NumberColumn(USER_COLUMNS[3], min_value=0.0, step=1.0, format="%.2f"),
            USER_COLUMNS[4]: st.column_config.NumberColumn(USER_COLUMNS[4], min_value=0.0, step=0.5, format="%.2f"),
        },
    )
    st.session_state.users_current = edited.copy()

scenario = scenario_from_frame(
    edited, st.session_state.cpu_cap, st.session_state.ram_cap, st.session_state.delta
)
report = validate_scenario(scenario)
results = run_all_algorithms(scenario) if report.ok else None
table = comparison_table(scenario, results) if results else None
users_long = all_user_tables(scenario, results) if results else None

with tab_dash:
    if report.errors:
        st.error("**Please fix these inputs before results can be computed:**\n\n"
                 + "\n".join(f"- {e}" for e in report.errors))
    for w in report.warnings:
        st.warning(w)

    if results:
        total = scenario.demands.sum(axis=0)
        st.info(
            f"Total demand: **{total[0]:.2f} cores** CPU ({total[0] / scenario.capacities[0]:.0%} of capacity) · "
            f"**{total[1]:.2f} GB** RAM ({total[1] / scenario.capacities[1]:.0%} of capacity) · "
            f"Disagreement fraction δ = **{scenario.disagreement_fraction:.2f}**"
        )
        st.subheader("2 · Results at a glance")
        show_results_overview(scenario, results, table)

        nbs = results[NASH]
        st.subheader("3 · Nash Bargaining solver status")
        if nbs.success:
            st.success(nbs.message)
        else:
            st.error(nbs.message + "  \nNo Nash Bargaining numbers are shown anywhere until this is fixed.")
        with st.expander("Solver details, disagreement point and verification"):
            attempts = pd.DataFrame(nbs.details.get("attempts", []))
            if not attempts.empty:
                attempts["issues"] = attempts.get("issues", pd.Series([[]] * len(attempts))).map(
                    lambda v: "none" if not v else "; ".join(v) if isinstance(v, list) else str(v)
                )
                st.markdown("**Optimization attempts**")
                st.dataframe(attempts, hide_index=True)
            D = disagreement_point(scenario)
            mask = bargaining_users(scenario)
            st.markdown("**Disagreement point and bargaining set**")
            st.dataframe(pd.DataFrame({
                "User": scenario.user_names,
                "Fallback CPU (cores)": nbs.details["disagreement_allocation"][:, 0],
                "Fallback RAM (GB)": nbs.details["disagreement_allocation"][:, 1],
                "Disagreement utility Dᵢ": D,
                "Takes part in bargaining": ["Yes" if m else "No" for m in mask],
            }), hide_index=True)
            st.caption(
                "A solution is marked successful only if the solver converged AND the allocation "
                "respects capacity, never exceeds demand, gives every bargaining user more than Dᵢ, "
                "and leaves no capacity idle while someone still wants it (Pareto check)."
            )

        st.subheader("4 · Allocation details per algorithm")
        for sub, name in zip(st.tabs(list(ALGORITHMS)), ALGORITHMS):
            with sub:
                res = results[name]
                st.caption(res.message)
                if res.success:
                    st.dataframe(styled_user_table(user_table(scenario, res)),
                                 hide_index=True)


def need_results(container) -> bool:
    if results is None:
        container.info("Results appear here once the inputs on the **Dashboard** tab are valid.")
        return False
    return True


# ---------------------------------------------------------------------------
# COMPARISON TAB
# ---------------------------------------------------------------------------
with tab_compare:
    if need_results(st):
        st.subheader("Algorithm comparison")
        if not results[NASH].success:
            st.error("Nash Bargaining failed for this input, so it is left out of the charts below.")
        st.dataframe(styled_comparison(table), hide_index=True)
        st.plotly_chart(charts.allocation_chart(scenario, results))
        st.plotly_chart(charts.utility_chart(scenario, results))
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.fairness_chart(table))
        c2.plotly_chart(charts.nash_product_chart(table))
        c3, c4 = st.columns(2)
        c3.plotly_chart(charts.utilization_chart(table, scenario))
        c4.plotly_chart(charts.demand_satisfaction_chart(table, scenario))
        with st.expander("How to read these charts"):
            st.markdown(
                "- **Allocation per user** – bars are what each algorithm gave; the red line is the demand.\n"
                "- **Utility** – 1.0 means the user got everything it asked for.\n"
                "- **Jain's index** – 1.0 means all users are equally satisfied; 1/n means one user gets everything.\n"
                "- **Nash product** – the quantity NBS maximizes. NBS should always be the highest (closest to 0).\n"
                "- **Utilization** – below 100% while demand is unmet means wasted (idle) capacity."
            )


# ---------------------------------------------------------------------------
# EXPERIMENT LAB
# ---------------------------------------------------------------------------
with tab_lab:
    st.subheader("🧪 Experiment Lab")
    st.caption("Every experiment re-runs the real algorithms; nothing is pre-computed.")
    options = ["Current dashboard scenario"] + list(PRESETS)
    st.selectbox("Scenario to experiment on", options, key="lab_choice")
    lab_name = st.session_state.lab_choice

    if lab_name == "Current dashboard scenario":
        lab_scenario = scenario if report.ok else None
        if lab_scenario is None:
            st.info("The dashboard inputs are invalid. Fix them or pick a preset scenario.")
    else:
        lab_scenario = preset_scenario(lab_name)
        st.markdown(f"**{lab_name}** — {PRESETS[lab_name]['description']}")
        st.button("Load this preset into the Dashboard", on_click=_load_preset_from_lab)
        st.dataframe(scenario_frame(lab_scenario), hide_index=True)

    if lab_scenario is not None:
        lab_results = run_all_algorithms(lab_scenario)
        lab_table = comparison_table(lab_scenario, lab_results)
        st.markdown("#### Snapshot of this scenario")
        st.dataframe(styled_comparison(lab_table[[
            "Algorithm", "Status", "CPU utilization", "RAM utilization", "Mean utility",
            "Min utility", "Jain's fairness index", "Weighted log Nash product", "Pareto efficient",
        ]]), hide_index=True)
        st.plotly_chart(charts.utility_chart(lab_scenario, lab_results),
                        key="lab_utility_chart")
        key = scenario_key(lab_scenario)

        st.markdown("#### Experiment A · Scarcity → abundance")
        st.caption("Capacity is set to *factor × total demand* for each resource, then all three "
                   "algorithms are re-run at every point.")
        a1, a2 = st.columns([2, 1])
        lo, hi = a1.slider("Capacity ÷ demand range", 0.1, 2.5, (0.2, 1.6), 0.05, key="cap_range")
        metric = a2.selectbox("Metric", ["Jain's fairness index", "Mean utility", "Min utility",
                                         "CPU utilization", "RAM utilization", "Weighted log Nash product"],
                              key="cap_metric")
        cap_df = cached_capacity_sweep(key, lo, hi, 25)
        st.plotly_chart(charts.capacity_sweep_chart(cap_df, metric))
        failed = int((cap_df["Status"] == "FAILED").sum())
        if failed:
            st.warning(f"{failed} sweep point(s) failed and are not plotted.")
        st.download_button("Download scarcity sweep (CSV)", to_csv_bytes(cap_df),
                           "experiment_capacity_sweep.csv", "text/csv", key="dl_cap")

        st.markdown("#### Experiment B · Bargaining power (weights)")
        st.caption("One user's bargaining weight is varied while the others keep theirs. "
                   "Watch that user's share of the gains grow.")
        b1, b2 = st.columns([2, 1])
        who = b1.selectbox("User whose weight changes", lab_scenario.user_names, key=f"weight_user_{hash(tuple(lab_scenario.user_names))}")
        w_hi = b2.slider("Maximum weight", 1.0, 20.0, 8.0, 0.5, key="weight_max")
        w_df = cached_weight_sweep(key, lab_scenario.user_names.index(who), w_hi, 20)
        st.plotly_chart(charts.per_user_sweep_chart(
            w_df, "Weight of selected user", f"NBS utilities as the weight of '{who}' grows"),
            )
        st.download_button("Download weight sweep (CSV)", to_csv_bytes(w_df),
                           "experiment_weight_sweep.csv", "text/csv", key="dl_w")

        st.markdown("#### Experiment C · Disagreement point")
        st.caption("δ is the fraction of each resource guaranteed as a fallback. A larger δ "
                   "protects small users: their disagreement utility, and hence their deal, rises.")
        d_df = cached_disagreement_sweep(key, 19)
        st.plotly_chart(charts.per_user_sweep_chart(
            d_df, "Disagreement fraction", "NBS utilities as the disagreement point rises"),
            )
        st.download_button("Download disagreement sweep (CSV)", to_csv_bytes(d_df),
                           "experiment_disagreement_sweep.csv", "text/csv", key="dl_d")


# ---------------------------------------------------------------------------
# EXPORT TAB
# ---------------------------------------------------------------------------
with tab_export:
    if need_results(st):
        st.subheader("📥 Download results")
        st.caption("Files are generated from the current Dashboard inputs. CSV files open in Excel.")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Per-user allocations (all algorithms)**")
            st.download_button("⬇️ allocations.csv", to_csv_bytes(users_long),
                               "allocations.csv", "text/csv", key="dl_alloc", type="primary")
            st.markdown("**Scenario inputs** (to reproduce the results)")
            st.download_button("⬇️ scenario_inputs.csv", to_csv_bytes(scenario_frame(scenario)),
                               "scenario_inputs.csv", "text/csv", key="dl_inputs")
        with e2:
            st.markdown("**Algorithm comparison (summary metrics)**")
            st.download_button("⬇️ algorithm_comparison.csv", to_csv_bytes(table),
                               "algorithm_comparison.csv", "text/csv", key="dl_cmp", type="primary")
            st.markdown("**Scarcity sweep for the current scenario**")
            sweep_now = cached_capacity_sweep(scenario_key(scenario), 0.2, 1.6, 25)
            st.download_button("⬇️ capacity_sweep.csv", to_csv_bytes(sweep_now),
                               "capacity_sweep.csv", "text/csv", key="dl_sweep_now")
        st.markdown("#### Preview: algorithm comparison")
        st.dataframe(table, hide_index=True)
        st.markdown("#### Preview: per-user allocations")
        st.dataframe(users_long, hide_index=True)


# ---------------------------------------------------------------------------
# VIVA GUIDE (educational section)
# ---------------------------------------------------------------------------
with tab_learn:
    st.subheader("📘 Viva guide: the model, the maths and the results")

    st.markdown("### 1. The problem")
    st.markdown(
        "A cloud cluster has a fixed amount of **CPU cores** and **RAM**. Several users (virtual "
        "machines, services, tenants) each **demand** some of both. When total demand exceeds "
        "capacity we must decide *who gets how much*. We want the decision to be:\n"
        "- **Efficient** – no capacity sits idle while someone still needs it.\n"
        "- **Fair** – no user is starved without a good reason.\n"
        "- **Principled** – based on a clear mathematical rule, not ad-hoc choices."
    )

    st.markdown("### 2. Notation")
    st.markdown(
        "| Symbol | Meaning |\n|---|---|\n"
        "| $n$ | number of users |\n"
        "| $C_r$ | capacity of resource $r$ (CPU or RAM) |\n"
        "| $d_{ir}$ | demand of user $i$ for resource $r$ |\n"
        "| $x_{ir}$ | amount given to user $i$ (the decision variable) |\n"
        "| $p_i$ | priority (Priority-Based only) |\n"
        "| $w_i$ | bargaining weight (Nash Bargaining only) |\n"
        "| $\\delta$ | disagreement fraction (fallback share) |"
    )
    st.markdown("**Constraints** every allocation must satisfy:")
    st.latex(r"\sum_{i=1}^{n} x_{ir} \le C_r \quad \text{(capacity)}, \qquad 0 \le x_{ir} \le d_{ir} \quad \text{(no over-allocation)}")

    st.markdown("### 3. Utility function")
    st.markdown("Each user's happiness is its **average demand satisfaction** over the resources it asked for:")
    st.latex(r"u_i(x) = \frac{1}{|R_i|} \sum_{r \in R_i} \frac{x_{ir}}{d_{ir}}, \qquad R_i = \{ r : d_{ir} > 0 \}")
    st.markdown(
        "So $u_i = 0$ means the user got nothing and $u_i = 1$ means it got everything it wanted. "
        "Dividing by demand makes the utility **unit-free**: CPU cores and GB of RAM become comparable."
    )

    st.markdown("### 4. Disagreement point")
    st.markdown(
        "In bargaining theory, the **disagreement point** $D_i$ is what a player gets if negotiations "
        "fail. Here each active user is guaranteed a fallback share:"
    )
    st.latex(r"b_{ir} = \min\!\left(d_{ir},\ \frac{\delta\, C_r}{n_{\text{active}}}\right), \qquad D_i = u_i(b)")
    st.markdown(
        "With $\\delta = 0$, $D_i = 0$ (no deal means nothing). A larger $\\delta$ strengthens the "
        "position of small users because their outside option is better."
    )

    st.markdown("### 5. The three algorithms")
    st.markdown("**(a) Equal Allocation** – everyone gets the same share, capped at demand:")
    st.latex(r"x_{ir} = \min\!\left(d_{ir},\ \frac{C_r}{n}\right)")
    st.markdown("Simple and symmetric, but shares that small users don't need stay **idle** → inefficient.")
    st.markdown(
        "**(b) Priority-Based Allocation** – users are sorted by priority; the highest group is served "
        "fully first, then the next, and so on. Users with equal priority share equally (water-filling). "
        "Efficient, but **low-priority users can be starved** (utility 0)."
    )
    st.markdown("**(c) Weighted Nash Bargaining Solution (NBS)** – solve the optimization problem:")
    st.latex(r"\max_{x} \ \prod_{i} \big(u_i(x) - D_i\big)^{w_i} \;\;\Longleftrightarrow\;\; \max_{x} \ \sum_{i} w_i \log\big(u_i(x) - D_i\big)")
    st.latex(r"\text{s.t.}\quad \sum_i x_{ir} \le C_r,\quad 0 \le x_{ir} \le d_{ir},\quad u_i(x) > D_i")
    st.markdown(
        "Taking the logarithm turns the product into a sum (same optimum, easier to compute). The "
        "objective is **concave** and the constraints are **linear**, so the problem is convex: "
        "the local optimum found by SciPy's **SLSQP** solver is the global optimum. If SLSQP fails, "
        "**trust-constr** is tried. The result is accepted only after checking feasibility, "
        "individual rationality ($u_i > D_i$) and Pareto efficiency; otherwise it is reported as **FAILED**."
    )
    st.markdown(
        "*Why the log helps fairness:* $\\log$ grows quickly near zero and slowly near one, so taking "
        "a little from a well-served user to help a starving user increases the sum. This is why NBS "
        "avoids starvation while still using all capacity."
    )

    st.markdown("### 6. Nash's bargaining axioms")
    st.markdown(
        "The (symmetric) NBS is the **only** solution satisfying all four of Nash's axioms:\n"
        "1. **Pareto efficiency** – no other allocation makes someone better off without hurting another.\n"
        "2. **Symmetry** – identical users get identical utility.\n"
        "3. **Invariance to affine utility transformations** – changing units (GB → MB) does not change the outcome.\n"
        "4. **Independence of irrelevant alternatives** – removing options that were not chosen does not change the choice.\n\n"
        "The weights $w_i$ relax symmetry to model unequal bargaining power. The test suite checks "
        "symmetry, unit invariance and Pareto efficiency numerically."
    )

    st.markdown("### 7. Nash Bargaining Solution ≠ Nash Equilibrium")
    st.markdown(
        "| | **Nash Bargaining Solution** | **Nash Equilibrium** |\n|---|---|---|\n"
        "| Game type | Cooperative (players negotiate a binding agreement) | Non-cooperative (each player acts alone) |\n"
        "| Defined by | Axioms / maximizing the Nash product | No player gains by changing only its own strategy |\n"
        "| Output | One agreed allocation | A strategy profile (may be several) |\n"
        "| Efficiency | Always Pareto efficient | Can be inefficient (e.g., Prisoner's Dilemma) |\n"
        "| Role of $D$ | Central: gains are measured from the disagreement point | No disagreement point |\n"
        "| In this project | ✅ what we compute | ❌ not computed – mentioned for contrast |\n\n"
        "Both are named after John Nash, which causes confusion. **Say clearly in the viva:** "
        "*\"We use the cooperative Nash Bargaining Solution, not a Nash Equilibrium.\"*"
    )

    st.markdown("### 8. Performance metrics")
    st.latex(r"\text{Utilization}_r = \frac{\sum_i x_{ir}}{C_r} \qquad \text{Demand satisfaction}_r = \frac{\sum_i x_{ir}}{\sum_i d_{ir}}")
    st.latex(r"J(u_1,\dots,u_k) = \frac{\left(\sum_i u_i\right)^2}{k \sum_i u_i^2}, \qquad \tfrac{1}{k} \le J \le 1")
    st.markdown(
        "**Jain's Fairness Index** equals 1 when all users are equally satisfied and $1/k$ when one "
        "user gets everything. It is computed over users with non-zero demand. "
        "**Pareto efficient = Yes** means no resource is idle while some user still wants it."
    )

    st.markdown("### 9. Worked example on your current inputs")
    if results is None:
        st.info("Enter valid inputs on the Dashboard to see the live worked example.")
    elif not results[NASH].success:
        st.error("Nash Bargaining failed for the current inputs, so no worked example is shown.")
    else:
        u_nbs = user_utilities(results[NASH].allocation, scenario.demands)
        D = disagreement_point(scenario)
        mask = bargaining_users(scenario)
        w_norm = np.where(mask, scenario.weights / scenario.weights[mask].sum() if mask.any() else 0, 0)
        gains = u_nbs - D
        ex = pd.DataFrame({
            "User": scenario.user_names,
            "Weight wᵢ (normalized)": np.round(w_norm, 4),
            "Dᵢ": np.round(D, 4),
            "uᵢ (NBS)": np.round(u_nbs, 4),
            "uᵢ − Dᵢ": np.round(gains, 4),
            "wᵢ·log(uᵢ − Dᵢ)": [
                round(float(w * np.log(g)), 4) if m and g > 0 else None
                for w, g, m in zip(w_norm, gains, mask)
            ],
        })
        st.dataframe(ex, hide_index=True)
        log_np = results[NASH].details["log_nash_product"]
        st.markdown(
            f"Sum of the last column = **{fmt_num(log_np, 4)}**. No other feasible allocation gives a "
            "larger value; the Comparison tab shows Equal and Priority-Based scoring lower."
        )

    st.markdown("### 10. Typical results to explain")
    st.markdown(
        "- **Scarcity:** Priority-Based gives some users utility 0 (Jain's index drops sharply). NBS "
        "spreads the shortage so everyone gets something, while still using 100% of capacity.\n"
        "- **Abundance:** NBS and Priority-Based both satisfy everyone. Equal Allocation may still "
        "leave capacity idle and some demands unmet → not Pareto efficient.\n"
        "- **Unequal demands:** Equal Allocation wastes the small users' unused shares; NBS hands them "
        "to the large user.\n"
        "- **Weights:** increasing $w_i$ never lowers user $i$'s utility (Experiment B).\n"
        "- **Disagreement point:** raising $\\delta$ protects users with small fallback options (Experiment C)."
    )

    st.markdown("### 11. Likely viva questions")
    qa = [
        ("Why maximize the log of the Nash product instead of the product?",
         "log is strictly increasing, so the maximizer is the same. The log turns a product into a sum, "
         "which is numerically stable and makes the problem concave."),
        ("Why is the solution guaranteed to be the global optimum?",
         "Utilities are linear in x, so log(uᵢ − Dᵢ) is concave; a positive-weighted sum of concave "
         "functions is concave; constraints are linear. Maximizing a concave function over a convex set "
         "has no spurious local optima."),
        ("What happens if the optimizer fails?",
         "Both SLSQP and trust-constr are tried. If neither returns a point that passes the "
         "feasibility, rationality and Pareto checks, the app shows FAILED and draws no NBS results."),
        ("Difference between priority and bargaining weight?",
         "Priority is a strict order (higher priority is served completely first). A weight only "
         "changes how the *gains* are split; every user still gets a positive share."),
        ("Why is Equal Allocation not always efficient?",
         "It caps each user at C/n. A user needing less than C/n leaves the rest unused even if "
         "others still need it."),
        ("What does the disagreement point represent in a cloud?",
         "The guaranteed minimum (e.g., an SLA-reserved share) a tenant would keep if no agreement "
         "were reached."),
    ]
    for q, a in qa:
        with st.expander(q):
            st.write(a)
