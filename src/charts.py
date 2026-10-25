"""Plotly figures used by the dashboard.

Every chart is built from real results passed in by the caller. Failed
algorithms (no allocation) are skipped and never drawn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .game_theory import disagreement_point, user_utilities
from .models import EQUAL, NASH, PRIORITY, RESOURCE_UNITS, RESOURCES, AllocationResult, Scenario

ALGORITHM_COLORS = {EQUAL: "#64748B", PRIORITY: "#F59E0B", NASH: "#4F46E5"}
USER_PALETTE = ["#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444",
                "#8B5CF6", "#EC4899", "#14B8A6", "#84CC16", "#F97316",
                "#6366F1", "#06B6D4"]


def _style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 16}},
        height=height,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        barmode="group",
        hovermode="closest",
    )
    return fig


def _ok(results: dict[str, AllocationResult]) -> dict[str, AllocationResult]:
    return {k: r for k, r in results.items() if r.success and r.allocation is not None}


def allocation_chart(scenario: Scenario, results: dict[str, AllocationResult]) -> go.Figure:
    """Grouped bars: what each user received of CPU and RAM, with demand markers."""
    titles = [f"{RESOURCES[r]} ({RESOURCE_UNITS[r]})" for r in range(scenario.n_resources)]
    fig = make_subplots(rows=1, cols=scenario.n_resources, subplot_titles=titles)
    names = scenario.user_names
    for r in range(scenario.n_resources):
        for name, res in _ok(results).items():
            fig.add_trace(
                go.Bar(
                    x=names, y=res.allocation[:, r], name=name,
                    marker_color=ALGORITHM_COLORS.get(name), legendgroup=name, showlegend=(r == 0),
                    hovertemplate="%{x}<br>" + name + ": %{y:.2f} " + RESOURCE_UNITS[r] + "<extra></extra>",
                ),
                row=1, col=r + 1,
            )
        fig.add_trace(
            go.Scatter(
                x=names, y=scenario.demands[:, r], mode="markers", name="Demand",
                marker={"symbol": "line-ew-open", "size": 34, "line": {"width": 3, "color": "#DC2626"}},
                legendgroup="demand", showlegend=(r == 0),
                hovertemplate="%{x}<br>Demand: %{y:.2f} " + RESOURCE_UNITS[r] + "<extra></extra>",
            ),
            row=1, col=r + 1,
        )
    return _style(fig, "Allocation per user (red line = demand)", height=440)


def utility_chart(scenario: Scenario, results: dict[str, AllocationResult]) -> go.Figure:
    """Grouped bars: user utility (average demand satisfaction) per algorithm."""
    fig = go.Figure()
    for name, res in _ok(results).items():
        u = user_utilities(res.allocation, scenario.demands)
        fig.add_trace(go.Bar(
            x=scenario.user_names, y=u, name=name, marker_color=ALGORITHM_COLORS.get(name),
            hovertemplate="%{x}<br>" + name + ": utility %{y:.3f}<extra></extra>",
        ))
    D = disagreement_point(scenario)
    if np.nanmax(np.nan_to_num(D, nan=0.0)) > 0:
        fig.add_trace(go.Scatter(
            x=scenario.user_names, y=D, mode="markers", name="Disagreement utility",
            marker={"symbol": "diamond", "size": 11, "color": "#DC2626"},
            hovertemplate="%{x}<br>Disagreement utility: %{y:.3f}<extra></extra>",
        ))
    fig.update_yaxes(title="Utility (0 = nothing, 1 = fully satisfied)", range=[0, 1.05])
    return _style(fig, "User satisfaction (utility) by algorithm")


def fairness_chart(comparison: pd.DataFrame) -> go.Figure:
    df = comparison[comparison["Status"] == "OK"]
    fig = go.Figure(go.Bar(
        x=df["Algorithm"], y=df["Jain's fairness index"],
        marker_color=[ALGORITHM_COLORS.get(a) for a in df["Algorithm"]],
        text=[f"{v:.3f}" if np.isfinite(v) else "N/A" for v in df["Jain's fairness index"]],
        textposition="outside",
        hovertemplate="%{x}<br>Jain's index: %{y:.4f}<extra></extra>",
    ))
    fig.update_yaxes(title="Jain's Fairness Index", range=[0, 1.12])
    return _style(fig, "Fairness (Jain's index, 1 = perfectly fair)", height=380)


def utilization_chart(comparison: pd.DataFrame, scenario: Scenario) -> go.Figure:
    df = comparison[comparison["Status"] == "OK"]
    fig = go.Figure()
    for name in df["Algorithm"]:
        row = df[df["Algorithm"] == name].iloc[0]
        values = [row[f"{res} utilization"] * 100 for res in RESOURCES[: scenario.n_resources]]
        fig.add_trace(go.Bar(
            x=list(RESOURCES[: scenario.n_resources]), y=values, name=name,
            marker_color=ALGORITHM_COLORS.get(name),
            text=[f"{v:.1f}%" for v in values], textposition="outside",
            hovertemplate="%{x}: %{y:.2f}% used<extra>" + name + "</extra>",
        ))
    fig.update_yaxes(title="Capacity used (%)", range=[0, 115])
    return _style(fig, "Resource utilization", height=380)


def demand_satisfaction_chart(comparison: pd.DataFrame, scenario: Scenario) -> go.Figure:
    df = comparison[comparison["Status"] == "OK"]
    fig = go.Figure()
    for name in df["Algorithm"]:
        row = df[df["Algorithm"] == name].iloc[0]
        values = [row[f"{res} demand satisfaction"] * 100 for res in RESOURCES[: scenario.n_resources]]
        fig.add_trace(go.Bar(
            x=list(RESOURCES[: scenario.n_resources]), y=values, name=name,
            marker_color=ALGORITHM_COLORS.get(name),
            text=[f"{v:.1f}%" if np.isfinite(v) else "N/A" for v in values], textposition="outside",
            hovertemplate="%{x}: %{y:.2f}% of total demand met<extra>" + name + "</extra>",
        ))
    fig.update_yaxes(title="Total demand satisfied (%)", range=[0, 115])
    return _style(fig, "Aggregate demand satisfaction", height=380)


def nash_product_chart(comparison: pd.DataFrame) -> go.Figure:
    """Weighted log Nash product. -inf means some user gained nothing (product = 0)."""
    df = comparison[comparison["Status"] == "OK"]
    values = df["Weighted log Nash product"].to_numpy(dtype=float)
    finite = np.isfinite(values)
    labels = [
        f"{v:.4f}" if np.isfinite(v)
        else ("−∞: a user gains nothing over D" if v == -np.inf else "undefined (nobody bargains)")
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=df["Algorithm"], y=np.where(finite, values, 0.0),
        marker_color=[ALGORITHM_COLORS.get(a) for a in df["Algorithm"]],
        text=labels, textposition="outside", cliponaxis=False,
        hovertext=labels, hovertemplate="%{x}<br>Σ wᵢ·log(uᵢ − Dᵢ) = %{hovertext}<extra></extra>",
    ))
    fig.update_yaxes(title="Σ wᵢ·log(uᵢ − Dᵢ)  (higher = better)", zeroline=True)
    return _style(fig, "Nash product objective (NBS maximizes this)", height=380)


def capacity_sweep_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure()
    for name in df["Algorithm"].unique():
        part = df[(df["Algorithm"] == name) & (df["Status"] == "OK")]
        fig.add_trace(go.Scatter(
            x=part["Capacity / demand"], y=part[metric], mode="lines+markers", name=name,
            line={"color": ALGORITHM_COLORS.get(name), "width": 3},
            hovertemplate="capacity = %{x:.2f} × demand<br>" + metric + ": %{y:.4f}<extra>" + name + "</extra>",
        ))
    fig.add_vline(x=1.0, line_dash="dot", line_color="#94A3B8",
                  annotation_text="capacity = demand", annotation_position="top")
    fig.update_xaxes(title="Capacity ÷ total demand  (left = scarcity, right = abundance)")
    fig.update_yaxes(title=metric)
    return _style(fig, f"{metric} vs. resource availability")


def per_user_sweep_chart(df: pd.DataFrame, x_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    for k, user in enumerate(df["User"].unique()):
        part = df[(df["User"] == user) & (df["Status"] == "OK")]
        fig.add_trace(go.Scatter(
            x=part[x_col], y=part["Utility"], mode="lines+markers", name=user,
            line={"color": USER_PALETTE[k % len(USER_PALETTE)], "width": 3},
            hovertemplate=x_col + " = %{x:.2f}<br>utility %{y:.4f}<extra>" + user + "</extra>",
        ))
    fig.update_xaxes(title=x_col)
    fig.update_yaxes(title="Utility under Nash Bargaining", range=[0, 1.05])
    return _style(fig, title)
