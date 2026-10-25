"""Performance metrics used to compare the algorithms.

utilization[r]          = sum_i x[i, r] / C[r]
demand satisfaction[r]  = sum_i x[i, r] / sum_i d[i, r]
user utility u_i        = average of x[i, r] / d[i, r] over the resources i demands
Jain's Fairness Index   J = (sum u_i)^2 / (k * sum u_i^2),   1/k <= J <= 1
                        (computed over the k users that have demand)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .game_theory import (
    PARETO_TOL,
    active_users,
    disagreement_point,
    user_utilities,
    weighted_log_nash_product,
)
from .models import RESOURCE_UNITS, RESOURCES, AllocationResult, Scenario


def jains_fairness_index(values: np.ndarray) -> float:
    """Jain's index of a vector, ignoring NaN entries.

    Returns NaN when it is undefined (no values, or every value is zero).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    denom = v.size * np.sum(v ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sum(v) ** 2 / denom)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def summary_metrics(scenario: Scenario, result: AllocationResult) -> dict:
    """One row of the comparison table for one algorithm."""
    row: dict = {"Algorithm": result.algorithm, "Status": "OK" if result.success else "FAILED"}
    keys = []
    for name in RESOURCES[: scenario.n_resources]:
        keys += [f"{name} utilization", f"{name} demand satisfaction", f"{name} idle"]
    keys += [
        "Mean utility", "Min utility", "Max utility", "Total utility",
        "Jain's fairness index", "Weighted log Nash product", "Pareto efficient",
    ]
    if not result.success or result.allocation is None:
        row.update({key: np.nan for key in keys})
        row["Pareto efficient"] = None
        return row

    x, d, C = result.allocation, scenario.demands, scenario.capacities
    used, total_demand = x.sum(axis=0), d.sum(axis=0)
    pareto = True
    for r, name in enumerate(RESOURCES[: scenario.n_resources]):
        row[f"{name} utilization"] = _safe_ratio(used[r], C[r])
        row[f"{name} demand satisfaction"] = _safe_ratio(used[r], total_demand[r])
        row[f"{name} idle"] = float(C[r] - used[r])
        idle = C[r] - used[r] > PARETO_TOL * C[r]
        unmet = np.any(d[:, r] - x[:, r] > PARETO_TOL * max(C[r], 1.0))
        if idle and unmet:
            pareto = False

    u = user_utilities(x, d)
    active = active_users(d)
    ua = u[active]
    has_users = ua.size > 0
    row["Mean utility"] = float(np.mean(ua)) if has_users else np.nan
    row["Min utility"] = float(np.min(ua)) if has_users else np.nan
    row["Max utility"] = float(np.max(ua)) if has_users else np.nan
    row["Total utility"] = float(np.sum(ua)) if has_users else np.nan
    row["Jain's fairness index"] = jains_fairness_index(ua)
    row["Weighted log Nash product"] = weighted_log_nash_product(u, scenario)
    row["Pareto efficient"] = pareto
    return row


def comparison_table(scenario: Scenario, results: dict[str, AllocationResult]) -> pd.DataFrame:
    """Algorithm-by-algorithm comparison (one row per algorithm)."""
    return pd.DataFrame([summary_metrics(scenario, res) for res in results.values()])


def user_table(scenario: Scenario, result: AllocationResult) -> pd.DataFrame:
    """Per-user allocation details for one algorithm."""
    d = scenario.demands
    D = disagreement_point(scenario)
    rows = []
    x = result.allocation if result.success and result.allocation is not None else None
    u = user_utilities(x, d) if x is not None else np.full(scenario.n_users, np.nan)
    for i, name in enumerate(scenario.user_names):
        row = {
            "Algorithm": result.algorithm,
            "User": name,
            "Priority": scenario.priorities[i],
            "Bargaining weight": scenario.weights[i],
        }
        for r, res_name in enumerate(RESOURCES[: scenario.n_resources]):
            unit = RESOURCE_UNITS[r]
            row[f"{res_name} demand ({unit})"] = d[i, r]
            row[f"{res_name} allocated ({unit})"] = x[i, r] if x is not None else np.nan
            row[f"{res_name} satisfaction"] = (
                x[i, r] / d[i, r] if (x is not None and d[i, r] > 0) else np.nan
            )
        row["Utility"] = u[i]
        row["Disagreement utility"] = D[i]
        row["Gain over disagreement"] = u[i] - D[i]
        rows.append(row)
    return pd.DataFrame(rows)


def all_user_tables(scenario: Scenario, results: dict[str, AllocationResult]) -> pd.DataFrame:
    """Per-user rows for every algorithm stacked into one long table."""
    return pd.concat([user_table(scenario, res) for res in results.values()], ignore_index=True)
