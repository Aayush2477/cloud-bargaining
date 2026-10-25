"""Experiment Lab: preset scenarios and parameter sweeps.

Every sweep re-runs the real algorithms; nothing here is pre-computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .algorithms import run_all_algorithms
from .game_theory import nash_bargaining_solution, user_utilities
from .metrics import jains_fairness_index, summary_metrics
from .models import NASH, Scenario

# Users of a typical small cloud cluster: (name, CPU cores, RAM GB, priority, weight)
_BASE_USERS = [
    ("Web Server", 8, 32, 3, 1.0),
    ("Database", 12, 48, 5, 1.0),
    ("Analytics", 16, 64, 2, 1.0),
    ("Batch Jobs", 6, 40, 1, 1.0),
    ("Dev/Test", 4, 16, 1, 1.0),
]

PRESETS: dict[str, dict] = {
    "Balanced baseline": {
        "description": "Moderate contention: total demand (46 cores, 200 GB) exceeds "
        "capacity (32 cores, 128 GB). Equal priorities except the database.",
        "cpu": 32.0, "ram": 128.0, "delta": 0.0,
        "users": _BASE_USERS,
    },
    "Resource scarcity": {
        "description": "Severe shortage: only about a third of the total demand is available. "
        "Shows how each algorithm shares pain when resources are scarce.",
        "cpu": 16.0, "ram": 64.0, "delta": 0.0,
        "users": _BASE_USERS,
    },
    "Abundant resources": {
        "description": "Capacity exceeds every demand. All sensible algorithms should satisfy "
        "everyone; Equal Allocation may still leave capacity idle.",
        "cpu": 64.0, "ram": 256.0, "delta": 0.0,
        "users": _BASE_USERS,
    },
    "Unequal priorities": {
        "description": "One mission-critical service has a much higher priority. "
        "Priority-Based starves low-priority users; NBS ignores priority (it uses weights).",
        "cpu": 24.0, "ram": 96.0, "delta": 0.0,
        "users": [
            ("Payments (critical)", 12, 48, 10, 1.0),
            ("Web Server", 8, 32, 2, 1.0),
            ("Analytics", 10, 40, 1, 1.0),
            ("Batch Jobs", 6, 24, 1, 1.0),
        ],
    },
    "Unequal demands": {
        "description": "One very large user next to several small ones. Equal Allocation wastes "
        "the small users' unused shares; NBS and water-filling re-use them.",
        "cpu": 32.0, "ram": 128.0, "delta": 0.0,
        "users": [
            ("ML Training (huge)", 40, 160, 1, 1.0),
            ("API Gateway", 2, 8, 1, 1.0),
            ("Cache", 2, 16, 1, 1.0),
            ("Monitoring", 1, 4, 1, 1.0),
            ("CI Runner", 4, 8, 1, 1.0),
        ],
    },
    "Different bargaining weights": {
        "description": "Same users as the baseline, but bargaining power differs (4, 2, 1, 1, 1). "
        "Higher weight = larger share of the gains in the Nash product.",
        "cpu": 32.0, "ram": 128.0, "delta": 0.1,
        "users": [
            ("Web Server", 8, 32, 3, 4.0),
            ("Database", 12, 48, 5, 2.0),
            ("Analytics", 16, 64, 2, 1.0),
            ("Batch Jobs", 6, 40, 1, 1.0),
            ("Dev/Test", 4, 16, 1, 1.0),
        ],
    },
}


def preset_users_frame(name: str) -> pd.DataFrame:
    """User table of a preset in the format used by the dashboard editor."""
    rows = PRESETS[name]["users"]
    return pd.DataFrame(
        [
            {"User": u, "CPU demand (cores)": float(c), "RAM demand (GB)": float(m),
             "Priority": float(p), "Bargaining weight": float(w)}
            for u, c, m, p, w in rows
        ]
    )


def preset_scenario(name: str) -> Scenario:
    preset = PRESETS[name]
    users = preset["users"]
    return Scenario(
        capacities=[preset["cpu"], preset["ram"]],
        demands=[[u[1], u[2]] for u in users],
        priorities=[u[3] for u in users],
        weights=[u[4] for u in users],
        user_names=[u[0] for u in users],
        disagreement_fraction=preset["delta"],
    )


def capacity_sweep(scenario: Scenario, factors) -> pd.DataFrame:
    """Scale capacity to factor x total demand and re-run all algorithms.

    factor < 1 means scarcity, factor > 1 means abundance. Resources with zero
    total demand keep their original capacity.
    """
    total = scenario.demands.sum(axis=0)
    rows = []
    for f in factors:
        caps = np.where(total > 0, f * total, scenario.capacities)
        caps = np.maximum(caps, 1e-6)
        s = scenario.copy(capacities=caps)
        for res in run_all_algorithms(s).values():
            row = summary_metrics(s, res)
            row["Capacity / demand"] = float(f)
            rows.append(row)
    return pd.DataFrame(rows)


def weight_sweep(scenario: Scenario, user_index: int, weight_values) -> pd.DataFrame:
    """Vary one user's bargaining weight and record every user's NBS utility."""
    rows = []
    for wv in weight_values:
        weights = scenario.weights.copy()
        weights[user_index] = float(wv)
        s = scenario.copy(weights=weights)
        res = nash_bargaining_solution(s)
        u = user_utilities(res.allocation, s.demands) if res.success else np.full(s.n_users, np.nan)
        for i, name in enumerate(s.user_names):
            rows.append({
                "Weight of selected user": float(wv), "User": name,
                "Utility": u[i], "Status": "OK" if res.success else "FAILED",
            })
    return pd.DataFrame(rows)


def disagreement_sweep(scenario: Scenario, deltas) -> pd.DataFrame:
    """Vary the disagreement fraction delta and record NBS utilities and fairness."""
    rows = []
    for delta in deltas:
        s = scenario.copy(disagreement_fraction=float(delta))
        res = nash_bargaining_solution(s)
        u = user_utilities(res.allocation, s.demands) if res.success else np.full(s.n_users, np.nan)
        for i, name in enumerate(s.user_names):
            rows.append({
                "Disagreement fraction": float(delta), "User": name, "Utility": u[i],
                "Jain's fairness index": jains_fairness_index(u) if res.success else np.nan,
                "Status": "OK" if res.success else "FAILED",
            })
    return pd.DataFrame(rows)


def nash_only_failed(results) -> bool:
    return not results[NASH].success
