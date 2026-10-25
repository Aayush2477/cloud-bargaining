"""The three allocation algorithms compared in this project.

1. Equal Allocation      every user gets C[r] / n of each resource (capped at demand)
2. Priority-Based        users are served strictly in order of priority
3. Nash Bargaining       weighted Nash Bargaining Solution (see game_theory.py)

Each algorithm treats every resource separately except Nash Bargaining,
which optimizes the joint utility over CPU and RAM together.
"""

from __future__ import annotations

import numpy as np

from .game_theory import nash_bargaining_solution
from .models import EQUAL, PRIORITY, AllocationResult, Scenario
from .validation import ensure_valid


def water_fill(capacity: float, demands: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted max-min fair split of one resource ("water-filling").

    Every user gets x_i = min(d_i, lambda * w_i), with lambda chosen so that
    either all capacity is used or every demand is met. Users that need less
    than their share are fully satisfied and the rest is re-shared.
    """
    demands = np.asarray(demands, dtype=float)
    weights = np.ones_like(demands) if weights is None else np.asarray(weights, dtype=float)
    x = np.zeros_like(demands)
    open_users = demands > 0
    remaining = float(capacity)
    while remaining > 1e-12 and open_users.any():
        share = np.zeros_like(demands)
        share[open_users] = remaining * weights[open_users] / weights[open_users].sum()
        need = demands - x
        can_finish = open_users & (need <= share)
        if not can_finish.any():
            # Nobody can be completed: give everyone their share and stop.
            x[open_users] += share[open_users]
            remaining = 0.0
            break
        x[can_finish] = demands[can_finish]
        remaining -= need[can_finish].sum()
        open_users &= ~can_finish
    return np.minimum(x, demands)


def equal_allocation(scenario: Scenario) -> AllocationResult:
    """Split every resource equally among ALL n users, capped at each demand.

    Capacity a user does not need is NOT given to anyone else; this shows the
    inefficiency (idle capacity) of a naive equal split.
    """
    ensure_valid(scenario)
    share = scenario.capacities / scenario.n_users
    x = np.minimum(scenario.demands, share[None, :])
    return AllocationResult(
        EQUAL, x, True,
        "Each user received min(demand, capacity / n). Unused shares stay idle.",
        {"equal_share": share},
    )


def priority_allocation(scenario: Scenario) -> AllocationResult:
    """Strict priority: serve the highest-priority users first.

    Users are grouped by priority (highest first). Each group takes as much of
    the remaining capacity as it needs; inside a group with equal priority the
    capacity is shared with equal water-filling, so ties are treated fairly.
    """
    ensure_valid(scenario)
    d, C = scenario.demands, scenario.capacities
    x = np.zeros_like(d)
    remaining = C.astype(float).copy()
    levels = sorted(set(scenario.priorities.tolist()), reverse=True)
    for level in levels:
        group = scenario.priorities == level
        for r in range(scenario.n_resources):
            if remaining[r] <= 0:
                continue
            x[group, r] = water_fill(remaining[r], d[group, r])
            remaining[r] -= x[group, r].sum()
        remaining = np.maximum(remaining, 0.0)
    return AllocationResult(
        PRIORITY, x, True,
        "Users served in descending priority order; ties shared equally.",
        {"priority_order": levels},
    )


def run_all_algorithms(scenario: Scenario) -> dict[str, AllocationResult]:
    """Run the three algorithms on the same scenario."""
    results = [equal_allocation(scenario), priority_allocation(scenario), nash_bargaining_solution(scenario)]
    return {res.algorithm: res for res in results}
