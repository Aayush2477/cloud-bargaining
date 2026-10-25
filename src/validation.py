"""Input validation.

Every scenario is checked before any algorithm runs. Errors stop the
computation; warnings are shown to the user but the computation continues.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import RESOURCES, Scenario

MAX_USERS = 50


class InvalidScenarioError(ValueError):
    """Raised when an algorithm is called with an invalid scenario."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _all_finite(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)))


def validate_scenario(scenario: Scenario) -> ValidationReport:
    """Check shapes, signs and ranges of every input. Never raises."""
    report = ValidationReport()
    C, d = scenario.capacities, scenario.demands
    p, w = scenario.priorities, scenario.weights

    # ---- capacities -------------------------------------------------------
    if C.ndim != 1 or C.size == 0:
        report.errors.append("Capacities must be a non-empty list of numbers.")
        return report
    if not _all_finite(C):
        report.errors.append("Capacities must be finite numbers (no blanks, NaN or infinity).")
    elif np.any(C <= 0):
        bad = [RESOURCES[r] if r < len(RESOURCES) else f"resource {r}" for r in np.where(C <= 0)[0]]
        report.errors.append(f"Capacity must be greater than zero for: {', '.join(bad)}.")

    # ---- users ------------------------------------------------------------
    n = d.shape[0]
    if d.ndim != 2 or d.shape[1] != C.size:
        report.errors.append(
            f"Demands must have one column per resource ({C.size}); got shape {d.shape}."
        )
        return report
    if n < 1:
        report.errors.append("At least one user is required.")
        return report
    if n > MAX_USERS:
        report.errors.append(f"At most {MAX_USERS} users are supported (got {n}).")

    if not _all_finite(d):
        report.errors.append("Every demand must be a finite number (check for empty cells).")
    elif np.any(d < 0):
        report.errors.append("Demands cannot be negative.")

    if p.shape != (n,):
        report.errors.append(f"Expected {n} priorities, got {p.size}.")
    elif not _all_finite(p) or np.any(p <= 0):
        report.errors.append("Every priority must be a finite number greater than zero.")

    if w.shape != (n,):
        report.errors.append(f"Expected {n} bargaining weights, got {w.size}.")
    elif not _all_finite(w) or np.any(w <= 0):
        report.errors.append("Every bargaining weight must be a finite number greater than zero.")

    names = scenario.user_names or []
    if len(names) != n:
        report.errors.append(f"Expected {n} user names, got {len(names)}.")
    else:
        stripped = [name.strip() for name in names]
        if any(name == "" for name in stripped):
            report.errors.append("User names cannot be empty.")
        elif len(set(stripped)) != len(stripped):
            report.errors.append("User names must be unique.")

    delta = scenario.disagreement_fraction
    if not np.isfinite(delta) or not (0.0 <= delta < 1.0):
        report.errors.append("Disagreement fraction must be in the range [0, 1).")

    if report.errors:
        return report

    # ---- warnings (computation still runs) --------------------------------
    total_demand = d.sum(axis=1)
    if np.all(total_demand == 0):
        report.warnings.append("All demands are zero: there is nothing to allocate.")
    else:
        zero_users = [names[i] for i in np.where(total_demand == 0)[0]]
        if zero_users:
            report.warnings.append(
                "Users with zero demand receive nothing and are excluded from "
                f"utility/fairness metrics: {', '.join(zero_users)}."
            )
    for r in range(C.size):
        too_big = [names[i] for i in np.where(d[:, r] > C[r])[0]]
        if too_big:
            report.warnings.append(
                f"{', '.join(too_big)} demand(s) more {RESOURCES[r]} than the whole system has; "
                "they can never be fully satisfied."
            )
    return report


def ensure_valid(scenario: Scenario) -> None:
    """Raise InvalidScenarioError if the scenario has any error."""
    report = validate_scenario(scenario)
    if not report.ok:
        raise InvalidScenarioError(" ".join(report.errors))
