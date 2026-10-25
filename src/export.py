"""CSV export helpers (used by the Export tab)."""

from __future__ import annotations

import pandas as pd

from .models import RESOURCE_UNITS, RESOURCES, Scenario


def scenario_frame(scenario: Scenario) -> pd.DataFrame:
    """Inputs of a scenario as a table, so exported results are reproducible."""
    rows = []
    for i, name in enumerate(scenario.user_names):
        row = {"User": name}
        for r, res in enumerate(RESOURCES[: scenario.n_resources]):
            row[f"{res} demand ({RESOURCE_UNITS[r]})"] = scenario.demands[i, r]
        row["Priority"] = scenario.priorities[i]
        row["Bargaining weight"] = scenario.weights[i]
        rows.append(row)
    capacity_row = {"User": "SYSTEM CAPACITY"}
    for r, res in enumerate(RESOURCES[: scenario.n_resources]):
        capacity_row[f"{res} demand ({RESOURCE_UNITS[r]})"] = scenario.capacities[r]
    capacity_row["Priority"] = None
    capacity_row["Bargaining weight"] = None
    rows.append(capacity_row)
    df = pd.DataFrame(rows)
    df["Disagreement fraction"] = scenario.disagreement_fraction
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 CSV with a BOM so Excel on Windows opens it correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")
