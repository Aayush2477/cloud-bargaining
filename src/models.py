"""Data containers shared by every module.

A *scenario* describes the cloud system (capacities) and the users
(demands, priorities, bargaining weights). An *allocation result* is what an
algorithm returns: an (n_users x n_resources) matrix plus a success flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# The two resources modelled in this project. Index 0 = CPU, index 1 = RAM.
RESOURCES: tuple[str, ...] = ("CPU", "RAM")
RESOURCE_UNITS: tuple[str, ...] = ("cores", "GB")

# Algorithm names used everywhere (UI, tables, CSV files).
EQUAL = "Equal Allocation"
PRIORITY = "Priority-Based"
NASH = "Nash Bargaining"
ALGORITHMS: tuple[str, ...] = (EQUAL, PRIORITY, NASH)


@dataclass
class Scenario:
    """Everything needed to run an allocation algorithm.

    capacities            shape (R,)    total CPU cores and RAM (GB) available
    demands               shape (n, R)  what each user asks for
    priorities            shape (n,)    importance of each user (higher = more important)
    weights               shape (n,)    bargaining power used in the weighted Nash product
    user_names            n labels
    disagreement_fraction delta in [0, 1): fraction of each resource reserved as
                          the "fallback" share if bargaining fails (the disagreement point)
    """

    capacities: np.ndarray
    demands: np.ndarray
    priorities: np.ndarray
    weights: np.ndarray
    user_names: Optional[list[str]] = None
    disagreement_fraction: float = 0.0

    def __post_init__(self) -> None:
        self.capacities = np.asarray(self.capacities, dtype=float).reshape(-1)
        self.demands = np.atleast_2d(np.asarray(self.demands, dtype=float))
        self.priorities = np.asarray(self.priorities, dtype=float).reshape(-1)
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        self.disagreement_fraction = float(self.disagreement_fraction)
        if self.user_names is None:
            self.user_names = [f"User {i + 1}" for i in range(self.demands.shape[0])]
        else:
            self.user_names = [str(name) for name in self.user_names]

    @property
    def n_users(self) -> int:
        return int(self.demands.shape[0])

    @property
    def n_resources(self) -> int:
        return int(self.demands.shape[1])

    def copy(self, **changes) -> "Scenario":
        """Return a modified copy (used by the Experiment Lab sweeps)."""
        data = dict(
            capacities=self.capacities.copy(),
            demands=self.demands.copy(),
            priorities=self.priorities.copy(),
            weights=self.weights.copy(),
            user_names=list(self.user_names),
            disagreement_fraction=self.disagreement_fraction,
        )
        data.update(changes)
        return Scenario(**data)


USER_COLUMNS: tuple[str, ...] = (
    "User", "CPU demand (cores)", "RAM demand (GB)", "Priority", "Bargaining weight",
)


def scenario_from_frame(users, cpu_capacity, ram_capacity, disagreement_fraction=0.0) -> Scenario:
    """Build a Scenario from the dashboard's user table (a pandas DataFrame).

    Blank or non-numeric cells become NaN so that validation reports them
    instead of the app crashing.
    """
    import pandas as pd  # local import keeps models.py light

    def numeric(column: str) -> np.ndarray:
        return pd.to_numeric(users[column], errors="coerce").to_numpy(dtype=float)

    names = ["" if pd.isna(v) else str(v).strip() for v in users["User"]]
    demands = np.column_stack([numeric(USER_COLUMNS[1]), numeric(USER_COLUMNS[2])]) if len(users) else np.zeros((0, 2))
    return Scenario(
        capacities=[
            float(cpu_capacity) if cpu_capacity is not None else np.nan,
            float(ram_capacity) if ram_capacity is not None else np.nan,
        ],
        demands=demands,
        priorities=numeric(USER_COLUMNS[3]),
        weights=numeric(USER_COLUMNS[4]),
        user_names=names,
        disagreement_fraction=disagreement_fraction,
    )


@dataclass
class AllocationResult:
    """Output of one allocation algorithm.

    allocation is None only when the algorithm failed. A failed result is
    never silently replaced by a made-up allocation.
    """

    algorithm: str
    allocation: Optional[np.ndarray]
    success: bool
    message: str
    details: dict = field(default_factory=dict)
