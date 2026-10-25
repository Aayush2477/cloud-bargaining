"""Game-Theoretic Bargaining for Fair and Efficient Allocation of
Computational Resources in Cloud Computing.

Package layout
--------------
models.py       Data containers (Scenario, AllocationResult)
validation.py   Input checks (capacities, demands, priorities, weights)
game_theory.py  Utilities, disagreement point, weighted Nash Bargaining Solution
algorithms.py   Equal, Priority-Based and Nash Bargaining allocation
metrics.py      Utilization, demand satisfaction, utility, Jain's Fairness Index
experiments.py  Preset scenarios and parameter sweeps for the Experiment Lab
export.py       CSV export helpers
"""

__version__ = "1.0.0"
