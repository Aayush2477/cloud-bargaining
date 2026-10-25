"""Game-theoretic core: utilities, disagreement point and the weighted
Nash Bargaining Solution (NBS).

Model (n users, R resources)
----------------------------
x[i, r]  amount of resource r given to user i            (decision variable)
d[i, r]  demand of user i for resource r
C[r]     capacity of resource r

Utility of user i (normalized demand satisfaction, between 0 and 1):

    u_i(x) = (1 / |R_i|) * sum_{r in R_i} x[i, r] / d[i, r]

where R_i is the set of resources user i actually demands (d[i, r] > 0).

Disagreement point D_i: the utility user i gets if bargaining breaks down.
Here each active user is guaranteed a reserved fallback share

    b[i, r] = min(d[i, r], delta * C[r] / n_active)

and D_i = u_i(b). With delta = 0 the disagreement point is D = 0.

Weighted Nash Bargaining Solution:

    maximize    sum_i w_i * log(u_i(x) - D_i)       (log of the weighted Nash product)
    subject to  sum_i x[i, r] <= C[r]                for every resource r
                0 <= x[i, r] <= d[i, r]              never give more than asked
                u_i(x) > D_i                         individual rationality

The objective is concave and the constraints are linear, so any local
optimum is the global optimum. It is solved numerically with SciPy.

NBS is NOT a Nash Equilibrium. NBS is a cooperative solution (users agree on
a joint allocation chosen by axioms: Pareto efficiency, symmetry,
independence of irrelevant alternatives, invariance to utility scaling).
A Nash Equilibrium belongs to non-cooperative games: a strategy profile in
which no single player can gain by changing only their own strategy.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .models import NASH, AllocationResult, Scenario
from .validation import ensure_valid

# Numerical tolerances
SATISFIED_TOL = 1e-9  # D_i >= 1 - tol  -> user already fully satisfied
IR_MARGIN = 1e-10  # u_i - D_i must stay above this inside the optimizer
FEAS_TOL = 1e-7  # relative tolerance for capacity / bound checks
PARETO_TOL = 1e-5  # relative tolerance for "leftover capacity while demand is unmet"


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def active_users(demands: np.ndarray) -> np.ndarray:
    """Boolean mask: users that demand at least one resource."""
    return np.asarray(demands).sum(axis=1) > 0


def utility_coefficients(demands: np.ndarray) -> np.ndarray:
    """Coefficients c[i, r] such that u_i(x) = sum_r c[i, r] * x[i, r].

    c[i, r] = 1 / (|R_i| * d[i, r]) if d[i, r] > 0, else 0.
    """
    demands = np.asarray(demands, dtype=float)
    coef = np.zeros_like(demands)
    demanded = demands > 0
    counts = demanded.sum(axis=1)
    for i in range(demands.shape[0]):
        if counts[i] > 0:
            coef[i, demanded[i]] = 1.0 / (counts[i] * demands[i, demanded[i]])
    return coef


def user_utilities(allocation: np.ndarray, demands: np.ndarray) -> np.ndarray:
    """Utility of every user. NaN for users with zero demand (undefined)."""
    allocation = np.asarray(allocation, dtype=float)
    utilities = (utility_coefficients(demands) * allocation).sum(axis=1)
    utilities[~active_users(demands)] = np.nan
    return utilities


# --------------------------------------------------------------------------
# Disagreement point
# --------------------------------------------------------------------------
def disagreement_allocation(scenario: Scenario) -> np.ndarray:
    """Fallback allocation b if the users fail to agree.

    Each active user is guaranteed delta * C[r] / n_active of every resource,
    capped at its own demand. It always fits in capacity because
    sum_i b[i, r] <= delta * C[r] < C[r].
    """
    d = scenario.demands
    active = active_users(d)
    n_active = int(active.sum())
    if n_active == 0:
        return np.zeros_like(d)
    share = scenario.disagreement_fraction * scenario.capacities / n_active
    b = np.minimum(d, share[None, :])
    b[~active] = 0.0
    return b


def disagreement_point(scenario: Scenario) -> np.ndarray:
    """D_i = u_i(b): utility each user gets if bargaining fails."""
    return user_utilities(disagreement_allocation(scenario), scenario.demands)


def bargaining_users(scenario: Scenario) -> np.ndarray:
    """Users that take part in the Nash product.

    A user participates if it has demand and is not already fully satisfied by
    the disagreement allocation (otherwise u_i - D_i can never be positive).
    """
    D = disagreement_point(scenario)
    active = active_users(scenario.demands)
    return active & (np.nan_to_num(D, nan=1.0) < 1.0 - SATISFIED_TOL)


def weighted_log_nash_product(
    utilities: np.ndarray, scenario: Scenario
) -> float:
    """sum_i w_i * log(u_i - D_i) over bargaining users (weights normalized to 1).

    Returns -inf if any bargaining user is not strictly better off than its
    disagreement utility (the Nash product is then 0), and NaN if nobody
    bargains (the Nash product is undefined).
    """
    mask = bargaining_users(scenario)
    if not mask.any():
        return float("nan")
    D = disagreement_point(scenario)
    gains = np.asarray(utilities, dtype=float)[mask] - D[mask]
    if np.any(~np.isfinite(gains)) or np.any(gains <= 0):
        return float("-inf")
    w = scenario.weights[mask] / scenario.weights[mask].sum()
    return float(np.sum(w * np.log(gains)))


# --------------------------------------------------------------------------
# Nash Bargaining Solution
# --------------------------------------------------------------------------
def _verify_solution(
    x: np.ndarray, scenario: Scenario, mask: np.ndarray, D: np.ndarray
) -> list[str]:
    """Return a list of problems with a candidate NBS allocation (empty = OK)."""
    issues: list[str] = []
    d, C = scenario.demands, scenario.capacities
    if not np.all(np.isfinite(x)):
        return ["allocation contains non-finite values"]
    scale = np.maximum(d, 1.0)
    if np.any(x < -FEAS_TOL * scale):
        issues.append("negative allocation")
    if np.any(x > d + FEAS_TOL * scale):
        issues.append("allocation exceeds demand")
    used = x.sum(axis=0)
    if np.any(used > C * (1 + FEAS_TOL)):
        issues.append("capacity constraint violated")
    u = user_utilities(x, d)
    if mask.any() and np.any(u[mask] - D[mask] <= 0):
        issues.append("a user is not better off than its disagreement point")
    # Pareto check: NBS must not leave capacity idle while a bargaining user
    # still wants that resource (utility is strictly increasing in x).
    leftover = C - used
    for r in range(C.size):
        unmet = (d[:, r] - x[:, r])[mask & (d[:, r] > 0)]
        if leftover[r] > PARETO_TOL * C[r] and np.any(unmet > PARETO_TOL * np.maximum(C[r], 1.0)):
            issues.append(f"not Pareto-efficient: resource {r} has idle capacity while demand is unmet")
    return issues


def nash_bargaining_solution(scenario: Scenario) -> AllocationResult:
    """Compute the weighted Nash Bargaining Solution by constrained optimization.

    Variables are scaled to y = x / d in [0, 1] for good numerical conditioning.
    SLSQP is tried first; trust-constr is used as a fallback. A solution is
    reported as successful only if the solver converged AND the allocation
    passes every feasibility, rationality and Pareto check.
    """
    ensure_valid(scenario)
    d, C, n, R = scenario.demands, scenario.capacities, scenario.n_users, scenario.n_resources

    b = disagreement_allocation(scenario)
    D = disagreement_point(scenario)
    active = active_users(d)
    mask = bargaining_users(scenario)

    # Users that do not bargain are fixed: zero-demand users get 0, users fully
    # satisfied by their fallback share keep their full demand (= b).
    base = np.zeros_like(d)
    fixed_satisfied = active & ~mask
    base[fixed_satisfied] = d[fixed_satisfied]

    details: dict = {
        "disagreement_allocation": b,
        "disagreement_point": D,
        "bargaining_users": mask,
        "attempts": [],
    }

    if not mask.any():
        details["method"] = "none (no bargaining needed)"
        details["log_nash_product"] = float("nan")
        return AllocationResult(
            NASH, base, True,
            "No bargaining needed: every user with demand is already fully "
            "satisfied by the disagreement allocation (or has zero demand).",
            details,
        )

    # ---- free variables: bargaining users x demanded resources -------------
    free = mask[:, None] & (d > 0)
    idx = np.argwhere(free)  # rows: (user, resource)
    k = len(idx)
    d_free = d[free]  # same order as idx (row-major)
    b_free = b[free]
    bargain_ids = np.where(mask)[0]
    row_of_user = {u: j for j, u in enumerate(bargain_ids)}

    coef = utility_coefficients(d)
    A = np.zeros((len(bargain_ids), k))  # u_bargain = A @ y
    M = np.zeros((R, k))  # resource usage = M @ y
    for col, (i, r) in enumerate(idx):
        A[row_of_user[i], col] = coef[i, r] * d[i, r]  # = 1/|R_i|
        M[r, col] = d[i, r]
    D_b = D[mask]
    w = scenario.weights[mask] / scenario.weights[mask].sum()
    cap_free = C - base.sum(axis=0)  # capacity left for bargaining users

    def objective(y: np.ndarray) -> float:
        gains = np.maximum(A @ y - D_b, 1e-300)
        return float(-np.sum(w * np.log(gains)))

    def gradient(y: np.ndarray) -> np.ndarray:
        gains = np.maximum(A @ y - D_b, 1e-300)
        return -(A.T @ (w / gains))

    # ---- strictly feasible starting point --------------------------------
    y_b = b_free / d_free
    room = d_free - b_free
    y0 = y_b.copy()
    for r in range(R):
        cols = idx[:, 1] == r
        need = room[cols].sum()
        if need > 0:
            spare = cap_free[r] - b_free[cols].sum()
            theta = min(1.0, 0.5 * spare / need)
            y0[cols] = y_b[cols] + theta * room[cols] / d_free[cols]

    lb, ub = np.zeros(k), np.ones(k)
    # Capacity rows are divided by C[r] so every constraint is O(1).
    M_scaled = M / C[:, None]
    cap_scaled = cap_free / C

    best = None
    for method in ("SLSQP", "trust-constr"):
        try:
            if method == "SLSQP":
                constraints = [
                    {"type": "ineq", "fun": lambda y: cap_scaled - M_scaled @ y, "jac": lambda y: -M_scaled},
                    {"type": "ineq", "fun": lambda y: A @ y - D_b - IR_MARGIN, "jac": lambda y: A},
                ]
                res = minimize(
                    objective, y0, jac=gradient, method="SLSQP",
                    bounds=Bounds(lb, ub), constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": 1000},
                )
            else:
                lin = LinearConstraint(
                    np.vstack([M_scaled, A]),
                    np.concatenate([np.full(R, -np.inf), D_b + IR_MARGIN]),
                    np.concatenate([cap_scaled, np.full(len(D_b), np.inf)]),
                )
                res = minimize(
                    objective, y0, jac=gradient, method="trust-constr",
                    bounds=Bounds(lb, ub), constraints=[lin],
                    options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 5000},
                )
        except Exception as exc:  # solver crashed: record and try the next one
            details["attempts"].append({"method": method, "success": False, "message": f"solver error: {exc}"})
            continue

        y = np.clip(res.x, 0.0, 1.0)
        x = base.copy()
        x[free] = y * d_free
        # Remove tiny numerical capacity overshoot (within tolerance only).
        for r in range(R):
            used = x[:, r].sum()
            if C[r] < used <= C[r] * (1 + FEAS_TOL):
                cols = free[:, r]
                x[cols, r] *= (cap_free[r]) / x[cols, r].sum()

        issues = _verify_solution(x, scenario, mask, D)
        attempt = {
            "method": method,
            "success": bool(res.success) and not issues,
            "solver_success": bool(res.success),
            "message": str(res.message),
            "iterations": int(getattr(res, "nit", getattr(res, "niter", -1))),
            "issues": issues,
        }
        details["attempts"].append(attempt)
        if attempt["success"]:
            best = (x, method, attempt)
            break

    if best is None:
        last = details["attempts"][-1] if details["attempts"] else {}
        reason = "; ".join(last.get("issues", [])) or last.get("message", "unknown error")
        details["method"] = "failed"
        details["log_nash_product"] = float("nan")
        return AllocationResult(
            NASH, None, False,
            f"Nash bargaining optimization FAILED (tried SLSQP and trust-constr): {reason}",
            details,
        )

    x, method, attempt = best
    u = user_utilities(x, d)
    details["method"] = method
    details["iterations"] = attempt["iterations"]
    details["log_nash_product"] = weighted_log_nash_product(u, scenario)
    return AllocationResult(
        NASH, x, True,
        f"Optimal weighted Nash bargaining allocation found with {method} "
        f"in {attempt['iterations']} iterations; feasibility and Pareto checks passed.",
        details,
    )
