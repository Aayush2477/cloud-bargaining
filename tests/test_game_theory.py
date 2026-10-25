"""Tests of the Nash Bargaining Solution against known closed-form answers,
the NBS axioms, and brute-force optimality checks."""

import numpy as np
import pytest

import src.game_theory as gt
from src.algorithms import run_all_algorithms, water_fill
from src.game_theory import (
    disagreement_allocation,
    disagreement_point,
    nash_bargaining_solution,
    user_utilities,
    utility_coefficients,
    weighted_log_nash_product,
)
from src.models import NASH, Scenario


def scen(capacities, demands, weights=None, delta=0.0, priorities=None):
    n = len(demands)
    return Scenario(
        capacities=capacities,
        demands=demands,
        priorities=priorities if priorities is not None else [1] * n,
        weights=weights if weights is not None else [1] * n,
        disagreement_fraction=delta,
    )


# ---------------- utilities & disagreement point --------------------------
def test_utility_is_average_satisfaction_over_demanded_resources():
    d = np.array([[10.0, 20.0], [0.0, 8.0], [0.0, 0.0]])
    x = np.array([[5.0, 20.0], [0.0, 2.0], [0.0, 0.0]])
    u = user_utilities(x, d)
    assert u[0] == pytest.approx((0.5 + 1.0) / 2)
    assert u[1] == pytest.approx(0.25)  # only RAM counts for user 1
    assert np.isnan(u[2])  # zero demand -> undefined
    assert np.allclose(utility_coefficients(d)[1], [0.0, 1 / 8])


def test_disagreement_point_zero_when_delta_zero():
    s = scen([10, 10], [[5, 5], [10, 10]])
    assert np.allclose(disagreement_allocation(s), 0)
    assert np.allclose(disagreement_point(s), 0)


def test_disagreement_allocation_is_feasible_and_capped():
    s = scen([10, 40], [[1, 30], [10, 30], [0, 0]], delta=0.5)
    b = disagreement_allocation(s)
    # 2 active users -> share = 0.5 * C / 2 = (2.5, 10)
    assert np.allclose(b, [[1, 10], [2.5, 10], [0, 0]])
    assert np.all(b.sum(axis=0) < s.capacities)


# ---------------- closed-form solutions -----------------------------------
def test_single_resource_symmetric_split():
    res = nash_bargaining_solution(scen([12], [[100], [100], [100]]))
    assert res.success
    assert np.allclose(res.allocation[:, 0], [4, 4, 4], atol=1e-5)


def test_single_resource_matches_weighted_water_filling():
    """With one resource and D = 0, maximizing sum w_i log(x_i/d_i) gives
    x_i = min(d_i, lambda * w_i): weighted water-filling."""
    rng = np.random.default_rng(42)
    for _ in range(25):
        n = rng.integers(2, 7)
        d = rng.uniform(1, 20, size=n)
        w = rng.uniform(0.2, 5, size=n)
        cap = rng.uniform(0.2, 1.2) * d.sum()
        res = nash_bargaining_solution(scen([cap], d[:, None], weights=w))
        assert res.success, res.message
        expected = water_fill(cap, d, w)
        assert np.allclose(res.allocation[:, 0], expected, atol=1e-4 * max(1, cap))


def test_single_resource_with_disagreement_point():
    """With fallback shares b, the solution is b + water-filling of the rest."""
    d = np.array([2.0, 10.0, 30.0])
    w = np.array([1.0, 2.0, 1.0])
    s = scen([24.0], d[:, None], weights=w, delta=0.3)
    res = nash_bargaining_solution(s)
    assert res.success
    b = disagreement_allocation(s)[:, 0]
    expected = b + water_fill(24.0 - b.sum(), d - b, w)
    assert np.allclose(res.allocation[:, 0], expected, atol=1e-4)


def test_abundant_resources_give_everyone_full_demand():
    d = [[4, 16], [8, 32], [2, 4]]
    res = nash_bargaining_solution(scen([100, 100], d))
    assert res.success
    assert np.allclose(res.allocation, d, atol=1e-5)


# ---------------- NBS axioms ----------------------------------------------
def test_symmetry_identical_users_get_identical_utility():
    res = nash_bargaining_solution(scen([10, 30], [[8, 20], [8, 20], [8, 20]]))
    u = user_utilities(res.allocation, np.array([[8, 20]] * 3, dtype=float))
    assert np.allclose(u, u[0], atol=1e-5)


def test_scale_invariance_changing_units_does_not_change_utilities():
    """GB -> MB (x1024) must not change anyone's utility."""
    base = scen([32, 128], [[8, 32], [12, 48], [16, 64], [6, 40]], weights=[2, 1, 1, 1])
    scaled = scen([32, 128 * 1024], [[8, 32 * 1024], [12, 48 * 1024], [16, 64 * 1024], [6, 40 * 1024]],
                  weights=[2, 1, 1, 1])
    r1, r2 = nash_bargaining_solution(base), nash_bargaining_solution(scaled)
    u1 = user_utilities(r1.allocation, base.demands)
    u2 = user_utilities(r2.allocation, scaled.demands)
    assert np.allclose(u1, u2, atol=1e-5)


def test_individual_rationality_and_pareto_efficiency():
    s = scen([20, 50], [[10, 30], [12, 10], [5, 40], [8, 8]], weights=[1, 3, 1, 2], delta=0.4)
    res = nash_bargaining_solution(s)
    assert res.success
    u = user_utilities(res.allocation, s.demands)
    assert np.all(u > disagreement_point(s))
    assert gt._verify_solution(res.allocation, s, gt.bargaining_users(s), disagreement_point(s)) == []


def test_higher_weight_never_hurts_that_user():
    utils = []
    for w0 in [0.5, 1, 2, 4, 8]:
        s = scen([16, 64], [[16, 64], [16, 64], [8, 40]], weights=[w0, 1, 1])
        res = nash_bargaining_solution(s)
        utils.append(user_utilities(res.allocation, s.demands)[0])
    assert all(b >= a - 1e-6 for a, b in zip(utils, utils[1:]))


# ---------------- optimality ----------------------------------------------
def test_nbs_beats_random_feasible_allocations():
    rng = np.random.default_rng(7)
    s = scen([24, 80], [[10, 40], [6, 50], [12, 20], [5, 30]], weights=[1, 2, 1, 1.5], delta=0.2)
    res = nash_bargaining_solution(s)
    best = weighted_log_nash_product(user_utilities(res.allocation, s.demands), s)
    b = disagreement_allocation(s)
    for _ in range(3000):
        # random feasible allocation that keeps everyone above their fallback share
        y = b + rng.random(s.demands.shape) * (s.demands - b)
        used = y.sum(axis=0)
        y = y * np.minimum(1.0, s.capacities / used)[None, :]
        val = weighted_log_nash_product(user_utilities(y, s.demands), s)
        assert val <= best + 1e-7


def test_nbs_has_highest_nash_product_among_algorithms():
    s = scen([32, 128], [[8, 32], [12, 48], [16, 64], [6, 40], [4, 16]],
             weights=[4, 2, 1, 1, 1], priorities=[3, 5, 2, 1, 1], delta=0.1)
    results = run_all_algorithms(s)
    values = {
        name: weighted_log_nash_product(user_utilities(r.allocation, s.demands), s)
        for name, r in results.items()
    }
    assert values[NASH] >= max(values.values()) - 1e-9


# ---------------- edge cases ----------------------------------------------
def test_zero_demand_users_get_nothing():
    s = scen([10, 10], [[0, 0], [5, 20], [5, 0]])
    res = nash_bargaining_solution(s)
    assert res.success
    assert np.allclose(res.allocation[0], 0)
    assert res.allocation[2, 1] == 0  # no RAM demand -> no RAM
    assert res.allocation.sum(axis=0) == pytest.approx([10, 10], abs=1e-6)


def test_all_zero_demands_is_trivial_success():
    res = nash_bargaining_solution(scen([10, 10], [[0, 0], [0, 0]]))
    assert res.success
    assert np.allclose(res.allocation, 0)
    assert "No bargaining needed" in res.message


def test_users_fully_satisfied_by_fallback_do_not_bargain():
    s = scen([10, 10], [[1, 1], [50, 50]], delta=0.5)  # fallback 2.5 >= demand 1
    res = nash_bargaining_solution(s)
    assert res.success
    assert not res.details["bargaining_users"][0]
    assert np.allclose(res.allocation[0], [1, 1])
    assert np.allclose(res.allocation[1], [9, 9], atol=1e-5)


def test_single_user_takes_everything_it_can():
    res = nash_bargaining_solution(scen([4, 64], [[8, 32]]))
    assert res.success
    assert np.allclose(res.allocation, [[4, 32]], atol=1e-6)


def test_optimizer_failure_is_reported_not_hidden(monkeypatch):
    """If both solvers fail, the result must be FAILED with no allocation."""

    class Fake:
        x = None
        success = False
        message = "simulated divergence"
        nit = 0

    def broken_minimize(fun, x0, **kwargs):
        fake = Fake()
        fake.x = np.full_like(x0, np.nan)
        return fake

    monkeypatch.setattr(gt, "minimize", broken_minimize)
    res = nash_bargaining_solution(scen([10, 10], [[8, 8], [8, 8]]))
    assert not res.success
    assert res.allocation is None
    assert "FAILED" in res.message
    assert [a["method"] for a in res.details["attempts"]] == ["SLSQP", "trust-constr"]


def test_solver_claiming_success_with_bad_point_is_rejected(monkeypatch):
    """A solver that says 'success' but returns a wasteful point fails verification."""

    class Fake:
        success = True
        message = "pretend converged"
        nit = 1

    def lazy_minimize(fun, x0, **kwargs):
        fake = Fake()
        fake.x = np.zeros_like(x0) + 1e-3  # leaves almost all capacity idle
        return fake

    monkeypatch.setattr(gt, "minimize", lazy_minimize)
    res = nash_bargaining_solution(scen([10, 10], [[8, 8], [8, 8]]))
    assert not res.success
    assert "Pareto" in res.message


def test_solver_exception_is_caught(monkeypatch):
    def crash(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gt, "minimize", crash)
    res = nash_bargaining_solution(scen([10, 10], [[8, 8], [8, 8]]))
    assert not res.success and res.allocation is None
    assert "boom" in res.message


def test_trust_constr_fallback_is_used_when_slsqp_fails(monkeypatch):
    real = gt.minimize

    def slsqp_breaks(fun, x0, method=None, **kwargs):
        if method == "SLSQP":
            raise RuntimeError("SLSQP unavailable")
        return real(fun, x0, method=method, **kwargs)

    monkeypatch.setattr(gt, "minimize", slsqp_breaks)
    res = nash_bargaining_solution(scen([12], [[100], [100], [100]]))
    assert res.success
    assert res.details["method"] == "trust-constr"
    assert np.allclose(res.allocation[:, 0], 4, atol=1e-3)
