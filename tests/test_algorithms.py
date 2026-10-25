import numpy as np
import pytest

from src.algorithms import equal_allocation, priority_allocation, run_all_algorithms, water_fill
from src.models import ALGORITHMS, Scenario


def scen(capacities, demands, priorities=None, weights=None, delta=0.0):
    n = len(demands)
    return Scenario(
        capacities=capacities,
        demands=demands,
        priorities=priorities if priorities is not None else [1] * n,
        weights=weights if weights is not None else [1] * n,
        disagreement_fraction=delta,
    )


# ---------------- water-filling -------------------------------------------
def test_water_fill_equal_split_when_all_demands_large():
    assert np.allclose(water_fill(9, [10, 10, 10]), [3, 3, 3])


def test_water_fill_redistributes_unused_share():
    # User 0 needs only 2, the other two share the remaining 8 equally.
    assert np.allclose(water_fill(10, [2, 10, 10]), [2, 4, 4])


def test_water_fill_weighted():
    assert np.allclose(water_fill(12, [100, 100], weights=[1, 2]), [4, 8])


def test_water_fill_abundant_gives_full_demand():
    assert np.allclose(water_fill(100, [3, 4, 0]), [3, 4, 0])


def test_water_fill_never_exceeds_capacity_or_demand():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = rng.uniform(0, 10, size=6) * (rng.random(6) > 0.2)
        w = rng.uniform(0.1, 5, size=6)
        cap = rng.uniform(0.1, 40)
        x = water_fill(cap, d, w)
        assert np.all(x >= -1e-12) and np.all(x <= d + 1e-9)
        assert x.sum() <= cap + 1e-9
        assert x.sum() == pytest.approx(min(cap, d.sum()), abs=1e-8)


# ---------------- equal allocation ----------------------------------------
def test_equal_allocation_caps_at_demand_and_leaves_idle_capacity():
    res = equal_allocation(scen([30, 60], [[2, 40], [20, 40], [20, 40]]))
    assert res.success
    # CPU share = 10 per user; user 0 only needs 2 -> 8 cores idle.
    assert np.allclose(res.allocation[:, 0], [2, 10, 10])
    assert np.allclose(res.allocation[:, 1], [20, 20, 20])


# ---------------- priority allocation -------------------------------------
def test_priority_serves_highest_priority_first():
    res = priority_allocation(scen([10, 10], [[8, 8], [8, 8]], priorities=[1, 5]))
    assert np.allclose(res.allocation, [[2, 2], [8, 8]])


def test_priority_ties_are_shared_equally():
    res = priority_allocation(scen([10, 10], [[8, 8], [8, 8], [8, 8]], priorities=[3, 3, 1]))
    assert np.allclose(res.allocation[0], [5, 5])
    assert np.allclose(res.allocation[1], [5, 5])
    assert np.allclose(res.allocation[2], [0, 0])


def test_priority_uses_all_capacity_when_demand_exceeds_it():
    res = priority_allocation(scen([10, 30], [[4, 10], [4, 10], [4, 20]], priorities=[1, 2, 3]))
    assert res.allocation.sum(axis=0) == pytest.approx([10, 30])


def test_run_all_returns_three_algorithms():
    results = run_all_algorithms(scen([10, 10], [[5, 5], [10, 2]]))
    assert list(results) == list(ALGORITHMS)
    for res in results.values():
        assert res.success
        assert res.allocation.shape == (2, 2)
        assert np.all(res.allocation.sum(axis=0) <= np.array([10, 10]) + 1e-9)
