import io

import numpy as np
import pandas as pd
import pytest

from src.algorithms import run_all_algorithms
from src.experiments import (
    PRESETS,
    capacity_sweep,
    disagreement_sweep,
    preset_scenario,
    preset_users_frame,
    weight_sweep,
)
from src.export import scenario_frame, to_csv_bytes
from src.metrics import all_user_tables, comparison_table, jains_fairness_index, summary_metrics
from src.models import ALGORITHMS, NASH, AllocationResult, Scenario


# ---------------- Jain's index --------------------------------------------
def test_jain_perfectly_fair():
    assert jains_fairness_index([0.5, 0.5, 0.5]) == pytest.approx(1.0)


def test_jain_one_user_gets_everything():
    assert jains_fairness_index([1, 0, 0, 0]) == pytest.approx(0.25)


def test_jain_known_value_and_nan_handling():
    # (1 + 0.5)^2 / (2 * (1 + 0.25)) = 2.25 / 2.5 = 0.9
    assert jains_fairness_index([1, 0.5, np.nan]) == pytest.approx(0.9)
    assert np.isnan(jains_fairness_index([0, 0]))
    assert np.isnan(jains_fairness_index([]))


# ---------------- summary metrics -----------------------------------------
def test_summary_metrics_values():
    s = Scenario(capacities=[10, 20], demands=[[4, 10], [10, 10]], priorities=[1, 1], weights=[1, 1])
    res = AllocationResult("Manual", np.array([[4, 10], [5, 10]], dtype=float), True, "")
    row = summary_metrics(s, res)
    assert row["CPU utilization"] == pytest.approx(0.9)
    assert row["RAM utilization"] == pytest.approx(1.0)
    assert row["CPU demand satisfaction"] == pytest.approx(9 / 14)
    assert row["Mean utility"] == pytest.approx((1.0 + 0.75) / 2)
    assert row["Min utility"] == pytest.approx(0.75)
    assert row["Pareto efficient"] is False  # 1 CPU idle while user 2 wants more


def test_failed_result_gives_nan_metrics():
    s = Scenario(capacities=[10, 20], demands=[[4, 10]], priorities=[1], weights=[1])
    row = summary_metrics(s, AllocationResult(NASH, None, False, "FAILED"))
    assert row["Status"] == "FAILED"
    assert np.isnan(row["Jain's fairness index"])


def test_comparison_and_user_tables_cover_all_algorithms():
    s = preset_scenario("Balanced baseline")
    results = run_all_algorithms(s)
    table = comparison_table(s, results)
    assert list(table["Algorithm"]) == list(ALGORITHMS)
    users = all_user_tables(s, results)
    assert len(users) == 3 * s.n_users
    assert users["Utility"].between(0, 1 + 1e-9).all()


# ---------------- presets & sweeps ----------------------------------------
@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_runs_and_nbs_succeeds(name):
    s = preset_scenario(name)
    results = run_all_algorithms(s)
    assert results[NASH].success, results[NASH].message
    assert len(preset_users_frame(name)) == s.n_users
    for res in results.values():
        assert np.all(res.allocation.sum(axis=0) <= s.capacities * (1 + 1e-7))
        assert np.all(res.allocation <= s.demands + 1e-7)


def test_capacity_sweep_utilization_and_satisfaction_trends():
    s = preset_scenario("Balanced baseline")
    df = capacity_sweep(s, [0.25, 0.5, 1.0, 1.5])
    assert len(df) == 4 * 3
    nbs = df[df["Algorithm"] == NASH].sort_values("Capacity / demand")
    assert (nbs["Status"] == "OK").all()
    # more capacity never lowers mean utility under NBS
    assert np.all(np.diff(nbs["Mean utility"].to_numpy()) >= -1e-6)
    # at capacity >= demand, everybody is fully satisfied
    assert nbs.iloc[-1]["Mean utility"] == pytest.approx(1.0, abs=1e-5)


def test_weight_sweep_is_monotone_for_selected_user():
    s = preset_scenario("Resource scarcity")
    df = weight_sweep(s, 0, [0.5, 1, 2, 4])
    u0 = df[df["User"] == s.user_names[0]]["Utility"].to_numpy()
    assert np.all(np.diff(u0) >= -1e-6)


def test_disagreement_sweep_runs():
    s = preset_scenario("Balanced baseline")
    df = disagreement_sweep(s, [0.0, 0.3, 0.6, 0.9])
    assert (df["Status"] == "OK").all()
    assert len(df) == 4 * s.n_users


# ---------------- export --------------------------------------------------
def test_csv_export_round_trip():
    s = preset_scenario("Unequal demands")
    results = run_all_algorithms(s)
    raw = to_csv_bytes(comparison_table(s, results))
    back = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    assert list(back["Algorithm"]) == list(ALGORITHMS)
    inputs = scenario_frame(s)
    assert inputs.iloc[-1]["User"] == "SYSTEM CAPACITY"
