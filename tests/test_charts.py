import plotly.graph_objects as go

from src import charts
from src.algorithms import run_all_algorithms
from src.experiments import capacity_sweep, disagreement_sweep, preset_scenario, weight_sweep
from src.metrics import comparison_table
from src.models import NASH, AllocationResult


def test_all_charts_build_for_every_preset():
    for name in ["Balanced baseline", "Resource scarcity", "Abundant resources"]:
        s = preset_scenario(name)
        results = run_all_algorithms(s)
        table = comparison_table(s, results)
        figs = [
            charts.allocation_chart(s, results),
            charts.utility_chart(s, results),
            charts.fairness_chart(table),
            charts.utilization_chart(table, s),
            charts.demand_satisfaction_chart(table, s),
            charts.nash_product_chart(table),
        ]
        assert all(isinstance(f, go.Figure) and len(f.data) > 0 for f in figs)


def test_failed_nbs_is_not_drawn():
    s = preset_scenario("Balanced baseline")
    results = run_all_algorithms(s)
    results[NASH] = AllocationResult(NASH, None, False, "FAILED (simulated)")
    fig = charts.utility_chart(s, results)
    assert NASH not in {trace.name for trace in fig.data}
    table = comparison_table(s, results)
    assert NASH not in list(charts.fairness_chart(table).data[0].x)


def test_sweep_charts_build():
    s = preset_scenario("Balanced baseline")
    assert charts.capacity_sweep_chart(capacity_sweep(s, [0.5, 1.0]), "Mean utility").data
    assert charts.per_user_sweep_chart(weight_sweep(s, 0, [1, 2]), "Weight of selected user", "t").data
    assert charts.per_user_sweep_chart(disagreement_sweep(s, [0, 0.5]), "Disagreement fraction", "t").data
