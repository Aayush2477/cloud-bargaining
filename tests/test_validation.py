import numpy as np
import pytest

from src.algorithms import equal_allocation
from src.models import Scenario
from src.validation import InvalidScenarioError, validate_scenario


def make(**overrides):
    data = dict(
        capacities=[10, 20],
        demands=[[4, 8], [6, 12]],
        priorities=[1, 2],
        weights=[1, 1],
        user_names=["A", "B"],
        disagreement_fraction=0.0,
    )
    data.update(overrides)
    return Scenario(**data)


def test_valid_scenario_has_no_errors():
    report = validate_scenario(make())
    assert report.ok
    assert report.errors == []


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"capacities": [0, 20]}, "greater than zero"),
        ({"capacities": [-5, 20]}, "greater than zero"),
        ({"capacities": [np.nan, 20]}, "finite"),
        ({"demands": [[-1, 8], [6, 12]]}, "negative"),
        ({"demands": [[np.nan, 8], [6, 12]]}, "finite"),
        ({"demands": [[4, 8, 1], [6, 12, 1]]}, "one column per resource"),
        ({"priorities": [0, 2]}, "priority"),
        ({"priorities": [1]}, "priorities"),
        ({"weights": [1, -1]}, "weight"),
        ({"weights": [1, np.inf]}, "weight"),
        ({"disagreement_fraction": 1.0}, "Disagreement"),
        ({"disagreement_fraction": -0.1}, "Disagreement"),
        ({"user_names": ["A", "A"]}, "unique"),
        ({"user_names": ["A", " "]}, "empty"),
    ],
)
def test_invalid_inputs_are_rejected(overrides, fragment):
    report = validate_scenario(make(**overrides))
    assert not report.ok
    assert any(fragment.lower() in e.lower() for e in report.errors), report.errors


def test_algorithms_refuse_invalid_scenarios():
    with pytest.raises(InvalidScenarioError):
        equal_allocation(make(capacities=[0, 20]))


def test_warnings_for_zero_demand_and_oversized_demand():
    report = validate_scenario(make(demands=[[0, 0], [15, 12]]))
    assert report.ok
    text = " ".join(report.warnings)
    assert "zero demand" in text
    assert "more CPU than the whole system" in text


def test_all_zero_demand_is_a_warning_not_an_error():
    report = validate_scenario(make(demands=[[0, 0], [0, 0]]))
    assert report.ok
    assert any("nothing to allocate" in w for w in report.warnings)
