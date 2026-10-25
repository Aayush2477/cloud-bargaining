"""End-to-end tests of the Streamlit UI using Streamlit's built-in AppTest.

These run the real app.py script headlessly and interact with widgets.
"""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [m.value for m in at.success] + [m.value for m in at.error]
    parts += [m.value for m in at.info] + [m.value for m in at.warning] + [m.value for m in at.caption]
    return "\n".join(str(p) for p in parts)


def test_app_starts_and_nbs_succeeds():
    at = run_app()
    assert any("Optimal weighted Nash bargaining allocation" in s.value for s in at.success)
    labels = [m.label for m in at.metric]
    assert labels.count("Jain's fairness") == 3


def test_load_each_preset_from_sidebar():
    at = run_app()
    for name in at.sidebar.selectbox[0].options:
        at.sidebar.selectbox[0].set_value(name).run()
        at.sidebar.button[0].click().run()  # "Load preset"
        assert not at.exception, (name, at.exception)
        assert at.session_state.loaded_preset == name
        assert not at.error, [e.value for e in at.error]


def test_changing_number_of_users_resizes_table():
    at = run_app()
    at.sidebar.number_input(key="n_users").set_value(7).run()
    assert not at.exception
    assert len(at.session_state.users_base) == 7
    at.sidebar.number_input(key="n_users").set_value(2).run()
    assert len(at.session_state.users_base) == 2
    assert not at.exception


def test_invalid_capacity_shows_error_and_no_results():
    at = run_app()
    at.sidebar.number_input(key="cpu_cap").set_value(0.0).run()
    assert not at.exception
    assert any("Capacity must be greater than zero" in e.value for e in at.error)
    assert len(at.metric) == 0


def test_disagreement_slider_and_reset():
    at = run_app()
    at.sidebar.slider(key="delta").set_value(0.5).run()
    assert not at.exception
    assert any("Optimal weighted Nash" in s.value for s in at.success)
    at.sidebar.button[1].click().run()  # Reset
    assert at.session_state.delta == 0.0


def test_experiment_lab_preset_and_load_button():
    at = run_app()
    at.selectbox(key="lab_choice").set_value("Unequal priorities").run()
    assert not at.exception
    load = [b for b in at.button if b.label == "Load this preset into the Dashboard"]
    assert load
    load[0].click().run()
    assert not at.exception
    assert at.session_state.loaded_preset == "Unequal priorities"
    at.selectbox(key="cap_metric").set_value("Min utility").run()
    at.slider(key="weight_max").set_value(12.0).run()
    assert not at.exception
