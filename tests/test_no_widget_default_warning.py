"""
No "created with a default value but also had its value set via the Session State
API" warning on Page 1 or Page 2 (Q-41, fixed 2026-10-07).

Streamlit prints that warning when a widget gets a `value=` / `index=` default AND its
key was written through st.session_state earlier in the same script run - which is
how Page 1 seeds its widget keys (`initialize_widget_keys`) and how the decision tabs
restore theirs (`restore_widget_from_storage`, `restore_from_persistent_storage`).
Those widgets are now driven by the session-state key only.

Streamlit shows the warning once per PROCESS (a module-level flag), so checking the
rendered page alone would pass vacuously once any earlier test had tripped it. The
test therefore resets that flag and also records every widget call that WOULD warn
(a new session-state value plus a non-None default), through the same policy function
Streamlit uses.
"""
from pathlib import Path

import pytest
from streamlit.elements.lib import policies
from streamlit.runtime.state import get_session_state
from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).resolve().parents[1] / "app_enhanced_new.py")
WARNING_TEXT = "was created with a default value but also had its value set via the Session State API"


@pytest.fixture
def offenders(monkeypatch):
    found = []
    original = policies.check_session_state_rules

    def spy(default_value, key, writes_allowed=True):
        if key is not None and default_value is not None:
            try:
                if get_session_state().is_new_state_value(key):
                    found.append(key)
            except Exception:   # no script-run context
                pass
        return original(default_value, key, writes_allowed)

    monkeypatch.setattr(policies, "check_session_state_rules", spy)
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    return found


def _no_warning(at, step, offenders):
    assert not at.exception, f"{step}: {[e.value for e in at.exception]}"
    shown = [str(w.value) for w in at.warning if WARNING_TEXT in str(w.value)]
    assert not shown, f"{step}: {shown}"
    assert not offenders, f"{step}: widgets with a default AND a Session-State value: {sorted(set(offenders))}"


def _click(at, label_part):
    button = [b for b in at.button if label_part in b.label]
    assert button, f"no button containing {label_part!r}"
    button[0].click().run()


def test_page1_and_page2_widgets_are_driven_by_session_state_only(offenders):
    at = AppTest.from_file(APP_FILE, default_timeout=300)
    at.run()
    _no_warning(at, "Page 1, first render", offenders)
    assert at.number_input(key="n_agents_input").value == at.session_state["n_agents"]

    # Page 1 edits keep working exactly as before (on_change mirrors into the canonical keys)
    at.number_input(key="n_agents_input").set_value(250).run()
    _no_warning(at, "after changing the number of agents", offenders)
    assert at.session_state["n_agents"] == 250
    at.radio(key="page1_simulation_mode").set_value("Monte-Carlo Study").run()
    _no_warning(at, "Monte-Carlo mode", offenders)
    at.number_input(key="n_runs_input").set_value(12).run()
    assert at.session_state["n_runs"] == 12
    for dist in ("generalised_gamma", "dagum", "lognormal"):
        at.selectbox(key="page1_income_distribution").set_value(dist).run()
        _no_warning(at, f"income distribution {dist}", offenders)
    at.radio(key="page1_simulation_mode").set_value("Single Run").run()

    # Page 2 with the four configurable decisions selected (all decision tabs render)
    _click(at, "Decision Parameters")
    _no_warning(at, "Page 2, first render", offenders)
    at.multiselect("page2_manual_multiselect").set_value(
        ["disclose_income", "disclose_documents", "donation_default",
         "rejected_transaction_defaults"]).run()
    _no_warning(at, "Page 2 with decisions selected", offenders)

    # values saved into the tab persistence dicts are restored on every rerun
    at.checkbox(key="di_tab_sigma_in_copula").check().run()
    at.slider(key="di_tab_sigma_coefficient_stochastic").set_value(0.5).run()
    at.slider(key="purchase_vs_bid_default_probability_y").set_value(0.3).run()
    at.run()
    _no_warning(at, "after tab edits and a rerun", offenders)
    assert at.slider(key="di_tab_sigma_coefficient_stochastic").value == 0.5
    assert at.slider(key="purchase_vs_bid_default_probability_y").value == 0.3

    # round trip through Page 1: every value survives, still no warning
    _click(at, "Back to Common Parameters")
    _no_warning(at, "back on Page 1", offenders)
    assert at.number_input(key="n_agents_input").value == 250
    _click(at, "Decision Parameters")
    _no_warning(at, "Page 2 again", offenders)
    assert at.checkbox(key="di_tab_sigma_in_copula").value is True
    assert at.slider(key="di_tab_sigma_coefficient_stochastic").value == 0.5
    assert at.slider(key="purchase_vs_bid_default_probability_y").value == 0.3
