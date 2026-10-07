"""
Widgets show - in the BROWSER - the value the script runs with (regression from
e51c3fc, fixed 2026-10-07).

e51c3fc removed ``value=`` / ``index=`` from ~37 widgets whose value lives in a
session-state key (to silence Streamlit's "created with a default value but also had
its value set via the Session State API" warning, Q-41). Streamlit sends a key's
value to the browser only in the run in which the key is written through
``st.session_state``; a key seeded in an EARLIER run (or kept from one) reaches a
browser that draws the widget for the first time only as the element's ``default=``
- which, without ``value=``, is the widget's built-in default (min / False / option
0). The browser shows it and sends it back, and the run uses it: the Donation tab's
σ coefficient and anchor weight became 0, its Research tick box unticked, the
Overview's P(Y) sliders 0, and a Donation run's mean was computed with anchor 0.

AppTest cannot see this - it reads the server-side value. ``tests/browser_sim.py``
models the frontend's widget-state manager on top of AppTest; these tests drive the
app through it, assert after EVERY run that the browser shows what the script
holds (``mismatches() == []``), and check the concrete defaults the controller saw
go wrong. All of them fail on a39d359 and pass with ``app/state/widgets.py``.
"""
import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from browser_sim import BrowserSim
from test_no_widget_default_warning import offenders  # noqa: F401  (fixture: Q-41 warning spy)

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = str(ROOT / "app_enhanced_new.py")

DEFAULT_PROBABILITY_KEYS = (
    "disclose_income_default_probability_y",
    "disclose_documents_default_probability_y",
    "purchase_vs_bid_default_probability_y",
)


def _start():
    sim = BrowserSim(AppTest.from_file(APP_FILE, default_timeout=300))
    sim.run()
    _consistent(sim, "Page 1, first render")
    return sim


def _consistent(sim, step):
    at = sim.at
    assert not at.exception, f"{step}: {[e.value for e in at.exception]}"
    bad = sim.mismatches()
    assert not bad, (f"{step}: the browser shows a different value than the script holds "
                     f"(key, browser, script): {bad}")


def _to_page2(sim):
    sim.click("Decision Parameters")
    _consistent(sim, "Page 2")


def _select_all(sim, on=True):
    sim.set("page2_select_all_checkbox", on)
    _consistent(sim, f"Select All = {on}")


def test_fresh_session_overview_and_donation_tab_show_their_defaults(offenders):
    """The controller's browser repro, step by step."""
    sim = _start()
    _to_page2(sim)

    # Overview with no decision selected: default-decision controls
    for key in DEFAULT_PROBABILITY_KEYS:
        assert sim.shown(key) == pytest.approx(0.5), key
    assert sim.shown("donation_default_default_value") == pytest.approx(0.10)
    assert sim.shown("final_donation_rate_default_value") == pytest.approx(0.10)
    assert sim.shown("vendor_choice_weights_default_param_price") is True

    _select_all(sim)
    assert sim.shown("tab_sigma_coefficient_stochastic") == pytest.approx(1.0)
    assert sim.shown("tab_anchor_weight") == pytest.approx(0.75)
    assert sim.shown("tab_sigma_in_research") is True
    assert sim.shown("tab_sigma_in_copula") is False

    # the next rerun sends the browser's values: the run must still use the defaults
    sim.run()
    _consistent(sim, "rerun after Select All")
    ss = sim.at.session_state
    assert ss["anchor_observed_weight"] == pytest.approx(0.75)
    assert ss["sigma_coefficient"] == pytest.approx(1.0)
    assert ss["sigma_in_research"] is True
    assert ss["disclose_income_default_probability_y"] == pytest.approx(0.5)
    assert ss["donation_default_default_value"] == pytest.approx(0.10)
    assert not offenders, sorted(set(offenders))


def test_user_values_survive_tab_and_page_switches(offenders):
    sim = _start()

    # Page 1 values, including one whose widget disappears with its distribution
    sim.set("n_agents_input", 250)
    sim.set("lognormal_mu_input", 11.0)
    sim.set("page1_income_distribution", "dagum")
    _consistent(sim, "dagum")
    sim.set("page1_income_distribution", "lognormal")
    _consistent(sim, "lognormal again")
    assert sim.shown("lognormal_mu_input") == pytest.approx(11.0)

    _to_page2(sim)
    sim.set("purchase_vs_bid_default_probability_y", 0.3)
    _consistent(sim, "Overview default edited")

    _select_all(sim)
    sim.set("tab_anchor_weight", 0.6)
    sim.set("tab_sigma_coefficient_stochastic", 1.3)
    sim.set("tab_sigma_in_copula", True)
    sim.set("di_wopb_widget", 0.4)
    _consistent(sim, "decision tabs edited")

    # Page 2 -> Page 1 -> Page 2 (every Page-2 widget is dropped and redrawn)
    sim.click("Back to Common Parameters")
    _consistent(sim, "back on Page 1")
    assert sim.shown("n_agents_input") == 250
    assert sim.shown("lognormal_mu_input") == pytest.approx(11.0)
    _to_page2(sim)
    assert sim.shown("tab_anchor_weight") == pytest.approx(0.6)
    assert sim.shown("tab_sigma_coefficient_stochastic") == pytest.approx(1.3)
    assert sim.shown("tab_sigma_in_copula") is True
    assert sim.shown("di_wopb_widget") == pytest.approx(0.4)

    # unselect every decision (the tabs disappear, the Overview defaults reappear) ...
    _select_all(sim, False)
    assert sim.shown("purchase_vs_bid_default_probability_y") == pytest.approx(0.3)
    assert sim.shown("disclose_income_default_probability_y") == pytest.approx(0.5)
    # ... and select them again
    _select_all(sim, True)
    assert sim.shown("tab_anchor_weight") == pytest.approx(0.6)
    assert sim.shown("tab_sigma_coefficient_stochastic") == pytest.approx(1.3)
    assert sim.shown("tab_sigma_in_copula") is True
    assert sim.shown("di_wopb_widget") == pytest.approx(0.4)

    sim.run()
    _consistent(sim, "final rerun")
    ss = sim.at.session_state
    assert ss["anchor_observed_weight"] == pytest.approx(0.6)
    assert ss["sigma_coefficient"] == pytest.approx(1.3)
    assert ss["sigma_in_copula"] is True
    assert ss["n_agents"] == 250
    assert not offenders, sorted(set(offenders))


def test_page1_reset_restores_the_defaults(offenders):
    """Q-39: the September reset raised StreamlitAPIException at its first widget-key
    write; as an on_click callback it resets everything, to the values a fresh session
    shows (tests/test_page1_reset_defaults.py checks every widget)."""
    sim = _start()
    sim.set("n_agents_input", 280)
    sim.set("seed_input", 7)
    sim.set("platform_markup_slider", 0.3)
    sim.set("page1_population_mode", "Research Specification")
    sim.set("page1_income_distribution", "dagum")
    _consistent(sim, "Page 1 edited")

    sim.click_key("reset_page1_defaults")
    _consistent(sim, "after Reset to Default Values")
    assert sim.shown("n_agents_input") == 1000
    assert sim.shown("seed_input") == 42
    assert sim.shown("platform_markup_slider") == pytest.approx(0.1)
    assert sim.shown("page1_population_mode") == "Copula (synthetic)"
    assert sim.shown("page1_income_distribution") == "lognormal"

    sim.run()
    _consistent(sim, "rerun after reset")
    ss = sim.at.session_state
    assert ss["n_agents"] == 1000
    assert ss["seed"] == 42
    assert ss["population_mode"] == "Copula (synthetic)"
    assert ss["sim_params"].platform_markup == pytest.approx(0.1)
    assert not offenders, sorted(set(offenders))


@pytest.mark.parametrize("button_key", [
    "di_reset_btn", "di_reload_btn", "dd_reset_btn", "dd_reload_btn",
    "reset_config_btn", "reset_intercept_btn", "rtd_reset_btn",
])
def test_decision_tab_resets_reach_the_browser(button_key, offenders):
    sim = _start()
    _to_page2(sim)
    _select_all(sim)
    sim.set("di_wopb_widget", 0.4)
    sim.set("tab_anchor_weight", 0.6)
    sim.click_key(button_key)
    _consistent(sim, f"after {button_key}")
    sim.run()
    _consistent(sim, f"rerun after {button_key}")
    if button_key == "di_reset_btn":
        assert sim.shown("di_wopb_widget") == pytest.approx(0.25)
    assert not offenders, sorted(set(offenders))


# --------------------------------------------------------------------------- #
# static: every session-state-driven widget goes through stateful()
# --------------------------------------------------------------------------- #

_VALUE_ARG = {"slider": "value", "number_input": "value", "checkbox": "value",
              "toggle": "value", "radio": "index", "selectbox": "index",
              "multiselect": "default", "select_slider": "value"}


def test_every_keyed_widget_without_a_value_is_drawn_through_stateful():
    """A keyed widget that passes no value= / index= / default= takes its value from
    session state, so it must be drawn through app.state.widgets.stateful (which
    pushes that value to a browser that has not drawn the widget before)."""
    offenders = []
    for path in sorted((ROOT / "app").rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _VALUE_ARG):
                continue
            kws = {k.arg for k in node.keywords}
            if "key" in kws and _VALUE_ARG[node.func.attr] not in kws:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} "
                                 f"{ast.unparse(node.func)}(key={ast.unparse(next(k.value for k in node.keywords if k.arg == 'key'))})")
    assert not offenders, ("keyed widgets with no value= drawn directly - use "
                           "stateful(st.<widget>, ...):\n  " + "\n  ".join(offenders))


def test_entry_point_rotates_the_render_log_before_any_page():
    source = (ROOT / "app_enhanced_new.py").read_text()
    assert "begin_script_run()" in source
    assert source.index("begin_script_run()") < source.index("render_page1()")


def test_no_stateful_widget_falls_back_to_its_builtin_default(monkeypatch, offenders):
    """Every stateful() widget's key is seeded before it is drawn - otherwise the
    widget would silently start at min / False / option 0."""
    import app.state.widgets as widgets

    unseeded = []
    original = widgets.sync_widget_key

    def spy(key, initial=widgets._MISSING, signature=None):
        import streamlit as st
        if key not in st.session_state and initial is widgets._MISSING \
                and not key.endswith("_add_selector"):   # an "add item" picker, empty by design
            unseeded.append(key)
        return original(key, initial, signature)

    monkeypatch.setattr(widgets, "sync_widget_key", spy)
    sim = _start()
    sim.set("page1_simulation_mode", "Monte-Carlo Study")
    for dist in ("generalised_gamma", "dagum", "lognormal"):
        sim.set("page1_income_distribution", dist)
    sim.set("page1_simulation_mode", "Single Run")
    _to_page2(sim)
    _select_all(sim)
    for button_key in ("di_reset_btn", "dd_reset_btn", "reset_config_btn", "rtd_reset_btn"):
        sim.click_key(button_key)
    assert not unseeded, f"widgets drawn with no seeded key: {sorted(set(unseeded))}"
    assert not offenders, sorted(set(offenders))


def test_a_steady_rerun_pushes_nothing_new():
    """A widget the browser already shows with the same arguments is NOT re-written:
    re-pushing on every rerun made number-input +/- clicks jump back a step
    (professor, 2026-09-17). The render-log signatures must therefore be stable."""
    sim = _start()
    sim.run()
    _to_page2(sim)
    _select_all(sim)
    sim.run()
    ss = sim.at.session_state
    current, previous = ss["_widget_render_log"], ss["_widget_render_log_previous"]
    assert len(current) > 30
    changed = [k for k in current if previous.get(k) != current[k]]
    assert not changed, f"widgets re-pushed on a rerun that changed nothing: {changed}"
