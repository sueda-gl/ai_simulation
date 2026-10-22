"""
The browser-state tests: the real app driven through ``tests/browser_sim.py``, the
model of the frontend's widget-state manager on top of Streamlit's AppTest.

Merged from five files (2026-10-21) - test_widget_browser_state.py,
test_page1_reset_defaults.py, test_tab_reset_defaults.py,
test_sigma_coefficient_zero.py and test_complete_simulation_gate_headings.py. Each
section below keeps its file's description (as a comment), helpers, tests and
parametrisations; where two files had a helper of the same name that differed (the
``_consistent`` check, ``_start``) each keeps its own under a section prefix.
"""
import ast
import copy
import dataclasses

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from browser_sim import BrowserSim
from conftest import APP_FILE, ROOT, start_app
from test_no_widget_default_warning import offenders  # noqa: F401  (fixture: the widget-default warning spy)

pytestmark = pytest.mark.slow

# config/decisions.yaml as the tab-reset and sigma-zero sections read it
CONFIG = yaml.safe_load((ROOT / "config" / "decisions.yaml").read_text())


# =============================================================================
# from test_widget_browser_state.py
# =============================================================================
# Widgets show - in the BROWSER - the value the script runs with (regression from
# e51c3fc, fixed 2026-10-07).
#
# e51c3fc removed ``value=`` / ``index=`` from ~37 widgets whose value lives in a
# session-state key (to silence Streamlit's "created with a default value but also had
# its value set via the Session State API" warning). Streamlit sends a key's
# value to the browser only in the run in which the key is written through
# ``st.session_state``; a key seeded in an EARLIER run (or kept from one) reaches a
# browser that draws the widget for the first time only as the element's ``default=``
# - which, without ``value=``, is the widget's built-in default (min / False / option
# 0). The browser shows it and sends it back, and the run uses it: the Donation tab's
# σ coefficient and anchor weight became 0, its Research tick box unticked, the
# Overview's P(Y) sliders 0, and a Donation run's mean was computed with anchor 0.
#
# AppTest cannot see this - it reads the server-side value. ``tests/browser_sim.py``
# models the frontend's widget-state manager on top of AppTest; these tests drive the
# app through it, assert after EVERY run that the browser shows what the script
# holds (``mismatches() == []``), and check the concrete defaults the controller saw
# go wrong. All of them fail on a39d359 and pass with ``app/state/widgets.py``.

DEFAULT_PROBABILITY_KEYS = (
    "disclose_income_default_probability_y",
    "disclose_documents_default_probability_y",
    "purchase_vs_bid_default_probability_y",
)


def _start():
    sim = BrowserSim(start_app())
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
    """The original reset raised StreamlitAPIException at its first widget-key
    write; as an on_click callback it resets everything, to the values a fresh session
    shows (test_reset_restores_exactly_what_a_fresh_session_shows checks every widget)."""
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
    re-pushing on every rerun made number-input +/- clicks jump back a step.
    The render-log signatures must therefore be stable."""
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


# =============================================================================
# from test_page1_reset_defaults.py
# =============================================================================
# Page 1 "Reset to Default Values" puts EVERY Page-1 setting back to exactly what a
# fresh session shows.
#
# Until then the reset wrote its own list of values, which disagreed with the first
# draw (Duration per Period 8 vs 1, Price Grid 9 vs 11, discount threshold 20000 vs
# 12500, NDIC 2 vs 10, NFIC 8 vs 10, Max Purchases 10 vs 50, Execution Mode Live
# Simulation vs Snapshot, and the hidden markup / bidding / carryover-probability
# fields). Now a fresh session and the reset both take their values from the
# ``SimulationParameters`` dataclass and ``SESSION_DEFAULTS`` (app/models.py) through
# ``page1_widget_values`` (app/pages/page1_common_params.py).
#
# The test drives two sessions through the browser model (tests/browser_sim.py): a
# fresh one, and one in which every Page-1 widget was changed before Reset was
# pressed. Both then take the same tour through every Page-1 view (Monte Carlo, the
# three income distributions, purchasing limits on, several vendors, upload mode) and
# must show - in the browser - the same value for every widget, hold the same
# ``sim_params`` (hidden fields included) and the same Page-1 session values at every
# step.

SESSION_VALUES = ("n_agents", "seed", "n_runs", "base_seed", "show_individual_agents",
                  "population_mode", "nfic_manually_set", "uniform_purchasing_limit")


def _page1_start():
    sim = BrowserSim(start_app())
    sim.run()
    _page1_consistent(sim, "first render")
    return sim


def _page1_consistent(sim, step):
    assert not sim.at.exception, f"{step}: {[e.value for e in sim.at.exception]}"
    bad = sim.mismatches()
    assert not bad, f"{step}: browser != script (key, browser, script): {bad}"


def _set(sim, key, value):
    sim.set(key, value)
    _page1_consistent(sim, f"{key} = {value!r}")


def _type(sim, key, text):
    sim.at.text_input(key=key).set_value(text)
    sim.run()
    _page1_consistent(sim, f"{key} = {text!r}")


def _snapshot(sim, view):
    """Everything Page 1 shows in the browser, plus the state behind it."""
    shown = {}
    for el in sim._widgets():
        key = getattr(el, "key", None)
        if key and key != "reset_page1_defaults":
            value = sim.shown(key)
            shown[key] = round(value, 9) if isinstance(value, float) else value
    for text in sim.at.text_input:
        shown[text.key] = text.value
    ss = sim.at.session_state
    return {
        "view": view,
        "widgets": shown,
        "sim_params": dataclasses.asdict(ss["sim_params"]),
        "session": {k: ss[k] for k in SESSION_VALUES if k in ss},
    }


def _tour(sim):
    """Visit every Page-1 view; the same steps on both sessions."""
    views = [_snapshot(sim, "default view")]
    _set(sim, "page1_simulation_mode", "Monte-Carlo Study")
    views.append(_snapshot(sim, "Monte-Carlo Study"))
    _set(sim, "page1_simulation_mode", "Single Run")
    for dist in ("generalised_gamma", "dagum"):
        _set(sim, "page1_income_distribution", dist)
        views.append(_snapshot(sim, dist))
    _set(sim, "page1_income_distribution", "lognormal")
    _set(sim, "page1_apply_limits", "Yes")
    views.append(_snapshot(sim, "purchasing limits on"))
    _set(sim, "page1_apply_limits", "No")
    _set(sim, "num_vendors_input", 3)
    views.append(_snapshot(sim, "3 vendors"))
    _set(sim, "page1_vendor_setup_mode", "Upload Vendor Config File")
    views.append(_snapshot(sim, "upload mode"))
    return views


def _change_everything(sim):
    """Every Page-1 widget to a non-default value."""
    _set(sim, "page1_simulation_execution_mode", "Live Simulation")
    _set(sim, "n_agents_input", 280)
    _set(sim, "seed_input", 7)
    _set(sim, "show_individual_agents_checkbox", True)
    _set(sim, "periods_input", 3)
    _set(sim, "duration_hours_input", 5)
    _set(sim, "single_vendor_price_input", 80.0)
    _set(sim, "single_vendor_products_input", 40)
    _set(sim, "single_vendor_carryover", True)
    _set(sim, "num_vendors_input", 4)
    _set(sim, "vendor_price_min_input", 60.0)
    _set(sim, "vendor_price_max_input", 140.0)
    _set(sim, "market_price_input", 90.0)
    _set(sim, "vendor_products_min_input", 60)
    _set(sim, "vendor_products_max_input", 120)
    _set(sim, "vendor_products_avg_input", 90)
    _set(sim, "page1_carryover_mode", "Use probability")
    _set(sim, "vendor_carryover_probability_slider", 0.7)
    _set(sim, "page1_carryover_mode", "All vendors have carryover")
    _set(sim, "page1_vendor_setup_mode", "Upload Vendor Config File")
    _set(sim, "platform_markup_slider", 0.3)
    _set(sim, "price_range_slider", 0.4)
    _set(sim, "bidding_percentage_slider", 0.6)
    _set(sim, "price_grid_input", 7)
    _set(sim, "lognormal_mu_input", 11.0)
    _set(sim, "lognormal_sigma_input", 0.8)
    _set(sim, "lognormal_min_input", 500.0)
    _type(sim, "lognormal_max_text_input", "300000")
    _set(sim, "page1_income_distribution", "generalised_gamma")
    _set(sim, "gg_k_input", 2.0)
    _set(sim, "gg_c_input", 3.0)
    _set(sim, "gg_lambda_input", 30000.0)
    _set(sim, "gg_min_input", 100.0)
    _type(sim, "gg_max_text_input", "500000")
    _set(sim, "page1_income_distribution", "dagum")
    _set(sim, "dagum_a_input", 3.0)
    _set(sim, "dagum_p_input", 2.0)
    _set(sim, "dagum_b_input", 30000.0)
    _set(sim, "dagum_min_input", 200.0)
    _type(sim, "dagum_max_text_input", "400000")
    _set(sim, "discount_threshold_input", 15000.0)
    _set(sim, "num_discount_categories_input", 3)
    _set(sim, "num_fixed_categories_input", 5)
    _set(sim, "artificial_limit_input", 25)
    _set(sim, "page1_apply_limits", "Yes")
    _set(sim, "uniform_purchasing_limit_input", 7)
    _set(sim, "purchasing_limit_0", 3)
    _set(sim, "page1_population_mode", "Research Specification")
    _set(sim, "page1_simulation_mode", "Monte-Carlo Study")
    _set(sim, "n_runs_input", 25)
    _set(sim, "base_seed_input", 9)


def test_reset_restores_exactly_what_a_fresh_session_shows(offenders):
    fresh = _page1_start()
    expected = _tour(fresh)

    sim = _page1_start()
    _change_everything(sim)
    changed = _snapshot(sim, "changed")
    assert changed["sim_params"] != expected[0]["sim_params"]

    sim.click_key("reset_page1_defaults")
    _page1_consistent(sim, "after Reset to Default Values")
    sim.run()
    _page1_consistent(sim, "rerun after reset")
    actual = _tour(sim)

    for want, got in zip(expected, actual):
        view = want["view"]
        assert got["widgets"] == want["widgets"], (
            f"{view}: widgets differ from a fresh session (key: reset vs fresh): "
            + str({k: (got["widgets"].get(k), want["widgets"].get(k))
                   for k in set(want["widgets"]) | set(got["widgets"])
                   if got["widgets"].get(k) != want["widgets"].get(k)}))
        assert got["sim_params"] == want["sim_params"], (
            f"{view}: sim_params differ (field: reset vs fresh): "
            + str({k: (got["sim_params"][k], want["sim_params"][k])
                   for k in want["sim_params"]
                   if got["sim_params"][k] != want["sim_params"][k]}))
        assert got["session"] == want["session"], f"{view}: {got['session']} vs {want['session']}"
    assert len(actual) == len(expected)

    # the specified values, as a fresh session shows them
    first = expected[0]["widgets"]
    assert first["duration_hours_input"] == 1
    assert first["price_grid_input"] == 11
    assert first["discount_threshold_input"] == pytest.approx(12500.0)
    assert first["num_discount_categories_input"] == 10
    assert first["num_fixed_categories_input"] == 10
    assert first["artificial_limit_input"] == 50
    assert first["page1_simulation_execution_mode"] == "Snapshot"
    assert first["page1_simulation_mode"] == "Single Run"
    assert first["page1_population_mode"] == "Copula (synthetic)"
    assert first["n_agents_input"] == 1000 and first["seed_input"] == 42
    mc = expected[1]["widgets"]
    assert mc["n_runs_input"] == 10 and mc["base_seed_input"] == 42
    assert not offenders, sorted(set(offenders))


def test_reset_values_have_one_definition():
    """No default literal in the reset: it is built from SimulationParameters,
    SESSION_DEFAULTS and the same page1_widget_values a fresh session seeds from."""
    import ast
    import inspect
    import textwrap

    from app.pages import page1_common_params as page1

    tree = ast.parse(textwrap.dedent(inspect.getsource(page1.reset_all_page1_defaults)))
    literals = [ast.unparse(n) for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)]
    assert not literals, f"numeric literals in reset_all_page1_defaults: {literals}"
    source = inspect.getsource(page1.initialize_widget_keys)
    assert "page1_widget_values" in source


# =============================================================================
# from test_tab_reset_defaults.py
# =============================================================================
# Every decision tab's "Reset … to Defaults" restores exactly what the tab showed on
# its first draw - and both come from config/decisions.yaml (single source of truth).
#
# Disclose Income's "Reset Config to Defaults" used to apply a hard-coded dict
# (``_DI_RESET_DEFAULTS``) on top of the file, whose ``stochastic.scale_factor: 1.0``
# set the σ coefficient to 1.0 while the shipped file - and therefore the first draw -
# say 0.1 (0.1 is the intended default). The resets now read the file:
# Disclose Income / Documents through ``research_default_config()``, Donation Default
# through ``research_defaults()``; Decision 4's reset re-runs the tab's own
# initialiser.
#
# Each test drives the real app (Page 2, Select All, population Research
# Specification so the σ controls show) through the browser model, records every
# setting of the tab on its first draw, changes several of them through their
# widgets, presses the tab's reset button and checks that every setting - server
# side and in the browser - is back to its first-draw value.

DONATION_KEYS = (
    "tab_anchor_weight", "anchor_observed_weight", "override_categorical_intercept",
    "override_continuous_intercept", "override_adjustment_shift", "donation_adjustment_shift",
    "donation_sigma_strategy", "donation_quintile_scale_factors", "page2_tab_income_spec_mode",
    "income_spec_mode", "tab_sigma_in_copula", "tab_sigma_in_research", "sigma_in_copula",
    "sigma_in_research", "donation_coeff_intercept_cat", "donation_coeff_intercept_cont",
    "donation_sigma_coefficient_setting", "sigma_coefficient",
)

# tab -> (settings it owns, widget edits, reset button)
TABS = {
    "disclose_income": (
        lambda key: key.startswith("di_"),
        [("di_tab_sigma_coefficient_stochastic", 0.55), ("di_wopb_widget", 0.4),
         ("di_wpb_widget", 0.7), ("di_override_intercept", 0.9),
         ("di_tab_sigma_strategy_stochastic", "quintile"), ("di_tab_income_mode", "Continuous only")],
        "di_reset_btn"),
    "disclose_documents": (
        lambda key: key.startswith("dd_"),
        [("dd_tab_sigma_coefficient_stochastic", 0.55), ("dd_override_intercept", -0.4),
         ("dd_tab_sigma_strategy_stochastic", "quintile"), ("dd_tab_income_mode", "Continuous only")],
        "dd_reset_btn"),
    "donation_default": (
        lambda key: key in DONATION_KEYS,
        # The σ controls are reset too (owner-approved 2026-10-22; until then this
        # reset left the drawn σ mode radio, coefficient slider and quintile sliders
        # at their values) - the test below covers them drawn and hidden.
        [("tab_anchor_weight", 0.6), ("override_adjustment_shift", -2.0),
         ("override_categorical_intercept", 1.0), ("tab_sigma_in_copula", True),
         ("tab_sigma_in_research", False), ("tab_sigma_coefficient_stochastic", 1.3),
         ("donation_tab_sigma_strategy_stochastic", "quintile")],
        "reset_config_btn"),
    "rejected_transaction_defaults": (
        lambda key: key.startswith("rtd_"),
        [("rtd_tab_sigma_coefficient", 0.55), ("rtd_tab_flex_observed_weight", 0.6),
         ("rtd_tab_intercept_loyalty", 0.3), ("rtd_tab_intercept_ttp", 0.1),
         ("rtd_tab_sigma_strategy", "quintile"), ("rtd_tab_income_mode", "Categorical only")],
        "rtd_reset_btn"),
}


def _settings(at, owns):
    state = at.session_state.filtered_state
    return {key: copy.deepcopy(value) for key, value in state.items()
            if owns(key) and not key.endswith("_btn") and not key.endswith("_btn_disabled")}


def _reset_consistent(sim, step):
    assert not sim.at.exception, f"{step}: {[e.value for e in sim.at.exception]}"
    assert not sim.mismatches(), f"{step}: browser differs from the script: {sim.mismatches()}"


def _first_draw():
    sim = BrowserSim(start_app())
    sim.run()
    sim.set("page1_population_mode", "Research Specification")
    assert sim.at.session_state["population_mode"] == "Research Specification"
    sim.click("Decision Parameters")
    sim.set("page2_select_all_checkbox", True)
    _reset_consistent(sim, "first draw")
    return sim


@pytest.mark.parametrize("tab", list(TABS))
def test_reset_restores_the_first_draw(tab):
    owns, edits, reset_button = TABS[tab]
    sim = _first_draw()
    first = _settings(sim.at, owns)
    widgets_first = {key: sim.shown(key) for key, _ in edits}
    assert first, tab

    for key, value in edits:
        sim.set(key, value)
        _reset_consistent(sim, f"{tab}: {key} = {value!r}")
    edited = _settings(sim.at, owns)
    assert edited != first, f"{tab}: the edits changed nothing"

    sim.click_key(reset_button)
    _reset_consistent(sim, f"{tab}: after reset")
    sim.run()
    _reset_consistent(sim, f"{tab}: rerun after reset")

    after = _settings(sim.at, owns)
    differs = {key: (first[key], after.get(key, "<absent>")) for key in first
               if after.get(key, "<absent>") != first[key]}
    assert not differs, f"{tab}: reset != first draw (key: (first draw, after reset)): {differs}"
    for key, _ in edits:
        assert sim.shown(key) == widgets_first[key], (tab, key)


def test_disclose_income_sigma_coefficient_is_the_config_value_after_reset():
    """The reported drift: first draw 0.1 (config), reset 1.0 (hard-coded dict)."""
    config_scale = CONFIG["disclose_income"]["stochastic"]["scale_factor"]
    assert config_scale == pytest.approx(0.1)   # the intended default
    sim = _first_draw()
    assert sim.shown("di_tab_sigma_coefficient_stochastic") == pytest.approx(config_scale)
    sim.set("di_tab_sigma_coefficient_stochastic", 1.3)
    sim.click_key("di_reset_btn")
    sim.run()
    _reset_consistent(sim, "after DI reset")
    assert sim.shown("di_tab_sigma_coefficient_stochastic") == pytest.approx(config_scale)
    assert sim.at.session_state["di_scale_factor"] == pytest.approx(config_scale)


def test_reset_defaults_are_read_from_the_config():
    """No tab keeps its own copy of the research defaults."""
    from app.pages.decision_tabs import disclose_documents, disclose_income, donation_default

    assert not hasattr(disclose_income, "_DI_RESET_DEFAULTS")
    assert not hasattr(disclose_documents, "_DD_RESET_DEFAULTS")
    assert disclose_income.research_default_config() == CONFIG["disclose_income"]
    assert disclose_documents.research_default_config() == CONFIG["disclose_documents"]

    donation = CONFIG["donation_default"]
    defaults = donation_default.research_defaults()
    assert defaults["intercepts"] == {
        "categorical": donation["regression_coefficients"]["categorical"]["intercept"],
        "continuous": donation["regression_coefficients"]["continuous"]["intercept"]}
    assert defaults["shift_value"] == donation["adjustment"]["shift_value"]
    assert defaults["anchor_observed_weight"] == donation["anchor_weights"]["observed"]
    assert defaults["sigma_strategy"] == donation["stochastic"]["sigma_strategy"]
    assert defaults["scale_factor"] == donation["stochastic"]["scale_factor"]
    assert defaults["quintile_scale_factors"] == {
        str(k): float(v) for k, v in donation["stochastic"]["quintile_scale_factors"].items()}


# the σ mode radio's options as the browser shows them
SIGMA_MODE_LABEL = {"overall": "Uniformly (single σ for all)", "quintile": "Quintiles (σ per budget level)"}
DONATION_QUINTILE_SLIDERS = tuple(f"donation_tab_sigma_q{level}_stochastic" for level in "12345")


def _donation_sigma_at_defaults(sim, step):
    """The Donation σ controls - shown in the browser and held server side - at the
    file's donation_default.stochastic values (Copula off, Research Specification on)."""
    _reset_consistent(sim, step)
    stochastic = CONFIG["donation_default"]["stochastic"]
    quintiles = {str(k): float(v) for k, v in stochastic["quintile_scale_factors"].items()}
    ss = sim.at.session_state
    assert stochastic["sigma_strategy"] == "overall"
    assert sim.shown("donation_tab_sigma_strategy_stochastic") == SIGMA_MODE_LABEL["overall"], step
    assert sim.element("donation_tab_sigma_strategy_stochastic").value == "overall", step
    assert sim.shown("tab_sigma_coefficient_stochastic") == pytest.approx(stochastic["scale_factor"]), step
    assert sim.shown("tab_sigma_in_copula") is False, step
    assert sim.shown("tab_sigma_in_research") is True, step
    assert ss["donation_sigma_strategy"] == "overall", step
    assert ss["donation_sigma_coefficient_setting"] == pytest.approx(stochastic["scale_factor"]), step
    assert ss["sigma_coefficient"] == pytest.approx(stochastic["scale_factor"]), step
    assert ss["donation_quintile_scale_factors"] == quintiles, step
    assert ss["sigma_in_copula"] is False and ss["sigma_in_research"] is True, step
    for key in DONATION_QUINTILE_SLIDERS:          # hidden in 'overall' mode
        assert not [el for el in sim._widgets() if getattr(el, "key", None) == key], (step, key)


def _settled_state(sim):
    """Session state without the per-run render log, and what the browser shows."""
    state = {key: copy.deepcopy(value) for key, value in sim.at.session_state.filtered_state.items()
             if not key.startswith("_widget_render_log")}
    return state, dict(sim.browser)


@pytest.mark.parametrize("mode_at_reset", ["quintile", "overall"])
def test_donation_reset_returns_the_sigma_controls_to_the_file_defaults(mode_at_reset):
    """The Donation tab's "Reset Config to Defaults" ("... sliders, radio buttons")
    returns the σ mode radio, the σ coefficient slider, the five quintile sliders and
    the two σ tick boxes to their defaults, like the other three tabs' resets - each
    control whether it is drawn or hidden when the button is pressed (parametrised by
    the σ mode on screen then). A plain rerun after the reset writes nothing and
    changes nothing, and a Page-1 round trip keeps the defaults."""
    sim = _first_draw()
    sim.set("tab_sigma_coefficient_stochastic", 1.3)
    sim.set("donation_tab_sigma_strategy_stochastic", "quintile")
    sim.set("donation_tab_sigma_q2_stochastic", 1.5)
    sim.set("donation_tab_sigma_q4_stochastic", 0.5)
    if mode_at_reset == "overall":
        sim.set("donation_tab_sigma_strategy_stochastic", "overall")
        assert sim.shown("tab_sigma_coefficient_stochastic") == pytest.approx(1.3)
    sim.set("tab_sigma_in_research", False)
    sim.set("tab_sigma_in_copula", True)
    _reset_consistent(sim, "edited")
    ss = sim.at.session_state
    assert sim.shown("donation_tab_sigma_strategy_stochastic") == SIGMA_MODE_LABEL[mode_at_reset]
    assert ss["donation_sigma_coefficient_setting"] == pytest.approx(1.3)
    assert ss["donation_quintile_scale_factors"] == {"1": 1.0, "2": 1.5, "3": 1.0, "4": 0.5, "5": 1.0}
    assert sim.shown("tab_sigma_in_research") is False and sim.shown("tab_sigma_in_copula") is True

    sim.click_key("reset_config_btn")
    _donation_sigma_at_defaults(sim, "after the reset")

    # a plain rerun: nothing pushed to the browser, nothing changed
    before = _settled_state(sim)
    sim.run()
    _donation_sigma_at_defaults(sim, "plain rerun after the reset")
    pushed = [getattr(el, "key", None) or el.label for el in sim._widgets() if el.proto.set_value]
    assert not pushed, f"widgets written through the Session State API on an unchanged rerun: {pushed}"
    assert _settled_state(sim) == before, "the plain rerun after the reset changed the session or the browser"

    # Page 1 and back: the defaults stay
    sim.click("Back to Common Parameters")
    _reset_consistent(sim, "Page 1")
    sim.click("Decision Parameters")
    _donation_sigma_at_defaults(sim, "back on Page 2")

    # the quintile sliders, hidden since the reset, come back at the file's values
    sim.set("donation_tab_sigma_strategy_stochastic", "quintile")
    _reset_consistent(sim, "σ mode quintile after the reset")
    stochastic = CONFIG["donation_default"]["stochastic"]
    for level, key in zip("12345", DONATION_QUINTILE_SLIDERS):
        assert sim.shown(key) == pytest.approx(float(stochastic["quintile_scale_factors"][level])), key


# =============================================================================
# from test_sigma_coefficient_zero.py
# =============================================================================
# A σ coefficient of 0 is a legal value: with the noise tick box on it means σ = 0,
# i.e. no noise (the specification: "if the tick box is on but the slider is at 0 then
# sigma is zero").
#
# Disclose Income's tab snapped 0 back to 1.0 on every render ("treat 0.0 as never
# intentionally set", plus a 0 -> 1.0 fallback for ``di_scale_factor``), and Disclose
# Documents' tab had the same ``or 1.0`` fallbacks. They worked around the browser
# showing a slider's built-in 0 (the e51c3fc regression), which 5f2cac8 fixed properly
# (app/state/widgets.py). Removed 2026-10-07; the first-draw default stays the
# config's ``stochastic.scale_factor`` (0.1 for Disclose Income).
#
# Driven through the browser model (tests/browser_sim.py): the value persists across
# reruns, a page switch and a tab hide / show, and a Disclose-Income-only run with
# σ coefficient 0 gives exactly the Disclose Income output of the run with the tick
# box off.

DI_COEFF = "di_tab_sigma_coefficient_stochastic"
DD_COEFF = "dd_tab_sigma_coefficient_stochastic"


def _sigma_consistent(sim, step):
    assert not sim.at.exception, f"{step}: {[e.value for e in sim.at.exception]}"
    assert not sim.mismatches(), f"{step}: browser != script: {sim.mismatches()}"


def _page2_research_specification():
    sim = BrowserSim(AppTest.from_file(APP_FILE, default_timeout=600))
    sim.run()
    sim.set("page1_population_mode", "Research Specification")
    sim.click("Decision Parameters")
    sim.set("page2_select_all_checkbox", True)
    _sigma_consistent(sim, "Page 2, all decisions selected")
    return sim


def _assert_zero(sim, widget_key, state_key, step):
    _sigma_consistent(sim, step)
    assert sim.shown(widget_key) == 0.0, f"{step}: browser shows {sim.shown(widget_key)}"
    assert sim.at.session_state[widget_key] == 0.0, step
    assert sim.at.session_state[state_key] == 0.0, step


@pytest.mark.parametrize("widget_key,state_key,tick_box", [
    (DI_COEFF, "di_scale_factor", "di_tab_sigma_enabled"),
    (DD_COEFF, "dd_scale_factor", None),
])
def test_sigma_coefficient_zero_persists(widget_key, state_key, tick_box):
    sim = _page2_research_specification()
    if tick_box:
        assert sim.shown(tick_box) is True
    if widget_key == DI_COEFF:   # the first draw is still the config value
        assert sim.shown(DI_COEFF) == pytest.approx(CONFIG["disclose_income"]["stochastic"]["scale_factor"])

    sim.set(widget_key, 0.0)
    _assert_zero(sim, widget_key, state_key, "set to 0")
    sim.run()
    sim.run()
    _assert_zero(sim, widget_key, state_key, "two reruns")

    sim.click("Back to Common Parameters")
    _sigma_consistent(sim, "Page 1")
    sim.click("Decision Parameters")
    _assert_zero(sim, widget_key, state_key, "back on Page 2")

    sim.set("page2_select_all_checkbox", False)
    _sigma_consistent(sim, "tabs hidden")
    sim.set("page2_select_all_checkbox", True)
    _assert_zero(sim, widget_key, state_key, "tabs shown again")


def _run_disclose_income_only(sim):
    sim.click_key("run_disclose_income_only_btn")
    assert not sim.at.exception, [e.value for e in sim.at.exception]
    ss = sim.at.session_state
    assert ss["page"] == "results"
    results = ss["simulation_results"]
    assert list(results) == ["categorical"], list(results)
    df = results["categorical"]
    columns = [c for c in df.columns if c.startswith("disclose_income")]
    assert "disclose_income" in columns and "disclose_income_anchored_pb" in columns
    sim.click("Back to Decision Parameters")
    _sigma_consistent(sim, "back on Page 2 after the run")
    return df[columns].copy()


def test_sigma_coefficient_zero_is_no_noise_in_the_run():
    sim = _page2_research_specification()

    noisy = _run_disclose_income_only(sim)                  # first draw: 0.1, box on
    assert (noisy["disclose_income_anchored_pb"]
            != noisy["disclose_income_anchored_pb_deterministic"]).any(), "σ 0.1 drew no noise"

    sim.set(DI_COEFF, 0.0)
    _assert_zero(sim, DI_COEFF, "di_scale_factor", "σ coefficient 0, box on")
    assert sim.shown("di_tab_sigma_enabled") is True
    zero = _run_disclose_income_only(sim)
    assert sim.shown(DI_COEFF) == 0.0, "the run reset the coefficient"

    sim.set("di_tab_sigma_enabled", False)
    _sigma_consistent(sim, "box off")
    off = _run_disclose_income_only(sim)

    assert (zero["disclose_income_anchored_pb"]
            == zero["disclose_income_anchored_pb_deterministic"]).all()
    assert zero.equals(off), "σ coefficient 0 with the box on differs from the box off"


def test_donation_coefficient_zero_reaches_the_engine_as_sigma_zero():
    """Donation's engine falls back to sigma_overall x scale_factor when sigma_value
    is 0 (src/decisions/donation_default.py); the seam passes the coefficient as
    scale_factor too, so a coefficient of 0 stays sigma 0 there as well."""
    from app.seam import build_plan as bp
    from app.seam.config_repo import get_config_repo
    from test_build_plan import base_state

    repo = get_config_repo()
    patch = bp.build_donation_patch(base_state(repo, sigma_coefficient=0.0), repo,
                                    "documentation", "categorical")
    stochastic = patch["stochastic"]
    assert stochastic["sigma_value"] == 0.0
    assert stochastic["scale_factor"] == 0.0


# =============================================================================
# from test_complete_simulation_gate_headings.py
# =============================================================================
# The "Run Complete Simulation" gate names the decision that blocks it.
#
# ``can_run_complete_simulation()`` reports one blocking issue per decision whose
# "Compare both" (or Compare-all population) setting would produce several
# configurations.  With a single issue, every gate - each decision tab's
# "🚀 Simulation Options" (``render_simulation_buttons``), the Page-2 Overview's
# "Complete Simulation" section and the results page - shows that decision's own
# "Configuration Required" heading.  "⚠️ Multiple Donation Configurations Detected"
# belongs to the donation block only: it used to be the ``else`` of the
# single-issue branch, i.e. the heading of any block type a gate did not list
# (Page 2 did not list Disclose Documents, so a Disclose Documents block was
# announced as a donation one).
#
# Reproduced in the app through the browser model: Page 2, Select All, the Decision
# 4 tab's income specification set to "Compare both" (the only blocking decision),
# then every tab's Simulation Options read.

DONATION_HEADING = "⚠️ **Multiple Donation Configurations Detected**"


def _gate_headings(at):
    """{tab label: [first line of each gate warning]} for every top-level Page-2 tab."""
    headings = {}
    for tab in at.tabs:
        if not (tab.label.startswith("🎯") or tab.label.startswith("📊")):
            continue   # the Decision 4 tab's own sub-tabs
        headings[tab.label] = [str(w.value).strip().splitlines()[0] for w in tab.warning
                               if "Configuration" in str(w.value).strip().splitlines()[0]]
    return headings


def _page2_all_selected():
    sim = BrowserSim(start_app())
    sim.run()
    sim.click("Decision Parameters")
    sim.set("page2_select_all_checkbox", True)
    assert not sim.at.exception
    return sim


@pytest.mark.parametrize("tab_key, session_key, heading", [
    ("rtd_tab_income_mode", "rtd_income_mode",
     "⚠️ **Rejected Transaction Defaults Configuration Required**"),
    ("dd_tab_income_mode", "dd_income_mode",
     "⚠️ **Disclose Documents Configuration Required**"),
])
def test_single_block_shows_its_own_heading_in_every_gate(tab_key, session_key, heading):
    sim = _page2_all_selected()
    sim.set(tab_key, "Compare both")
    at = sim.at
    assert not at.exception
    assert at.session_state[session_key] == "Compare both"

    headings = _gate_headings(at)

    decision_tabs = [label for label in headings if label.startswith("🎯")]
    assert len(decision_tabs) == 13
    for label, found in headings.items():
        assert found == [heading], (label, found)
        assert DONATION_HEADING not in found, label
    # every decision tab's Complete Simulation button is disabled
    disabled = [b for b in at.button if b.key and b.key.endswith("_btn_disabled")]
    assert len(disabled) == 13 and all(b.disabled for b in disabled)


def test_donation_block_keeps_the_donation_heading():
    """The donation block itself still reads 'Multiple Donation Configurations Detected'."""
    sim = _page2_all_selected()
    sim.set("page2_tab_income_spec_mode", "Compare both")
    at = sim.at
    assert not at.exception
    headings = _gate_headings(at)
    assert headings and all(found == [DONATION_HEADING] for found in headings.values()), headings
