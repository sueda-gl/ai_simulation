"""
Every decision tab's "Reset … to Defaults" restores exactly what the tab showed on
its first draw - and both come from config/decisions.yaml (single source of truth).

Disclose Income's "Reset Config to Defaults" used to apply a hard-coded dict
(``_DI_RESET_DEFAULTS``) on top of the file, whose ``stochastic.scale_factor: 1.0``
set the σ coefficient to 1.0 while the shipped file - and therefore the first draw -
say 0.1 (owner ruling R-DI01, Q-59: 0.1 is correct). The resets now read the file:
Disclose Income / Documents through ``research_default_config()``, Donation Default
through ``research_defaults()``; Decision 4's reset re-runs the tab's own
initialiser.

Each test drives the real app (Page 2, Select All, population Research
Specification so the σ controls show) through the browser model, records every
setting of the tab on its first draw, changes several of them through their
widgets, presses the tab's reset button and checks that every setting - server
side and in the browser - is back to its first-draw value.
"""
import copy

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from browser_sim import BrowserSim
from test_widget_browser_state import APP_FILE, ROOT

CONFIG = yaml.safe_load((ROOT / "config" / "decisions.yaml").read_text())

DONATION_KEYS = (
    "tab_anchor_weight", "anchor_observed_weight", "override_categorical_intercept",
    "override_continuous_intercept", "override_adjustment_shift", "donation_adjustment_shift",
    "donation_sigma_strategy", "donation_quintile_scale_factors", "page2_tab_income_spec_mode",
    "income_spec_mode", "tab_sigma_in_copula", "tab_sigma_in_research", "sigma_in_copula",
    "sigma_in_research", "donation_coeff_intercept_cat", "donation_coeff_intercept_cont",
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
        # Q-44 (deferred, kept): this reset leaves the live σ controls alone - the σ
        # coefficient slider and the σ strategy / quintile widgets keep their values
        # (it writes donation_sigma_strategy, but the strategy radio's own key wins on
        # the next render) - so they are not edited here.
        [("tab_anchor_weight", 0.6), ("override_adjustment_shift", -2.0),
         ("override_categorical_intercept", 1.0), ("tab_sigma_in_copula", True),
         ("tab_sigma_in_research", False)],
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


def _consistent(sim, step):
    assert not sim.at.exception, f"{step}: {[e.value for e in sim.at.exception]}"
    assert not sim.mismatches(), f"{step}: browser differs from the script: {sim.mismatches()}"


def _first_draw():
    sim = BrowserSim(AppTest.from_file(APP_FILE, default_timeout=300))
    sim.run()
    sim.set("page1_population_mode", "Research Specification")
    assert sim.at.session_state["population_mode"] == "Research Specification"
    sim.click("Decision Parameters")
    sim.set("page2_select_all_checkbox", True)
    _consistent(sim, "first draw")
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
        _consistent(sim, f"{tab}: {key} = {value!r}")
    edited = _settings(sim.at, owns)
    assert edited != first, f"{tab}: the edits changed nothing"

    sim.click_key(reset_button)
    _consistent(sim, f"{tab}: after reset")
    sim.run()
    _consistent(sim, f"{tab}: rerun after reset")

    after = _settings(sim.at, owns)
    differs = {key: (first[key], after.get(key, "<absent>")) for key in first
               if after.get(key, "<absent>") != first[key]}
    assert not differs, f"{tab}: reset != first draw (key: (first draw, after reset)): {differs}"
    for key, _ in edits:
        assert sim.shown(key) == widgets_first[key], (tab, key)


def test_disclose_income_sigma_coefficient_is_the_config_value_after_reset():
    """The reported drift: first draw 0.1 (config), reset 1.0 (hard-coded dict)."""
    config_scale = CONFIG["disclose_income"]["stochastic"]["scale_factor"]
    assert config_scale == pytest.approx(0.1)   # R-DI01
    sim = _first_draw()
    assert sim.shown("di_tab_sigma_coefficient_stochastic") == pytest.approx(config_scale)
    sim.set("di_tab_sigma_coefficient_stochastic", 1.3)
    sim.click_key("di_reset_btn")
    sim.run()
    _consistent(sim, "after DI reset")
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
    assert defaults["quintile_scale_factors"] == {
        str(k): float(v) for k, v in donation["stochastic"]["quintile_scale_factors"].items()}
