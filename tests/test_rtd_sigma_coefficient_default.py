"""
Decision 4's σ Coefficient defaults to 0.5 (owner ruling 2026-10-07, from the updated
user manual; it was 1.0): the Uniformly coefficient and every per-budget-level
coefficient in Quintiles mode. The value lives in config/decisions.yaml
(``rejected_transaction_defaults.stochastic.mechanisms.*.scale_factor`` /
``quintile_scale_factors``); the tab's first draw and "Reset Decision 4 Settings to
Defaults" read it from there, and a run with σ on draws with base σ × 0.5.

Copula draws use the same decision-wide coefficient (the seam passes
``rtd_scale_factor`` for both population modes); the Copula-only caption states the
coefficient actually in effect instead of a fixed "coefficient 1.0".

Stata parity is unaffected (it is checked deterministically, σ off).
"""
import numpy as np
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from app.seam.config_repo import get_config_repo, rtd_sigma_coefficient_defaults
from src.decisions.rejected_transaction_defaults import MECHANISMS, SIGMA_OVERALL

from test_tab_reset_defaults import _consistent, _first_draw
from test_widget_browser_state import ROOT

CONFIG = yaml.safe_load((ROOT / "config" / "decisions.yaml").read_text())
RTD = CONFIG["rejected_transaction_defaults"]
LEVELS = ("1", "2", "3", "4", "5")
DEFAULT = 0.5


def test_config_coefficients_are_0_5_for_every_element_and_level():
    mechs = RTD["stochastic"]["mechanisms"]
    assert set(mechs) == set(MECHANISMS)
    for name, m in mechs.items():
        assert m["scale_factor"] == DEFAULT, name
        assert {str(k): v for k, v in m["quintile_scale_factors"].items()} == {
            lvl: DEFAULT for lvl in LEVELS}, name
    assert rtd_sigma_coefficient_defaults(RTD) == (DEFAULT, {lvl: DEFAULT for lvl in LEVELS})
    assert get_config_repo().rtd_sigma_coefficient_defaults() == (
        DEFAULT, {lvl: DEFAULT for lvl in LEVELS})


def test_disagreeing_element_coefficients_are_rejected():
    """The tab applies ONE coefficient set to all five elements."""
    block = yaml.safe_load(yaml.safe_dump(RTD))
    block["stochastic"]["mechanisms"]["wtp"]["scale_factor"] = 0.7
    with pytest.raises(ValueError):
        rtd_sigma_coefficient_defaults(block)


def _show_quintile_sliders(sim):
    sim.set("rtd_tab_sigma_strategy", "quintile")
    _consistent(sim, "quintile mode")
    return {lvl: sim.shown(f"rtd_tab_sigma_q{lvl}") for lvl in LEVELS}


def test_first_draw_and_reset_show_0_5():
    sim = _first_draw()
    assert sim.shown("rtd_tab_sigma_coefficient") == pytest.approx(DEFAULT)
    assert sim.at.session_state["rtd_scale_factor"] == pytest.approx(DEFAULT)
    assert sim.at.session_state["rtd_quintile_scale_factors"] == {lvl: DEFAULT for lvl in LEVELS}
    assert _show_quintile_sliders(sim) == {lvl: pytest.approx(DEFAULT) for lvl in LEVELS}

    # change both coefficient sets, then reset
    for lvl, value in zip(LEVELS, (0.1, 0.2, 1.3, 1.4, 2.0)):
        sim.set(f"rtd_tab_sigma_q{lvl}", value)
    sim.set("rtd_tab_sigma_strategy", "overall")
    sim.set("rtd_tab_sigma_coefficient", 1.7)
    _consistent(sim, "edited")
    assert sim.at.session_state["rtd_scale_factor"] == pytest.approx(1.7)

    sim.click_key("rtd_reset_btn")
    sim.run()
    _consistent(sim, "after reset")
    assert sim.shown("rtd_tab_sigma_coefficient") == pytest.approx(DEFAULT)
    assert sim.at.session_state["rtd_scale_factor"] == pytest.approx(DEFAULT)
    assert sim.at.session_state["rtd_quintile_scale_factors"] == {lvl: DEFAULT for lvl in LEVELS}
    assert _show_quintile_sliders(sim) == {lvl: pytest.approx(DEFAULT) for lvl in LEVELS}


def _rtd_research_spec_script():
    """Decision 4 tab in Research Specification mode (σ on by default)."""
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()
    st.session_state.population_mode = 'Research Specification'
    st.session_state.n_agents = 60

    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()


def test_run_with_sigma_on_draws_with_base_sigma_times_0_5():
    at = AppTest.from_function(_rtd_research_spec_script)
    at.run(timeout=600)
    assert not at.exception
    assert at.session_state["rtd_sigma_enabled"] is True
    assert at.session_state["rtd_scale_factor"] == pytest.approx(DEFAULT)

    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    results = at.session_state['simulation_results']
    df = next(iter(results.values()))
    assert len(df) > 0
    # each element's effective σ = its base σ (config sigma_overall) × 0.5
    base = {m: RTD["stochastic"]["mechanisms"][m]["sigma_overall"] for m in MECHANISMS}
    assert base["ttp"] == pytest.approx(SIGMA_OVERALL["ttp"])
    for col, mech in (("rtd_sigma_used_ttp", "ttp"), ("rtd_sigma_used_wtp", "wtp"),
                      ("rtd_sigma_used_flex", "flexibility")):
        assert col in df.columns, col
        assert np.allclose(df[col].astype(float), base[mech] * DEFAULT, rtol=0, atol=1e-12), (
            col, df[col].unique())
    assert df["rtd_sigma_used_ttp"].iloc[0] == pytest.approx(0.395446 * 0.5)


def test_copula_only_caption_states_the_coefficient_in_effect():
    """The caption used to say "coefficient 1.0" regardless of the setting; Copula draws
    actually use the decision-wide coefficient (default 0.5)."""
    at = AppTest.from_function(_rtd_research_spec_script)
    at.run(timeout=600)
    at.checkbox("rtd_tab_sigma_in_copula").set_value(True).run(timeout=600)
    at.checkbox("rtd_tab_sigma_enabled").set_value(False).run(timeout=600)
    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    copula = [c for c in captions if c.startswith("Copula draws use")]
    assert copula == ["Copula draws use each element's base σ × the decision's σ coefficient "
                      "(0.50). Check the Research Specification box to configure σ settings."]
