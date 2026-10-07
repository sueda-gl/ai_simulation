"""
A σ coefficient of 0 is a legal value: with the noise tick box on it means σ = 0,
i.e. no noise (owner ruling R6: "if the tick box is on but the slider is at 0 then
sigma is zero").

Disclose Income's tab snapped 0 back to 1.0 on every render ("treat 0.0 as never
intentionally set", plus a 0 -> 1.0 fallback for ``di_scale_factor``), and Disclose
Documents' tab had the same ``or 1.0`` fallbacks. They worked around the browser
showing a slider's built-in 0 (the e51c3fc regression), which 5f2cac8 fixed properly
(app/state/widgets.py). Removed 2026-10-07; the first-draw default stays the
config's ``stochastic.scale_factor`` (0.1 for Disclose Income, R-DI01).

Driven through the browser model (tests/browser_sim.py): the value persists across
reruns, a page switch and a tab hide / show, and a Disclose-Income-only run with
σ coefficient 0 gives exactly the Disclose Income output of the run with the tick
box off.
"""
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from browser_sim import BrowserSim
from test_widget_browser_state import APP_FILE, ROOT

CONFIG = yaml.safe_load((ROOT / "config" / "decisions.yaml").read_text())

DI_COEFF = "di_tab_sigma_coefficient_stochastic"
DD_COEFF = "dd_tab_sigma_coefficient_stochastic"


def _consistent(sim, step):
    assert not sim.at.exception, f"{step}: {[e.value for e in sim.at.exception]}"
    assert not sim.mismatches(), f"{step}: browser != script: {sim.mismatches()}"


def _page2_research_specification():
    sim = BrowserSim(AppTest.from_file(APP_FILE, default_timeout=600))
    sim.run()
    sim.set("page1_population_mode", "Research Specification")
    sim.click("Decision Parameters")
    sim.set("page2_select_all_checkbox", True)
    _consistent(sim, "Page 2, all decisions selected")
    return sim


def _assert_zero(sim, widget_key, state_key, step):
    _consistent(sim, step)
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
    if widget_key == DI_COEFF:   # the first draw is still the config value (R-DI01)
        assert sim.shown(DI_COEFF) == pytest.approx(CONFIG["disclose_income"]["stochastic"]["scale_factor"])

    sim.set(widget_key, 0.0)
    _assert_zero(sim, widget_key, state_key, "set to 0")
    sim.run()
    sim.run()
    _assert_zero(sim, widget_key, state_key, "two reruns")

    sim.click("Back to Common Parameters")
    _consistent(sim, "Page 1")
    sim.click("Decision Parameters")
    _assert_zero(sim, widget_key, state_key, "back on Page 2")

    sim.set("page2_select_all_checkbox", False)
    _consistent(sim, "tabs hidden")
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
    _consistent(sim, "back on Page 2 after the run")
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
    _consistent(sim, "box off")
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
