"""
Page 1 "Reset to Default Values" puts EVERY Page-1 setting back to exactly what a
fresh session shows (owner ruling Q-39, 2026-10-07).

Until then the reset wrote its own list of values, which disagreed with the first
draw (Duration per Period 8 vs 1, Price Grid 9 vs 11, discount threshold 20000 vs
12500, NDIC 2 vs 10, NFIC 8 vs 10, Max Purchases 10 vs 50, Execution Mode Live
Simulation vs Snapshot, and the hidden markup / bidding / carryover-probability
fields). Now a fresh session and the reset both take their values from the
``SimulationParameters`` dataclass and ``SESSION_DEFAULTS`` (app/models.py) through
``page1_widget_values`` (app/pages/page1_common_params.py).

The test drives two sessions through the browser model (tests/browser_sim.py): a
fresh one, and one in which every Page-1 widget was changed before Reset was
pressed. Both then take the same tour through every Page-1 view (Monte Carlo, the
three income distributions, purchasing limits on, several vendors, upload mode) and
must show - in the browser - the same value for every widget, hold the same
``sim_params`` (hidden fields included) and the same Page-1 session values at every
step.
"""
import dataclasses
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from browser_sim import BrowserSim
from test_no_widget_default_warning import offenders  # noqa: F401  (fixture: Q-41 warning spy)

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = str(ROOT / "app_enhanced_new.py")

SESSION_VALUES = ("n_agents", "seed", "n_runs", "base_seed", "show_individual_agents",
                  "population_mode", "nfic_manually_set", "uniform_purchasing_limit")


def _start():
    sim = BrowserSim(AppTest.from_file(APP_FILE, default_timeout=300))
    sim.run()
    _consistent(sim, "first render")
    return sim


def _consistent(sim, step):
    assert not sim.at.exception, f"{step}: {[e.value for e in sim.at.exception]}"
    bad = sim.mismatches()
    assert not bad, f"{step}: browser != script (key, browser, script): {bad}"


def _set(sim, key, value):
    sim.set(key, value)
    _consistent(sim, f"{key} = {value!r}")


def _type(sim, key, text):
    sim.at.text_input(key=key).set_value(text)
    sim.run()
    _consistent(sim, f"{key} = {text!r}")


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
    fresh = _start()
    expected = _tour(fresh)

    sim = _start()
    _change_everything(sim)
    changed = _snapshot(sim, "changed")
    assert changed["sim_params"] != expected[0]["sim_params"]

    sim.click_key("reset_page1_defaults")
    _consistent(sim, "after Reset to Default Values")
    sim.run()
    _consistent(sim, "rerun after reset")
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

    # the values the owner named, as a fresh session shows them
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
