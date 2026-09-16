"""
Regression tests for the reported glitch:

    "when making changes to the stochastic component options, the whole
     simulation is sometimes reset with decisions deleted and returning
     the common parameters page"

Two things are covered here.

1. UI invariants (test_* below, the bulk of the file): driving every stochastic
   control on the four decision tabs that have one - Disclose Income (1),
   Disclose Documents (2), Donation Default (3) and Rejected Transaction
   Defaults (4) - must never move the app off Page 2, never drop the selected
   decisions and never raise. Covered: both checkboxes per tab, the sigma-mode
   radios (uniform <-> per-quintile), the sigma-coefficient and per-quintile
   sliders, and the same toggles after a Page 1 round trip, after a population
   mode change, after an income-specification change and after the per-element
   and whole-decision reset buttons.

2. The crash that CAN take the app down (and therefore send the user back to a
   fresh session on Page 1): config/decisions.yaml is re-read by every decision
   tab on every rerun and is rewritten in place by the tabs' reset buttons. A
   reader landing inside a non-atomic write sees a truncated document, and when
   that happens in initialize_session_state() the script dies before the page
   router runs. app.models.read_yaml_config / write_yaml_config close that
   window; the last three tests pin that behaviour down.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = str(REPO_ROOT / "app_enhanced_new.py")
DECISIONS_YAML = REPO_ROOT / "config" / "decisions.yaml"

DECISIONS = [
    "disclose_income",
    "disclose_documents",
    "donation_default",
    "rejected_transaction_defaults",
]

# Stochastic-component checkboxes, one "copula" + one "research spec" per tab.
STOCHASTIC_CHECKBOXES = [
    "di_tab_sigma_in_copula",
    "di_tab_sigma_enabled",
    "dd_tab_sigma_in_copula",
    "dd_tab_sigma_enabled",
    "tab_sigma_in_copula",       # donation_default
    "tab_sigma_in_research",     # donation_default
    "rtd_tab_sigma_in_copula",
    "rtd_tab_sigma_enabled",
]

# "Apply sigma uniformly or per budget level?" radios.
SIGMA_STRATEGY_RADIOS = [
    "di_tab_sigma_strategy_stochastic",
    "dd_tab_sigma_strategy_stochastic",
    "donation_tab_sigma_strategy_stochastic",
    "rtd_tab_sigma_strategy",
]

# sigma-coefficient / anchor-mix sliders.
STOCHASTIC_SLIDERS = [
    "di_tab_sigma_coefficient_stochastic",
    "dd_tab_sigma_coefficient_stochastic",
    "tab_sigma_coefficient_stochastic",
    "rtd_tab_sigma_coefficient",
    "tab_anchor_weight",
    "rtd_tab_flex_observed_weight",
]


@pytest.fixture(autouse=True, scope="module")
def preserve_decisions_yaml():
    """Reset buttons rewrite config/decisions.yaml - never leave it changed."""
    original = DECISIONS_YAML.read_bytes()
    try:
        yield
    finally:
        DECISIONS_YAML.write_bytes(original)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def page2(decisions=None):
    """A fresh app, navigated to Page 2 with `decisions` selected."""
    decisions = list(DECISIONS if decisions is None else decisions)
    at = AppTest.from_file(APP_FILE, default_timeout=300)
    at.run()
    # Navigate the way the user does, via the Page 1 nav button.
    nav = [b for b in at.button if "Decision Parameters" in b.label]
    assert nav, "Page 1 is missing the 'Next: Decision Parameters' button"
    nav[0].click().run()
    assert at.session_state["page"] == "page2"
    at.multiselect("page2_manual_multiselect").set_value(decisions).run()
    assert_intact(at, "after selecting decisions", decisions)
    return at


def assert_intact(at, note, decisions=None):
    """The three invariants the professor's glitch violates."""
    decisions = list(DECISIONS if decisions is None else decisions)
    assert not at.exception, (
        f"{note}: script raised "
        + " | ".join(f"{type(e.value).__name__}: {e.value}" for e in at.exception)
    )
    assert at.session_state["page"] == "page2", (
        f"{note}: app left Page 2 (page={at.session_state['page']!r})"
    )
    assert list(at.session_state["decision_params"].selected_decisions) == decisions, (
        f"{note}: selected decisions changed to "
        f"{list(at.session_state['decision_params'].selected_decisions)!r}"
    )


def widget_keys(elements):
    return [e.key for e in elements if e.key]


def go_back_to_page1(at):
    back = [b for b in at.button if "Back to Common Parameters" in b.label]
    assert back, "Page 2 is missing the 'Back to Common Parameters' button"
    back[0].click().run()
    return at


def go_forward_to_page2(at):
    nav = [b for b in at.button if "Decision Parameters" in b.label]
    assert nav, "Page 1 is missing the 'Next: Decision Parameters' button"
    nav[0].click().run()
    return at


# ---------------------------------------------------------------------------
# 1. the stochastic controls themselves
# ---------------------------------------------------------------------------
def test_all_stochastic_widgets_are_rendered():
    """Guard against the lists above silently going stale."""
    at = page2()
    present_cb = set(widget_keys(at.checkbox))
    missing = [k for k in STOCHASTIC_CHECKBOXES if k not in present_cb]
    assert not missing, f"stochastic checkboxes missing from Page 2: {missing}"

    present_radio = set(widget_keys(at.radio))
    missing = [k for k in SIGMA_STRATEGY_RADIOS if k not in present_radio]
    assert not missing, f"sigma-mode radios missing from Page 2: {missing}"

    present_slider = set(widget_keys(at.slider))
    missing = [k for k in STOCHASTIC_SLIDERS if k not in present_slider]
    assert not missing, f"stochastic sliders missing from Page 2: {missing}"


@pytest.mark.parametrize("key", STOCHASTIC_CHECKBOXES)
def test_toggling_a_stochastic_checkbox_keeps_the_session(key):
    at = page2()
    original = at.checkbox(key).value
    at.checkbox(key).set_value(not original).run()
    assert_intact(at, f"after toggling {key} -> {not original}")
    at.checkbox(key).set_value(original).run()
    assert_intact(at, f"after toggling {key} back -> {original}")


def test_toggling_every_stochastic_checkbox_in_one_session():
    """The user flips several boxes in a row without reloading."""
    at = page2()
    for _ in range(3):
        for key in STOCHASTIC_CHECKBOXES:
            at.checkbox(key).set_value(not at.checkbox(key).value).run()
            assert_intact(at, f"after toggling {key}")


@pytest.mark.parametrize("key", SIGMA_STRATEGY_RADIOS)
def test_switching_sigma_mode_keeps_the_session(key):
    at = page2()
    for option in ("quintile", "overall", "quintile", "overall"):
        at.radio(key).set_value(option).run()
        assert_intact(at, f"after setting {key} = {option}")


@pytest.mark.parametrize("key", STOCHASTIC_SLIDERS)
def test_moving_a_sigma_slider_keeps_the_session(key):
    at = page2()
    original = at.slider(key).value
    target = 0.5 if original != 0.5 else 1.5
    at.slider(key).set_value(target).run()
    assert_intact(at, f"after setting {key} = {target}")
    at.slider(key).set_value(original).run()
    assert_intact(at, f"after restoring {key} = {original}")


def test_per_quintile_sliders_keep_the_session():
    """Per-budget-level sliders only exist once sigma mode is 'quintile'."""
    at = page2()
    for key in SIGMA_STRATEGY_RADIOS:
        at.radio(key).set_value("quintile").run()
        assert_intact(at, f"after setting {key} = quintile")

    quintile_sliders = [k for k in widget_keys(at.slider)
                        if "_q1" in k or "_q2" in k or "_q3" in k
                        or "_q4" in k or "_q5" in k]
    assert len(quintile_sliders) >= 20, (
        f"expected 5 per-quintile sliders per tab, found {quintile_sliders}"
    )
    for key in quintile_sliders:
        original = at.slider(key).value
        target = 0.5 if original != 0.5 else 1.5
        at.slider(key).set_value(target).run()
        assert_intact(at, f"after setting per-quintile slider {key} = {target}")


# ---------------------------------------------------------------------------
# 2. the same toggles after the navigation / mode changes the professor uses
# ---------------------------------------------------------------------------
def test_toggle_survives_a_page1_round_trip():
    at = page2()
    at.checkbox("rtd_tab_sigma_in_copula").set_value(True).run()
    assert_intact(at, "after the first toggle")

    go_back_to_page1(at)
    assert not at.exception
    assert at.session_state["page"] == "page1"

    go_forward_to_page2(at)
    assert_intact(at, "after returning to Page 2")
    assert at.checkbox("rtd_tab_sigma_in_copula").value is True, (
        "the stochastic setting was lost across a Page 1 round trip"
    )

    for key in STOCHASTIC_CHECKBOXES:
        at.checkbox(key).set_value(not at.checkbox(key).value).run()
        assert_intact(at, f"after the round trip, toggling {key}")


@pytest.mark.parametrize(
    "population_mode",
    ["Copula (synthetic)", "Research Specification", "Research Baseline", "Compare all"],
)
def test_toggle_after_changing_population_mode(population_mode):
    at = AppTest.from_file(APP_FILE, default_timeout=300)
    at.run()
    at.radio("page1_population_mode").set_value(population_mode).run()
    assert not at.exception
    go_forward_to_page2(at)
    at.multiselect("page2_manual_multiselect").set_value(DECISIONS).run()
    assert_intact(at, f"on Page 2 with population mode {population_mode}")

    for key in STOCHASTIC_CHECKBOXES:
        at.checkbox(key).set_value(not at.checkbox(key).value).run()
        assert_intact(at, f"population mode {population_mode}, toggling {key}")


@pytest.mark.parametrize("radio_key", ["page2_tab_income_spec_mode", "di_tab_income_mode",
                                       "dd_tab_income_mode", "rtd_tab_income_mode"])
def test_toggle_after_changing_income_specification(radio_key):
    at = page2()
    for option in at.radio(radio_key).options:
        at.radio(radio_key).set_value(option).run()
        assert_intact(at, f"after setting {radio_key} = {option}")
        for key in STOCHASTIC_CHECKBOXES:
            at.checkbox(key).set_value(not at.checkbox(key).value).run()
            assert_intact(at, f"{radio_key}={option}, toggling {key}")


RESET_BUTTONS = [
    "rtd_reset_btn",             # "Reset Decision 4 Settings to Defaults"
    "rtd_reset_ttp_btn",
    "rtd_reset_loyalty_btn",
    "rtd_reset_wtp_btn",
    "rtd_reset_risk_taking_btn",
    "rtd_reset_flexibility_btn",
    "di_reset_btn",
    "dd_reset_btn",
    "reset_config_btn",          # donation_default
    "di_reload_btn",
    "dd_reload_btn",
    "reset_intercept_btn",
    "reset_adjustment_btn",
]

# "Reset Config to Defaults" on the Disclose Documents tab writes
# stochastic.sigma_value = 0, which un-checks BOTH of that tab's stochastic
# boxes and therefore removes the sigma-mode radio / sliders from the page in
# the same run that calls st.rerun(). AppTest keeps the aborted run's deltas in
# its element tree, so it can no longer collect widget states afterwards. The
# click itself is still asserted below; only the follow-up toggling is skipped.
RESET_BUTTONS_NOT_DRIVABLE_AFTERWARDS = {"dd_reset_btn"}


@pytest.mark.parametrize("button_key", RESET_BUTTONS)
def test_reset_button_keeps_the_session(button_key):
    """Reset buttons delete widget keys and st.rerun(); that must not take the
    app off Page 2, drop the decisions, or raise."""
    at = page2()
    at.button(button_key).click().run()
    assert_intact(at, f"after clicking {button_key}")


@pytest.mark.parametrize(
    "button_key",
    [b for b in RESET_BUTTONS if b not in RESET_BUTTONS_NOT_DRIVABLE_AFTERWARDS],
)
def test_toggle_after_a_reset_button(button_key):
    at = page2()
    at.button(button_key).click().run()
    assert_intact(at, f"after clicking {button_key}")

    for key in STOCHASTIC_CHECKBOXES:
        at.checkbox(key).set_value(not at.checkbox(key).value).run()
        assert_intact(at, f"after {button_key}, toggling {key}")


def test_toggle_then_change_the_decision_selection():
    at = page2()
    at.checkbox("rtd_tab_sigma_in_copula").set_value(True).run()
    assert_intact(at, "after toggling")

    fewer = DECISIONS[:2]
    at.multiselect("page2_manual_multiselect").set_value(fewer).run()
    assert_intact(at, "after narrowing the selection", fewer)

    at.multiselect("page2_manual_multiselect").set_value(DECISIONS).run()
    assert_intact(at, "after restoring the selection")

    for key in STOCHASTIC_CHECKBOXES:
        at.checkbox(key).set_value(not at.checkbox(key).value).run()
        assert_intact(at, f"after re-selecting, toggling {key}")


# ---------------------------------------------------------------------------
# 3. the config-file race that can actually kill the app
# ---------------------------------------------------------------------------
def test_read_yaml_config_survives_a_half_written_file(tmp_path):
    """A reader landing inside a non-atomic write must not blow up.

    Without this, a truncated config/decisions.yaml makes
    initialize_session_state() raise 'NoneType object is not subscriptable' at
    app_enhanced_new.py:29 - the app dies before the page router runs.
    """
    from app.models import read_yaml_config

    target = tmp_path / "decisions.yaml"
    full = DECISIONS_YAML.read_text()

    # The state a reader sees between open(path, 'w') and the dump completing.
    target.write_text("")
    assert yaml.safe_load(target.read_text()) is None, (
        "sanity: a plain read of the truncated file yields None - this is what "
        "used to crash initialize_session_state()"
    )

    def repair():
        time.sleep(0.06)          # inside read_yaml_config's retry window
        target.write_text(full)

    writer = threading.Thread(target=repair, daemon=True)
    writer.start()
    try:
        config = read_yaml_config(target)
    finally:
        writer.join(timeout=5)

    assert isinstance(config, dict) and "donation_default" in config, (
        "read_yaml_config did not recover once the writer finished"
    )


def test_write_yaml_config_is_atomic(tmp_path):
    """The file must never be observable in a truncated state."""
    from app.models import write_yaml_config

    target = tmp_path / "decisions.yaml"
    data = yaml.safe_load(DECISIONS_YAML.read_text())
    write_yaml_config(target, data)
    assert yaml.safe_load(target.read_text())["donation_default"]

    seen_bad = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                parsed = yaml.safe_load(target.read_text())
            except Exception as exc:          # pragma: no cover - the bug we fixed
                seen_bad.append(repr(exc))
            else:
                if not isinstance(parsed, dict) or "donation_default" not in parsed:
                    seen_bad.append(f"partial document: {type(parsed).__name__}")

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        for _ in range(30):
            write_yaml_config(target, data)
    finally:
        stop.set()
        t.join(timeout=5)

    assert not seen_bad, f"reader observed a partial file: {seen_bad[:3]}"
    # no temp files left behind
    assert [p.name for p in tmp_path.iterdir()] == ["decisions.yaml"]


def test_saving_one_decision_does_not_change_the_others(tmp_path, monkeypatch):
    """`save_disclose_documents_config` rewrites the whole YAML document; make
    sure the other decisions' values survive it byte-for-byte in value terms."""
    from app.pages.decision_tabs import disclose_documents as dd

    target = tmp_path / "decisions.yaml"
    shutil.copy(DECISIONS_YAML, target)
    monkeypatch.setattr(dd, "CONFIG_PATH", target)

    before = yaml.safe_load(target.read_text())
    assert dd.save_disclose_documents_config({"stochastic.scale_factor": 1.25}) is True
    after = yaml.safe_load(target.read_text())

    assert after["disclose_documents"]["stochastic"]["scale_factor"] == 1.25
    for name, section in before.items():
        if name == "disclose_documents":
            continue
        assert after[name] == section, f"saving disclose_documents changed {name}"
