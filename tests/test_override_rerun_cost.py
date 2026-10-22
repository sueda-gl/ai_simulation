"""
Reported problem: "When clicking a few times to change the Override value for any of the
decision elements, the simulation page freezes/blanks for a few seconds."

Three causes, three guards, and a fourth section for the Donation intercept inputs
(the +/- click loss, fixed in 8fe282e):

1. Every Page-2 rerun re-parsed config/decisions.yaml - fourteen times with
   Decisions 1-4 selected (~0.06 s each, ~80 % of the rerun). `read_yaml_file`
   caches the parse per (path, mtime, size) and hands out deep copies.
2. Each +/- click reran the WHOLE page (every selected decision tab), fading it
   while the run lasted. The Decision 4 Override value is now an st.fragment: a
   click reruns only that control.
3. A click while a run was going stopped that run early; its partial widget render
   log, taken as "the previous run", made the next run re-push the Override value
   to the browser with a stale value - the number jumped. After a stopped run the
   render log is merged, not replaced (app/state/widgets.py).
"""
import os
import time
import types

import pytest
import yaml

import app.state.widgets as widgets
from app.seam.config_repo import read_yaml_file

from conftest import DECISIONS_YAML, load_decisions_yaml

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# 1. cached YAML
# ---------------------------------------------------------------------------
def test_read_yaml_file_equals_a_fresh_load_and_is_a_private_copy():
    fresh = load_decisions_yaml()
    first = read_yaml_file(DECISIONS_YAML)
    assert first == fresh
    first["rejected_transaction_defaults"]["intercepts"]["ttp"] = 123.0
    first.pop("donation_default")
    assert read_yaml_file(DECISIONS_YAML) == fresh      # callers cannot poison the cache


def test_read_yaml_file_rereads_an_edited_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\n")
    assert read_yaml_file(path) == {"a": 1}
    time.sleep(0.01)
    path.write_text("a: 22\n")
    os.utime(path, ns=(time.time_ns(), time.time_ns() + 1_000_000))
    assert read_yaml_file(path) == {"a": 22}


def _page2_script():
    import streamlit as st
    from app.models import initialize_session_state
    from app.state.widgets import begin_script_run, end_script_run
    initialize_session_state()
    begin_script_run()
    st.session_state.page = 'page2'
    from app.pages import render_page2
    render_page2()
    end_script_run()


def test_page2_rerun_parses_no_yaml(monkeypatch):
    """With Decisions 1-4 selected, a rerun after an Override click parses no YAML
    (before: 14 parses, ~0.8 s of the ~0.9 s profiled rerun)."""
    from streamlit.testing.v1 import AppTest

    decisions = ['disclose_income', 'disclose_documents', 'donation_default',
                 'rejected_transaction_defaults']
    at = AppTest.from_function(_page2_script, default_timeout=600)
    at.session_state['page2_manual_multiselect'] = decisions
    at.session_state['page2_manual_selections'] = decisions
    at.run()
    assert not at.exception

    calls = []
    real = yaml.safe_load
    monkeypatch.setattr(yaml, "safe_load", lambda *a, **k: calls.append(1) or real(*a, **k))
    at.number_input(key="rtd_tab_intercept_ttp").increment().run()
    assert not at.exception
    assert at.session_state["rtd_intercept_ttp"] == pytest.approx(0.01)
    assert calls == []


# ---------------------------------------------------------------------------
# 2. the Override value is a fragment
# ---------------------------------------------------------------------------
def test_override_value_is_drawn_as_a_fragment():
    import inspect
    from app.pages.decision_tabs import rejected_transaction as rt
    src = inspect.getsource(rt.render_mechanism_subtab)
    assert "render_intercept_control_fragment(config, mech)" in src
    assert "render_intercept_control(config, mech)" not in src
    # st.fragment wraps the plain renderer (functools.wraps keeps the name)
    assert rt.render_intercept_control_fragment is not rt.render_intercept_control
    assert rt.render_intercept_control_fragment.__wrapped__ is rt.render_intercept_control


# ---------------------------------------------------------------------------
# 3. render log after a run stopped early
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_session(monkeypatch):
    state = {}
    monkeypatch.setattr(widgets, "st", types.SimpleNamespace(session_state=state))
    return state


def test_completed_runs_replace_the_previous_log(fake_session):
    fake_session["w"] = 0.0
    widgets.begin_script_run()
    widgets.sync_widget_key("w", signature="sig")
    widgets.end_script_run()
    widgets.begin_script_run()
    assert fake_session[widgets.PREVIOUS_RENDER_LOG_KEY] == {"w": "sig"}
    widgets.end_script_run()                     # 'w' not drawn in this run
    widgets.begin_script_run()
    assert fake_session[widgets.PREVIOUS_RENDER_LOG_KEY] == {}


def test_a_stopped_run_merges_its_log_and_does_not_repush(fake_session):
    """Run 1 completes and draws the Override value; run 2 is stopped by a click
    before reaching it; run 3 must NOT re-push the value (that push carried a stale
    value and made the number jump)."""
    fake_session["early"] = 1
    fake_session["override"] = 0.03
    widgets.begin_script_run()
    widgets.sync_widget_key("early", signature="e")
    widgets.sync_widget_key("override", signature="o")
    widgets.end_script_run()

    widgets.begin_script_run()                   # run 2: stopped after 'early'
    widgets.sync_widget_key("early", signature="e")
    # (no end_script_run: a RerunException ended it)

    widgets.begin_script_run()                   # run 3
    previous = fake_session[widgets.PREVIOUS_RENDER_LOG_KEY]
    assert previous == {"early": "e", "override": "o"}

    writes = []

    class Recorder(dict):
        def __setitem__(self, k, v):
            writes.append(k)
            super().__setitem__(k, v)
    rec = Recorder(fake_session)
    widgets.st.session_state = rec
    widgets.sync_widget_key("override", signature="o")
    assert "override" not in writes              # browser still holds it: no push


def test_fragment_rerun_does_not_repush(fake_session):
    """A fragment rerun calls no begin_script_run: the widget sits in the CURRENT log
    (drawn by the last full run) - no push, even when the previous log lacks it."""
    fake_session["override"] = 0.0
    widgets.begin_script_run()
    widgets.sync_widget_key("override", signature="o")   # first draw (full run)
    widgets.end_script_run()

    writes = []

    class Recorder(dict):
        def __setitem__(self, k, v):
            writes.append(k)
            super().__setitem__(k, v)
    widgets.st.session_state = Recorder(fake_session)
    widgets.sync_widget_key("override", signature="o")   # fragment rerun
    assert "override" not in writes


# ---------------------------------------------------------------------------
# 4. the Donation intercept override inputs are not re-pushed by an unchanged rerun
# ---------------------------------------------------------------------------
# Six quick "+" clicks on either Donation Default intercept input landed only 2-4
# steps (measured in a browser, 2026-10-22: 3/6, 2/6, 4/6, 4/6 at 120 ms between
# clicks; every other number input kept all six). The two keys were the only number
# inputs restored by writing the stored override OVER the widget key on every rerun,
# and a key written through the Session State API is re-sent to the browser with
# that run's response, overwriting a click that arrived while the run was in flight.
# AppTest cannot time clicks, but it can see the cause: the element proto's
# ``set_value`` flag is True exactly when the key was written in that run (the flag
# tests/browser_sim.py models the browser with). After a rerun in which the tab was
# drawn and nothing changed, it must be False for both inputs - as it already is for
# the Distribution Shift Value input and the Decision 4 intercept inputs.

DONATION_INTERCEPT_INPUTS = {
    "categorical only": ["override_categorical_intercept"],
    "continuous only": ["override_continuous_intercept"],
    "Compare both": ["override_categorical_intercept", "override_continuous_intercept"],
}
CANONICAL_INTERCEPT = {"override_categorical_intercept": "donation_coeff_intercept_cat",
                       "override_continuous_intercept": "donation_coeff_intercept_cont"}


def _donation_page2():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_page2_script, default_timeout=600)
    at.session_state['page2_manual_multiselect'] = ['donation_default']
    at.session_state['page2_manual_selections'] = ['donation_default']
    at.run()
    assert not at.exception
    return at


def _pushed(at, key):
    return bool(at.number_input(key=key).proto.set_value)


def test_the_push_flag_is_observable():
    """Control: the run that seeds an input's key does push it (so a False below means
    'not written', not 'not observable')."""
    at = _donation_page2()
    assert _pushed(at, "override_categorical_intercept")          # seeded in this run
    at.radio(key="page2_tab_income_spec_mode").set_value("continuous only").run()
    assert _pushed(at, "override_continuous_intercept")           # first drawn in this run


@pytest.mark.parametrize("mode", list(DONATION_INTERCEPT_INPUTS))
def test_unchanged_rerun_does_not_repush_the_donation_intercept_inputs(mode):
    at = _donation_page2()
    at.radio(key="page2_tab_income_spec_mode").set_value(mode).run()
    at.run()                                                      # the tab drawn, nothing changed
    assert not at.exception
    for key in DONATION_INTERCEPT_INPUTS[mode] + ["override_adjustment_shift"]:
        assert not _pushed(at, key), f"{key} was written through the Session State API on an unchanged rerun"

    # a click is the user's own value: it is saved, reaches the seam's key and the
    # toast, and the next unchanged rerun pushes nothing either
    for key in DONATION_INTERCEPT_INPUTS[mode]:
        before = at.number_input(key=key).value
        at.number_input(key=key).increment().run()
        assert not at.exception
        assert at.number_input(key=key).value == pytest.approx(before + 0.001)
        assert at.session_state[CANONICAL_INTERCEPT[key]] == pytest.approx(before + 0.001)
        assert any("auto-saved" in str(t.value) for t in at.toast), [t.value for t in at.toast]
        at.run()
        assert not _pushed(at, key)
        assert at.number_input(key=key).value == pytest.approx(before + 0.001)


def test_a_reset_before_the_input_was_ever_drawn_does_not_shadow_a_later_override():
    """The resets write BOTH intercept keys, including the one whose input the income
    mode does not draw; Streamlit keeps that write under the user key and lets it
    resurface once the input has been drawn and culled. The typed override must
    still win when the input comes back (the input is seeded on every return)."""
    at = _donation_page2()                                        # categorical: the Continuous input never drawn
    at.button(key="reset_intercept_btn").click().run()
    at.run()
    at.radio(key="page2_tab_income_spec_mode").set_value("continuous only").run()
    at.number_input(key="override_continuous_intercept").increment().run()
    typed = at.number_input(key="override_continuous_intercept").value
    assert at.session_state["donation_coeff_intercept_cont"] == pytest.approx(typed)
    at.radio(key="page2_tab_income_spec_mode").set_value("categorical only").run()
    at.radio(key="page2_tab_income_spec_mode").set_value("continuous only").run()
    assert at.number_input(key="override_continuous_intercept").value == pytest.approx(typed)
    assert at.session_state["donation_coeff_intercept_cont"] == pytest.approx(typed)
    at.run()
    assert not _pushed(at, "override_continuous_intercept")
