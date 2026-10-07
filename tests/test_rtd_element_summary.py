"""
Lavie #8 (professor 2026-10): after "Use This Config" on a Decision 4 ELEMENT page, the
selected-configuration summary must not show list-length averages unrelated to that
element. The summary keeps only what belongs to the element:

* whole decision run                         -> options list length AND integrated default list length;
* Options List Length (ttp) element          -> options list length only;
* Integrated Default List element            -> integrated default list length only;
* Loyalty / WTP / Risk-Taking / Flexibility  -> neither.

Checked on the results page (the selected-configuration card and the "saved
configuration(s) will be used" line of the Run Complete Simulation section) and on the
Page-2 saved-configuration display (rendered in the sidebar here, to keep it apart).
"""
import pytest

from app.reports.rtd import rtd_summary_metric_keys


def test_summary_metric_keys_per_element():
    assert rtd_summary_metric_keys(None) == ('mean_choice_length', 'mean_default_list_length')
    assert rtd_summary_metric_keys('ttp') == ('mean_choice_length',)
    assert rtd_summary_metric_keys('aggregation') == ('mean_default_list_length',)
    for element in ('loyalty', 'wtp', 'risk_taking', 'flexibility'):
        assert rtd_summary_metric_keys(element) == ()


def _app_script():
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()
    st.session_state.population_mode = 'Research Baseline'
    st.session_state.n_agents = 40

    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()

    if st.session_state.get('simulation_results'):
        from app.pages.results.main_results import render_single_run_results
        render_single_run_results()

    with st.sidebar:
        from app.pages.page2_decisions import render_selected_rejected_transaction_config_display
        render_selected_rejected_transaction_config_display()


def _main_captions(at):
    return [str(c.value) for c in at.main.get("caption")]


def _sidebar_texts(at):
    out = [str(c.value) for c in at.sidebar.get("caption")]
    out += [str(m.label) for m in at.sidebar.get("metric")]
    return out


@pytest.mark.parametrize("button,element,expect_options,expect_integrated", [
    ("rtd_run_loyalty_btn", "loyalty", False, False),
    ("rtd_run_flexibility_btn", "flexibility", False, False),
    ("rtd_run_ttp_btn", "ttp", True, False),
    ("rtd_run_aggregation_btn", "aggregation", False, True),
    ("run_rejected_transaction_defaults_only_btn", None, True, True),
])
def test_selected_config_summary_shows_only_the_page_element(button, element,
                                                              expect_options, expect_integrated):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_app_script)
    at.run(timeout=600)
    assert not at.exception
    at.button(key=button).click().run(timeout=600)
    assert not at.exception
    result_key = next(iter(at.session_state['simulation_results'].keys()))

    at.button(key=f"rtd_inline_select_{result_key}").click().run(timeout=600)
    assert not at.exception
    cfg = at.session_state['selected_decision_configs']['rejected_transaction_defaults']
    assert cfg['run_element'] == element
    # both averages are still recorded in the configuration; only the display changes
    assert 'mean_choice_length' in cfg['metrics'] and 'mean_default_list_length' in cfg['metrics']

    # results page: the selected-configuration card is shown (Clear + Run Complete)
    assert at.button(key='clear_rtd_selection') is not None
    captions = _main_captions(at)
    card_options = [c for c in captions if c.startswith("Avg options list length:")]
    card_integrated = [c for c in captions if c.startswith("Avg integrated default list length:")]
    assert bool(card_options) is expect_options
    assert bool(card_integrated) is expect_integrated
    used = [c for c in captions if c.strip().startswith("✅ Rejected Transaction Defaults:")]
    assert len(used) == 1
    assert ("avg options list length" in used[0]) is expect_options
    if not expect_options:
        assert ("avg integrated default list length" in used[0]) is expect_integrated

    # Page 2 saved-configuration display
    page2 = _sidebar_texts(at)
    assert ("Avg. Options List Length" in page2) is expect_options
    page2_integrated = [t for t in page2 if t.lower().startswith("avg. integrated default list length")]
    assert bool(page2_integrated) is expect_integrated
