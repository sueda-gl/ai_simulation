"""
Lavie 2026-10: "When running decision elements only, below the '4. Rejected Transaction
Defaults' title, add a subtitle for the decision element to which the results
correspond, e.g., in the results for Option List Length, add '4.1 Options List Length
(Tendency to Plan)' subtitle. Same for other decision elements."

The subtitle is the element's Decision 4 sub-tab number and name, drawn directly under
the decision title with the title's own heading class; whole-decision runs get none.
"""
import pytest

from app.reports.rtd import RTD_ELEMENT_SUBTITLES, rtd_element_subtitle

EXPECTED = {
    'rtd_run_ttp_btn': "4.1 Options List Length (Tendency to Plan)",
    'rtd_run_loyalty_btn': "4.2 Loyalty Ranking",
    'rtd_run_wtp_btn': "4.3 Willingness-to-Pay Ranking",
    'rtd_run_risk_taking_btn': "4.4 Risk-Taking Ranking",
    'rtd_run_flexibility_btn': "4.5 Flexibility Ranking",
    'rtd_run_aggregation_btn': "4.6 Integrated Default List (Rank Aggregation)",
}


def test_subtitles_follow_the_decision_4_tab():
    from app.pages.decision_tabs.rejected_transaction import (
        AGGREGATION_TITLE, MECH_TITLES)
    for mech, tab_title in MECH_TITLES.items():
        index, name = RTD_ELEMENT_SUBTITLES[mech]
        assert tab_title == f"{index}. {name}"
    index, name = RTD_ELEMENT_SUBTITLES['aggregation']
    assert AGGREGATION_TITLE == f"{index}. {name}"
    assert rtd_element_subtitle(4, 'ttp') == "4.1 Options List Length (Tendency to Plan)"
    assert rtd_element_subtitle(4, None) is None


def _app_script():
    import streamlit as st
    from app.models import initialize_session_state
    initialize_session_state()
    st.session_state.population_mode = st.session_state.get('_test_population',
                                                            'Research Baseline')
    st.session_state.n_agents = 40
    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()
    if st.session_state.get('simulation_results'):
        from app.pages.results.main_results import render_single_run_results
        render_single_run_results()


def _markdown(at):
    return [str(m.value) for m in at.main.get("markdown")]


def _subtitles(at):
    return [m for m in _markdown(at) if m.startswith('<h5 class="subsection-header"')]


def _title_index(md):
    return next(i for i, m in enumerate(md)
                if 'subsection-header' in m and '4. Rejected Transaction Defaults' in m)


@pytest.mark.parametrize("button, subtitle", list(EXPECTED.items()))
def test_element_run_shows_its_subtitle_under_the_decision_title(button, subtitle):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_app_script, default_timeout=600)
    at.run()
    at.button(key=button).click().run()
    assert not at.exception
    md = _markdown(at)
    title = _title_index(md)
    assert md[title].startswith('<h4 class="subsection-header">')
    # directly below the title, same heading class, no extra bold
    assert md[title + 1] == (f'<h5 class="subsection-header" style="margin-top:0">'
                             f'{subtitle}</h5>')
    assert len(_subtitles(at)) == 1


def test_whole_decision_run_has_no_element_subtitle():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_app_script, default_timeout=600)
    at.run()
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run()
    assert not at.exception
    assert _subtitles(at) == []


def test_comparison_element_run_shows_the_subtitle_once():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_app_script, default_timeout=600)
    at.session_state['_test_population'] = 'Compare all'
    at.run()
    at.button(key='rtd_run_wtp_btn').click().run()
    assert not at.exception
    md = _markdown(at)
    assert md[_title_index(md) + 1].endswith('>4.3 Willingness-to-Pay Ranking</h5>')
    assert len(_subtitles(at)) == 1
