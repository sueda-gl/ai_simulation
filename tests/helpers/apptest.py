"""Readers for a Streamlit ``AppTest`` page, and the Decision 4 app script that
tests/test_rtd_batch4_ui.py and tests/test_rtd_results_ui.py both drive."""


def all_markdown(at):
    """The text of every ``st.markdown`` element, in the order AppTest lists them, one per line."""
    return "\n".join(str(m.value) for m in at.markdown)


def all_captions(at):
    """The text of every ``st.caption`` element, in the order AppTest lists them."""
    return [str(c.value) for c in at.caption]


def rtd_tab_and_results_script():
    """Decision 4 tab + (once results exist) the results page, end to end.

    Research Baseline, 60 agents; a test sets the tab's income mode or presses its run
    buttons. ``AppTest.from_function`` runs this function's source on its own, so the
    imports stay inside it.
    """
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()
    st.session_state.population_mode = 'Research Baseline'
    st.session_state.n_agents = 60

    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()

    if st.session_state.get('simulation_results'):
        from app.pages.results.main_results import render_single_run_results
        render_single_run_results()
