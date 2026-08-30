"""Regression tests: individual Decision 1 / Decision 2 runs with the
"Continuous only" income specification must render results.

Bug (professor report): in "Compare all" population mode, an individual
disclose_income or disclose_documents run with "Continuous only" produced
result keys copula_continuous / research_spec_continuous /
research_baseline_continuous, but the results page resolved the income type
with a case-sensitive check (income_spec_mode == "continuous only", the
lowercase donation-tab label) while these decisions sync the capitalized
"Continuous only" into income_spec_mode. The comparison grid then looked up
the *_categorical keys and rendered only "results not available" captions -
an empty results page. "Compare both" (and "Categorical only") worked.

Fixed by normalizing the check case-insensitively in
app/pages/results/comparisons.py and app/pages/results/main_results.py.
"""


def _di_compare_all_app_script():
    """Decision 1 tab in Compare-all population mode + the results page."""
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()
    st.session_state.population_mode = 'Compare all'
    st.session_state.n_agents = 60

    from app.pages.decision_tabs.disclose_income import render_disclose_income_tab
    render_disclose_income_tab()

    if st.session_state.get('simulation_results'):
        from app.pages.results.main_results import render_single_run_results
        render_single_run_results()


def _dd_compare_all_app_script():
    """Decision 2 tab in Compare-all population mode + the results page."""
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()
    st.session_state.population_mode = 'Compare all'
    st.session_state.n_agents = 60

    from app.pages.decision_tabs.disclose_documents import render_disclose_documents_tab
    render_disclose_documents_tab()

    if st.session_state.get('simulation_results'):
        from app.pages.results.main_results import render_single_run_results
        render_single_run_results()


CONTINUOUS_KEYS = ['copula_continuous', 'research_baseline_continuous',
                   'research_spec_continuous']


def _run_continuous_only(app_script, tab_key, ss_key, run_btn_key):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(app_script)
    at.session_state[tab_key] = 'Continuous only'
    at.session_state[ss_key] = 'Continuous only'
    at.run(timeout=600)
    assert not at.exception

    at.button(key=run_btn_key).click().run(timeout=600)
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _assert_grid_rendered(at):
    """The Compare-all grid must show every continuous cell, none missing."""
    assert sorted(at.session_state['simulation_results'].keys()) == CONTINUOUS_KEYS

    captions = [str(c.value) for c in at.caption]
    missing = [c for c in captions if 'results not available' in c]
    assert not missing, f"empty comparison cells rendered: {missing}"

    md = "\n".join(str(m.value) for m in at.markdown)
    assert "All Population Modes Comparison" in md

    # one non-empty overview cell per population mode
    metric_labels = [m.label for m in at.metric]
    assert metric_labels.count("Total Agents") >= 3, metric_labels


def test_disclose_income_continuous_only_compare_all_renders():
    at = _run_continuous_only(_di_compare_all_app_script,
                              'di_tab_income_mode', 'di_income_mode',
                              'run_disclose_income_only_btn')
    _assert_grid_rendered(at)
    metric_labels = [m.label for m in at.metric]
    assert metric_labels.count("Disclose Income (Y)") >= 3, metric_labels


def test_disclose_documents_continuous_only_compare_all_renders():
    at = _run_continuous_only(_dd_compare_all_app_script,
                              'dd_tab_income_mode', 'dd_income_mode',
                              'run_disclose_documents_only_btn')
    _assert_grid_rendered(at)
    metric_labels = [m.label for m in at.metric]
    assert metric_labels.count("Disclose Documents Rate (qualified)") >= 3, metric_labels
