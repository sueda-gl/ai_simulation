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

Second bug (user report, screenshots): in SINGLE population mode an individual
disclose_income (or disclose_documents) run rendered its full results once in
the Decision Results section ("✅ 1. Disclose Income (Custom Parameters)":
metrics, distribution, statistics, classification) and then AGAIN via the
generic income-type overview ("📊 Simulation Overview" +
"Simulation Overview (Continuous)" + "📊 Disclose Income Analysis
(Continuous)") - the same histogram/statistics/classification twice. This
affected both "Continuous only" and "Categorical only" (it predates the
case-sensitivity fix). Fixed in main_results.py by skipping the generic
overview for individual single-mode runs whose decision-specific section
already rendered; comparison modes ("Compare all" / "Compare both") and the
Decision 4 summary-first layout are unchanged.
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


# ---------------------------------------------------------------------------
# SINGLE population mode: no duplicate generic overview after the decision's
# own results section
# ---------------------------------------------------------------------------

def _di_single_mode_app_script():
    """Decision 1 tab in the default SINGLE population mode + the results page."""
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()  # population_mode default: 'Copula (synthetic)'
    st.session_state.n_agents = 60

    from app.pages.decision_tabs.disclose_income import render_disclose_income_tab
    render_disclose_income_tab()

    if st.session_state.get('simulation_results'):
        from app.pages.results.main_results import render_single_run_results
        render_single_run_results()


def _dd_single_mode_app_script():
    """Decision 2 tab in the default SINGLE population mode + the results page."""
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()
    st.session_state.n_agents = 60

    from app.pages.decision_tabs.disclose_documents import render_disclose_documents_tab
    render_disclose_documents_tab()

    if st.session_state.get('simulation_results'):
        from app.pages.results.main_results import render_single_run_results
        render_single_run_results()


def _assert_no_duplicate_overview(at, decision_header):
    """The decision section renders exactly once; the generic income-type
    overview block ('📊 Simulation Overview' + show_overview's
    'Simulation Overview (...)' / '... Analysis (...)') must NOT repeat it."""
    md = [str(m.value) for m in at.markdown]
    assert sum(decision_header in m for m in md) == 1, \
        [m for m in md if decision_header in m]

    assert not any("📊 Simulation Overview" in m for m in md), \
        "generic overview header duplicated the decision results"
    subheaders = [str(s.value) for s in at.subheader]
    dup = [s for s in subheaders
           if s.startswith("Simulation Overview (") or "Analysis (" in s]
    assert not dup, f"duplicate generic analysis block rendered: {dup}"


def test_disclose_income_single_mode_continuous_only_no_duplicate_block():
    at = _run_continuous_only(_di_single_mode_app_script,
                              'di_tab_income_mode', 'di_income_mode',
                              'run_disclose_income_only_btn')
    assert sorted(at.session_state['simulation_results'].keys()) == ['continuous']

    # Decision-specific section rendered once, with its own metrics
    _assert_no_duplicate_overview(at, "✅ 1. Disclose Income (Custom Parameters)")
    metric_labels = [m.label for m in at.metric]
    assert metric_labels.count("Disclosure Rate") == 1, metric_labels
    assert metric_labels.count("Disclosed income (Y)") == 1, metric_labels
    # generic overview's headline metric must not appear
    assert "Disclose Income (Y)" not in metric_labels, metric_labels


def test_disclose_documents_single_mode_continuous_only_no_duplicate_block():
    at = _run_continuous_only(_dd_single_mode_app_script,
                              'dd_tab_income_mode', 'dd_income_mode',
                              'run_disclose_documents_only_btn')
    assert sorted(at.session_state['simulation_results'].keys()) == ['continuous']

    _assert_no_duplicate_overview(at, "✅ 2. Disclose Documents (Custom Parameters)")
    metric_labels = [m.label for m in at.metric]
    assert metric_labels.count("Disclosed documents (Y)") == 1, metric_labels
    # generic overview's headline metrics must not appear
    assert "Disclose Documents Rate (qualified)" not in metric_labels, metric_labels


def test_disclose_income_single_mode_compare_both_grid_unchanged():
    """'Compare both' income spec in single population mode: the per-income-type
    overview grid IS the primary display and must still render (one overview +
    one analysis cell per income type, no decision-results duplication)."""
    at = _run_continuous_only(_di_single_mode_app_script,
                              'di_tab_income_mode', 'di_income_mode',
                              'run_disclose_income_only_btn')
    # switch to Compare both and re-run
    at.session_state['di_tab_income_mode'] = 'Compare both'
    at.session_state['di_income_mode'] = 'Compare both'
    at.run(timeout=600)
    at.button(key='run_disclose_income_only_btn').click().run(timeout=600)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert sorted(at.session_state['simulation_results'].keys()) == \
        ['categorical', 'continuous']

    subheaders = [str(s.value) for s in at.subheader]
    assert subheaders.count("Simulation Overview (Categorical)") == 1, subheaders
    assert subheaders.count("Simulation Overview (Continuous)") == 1, subheaders
    assert subheaders.count("📊 Disclose Income Analysis (Categorical)") == 1, subheaders
    assert subheaders.count("📊 Disclose Income Analysis (Continuous)") == 1, subheaders
