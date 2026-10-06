"""
Complete-run income labelling (2026-10-07).

A complete simulation names its sub-runs / result keys after the GLOBAL income mode,
but Decision 4 runs with its own tab's income mode (``build_rejected_transaction_patch``;
a tab set to "Compare both" falls back to continuous). The run metadata now records
the mode every income-dependent decision actually ran with, and the results page labels
the Decision 4 section with it - plus an info note when the tab said "Compare both".
The numbers do not change: Decision 4 still runs continuous.
"""
import pytest

from app.seam import build_plan as bp
from app.seam.config_repo import get_config_repo
from test_build_plan import ALL, base_state

RTD = "rejected_transaction_defaults"
OTHERS = [d for d in ALL if d != RTD]


@pytest.fixture(scope="module")
def repo():
    return get_config_repo()


@pytest.fixture(scope="module")
def default_values():
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    return DEFAULT_DECISION_VALUES


def _complete(repo, default_values, **overrides):
    state = base_state(repo, custom_decisions=[RTD], default_decisions=OTHERS, **overrides)
    return bp.build_run_plan(state, repo, default_decision_values=default_values)


def test_complete_run_records_decision_4_own_mode(repo, default_values):
    plan = _complete(repo, default_values, income_spec_mode="categorical only",
                     rtd_income_mode="Continuous only")
    assert plan.result_keys == ("categorical",)
    modes = plan.metadata.decision_income_modes["categorical"]
    assert modes == {"disclose_income": "categorical", "disclose_documents": "categorical",
                     "donation_default": "categorical", RTD: "continuous"}
    # the recorded mode is the mode the patch hands the engine
    assert plan.sub_runs[0].decision_config_patches[RTD]["income_mode"] == "continuous"
    assert plan.metadata.rtd_compare_both_fallback is False


def test_complete_run_with_compare_both_tab_falls_back_to_continuous(repo, default_values):
    plan = _complete(repo, default_values, income_spec_mode="categorical only",
                     rtd_income_mode="Compare both")
    assert plan.sub_runs[0].decision_config_patches[RTD]["income_mode"] == "continuous"
    assert plan.metadata.decision_income_modes["categorical"][RTD] == "continuous"
    assert plan.metadata.rtd_compare_both_fallback is True


def test_complete_compare_both_run_keeps_decision_4_on_its_tab_mode(repo, default_values):
    plan = _complete(repo, default_values, income_spec_mode="Compare both",
                     rtd_income_mode="Categorical only")
    assert plan.result_keys == ("categorical", "continuous")
    for key in plan.result_keys:
        modes = plan.metadata.decision_income_modes[key]
        assert modes[RTD] == "categorical"                 # its own tab mode in both sub-runs
        assert modes["donation_default"] == key            # the sub-run's mode
        assert modes["disclose_income"] == key


def test_individual_decision_4_compare_both_is_not_a_fallback(repo, default_values):
    state = base_state(repo, selected=[RTD], custom_decisions=[RTD], default_decisions=[],
                       rtd_income_mode="Compare both")
    plan = bp.build_run_plan(state, repo, default_decision_values=default_values)
    assert plan.metadata.decision_income_modes == {"categorical": {RTD: "categorical"},
                                                   "continuous": {RTD: "continuous"}}
    assert plan.metadata.rtd_compare_both_fallback is False


# ---------------------------------------------------------------------------
# Results page: the Decision 4 section is labelled with the mode it really ran
# ---------------------------------------------------------------------------
def _results_script():
    import streamlit as st
    from app.models import initialize_session_state, ALL_DECISIONS

    initialize_session_state()
    if not st.session_state.get('simulation_results'):
        st.session_state.population_mode = 'Research Baseline'
        st.session_state.income_spec_mode = 'categorical only'
        st.session_state.n_agents = 30
        st.session_state.rtd_income_mode = 'Compare both'
        st.session_state.decision_params.selected_decisions = list(ALL_DECISIONS)
        st.session_state.custom_decisions = ['rejected_transaction_defaults']
        st.session_state.default_decisions = [d for d in ALL_DECISIONS
                                              if d != 'rejected_transaction_defaults']
        from app.seam.config_repo import get_config_repo
        from app.seam.execute import execute
        from app.simulation import _run_metadata_dict, build_plan_from_session
        plan = build_plan_from_session()
        st.session_state.simulation_results = execute(plan, config_repo=get_config_repo())
        st.session_state._run_metadata = _run_metadata_dict(plan.metadata)
        # the user changes the tab AFTER the run: the page must still describe the run
        st.session_state.rtd_income_mode = 'Categorical only'

    from app.pages.results.main_results import render_single_run_results
    render_single_run_results()


def test_results_page_labels_decision_4_with_the_mode_it_ran():
    from streamlit.testing.v1 import AppTest
    from app.pages.results.main_results import RTD_COMPARE_BOTH_NOTE

    at = AppTest.from_function(_results_script)
    at.run(timeout=900)
    assert not at.exception
    meta = at.session_state['_run_metadata']
    assert meta['result_keys'] == ['categorical']
    assert meta['decision_income_modes']['categorical'][RTD] == 'continuous'
    assert meta['rtd_compare_both_fallback'] is True
    df = at.session_state['simulation_results']['categorical']
    assert set(df['rtd_income_mode']) == {'continuous'}             # numbers unchanged

    captions = [str(c.value) for c in at.caption]
    assert "⚙️ **Current Settings:** Continuous only" in captions   # not the key's 'categorical'
    assert "⚙️ **Current Settings:** Categorical only" not in captions
    assert "⚙️ **Current Settings:** Compare both" not in captions
    infos = [str(i.value) for i in at.info]
    assert infos.count(RTD_COMPARE_BOTH_NOTE) == 1
    assert "Decision 4 ran with **continuous** income" in RTD_COMPARE_BOTH_NOTE
