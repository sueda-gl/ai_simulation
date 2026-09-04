"""
Complete (combined) simulation with Decision 4 as the customised decision: the
Decision 4 section of the results page must show ONLY the integrated ranking - never
the other decisions' analyses (donation rate, disclose income, ...) and never the
comparison scaffolding - even when a comparison setting ("Compare both" income spec
from the Decision 4 tab, or "Compare all" population mode) is still active in the
session from the earlier individual run. Regression for the professor's screenshot
(2026-09-04): the Decision 4 section of a complete run rendered "Categorical Income
Treatment" + "Simulation Overview (Categorical)" + "Donation Rate Analysis".
"""


def _combined_run_script():
    import streamlit as st
    from app.models import initialize_session_state, ALL_DECISIONS

    initialize_session_state()
    st.session_state.population_mode = 'Research Baseline'
    st.session_state.income_spec_mode = 'Compare both'     # stale comparison setting
    st.session_state.n_agents = 30

    if not st.session_state.get('simulation_results'):
        from src.orchestrator_baseline import OrchestratorBaseline
        from app.simulation import _apply_rejected_transaction_config, _load_original_participants
        orch = OrchestratorBaseline()
        _apply_rejected_transaction_config(orch, "baseline", "categorical")
        agents = _load_original_participants(30, 1, random_sample=False)
        df = orch.run_simulation(30, 1, None, agents_df=agents)     # every decision
        st.session_state.simulation_results = {'categorical': df}
        st.session_state.custom_decisions = ['rejected_transaction_defaults']
        st.session_state.default_decisions = [d for d in ALL_DECISIONS
                                              if d != 'rejected_transaction_defaults']

    from app.pages.results.main_results import render_single_run_results
    render_single_run_results()


def _texts(at):
    out = []
    for coll in (at.markdown, at.subheader, at.header, at.caption, at.info, at.success):
        out.extend(str(e.value) for e in coll)
    return "\n".join(out)


def test_combined_run_decision4_section_shows_integrated_ranking_only():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_combined_run_script)
    at.run(timeout=900)
    assert not at.exception
    df = at.session_state['simulation_results']['categorical']
    assert 'rtd_default_list' in df.columns and 'donation_default' in df.columns

    text = _texts(at)
    # Decision 4 section present with the integrated ranking only
    assert "4. Rejected Transaction Defaults (Custom Parameters)" in text
    assert "6️⃣ Integrated Default List (Rank Aggregation)" in text
    for element in ("1️⃣ Options List Length (Tendency to Plan)", "2️⃣ Loyalty Ranking",
                    "3️⃣ Willingness-to-Pay Ranking", "4️⃣ Risk-Taking Ranking",
                    "5️⃣ Cognitive Flexibility Ranking"):
        assert element not in text, element
    # no comparison scaffolding and no other decision's analysis inside the section
    assert "Categorical Income Treatment" not in text
    assert "Continuous Income Treatment" not in text
    assert "Simulation Overview (Categorical)" not in text
    assert "Donation Rate Analysis" not in text
    assert "Disclose Income Analysis" not in text
