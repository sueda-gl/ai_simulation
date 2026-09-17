"""
Complete (combined) simulation with Decision 4 as the customised decision.

Per the professor's 2026-09 display specification the Decision 4 section of such a run
shows ONLY the integrated results - the "% of agents with default list length" chart
with its table and the "% of first integrated default option" chart with its table and
the option explanations. No element sections, no descriptive paragraph, no tie
statistics and no "most common integrated default lists" table; the element values stay
in the run's agent-level Excel.

It must also never show the other decisions' analyses (donation rate, disclose income,
...) or the comparison scaffolding - even when a comparison setting ("Compare both"
income spec from the Decision 4 tab, or "Compare all" population mode) is still active
in the session from an earlier individual run. Regression for the professor's
screenshot (2026-09-04): the Decision 4 section of a complete run rendered "Categorical
Income Treatment" + "Simulation Overview (Categorical)" + "Donation Rate Analysis".
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


ELEMENT_SECTIONS = ("1️⃣ Options List Length (Tendency to Plan)", "2️⃣ Loyalty Ranking",
                    "3️⃣ Willingness-to-Pay Ranking", "4️⃣ Risk-Taking Ranking",
                    "5️⃣ Flexibility Ranking")


def _texts(at):
    out = []
    for coll in (at.markdown, at.subheader, at.header, at.caption, at.info, at.success):
        out.extend(str(e.value) for e in coll)
    return "\n".join(out)


def test_combined_run_decision4_section_shows_integrated_results_only():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_combined_run_script)
    at.run(timeout=900)
    assert not at.exception
    df = at.session_state['simulation_results']['categorical']
    assert 'rtd_default_list' in df.columns and 'donation_default' in df.columns

    text = _texts(at)
    # Decision 4 section present with the integrated results only
    assert "4. Rejected Transaction Defaults (Custom Parameters)" in text
    assert "6️⃣ Integrated Default List" in text
    for element in ELEMENT_SECTIONS:
        assert element not in text, element
    # the option explanations travel with the first-option chart
    md = [str(m.value) for m in at.markdown]
    assert md.count("Option 3: current vendor at PN price") == 1
    assert md.count("Option 5: forgo the transaction") == 1
    # no descriptive paragraph, no tie statistics, no most-common-lists table
    assert "presents each customer's final pre-selected default list" not in text
    assert "Tie statistics of the Kemeny aggregation" not in text
    assert "Stage that settled the ranking" not in text
    assert "Kemeny outcome" not in text
    assert "Most common integrated default lists" not in text
    assert "List cut by" not in text
    # no score charts -> no Min/Max captions
    assert not any(str(c.value).startswith("Min ") for c in at.caption)
    # no comparison scaffolding and no other decision's analysis inside the section
    assert "Categorical Income Treatment" not in text
    assert "Continuous Income Treatment" not in text
    assert "Income-Independent Elements" not in text
    assert "Simulation Overview (Categorical)" not in text
    assert "Donation Rate Analysis" not in text
    assert "Disclose Income Analysis" not in text
    # professor 2026-09-17: the complete simulation offers the same Decision 4 workbook
    # as the whole-decision run
    labels = [str(e.label) for e in at.get("download_button")]
    assert labels.count("📊 Download Decision 4 Excel (all elements)") == 1


def test_combined_agent_level_export_carries_every_decision4_variable():
    """The element values must stay available in the complete simulation's agent-level
    Excel even though the page shows only the integrated results."""
    from streamlit.testing.v1 import AppTest
    from app.pages.results.components.export_section import _build_agent_level_dataframe

    at = AppTest.from_function(_combined_run_script)
    at.run(timeout=900)
    assert not at.exception
    df = at.session_state['simulation_results']['categorical']

    agent_df = _build_agent_level_dataframe(df)
    assert len(agent_df) == len(df)
    columns = set(agent_df.columns)
    # inputs + the z-scores the equations use
    for col in ('rtd_Conscientiousness', 'rtd_Education', 'rtd_stdactions',
                'rtd_z_extraversionbig5', 'rtd_z_agreeable', 'rtd_z_neuroticismbig5',
                'rtd_z_conscientiousnessbig5', 'rtd_z_opennessbig5', 'rtd_reducation',
                'rtd_z_net_income', 'rtd_z_stdactions'):
        assert col in columns, col
    # element scores / intermediates / segments / lists
    for col in ('rtd_weighted_ttp', 'rtd_weighted_ttp06', 'rtd_choice_length_deterministic',
                'rtd_choice_length', 'rtd_flex_ivw', 'rtd_flex_z_ivw'):
        assert col in columns, col
    for element in ('rtd_loyalty', 'rtd_wtp', 'rtd_risk_taking', 'rtd_flexibility'):
        for suffix in ('_score', '_z', '_segment_deterministic', '_segment', '_ranking'):
            assert f'{element}{suffix}' in columns, f'{element}{suffix}'
    # integration fields
    for col in ('rtd_default_list', 'rtd_integrated_ranking', 'rtd_consensus_kemeny_status',
                'rtd_consensus_n_kemeny_optimal', 'rtd_consensus_is_kemeny_optimal',
                'rtd_consensus_settled_by', 'rtd_consensus_truncated_by',
                'rtd_default_list_length'):
        assert col in columns, col
    assert 'rtd_consensus_ranking' not in columns     # renamed (professor 2026-09-17)
