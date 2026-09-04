"""
UI / plumbing tests for the Decision 4 per-element run workflow (Batch 4):

- Per-element Run buttons on the Decision 4 tab set st.session_state.rtd_run_element
  and trigger the SAME individual-decision run as the whole-decision button; the
  whole-decision button clears the flag again.
- The results page presents ONLY the selected element's section (title, summary,
  charts, per-element Excel) on a per-element run, and all four sections on a
  whole-decision run; in Compare-both income mode the element filter applies inside
  each comparison column.
- The per-element and whole-decision (per-element-sheets) Excel builders produce
  the specified sheet names and column sets, with choice1..choice5 blank beyond the
  list length.
- A Decision 4-only run's Export Results section offers no transaction-level Excel,
  and the Simulation Overview shows D4 metrics (Total Agents / Avg. options list
  length).
"""
import numpy as np
import pandas as pd
import pytest

from app.pages.results.visualizations.transaction_viz import (
    _prepare_rtd_element_export,
    _prepare_rtd_model_export,
    _RTD_ELEMENT_SHEETS,
)

RTD_SECTION_TITLES = {
    'ttp': "1️⃣ Options List Length (Tendency to Plan)",
    'loyalty': "2️⃣ Loyalty Ranking",
    'wtp': "3️⃣ Willingness-to-Pay Ranking",
    'risk_taking': "4️⃣ Risk-Taking Ranking",
    'flexibility': "5️⃣ Cognitive Flexibility Ranking",
}


def _build_rtd_frame(inc_mode):
    """Individual Decision 4 model run on the first 60 original participants."""
    from src.orchestrator_baseline import OrchestratorBaseline
    from app.simulation import _apply_rejected_transaction_config

    orch = OrchestratorBaseline()
    _apply_rejected_transaction_config(orch, "baseline", inc_mode)
    agents = orch.original_data.iloc[:60].copy()
    agents.index = range(len(agents))
    return orch.run_simulation(len(agents), 123, ['rejected_transaction_defaults'],
                               agents_df=agents)


@pytest.fixture(scope="module")
def rtd_frame():
    return _build_rtd_frame("continuous")


@pytest.fixture(scope="module")
def rtd_frame_cat():
    return _build_rtd_frame("categorical")


# ---------------------------------------------------------------------------
# Excel builders (verification item iii)
# ---------------------------------------------------------------------------
def test_whole_decision_workbook_sheets_and_columns(rtd_frame):
    sheets = _prepare_rtd_model_export(rtd_frame)
    # professor 2026-09: the final ranking first, then all elements side by side
    # (inputs -> element by element -> final), then one sheet per element
    assert list(sheets.keys()) == ['Integrated Default List', 'All Elements',
                                   'Options List Length', 'Loyalty',
                                   'Willingness-to-Pay', 'Risk-Taking',
                                   'Cognitive Flexibility']
    assert list(sheets['All Elements'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'Agreeable', 'NeuroticismBig5',
        'ConscientiousnessBig5', 'OpennessBig5', 'Education', 'income', 'stdactions',
        'weighted_ttp', 'weighted_ttp06', 'choice_length_deterministic', 'choice_length',
        'loyalty_score', 'z_loyalty', 'loyalty_segment_deterministic', 'loyalty_segment', 'loyalty_list',
        'WTP_score', 'z_WTP', 'WTP_segment_deterministic', 'WTP_segment', 'WTP_list',
        'RT_score', 'z_RT', 'RT_segment_deterministic', 'RT_segment', 'RT_list',
        'Flexibility_calculated_ivw', 'z_Flexibility_calculated_ivw', 'z_stdactions',
        'Flexibility_score', 'z_Flexibility', 'Flexibility_segment_deterministic',
        'Flexibility_segment', 'Flexibility_list',
        'consensus_ranking', 'settled_by', 'truncated_by', 'default_list_length',
        'final_choice1', 'final_choice2', 'final_choice3', 'final_choice4', 'final_choice5']

    assert list(sheets['Options List Length'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'Agreeable', 'NeuroticismBig5',
        'ConscientiousnessBig5', 'Education', 'weighted_ttp', 'weighted_ttp06',
        'choice_length_deterministic', 'choice_length', 'sigma_used_ttp']
    assert list(sheets['Loyalty'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'OpennessBig5', 'Agreeable',
        'loyalty_score', 'z_loyalty', 'loyalty_segment_deterministic',
        'loyalty_segment', 'choice1', 'choice2', 'choice3', 'choice4', 'choice5',
        'sigma_used_loyalty']
    assert list(sheets['Willingness-to-Pay'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'Agreeable', 'income',
        'WTP_score', 'z_WTP', 'WTP_segment_deterministic', 'WTP_segment',
        'choice1', 'choice2', 'choice3', 'choice4', 'choice5', 'sigma_used_WTP']
    assert list(sheets['Risk-Taking'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'OpennessBig5', 'Agreeable',
        'ConscientiousnessBig5', 'NeuroticismBig5', 'income',
        'RT_score', 'z_RT', 'RT_segment_deterministic', 'RT_segment',
        'choice1', 'choice2', 'choice3', 'choice4', 'choice5', 'sigma_used_RT']
    # Cognitive Flexibility: Big 5 + stdactions inputs, the Stata intermediates, the
    # anchored score / its z, segments, choices, sigma.
    assert list(sheets['Cognitive Flexibility'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'OpennessBig5', 'NeuroticismBig5', 'Agreeable',
        'ConscientiousnessBig5', 'stdactions',
        'Flexibility_calculated_ivw', 'z_Flexibility_calculated_ivw', 'z_stdactions',
        'Flexibility_score', 'z_Flexibility', 'Flexibility_segment_deterministic',
        'Flexibility_segment', 'choice1', 'choice2', 'choice3', 'choice4', 'choice5',
        'sigma_used_Flexibility']
    # Section-6 rank aggregation sheet: the mechanism inputs, the consensus ranking,
    # the tie-break diagnostics and the truncated integrated default list.
    assert list(sheets['Integrated Default List'].columns) == [
        'Agent ID', 'choice_length', 'loyalty_list', 'WTP_list', 'RT_list', 'Flexibility_list',
        'consensus_ranking', 'kemeny_status', 'n_kemeny_optimal', 'settled_by',
        'is_kemeny_optimal', 'truncated_by', 'default_list_length',
        'choice1', 'choice2', 'choice3', 'choice4', 'choice5']

    for sheet in sheets.values():
        assert len(sheet) == len(rtd_frame)


def test_element_exports_only_own_variables(rtd_frame):
    ttp = _prepare_rtd_element_export(rtd_frame, 'ttp')
    assert list(ttp.columns) == [
        'Agent ID', 'ExtraversionBig5', 'Agreeable', 'NeuroticismBig5',
        'ConscientiousnessBig5', 'Education', 'weighted_ttp', 'choice_length']

    loyalty = _prepare_rtd_element_export(rtd_frame, 'loyalty')
    assert list(loyalty.columns) == [
        'Agent ID', 'ExtraversionBig5', 'OpennessBig5', 'Agreeable',
        'loyalty_score', 'z_loyalty', 'loyalty_segment',
        'choice1', 'choice2', 'choice3', 'choice4', 'choice5']

    wtp = _prepare_rtd_element_export(rtd_frame, 'wtp')
    assert list(wtp.columns) == [
        'Agent ID', 'ExtraversionBig5', 'Agreeable', 'income',
        'WTP_score', 'z_WTP', 'WTP_segment',
        'choice1', 'choice2', 'choice3', 'choice4', 'choice5']

    rt = _prepare_rtd_element_export(rtd_frame, 'risk_taking')
    assert list(rt.columns) == [
        'Agent ID', 'ExtraversionBig5', 'OpennessBig5', 'Agreeable',
        'ConscientiousnessBig5', 'NeuroticismBig5', 'income',
        'RT_score', 'z_RT', 'RT_segment',
        'choice1', 'choice2', 'choice3', 'choice4', 'choice5']

    flex = _prepare_rtd_element_export(rtd_frame, 'flexibility')
    assert list(flex.columns) == [
        'Agent ID', 'ExtraversionBig5', 'OpennessBig5', 'NeuroticismBig5', 'Agreeable',
        'ConscientiousnessBig5', 'stdactions',
        'Flexibility_calculated_ivw', 'z_Flexibility_calculated_ivw', 'z_stdactions',
        'Flexibility_score', 'z_Flexibility', 'Flexibility_segment',
        'choice1', 'choice2', 'choice3', 'choice4', 'choice5']

    # exclusivity: no foreign independent variables leak into an element's file
    assert 'income' not in ttp.columns and 'income' not in loyalty.columns
    assert 'OpennessBig5' not in ttp.columns and 'OpennessBig5' not in wtp.columns
    assert 'Assigned Allowance Level' not in wtp.columns  # continuous frame
    assert 'income' not in flex.columns and 'stdactions' not in rt.columns


def test_categorical_frame_adds_allowance_level(rtd_frame_cat):
    assert str(rtd_frame_cat['rtd_income_mode'].iloc[0]) == 'categorical'
    sheets = _prepare_rtd_model_export(rtd_frame_cat)
    for name in ('Willingness-to-Pay', 'Risk-Taking'):
        assert 'Assigned Allowance Level' in sheets[name].columns
    for name in ('Options List Length', 'Loyalty', 'Cognitive Flexibility'):
        assert 'Assigned Allowance Level' not in sheets[name].columns
    for mech in ('wtp', 'risk_taking'):
        assert 'Assigned Allowance Level' in \
            _prepare_rtd_element_export(rtd_frame_cat, mech).columns
    for mech in ('ttp', 'loyalty', 'flexibility'):
        assert 'Assigned Allowance Level' not in \
            _prepare_rtd_element_export(rtd_frame_cat, mech).columns


def test_choice_sequences_follow_segment_and_blank_beyond_length(rtd_frame):
    """choice1..choice5 hold the Stata-direction priority tail: segment s gets the
    sequence from position s (choice1 = seq[s-1]); positions beyond are blank."""
    from src.decisions.rejected_transaction_defaults import PRIORITY_SEQUENCES
    for mech, seg_col in (('loyalty', 'loyalty_segment'), ('wtp', 'WTP_segment'),
                          ('risk_taking', 'RT_segment'), ('flexibility', 'Flexibility_segment')):
        out = _prepare_rtd_element_export(rtd_frame, mech)
        seq = PRIORITY_SEQUENCES[mech]
        segs = out[seg_col].astype(int)
        assert segs.min() < 5, f"{mech}: need a short list to test blanks"
        for _, row in out.iterrows():
            s = int(row[seg_col])
            tail = seq[s - 1:]
            for pos in range(1, 6):
                val = row[f'choice{pos}']
                if pos <= len(tail):
                    assert int(val) == tail[pos - 1]
                else:
                    assert pd.isna(val)


# ---------------------------------------------------------------------------
# AppTest end-to-end (verification items i, ii, iv, v)
# ---------------------------------------------------------------------------
def _rtd_app_script():
    """Decision 4 tab + (once results exist) the results page, end to end."""
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


def _all_markdown(at):
    return "\n".join(str(m.value) for m in at.markdown)


def _all_captions(at):
    return "\n".join(str(c.value) for c in at.caption)


def _download_labels(at):
    return [str(e.label) for e in at.get("download_button")]


def test_apptest_element_run_then_whole_run():
    """Clicking 'Run Loyalty Ranking Only' presents ONLY the Loyalty section
    (summary + charts + download, no other elements, no transaction-level export,
    D4 overview metrics); clicking the whole-decision button afterwards clears the
    element flag and brings all four sections back."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_app_script)
    at.run(timeout=600)
    assert not at.exception

    # (A) all five per-element Run buttons exist on the tab
    for mech in ('ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility'):
        assert at.button(key=f'rtd_run_{mech}_btn') is not None

    # (i) per-element run: Loyalty only
    at.button(key='rtd_run_loyalty_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] == 'loyalty'
    assert at.session_state['custom_decisions'] == ['rejected_transaction_defaults']

    md = _all_markdown(at)
    assert RTD_SECTION_TITLES['loyalty'] in md
    for other in ('ttp', 'wtp', 'risk_taking', 'flexibility'):
        assert RTD_SECTION_TITLES[other] not in md
    # summary line (Mean/SD/Min/Max/N) rendered exactly once (one element section)
    captions = _all_captions(at)
    assert captions.count("· SD ") == 1
    # element-scoped export description; no transaction-level file anywhere
    assert "Decision 4 Results Export (Loyalty element)" in md
    assert "Transaction-Level" not in md
    metric_labels = [m.label for m in at.metric]
    assert "Total Transactions" not in metric_labels
    # element-aware overview: a Loyalty-only run shows the Loyalty metric, NOT the
    # whole-decision options-list-length metric
    assert "Mean Loyalty score" in metric_labels
    assert "Avg. Options List Length" not in metric_labels
    # only the Loyalty element download (+ the D4 agent-level export) is offered
    dls = _download_labels(at)
    assert "📊 Download Loyalty Ranking Excel" in dls
    assert "📊 Download Decision 4 Agent-Level Excel" in dls
    assert not any("Options List Length" in l or "Willingness-to-Pay" in l
                   or "Risk-Taking" in l or "Cognitive Flexibility" in l
                   or "all elements" in l or "Transaction-Level" in l for l in dls)

    # (ii) whole-decision run: flag cleared; ONLY the integrated ranking is shown
    # (professor 2026-09: "Run Rejected Transaction Defaults Only" produces the
    # integrated ranking only; the elements stay in the Excel)
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] is None

    md = _all_markdown(at)
    for mech in ('ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility'):
        assert RTD_SECTION_TITLES[mech] not in md
    assert "6️⃣ Integrated Default List (Rank Aggregation)" in md
    assert "Option 5 (forgo the transaction) is the final option of the produced list" in md
    assert _all_captions(at).count("· SD ") == 0

    # (iv) D4-only export/overview: agent-level workbook only, D4 metrics
    assert "Transaction-Level" not in md
    assert "one row per agent with the decision 4 element results" in md.lower()
    metric_labels = [m.label for m in at.metric]
    assert "Total Transactions" not in metric_labels
    assert "Avg. Options List Length" in metric_labels
    assert "Total Agents" in metric_labels
    # the integrated-list download + the whole-decision workbook (all elements) +
    # the export section's agent-level workbook; NO per-element downloads and no
    # transaction-level file
    dls = _download_labels(at)
    for label in ("📊 Download Integrated Default List Excel",
                  "📊 Download Decision 4 Excel (all elements)",
                  "📊 Download Decision 4 Agent-Level Excel"):
        assert label in dls, label
    for label in ("📊 Download Options List Length Excel",
                  "📊 Download Loyalty Ranking Excel",
                  "📊 Download Willingness-to-Pay Ranking Excel",
                  "📊 Download Risk-Taking Ranking Excel",
                  "📊 Download Cognitive Flexibility Ranking Excel"):
        assert label not in dls, label
    assert not any("Transaction-Level" in l for l in dls)


def test_apptest_compare_both_element_filter_per_column():
    """(v) Compare-both income: a per-element run filters EACH comparison column to
    the selected element (one Loyalty section per income mode, no other sections)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_app_script)
    at.session_state['rtd_income_mode'] = 'Compare both'
    at.run(timeout=600)
    assert not at.exception

    at.button(key='rtd_run_loyalty_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] == 'loyalty'

    md = _all_markdown(at)
    # one Loyalty section per income-mode column, nothing else
    assert md.count(RTD_SECTION_TITLES['loyalty']) == 2
    for other in ('ttp', 'wtp', 'risk_taking', 'flexibility'):
        assert RTD_SECTION_TITLES[other] not in md
    assert "Transaction-Level" not in md
    # element-scoped D4-only export path is used for the comparison run as well:
    # one config-prefixed Loyalty sheet per income mode in the workbook preview
    assert "Decision 4 Results Export (Loyalty element)" in md
    assert "Cat Loyalty Sheet:" in md and "Cont Loyalty Sheet:" in md
    dls = _download_labels(at)
    assert dls.count("📊 Download Loyalty Ranking Excel") == 2  # one per column
    assert "📊 Download Decision 4 Agent-Level Excel" in dls
