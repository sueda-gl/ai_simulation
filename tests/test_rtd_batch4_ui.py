"""
UI / plumbing tests for the Decision 4 results display and exports.

Covers the professor's 2026-09 display specification:

- score histograms use Stata's DEFAULT bin rule, k = round(min(sqrt(N), 10*log10(N)))
  equal-width bins spanning min..max, with histnorm='probability' (bar heights sum
  to 1); the caption under a score chart shows Min and Max only;
- allocation charts order the categories by the element's priority sequence REVERSED
  and read each agent's first choice from the element's RANKING column (never from the
  segment, so the charts are agnostic to the segment -> sequence mapping direction);
- five option-explanation lines are printed under every allocation chart;
- run shapes: a per-element run shows only that element; "Run Rejected Transaction
  Defaults Only" shows all five elements + the integrated first-option chart but no tie
  statistics; "Run Integrated Default List Only" shows the integrated charts + the tie
  statistics; in Compare-both income the income-free elements render ONCE;
- the per-element Excel carries the element's raw inputs AND the z-scores its equation
  uses, its intermediates, both segments, the sigma and choice1..choice5; the
  whole-decision workbook leads with one merged 'Integrated Default List' sheet holding
  every variable of every element, followed by one sheet per element.
"""
import numpy as np
import pandas as pd
import pytest

from src.decisions.rejected_transaction_defaults import PRIORITY_SEQUENCES
import app.pages.results.visualizations.transaction_viz as viz
from app.pages.results.visualizations.transaction_viz import (
    _prepare_rtd_element_export,
    _prepare_rtd_integrated_export,
    _prepare_rtd_model_export,
    _RTD_ELEMENT_SHEETS,
    _RTD_OPTION_LINES,
)

RTD_SECTION_TITLES = {
    'ttp': "1️⃣ Options List Length (Tendency to Plan)",
    'loyalty': "2️⃣ Loyalty Ranking",
    'wtp': "3️⃣ Willingness-to-Pay Ranking",
    'risk_taking': "4️⃣ Risk-Taking Ranking",
    'flexibility': "5️⃣ Flexibility Ranking",
}
INTEGRATED_SECTION = "6️⃣ Integrated Default List"
TIE_STATS_SECTION = "Tie statistics of the Kemeny aggregation"
RANKING_COL_KEYS = {'loyalty': 'loyalty', 'wtp': 'wtp',
                    'risk_taking': 'rt', 'flexibility': 'flex'}


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
# A.1 / A.2 - histogram bins, density normalisation, Min/Max caption
# ---------------------------------------------------------------------------
def test_density_hist_uses_stata_default_bins_and_probability_normalisation(monkeypatch):
    """Stata's default rule - k = round(min(sqrt(N), 10*log10(N))) equal-width bins
    spanning min..max - so the chart reproduces the design document's figures (N = 500
    -> min(22.36, 26.99) = 22.36 -> 22 bins; N = 280 -> 17). histnorm='probability', so
    the bar heights are proportions that sum to 1 (professor: "Did you standardize the
    density values so that the values sum up to 1?").

    Stata-parity of the bin counts themselves lives in
    tests/test_rtd_histogram_stata_parity.py."""
    captured = {}
    monkeypatch.setattr(viz.st, 'plotly_chart',
                        lambda fig, **kw: captured.__setitem__('fig', fig))
    s = pd.Series(np.random.default_rng(7).normal(size=500))
    expected_k = viz._rtd_stata_bin_count(len(s))
    assert expected_k == 22

    viz._rtd_density_hist(s, "title", "x", "k")

    trace = captured['fig'].data[0]
    assert trace.histnorm == 'probability'
    start, size = float(trace.xbins.start), float(trace.xbins.size)
    assert start == pytest.approx(float(s.min()))
    assert size == pytest.approx((float(s.max()) - float(s.min())) / expected_k)
    assert round((float(trace.xbins.end) - start) / size) == expected_k

    # every observation falls inside the k bins -> the plotted proportions sum to 1
    edges, counts = viz._rtd_stata_bins(s)
    assert len(counts) == expected_k
    assert edges[0] == pytest.approx(start)
    assert counts.sum() == len(s)
    assert (counts / len(s)).sum() == pytest.approx(1.0)


def test_density_hist_bins_a_280_agent_score_into_17_bins(monkeypatch):
    """The document's sample size: 280 participants -> 17 bins, empty bins kept at zero
    height (Stata just does not draw them)."""
    captured = {}
    monkeypatch.setattr(viz.st, 'plotly_chart',
                        lambda fig, **kw: captured.__setitem__('fig', fig))
    s = pd.Series(np.random.default_rng(3).normal(size=280))
    viz._rtd_density_hist(s, "title", "x", "k280")
    trace = captured['fig'].data[0]
    start, size = float(trace.xbins.start), float(trace.xbins.size)
    assert round((float(trace.xbins.end) - start) / size) == 17


def test_score_caption_shows_min_and_max_only(monkeypatch):
    captions = []
    monkeypatch.setattr(viz.st, 'caption', lambda text, **kw: captions.append(text))
    viz._rtd_score_stats_caption(pd.Series([0.0, 1.0, 2.0, 3.0]))
    assert captions == ["Min 0.0000 · Max 3.0000"]
    for dropped in ("Mean", "SD", "N "):
        assert dropped not in captions[0]


# ---------------------------------------------------------------------------
# A.4 / A.5 - reversed category order, ranking-derived first choice, option lines
# ---------------------------------------------------------------------------
def test_reversed_sequence_order_matches_the_specification():
    """Least likely option on the left, most likely on the right."""
    assert viz._rtd_reversed_sequence(PRIORITY_SEQUENCES['loyalty']) == [2, 5, 4, 1, 3]
    assert viz._rtd_reversed_sequence(PRIORITY_SEQUENCES['wtp']) == [5, 4, 1, 2, 3]
    assert viz._rtd_reversed_sequence(PRIORITY_SEQUENCES['risk_taking']) == [5, 3, 1, 2, 4]
    assert viz._rtd_reversed_sequence(PRIORITY_SEQUENCES['flexibility']) == [5, 1, 3, 4, 2]


def test_first_choice_is_read_from_the_ranking_column(rtd_frame):
    """Direction-agnostic by construction: the helper returns ranking[0], never a
    value derived from the segment."""
    for mech, col_key in RANKING_COL_KEYS.items():
        firsts = viz._rtd_first_choice(rtd_frame, col_key)
        expected = rtd_frame[f'rtd_{col_key}_ranking'].apply(lambda l: l[0])
        assert list(firsts) == list(expected)
        # and it is genuinely the model's list head, whichever mapping direction is used
        assert set(firsts) <= set(PRIORITY_SEQUENCES[mech])


def test_ranking_allocation_chart_uses_reversed_order_and_ranking_shares(rtd_frame, monkeypatch):
    bars = []
    monkeypatch.setattr(viz, '_rtd_fraction_bar',
                        lambda labels, fractions, *a, **k: bars.append((list(labels), list(fractions))))
    monkeypatch.setattr(viz, '_rtd_density_hist', lambda *a, **k: None)

    for mech, col_key in RANKING_COL_KEYS.items():
        bars.clear()
        viz._render_rtd_ranking_section(rtd_frame, 'd4', '', lambda a, b: (a(), b()),
                                        mech, False)
        labels, fractions = bars[0]
        order = viz._rtd_reversed_sequence(PRIORITY_SEQUENCES[mech])
        assert labels == [f"Option {o}" for o in order]
        firsts = rtd_frame[f'rtd_{col_key}_ranking'].apply(lambda l: l[0])
        assert fractions == pytest.approx([(firsts == o).mean() for o in order])


def test_option_explanation_lines(monkeypatch):
    """Five separate plain-markdown lines (not a caption), one per option."""
    lines = []
    monkeypatch.setattr(viz.st, 'markdown', lambda text, **kw: lines.append(text))
    viz._rtd_option_lines()
    assert lines == [
        "Option 1: higher price category, same vendor",
        "Option 2: other vendor at lower PN price",
        "Option 3: current vendor at PN price",
        "Option 4: place a bid",
        "Option 5: forgo the transaction",
    ]
    assert all("**" not in line for line in lines)


# ---------------------------------------------------------------------------
# C - Excel builders
# ---------------------------------------------------------------------------
INTEGRATED_SHEET_COLUMNS = [
    'Agent ID',
    # raw inputs
    'ExtraversionBig5', 'Agreeable', 'NeuroticismBig5', 'ConscientiousnessBig5',
    'OpennessBig5', 'Education', 'income', 'stdactions',
    # z-scores the equations use
    'z_extraversionbig5', 'z_agreeable', 'z_neuroticismbig5',
    'z_conscientiousnessbig5', 'z_opennessbig5', 'reducation', 'z_net_income',
    # 1 Options List Length
    'weighted_ttp', 'weighted_ttp06', 'choice_length_deterministic', 'choice_length',
    # 2 Loyalty
    'loyalty_score', 'z_loyalty', 'loyalty_segment_deterministic', 'loyalty_segment',
    'loyalty_list',
    # 3 Willingness-to-Pay
    'WTP_score', 'z_WTP', 'WTP_segment_deterministic', 'WTP_segment', 'WTP_list',
    # 4 Risk-Taking
    'RT_score', 'z_RT', 'RT_segment_deterministic', 'RT_segment', 'RT_list',
    # 5 Flexibility
    'Flexibility_calculated_ivw', 'z_Flexibility_calculated_ivw', 'z_stdactions',
    'Flexibility_score', 'z_Flexibility', 'Flexibility_segment_deterministic',
    'Flexibility_segment', 'Flexibility_list',
    # 6 integration
    'consensus_ranking', 'kemeny_status', 'n_kemeny_optimal', 'is_kemeny_optimal',
    'settled_by', 'truncated_by', 'default_list_length',
    'final_choice1', 'final_choice2', 'final_choice3', 'final_choice4', 'final_choice5',
]


def test_whole_decision_workbook_sheets_and_columns(rtd_frame):
    sheets = _prepare_rtd_model_export(rtd_frame)
    # professor 2026-09: ONE merged integrated sheet first (the old separate
    # 'All Elements' sheet is gone), then one self-contained sheet per element
    assert list(sheets.keys()) == ['Integrated Default List', 'Options List Length',
                                   'Loyalty', 'Willingness-to-Pay', 'Risk-Taking',
                                   'Flexibility']
    assert 'All Elements' not in sheets
    assert list(sheets['Integrated Default List'].columns) == INTEGRATED_SHEET_COLUMNS

    assert list(sheets['Options List Length'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'z_extraversionbig5', 'Agreeable', 'z_agreeable',
        'NeuroticismBig5', 'z_neuroticismbig5', 'ConscientiousnessBig5',
        'z_conscientiousnessbig5', 'Education', 'reducation',
        'weighted_ttp', 'weighted_ttp06', 'choice_length_deterministic',
        'choice_length', 'sigma_used_ttp']
    assert list(sheets['Loyalty'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'z_extraversionbig5', 'OpennessBig5',
        'z_opennessbig5', 'Agreeable', 'z_agreeable',
        'loyalty_score', 'z_loyalty', 'loyalty_segment_deterministic', 'loyalty_segment',
        'sigma_used_loyalty', 'choice1', 'choice2', 'choice3', 'choice4', 'choice5']
    assert list(sheets['Willingness-to-Pay'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'z_extraversionbig5', 'Agreeable', 'z_agreeable',
        'income', 'z_net_income',
        'WTP_score', 'z_WTP', 'WTP_segment_deterministic', 'WTP_segment',
        'sigma_used_WTP', 'choice1', 'choice2', 'choice3', 'choice4', 'choice5']
    assert list(sheets['Risk-Taking'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'z_extraversionbig5', 'OpennessBig5',
        'z_opennessbig5', 'Agreeable', 'z_agreeable', 'ConscientiousnessBig5',
        'z_conscientiousnessbig5', 'NeuroticismBig5', 'z_neuroticismbig5',
        'income', 'z_net_income',
        'RT_score', 'z_RT', 'RT_segment_deterministic', 'RT_segment',
        'sigma_used_RT', 'choice1', 'choice2', 'choice3', 'choice4', 'choice5']
    assert list(sheets['Flexibility'].columns) == [
        'Agent ID', 'ExtraversionBig5', 'z_extraversionbig5', 'OpennessBig5',
        'z_opennessbig5', 'NeuroticismBig5', 'z_neuroticismbig5', 'Agreeable',
        'z_agreeable', 'ConscientiousnessBig5', 'z_conscientiousnessbig5',
        'stdactions', 'z_stdactions',
        'Flexibility_calculated_ivw', 'z_Flexibility_calculated_ivw',
        'Flexibility_score', 'z_Flexibility', 'Flexibility_segment_deterministic',
        'Flexibility_segment', 'sigma_used_Flexibility',
        'choice1', 'choice2', 'choice3', 'choice4', 'choice5']

    for sheet in sheets.values():
        assert len(sheet) == len(rtd_frame)

    # the per-element file and the element's workbook sheet are the same frame
    for mech, name in _RTD_ELEMENT_SHEETS.items():
        assert list(_prepare_rtd_element_export(rtd_frame, mech).columns) == \
            list(sheets[name].columns)


def test_integrated_sheet_is_the_aggregation_download(rtd_frame):
    """'Run Integrated Default List Only' downloads the SAME first sheet as the
    whole-decision workbook."""
    integrated = _prepare_rtd_integrated_export(rtd_frame)
    assert list(integrated.columns) == INTEGRATED_SHEET_COLUMNS
    assert len(integrated) == len(rtd_frame)
    # the final list columns come from the model's own integrated list
    for _, row in integrated.head(10).iterrows():
        length = int(row['default_list_length'])
        for pos in range(1, 6):
            if pos <= length:
                assert not pd.isna(row[f'final_choice{pos}'])
            else:
                assert pd.isna(row[f'final_choice{pos}'])


def test_element_exports_only_own_variables(rtd_frame):
    ttp = _prepare_rtd_element_export(rtd_frame, 'ttp')
    loyalty = _prepare_rtd_element_export(rtd_frame, 'loyalty')
    wtp = _prepare_rtd_element_export(rtd_frame, 'wtp')
    rt = _prepare_rtd_element_export(rtd_frame, 'risk_taking')
    flex = _prepare_rtd_element_export(rtd_frame, 'flexibility')

    # exclusivity: no foreign independent variables leak into an element's file
    assert 'income' not in ttp.columns and 'income' not in loyalty.columns
    assert 'z_net_income' not in ttp.columns and 'z_net_income' not in loyalty.columns
    assert 'OpennessBig5' not in ttp.columns and 'OpennessBig5' not in wtp.columns
    assert 'Assigned Allowance Level' not in wtp.columns  # continuous frame
    assert 'income' not in flex.columns and 'stdactions' not in rt.columns
    # every element carries the z-scores its own equation multiplies
    assert 'reducation' in ttp.columns          # TTP's education term
    assert 'z_stdactions' in flex.columns       # Flexibility's observed anchor
    assert 'z_agreeable' in loyalty.columns
    # both segments and the sigma used are present for every ranking element
    for out, stata in ((loyalty, 'loyalty'), (wtp, 'WTP'), (rt, 'RT'), (flex, 'Flexibility')):
        assert f'{stata}_segment_deterministic' in out.columns
        assert f'{stata}_segment' in out.columns
        assert f'sigma_used_{stata}' in out.columns


def test_categorical_frame_uses_allowance_level_instead_of_z_income(rtd_frame_cat):
    assert str(rtd_frame_cat['rtd_income_mode'].iloc[0]) == 'categorical'
    sheets = _prepare_rtd_model_export(rtd_frame_cat)
    for name in ('Willingness-to-Pay', 'Risk-Taking'):
        assert 'Assigned Allowance Level' in sheets[name].columns
        # the categorical equations use the budget-level dummies, not z_net_income
        assert 'z_net_income' not in sheets[name].columns
    for name in ('Options List Length', 'Loyalty', 'Flexibility'):
        assert 'Assigned Allowance Level' not in sheets[name].columns
    for mech in ('wtp', 'risk_taking'):
        assert 'Assigned Allowance Level' in \
            _prepare_rtd_element_export(rtd_frame_cat, mech).columns
    for mech in ('ttp', 'loyalty', 'flexibility'):
        assert 'Assigned Allowance Level' not in \
            _prepare_rtd_element_export(rtd_frame_cat, mech).columns
    integrated = sheets['Integrated Default List']
    assert 'Assigned Allowance Level' in integrated.columns
    assert 'z_net_income' not in integrated.columns


def test_choice_columns_mirror_the_model_ranking_and_blank_beyond_length(rtd_frame):
    """choice1..choice5 reproduce the element's OWN ranking column (direction-agnostic:
    nothing here re-derives the segment -> priority-sequence mapping); positions beyond
    the list length are blank."""
    for mech, col_key in RANKING_COL_KEYS.items():
        out = _prepare_rtd_element_export(rtd_frame, mech)
        rankings = rtd_frame[f'rtd_{col_key}_ranking']
        assert rankings.apply(len).min() < 5, f"{mech}: need a short list to test blanks"
        for (_, row), ranking in zip(out.iterrows(), rankings):
            for pos in range(1, 6):
                val = row[f'choice{pos}']
                if pos <= len(ranking):
                    assert int(val) == ranking[pos - 1]
                else:
                    assert pd.isna(val)


# ---------------------------------------------------------------------------
# B - run shapes, end to end
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
    return [str(c.value) for c in at.caption]


def _download_labels(at):
    return [str(e.label) for e in at.get("download_button")]


def _option_line_counts(at):
    md = [str(m.value) for m in at.markdown]
    return [md.count(line) for line in _RTD_OPTION_LINES]


def test_apptest_per_element_run_then_whole_run_then_aggregation_run():
    """The three individual Decision 4 run shapes, in one session."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_app_script)
    at.run(timeout=600)
    assert not at.exception

    # (A) all five per-element Run buttons + the integrated-list Run button exist
    for mech in ('ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility'):
        assert at.button(key=f'rtd_run_{mech}_btn') is not None
    assert at.button(key='rtd_run_aggregation_btn') is not None

    # ---- (B.1) per-element run: Loyalty only ----
    at.button(key='rtd_run_loyalty_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] == 'loyalty'
    assert at.session_state['custom_decisions'] == ['rejected_transaction_defaults']

    md = _all_markdown(at)
    assert RTD_SECTION_TITLES['loyalty'] in md
    for other in ('ttp', 'wtp', 'risk_taking', 'flexibility'):
        assert RTD_SECTION_TITLES[other] not in md
    assert INTEGRATED_SECTION not in md
    # Min/Max-only caption, rendered once (one element section)
    captions = _all_captions(at)
    assert sum(c.startswith("Min ") and "· Max " in c for c in captions) == 1
    assert not any("· SD " in c for c in captions)
    # five option lines under the single allocation chart
    assert _option_line_counts(at) == [1, 1, 1, 1, 1]
    # A.3: no "Mean <element> score" headline metric for a ranking element run
    metric_labels = [m.label for m in at.metric]
    assert "Mean Loyalty score" not in metric_labels
    assert "Avg. Options List Length" not in metric_labels
    assert "Total Agents" in metric_labels
    assert "Total Transactions" not in metric_labels
    # element-scoped export description; no transaction-level file anywhere
    assert "Decision 4 Results Export (Loyalty element)" in md
    assert "Transaction-Level" not in md
    dls = _download_labels(at)
    assert "📊 Download Loyalty Ranking Excel" in dls
    assert "📊 Download Decision 4 Agent-Level Excel" in dls
    assert not any("Options List Length" in l or "Willingness-to-Pay" in l
                   or "Risk-Taking" in l or "Flexibility" in l
                   or "all elements" in l or "Transaction-Level" in l for l in dls)

    # ---- (B.2) whole-decision run: all five elements + the integrated first option ----
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] is None

    md = _all_markdown(at)
    for mech in ('ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility'):
        assert RTD_SECTION_TITLES[mech] in md, mech
    assert INTEGRATED_SECTION in md
    # the descriptive paragraph and the "elements are not shown" caption are gone
    assert "presents each customer's final pre-selected default list" not in md
    assert "The individual elements' results are not shown" not in "\n".join(_all_captions(at))
    # no tie statistics, no truncation table, no most-common-lists table in this run
    assert TIE_STATS_SECTION not in md
    assert "Stage that settled the ranking" not in md
    assert "Kemeny outcome" not in md
    assert "Most common integrated default lists" not in md
    assert "List cut by" not in md
    # one Min/Max caption per element section, five option-line blocks (5 elements +
    # the integrated first-option chart)
    captions = _all_captions(at)
    assert sum(c.startswith("Min ") and "· Max " in c for c in captions) == 5
    assert _option_line_counts(at) == [6, 6, 6, 6, 6]
    # D4-only export/overview: agent-level workbook only, D4 metrics
    assert "Transaction-Level" not in md
    assert "one row per agent with the decision 4 element results" in md.lower()
    metric_labels = [m.label for m in at.metric]
    assert "Avg. Options List Length" in metric_labels
    assert "Min Options List Length" in metric_labels
    assert "Max Options List Length" in metric_labels
    # the whole-decision workbook + the export section's agent-level workbook; no
    # per-element downloads (the workbook carries the same sheets)
    dls = _download_labels(at)
    for label in ("📊 Download Decision 4 Excel (all elements)",
                  "📊 Download Decision 4 Agent-Level Excel"):
        assert label in dls, label
    for label in ("📊 Download Options List Length Excel",
                  "📊 Download Loyalty Ranking Excel",
                  "📊 Download Willingness-to-Pay Ranking Excel",
                  "📊 Download Risk-Taking Ranking Excel",
                  "📊 Download Flexibility Ranking Excel",
                  "📊 Download Integrated Default List Excel"):
        assert label not in dls, label

    # ---- (B.3) "Run Integrated Default List Only": integrated charts + tie stats ----
    at.button(key='rtd_run_aggregation_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] == 'aggregation'

    md = _all_markdown(at)
    for mech in ('ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility'):
        assert RTD_SECTION_TITLES[mech] not in md, mech
    assert INTEGRATED_SECTION in md
    assert TIE_STATS_SECTION in md
    assert "% of agents with initial ties after Kemeny" in md
    assert "Kemeny outcome" in md
    assert "Stage that settled the ranking" in md
    assert "Kemeny-optimal for" in md
    # no element score charts -> no Min/Max captions
    assert not any(c.startswith("Min ") for c in _all_captions(at))
    # the integrated allocation chart still carries the option explanations
    assert _option_line_counts(at) == [1, 1, 1, 1, 1]
    # A.3: the aggregation run's headline metric
    metric_labels = [m.label for m in at.metric]
    assert "Avg. integrated default list length" in metric_labels
    assert "Avg. Options List Length" not in metric_labels
    dls = _download_labels(at)
    assert "📊 Download Integrated Default List Excel" in dls
    assert "📊 Download Decision 4 Excel (all elements)" not in dls
    assert "Decision 4 Integrated Default List Export" in md


def test_apptest_ttp_element_run_keeps_the_options_list_length_metric():
    """A.3: the Options List Length element run keeps the headline metric (+min/max)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_app_script)
    at.run(timeout=600)
    at.button(key='rtd_run_ttp_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] == 'ttp'

    md = _all_markdown(at)
    assert RTD_SECTION_TITLES['ttp'] in md
    for other in ('loyalty', 'wtp', 'risk_taking', 'flexibility'):
        assert RTD_SECTION_TITLES[other] not in md
    # "customer" wording replaced by "agent"
    assert "presents how many default options each agent pre-selects (0-5)" in md
    metric_labels = [m.label for m in at.metric]
    assert "Avg. Options List Length" in metric_labels
    assert "Min Options List Length" in metric_labels
    assert "Max Options List Length" in metric_labels
    assert "📊 Download Options List Length Excel" in _download_labels(at)


def test_apptest_compare_both_renders_income_free_elements_once():
    """(B.5a) Compare-both income: an income-FREE element (Loyalty) is identical under
    both income specifications, so it is rendered ONCE with a note instead of once per
    income treatment."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_app_script)
    at.session_state['rtd_income_mode'] = 'Compare both'
    at.run(timeout=600)
    assert not at.exception

    at.button(key='rtd_run_loyalty_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] == 'loyalty'
    assert sorted(at.session_state['simulation_results'].keys()) == ['categorical', 'continuous']

    md = _all_markdown(at)
    assert md.count(RTD_SECTION_TITLES['loyalty']) == 1
    assert "Income-Independent Elements" in md
    assert any("identical for both income specifications" in c.lower()
               for c in _all_captions(at))
    for other in ('ttp', 'wtp', 'risk_taking', 'flexibility'):
        assert RTD_SECTION_TITLES[other] not in md
    assert "Transaction-Level" not in md
    # the income-treatment groups still carry their overview cells (and the per-config
    # "Use This Config" buttons), so a configuration can still be selected
    assert "Categorical Income Treatment" in md and "Continuous Income Treatment" in md
    button_keys = [b.key for b in at.button]
    assert 'rtd_inline_select_categorical' in button_keys
    assert 'rtd_inline_select_continuous' in button_keys
    # element-scoped D4-only export path still exports both configurations
    assert "Decision 4 Results Export (Loyalty element)" in md
    assert "Cat Loyalty Sheet:" in md and "Cont Loyalty Sheet:" in md
    dls = _download_labels(at)
    assert dls.count("📊 Download Loyalty Ranking Excel") == 1   # rendered once
    assert "📊 Download Decision 4 Agent-Level Excel" in dls


def test_apptest_compare_both_repeats_income_dependent_elements():
    """(B.5a) Willingness-to-Pay DOES use income, so it is rendered per income
    treatment - one section per result key."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_app_script)
    at.session_state['rtd_income_mode'] = 'Compare both'
    at.run(timeout=600)
    at.button(key='rtd_run_wtp_btn').click().run(timeout=600)
    assert not at.exception

    md = _all_markdown(at)
    assert md.count(RTD_SECTION_TITLES['wtp']) == 2
    assert "Income-Independent Elements" not in md
    for other in ('ttp', 'loyalty', 'risk_taking', 'flexibility'):
        assert RTD_SECTION_TITLES[other] not in md
