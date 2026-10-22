"""
The Decision 4 results page, driven through Streamlit's AppTest: element subtitles,
the selected-configuration summary, the score charts, the complete-run section, the
"Use This Config" flow and the comparison layouts. Pinned against the Decision 4 display
specification (and, for the Research Baseline options-list-length chart, the length
distribution 20/92/95/57/14/2 of Stata_File_Decision4_050926.dta).

Merged from six files (2026-10-21) - test_rtd_combined_run_display.py,
test_rtd_comparison_layout_ui.py, test_rtd_config_selection.py,
test_rtd_element_subtitle.py, test_rtd_element_summary.py and
test_rtd_score_chart_matches_stats.py. Each section below keeps its file's
description (as a comment), app script, helpers, tests and parametrisations. The two
complete-run tests share one module-scoped run (``combined_run``): their setups were
identical. Three of the files each had an app script named ``_app_script``; they
differed, so each keeps its own under a section prefix (``_subtitle_app_script``,
``_summary_app_script``, ``_score_chart_app_script``). The comparison-layout file's
single-mode and Compare-both scripts were identical to each other and to
tests/test_rtd_batch4_ui.py's; all three now use
``helpers.apptest.rtd_tab_and_results_script``.
"""
import numpy as np
import pandas as pd
import pytest

import app.pages.results.visualizations.transaction_viz as viz
from app.reports.rtd import (
    RTD_ELEMENT_SUBTITLES, RTD_SCORE_CHARTS, rtd_element_subtitle, rtd_score_chart_data,
    rtd_summary_metric_keys,
)
from src.decisions.rejected_transaction_defaults import _RESCALE
from helpers.apptest import all_markdown, rtd_tab_and_results_script
from ui_strings import (
    CATEGORICAL_TREATMENT, CONTINUOUS_TREATMENT, D4_SUBTITLE_AGGREGATION,
    D4_SUBTITLE_FLEXIBILITY, D4_SUBTITLE_LOYALTY, D4_SUBTITLE_RISK_TAKING, D4_SUBTITLE_TTP,
    D4_SUBTITLE_WTP, DOWNLOAD_D4_ALL_ELEMENTS, SELECTED_LABEL,
)

pytestmark = pytest.mark.slow


# =============================================================================
# from test_rtd_combined_run_display.py
# =============================================================================
# Complete (combined) simulation with Decision 4 as the customised decision.
#
# Per the display specification, the Decision 4 section of such a run
# shows ONLY the integrated results - the "% of agents with default list length" chart
# with its table and the "% of first integrated default option" chart with its table and
# the option explanations. No element sections, no descriptive paragraph, no tie
# statistics and no "most common integrated default lists" table; the element values stay
# in the run's agent-level Excel.
#
# It must also never show the other decisions' analyses (donation rate, disclose income,
# ...) or the comparison scaffolding - even when a comparison setting ("Compare both"
# income spec from the Decision 4 tab, or "Compare all" population mode) is still active
# in the session from an earlier individual run. Regression for a reported
# screenshot: the Decision 4 section of a complete run rendered "Categorical
# Income Treatment" + "Simulation Overview (Categorical)" + "Donation Rate Analysis".
#
# Ported from the original repo (e973a99) to the current layout: the participants come
# from src.engine.sampling (the app path), the agent-level frame from
# app.reports.build_agent_level_dataframe, and the run shape from _run_metadata.

def _combined_run_script():
    import streamlit as st
    from app.models import initialize_session_state, ALL_DECISIONS

    initialize_session_state()
    st.session_state.population_mode = 'Research Baseline'
    st.session_state.income_spec_mode = 'Compare both'     # stale comparison setting
    st.session_state.n_agents = 30

    if not st.session_state.get('simulation_results'):
        from src.engine.core import Engine
        from app.simulation import _apply_rejected_transaction_config
        from src.engine.sampling import load_original_participants
        orch = Engine("baseline")
        _apply_rejected_transaction_config(orch, "baseline", "categorical")
        agents = load_original_participants(30)
        df = orch.run_simulation(30, 1, None, agents_df=agents)     # every decision
        st.session_state.simulation_results = {'categorical': df}
        st.session_state.custom_decisions = ['rejected_transaction_defaults']
        st.session_state.default_decisions = [d for d in ALL_DECISIONS
                                              if d != 'rejected_transaction_defaults']
        # what the run actually used (a complete run is single-mode)
        st.session_state._run_metadata = {
            'effective_population_mode': 'Research Baseline',
            'effective_income_mode': 'categorical only',
            'result_keys': ['categorical'],
            'custom_decisions': list(st.session_state.custom_decisions),
            'default_decisions': list(st.session_state.default_decisions),
            'seed': 1, 'n_agents': 30, 'is_comparison': False, 'num_results': 1,
        }

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


@pytest.fixture(scope="module")
def combined_run():
    """The complete run both tests below inspect (their setups were identical: one
    AppTest of _combined_run_script, run once with timeout 900)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_combined_run_script)
    at.run(timeout=900)
    return at


def test_combined_run_decision4_section_shows_integrated_results_only(combined_run):
    at = combined_run
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
    assert CATEGORICAL_TREATMENT not in text
    assert CONTINUOUS_TREATMENT not in text
    assert "Income-Independent Elements" not in text
    assert "Simulation Overview (Categorical)" not in text
    assert "Donation Rate Analysis" not in text
    assert "Disclose Income Analysis" not in text
    # the complete simulation offers the same Decision 4 workbook
    # as the whole-decision run
    labels = [str(e.label) for e in at.get("download_button")]
    assert labels.count(DOWNLOAD_D4_ALL_ELEMENTS) == 1


def test_combined_agent_level_export_carries_every_decision4_variable(combined_run):
    """The element values must stay available in the complete simulation's agent-level
    Excel even though the page shows only the integrated results."""
    from app.reports import build_agent_level_dataframe as _build_agent_level_dataframe

    at = combined_run
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
    for col in ('rtd_default_list', 'rtd_integrated_ranking', 'rtd_integrated_kemeny_status',
                'rtd_integrated_n_kemeny_optimal', 'rtd_integrated_is_kemeny_optimal',
                'rtd_integrated_settled_by', 'rtd_integrated_truncated_by',
                'rtd_default_list_length'):
        assert col in columns, col
    # "integrated", never "consensus"
    assert not any('consensus' in c.lower() for c in columns)


# =============================================================================
# from test_rtd_comparison_layout_ui.py
# =============================================================================
# UI layout tests for the Decision 4 comparison results page.
#
# For an INDIVIDUAL Decision 4 run in a comparison mode, the summary and details
# must be INTERLEAVED per income treatment inside the "📋 Decision Results"
# section:
#
#     📋 Decision Results (+ green custom-parameters banner)
#     Categorical Income Treatment
#         [overview cells - one per population mode]
#         [detailed element sections - one column per population mode]
#     Continuous Income Treatment
#         [overview cells]
#         [detailed sections]
#
# instead of the old split layout (both treatment summaries first, then both
# detail rows). Single-income comparison runs degenerate to one untitled group
# (overview row then detail row), and plain single-mode runs keep the existing
# summary-first layout ("📊 Simulation Overview" before "📋 Decision Results").

# ---------------------------------------------------------------------------
# App scripts and helpers
# ---------------------------------------------------------------------------
def _rtd_compare_all_app_script():
    """Decision 4 tab in Compare-all population mode + the results page."""
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()
    st.session_state.population_mode = 'Compare all'
    st.session_state.n_agents = 60

    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()

    if st.session_state.get('simulation_results'):
        from app.pages.results.main_results import render_single_run_results
        render_single_run_results()


def _ordered_texts(at):
    """All text-bearing elements of the main block in document order, prefixed
    with their element type ('markdown:', 'subheader:', 'success:', ...)."""
    texts = []
    for node in at.main:
        node_type = getattr(node, "type", "")
        if node_type in ("markdown", "subheader", "header", "title",
                         "caption", "success"):
            texts.append(f"{node_type}:{node.value}")
    return texts


def _first_index(texts, needle):
    for i, t in enumerate(texts):
        if needle in t:
            return i
    raise AssertionError(f"{needle!r} not found on the page")


def _count(texts, needle):
    return sum(needle in t for t in texts)


WTP_SECTION = "3️⃣ Willingness-to-Pay Ranking"
RT_SECTION = "4️⃣ Risk-Taking Ranking"
INTEGRATED_SECTION = "6️⃣ Integrated Default List"
ALL_SECTIONS = ("1️⃣ Options List Length (Tendency to Plan)",
                "2️⃣ Loyalty Ranking",
                "3️⃣ Willingness-to-Pay Ranking",
                "4️⃣ Risk-Taking Ranking",
                "5️⃣ Flexibility Ranking")
# Elements whose equations carry no income term: rendered ONCE when both income
# treatments are on the page.
INCOME_FREE_SECTIONS = ("1️⃣ Options List Length (Tendency to Plan)",
                        "2️⃣ Loyalty Ranking",
                        "5️⃣ Flexibility Ranking")


def _assert_interleaved_two_groups(texts, first_detail_marker):
    """Order: Decision Results header < custom-parameters banner
    < Categorical title < categorical overview < categorical details
    < Continuous title < continuous overview < continuous details."""
    idx_results = _first_index(texts, "📋 Decision Results")
    idx_banner = _first_index(texts, "custom parameters")
    idx_cat_title = _first_index(texts, CATEGORICAL_TREATMENT)
    idx_cat_overview = _first_index(texts, "Simulation Overview (Copula, Cat)")
    idx_cont_title = _first_index(texts, CONTINUOUS_TREATMENT)
    idx_cont_overview = _first_index(texts, "Simulation Overview (Copula, Cont)")

    detail_indexes = [i for i, t in enumerate(texts) if first_detail_marker in t]
    cat_details = [i for i in detail_indexes if idx_cat_title < i < idx_cont_title]
    cont_details = [i for i in detail_indexes if i > idx_cont_title]
    assert cat_details, "no categorical detail sections between the group titles"
    assert cont_details, "no continuous detail sections after the Continuous title"
    idx_cat_detail = cat_details[0]
    idx_cont_detail = cont_details[0]

    order = [idx_results, idx_banner, idx_cat_title, idx_cat_overview,
             idx_cat_detail, idx_cont_title, idx_cont_overview, idx_cont_detail]
    assert order == sorted(order), (
        f"interleaved order violated: results={idx_results} banner={idx_banner} "
        f"cat_title={idx_cat_title} cat_overview={idx_cat_overview} "
        f"cat_detail={idx_cat_detail} cont_title={idx_cont_title} "
        f"cont_overview={idx_cont_overview} cont_detail={idx_cont_detail}")
    assert idx_cat_overview < idx_cat_detail
    assert idx_cont_overview < idx_cont_detail

    # Each treatment title appears exactly once; the categorical overview row
    # sits fully inside the categorical group (before the Continuous title).
    assert _count(texts, CATEGORICAL_TREATMENT) == 1
    assert _count(texts, CONTINUOUS_TREATMENT) == 1
    for overview in ("Simulation Overview (Copula, Cat)",
                     "Simulation Overview (Research Spec, Cat)",
                     "Simulation Overview (Research Baseline, Cat)"):
        assert idx_cat_title < _first_index(texts, overview) < idx_cont_title
    for overview in ("Simulation Overview (Copula, Cont)",
                     "Simulation Overview (Research Spec, Cont)",
                     "Simulation Overview (Research Baseline, Cont)"):
        assert _first_index(texts, overview) > idx_cont_title

    # The old split layout's summary grid must be gone.
    joined = "\n".join(texts)
    assert "All Population Modes Comparison" not in joined
    assert "Income Specification Comparison" not in joined
    return joined


def _assert_shared_overview_layout(texts, first_detail_marker):
    """Layout when the income-free elements are shared (whole-decision run, both
    income treatments) - the overview sits ABOVE the graphs:
    Decision Results header < custom-parameters banner < ONE income-neutral overview
    row (per population mode) < Income-Independent Elements < Categorical title
    < categorical details < Continuous title < continuous details."""
    idx_results = _first_index(texts, "📋 Decision Results")
    idx_banner = _first_index(texts, "custom parameters")
    idx_overview = _first_index(texts, "Simulation Overview (Copula)")
    idx_shared = _first_index(texts, "Income-Independent Elements")
    idx_cat_title = _first_index(texts, CATEGORICAL_TREATMENT)
    idx_cont_title = _first_index(texts, CONTINUOUS_TREATMENT)

    detail_indexes = [i for i, t in enumerate(texts) if first_detail_marker in t]
    cat_details = [i for i in detail_indexes if idx_cat_title < i < idx_cont_title]
    cont_details = [i for i in detail_indexes if i > idx_cont_title]
    assert cat_details, "no categorical detail sections between the group titles"
    assert cont_details, "no continuous detail sections after the Continuous title"

    order = [idx_results, idx_banner, idx_overview, idx_shared, idx_cat_title,
             cat_details[0], idx_cont_title, cont_details[0]]
    assert order == sorted(order), f"overview-above-graphs order violated: {order}"

    # exactly one overview per population mode, all above the shared block, and
    # none of the old per-treatment overview rows
    for overview in ("Simulation Overview (Copula)",
                     "Simulation Overview (Research Spec)",
                     "Simulation Overview (Research Baseline)"):
        assert idx_banner < _first_index(texts, overview) < idx_shared, overview
    assert _count(texts, "Simulation Overview") == 3
    joined = "\n".join(texts)
    assert "Simulation Overview (Copula, Cat)" not in joined
    assert "Simulation Overview (Copula, Cont)" not in joined
    assert _count(texts, CATEGORICAL_TREATMENT) == 1
    assert _count(texts, CONTINUOUS_TREATMENT) == 1
    assert "All Population Modes Comparison" not in joined
    assert "Income Specification Comparison" not in joined
    return joined


# ---------------------------------------------------------------------------
# (a) Compare all population x Compare both income (6 result keys)
# ---------------------------------------------------------------------------
def test_apptest_compare_all_compare_both_interleaved():
    """Per-element (WTP) run AND whole-decision run both render
    title -> overview row -> detail row per income treatment."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_compare_all_app_script)
    at.session_state['rtd_tab_income_mode'] = 'Compare both'
    at.session_state['rtd_income_mode'] = 'Compare both'
    at.run(timeout=600)
    assert not at.exception

    # -- per-element run (WTP) --
    at.button(key='rtd_run_wtp_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] == 'wtp'
    assert sorted(at.session_state['simulation_results'].keys()) == [
        'copula_categorical', 'copula_continuous',
        'research_baseline_categorical', 'research_baseline_continuous',
        'research_spec_categorical', 'research_spec_continuous']

    texts = _ordered_texts(at)
    joined = _assert_interleaved_two_groups(texts, WTP_SECTION)

    # per-element filter still applies inside the interleaved layout: Willingness-to-Pay
    # USES income, so it renders once per result-key column and nothing is shared
    assert _count(texts, WTP_SECTION) == 6
    assert "Income-Independent Elements" not in joined
    for other in ALL_SECTIONS:
        if other != WTP_SECTION:
            assert other not in joined
    # a ranking element's run has NO element headline metric
    metric_labels = [m.label for m in at.metric]
    assert "Mean Willingness-to-Pay score" not in metric_labels
    assert "Avg. Options List Length" not in metric_labels
    assert metric_labels.count("Total Agents") >= 6

    # -- whole-decision run --
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] is None

    texts = _ordered_texts(at)
    # whole-decision run: every element section, then the
    # integrated first-option chart, per result-key column - except the income-free
    # elements, which are identical under both income specifications and therefore
    # render ONCE per population mode above the treatment groups, with the overview
    # row above them (overview above the graphs).
    joined = _assert_shared_overview_layout(texts, INTEGRATED_SECTION)
    assert _count(texts, INTEGRATED_SECTION) == 6  # one per result-key column
    idx_shared = _first_index(texts, "Income-Independent Elements")
    idx_cat_title = _first_index(texts, CATEGORICAL_TREATMENT)
    assert idx_shared < idx_cat_title
    assert any("identical for both income specifications" in t.lower() for t in texts)
    for section in INCOME_FREE_SECTIONS:
        assert _count(texts, section) == 3, section       # once per population mode
        assert _first_index(texts, section) < idx_cat_title
    for section in (WTP_SECTION, RT_SECTION):
        assert _count(texts, section) == 6, section       # once per result key
        assert _first_index(texts, section) > idx_cat_title
    # the whole-decision run shows no tie statistics
    assert "Tie statistics of the Kemeny aggregation" not in joined
    metric_labels = [m.label for m in at.metric]
    assert metric_labels.count("Avg. Options List Length") == 3   # one overview per population mode


# ---------------------------------------------------------------------------
# (b) Compare all population x single income mode (3 result keys, ONE group)
# ---------------------------------------------------------------------------
def test_apptest_compare_all_single_income_one_group():
    """One untitled group: overview row then detail row, no duplicate overview
    at the end of the page, no treatment titles - and nothing is split out as
    income-independent (only one income treatment is on the page)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_compare_all_app_script)
    at.session_state['rtd_tab_income_mode'] = 'Continuous only'
    at.session_state['rtd_income_mode'] = 'Continuous only'
    at.run(timeout=600)
    assert not at.exception

    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    assert sorted(at.session_state['simulation_results'].keys()) == [
        'copula_continuous', 'research_baseline_continuous',
        'research_spec_continuous']

    texts = _ordered_texts(at)
    joined = "\n".join(texts)

    idx_results = _first_index(texts, "📋 Decision Results")
    idx_overview = _first_index(texts, "Simulation Overview (Copula, Cont)")
    idx_detail = _first_index(texts, INTEGRATED_SECTION)
    assert idx_results < idx_overview < idx_detail

    # single group: no income-treatment titles, no shared block, no old summary grid
    assert CATEGORICAL_TREATMENT not in joined
    assert CONTINUOUS_TREATMENT not in joined
    assert "Income-Independent Elements" not in joined
    assert "All Population Modes Comparison" not in joined
    # all five element sections render inside the single group, once per column
    for section in ALL_SECTIONS:
        assert _count(texts, section) == 3, section

    # exactly one overview cell per population mode, all BEFORE the details
    # (i.e. no duplicated overview block at the end of the page)
    assert _count(texts, "Simulation Overview") == 3
    for overview in ("Simulation Overview (Copula, Cont)",
                     "Simulation Overview (Research Spec, Cont)",
                     "Simulation Overview (Research Baseline, Cont)"):
        assert _first_index(texts, overview) < idx_detail


# ---------------------------------------------------------------------------
# (c) Plain single-mode run: existing summary-first layout must not regress
# ---------------------------------------------------------------------------
def test_apptest_single_mode_summary_first_unchanged():
    """A plain single-mode Decision 4 run still shows '📊 Simulation Overview'
    at the top, BEFORE the '📋 Decision Results' section."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(rtd_tab_and_results_script)
    at.session_state['rtd_tab_income_mode'] = 'Continuous only'
    at.session_state['rtd_income_mode'] = 'Continuous only'
    at.run(timeout=600)
    assert not at.exception

    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    assert list(at.session_state['simulation_results'].keys()) == ['continuous']

    texts = _ordered_texts(at)
    joined = "\n".join(texts)

    idx_summary = _first_index(texts, "📊 Simulation Overview")
    idx_results = _first_index(texts, "📋 Decision Results")
    idx_detail = _first_index(texts, INTEGRATED_SECTION)
    assert idx_summary < idx_results < idx_detail

    # no comparison scaffolding on a single-mode run
    assert "Income Treatment" not in joined
    assert "All Population Modes Comparison" not in joined


# ---------------------------------------------------------------------------
# (d) A selected configuration replaces the comparison with that config alone
# ---------------------------------------------------------------------------
def test_apptest_selected_config_hides_the_other_configurations():
    """(B.5b) Once "Use This Config" has stored a configuration, the Decision 4 results
    show ONLY that configuration (all its elements and its integrated results) and the
    Excel export covers only it."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(rtd_tab_and_results_script)
    at.session_state['rtd_income_mode'] = 'Compare both'
    at.run(timeout=600)
    assert not at.exception

    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    assert sorted(at.session_state['simulation_results'].keys()) == ['categorical', 'continuous']

    texts = _ordered_texts(at)
    joined = "\n".join(texts)
    # before the selection: both income treatments are presented
    assert CATEGORICAL_TREATMENT in joined
    assert CONTINUOUS_TREATMENT in joined
    assert _count(texts, INTEGRATED_SECTION) == 2

    # select the continuous configuration (the handler calls st.rerun(); a follow-up
    # run gives the post-rerun page, as the other Decision 4 config tests do)
    at.button(key='rtd_inline_select_continuous').click().run(timeout=600)
    assert not at.exception
    at.run(timeout=600)
    assert not at.exception
    assert at.session_state['selected_decision_configs'][
        'rejected_transaction_defaults']['result_key'] == 'continuous'

    texts = _ordered_texts(at)
    joined = "\n".join(texts)
    # only the selected configuration's results remain
    assert CATEGORICAL_TREATMENT not in joined
    assert CONTINUOUS_TREATMENT not in joined
    assert "Income-Independent Elements" not in joined
    assert _count(texts, INTEGRATED_SECTION) == 1
    for section in ALL_SECTIONS:
        assert _count(texts, section) == 1, section
    assert any("Showing the selected configuration only" in t for t in texts)
    # the export section covers the selected configuration only (no per-config prefixes)
    assert "configurations - sheet names are prefixed" not in joined
    assert any("Selected configuration only." in t for t in texts)
    labels = [str(e.label) for e in at.get("download_button")]
    assert labels.count(DOWNLOAD_D4_ALL_ELEMENTS) == 1


# ---------------------------------------------------------------------------
# (e) Income-free per-element run under both income treatments: overview above
#     the graphs
# ---------------------------------------------------------------------------
def test_apptest_income_free_element_run_overview_above_graphs():
    """A per-element run of an income-free element (Flexibility) with Compare-both
    income used to render the shared graphs first and the overview rows below them.
    Now ONE income-neutral overview (per population mode) sits above the graphs, the
    element renders once, and both income treatments keep a "Use This Config" button."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(rtd_tab_and_results_script)
    at.session_state['rtd_income_mode'] = 'Compare both'
    at.run(timeout=600)
    assert not at.exception

    at.button(key='rtd_run_flexibility_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] == 'flexibility'
    assert sorted(at.session_state['simulation_results'].keys()) == ['categorical', 'continuous']

    texts = _ordered_texts(at)
    joined = "\n".join(texts)
    flex_section = "5️⃣ Flexibility Ranking"
    idx_overview = _first_index(texts, "Simulation Overview (Research Baseline)")
    idx_shared = _first_index(texts, "Income-Independent Elements")
    idx_flex = _first_index(texts, flex_section)
    assert idx_overview < idx_shared < idx_flex
    # income-neutral overview only: no per-treatment overview rows any more
    assert "Simulation Overview (Categorical)" not in joined
    assert "Simulation Overview (Continuous)" not in joined
    assert _count(texts, flex_section) == 1
    for other in ALL_SECTIONS:
        if other != flex_section:
            assert other not in joined, other
    # both income treatments remain selectable (the saved config carries the income mode)
    button_keys = [b.key for b in at.button]
    assert 'rtd_inline_select_categorical' in button_keys
    assert 'rtd_inline_select_continuous' in button_keys
    assert not any(str(c.value).startswith("📊 Quick Summary") for c in at.caption)


# =============================================================================
# from test_rtd_config_selection.py
# =============================================================================
# Decision 4 configuration selection - the same "Use This Config" flow as the other
# modelled decisions (disclose_income is the reference):
#
# - An individual Decision 4 run shows a "🎯 Use This Config" button UNDER its results
#   (after the workbook download, like the other decisions);
#   clicking it stores the run's tab settings (income mode, per-element intercepts,
#   rank-aggregation settings, stochastic UI) plus metrics in the
#   unified selected_decision_configs store and the button turns into "✅ Selected".
# - Page 2 "Saved Decision Configurations" shows the saved Decision 4 configuration with
#   a Clear button; the results page shows the selected configuration (Clear) and the
#   "Run Complete Simulation" section, like disclose_income.
# - Combined / complete simulations apply the saved MODEL settings (income mode,
#   intercepts, aggregation) instead of the tab state; individual Decision 4
#   runs keep reflecting the tab.
# - The complete-simulation gate blocks a Decision 4 "Compare both" (or Compare-all
#   population) selection until one configuration is selected, like disclose_income.

def _rtd_config_app_script():
    """Decision 4 tab + results (once run) + the Page-2 saved-config display, and the
    combined-run application probes stashed into session state for assertions."""
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

    from app.pages.page2_decisions import render_selected_rejected_transaction_config_display
    st.markdown("PAGE2_SAVED_CONFIGS_MARKER")
    render_selected_rejected_transaction_config_display()

    # Probe: how a combined run (inc_mode=None) vs an individual run (explicit inc_mode)
    # would configure the orchestrator with the current saved config / tab state.
    if st.session_state.get('_probe_apply'):
        from src.engine.core import Engine
        from app.simulation import _apply_rejected_transaction_config
        probe = {}
        for label, inc_mode in (('combined', None), ('individual', 'continuous')):
            orch = Engine("baseline")
            _apply_rejected_transaction_config(orch, "baseline", inc_mode)
            cfg = orch.config['rejected_transaction_defaults']
            probe[label] = {
                'income_mode': cfg.get('income_mode'),
                'intercepts': dict(cfg.get('intercepts', {})),
                'aggregation': dict(cfg.get('aggregation', {})),
            }
        st.session_state['_probe_result'] = probe

    if st.session_state.get('_probe_gate'):
        # a Decision 4 selection on Page 2 (the gate reads decision_params.selected_decisions)
        st.session_state.decision_params.selected_decisions = ['rejected_transaction_defaults']
        from app.pages.decision_execution import can_run_complete_simulation
        can_run, reason, count, block_type, issues = can_run_complete_simulation()
        st.session_state['_gate_result'] = {
            'can_run': can_run, 'reason': reason, 'count': count,
            'block_type': block_type, 'issues': [i['decision'] for i in issues]}


def _success_texts(at):
    return [str(e.value) for e in at.success]


def _saved_configs(at):
    try:
        return dict(at.session_state['selected_decision_configs'])
    except KeyError:
        return {}


def test_apptest_select_display_apply_and_clear():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_config_app_script)
    at.run(timeout=600)
    assert not at.exception
    assert 'rejected_transaction_defaults' not in _saved_configs(at)

    # individual Decision 4 run -> "Use This Config" offered under the results
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['custom_decisions'] == ['rejected_transaction_defaults']
    results = at.session_state['simulation_results']
    result_key = next(iter(results.keys()))
    assert result_key == 'continuous'            # tab default income mode
    select_btn = at.button(key=f'rtd_inline_select_{result_key}')
    assert select_btn is not None
    assert "4. Rejected Transaction Defaults" not in all_markdown(at).split("PAGE2_SAVED_CONFIGS_MARKER")[-1]
    # no "Quick Summary" caption next to the button
    captions = [str(c.value) for c in at.caption]
    assert not any(c.startswith("📊 Quick Summary") for c in captions)
    # no "Run Complete Simulation" offer before a configuration is selected
    assert "🚀 Run Complete Simulation" not in all_markdown(at).split("PAGE2_SAVED_CONFIGS_MARKER")[0]

    # select it
    select_btn.click().run(timeout=600)
    assert not at.exception
    configs = at.session_state['selected_decision_configs']
    assert 'rejected_transaction_defaults' in configs
    cfg = configs['rejected_transaction_defaults']
    assert cfg['result_key'] == result_key
    assert cfg['income_mode'] == 'Continuous only'
    assert cfg['population_mode'] == 'Research Baseline'
    assert cfg['source'] == 'individual_rejected_transaction_defaults_run'
    assert cfg['total_agents'] == 60
    assert set(cfg['params']['intercepts']) == {'ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility'}
    assert cfg['params']['intercepts']['ttp'] == pytest.approx(0.0)      # research default (0 since 2026-09-17)
    assert cfg['params']['aggregation'] == {'enabled': True}
    assert cfg['params']['flexibility_anchor'] == {'observed_weight': 0.25, 'calculated_weight': 0.75}
    assert cfg['params']['stochastic']['sigma_strategy'] == 'overall'
    assert 'mean_choice_length' in cfg['metrics'] and 'first_option_shares' in cfg['metrics']
    assert abs(cfg['metrics']['mean_choice_length'] - results[result_key]['rtd_choice_length'].mean()) < 1e-9
    # button replaced by the selected marker, which is itself an Unselect button
    # (unselecting must be possible at any point); Page 2 display present
    assert f'rtd_inline_select_{result_key}' not in [b.key for b in at.button]   # replaced by the marker
    unselect_btn = at.button(key=f'rtd_inline_unselect_{result_key}')
    assert unselect_btn is not None and SELECTED_LABEL in str(unselect_btn.label)
    assert at.button(key='rtd_tab_unselect_btn') is not None                    # on the Decision 4 tab too
    page2 = all_markdown(at).split("PAGE2_SAVED_CONFIGS_MARKER")[-1]
    assert "4. Rejected Transaction Defaults" in page2
    assert any("Rejected Transaction Defaults Configuration" in t and "Research Baseline + Continuous only" in t
               for t in _success_texts(at))
    # like disclose_income: the results page now shows the selected configuration
    # (with Clear) and offers "Run Complete Simulation" right there
    results_md = all_markdown(at).split("PAGE2_SAVED_CONFIGS_MARKER")[0]
    assert "🚀 Run Complete Simulation" in results_md
    assert at.button(key='run_complete_from_results') is not None
    assert at.button(key='clear_rtd_selection') is not None
    assert any("Selected Rejected Transaction Defaults Configuration" in t for t in _success_texts(at))

    # combined-run application: change the tab afterwards (through its widgets) ->
    # combined runs still use the saved model settings, individual runs follow the tab.
    # The aggregation enable checkbox was removed from the tab (the rank aggregation is
    # always on); tolerate either tab state so this test does not depend on it.
    at.number_input(key='rtd_tab_intercept_ttp').set_value(0.1)   # within the raw-scale β0 range ±0.25
    agg_toggle = next((c for c in at.checkbox if c.key == 'rtd_tab_aggregation_enabled'), None)
    if agg_toggle is not None:
        agg_toggle.uncheck()
    expected_agg = agg_toggle is None
    at.radio(key='rtd_tab_income_mode').set_value('Categorical only')
    at.session_state['_probe_apply'] = True
    at.run(timeout=600)
    assert at.session_state['rtd_intercept_ttp'] == pytest.approx(0.1)
    assert at.session_state['rtd_aggregation_enabled'] is expected_agg
    assert not at.exception
    probe = at.session_state['_probe_result']
    assert probe['combined']['income_mode'] == 'continuous'
    assert probe['combined']['intercepts']['ttp'] == pytest.approx(0.0)
    assert probe['combined']['aggregation']['enabled'] is True
    assert probe['individual']['income_mode'] == 'continuous'      # explicit inc_mode of the sub-run
    assert probe['individual']['intercepts']['ttp'] == pytest.approx(0.1)
    assert probe['individual']['aggregation']['enabled'] is expected_agg

    # clear (the button sits below the display it removes; a follow-up run shows the
    # post-clear page like the st.rerun() does in the app)
    at.button(key='clear_rejected_transaction_config').click().run(timeout=600)
    assert not at.exception
    at.run(timeout=600)
    assert 'rejected_transaction_defaults' not in at.session_state['selected_decision_configs']
    assert "4. Rejected Transaction Defaults" not in all_markdown(at).split("PAGE2_SAVED_CONFIGS_MARKER")[-1]
    # the results-page complete-simulation offer disappears with the selection
    button_keys = [b.key for b in at.button]
    assert 'run_complete_from_results' not in button_keys
    assert 'clear_rtd_selection' not in button_keys
    assert 'rtd_tab_unselect_btn' not in button_keys
    # ... and the alternatives are selectable again
    assert f'rtd_inline_select_{result_key}' in button_keys


def test_apptest_complete_simulation_gate_requires_a_selected_config():
    """Decision 4 selected with 'Compare both' -> 2 configurations -> the complete
    simulation is blocked until one is selected; a saved config lifts the block and
    is named in the success reason."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_config_app_script)
    at.run(timeout=600)
    at.radio(key='rtd_tab_income_mode').set_value('Compare both')
    at.session_state['_probe_gate'] = True
    at.run(timeout=600)
    assert at.session_state['rtd_income_mode'] == 'Compare both'
    assert not at.exception
    gate = at.session_state['_gate_result']
    assert gate['can_run'] is False
    assert gate['block_type'] == 'rejected_transaction_defaults'
    assert gate['issues'] == ['rejected_transaction_defaults']
    assert gate['count'] == 2

    # a selected configuration lifts the block
    at.session_state['selected_decision_configs'] = {'rejected_transaction_defaults': {
        'result_key': 'continuous', 'params': {'income_mode': 'Compare both'},
        'income_mode': 'Continuous only', 'population_mode': 'Research Baseline',
        'metrics': {}, 'total_agents': 60, 'source': 'individual_rejected_transaction_defaults_run',
        'original_seed': 42, 'original_n_agents': 60}}
    at.run(timeout=600)
    assert not at.exception
    gate = at.session_state['_gate_result']
    assert gate['can_run'] is True
    assert "Rejected Transaction Defaults: Research Baseline + Continuous only" in gate['reason']

    # single income mode never blocks
    at.session_state['selected_decision_configs'] = {}
    at.radio(key='rtd_tab_income_mode').set_value('Continuous only')
    at.run(timeout=600)
    assert at.session_state['_gate_result']['can_run'] is True


# =============================================================================
# from test_rtd_element_subtitle.py
# =============================================================================
# Requirement: "When running decision elements only, below the '4. Rejected Transaction
# Defaults' title, add a subtitle for the decision element to which the results
# correspond, e.g., in the results for Option List Length, add '4.1 Options List Length
# (Tendency to Plan)' subtitle. Same for other decision elements."
#
# The subtitle is the element's Decision 4 sub-tab number and name, drawn directly under
# the decision title with the title's own heading class; whole-decision runs get none.

EXPECTED = {
    'rtd_run_ttp_btn': D4_SUBTITLE_TTP,
    'rtd_run_loyalty_btn': D4_SUBTITLE_LOYALTY,
    'rtd_run_wtp_btn': D4_SUBTITLE_WTP,
    'rtd_run_risk_taking_btn': D4_SUBTITLE_RISK_TAKING,
    'rtd_run_flexibility_btn': D4_SUBTITLE_FLEXIBILITY,
    'rtd_run_aggregation_btn': D4_SUBTITLE_AGGREGATION,
}


def test_subtitles_follow_the_decision_4_tab():
    from app.pages.decision_tabs.rejected_transaction import (
        AGGREGATION_TITLE, MECH_TITLES)
    for mech, tab_title in MECH_TITLES.items():
        index, name = RTD_ELEMENT_SUBTITLES[mech]
        assert tab_title == f"{index}. {name}"
    index, name = RTD_ELEMENT_SUBTITLES['aggregation']
    assert AGGREGATION_TITLE == f"{index}. {name}"
    assert rtd_element_subtitle(4, 'ttp') == D4_SUBTITLE_TTP
    assert rtd_element_subtitle(4, None) is None


def _subtitle_app_script():
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
    at = AppTest.from_function(_subtitle_app_script, default_timeout=600)
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
    at = AppTest.from_function(_subtitle_app_script, default_timeout=600)
    at.run()
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run()
    assert not at.exception
    assert _subtitles(at) == []


def test_comparison_element_run_shows_the_subtitle_once():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_subtitle_app_script, default_timeout=600)
    at.session_state['_test_population'] = 'Compare all'
    at.run()
    at.button(key='rtd_run_wtp_btn').click().run()
    assert not at.exception
    md = _markdown(at)
    assert md[_title_index(md) + 1].endswith(f'>{D4_SUBTITLE_WTP}</h5>')
    assert len(_subtitles(at)) == 1


# =============================================================================
# from test_rtd_element_summary.py
# =============================================================================
# After "Use This Config" on a Decision 4 ELEMENT page, the
# selected-configuration summary must not show list-length averages unrelated to that
# element. The summary keeps only what belongs to the element:
#
# * whole decision run                         -> options list length AND integrated default list length;
# * Options List Length (ttp) element          -> options list length only;
# * Integrated Default List element            -> integrated default list length only;
# * Loyalty / WTP / Risk-Taking / Flexibility  -> neither.
#
# Checked on the results page (the selected-configuration card and the "saved
# configuration(s) will be used" line of the Run Complete Simulation section) and on the
# Page-2 saved-configuration display (rendered in the sidebar here, to keep it apart).

def test_summary_metric_keys_per_element():
    assert rtd_summary_metric_keys(None) == ('mean_choice_length', 'mean_default_list_length')
    assert rtd_summary_metric_keys('ttp') == ('mean_choice_length',)
    assert rtd_summary_metric_keys('aggregation') == ('mean_default_list_length',)
    for element in ('loyalty', 'wtp', 'risk_taking', 'flexibility'):
        assert rtd_summary_metric_keys(element) == ()


def _summary_app_script():
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

    at = AppTest.from_function(_summary_app_script)
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


# =============================================================================
# from test_rtd_score_chart_matches_stats.py
# =============================================================================
# Reported problem: "For the Options List Length results, the graph for Tendency to Plan
# score for Research Spec is identical to that for Research Baseline even though the
# statistics are different. This problem occurs for all the other decision elements as
# well."
#
# Root cause: every score chart plotted the DETERMINISTIC score (weighted_ttp, the
# standardized loyalty / WTP / RT / anchored-flexibility scores), which is the same for
# Research Specification and Research Baseline (same 280 participants), while the
# allocation charts, their tables and the overview statistics count the outcome of the
# Normal(anchor, sigma) DRAW (document: draw_k ~ Normal(anchor, sigma), re-rescaled over
# the population and re-binned).
#
# Now, whenever an element's scores were drawn, its chart plots the drawn scores, so
# that the plotted values, re-cut into equal-width bins over their own min..max exactly
# as the engine does, reproduce the very lengths / segments the statistics count -
# agent by agent. Research Baseline (no draw) is unchanged: the deterministic score,
# the Stata figure.

SEGMENT_COLS = {'loyalty': 'rtd_loyalty_segment', 'wtp': 'rtd_wtp_segment',
                'risk_taking': 'rtd_rt_segment', 'flexibility': 'rtd_flex_segment'}


def _score_chart_app_script():
    import streamlit as st
    from app.models import initialize_session_state
    initialize_session_state()
    st.session_state.population_mode = st.session_state.get('_test_population',
                                                            'Research Specification')
    st.session_state.n_agents = 280
    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()


def _run(population):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_score_chart_app_script, default_timeout=600)
    at.session_state['_test_population'] = population
    at.run()
    assert not at.exception
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run()
    assert not at.exception
    results = at.session_state['simulation_results']
    return {k: v for k, v in results.items()}


@pytest.fixture(scope="module")
def spec_frame():
    return next(iter(_run('Research Specification').values()))


@pytest.fixture(scope="module")
def baseline_frame():
    return next(iter(_run('Research Baseline').values()))


def _rebin(values, mech):
    """The engine's re-binning of drawn values: min-max over the population, then
    floor(low + (span - 0.0001) * u), clipped to the element's range."""
    x = np.asarray(values, dtype=float)
    low, span = _RESCALE[mech]
    u = (x - x.min()) / (x.max() - x.min())
    out = np.floor(low + (span - 0.0001) * u).astype(int)
    return np.clip(out, int(low), int(low) + int(span) - 1)


def _plotted(df, mech, monkeypatch):
    """The series the results page hands to the histogram for this element."""
    captured = {}
    monkeypatch.setattr(viz, '_rtd_density_hist',
                        lambda series, title, x_title, key: captured.update(
                            series=pd.Series(series), title=title, x_title=x_title))

    def both(a, b):
        a()
    if mech == 'ttp':
        viz._render_rtd_ttp_section(df, 'rtd', '', both, False)
    else:
        viz._render_rtd_ranking_section(df, 'rtd', '', both, mech, False)
    return captured


@pytest.mark.parametrize("mech", list(RTD_SCORE_CHARTS))
def test_spec_chart_plots_the_values_the_statistics_count(spec_frame, monkeypatch, mech):
    df = spec_frame
    plotted = _plotted(df, mech, monkeypatch)
    series = plotted['series'].to_numpy(dtype=float)
    assert "stochastic draw" in plotted['title']
    if mech == 'ttp':
        counted = df['rtd_choice_length'].astype(int).to_numpy()
    else:
        counted = df[SEGMENT_COLS[mech]].astype(int).to_numpy()
    # graph and statistics describe the same values, agent by agent
    assert (_rebin(series, mech) == counted).all()
    # ... which the deterministic score (the Baseline graph) does not
    det_col = RTD_SCORE_CHARTS[mech][0]
    assert not np.allclose(series, df[det_col].to_numpy(dtype=float))


@pytest.mark.parametrize("mech", list(RTD_SCORE_CHARTS))
def test_baseline_chart_is_unchanged(baseline_frame, monkeypatch, mech):
    """Research Baseline: the deterministic score (the Stata figure), as before."""
    df = baseline_frame
    plotted = _plotted(df, mech, monkeypatch)
    det_col, title = RTD_SCORE_CHARTS[mech][:2]
    assert plotted['title'] == title and plotted['x_title'] == title
    assert plotted['series'].to_numpy(dtype=float) == pytest.approx(
        df[det_col].to_numpy(dtype=float))
    if mech == 'ttp':
        assert (_rebin(plotted['series'], mech)
                == df['rtd_choice_length'].astype(int).to_numpy()).all()


def test_spec_and_baseline_charts_differ_while_baseline_matches_stata(spec_frame,
                                                                     baseline_frame):
    spec, _, _, drawn = rtd_score_chart_data(spec_frame, 'ttp')
    base, _, _, base_drawn = rtd_score_chart_data(baseline_frame, 'ttp')
    assert drawn and not base_drawn
    assert np.histogram(spec, bins=16)[0].tolist() != np.histogram(base, bins=16)[0].tolist()
    # Baseline still reproduces the .dta's length distribution
    assert baseline_frame['rtd_choice_length'].value_counts().to_dict() == \
        {0: 20, 1: 92, 2: 95, 3: 57, 4: 14, 5: 2}


def test_undrawn_agents_fall_back_to_their_anchor():
    """Quintiles mode with one budget level's coefficient at 0: those agents carry no
    draw; the chart plots their anchor (the value the draw range used for them)."""
    df = pd.DataFrame({
        'rtd_weighted_ttp': [0.0, 0.1, 0.2],
        'rtd_weighted_ttp06': [0.0, 3.0, 5.9999],
        'rtd_ttp_draw': [0.4, np.nan, 6.2],
    })
    series, title, x_title, drawn = rtd_score_chart_data(df, 'ttp')
    assert drawn
    assert series.tolist() == [0.4, 3.0, 6.2]


def test_no_draw_column_or_all_missing_means_deterministic():
    df = pd.DataFrame({'rtd_loyalty_z': [0.1, 0.2], 'rtd_loyalty_draw': [np.nan, np.nan]})
    series, title, _, drawn = rtd_score_chart_data(df, 'loyalty')
    assert not drawn and title == "Loyalty score"
    assert series.tolist() == [0.1, 0.2]
