"""
UI layout tests for the Decision 4 comparison results page.

For an INDIVIDUAL Decision 4 run in a comparison mode, the summary and details
must be INTERLEAVED per income treatment inside the "📋 Decision Results"
section:

    📋 Decision Results (+ green custom-parameters banner)
    Categorical Income Treatment
        [overview cells - one per population mode]
        [detailed element sections - one column per population mode]
    Continuous Income Treatment
        [overview cells]
        [detailed sections]

instead of the old split layout (both treatment summaries first, then both
detail rows). Single-income comparison runs degenerate to one untitled group
(overview row then detail row), and plain single-mode runs keep the existing
summary-first layout ("📊 Simulation Overview" before "📋 Decision Results").
"""


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


def _rtd_single_mode_app_script():
    """Decision 4 tab in a plain single mode + the results page."""
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
ALL_SECTIONS = ("1️⃣ Options List Length (Tendency to Plan)",
                "2️⃣ Loyalty Ranking",
                "3️⃣ Willingness-to-Pay Ranking",
                "4️⃣ Risk-Taking Ranking")


def _assert_interleaved_two_groups(texts, first_detail_marker):
    """Order: Decision Results header < custom-parameters banner
    < Categorical title < categorical overview < categorical details
    < Continuous title < continuous overview < continuous details."""
    idx_results = _first_index(texts, "📋 Decision Results")
    idx_banner = _first_index(texts, "custom parameters")
    idx_cat_title = _first_index(texts, "Categorical Income Treatment")
    idx_cat_overview = _first_index(texts, "Simulation Overview (Copula, Cat)")
    idx_cont_title = _first_index(texts, "Continuous Income Treatment")
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
    assert _count(texts, "Categorical Income Treatment") == 1
    assert _count(texts, "Continuous Income Treatment") == 1
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

    # per-element filter still applies inside the interleaved layout:
    # one WTP section per result-key column, no other element sections
    assert _count(texts, WTP_SECTION) == 6
    for other in ALL_SECTIONS:
        if other != WTP_SECTION:
            assert other not in joined
    # element-aware overview metric in every overview cell
    metric_labels = [m.label for m in at.metric]
    assert metric_labels.count("Mean Willingness-to-Pay score") == 6
    assert "Avg. Options List Length" not in metric_labels

    # -- whole-decision run --
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['rtd_run_element'] is None

    texts = _ordered_texts(at)
    joined = _assert_interleaved_two_groups(texts, ALL_SECTIONS[0])
    for section in ALL_SECTIONS:
        assert _count(texts, section) == 6  # one per result-key column
    metric_labels = [m.label for m in at.metric]
    assert metric_labels.count("Avg. Options List Length") == 6


# ---------------------------------------------------------------------------
# (b) Compare all population x single income mode (3 result keys, ONE group)
# ---------------------------------------------------------------------------
def test_apptest_compare_all_single_income_one_group():
    """One untitled group: overview row then detail row, no duplicate overview
    at the end of the page, no treatment titles."""
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
    idx_detail = _first_index(texts, ALL_SECTIONS[0])
    assert idx_results < idx_overview < idx_detail

    # single group: no income-treatment titles, no old summary grid
    assert "Categorical Income Treatment" not in joined
    assert "Continuous Income Treatment" not in joined
    assert "All Population Modes Comparison" not in joined

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

    at = AppTest.from_function(_rtd_single_mode_app_script)
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
    idx_detail = _first_index(texts, ALL_SECTIONS[0])
    assert idx_summary < idx_results < idx_detail

    # no comparison scaffolding on a single-mode run
    assert "Income Treatment" not in joined
    assert "All Population Modes Comparison" not in joined
