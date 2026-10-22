# app/pages/results/comparisons.py
"""
Comparison functions for different simulation modes.

The grids are shaped by the run that produced the results, not by the live
mode keys: the
RunContext says whether the run compared both income specifications, and which
single income type its keys carry.
"""
import streamlit as st
from app.pages.results.overview import show_overview
from app.pages.results.run_context import RunContext


def should_enable_selection(ctx=None):
    """Check if selection buttons should be enabled for individual decision runs
    (donation_default, disclose_income, disclose_documents or rejected_transaction_defaults)"""
    ctx = ctx if ctx is not None else RunContext.from_session()
    return ctx.should_enable_selection


# The three population modes of a Compare-all grid, in column order: result-key
# prefix, heading, short name (cell titles) and long name (the "not available"
# captions).
_POPULATION_COLUMNS = (
    ("copula", "🧬 Copula (Synthetic)", "Copula", "Copula"),
    ("research_spec", "📄 Research Specification", "Research Spec", "Research Specification"),
    ("research_baseline", "⚖️ Research Baseline", "Research Baseline", "Research Baseline"),
)


def _population_cell(results_dict, heading, result_key, title_suffix, missing_caption, enable_selection):
    """One cell of a Compare-all grid: the population heading, then that
    configuration's overview (or a caption when the run has no such result)."""
    st.markdown(heading)
    if result_key in results_dict:
        show_overview(
            results_dict[result_key],
            title_suffix,
            result_key=result_key,
            enable_selection=enable_selection
        )
    else:
        st.caption(missing_caption)


def render_all_modes_comparison(results_dict, ctx=None):
    """Render comparison of all three population modes"""
    ctx = ctx if ctx is not None else RunContext.from_session()
    enable_selection = should_enable_selection(ctx)

    st.markdown("### 🔬 All Population Modes Comparison")

    if ctx.is_compare_both:
        # 3x2 grid: copula vs research_spec vs research_baseline x categorical vs continuous
        income_rows = (("categorical", "Cat", "#### Categorical Income Treatment"),
                       ("continuous", "Cont", "#### Continuous Income Treatment"))
        for row, (income, abbreviation, row_heading) in enumerate(income_rows):
            if row:
                st.markdown("---")
            st.markdown(row_heading)
            for column, (population, heading, short_name, long_name) in zip(
                    st.columns(3), _POPULATION_COLUMNS):
                with column:
                    _population_cell(
                        results_dict, f"**{heading}**", f"{population}_{income}",
                        f" ({short_name}, {abbreviation})",
                        f"{long_name} {income} results not available", enable_selection)
    else:
        # Single income mode, compare all three population modes.
        # The income type is the run's own (case-insensitive: donation uses
        # "continuous only" while Decisions 1/2/4 use "Continuous only").
        income_type = ctx.income_type
        for column, (population, heading, short_name, long_name) in zip(
                st.columns(3), _POPULATION_COLUMNS):
            with column:
                _population_cell(
                    results_dict, f"#### {heading}", f"{population}_{income_type}",
                    f" ({short_name}, {income_type.title()})",
                    f"{long_name} {income_type} results not available", enable_selection)


def render_income_comparison(results_dict, ctx=None):
    """Render income specification comparison results"""
    ctx = ctx if ctx is not None else RunContext.from_session()
    enable_selection = should_enable_selection(ctx)

    st.markdown("### 📊 Income Specification Comparison")

    col_cat, col_cont = st.columns(2, gap="large")

    with col_cat:
        st.markdown("#### 📋 Categorical Income")
        if "categorical" in results_dict:
            show_overview(
                results_dict["categorical"],
                " (Categorical)",
                result_key="categorical",
                enable_selection=enable_selection
            )
        else:
            st.caption("Categorical results not available")

    with col_cont:
        st.markdown("#### 📈 Continuous Income")
        if "continuous" in results_dict:
            show_overview(
                results_dict["continuous"],
                " (Continuous)",
                result_key="continuous",
                enable_selection=enable_selection
            )
        else:
            st.caption("Continuous results not available")
