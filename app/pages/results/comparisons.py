# app/pages/results/comparisons.py
"""
Comparison functions for different simulation modes.

The grids are shaped by the run that produced the results (ruling R28): the
RunContext says whether the run compared both income specifications, and which
single income type its keys carry.
"""
import streamlit as st
from app.components import show_overview
from app.pages.results.run_context import RunContext


def should_enable_selection(ctx=None):
    """Check if selection buttons should be enabled for individual decision runs
    (donation_default, disclose_income, disclose_documents or rejected_transaction_defaults)"""
    ctx = ctx if ctx is not None else RunContext.from_session()
    return ctx.should_enable_selection


def render_all_modes_comparison(results_dict, ctx=None):
    """Render comparison of all three population modes"""
    ctx = ctx if ctx is not None else RunContext.from_session()
    enable_selection = should_enable_selection(ctx)

    st.markdown("### 🔬 All Population Modes Comparison")

    if ctx.is_compare_both:
        # 3x2 grid: copula vs research_spec vs research_baseline x categorical vs continuous
        st.markdown("#### Categorical Income Treatment")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🧬 Copula (Synthetic)**")
            if "copula_categorical" in results_dict:
                show_overview(
                    results_dict["copula_categorical"],
                    " (Copula, Cat)",
                    result_key="copula_categorical",
                    enable_selection=enable_selection
                )
            else:
                st.caption("Copula categorical results not available")

        with col2:
            st.markdown("**📄 Research Specification**")
            if "research_spec_categorical" in results_dict:
                show_overview(
                    results_dict["research_spec_categorical"],
                    " (Research Spec, Cat)",
                    result_key="research_spec_categorical",
                    enable_selection=enable_selection
                )
            else:
                st.caption("Research Specification categorical results not available")

        with col3:
            st.markdown("**⚖️ Research Baseline**")
            if "research_baseline_categorical" in results_dict:
                show_overview(
                    results_dict["research_baseline_categorical"],
                    " (Research Baseline, Cat)",
                    result_key="research_baseline_categorical",
                    enable_selection=enable_selection
                )
            else:
                st.caption("Research Baseline categorical results not available")

        st.markdown("---")
        st.markdown("#### Continuous Income Treatment")
        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown("**🧬 Copula (Synthetic)**")
            if "copula_continuous" in results_dict:
                show_overview(
                    results_dict["copula_continuous"],
                    " (Copula, Cont)",
                    result_key="copula_continuous",
                    enable_selection=enable_selection
                )
            else:
                st.caption("Copula continuous results not available")

        with col5:
            st.markdown("**📄 Research Specification**")
            if "research_spec_continuous" in results_dict:
                show_overview(
                    results_dict["research_spec_continuous"],
                    " (Research Spec, Cont)",
                    result_key="research_spec_continuous",
                    enable_selection=enable_selection
                )
            else:
                st.caption("Research Specification continuous results not available")

        with col6:
            st.markdown("**⚖️ Research Baseline**")
            if "research_baseline_continuous" in results_dict:
                show_overview(
                    results_dict["research_baseline_continuous"],
                    " (Research Baseline, Cont)",
                    result_key="research_baseline_continuous",
                    enable_selection=enable_selection
                )
            else:
                st.caption("Research Baseline continuous results not available")
    else:
        # Single income mode, compare all three population modes.
        # The income type is the run's own (case-insensitive: donation uses
        # "continuous only" while Decisions 1/2/4 use "Continuous only").
        income_type = ctx.income_type
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🧬 Copula (Synthetic)")
            copula_key = f"copula_{income_type}"
            if copula_key in results_dict:
                show_overview(
                    results_dict[copula_key],
                    f" (Copula, {income_type.title()})",
                    result_key=copula_key,
                    enable_selection=enable_selection
                )
            else:
                st.caption(f"Copula {income_type} results not available")

        with col2:
            st.markdown("#### 📄 Research Specification")
            research_spec_key = f"research_spec_{income_type}"
            if research_spec_key in results_dict:
                show_overview(
                    results_dict[research_spec_key],
                    f" (Research Spec, {income_type.title()})",
                    result_key=research_spec_key,
                    enable_selection=enable_selection
                )
            else:
                st.caption(f"Research Specification {income_type} results not available")

        with col3:
            st.markdown("#### ⚖️ Research Baseline")
            baseline_key = f"research_baseline_{income_type}"
            if baseline_key in results_dict:
                show_overview(
                    results_dict[baseline_key],
                    f" (Research Baseline, {income_type.title()})",
                    result_key=baseline_key,
                    enable_selection=enable_selection
                )
            else:
                st.caption(f"Research Baseline {income_type} results not available")


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
