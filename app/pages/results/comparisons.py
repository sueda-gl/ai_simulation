# app/pages/results/comparisons.py
"""
Comparison functions for different simulation modes.
"""
import streamlit as st
from app.components import show_overview, show_dependent_variable_comparison


def should_enable_selection():
    """Check if selection buttons should be enabled for individual decision runs (donation_default or disclose_income)"""
    return (
        hasattr(st.session_state, 'custom_decisions') and
        st.session_state.custom_decisions in [['donation_default'], ['disclose_income'], ['disclose_documents']] and
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) == 0  # Individual runs have empty default_decisions
    )


def render_population_comparison(results_dict):
    """Render population mode comparison results"""
    st.markdown("### 🔬 Population Mode Comparison")
    
    if st.session_state.income_spec_mode == "Compare both":
        # 2x2 grid: copula vs doc_mode x categorical vs continuous
        st.markdown("#### Copula (Synthetic Agents)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Categorical Income**")
            if "copula_categorical" in results_dict:
                show_overview(
                    results_dict["copula_categorical"], 
                    " (Copula, Cat)",
                    result_key="copula_categorical",
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Categorical results not available")
        with col2:
            st.markdown("**Continuous Income**")
            if "copula_continuous" in results_dict:
                show_overview(
                    results_dict["copula_continuous"], 
                    " (Copula, Cont)",
                    result_key="copula_continuous",
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Continuous results not available")
        
        st.markdown("---")
        st.markdown("#### Research Mode (Original + Stochastic)")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Categorical Income**")
            if "doc_mode_categorical" in results_dict:
                show_overview(
                    results_dict["doc_mode_categorical"], 
                    " (Research, Cat)",
                    result_key="doc_mode_categorical",
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Categorical results not available")
        with col4:
            st.markdown("**Continuous Income**")
            if "doc_mode_continuous" in results_dict:
                show_overview(
                    results_dict["doc_mode_continuous"], 
                    " (Research, Cont)",
                    result_key="doc_mode_continuous",
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Continuous results not available")
    else:
        # Single income mode, compare population modes
        income_type = "continuous" if st.session_state.income_spec_mode == "continuous only" else "categorical"
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧬 Copula (Synthetic)")
            copula_key = f"copula_{income_type}"
            if copula_key in results_dict:
                show_overview(
                    results_dict[copula_key], 
                    f" (Copula, {income_type.title()})",
                    result_key=copula_key,
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption(f"Copula {income_type} results not available")
        
        with col2:
            st.markdown("#### 📄 Research Mode")
            doc_key = f"doc_mode_{income_type}"
            if doc_key in results_dict:
                show_overview(
                    results_dict[doc_key], 
                    f" (Research, {income_type.title()})",
                    result_key=doc_key,
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption(f"Research {income_type} results not available")


def render_all_modes_comparison(results_dict):
    """Render comparison of all three population modes"""
    st.markdown("### 🔬 All Population Modes Comparison")
    
    if st.session_state.income_spec_mode == "Compare both":
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
                    enable_selection=should_enable_selection()
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
                    enable_selection=should_enable_selection()
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
                    enable_selection=should_enable_selection()
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
                    enable_selection=should_enable_selection()
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
                    enable_selection=should_enable_selection()
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
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Research Baseline continuous results not available")
    else:
        # Single income mode, compare all three population modes
        income_type = "continuous" if st.session_state.income_spec_mode == "continuous only" else "categorical"
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🧬 Copula (Synthetic)")
            copula_key = f"copula_{income_type}"
            if copula_key in results_dict:
                show_overview(
                    results_dict[copula_key], 
                    f" (Copula, {income_type.title()})",
                    result_key=copula_key,
                    enable_selection=should_enable_selection()
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
                    enable_selection=should_enable_selection()
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
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption(f"Research Baseline {income_type} results not available")


def render_dependent_variable_results(results_dict):
    """Render dependent variable resampling results"""
    st.markdown("### 📊 Dependent Variable Resampling")
    st.caption("This mode resamples from the empirical distribution of donation rates computed from the original 280 participants. No trait information is preserved.")
    
    df = results_dict["depvar"]
    show_dependent_variable_comparison(df)


def render_income_comparison(results_dict):
    """Render income specification comparison results"""
    st.markdown("### 📊 Income Specification Comparison")
    
    col_cat, col_cont = st.columns(2, gap="large")
    
    with col_cat:
        st.markdown("#### 📋 Categorical Income")
        if "categorical" in results_dict:
            show_overview(
                results_dict["categorical"], 
                " (Categorical)",
                result_key="categorical",
                enable_selection=should_enable_selection()
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
                enable_selection=should_enable_selection()
            )
        else:
            st.caption("Continuous results not available")


def render_disclose_income_all_modes_comparison(results_dict):
    """Render comparison of all three population modes for disclose_income"""
    from app.components import show_disclose_income_overview
    
    st.markdown("### 🔬 Disclose Income - All Population Modes Comparison")
    
    if st.session_state.income_spec_mode == "Compare both":
        # 3x2 grid: copula vs research_spec vs research_baseline x categorical vs continuous
        st.markdown("#### Categorical Income Treatment")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🧬 Copula (Synthetic)**")
            if "copula_categorical" in results_dict:
                show_disclose_income_overview(
                    results_dict["copula_categorical"], 
                    " (Copula, Cat)",
                    result_key="copula_categorical",
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Copula categorical results not available")
        
        with col2:
            st.markdown("**📄 Research Specification**")
            if "research_spec_categorical" in results_dict:
                show_disclose_income_overview(
                    results_dict["research_spec_categorical"], 
                    " (Research Spec, Cat)",
                    result_key="research_spec_categorical",
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Research Specification categorical results not available")
        
        with col3:
            st.markdown("**⚖️ Research Baseline**")
            if "research_baseline_categorical" in results_dict:
                show_disclose_income_overview(
                    results_dict["research_baseline_categorical"], 
                    " (Research Baseline, Cat)",
                    result_key="research_baseline_categorical",
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Research Baseline categorical results not available")
        
        st.markdown("---")
        st.markdown("#### Continuous Income Treatment")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown("**🧬 Copula (Synthetic)**")
            if "copula_continuous" in results_dict:
                show_disclose_income_overview(
                    results_dict["copula_continuous"], 
                    " (Copula, Cont)",
                    result_key="copula_continuous",
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Copula continuous results not available")
        
        with col5:
            st.markdown("**📄 Research Specification**")
            if "research_spec_continuous" in results_dict:
                show_disclose_income_overview(
                    results_dict["research_spec_continuous"], 
                    " (Research Spec, Cont)",
                    result_key="research_spec_continuous",
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Research Specification continuous results not available")
        
        with col6:
            st.markdown("**⚖️ Research Baseline**")
            if "research_baseline_continuous" in results_dict:
                show_disclose_income_overview(
                    results_dict["research_baseline_continuous"], 
                    " (Research Baseline, Cont)",
                    result_key="research_baseline_continuous",
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption("Research Baseline continuous results not available")
    else:
        # Single income mode, compare all three population modes
        income_type = "continuous" if st.session_state.income_spec_mode == "continuous only" else "categorical"
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🧬 Copula (Synthetic)")
            copula_key = f"copula_{income_type}"
            if copula_key in results_dict:
                show_disclose_income_overview(
                    results_dict[copula_key], 
                    f" (Copula, {income_type.title()})",
                    result_key=copula_key,
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption(f"Copula {income_type} results not available")
        
        with col2:
            st.markdown("#### 📄 Research Specification")
            research_spec_key = f"research_spec_{income_type}"
            if research_spec_key in results_dict:
                show_disclose_income_overview(
                    results_dict[research_spec_key], 
                    f" (Research Spec, {income_type.title()})",
                    result_key=research_spec_key,
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption(f"Research Specification {income_type} results not available")
        
        with col3:
            st.markdown("#### ⚖️ Research Baseline")
            baseline_key = f"research_baseline_{income_type}"
            if baseline_key in results_dict:
                show_disclose_income_overview(
                    results_dict[baseline_key], 
                    f" (Research Baseline, {income_type.title()})",
                    result_key=baseline_key,
                    enable_selection=should_enable_selection()
                )
            else:
                st.caption(f"Research Baseline {income_type} results not available")
    
    # Excel export is handled in the main Export section (export_section.py)
    # to avoid duplicate download buttons on the page


def render_disclose_income_comparison(results_dict):
    """Render income specification comparison results for disclose_income"""
    from app.components import show_disclose_income_overview
    
    st.markdown("### 📊 Disclose Income - Income Specification Comparison")
    
    col_cat, col_cont = st.columns(2, gap="large")
    
    with col_cat:
        st.markdown("#### 📋 Categorical Income")
        if "categorical" in results_dict:
            show_disclose_income_overview(
                results_dict["categorical"], 
                " (Categorical)",
                result_key="categorical",
                enable_selection=should_enable_selection()
            )
        else:
            st.caption("Categorical results not available")
    
    with col_cont:
        st.markdown("#### 📈 Continuous Income") 
        if "continuous" in results_dict:
            show_disclose_income_overview(
                results_dict["continuous"], 
                " (Continuous)",
                result_key="continuous",
                enable_selection=should_enable_selection()
            )
        else:
            st.caption("Continuous results not available")
    
    # Excel export is handled in the main Export section (export_section.py)
    # to avoid duplicate download buttons on the page
