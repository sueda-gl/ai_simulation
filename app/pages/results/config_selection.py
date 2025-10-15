# app/pages/results/config_selection.py
"""
Configuration selection UI for donation decision results.
"""
import streamlit as st
import pandas as pd
from app.pages.decision_execution import (
    save_selected_configuration,
    format_result_name,
    is_configuration_selected,
    clear_selected_configuration
)


def render_configuration_selection_ui(results_dict):
    """Render configuration selection UI for donation decision results"""
    
    # Only show if we have donation_default results and this is from individual decision run
    if not results_dict:
        return
        
    # Check if any results have donation_default column
    has_donation_results = any(
        'donation_default' in df.columns 
        for df in results_dict.values() 
        if isinstance(df, pd.DataFrame) and not df.empty
    )
    
    if not has_donation_results:
        return
    
    # Check if this is from an individual donation decision run
    # This should only show for individual donation runs, not combined simulations
    is_individual_donation_run = (
        hasattr(st.session_state, 'custom_decisions') and 
        st.session_state.custom_decisions == ['donation_default'] and
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) == 0  # Individual runs have empty default_decisions
    )
    
    if not is_individual_donation_run:
        return
    
    # Show configuration selection interface
    st.markdown("---")
    st.markdown('<h3 class="section-header">🎯 Select Configuration for Combined Simulations</h3>', unsafe_allow_html=True)
    st.caption(f"Choose your preferred configuration from {len(results_dict)} available result(s) to use in complete simulations")
    
    # Show current selection status if any
    if hasattr(st.session_state, 'selected_donation_config'):
        config = st.session_state.selected_donation_config
        with st.container():
            st.success(f"✅ **Selected Configuration**: {format_result_name(config['result_key'])}")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"Selected at {config['selected_timestamp'].strftime('%H:%M:%S')} - Avg Donation: {config['metrics']['mean_donation']:.2%}")
            with col2:
                if st.button("🗑️ Clear Selection", help="Clear the selected configuration"):
                    clear_selected_configuration()
                    st.rerun()
    
    # Configuration selection cards
    cols = st.columns(min(len(results_dict), 3))  # Max 3 columns for better layout
    
    for idx, (result_key, result_df) in enumerate(results_dict.items()):
        col_idx = idx % 3
        
        with cols[col_idx]:
            render_configuration_card(result_key, result_df)


def render_configuration_card(result_key, result_df):
    """Render a single configuration selection card"""
    
    if result_df.empty or 'donation_default' not in result_df.columns:
        return
    
    # Check if this configuration is currently selected
    is_selected = is_configuration_selected(result_key)
    
    # Calculate key metrics - always use truncated
    donation_col = 'donation_default'
    
    mean_donation = result_df[donation_col].mean()
    std_donation = result_df[donation_col].std()
    median_donation = result_df[donation_col].median()
    
    # Create card with conditional styling
    card_class = "selected-config-card" if is_selected else "config-card"
    
    with st.container():
        # Card header with selection indicator
        if is_selected:
            st.success(f"✅ **{format_result_name(result_key)}**")
        else:
            st.info(f"📊 **{format_result_name(result_key)}**")
        
        # Key metrics
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric("Mean", f"{mean_donation:.2%}")
            st.metric("Std Dev", f"{std_donation:.2%}")
        
        with metric_col2:
            st.metric("Median", f"{median_donation:.2%}")
            st.metric("Agents", f"{len(result_df):,}")
        
        # Configuration details in smaller text
        config_details = extract_configuration_details_from_key(result_key)
        st.caption(f"Population: {config_details['population_short']}")
        st.caption(f"Income: {config_details['income_short']}")
        
        # Selection button
        button_type = "secondary" if is_selected else "primary"
        button_text = "✅ Selected" if is_selected else "🎯 Use This Config"
        button_disabled = is_selected
        
        if st.button(
            button_text, 
            type=button_type, 
            key=f"select_config_{result_key}",
            disabled=button_disabled,
            use_container_width=True,
            help="Select this configuration for use in combined simulations"
        ):
            save_selected_configuration(result_key, result_df)
            st.success(f"Selected: {format_result_name(result_key)}")
            st.rerun()


def extract_configuration_details_from_key(result_key):
    """Extract short display details from result key for UI"""
    
    # Population mode short names
    if 'copula' in result_key:
        population_short = "Copula"
    elif 'research_spec' in result_key or 'documentation' in result_key:
        population_short = "Research Spec"
    elif 'baseline' in result_key:
        population_short = "Baseline"
    else:
        population_short = "Single Mode"
    
    # Income mode short names
    if 'categorical' in result_key:
        income_short = "Categorical"
    elif 'continuous' in result_key:
        income_short = "Continuous"
    else:
        income_short = "Single Mode"
    
    return {
        'population_short': population_short,
        'income_short': income_short
    }
