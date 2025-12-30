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
    clear_selected_configuration,
    run_combined_simulation
)
from app.models import ALL_DECISIONS


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
    
    # Check if config is already selected
    has_selected_config = (
        'selected_donation_config' in st.session_state and 
        st.session_state.selected_donation_config is not None
    )
    
    # SINGLE CONFIG SCENARIO: Show save button when only one config exists and not yet saved
    if len(results_dict) == 1 and not has_selected_config:
        st.markdown("---")
        st.markdown('<h3 class="section-header">💾 Save Configuration</h3>', unsafe_allow_html=True)
        
        # Get the single result
        result_key = next(iter(results_dict.keys()))
        result_df = results_dict[result_key]
        
        if not result_df.empty and 'donation_default' in result_df.columns:
            # Calculate metrics for display
            donation_col = 'donation_default'
            mean_donation = result_df[donation_col].mean()
            std_donation = result_df[donation_col].std()
            median_donation = result_df[donation_col].median()
            
            # Get population and income mode for display
            population_mode = st.session_state.get('population_mode', 'Unknown')
            income_spec_mode = st.session_state.get('income_spec_mode', 'Unknown')
            
            with st.container():
                st.info(f"📊 **{population_mode}** + **{income_spec_mode}**")
                
                # Show key metrics
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Mean", f"{mean_donation:.2%}")
                with metric_cols[1]:
                    st.metric("Std Dev", f"{std_donation:.2%}")
                with metric_cols[2]:
                    st.metric("Median", f"{median_donation:.2%}")
                with metric_cols[3]:
                    st.metric("Agents", f"{len(result_df):,}")
                
                st.caption("💡 Save this configuration to use it in the complete simulation and link it to final_donation_rate")
                
                # Save button
                if st.button(
                    "💾 Save This Configuration",
                    type="primary",
                    use_container_width=True,
                    key="save_single_config",
                    help="Save this configuration for use in combined simulations"
                ):
                    save_selected_configuration(result_key, result_df)
                    st.success(f"✅ Configuration saved: {population_mode} + {income_spec_mode}")
                    st.rerun()
        return
    
    # MULTIPLE CONFIGS or ALREADY SAVED: Show selected config and "Run Complete Simulation"
    if has_selected_config:
        st.markdown("---")
        config = st.session_state.selected_donation_config
        with st.container():
            st.success(f"✅ **Selected Configuration**: {format_result_name(config['result_key'])}")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"Selected at {config['selected_timestamp'].strftime('%H:%M:%S')} - Avg Donation: {config['metrics']['mean_donation']:.2%}")
            with col2:
                if st.button("🗑️ Clear Selection", help="Clear the selected configuration", key="clear_selection_top"):
                    clear_selected_configuration()
                    st.rerun()
        
        # Add "Run Complete Simulation" button right here on Results page
        render_complete_simulation_section()


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


def render_complete_simulation_section():
    """Render the complete simulation section with Run button after configuration selection"""
    
    st.markdown("---")
    st.markdown('<h3 class="section-header">🚀 Run Complete Simulation</h3>', unsafe_allow_html=True)
    
    # Get selected decisions from session state
    selected_decisions = getattr(st.session_state.decision_params, 'selected_decisions', [])
    
    # Calculate unselected decisions
    unselected_decisions = [d for d in ALL_DECISIONS if d not in selected_decisions]
    
    # Get selected configuration details
    config = st.session_state.selected_donation_config
    
    # Simple button without verbose info
    if st.button(
        "🚀 Run Complete Simulation",
        type="primary",
        use_container_width=True,
        key="run_complete_from_results",
        help=f"Execute all {len(ALL_DECISIONS)} decisions with the selected configuration"
    ):
        # Execute the combined simulation
        with st.spinner("🔄 Running complete simulation..."):
            run_combined_simulation(selected_decisions)
            
            # After simulation completes, show success message
            if hasattr(st.session_state, 'simulation_results') and st.session_state.simulation_results:
                st.success("✅ Complete simulation finished!")
