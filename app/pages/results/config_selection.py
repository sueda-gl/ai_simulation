# app/pages/results/config_selection.py
"""
Configuration selection UI for decision results (donation_default and disclose_income).
"""
import streamlit as st
import pandas as pd
from app.pages.decision_execution import (
    save_selected_configuration,
    format_result_name,
    is_configuration_selected,
    clear_selected_configuration,
    run_combined_simulation,
    can_run_complete_simulation,
    get_selected_decision_configs,
    get_decision_config,
    clear_decision_config
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
    
    # Check if config is already selected (only count explicitly saved configs, not auto-implied)
    has_selected_config = False
    if 'selected_donation_config' in st.session_state and st.session_state.selected_donation_config is not None:
        config = st.session_state.selected_donation_config
        # Only count as "selected" if it was explicitly saved, not auto-implied
        has_selected_config = config.get('source') != 'auto_implied_single_config'
    
    # SINGLE CONFIG SCENARIO: Show save button when only one config exists and not yet explicitly saved
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
        # Use donation_income_mode (primary) with fallback to income_spec_mode (legacy)
        donation_income_mode = config.get('donation_income_mode', config.get('income_spec_mode', 'unknown'))
        with st.container():
            st.success(f"✅ **Selected Donation Configuration**: {format_result_name(config['result_key'])}")
            st.caption(f"Income mode for Donation Default: {donation_income_mode}")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"Selected at {config['selected_timestamp'].strftime('%H:%M:%S')} - Avg Donation: {config['metrics']['mean_donation']:.2%}")
            with col2:
                if st.button("🗑️ Clear Selection", help="Clear the selected configuration", key="clear_selection_top"):
                    clear_selected_configuration()
                    st.rerun()
        
        # Add "Run Complete Simulation" button right here on Results page
        # The render_complete_simulation_section will check can_run_complete_simulation() internally
        render_complete_simulation_section()


def render_disclose_income_config_selection_ui(results_dict):
    """Render configuration selection UI for disclose_income decision results"""
    
    if not results_dict:
        return
    
    # Check if any results have disclose_income column
    has_di_results = any(
        'disclose_income' in df.columns 
        for df in results_dict.values() 
        if isinstance(df, pd.DataFrame) and not df.empty
    )
    
    if not has_di_results:
        return
    
    # Check if this is from an individual disclose_income decision run
    is_individual_di_run = (
        hasattr(st.session_state, 'custom_decisions') and 
        st.session_state.custom_decisions == ['disclose_income'] and
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) == 0
    )
    
    if not is_individual_di_run:
        return
    
    # Check if config is already selected (from unified or legacy storage)
    # Only count explicitly saved configs, not auto-implied ones
    di_config = get_decision_config('disclose_income')
    has_selected_config = di_config is not None and di_config.get('source') != 'auto_implied_single_config'
    
    if not has_selected_config:
        legacy_config = st.session_state.get('selected_disclose_income_config')
        has_selected_config = (
            legacy_config is not None and 
            legacy_config.get('source') != 'auto_implied_single_config'
        )
    
    # If config is selected, show the selected config info and potentially the Run Complete Simulation button
    if has_selected_config:
        st.markdown("---")
        
        # Get config from unified storage first, then legacy
        config = get_decision_config('disclose_income')
        if config is None:
            config = st.session_state.selected_disclose_income_config
        
        income_mode = config.get('params', {}).get('income_mode', config.get('income_mode', 'Unknown'))
        
        with st.container():
            st.success(f"✅ **Selected Disclose Income Configuration**: {income_mode}")
            
            # Show metrics if available
            metrics = config.get('metrics', {})
            if metrics:
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    y_rate = metrics.get('y_rate', 0)
                    st.caption(f"Y Rate: {y_rate:.2%}")
                with col2:
                    timestamp = config.get('selected_timestamp')
                    if timestamp:
                        st.caption(f"Selected at {timestamp.strftime('%H:%M:%S')}")
                with col3:
                    if st.button("🗑️ Clear", help="Clear the selected configuration", key="clear_di_selection"):
                        clear_decision_config('disclose_income')
                        st.rerun()
        
        # CRITICAL: Check if complete simulation can actually run before showing the section
        can_run, reason, config_count, block_type, *_ = can_run_complete_simulation()
        
        # Only show Run Complete Simulation section if simulation is viable OR if it's blocked 
        # (so we can show the user why it's blocked and what to do)
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
    
    # CRITICAL: Check if complete simulation can run FIRST, before showing any config info
    result = can_run_complete_simulation()
    can_run, reason, config_count, block_type = result[:4]
    blocking_issues = result[4] if len(result) > 4 else []
    
    # Get selected decisions from session state
    selected_decisions = getattr(st.session_state.decision_params, 'selected_decisions', [])
    
    # Calculate unselected decisions
    unselected_decisions = [d for d in ALL_DECISIONS if d not in selected_decisions]
    
    if not can_run:
        # FIX: Show ALL blocking issues, not just the first one
        if blocking_issues and len(blocking_issues) > 1:
            # Multiple issues - show them all together
            st.error(f"⚠️ **{len(blocking_issues)} Configuration Issues Detected**")
            for i, issue in enumerate(blocking_issues, 1):
                if issue['block_type'] == "disclose_income":
                    st.warning(f"""
**Issue {i}: Disclose Income**

{issue['reason']}

**Action Required:**
1. Run **Disclose Income Only** from the Disclose Income tab
2. Click **"Use This Config"** on the result you want to use
                    """)
                else:
                    # donation_config block type
                    st.warning(f"""
**Issue {i}: Donation Default**

{issue['reason']}

**Action Required:**
1. Run **Donation Default Only** from the Donation Default tab
2. Click **"Use This Config"** on the result you want to use
                    """)
        else:
            # Single issue - show original format
            if block_type == "disclose_income":
                st.warning(f"""
⚠️ **Disclose Income Configuration Required**

{reason}

**Action Required:**
1. Run **Disclose Income Only** from the Disclose Income tab
2. Click **"Use This Config"** on the result you want to use
3. Return here to run complete simulation
                """)
            elif block_type == "donation_config":
                st.warning(f"""
⚠️ **Donation Default Configuration Required**

{reason}

**Action Required:**
1. Run **Donation Default Only** from the Donation Default tab
2. Click **"Use This Config"** on the result you want to use
3. Return here to run complete simulation
                """)
            else:
                st.warning(f"""
⚠️ **Configuration Issue**

{reason}
                """)
        
        # Disabled button
        help_text = f"{len(blocking_issues)} configuration issue(s) detected" if len(blocking_issues) > 1 else ("Select a Disclose Income config first" if block_type == "disclose_income" else ("Select a Donation Default config first" if block_type == "donation_config" else "Configuration issue detected"))
            
        st.button(
            "🚀 Run Complete Simulation",
            type="primary",
            use_container_width=True,
            disabled=True,
            key="run_complete_from_results_disabled",
            help=help_text
        )
    else:
        # Simulation CAN run - now show saved configs that will be used
        # Only show explicitly saved configs (not auto-implied ones)
        saved_configs = get_selected_decision_configs()
        explicit_configs = {
            k: v for k, v in saved_configs.items() 
            if v.get('source') != 'auto_implied_single_config'
        }
        
        if explicit_configs:
            st.info(f"📋 **{len(explicit_configs)} saved configuration(s) will be used:**")
            for decision_name, config in explicit_configs.items():
                decision_title = decision_name.replace('_', ' ').title()
                if decision_name == 'donation_default':
                    income_mode = config.get('params', {}).get('income_mode', 
                        config.get('income_spec_mode', 'Unknown'))
                    mean_val = config.get('metrics', {}).get('mean_donation', 0)
                    st.caption(f"  ✅ {decision_title}: {income_mode} (mean: {mean_val:.2%})")
                elif decision_name == 'disclose_income':
                    income_mode = config.get('params', {}).get('income_mode',
                        config.get('income_mode', 'Unknown'))
                    y_rate = config.get('metrics', {}).get('y_rate', 0)
                    st.caption(f"  ✅ {decision_title}: {income_mode} (Y rate: {y_rate:.2%})")
                else:
                    st.caption(f"  ✅ {decision_title}")
        else:
            # No explicit configs but simulation can run (single mode)
            st.info("📋 **Using current UI settings** (no explicit configurations saved)")
        
        # Enabled button
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
