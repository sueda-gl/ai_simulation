# app/pages/page2_decisions.py
"""
Page 2: Decision-Specific Parameters for the Enhanced AI Agent Simulation.
"""
import streamlit as st
from app.models import ALL_DECISIONS
from app.pages.navigation import render_navigation
from app.pages.decision_tabs import render_decision_tab
from app.pages.decision_tabs.global_parameters import render_global_parameters_readonly
from app.pages.decision_tabs.default_config import render_default_decisions_config
from app.pages.decision_execution import run_combined_simulation, DEFAULT_DECISION_VALUES, can_run_complete_simulation


def format_decision_title(decision_name):
    """Format decision name for display, with special handling for specific decisions"""
    if decision_name == "purchase_vs_bid":
        return "Purchase Now Vs Bid"
    return decision_name.replace('_', ' ').title()


def render_overview_tab(selected_decisions):
    """Render the overview tab with combined execution option"""
    
    # Display Global Parameters
    render_global_parameters_readonly()
    
    # Show selected donation configuration if available
    if 'donation_default' in selected_decisions:
        render_selected_donation_config_display()
    
    # NEW: Show default decisions configuration for unselected decisions
    st.markdown("---")
    render_default_decisions_config(selected_decisions, ALL_DECISIONS)
    
    # Add combined run button
    st.markdown("---")
    st.markdown('<h3 class="section-header">🚀 Complete Simulation</h3>', unsafe_allow_html=True)
    
    # Calculate unselected decisions (ALL_DECISIONS is already imported at module level)
    unselected_decisions = [d for d in ALL_DECISIONS if d not in selected_decisions]
    
    # Check if complete simulation can run (validation for multiple configurations)
    can_run, reason, config_count = can_run_complete_simulation()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if not can_run:
            # Show warning about multiple configurations
            st.warning(f"""
⚠️ **Multiple Donation Configurations Detected**

{reason}

**Action Required:**
1. Go to the **Donation Default** tab
2. Run **donation_default only**
3. Select your preferred configuration from results
4. Return here to run complete simulation
            """)
        elif config_count > 1 and hasattr(st.session_state, 'selected_donation_config'):
            # Show selected configuration info
            config = st.session_state.selected_donation_config
            st.success(f"✅ **Using selected configuration**: {config['population_mode']} + {config['income_spec_mode']}")
            
            if len(selected_decisions) == 0:
                st.info(f"🎯 All {len(ALL_DECISIONS)} decisions will use default values")
            elif len(unselected_decisions) == 0:
                st.info(f"🎯 All {len(selected_decisions)} decisions with custom parameters")
            else:
                st.info(f"🎯 {len(selected_decisions)} custom + {len(unselected_decisions)} default decisions")
        else:
            # Single configuration mode - show normal info
            if len(selected_decisions) == 0:
                # All decisions use defaults
                st.info(f"🎯 **Complete end-to-end simulation**: All {len(ALL_DECISIONS)} decisions will use default values")
                st.caption(f"💡 All decisions will use configured default values shown above")
            elif len(unselected_decisions) == 0:
                # All decisions use custom parameters
                st.info(f"🎯 **Complete end-to-end simulation**: All {len(selected_decisions)} decisions with custom parameters")
                st.caption(f"✅ All decisions use custom parameters")
            else:
                # Mixed: some custom, some defaults
                st.info(f"🎯 **Complete end-to-end simulation**: {len(selected_decisions)} decisions with custom parameters + {len(unselected_decisions)} decisions with default values")
                st.caption(f"💡 Unselected decisions will use configured default values shown above")
    
    with col2:
        if not can_run:
            # Disabled button
            st.button(
                "🚀 Run Complete Simulation", 
                type="primary", 
                use_container_width=True, 
                disabled=True,
                key="run_complete_simulation_disabled",
                help="Select a donation configuration first"
            )
        else:
            # Enabled button
            if st.button("🚀 Run Complete Simulation", type="primary", use_container_width=True, key="run_complete_simulation"):
                run_combined_simulation(selected_decisions)
    


def render_page2():
    """Render Page 2: Decision-Specific Parameters"""
    st.markdown('<h2 class="page-header">Page 2: Decision-Specific Parameters</h2>', unsafe_allow_html=True)
    
    # Decision selection
    st.markdown('<h3 class="section-header">🎯 Decision Selection</h3>', unsafe_allow_html=True)
    
    # Initialize states properly
    if "page2_manual_selections" not in st.session_state:
        st.session_state.page2_manual_selections = []
    
    if "page2_select_all_state" not in st.session_state:
        st.session_state.page2_select_all_state = False
    
    # Multi-select with "Select All" functionality
    select_all = st.checkbox(
        "Select All Decisions", 
        value=st.session_state.page2_select_all_state,
        key="page2_select_all_checkbox",
        on_change=lambda: setattr(st.session_state, 'page2_select_all_state', st.session_state.page2_select_all_checkbox)
    )
    
    if select_all:
        # Show disabled multiselect with all decisions
        selected_decisions = st.multiselect(
            "Selected Decisions",
            ALL_DECISIONS,
            default=ALL_DECISIONS,
            help="All decisions are selected",
            disabled=True
        )
        # Set all decisions as selected
        selected_decisions = ALL_DECISIONS
    else:
        # Manual selection mode
        selected_decisions = st.multiselect(
            "Select Decisions to Run",
            ALL_DECISIONS,
            default=st.session_state.page2_manual_selections,
            key="page2_manual_multiselect",
            help="Select one or more decisions to run",
            placeholder="Choose decisions..."
        )
        # Update manual selections for persistence
        st.session_state.page2_manual_selections = selected_decisions
    
    # Store the final selected decisions
    st.session_state.decision_params.selected_decisions = selected_decisions
    
    # Show informational message based on selection state
    if not selected_decisions:
        st.info("ℹ️ No decisions selected for custom configuration. You can run the complete simulation using default values for all decisions, or select specific decisions to customize their parameters.")
    
    # Create tabs - Overview tab is always present
    if selected_decisions:
        # Overview + decision-specific tabs
        tab_names = ["📊 Overview"] + [f"🎯 {format_decision_title(d)}" for d in selected_decisions]
        tabs = st.tabs(tab_names)
        
        # Overview Tab
        with tabs[0]:
            render_overview_tab(selected_decisions)
        
        # Decision-specific tabs
        for i, decision in enumerate(selected_decisions):
            with tabs[i + 1]:
                render_decision_tab(decision)
    else:
        # Only Overview tab when no decisions are selected
        tabs = st.tabs(["📊 Overview"])
        
        # Overview Tab
        with tabs[0]:
            render_overview_tab(selected_decisions)
    
    # Navigation
    render_navigation('page2')


def render_selected_donation_config_display():
    """Display the selected donation configuration in the overview tab"""
    
    if not hasattr(st.session_state, 'selected_donation_config'):
        # No configuration selected yet
        st.markdown('<h3 class="section-header">🎯 Donation Configuration</h3>', unsafe_allow_html=True)
        st.info("💡 **No donation configuration selected yet**")
        st.caption("Run the donation decision individually first, then select your preferred configuration from the results.")
        return
    
    config = st.session_state.selected_donation_config
    
    st.markdown('<h3 class="section-header">🎯 Selected Donation Configuration</h3>', unsafe_allow_html=True)
    
    # Main configuration display
    with st.container():
        st.success(f"✅ **Configuration Selected**: {config['population_mode']} + {config['income_spec_mode']}")
        
        # Metrics and details in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Population Mode", config['population_mode'])
        
        with col2:
            st.metric("Income Mode", config['income_spec_mode'])
        
        with col3:
            st.metric("Avg Donation Rate", f"{config['metrics']['mean_donation']:.2%}")
        
        with col4:
            st.metric("Total Agents", f"{config['total_agents']:,}")
        
        # Additional details in expandable section
        with st.expander("📊 Configuration Details", expanded=False):
            
            # Coefficient summary
            st.markdown("**🔢 Regression Coefficients:**")
            coeffs = config['coefficients']
            
            coeff_col1, coeff_col2 = st.columns(2)
            
            with coeff_col1:
                st.metric("Intercept", f"{coeffs['intercept']:.6f}")
                st.metric("Honesty-Humility", f"{coeffs['beta_hh']:.6f}")
                
                # Group effects
                st.markdown("**👥 Group Effects:**")
                for group, coeff in coeffs['beta_group'].items():
                    st.caption(f"{group}: {coeff:.6f}")
            
            with coeff_col2:
                # Income effects
                if config['income_spec_mode'] == 'categorical only':
                    st.markdown("**💰 Income Quintile Effects:**")
                    for quintile, coeff in coeffs['beta_income_q'].items():
                        st.caption(f"{quintile}: {coeff:.6f}")
                else:
                    st.metric("Linear Income Coeff", f"{coeffs['beta_income_linear']:.6f}")
                
                # Study effects  
                st.markdown("**🎓 Study Programme Effects:**")
                for study, coeff in coeffs['beta_study'].items():
                    st.caption(f"{study}: {coeff:.6f}")
            
            # Stochastic parameters
            st.markdown("**🎲 Stochastic Parameters:**")
            stoch = config['stochastic_params']
            
            stoch_col1, stoch_col2 = st.columns(2)
            
            with stoch_col1:
                st.metric("Sigma Value", f"{stoch['stochastic']['sigma_value']:.4f}")
                st.metric("Sigma Coefficient", f"{stoch['stochastic']['sigma_coefficient']:.2f}")
            
            with stoch_col2:
                st.metric("Observed Weight", f"{stoch['anchor_weights']['observed']:.2f}")
                st.metric("Predicted Weight", f"{stoch['anchor_weights']['predicted']:.2f}")
            
            # Selection metadata
            st.markdown("**ℹ️ Selection Info:**")
            st.caption(f"Selected at: {config['selected_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.caption(f"Source: {config['source']}")
        
        # Action buttons
        action_col1, action_col2 = st.columns([3, 1])
        
        with action_col1:
            st.caption("This configuration will be used for the donation decision in complete simulations")
        
        with action_col2:
            if st.button("🗑️ Clear", help="Clear the selected configuration", key="clear_donation_config"):
                from app.pages.decision_execution import clear_selected_configuration
                clear_selected_configuration()
                st.rerun()
