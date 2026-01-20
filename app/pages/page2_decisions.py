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
from app.pages.decision_execution import run_combined_simulation, DEFAULT_DECISION_VALUES, can_run_complete_simulation, auto_populate_single_donation_config


def initialize_page2_widget_keys():
    """Initialize all Page 2 widget keys to preserve values across navigation.
    
    CRITICAL: This function ensures donation decision parameters persist when:
    - Navigating from Results back to Page 2
    - Switching between Overview and decision tabs
    - Running simulations
    
    Similar to Page 1's initialize_widget_keys(), but for donation-specific parameters.
    """
    
    # Initialize donation widget keys (checkboxes and sliders)
    # These are the widget keys that store UI state
    if "tab_sigma_in_copula" not in st.session_state:
        st.session_state.tab_sigma_in_copula = st.session_state.get('sigma_in_copula', False)
    
    if "tab_sigma_in_research" not in st.session_state:
        st.session_state.tab_sigma_in_research = st.session_state.get('sigma_in_research', True)
    
    if "tab_sigma_in_copula_compare" not in st.session_state:
        st.session_state.tab_sigma_in_copula_compare = st.session_state.get('sigma_in_copula', False)
    
    if "tab_sigma_in_research_compare" not in st.session_state:
        st.session_state.tab_sigma_in_research_compare = st.session_state.get('sigma_in_research', True)
    
    # Initialize slider widget keys for different population modes
    if "tab_sigma_coefficient" not in st.session_state:
        st.session_state.tab_sigma_coefficient = st.session_state.get('sigma_coefficient', 1.0)
    
    if "tab_sigma_coefficient_research" not in st.session_state:
        st.session_state.tab_sigma_coefficient_research = st.session_state.get('sigma_coefficient', 1.0)
    
    if "tab_sigma_coefficient_compare" not in st.session_state:
        st.session_state.tab_sigma_coefficient_compare = st.session_state.get('sigma_coefficient', 1.0)
    
    if "tab_anchor_weight" not in st.session_state:
        st.session_state.tab_anchor_weight = st.session_state.get('anchor_observed_weight', 0.75)
    
    # Initialize income spec mode widget key
    if "page2_tab_income_spec_mode" not in st.session_state:
        # Map current income_spec_mode to radio button options
        current_mode = st.session_state.get('income_spec_mode', 'categorical only')
        if current_mode in ["categorical only", "continuous only", "Compare both"]:
            st.session_state.page2_tab_income_spec_mode = current_mode
        elif current_mode in ["compare both", "compare side-by-side"]:
            st.session_state.page2_tab_income_spec_mode = "Compare both"
        else:
            st.session_state.page2_tab_income_spec_mode = "categorical only"
    
    # CRITICAL: ALWAYS sync non-prefixed variables from widget keys
    # These non-prefixed variables are what the simulation actually reads
    # We MUST sync them on every page load to preserve user's values after navigation
    
    population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
    
    # Sync sigma coefficient based on current population mode
    # ALWAYS sync if widget key exists, regardless of whether non-prefixed variable exists
    if population_mode == "Copula (synthetic)":
        # Use Copula widget key
        if 'tab_sigma_coefficient' in st.session_state:
            st.session_state.sigma_coefficient = st.session_state.tab_sigma_coefficient
            st.session_state.sigma_value_ui = 9.8995 * st.session_state.tab_sigma_coefficient
        if 'tab_sigma_in_copula' in st.session_state:
            st.session_state.sigma_in_copula = st.session_state.tab_sigma_in_copula
        else:
            # If widget key doesn't exist, ensure non-prefixed variable exists
            if 'sigma_coefficient' not in st.session_state:
                st.session_state.sigma_coefficient = 1.0
                st.session_state.sigma_value_ui = 9.8995
            if 'sigma_in_copula' not in st.session_state:
                st.session_state.sigma_in_copula = False
    elif population_mode == "Research Specification":
        # Use Research widget key
        if 'tab_sigma_coefficient_research' in st.session_state:
            st.session_state.sigma_coefficient = st.session_state.tab_sigma_coefficient_research
            st.session_state.sigma_value_ui = 9.8995 * st.session_state.tab_sigma_coefficient_research
        if 'tab_sigma_in_research' in st.session_state:
            st.session_state.sigma_in_research = st.session_state.tab_sigma_in_research
        else:
            # If widget key doesn't exist, ensure non-prefixed variable exists
            if 'sigma_coefficient' not in st.session_state:
                st.session_state.sigma_coefficient = 1.0
                st.session_state.sigma_value_ui = 9.8995
            if 'sigma_in_research' not in st.session_state:
                st.session_state.sigma_in_research = True
    elif population_mode == "Research Baseline":
        # Research Baseline has no stochastic component - ALWAYS set to 0
        st.session_state.sigma_coefficient = 0.0
        st.session_state.sigma_value_ui = 0.0
        st.session_state.sigma_in_copula = False
        st.session_state.sigma_in_research = False
    elif population_mode == "Compare all":
        # Use Compare widget key
        if 'tab_sigma_coefficient_compare' in st.session_state:
            st.session_state.sigma_coefficient = st.session_state.tab_sigma_coefficient_compare
            st.session_state.sigma_value_ui = 9.8995 * st.session_state.tab_sigma_coefficient_compare
        if 'tab_sigma_in_copula_compare' in st.session_state:
            st.session_state.sigma_in_copula = st.session_state.tab_sigma_in_copula_compare
        if 'tab_sigma_in_research_compare' in st.session_state:
            st.session_state.sigma_in_research = st.session_state.tab_sigma_in_research_compare
        else:
            # If widget key doesn't exist, ensure non-prefixed variable exists
            if 'sigma_coefficient' not in st.session_state:
                st.session_state.sigma_coefficient = 1.0
                st.session_state.sigma_value_ui = 9.8995
            if 'sigma_in_copula' not in st.session_state:
                st.session_state.sigma_in_copula = False
            if 'sigma_in_research' not in st.session_state:
                st.session_state.sigma_in_research = True
    
    # ALWAYS sync anchor weight (common across all modes)
    if 'tab_anchor_weight' in st.session_state:
        st.session_state.anchor_observed_weight = st.session_state.tab_anchor_weight
    elif 'anchor_observed_weight' not in st.session_state:
        st.session_state.anchor_observed_weight = 0.75
    
    # ALWAYS sync income spec mode
    if 'page2_tab_income_spec_mode' in st.session_state:
        st.session_state.income_spec_mode = st.session_state.page2_tab_income_spec_mode
    elif 'income_spec_mode' not in st.session_state:
        st.session_state.income_spec_mode = 'categorical only'
    
    # CRITICAL: Initialize default decision parameter keys
    # These are the keys used by default decision widgets in the Overview tab
    # They MUST be initialized even when the Overview tab hasn't rendered yet
    # This prevents loss of default decision configurations when navigating
    
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    
    for decision_name, default_value in DEFAULT_DECISION_VALUES.items():
        if isinstance(default_value, dict):
            decision_type = default_value.get("type")
            
            # Initialize random probability decision keys
            if decision_type == "random_probability":
                prob_key = f"{decision_name}_default_probability_y"
                if prob_key not in st.session_state:
                    st.session_state[prob_key] = default_value.get("probability_y", 0.5)
            
            # Initialize checkbox selection decision keys
            elif decision_type == "checkbox_selection":
                selection_key = f"{decision_name}_default_params"
                if selection_key not in st.session_state:
                    st.session_state[selection_key] = default_value.get("default_selection", []).copy()
                
                # Initialize individual checkbox keys
                parameters = default_value.get("parameters", {})
                default_selection = default_value.get("default_selection", [])
                for param_key in parameters.keys():
                    checkbox_key = f"{decision_name}_default_param_{param_key}"
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = param_key in default_selection
            
            # Initialize radio selection decision keys
            elif decision_type == "radio_selection":
                selection_key = f"{decision_name}_default_selection"
                if selection_key not in st.session_state:
                    st.session_state[selection_key] = default_value.get("default_option", "")
            
            # Initialize prioritized selection decision keys
            elif decision_type == "prioritized_selection":
                template_key = f"{decision_name}_priority_template"
                if template_key not in st.session_state:
                    st.session_state[template_key] = default_value.get("priority_template", []).copy()
        
        # Initialize numeric default decision keys
        elif isinstance(default_value, (int, float)):
            value_key = f"{decision_name}_default_value"
            if value_key not in st.session_state:
                st.session_state[value_key] = default_value
    
    # Auto-populate selected_donation_config when only one configuration exists
    # This ensures the Overview tab shows the current config without requiring explicit selection
    # NOTE: The sync of final_donation_rate happens INSIDE this function if it populates/updates
    auto_populate_single_donation_config()


def format_decision_title(decision_name, include_number=False):
    """Format decision name for display, with special handling for specific decisions
    
    Args:
        decision_name: The decision name to format
        include_number: If True, prepend decision number (e.g., "1. Decision Name")
    
    Returns:
        Formatted decision title string
    """
    # Get decision number from ALL_DECISIONS
    decision_number = None
    if include_number and decision_name in ALL_DECISIONS:
        decision_number = ALL_DECISIONS.index(decision_name) + 1
    
    # Format the title
    if decision_name == "purchase_vs_bid":
        title = "Purchase Now Vs Bid"
    elif decision_name == "purchasing_quantity":
        title = "Purchase Request Quantity"
    elif decision_name == "purchasing_frequency":
        title = "Purchase Request Frequency"
    else:
        title = decision_name.replace('_', ' ').title()
    
    # Add number prefix if requested
    if decision_number is not None:
        return f"{decision_number}. {title}"
    return title


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
    can_run, reason, config_count, block_type = can_run_complete_simulation()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if not can_run:
            # Show warning based on block type
            if block_type == "disclose_income":
                st.warning(f"""
⚠️ **Disclose Income Configuration Required**

{reason}

**Action Required:**
1. Go to the **Disclose Income** tab
2. Change "Income Specification for Disclosure Model" from "Compare both" to either **"Categorical only"** or **"Continuous only"**
3. Return here to run complete simulation
                """)
            else:
                # donation_config block type
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
            donation_income_mode = config.get('donation_income_mode', config.get('income_spec_mode', 'unknown'))
            st.success(f"✅ **Using selected donation configuration**: {config['population_mode']} + {donation_income_mode}")
            
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
            # Disabled button with appropriate help text
            help_text = "Change Disclose Income to single mode first" if block_type == "disclose_income" else "Select a donation configuration first"
            st.button(
                "🚀 Run Complete Simulation", 
                type="primary", 
                use_container_width=True, 
                disabled=True,
                key="run_complete_simulation_disabled",
                help=help_text
            )
        else:
            # Enabled button
            if st.button("🚀 Run Complete Simulation", type="primary", use_container_width=True, key="run_complete_simulation"):
                run_combined_simulation(selected_decisions)
    


def render_page2():
    """Render Page 2: Decision-Specific Parameters"""
    st.markdown('<h2 class="page-header">Page 2: Decision-Specific Parameters</h2>', unsafe_allow_html=True)
    
    # Initialize widget keys to preserve values across navigation
    # CRITICAL: This ensures donation parameters persist when navigating from Results back to Page 2
    initialize_page2_widget_keys()
    
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
            format_func=lambda d: format_decision_title(d, include_number=True),
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
            format_func=lambda d: format_decision_title(d, include_number=True),
            key="page2_manual_multiselect",
            help="Select one or more decisions to run",
            placeholder="Choose decisions..."
        )
        # Update manual selections for persistence
        st.session_state.page2_manual_selections = selected_decisions
    
    # Store the final selected decisions (sorted in chronological order)
    # Ensure decisions are always displayed in chronological execution order
    selected_set = set(selected_decisions)
    selected_decisions_ordered = [d for d in ALL_DECISIONS if d in selected_set]
    st.session_state.decision_params.selected_decisions = selected_decisions_ordered
    
    # Show informational message based on selection state
    if not selected_decisions_ordered:
        st.info("ℹ️ No decisions selected for custom configuration. You can run the complete simulation using default values for all decisions, or select specific decisions to customize their parameters.")
    
    # Create tabs - Overview tab is always present
    if selected_decisions_ordered:
        # Overview + decision-specific tabs (in chronological order with numbers)
        tab_names = ["📊 Overview"] + [f"🎯 {format_decision_title(d, include_number=True)}" for d in selected_decisions_ordered]
        tabs = st.tabs(tab_names)
        
        # Overview Tab
        with tabs[0]:
            render_overview_tab(selected_decisions_ordered)
        
        # Decision-specific tabs
        for i, decision in enumerate(selected_decisions_ordered):
            with tabs[i + 1]:
                render_decision_tab(decision)
    else:
        # Only Overview tab when no decisions are selected
        tabs = st.tabs(["📊 Overview"])
        
        # Overview Tab
        with tabs[0]:
            render_overview_tab(selected_decisions_ordered)
    
    # Navigation
    render_navigation('page2')


def render_selected_donation_config_display():
    """Display the selected donation configuration in the overview tab"""
    
    # Common header for the section
    st.markdown('<h3 class="section-header">🎯 Selected Decision Parameters</h3>', unsafe_allow_html=True)
    
    # Check if config exists (use 'in' operator for Streamlit session state)
    has_config = 'selected_donation_config' in st.session_state and st.session_state.selected_donation_config is not None
    
    if not has_config:
        # No configuration - try auto-populate one more time
        auto_populate_single_donation_config()
        # Re-check after auto-populate attempt
        has_config = 'selected_donation_config' in st.session_state and st.session_state.selected_donation_config is not None
    
    if not has_config:
        # Still no config - show the "not selected" message
        st.markdown("#### 3. Donation Default")
        st.info("💡 **No donation configuration selected yet**")
        st.caption("Run the donation decision individually first, then select your preferred configuration from the results.")
        return
    
    config = st.session_state.selected_donation_config
    is_auto_implied = config.get('source') == 'auto_implied_single_config'
    
    st.markdown("#### 3. Donation Default")
    
    # Main configuration display
    # Use donation_income_mode (primary) with fallback to income_spec_mode (legacy)
    donation_income_mode = config.get('donation_income_mode', config.get('income_spec_mode', 'unknown'))
    
    with st.container():
        st.success(f"✅ **Donation Default Configuration**: {config['population_mode']} + {donation_income_mode}")
        
        # Metrics and details in columns
        if is_auto_implied:
            # For auto-implied configs, show only mode info (no simulation metrics yet)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Population Mode", config['population_mode'])
            
            with col2:
                st.metric("Income Mode", donation_income_mode)
            
            with col3:
                st.metric("Agents", f"{config['total_agents']:,}")
        else:
            # For explicitly selected configs, show full metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Population Mode", config['population_mode'])
            
            with col2:
                st.metric("Income Mode", donation_income_mode)
            
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
                # Income effects - use donation_income_mode defined above
                if 'categorical' in donation_income_mode.lower():
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
            st.markdown("**ℹ️ Configuration Info:**")
            if is_auto_implied:
                st.caption("Source: Auto-detected (single configuration)")
                st.caption("💡 This configuration was automatically detected from your Page 1 and Donation Default tab settings")
            else:
                st.caption(f"Selected at: {config['selected_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.caption(f"Source: {config['source']}")
        
        # Action buttons
        action_col1, action_col2 = st.columns([3, 1])
        
        with action_col1:
            if is_auto_implied:
                st.caption("This single configuration will be used for the donation decision in complete simulations")
            else:
                st.caption("This selected configuration will be used for the donation decision in complete simulations")
        
        with action_col2:
            # Only show Clear button for explicitly selected configs (not auto-implied)
            if not is_auto_implied:
                if st.button("🗑️ Clear", help="Clear the selected configuration", key="clear_donation_config"):
                    from app.pages.decision_execution import clear_selected_configuration
                    clear_selected_configuration()
                    st.rerun()
