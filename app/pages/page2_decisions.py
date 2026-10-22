# app/pages/page2_decisions.py
"""
Page 2: Decision-Specific Parameters for the Enhanced AI Agent Simulation.
"""
import streamlit as st
from app.state.widgets import stateful
from app.seam.config_repo import get_config_repo
from app.models import ALL_DECISIONS, DONATION_SIGMA_OVERALL
from app.pages.navigation import render_navigation
from app.pages.decision_tabs import render_decision_tab
from app.pages.decision_tabs.global_parameters import render_global_parameters_readonly
from app.pages.decision_tabs.default_config import render_default_decisions_config
from app.pages.decision_execution import run_combined_simulation, can_run_complete_simulation, format_decision_title, render_blocking_issues, current_donation_configuration
from app.state.saved_configs import clear_decision_config, get_decision_config
from app.reports.rtd import rtd_summary_metric_keys


def initialize_page2_widget_keys():
    """Mirror the canonical Donation / Disclose Income keys from their widget keys
    at the top of every Page-2 run.

    The decision tabs mirror these keys on every render, but the Overview tab is
    drawn BEFORE them, and in the run that processes a change to one of these
    widgets it reads the canonical keys (its donation card: the sigma coefficient,
    the tick boxes, the anchor weight; the run gate: the income mode), so without
    this mirror it would show the previous value for one rerun.

    The widget keys themselves are NOT seeded here: each tab seeds its own keys
    right before its widgets (``decision_tabs._tab_state.restore`` /
    ``seed_if_absent``), in the run that draws them. A widget key written in a run
    that does not draw the widget stays in Streamlit's old session state under the
    user key, is never refreshed once the widget exists (later writes go under the
    widget id), and resurfaces with that stale value once Streamlit culls the
    widget - in front of the value the user typed (verified 2026-10-21/22).
    """

    # Sync donation sigma coefficient from unified widget key
    if 'tab_sigma_coefficient_stochastic' in st.session_state:
        st.session_state.sigma_coefficient = st.session_state.tab_sigma_coefficient_stochastic
        st.session_state.sigma_value_ui = DONATION_SIGMA_OVERALL * st.session_state.tab_sigma_coefficient_stochastic
    elif 'sigma_coefficient' not in st.session_state:
        st.session_state.sigma_coefficient = 1.0
        st.session_state.sigma_value_ui = DONATION_SIGMA_OVERALL

    # Sync donation sigma checkboxes from canonical widget keys
    if 'tab_sigma_in_copula' in st.session_state:
        st.session_state.sigma_in_copula = st.session_state.tab_sigma_in_copula
    elif 'sigma_in_copula' not in st.session_state:
        st.session_state.sigma_in_copula = False

    if 'tab_sigma_in_research' in st.session_state:
        st.session_state.sigma_in_research = st.session_state.tab_sigma_in_research
    elif 'sigma_in_research' not in st.session_state:
        st.session_state.sigma_in_research = True

    # Sync disclose_income copula stochastic from canonical widget key
    if 'di_tab_sigma_in_copula' in st.session_state:
        st.session_state.di_sigma_in_copula = st.session_state.di_tab_sigma_in_copula
    elif 'di_sigma_in_copula' not in st.session_state:
        st.session_state.di_sigma_in_copula = False
    
    # ALWAYS sync anchor weight (common across all modes)
    if 'tab_anchor_weight' in st.session_state:
        st.session_state.anchor_observed_weight = st.session_state.tab_anchor_weight
    elif 'anchor_observed_weight' not in st.session_state:
        st.session_state.anchor_observed_weight = float(
            get_config_repo().decision('donation_default')['anchor_weights']['observed'])
    
    # ALWAYS sync income spec mode
    if 'page2_tab_income_spec_mode' in st.session_state:
        st.session_state.income_spec_mode = st.session_state.page2_tab_income_spec_mode
    elif 'income_spec_mode' not in st.session_state:
        st.session_state.income_spec_mode = 'categorical only'
    
    # The default-decision parameter keys the Overview tab's widgets use
    # ({decision_name}_default_probability_y, _default_params, _default_param_*,
    # _default_selection, _priority_template, _default_value) are seeded by
    # models.initialize_default_decision_parameters, which
    # models.initialize_session_state runs at the top of every script run.


def render_overview_tab(selected_decisions):
    """Render the overview tab with combined execution option"""
    
    # Display Global Parameters
    render_global_parameters_readonly()
    
    # Saved configs show whether or not their decision is selected for customization,
    # so they stay visible after navigation; one section header for all of them
    st.markdown('<h3 class="section-header">🎯 Saved Decision Configurations</h3>', unsafe_allow_html=True)
    
    # Render the config displays - they handle the "no config" case internally
    render_selected_donation_config_display()
    render_selected_disclose_income_config_display()
    render_selected_disclose_documents_config_display()
    render_selected_rejected_transaction_config_display()

    # Default-decision configuration for the unselected decisions
    st.markdown("---")
    render_default_decisions_config(selected_decisions, ALL_DECISIONS)
    
    # Add combined run button
    st.markdown("---")
    st.markdown('<h3 class="section-header">🚀 Complete Simulation</h3>', unsafe_allow_html=True)
    
    # Calculate unselected decisions (ALL_DECISIONS is already imported at module level)
    unselected_decisions = [d for d in ALL_DECISIONS if d not in selected_decisions]
    
    # Check if complete simulation can run (validation for multiple configurations)
    result = can_run_complete_simulation()
    can_run, reason, config_count, block_type = result[:4]
    blocking_issues = result[4] if len(result) > 4 else []
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if not can_run:
            # Show ALL blocking issues at once, not just the first one
            render_blocking_issues("overview", reason, block_type, blocking_issues)
        elif config_count > 1:
            dd_config = get_decision_config('donation_default')
            if dd_config:
                donation_income_mode = dd_config.get('donation_income_mode', dd_config.get('income_spec_mode', 'unknown'))
                st.success(f"✅ **Using selected donation configuration**: {dd_config['population_mode']} + {donation_income_mode}")
            
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
            help_text = {
                "disclose_income": "Change Disclose Income to single mode first",
                "rejected_transaction_defaults": "Select a Rejected Transaction Defaults configuration first",
                "disclose_documents": "Select a Disclose Documents configuration first",
            }.get(block_type, "Select a donation configuration first")
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
        # Manual selection mode.
        # CRITICAL: do NOT pass a changing `default=` together with a stable `key=` -
        # when the default changes between reruns, Streamlit treats the multiselect as
        # a NEW widget and resets it, silently dropping the user's in-flight click
        # (symptom: every second decision selected gets automatically unselected).
        # Instead, seed the widget's keyed state once from the persisted mirror.
        if "page2_manual_multiselect" not in st.session_state:
            st.session_state.page2_manual_multiselect = [
                d for d in st.session_state.page2_manual_selections if d in ALL_DECISIONS
            ]
        selected_decisions = stateful(st.multiselect,
            "Select Decisions to Run",
            ALL_DECISIONS,
            format_func=lambda d: format_decision_title(d, include_number=True),
            key="page2_manual_multiselect",
            help="Select one or more decisions to run",
            placeholder="Choose decisions..."
        )
        # Update manual selections for persistence (survives the Select-All branch
        # un-rendering the widget, which clears keyed state)
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

    config = get_decision_config('donation_default')
    is_from_live_settings = False

    if config is None:
        # Nothing pinned: the same card is rendered straight from the current
        # Page 1 / Donation Default tab settings.
        config = current_donation_configuration()
        is_from_live_settings = config is not None

    if config is None:
        return

    st.markdown("#### 3. Donation Default")
    
    # Main configuration display
    # Use donation_income_mode (primary) with fallback to income_spec_mode (older records)
    donation_income_mode = config.get('donation_income_mode', config.get('income_spec_mode', 'unknown'))
    
    with st.container():
        st.success(f"✅ **Donation Default Configuration**: {config['population_mode']} + {donation_income_mode}")
        
        # Metrics and details in columns
        if is_from_live_settings:
            # Rendered from the live settings: show only mode info (no simulation metrics yet)
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
            if is_from_live_settings:
                st.caption("Source: Auto-detected (single configuration)")
                st.caption("💡 This configuration was automatically detected from your Page 1 and Donation Default tab settings")
            else:
                st.caption(f"Selected at: {config['selected_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.caption(f"Source: {config['source']}")
        
        # Action buttons
        action_col1, action_col2 = st.columns([3, 1])
        
        with action_col1:
            if is_from_live_settings:
                st.caption("This single configuration will be used for the donation decision in complete simulations")
            else:
                st.caption("This selected configuration will be used for the donation decision in complete simulations")
        
        with action_col2:
            # Only show Clear button for explicitly selected configs
            if not is_from_live_settings:
                if st.button("🗑️ Clear", help="Clear the selected configuration", key="clear_donation_config"):
                    clear_decision_config('donation_default')
                    st.rerun()


def render_selected_disclose_income_config_display():
    """Display the selected disclose income configuration in the overview tab"""
    
    config = get_decision_config('disclose_income')
    
    if config is None:
        return
    
    st.markdown("#### 1. Disclose Income")
    
    # Get income mode and population mode from config
    income_mode = config.get('income_mode', config.get('params', {}).get('income_mode', 'Unknown'))
    population_mode = config.get('population_mode', st.session_state.get('population_mode', 'Unknown'))
    
    with st.container():
        st.success(f"✅ **Disclose Income Configuration**: {population_mode} + {income_mode}")
        
        # Metrics display
        metrics = config.get('metrics', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Population Mode", population_mode)
        
        with col2:
            st.metric("Income Mode", income_mode)
        
        with col3:
            y_rate = metrics.get('y_rate', 0)
            st.metric("Y Rate", f"{y_rate:.2%}")
        
        with col4:
            st.metric("Agents", f"{config.get('total_agents', 0):,}")
        
        # Additional details in expandable section
        with st.expander("📊 Configuration Details", expanded=False):
            params = config.get('params', {})
            
            st.markdown("**🔢 Model Parameters:**")
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                st.metric("Intercept", f"{params.get('intercept', 0):.4f}")
                st.metric("WoPB", f"{params.get('anchor_weights', {}).get('observed_prosocial', 0):.4f}")
                st.metric("WPB", f"{params.get('anchor_weights', {}).get('prosocial_weight', 0):.4f}")
            
            with param_col2:
                stochastic = params.get('stochastic', {})
                st.metric("Scale Factor", f"{stochastic.get('scale_factor', 1.0):.2f}")
                st.metric("Sigma Strategy", stochastic.get('sigma_strategy', 'overall'))
            
            st.markdown("**ℹ️ Configuration Info:**")
            st.caption(f"Selected at: {config.get('selected_timestamp', 'Unknown')}")
            st.caption(f"Original seed: {config.get('original_seed', 'Unknown')}")
            st.caption(f"Source: {config.get('source', 'Unknown')}")
        
        # Action buttons
        action_col1, action_col2 = st.columns([3, 1])
        
        with action_col1:
            st.caption("This configuration will be used for the disclose income decision in complete simulations")
        
        with action_col2:
            if st.button("🗑️ Clear", help="Clear the selected configuration", key="clear_disclose_income_config"):
                clear_decision_config('disclose_income')
                st.rerun()


def render_selected_disclose_documents_config_display():
    """Display the selected disclose documents configuration in the overview tab -
    mirrors render_selected_disclose_income_config_display (no anchor weights: the
    Disclose Documents model has none)."""
    
    config = get_decision_config('disclose_documents')
    
    if config is None:
        return
    
    st.markdown("#### 2. Disclose Documents")
    
    # Get income mode and population mode from config
    income_mode = config.get('income_mode', config.get('params', {}).get('income_mode', 'Unknown'))
    population_mode = config.get('population_mode', st.session_state.get('population_mode', 'Unknown'))
    
    with st.container():
        st.success(f"✅ **Disclose Documents Configuration**: {population_mode} + {income_mode}")
        
        # Metrics display
        metrics = config.get('metrics', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Population Mode", population_mode)
        
        with col2:
            st.metric("Income Mode", income_mode)
        
        with col3:
            y_rate = metrics.get('y_rate', 0)
            st.metric("Y Rate", f"{y_rate:.2%}")
        
        with col4:
            st.metric("Agents", f"{config.get('total_agents', 0):,}")
        
        # Additional details in expandable section
        with st.expander("📊 Configuration Details", expanded=False):
            params = config.get('params', {})
            
            st.markdown("**🔢 Model Parameters:**")
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                st.metric("Intercept", f"{params.get('intercept', 0):.4f}")
            
            with param_col2:
                stochastic = params.get('stochastic', {})
                st.metric("Scale Factor", f"{stochastic.get('scale_factor', 1.0):.2f}")
                st.metric("Sigma Strategy", stochastic.get('sigma_strategy', 'overall'))
            
            st.markdown("**ℹ️ Configuration Info:**")
            st.caption(f"Selected at: {config.get('selected_timestamp', 'Unknown')}")
            st.caption(f"Original seed: {config.get('original_seed', 'Unknown')}")
            st.caption(f"Source: {config.get('source', 'Unknown')}")
        
        # Action buttons
        action_col1, action_col2 = st.columns([3, 1])
        
        with action_col1:
            st.caption("This configuration will be used for the disclose documents decision in complete simulations")
        
        with action_col2:
            if st.button("🗑️ Clear", help="Clear the selected configuration", key="clear_disclose_documents_config"):
                clear_decision_config('disclose_documents')
                st.rerun()


def render_selected_rejected_transaction_config_display():
    """Display the selected Rejected Transaction Defaults (Decision 4) configuration in
    the overview tab - mirrors render_selected_disclose_income_config_display."""

    config = get_decision_config('rejected_transaction_defaults')

    # Only explicit "Use This Config" selections are stored.
    if config is None:
        return

    st.markdown("#### 4. Rejected Transaction Defaults")

    params = config.get('params', {}) or {}
    income_mode = config.get('income_mode', params.get('income_mode', 'Unknown'))
    population_mode = config.get('population_mode', st.session_state.get('population_mode', 'Unknown'))
    metrics = config.get('metrics', {}) or {}
    shown = rtd_summary_metric_keys(config.get('run_element'))

    with st.container():
        st.success(f"✅ **Rejected Transaction Defaults Configuration**: {population_mode} + {income_mode}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Population Mode", population_mode)
        with col2:
            st.metric("Income Mode", income_mode)
        with col3:
            # Only the average that belongs to the element the configuration
            # was selected from (both for a whole-decision run, none for Loyalty / WTP /
            # Risk-Taking / Flexibility)
            if 'mean_choice_length' in shown:
                mean_len = metrics.get('mean_choice_length')
                st.metric("Avg. Options List Length", f"{mean_len:.2f}" if mean_len is not None else "n/a")
            elif 'mean_default_list_length' in shown and 'mean_default_list_length' in metrics:
                st.metric("Avg. Integrated Default List Length", f"{metrics['mean_default_list_length']:.2f}")
        with col4:
            st.metric("Agents", f"{config.get('total_agents', 0):,}")

        with st.expander("📊 Configuration Details", expanded=False):
            st.markdown("**🔢 Element Intercepts:**")
            intercepts = params.get('intercepts', {}) or {}
            labels = [('ttp', 'β₀ Options List Length'), ('loyalty', 'β₁ Loyalty'),
                      ('wtp', 'β₂ Willingness-to-Pay'), ('risk_taking', 'β₃ Risk-Taking'),
                      ('flexibility', 'β₄ Flexibility')]
            int_cols = st.columns(len(labels))
            for col, (mech, label) in zip(int_cols, labels):
                with col:
                    st.metric(label, f"{float(intercepts.get(mech, 0.0)):.4f}")
            flex_anchor = params.get('flexibility_anchor') or {}
            if flex_anchor:
                # a weight the record lacks is the file's (rejected_transaction_defaults.flexibility_anchor)
                file_flex = get_config_repo().decision('rejected_transaction_defaults')['flexibility_anchor']
                st.caption(f"Flexibility anchor mix: W_OFlex = "
                           f"{float(flex_anchor.get('observed_weight', file_flex['observed_weight'])):.2f} observed · "
                           f"W_CFlex = {float(flex_anchor.get('calculated_weight', file_flex['calculated_weight'])):.2f} calculated")

            det_col1, det_col2 = st.columns(2)
            with det_col1:
                st.markdown("**🎲 Stochastic Settings (at save time):**")
                stochastic = params.get('stochastic', {}) or {}
                st.caption(f"Research Specification draws: {'on' if stochastic.get('sigma_enabled', True) else 'off'} · "
                           f"Copula draws: {'on' if stochastic.get('sigma_in_copula', False) else 'off'}")
                st.caption(f"σ mode: {stochastic.get('sigma_strategy', 'overall')} · "
                           f"σ coefficient: {float(stochastic.get('scale_factor', get_config_repo().rtd_sigma_coefficient_defaults()[0])):.2f}")
            with det_col2:
                st.markdown("**🔗 Rank Aggregation:**")
                agg = params.get('aggregation', {}) or {}
                st.caption(f"Integrated default list: {'enabled' if agg.get('enabled', True) else 'disabled'} "
                           "(Kemeny-Young, ties settled Schulze → Copeland → footrule → random)")
                shares = metrics.get('first_option_shares')
                if shares:
                    st.caption("First integrated option: " + ", ".join(
                        f"Option {o}: {float(v) * 100:.1f}%" for o, v in sorted(shares.items(), key=lambda kv: int(kv[0]))))
                if 'mean_default_list_length' in metrics and 'mean_default_list_length' in shown:
                    st.caption(f"Avg. integrated default list length: {metrics['mean_default_list_length']:.2f}")

            st.markdown("**ℹ️ Configuration Info:**")
            st.caption(f"Selected at: {config.get('selected_timestamp', 'Unknown')}")
            st.caption(f"Original seed: {config.get('original_seed', 'Unknown')}")
            st.caption(f"Source: {config.get('source', 'Unknown')}")

        action_col1, action_col2 = st.columns([3, 1])
        with action_col1:
            st.caption("This configuration will be used for the rejected transaction defaults decision in complete simulations")
        with action_col2:
            if st.button("🗑️ Clear", help="Clear the selected configuration", key="clear_rejected_transaction_config"):
                clear_decision_config('rejected_transaction_defaults')
                st.rerun()
