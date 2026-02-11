# app/pages/decision_execution.py
"""
Decision execution functions for running individual and combined simulations.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from app.simulation import run_simulation_from_sidebar
from app.models import ALL_DECISIONS


def format_decision_title(decision_name, include_number=False):
    """Format decision name for display, with special handling for specific decisions
    
    Args:
        decision_name: The decision name to format
        include_number: If True, prepend decision number (e.g., "1. Decision Name")
    
    Returns:
        Formatted decision title string
    """
    from app.models import ALL_DECISIONS
    
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


def can_run_complete_simulation():
    """
    Determine if complete simulation can run based on configuration state.
    
    This prevents running all decisions when:
    1. Any decision is in "Compare both" mode without a saved config selected
    2. Multiple configurations would be generated without an explicit selection
    
    IMPORTANT: Only check config requirements for decisions the user actually selected.
    Unselected decisions will use defaults and don't need explicit config selection.
    
    FIXED: Now accumulates ALL blocking issues instead of returning on first one.
    This allows the UI to display all conflicts at once.
    
    Returns:
        tuple: (can_run: bool, reason: str, config_count: int, block_type: str or None, blocking_issues: list)
            - can_run: Whether complete simulation is allowed
            - reason: Human-readable explanation (combined if multiple issues)
            - config_count: Number of configurations that would be generated
            - block_type: Primary block type for backward compat - "disclose_income", "donation_config", or None
            - blocking_issues: List of all blocking issues with details (NEW!)
    """
    # Use unified config system
    configs = get_selected_decision_configs()
    
    # Accumulator for ALL blocking issues
    blocking_issues = []
    
    # Get the user's selected decisions - only check requirements for these
    selected_decisions = []
    if hasattr(st.session_state, 'decision_params') and hasattr(st.session_state.decision_params, 'selected_decisions'):
        selected_decisions = st.session_state.decision_params.selected_decisions or []
    
    # Also include decisions that have saved configs (user explicitly saved a config)
    for decision_name in configs:
        if configs[decision_name].get('source') != 'auto_implied_single_config':
            if decision_name not in selected_decisions:
                selected_decisions = list(selected_decisions) + [decision_name]
    
    # ========================================================================
    # CALCULATE ALL COUNTS UPFRONT (shared between disclose_income and donation_default)
    # ========================================================================
    
    # Population modes: "Compare all" generates 3, others generate 1
    population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
    population_count = 3 if population_mode == "Compare all" else 1
    
    # Disclose income modes: "Compare both" generates 2, others generate 1
    di_income_mode = st.session_state.get('di_income_mode', 'Categorical only')
    di_income_count = 2 if ('compare' in str(di_income_mode).lower() or 'both' in str(di_income_mode).lower()) else 1
    
    # Donation default income modes: "Compare both" generates 2, others generate 1
    # FIX: Get donation_default's ACTUAL income mode from its dedicated storage
    # The global income_spec_mode can be contaminated when running disclose_income with "Compare both"
    # (simulation.py syncs di_income_mode to income_spec_mode for results display)
    # Priority: 1) donation tab persistence, 2) page2_tab_income_spec_mode widget, 3) global income_spec_mode
    donation_income_mode = None
    if hasattr(st.session_state, 'donation_tab_persistence') and 'income_spec_mode' in st.session_state.donation_tab_persistence:
        donation_income_mode = st.session_state.donation_tab_persistence['income_spec_mode']
    elif 'page2_tab_income_spec_mode' in st.session_state:
        donation_income_mode = st.session_state.page2_tab_income_spec_mode
    else:
        donation_income_mode = st.session_state.get('income_spec_mode', 'categorical only')
    
    donation_income_count = 2 if ('compare' in str(donation_income_mode).lower() or 'both' in str(donation_income_mode).lower()) else 1
    
    # ========================================================================
    # CHECK DISCLOSE_INCOME (only if selected)
    # FIX: Now correctly multiplies by population_count like donation_default does
    # ========================================================================
    
    disclose_income_selected = 'disclose_income' in selected_decisions
    di_total_configs = population_count * di_income_count  # FIX: Include population_count!
    has_disclose_income_config = False
    di_saved_mode = None
    
    if disclose_income_selected:
        # Check if user has a saved disclose_income config (not auto-implied)
        if 'disclose_income' in configs:
            di_config = configs['disclose_income']
            if di_config.get('source') != 'auto_implied_single_config':
                has_disclose_income_config = True
                di_saved_mode = di_config.get('params', {}).get('income_mode', 
                    di_config.get('income_mode', 'Unknown'))
        
        # Also check legacy storage
        if not has_disclose_income_config and hasattr(st.session_state, 'selected_disclose_income_config'):
            legacy_config = st.session_state.selected_disclose_income_config
            if legacy_config and legacy_config.get('source') != 'auto_implied_single_config':
                has_disclose_income_config = True
                di_saved_mode = legacy_config.get('income_mode', legacy_config.get('params', {}).get('income_mode'))
        
        # If multiple configs and no saved selection, add to blocking issues
        if di_total_configs > 1 and not has_disclose_income_config:
            blocking_issues.append({
                'decision': 'disclose_income',
                'block_type': 'disclose_income',
                'config_count': di_total_configs,
                'reason': f"Disclose Income has {di_total_configs} configurations (population: {population_count}, income: {di_income_count}) - please run disclose_income only and select one"
            })
    
    # ========================================================================
    # CHECK DONATION_DEFAULT (only if selected)
    # ========================================================================
    
    donation_default_selected = 'donation_default' in selected_decisions
    donation_total_configs = population_count * donation_income_count
    has_donation_config = False
    donation_saved_info = None
    
    if donation_default_selected:
        # Check if user has a saved donation_default config (not auto-implied)
        if 'donation_default' in configs:
            config = configs['donation_default']
            if config.get('source') != 'auto_implied_single_config':
                has_donation_config = True
                pop_mode = config.get('population_mode', st.session_state.get('population_mode', 'Unknown'))
                inc_mode = config.get('params', {}).get('income_mode', config.get('income_spec_mode', 'Unknown'))
                donation_saved_info = f"{pop_mode} + {inc_mode}"
        
        # Also check legacy storage
        if not has_donation_config and hasattr(st.session_state, 'selected_donation_config'):
            legacy_config = st.session_state.selected_donation_config
            if legacy_config and legacy_config.get('source') != 'auto_implied_single_config':
                has_donation_config = True
                pop_mode = legacy_config.get('population_mode', 'Unknown')
                inc_mode = legacy_config.get('income_spec_mode', 'Unknown')
                donation_saved_info = f"{pop_mode} + {inc_mode}"
        
        # If multiple configs and no saved selection, add to blocking issues
        if donation_total_configs > 1 and not has_donation_config:
            blocking_issues.append({
                'decision': 'donation_default',
                'block_type': 'donation_config',
                'config_count': donation_total_configs,
                'reason': f"Donation Default has {donation_total_configs} configurations (population: {population_count}, income: {donation_income_count}) - please run donation_default only and select one"
            })
    
    # ========================================================================
    # DETERMINE RESULT
    # ========================================================================
    
    # If there are any blocking issues, return them ALL
    if blocking_issues:
        # Combine all reasons for display
        combined_reasons = "\n\n".join([issue['reason'] for issue in blocking_issues])
        # Use first block_type for backward compatibility
        primary_block_type = blocking_issues[0]['block_type']
        # Total config count is max of all (represents worst case)
        max_config_count = max(issue['config_count'] for issue in blocking_issues)
        
        return (False, combined_reasons, max_config_count, primary_block_type, blocking_issues)
    
    # No blocking issues - build success message
    config_parts = []
    
    # Show disclose_income config if selected and saved
    if has_disclose_income_config and di_saved_mode:
        config_parts.append(f"Disclose Income: {di_saved_mode}")
    
    # Show donation_default config if selected and saved
    if has_donation_config and donation_saved_info:
        config_parts.append(f"Donation Default: {donation_saved_info}")
    
    # Determine total config count for display
    total_configs = max(di_total_configs if disclose_income_selected else 1,
                       donation_total_configs if donation_default_selected else 1)
    
    if config_parts:
        return (True, f"Using saved configuration(s): {', '.join(config_parts)}", total_configs, None, [])
    elif not disclose_income_selected and not donation_default_selected:
        return (True, "Using default values for all decisions", 1, None, [])
    else:
        return (True, "Single configuration", 1, None, [])


def get_implied_single_configuration():
    """
    Returns the implied configuration when only one exists, otherwise None.
    
    This allows the system to auto-populate selected_donation_config when the user
    has configured a single population mode + income mode combination, without
    requiring them to explicitly run donation_default first.
    
    Returns:
        dict: Configuration object if only one config exists, None otherwise
    """
    try:
        # Check configuration count
        # Note: Using *_ to handle the new 5th return value (blocking_issues list)
        can_run, reason, config_count, block_type, *_ = can_run_complete_simulation()
        
        # Only return implied config when exactly one configuration exists
        if config_count != 1:
            return None
        
        # Get current mode settings from session state
        population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
        income_spec_mode = st.session_state.get('income_spec_mode', 'categorical only')
        
        # Try to get current coefficient values (may fail if not loaded yet)
        try:
            coefficients = get_current_coefficients()
        except Exception:
            # Fallback: create minimal coefficients structure
            coefficients = {
                'intercept': 0.0,
                'beta_group': {'MidSub': 0.0, 'NoSub': 0.0, 'FullSub': 0.0},
                'beta_income_q': {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0, 'Q4': 0.0, 'Q5': 0.0},
                'beta_income_linear': 0.0,
                'beta_study': {'Incoming': 0.0, 'Law5yr': 0.0, 'UG3yr': 0.0, 'Grad2yr': 0.0},
                'beta_hh': 0.0
            }
        
        # Try to get current stochastic parameters  
        try:
            stochastic_params = get_current_stochastic_params()
        except Exception:
            # Fallback: create minimal stochastic structure
            stochastic_params = {
                'stochastic': {
                    'sigma_value': 9.8995,
                    'sigma_coefficient': 1.0,
                    'sigma_in_copula': False,
                    'sigma_in_research': True
                },
                'anchor_weights': {
                    'observed': 0.75,
                    'predicted': 0.25
                }
            }
        
        # Build implied configuration (without metrics since simulation hasn't run)
        # NOTE: donation_income_mode is the PRIMARY key for donation-specific income mode
        config = {
            'result_key': f"implied_{population_mode.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{income_spec_mode.replace(' ', '_')}",
            'population_mode': population_mode,
            # NEW: donation-specific income mode (primary)
            'donation_income_mode': income_spec_mode,
            # DEPRECATED: kept for backwards compatibility
            'income_spec_mode': income_spec_mode,
            'coefficients': coefficients,
            'stochastic_params': stochastic_params,
            'metrics': {
                'mean_donation': st.session_state.get('final_donation_rate_default_value', 0.10),
                'std_donation': None,
                'median_donation': None,
                'min_donation': None,
                'max_donation': None,
                'q25_donation': None,
                'q75_donation': None,
                'donation_column_used': 'donation_default'
            },
            'selected_timestamp': datetime.now(),
            'total_agents': st.session_state.get('n_agents', 1000),
            'source': 'auto_implied_single_config',
            'original_seed': st.session_state.get('seed_input', st.session_state.get('seed', 42)),
            'original_n_agents': st.session_state.get('n_agents', 1000)
        }
        
        return config
    except Exception:
        # If anything fails, return None (don't auto-populate)
        return None


def auto_populate_single_donation_config():
    """
    Auto-populate selected_donation_config when only one configuration exists.
    
    This should be called during page initialization to ensure the Overview tab
    shows the current configuration settings without requiring explicit selection.
    
    IMPORTANT: Auto-implied configs are ONLY for UI convenience. They should NOT
    be treated as "saved" configs for the Run Complete Simulation feature.
    When user is in "Compare both" mode, any auto-implied configs must be cleared.
    
    Returns:
        bool: True if config was auto-populated, False otherwise
    """
    try:
        # First, check if donation_default is in the user's selected_decisions
        # Don't auto-populate configs for decisions the user hasn't selected
        selected_decisions = []
        if hasattr(st.session_state, 'decision_params') and hasattr(st.session_state.decision_params, 'selected_decisions'):
            selected_decisions = st.session_state.decision_params.selected_decisions or []
        
        if 'donation_default' not in selected_decisions:
            # User hasn't selected donation_default - don't create auto-implied config
            return False
        
        # Next, check if we're in a "Compare both" mode - if so, clear any auto-implied configs
        population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
        income_spec_mode = st.session_state.get('income_spec_mode', 'categorical only')
        
        is_compare_mode = (
            population_mode == "Compare all" or 
            'compare' in str(income_spec_mode).lower() or 
            'both' in str(income_spec_mode).lower()
        )
        
        if is_compare_mode:
            # In compare mode - clear any auto-implied configs from BOTH legacy and unified storage
            if 'selected_donation_config' in st.session_state:
                existing = st.session_state.selected_donation_config
                if existing and existing.get('source') == 'auto_implied_single_config':
                    del st.session_state.selected_donation_config
                    # Also clear from unified storage
                    configs = get_selected_decision_configs()
                    if 'donation_default' in configs and configs['donation_default'].get('source') == 'auto_implied_single_config':
                        del configs['donation_default']
            return False
        
        # Check if already has a configuration (use 'in' for Streamlit session state)
        has_existing = 'selected_donation_config' in st.session_state and st.session_state.selected_donation_config is not None
        
        if has_existing:
            # Check if the existing config matches current settings
            current_pop = st.session_state.get('population_mode', 'Copula (synthetic)')
            current_income = st.session_state.get('income_spec_mode', 'categorical only')
            existing = st.session_state.selected_donation_config
            
            # If modes changed, clear the old config so we can re-evaluate
            if existing.get('population_mode') != current_pop or existing.get('income_spec_mode') != current_income:
                # Modes have changed - check if we should update
                can_run, reason, config_count, block_type, *_ = can_run_complete_simulation()
                if config_count == 1:
                    # Single config now - only auto-populate if the existing was auto-implied
                    # Don't overwrite explicitly saved configs
                    if existing.get('source') == 'auto_implied_single_config':
                        implied_config = get_implied_single_configuration()
                        if implied_config:
                            st.session_state.selected_donation_config = implied_config
                            
                            # Sync final_donation_rate_default_value with implied config's mean donation
                            mean_donation = implied_config['metrics']['mean_donation']
                            st.session_state.final_donation_rate_default_value = mean_donation
                            
                            if '_persistent_defaults' not in st.session_state:
                                st.session_state._persistent_defaults = {}
                            st.session_state._persistent_defaults['final_donation_rate_default_value'] = mean_donation
                            
                            return True
                else:
                    # Multiple configs now - clear the old single-config selection if it was auto-implied
                    if existing.get('source') == 'auto_implied_single_config':
                        del st.session_state.selected_donation_config
                        # Also clear from unified storage
                        configs = get_selected_decision_configs()
                        if 'donation_default' in configs and configs['donation_default'].get('source') == 'auto_implied_single_config':
                            del configs['donation_default']
            return False
        
        # No existing config - try to get implied single configuration
        implied_config = get_implied_single_configuration()
        
        if implied_config:
            # Auto-populate with the single implied configuration
            st.session_state.selected_donation_config = implied_config
            
            # Sync final_donation_rate_default_value with implied config's mean donation
            # This happens only ONCE when the config is auto-populated
            mean_donation = implied_config['metrics']['mean_donation']
            st.session_state.final_donation_rate_default_value = mean_donation
            
            if '_persistent_defaults' not in st.session_state:
                st.session_state._persistent_defaults = {}
            st.session_state._persistent_defaults['final_donation_rate_default_value'] = mean_donation
            
            return True
        
        return False
    except Exception:
        # If anything fails, don't crash - just return False
        return False


def render_simulation_buttons(decision_name, selected_decisions):
    """
    Render both individual and complete simulation buttons for a decision tab.
    
    This provides a consistent interface across all decision tabs, allowing users to:
    1. Run only the current decision (for testing/validation)
    2. Run the complete simulation with all 13 decisions
    
    Args:
        decision_name: Name of the current decision (e.g., "donation_default")
        selected_decisions: List of all selected decisions from session state
    """
    st.markdown("---")
    st.markdown('<h3 class="section-header">🚀 Simulation Options</h3>', unsafe_allow_html=True)
    
    # Safety check: ensure selected_decisions is a list
    if selected_decisions is None or not isinstance(selected_decisions, list):
        selected_decisions = []
    
    # Calculate unselected decisions for informational purposes
    unselected_decisions = [d for d in ALL_DECISIONS if d not in selected_decisions]
    
    # Display context in two columns
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"**🔬 Test Run**\n\nTest only {format_decision_title(decision_name)} with current parameters")
        st.caption("Quick validation of this decision's configuration")
    with col_info2:
        st.info(f"**🎯 Complete Simulation**\n\nRun all {len(ALL_DECISIONS)} decisions end-to-end")
        if len(unselected_decisions) > 0:
            st.caption(f"{len(selected_decisions)} custom + {len(unselected_decisions)} defaults")
        else:
            st.caption("All decisions use custom parameters")
    
    # Render action buttons in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            f"🔬 Run {format_decision_title(decision_name)} Only", 
            type="primary", 
            use_container_width=True,
            key=f"run_{decision_name}_only_btn",
            help=f"Execute only {format_decision_title(decision_name)} to test and validate your parameters"
        ):
            run_individual_decision(decision_name)
    
    with col2:
        # Show detailed breakdown in expander
        with st.expander("📊 View Complete Simulation Configuration", expanded=False):
            st.markdown("**What will run in Complete Simulation:**")
            
            if len(selected_decisions) > 0:
                st.markdown(f"**✅ Custom Parameters ({len(selected_decisions)} decisions):**")
                for i, dec in enumerate(selected_decisions, 1):
                    icon = "🎯" if dec == decision_name else "✓"
                    label = " **(current tab)**" if dec == decision_name else ""
                    st.caption(f"{icon} {format_decision_title(dec, include_number=True)}{label}")
            
            if len(unselected_decisions) > 0:
                st.markdown(f"\n**🔧 Default Values ({len(unselected_decisions)} decisions):**")
                for i, dec in enumerate(unselected_decisions, 1):
                    st.caption(f"{format_decision_title(dec, include_number=True)}")
        
        # Check if complete simulation can run (validation for multiple configurations)
        result = can_run_complete_simulation()
        can_run, reason, config_count, block_type = result[:4]
        blocking_issues = result[4] if len(result) > 4 else []
        
        if not can_run:
            # Disabled button with explanation
            help_text = f"{len(blocking_issues)} configuration issue(s) detected" if len(blocking_issues) > 1 else ("Disclose Income is in Compare mode" if block_type == "disclose_income" else "Multiple configurations detected - select one first")
            st.button(
                "🎯 Run Complete Simulation", 
                type="primary",
                use_container_width=True,
                disabled=True,
                key=f"run_complete_from_{decision_name}_btn_disabled",
                help=help_text
            )
            
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
1. Go to the **Disclose Income** tab
2. Run **disclose_income only** and select one configuration
3. Or change to **"Categorical only"** or **"Continuous only"** mode
                        """)
                    else:
                        # donation_config block type
                        st.warning(f"""
**Issue {i}: Donation Default**

{issue['reason']}

**Action Required:**
1. Go to the **Donation Default** tab
2. Run **donation_default only** and select one configuration
                        """)
            else:
                # Single issue - show original format
                if block_type == "disclose_income":
                    st.warning(f"""
⚠️ **Disclose Income Configuration Required**

{reason}

**Action Required:**

1. Go to the **Disclose Income** tab
2. Change "Income Specification for Disclosure Model" from "Compare both" to either **"Categorical only"** or **"Continuous only"**
3. Return here and click **Run Complete Simulation**

This ensures all decisions produce a single result set.
                    """)
                else:
                    # donation_config block type
                    st.warning(f"""
⚠️ **Multiple Donation Configurations Detected**

{reason}

**Action Required:**

1. Go to the **Donation Default** tab
2. Run **donation_default only**
3. **Select one configuration** from the results
4. Return here and click **Run Complete Simulation**

This ensures all decisions use consistent settings.
                    """)
            
        else:
            # Enabled button - can proceed
            # Show info about selected config if applicable
            # CRITICAL: Only show "Using:" message for EXPLICITLY saved configs, not auto-implied ones
            if config_count > 1 and hasattr(st.session_state, 'selected_donation_config'):
                config = st.session_state.selected_donation_config
                # Only show message if this was explicitly selected by user
                if config and config.get('source') != 'auto_implied_single_config':
                    st.success(f"✅ Using: {config['population_mode']} + {config['income_spec_mode']}")
            
            if st.button(
                "🎯 Run Complete Simulation", 
                type="primary",
                use_container_width=True,
                key=f"run_complete_from_{decision_name}_btn",
                help=f"Run all {len(ALL_DECISIONS)} decisions with current configuration"
            ):
                # Show confirmation info before running
                with st.spinner("🔄 Preparing complete simulation..."):
                    st.info(f"""
                    **🚀 Starting complete end-to-end simulation**
                    
                    - Running all {len(ALL_DECISIONS)} decisions in sequence
                    - Current decision ({format_decision_title(decision_name)}) will use your configured parameters above
                    - {len(selected_decisions)} decisions total with custom parameters
                    - {len(unselected_decisions)} decisions using default values
                    """)
                    
                    # Execute the combined simulation
                    run_combined_simulation(selected_decisions)

# Default values for unselected decisions
DEFAULT_DECISION_VALUES = {
    "donation_default": 0.10,  # 10%
    "disclose_income": {
        "type": "random_probability",
        "probability_y": 0.5,  # 50% chance of Y (disclosing)
        "options": ["Y", "N"],
        "description": "Probability of disclosing income for Fixed status"
    },
    "disclose_documents": {
        "type": "random_probability",
        "probability_y": 0.5,  # 50% chance of Y (disclosing)
        "options": ["Y", "N"],
        "description": "Probability of disclosing documents (applies only to agents qualified for discount: income < threshold)"
    },
    "rejected_transaction_defaults": {
        "type": "prioritized_selection",
        "priority_template": ["forgo_transaction"],  # Default: all agents use Option 5 only
        "options": [
            ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
            ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"), 
            ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"),
            ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
            ("forgo_transaction", "Option 5: Forgo the purchase request")
        ],
        "description": "Each agent gets a prioritized list. If Option 5 is included, it must be last."
    },
    "vendor_choice_weights": {
        "type": "checkbox_selection",
        "default_selection": ["price", "quality", "proximity", "sustainability"],
        "parameters": {
            "price": {"name": "Price", "description": "the product price offered to the customer"},
            "quality": {"name": "Quality", "description": "product quality based on customer ratings"},
            "proximity": {"name": "Proximity", "description": "the proximity of vendor to customer"},
            "sustainability": {"name": "Sustainability", "description": "vendor sustainability rating"}
        }
    },
    "purchasing_quantity": "RANDOM_WITHIN_LIMIT",  # Random within purchasing limit
    "purchasing_frequency": "CALCULATED",  # Consumption quantity / Number of Periods
    "vendor_selection": "deterministic",  # Deterministic based on highest weighted vendor-product score
    "purchase_vs_bid": {
        "type": "random_probability",
        "probability_y": 0.5,  # 50% chance of Purchase Now (vs bid)
        "options": ["Purchase Now", "bid"],
        "description": "Probability of Purchase Now vs bidding (applies only to REGULAR customers - those who did not disclose income)"
    },
    "bid_value": "RANDOM_WITHIN_RANGE",  # Random within bidding price range
    "rejected_transaction_option": {
        "type": "radio_selection",
        "default_option": "forgo_transaction", 
        "options": [
            ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
            ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
            ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
            ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
            ("forgo_transaction", "Option 5: Forgo the purchase request")
        ]
    },
    "rejected_bid_value": "NA",  # Not relevant given Option 5
    "final_donation_rate": 0.10  # Keep default 10%
}

# Description text for display purposes
DEFAULT_DECISION_DESCRIPTIONS = {
    "donation_default": "10%",
    "disclose_income": "configurable probability Y/N (default 50% each)", 
    "disclose_documents": "configurable probability Y/N (applies only to agents with income < discount threshold, default 50% each)",
    "rejected_transaction_defaults": "Selected option for handling rejected transactions will be applied to all agents",
    "vendor_choice_weights": "equal weight distribution among selected parameters (Price, Quality, Proximity, Sustainability)",
    "purchasing_quantity": "random within purchasing limit",
    "purchasing_frequency": "Consumption quantity divided by Number of Periods",
    "vendor_selection": "deterministic based on highest weighted vendor-product score",
    "purchase_vs_bid": "configurable probability Purchase Now/bid for REGULAR customers only (default 50% each)",
    "bid_value": "random within bidding price range (only for REGULAR customers who chose to bid)",
    "rejected_transaction_option": "Selected specific option for transaction rejection handling will be used",
    "rejected_bid_value": "Default handling for rejected bid values will be applied",
    "final_donation_rate": "Default donation rate will be maintained"
}


def get_actual_default_value(decision_name, sim_params=None):
    """
    Get the actual default value for a decision, handling random generation where needed.
    This function returns values that can be used directly by the simulation.
    
    Priority order:
    1. Pre-configured default from Page 2 Overview tab ({decision_name}_default_*)
    2. Post-simulation adjustment from Results page ({decision_name}_*)
    3. Hard-coded default from DEFAULT_DECISION_VALUES
    """
    import random
    import streamlit as st
    
    base_value = DEFAULT_DECISION_VALUES.get(decision_name)
    
    # NEW: Handle parametric random decisions with configurable probabilities
    if isinstance(base_value, dict) and base_value.get("type") == "random_probability":
        # Priority 1: Check for pre-configured default from Overview tab
        pre_config_key = f"{decision_name}_default_probability_y"
        # Priority 2: Check for post-simulation adjustment from Results page
        post_sim_key = f"{decision_name}_probability_y"
        # Priority 3: Use hard-coded default
        
        probability_y = st.session_state.get(
            pre_config_key, 
            st.session_state.get(
                post_sim_key, 
                base_value.get("probability_y", 0.5)
            )
        )
        
        options = base_value.get("options", ["Y", "N"])
        
        # Weighted random choice
        if random.random() < probability_y:
            return options[0]  # First option (Y or purchase)
        else:
            return options[1]  # Second option (N or bid)
    
    # Handle prioritized selection decisions (rejected_transaction_defaults)
    elif isinstance(base_value, dict) and base_value.get("type") == "prioritized_selection":
        # Priority 1: Check for configured priority template from Overview tab
        pre_config_key = f"{decision_name}_priority_template"
        
        # Priority 2: Use hard-coded default template
        priority_template = st.session_state.get(
            pre_config_key,
            base_value.get("priority_template", ["forgo_transaction"])
        )
        return priority_template
    
    # Handle radio selection decisions (rejected transaction options)
    elif isinstance(base_value, dict) and base_value.get("type") == "radio_selection":
        # Priority 1: Check for pre-configured default from Overview tab
        pre_config_key = f"{decision_name}_default_selection"
        
        # Priority 2: Check for post-simulation adjustments (legacy keys)
        if decision_name == "rejected_transaction_defaults":
            post_sim_key = "rejected_transaction_defaults_option"
        elif decision_name == "rejected_transaction_option":
            post_sim_key = "rejected_transaction_option_selection"
        else:
            post_sim_key = f"{decision_name}_selection"
        
        # Priority 3: Use hard-coded default
        selected_value = st.session_state.get(
            pre_config_key,
            st.session_state.get(
                post_sim_key,
                base_value.get("default_option", "forgo_transaction")
            )
        )
        return selected_value
    
    # Handle checkbox selection decisions (vendor choice weights)
    elif isinstance(base_value, dict) and base_value.get("type") == "checkbox_selection":
        # Priority 1: Check for pre-configured default from Overview tab
        pre_config_key = f"{decision_name}_default_params"
        # Priority 2: Check for post-simulation adjustment
        post_sim_key = "vendor_choice_weights_selection"
        
        selected_params = st.session_state.get(
            pre_config_key,
            st.session_state.get(
                post_sim_key,
                base_value.get("default_selection", [])
            )
        )
        
        # Calculate equal weights for selected parameters
        if len(selected_params) > 0:
            weight_per_param = 1.0 / len(selected_params)
            weights = {}
            
            # Set weights for all parameters
            for param_key in base_value.get("parameters", {}).keys():
                if param_key in selected_params:
                    weights[param_key] = weight_per_param
                else:
                    weights[param_key] = 0.0
            
            return weights
        else:
            # Fallback to equal weights if nothing selected
            params = list(base_value.get("parameters", {}).keys())
            if params:
                weight_per_param = 1.0 / len(params)
                return {param: weight_per_param for param in params}
            else:
                return {"price": 0.25, "quality": 0.25, "proximity": 0.25, "sustainability": 0.25}
    
    # Handle numeric defaults (donation_default, final_donation_rate, etc.)
    elif isinstance(base_value, (int, float)):
        # Priority 1: Check for pre-configured default from Overview tab
        pre_config_key = f"{decision_name}_default_value"
        # Priority 2: Check for post-simulation adjustment (specific keys)
        post_sim_key = f"{decision_name}_config"
        
        # Priority 3: Use hard-coded default
        return st.session_state.get(
            pre_config_key,
            st.session_state.get(
                post_sim_key,
                base_value
            )
        )
    
    # Handle random within purchasing limit
    elif base_value == "RANDOM_WITHIN_LIMIT":
        # This needs to be handled per agent based on their income category
        # Return a placeholder that the simulation will interpret
        return "RANDOM_WITHIN_LIMIT"
    
    # Handle calculated purchasing frequency
    elif base_value == "CALCULATED":
        # This will be calculated based on purchasing quantity / period duration
        # Return a placeholder that the simulation will interpret
        return "CALCULATED"
    
    # Handle random bid value within range
    elif base_value == "RANDOM_WITHIN_RANGE":
        # This needs market price and bidding range from sim_params
        # Return a placeholder that the simulation will interpret
        return "RANDOM_WITHIN_RANGE"
    
    # For all other values (numbers, dictionaries, strings), return as-is
    else:
        return base_value


def run_individual_decision(decision_name):
    """Run a single decision simulation.
    
    Each decision uses its OWN settings from its respective tab.
    No global overrides are applied.
    """
    with st.spinner(f"Running {decision_name} simulation..."):
        try:
            # Store original state values to restore later
            original_decisions = st.session_state.decision_params.selected_decisions.copy()
            original_custom_decisions = getattr(st.session_state, 'custom_decisions', [])
            original_default_decisions = getattr(st.session_state, 'default_decisions', [])
            
            # FIX: Store in session state for restoration after st.rerun()
            # This is needed because run_simulation_from_sidebar() may call st.rerun()
            # BEFORE this function's restoration code can execute
            # NOTE: Only store selected_decisions - custom_decisions/default_decisions should
            # reflect the current run (needed for should_enable_selection())
            st.session_state._pending_decisions_restore = {
                'selected_decisions': original_decisions
            }
            
            # For donation_default, clear any saved configuration to allow fresh run with current tab settings
            if decision_name == "donation_default" and hasattr(st.session_state, 'selected_donation_config'):
                st.info("🔄 Clearing saved donation configuration - will use current tab settings")
                delattr(st.session_state, 'selected_donation_config')
            
            # For disclose_income, clear any saved configuration to allow fresh run with current tab settings
            if decision_name == "disclose_income":
                # Clear from unified storage
                if 'selected_decision_configs' in st.session_state and 'disclose_income' in st.session_state.selected_decision_configs:
                    del st.session_state.selected_decision_configs['disclose_income']
                # Clear from legacy storage
                if hasattr(st.session_state, 'selected_disclose_income_config'):
                    delattr(st.session_state, 'selected_disclose_income_config')
            
            # Modify selected decisions for simulation
            st.session_state.decision_params.selected_decisions = [decision_name]
            
            # If this is donation_default, collect and apply coefficient parameters
            if decision_name == "donation_default":
                # Collect regression coefficients from YAML-loaded session state
                coeffs = get_current_coefficients()
                coeffs['income_mode'] = st.session_state.get('income_spec_mode', 'categorical')
                
                # Store the coefficients in decision_params for the simulation
                if not hasattr(st.session_state, 'custom_coefficients'):
                    st.session_state.custom_coefficients = {}
                st.session_state.custom_coefficients['donation_default'] = coeffs
            
            # Set state variables correctly for individual runs
            # This ensures the results display shows only the executed decision
            st.session_state.custom_decisions = [decision_name]  # Only this decision was run with custom parameters
            st.session_state.default_decisions = []  # No decisions used default values (since only one was run)
            
            # Run simulation - check if Monte Carlo or Single Run mode
            if st.session_state.sim_params.simulation_mode == "Monte-Carlo Study":
                # Run Monte Carlo study
                from app.simulation import run_monte_carlo_study
                mc_summary, mc_detailed, output_log = run_monte_carlo_study()
                if mc_summary is not None:
                    st.session_state.mc_results = {
                        'summary': mc_summary,
                        'detailed': mc_detailed,
                        'log': output_log
                    }
                    st.session_state.simulation_results = None
                    st.success(f"✅ Monte Carlo study for {decision_name} complete!")
                    
                    # Show Monte Carlo preview
                    if 'donation_default' in mc_summary['decision'].values:
                        donation_row = mc_summary[mc_summary['decision'] == 'donation_default'].iloc[0]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Donation Rate", f"{donation_row['mean']:.2%}")
                        with col2:
                            st.metric("Std Deviation", f"{donation_row['std']:.2%}")
                        with col3:
                            st.metric("Number of Runs", int(donation_row['runs']))
                    
                    # Navigate to results page
                    st.session_state.page = 'results'
                    st.info("🔄 Redirecting to Results page to view full Monte Carlo analysis...")
                    
                    # FIX: Restore ONLY selected_decisions BEFORE rerun
                    # Do NOT restore custom_decisions/default_decisions - they should reflect the current run
                    # (needed for should_enable_selection() to show "Use This Config" button)
                    st.session_state.decision_params.selected_decisions = original_decisions
                    # Clean up the pending restore flag
                    if hasattr(st.session_state, '_pending_decisions_restore'):
                        del st.session_state._pending_decisions_restore
                    
                    st.rerun()
                else:
                    st.error("❌ Monte Carlo simulation returned no results")
            else:
                # Run single simulation
                run_simulation_from_sidebar()
            
            # Store in individual results (only for single run mode)
            if st.session_state.sim_params.simulation_mode != "Monte-Carlo Study" and st.session_state.simulation_results:
                if 'individual_results' not in st.session_state:
                    st.session_state.individual_results = {}
                
                st.session_state.individual_results[decision_name] = st.session_state.simulation_results
                st.success(f"✅ {decision_name} simulation complete!")
                
                # Show preview of results
                results = next(iter(st.session_state.simulation_results.values()))
                if results is not None and not results.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Agents Simulated", f"{len(results):,}")
                    with col2:
                        if decision_name == "donation_default":
                            if 'donation_default' in results.columns:
                                st.metric("Average Donation Rate", f"{results['donation_default'].mean():.2%}")
            
            # Restore ONLY selected_decisions - keep custom_decisions/default_decisions as-is
            # because they're needed for should_enable_selection() to show "Use This Config" button
            st.session_state.decision_params.selected_decisions = original_decisions
            # Do NOT restore custom_decisions and default_decisions
            
            # Clear pending restore flag since we restored successfully
            if hasattr(st.session_state, '_pending_decisions_restore'):
                del st.session_state._pending_decisions_restore
            
        except Exception as e:
            # Restore selected_decisions on exception to ensure state is consistent
            st.session_state.decision_params.selected_decisions = original_decisions
            # Do NOT restore custom_decisions and default_decisions
            if hasattr(st.session_state, '_pending_decisions_restore'):
                del st.session_state._pending_decisions_restore
            
            st.error(f"❌ Error running {decision_name}: {str(e)}")
            import traceback
            st.text(traceback.format_exc())


def _validate_income_mode_compatibility(selected_decisions, unselected_decisions, using_selected_donation_config):
    """
    Validate that income modes are compatible across decisions and show warning if mismatched.
    
    This is informational only - we proceed with user's explicit settings.
    Each decision uses its own configured income mode independently.
    """
    # Determine donation_default income mode
    if using_selected_donation_config and hasattr(st.session_state, 'selected_donation_config') and st.session_state.selected_donation_config:
        config = st.session_state.selected_donation_config
        donation_mode = config.get('donation_income_mode', config.get('income_spec_mode', 'categorical only'))
    else:
        donation_mode = st.session_state.get('income_spec_mode', 'categorical only')
    
    # Determine disclose_income mode
    di_mode = st.session_state.get('di_income_mode', 'Categorical only')
    
    # Normalize for comparison
    def normalize_mode(mode):
        mode_lower = str(mode).lower()
        if 'continuous' in mode_lower:
            return 'continuous'
        elif 'categorical' in mode_lower:
            return 'categorical'
        elif 'compare' in mode_lower or 'both' in mode_lower:
            return 'compare'
        return 'categorical'
    
    donation_normalized = normalize_mode(donation_mode)
    di_normalized = normalize_mode(di_mode)
    
    # Check for mismatch (ignore if either is in "compare" mode)
    if donation_normalized != 'compare' and di_normalized != 'compare':
        if donation_normalized != di_normalized:
            st.warning(f"""
⚠️ **Income Mode Mismatch Detected**

- **Donation Default**: {donation_mode} ({donation_normalized})
- **Disclose Income**: {di_mode} ({di_normalized})

Each decision will use its own configured income mode. This is intentional - 
you have configured different income specifications for each decision.

If you want them to match, update the settings on the respective decision tabs.
            """)


def run_combined_simulation(selected_decisions):
    """Run complete simulation with selected decisions using custom parameters and unselected decisions using defaults.
    
    NOTE: Each decision uses its OWN income mode setting independently.
    - donation_default uses selected_donation_config.donation_income_mode (if saved) or income_spec_mode
    - disclose_income uses di_income_mode from its tab settings
    - Other decisions use their own respective settings
    
    We no longer override global income_spec_mode from selected_donation_config.
    
    FIXED: Now properly adds ALL decisions with saved configs to effective_selected_decisions,
    not just disclose_income. This ensures manually selected decisions get executed.
    """
    
    # Store information about selected vs default decisions
    effective_selected_decisions = list(selected_decisions)
    
    # Track which decisions have saved configs for display purposes
    saved_config_info = {}  # decision_name -> display info
    
    # ========================================================================
    # FIX: GENERICALLY ADD ALL DECISIONS WITH SAVED CONFIGS TO EFFECTIVE LIST
    # This ensures any decision with a saved config is treated as "custom/selected"
    # ========================================================================
    
    # Check unified storage for ANY decision with saved config
    if 'selected_decision_configs' in st.session_state:
        for decision_name, config in st.session_state.selected_decision_configs.items():
            # Skip auto-implied configs (they don't count as explicit selection)
            if config.get('source') == 'auto_implied_single_config':
                continue
            
            # Add to effective_selected_decisions if not already there
            if decision_name not in effective_selected_decisions:
                effective_selected_decisions.append(decision_name)
            
            # Store info for display
            if decision_name == 'disclose_income':
                saved_config_info['disclose_income'] = config.get('income_mode', 
                    config.get('params', {}).get('income_mode', 'Unknown'))
            elif decision_name == 'donation_default':
                pop_mode = config.get('population_mode', 'Unknown')
                inc_mode = config.get('params', {}).get('income_mode', 
                    config.get('income_spec_mode', 'Unknown'))
                saved_config_info['donation_default'] = f"{pop_mode} + {inc_mode}"
            else:
                # Generic handling for other decisions
                saved_config_info[decision_name] = "custom config"
    
    # Also check legacy storage for backward compatibility
    if hasattr(st.session_state, 'selected_disclose_income_config'):
        legacy_config = st.session_state.selected_disclose_income_config
        if legacy_config and legacy_config.get('source') != 'auto_implied_single_config':
            if 'disclose_income' not in effective_selected_decisions:
                effective_selected_decisions.append('disclose_income')
            if 'disclose_income' not in saved_config_info:
                saved_config_info['disclose_income'] = legacy_config.get('income_mode', 
                    legacy_config.get('params', {}).get('income_mode', 'Unknown'))
    
    if hasattr(st.session_state, 'selected_donation_config'):
        legacy_config = st.session_state.selected_donation_config
        if legacy_config and legacy_config.get('source') != 'auto_implied_single_config':
            if 'donation_default' not in effective_selected_decisions:
                effective_selected_decisions.append('donation_default')
            if 'donation_default' not in saved_config_info:
                pop_mode = legacy_config.get('population_mode', 'Unknown')
                inc_mode = legacy_config.get('donation_income_mode', 
                    legacy_config.get('income_spec_mode', 'Unknown'))
                saved_config_info['donation_default'] = f"{pop_mode} + {inc_mode}"
    
    unselected_decisions = [d for d in ALL_DECISIONS if d not in effective_selected_decisions]
    
    # ========================================================================
    # DISPLAY INFO ABOUT SAVED CONFIGS
    # ========================================================================
    
    # Show info about saved configs being used
    if 'donation_default' in saved_config_info:
        st.info(f"🎯 **Donation Default** will use saved config: {saved_config_info['donation_default']}")
        st.session_state._using_selected_config = True
    
    if 'disclose_income' in saved_config_info:
        di_saved_mode = saved_config_info['disclose_income']
        st.info(f"📋 **Disclose Income** will use saved config: {di_saved_mode}")
        # Update di_income_mode to match saved config for consistency
        if di_saved_mode and di_saved_mode != 'Unknown':
            st.session_state.di_income_mode = di_saved_mode
    elif 'disclose_income' in effective_selected_decisions:
        # Decision is selected but no saved config - show current mode
        di_mode = st.session_state.get('di_income_mode', 'Categorical only')
        st.info(f"📋 **Disclose Income** will use: {di_mode}")
    
    # Validate income mode compatibility and show warning if mismatched
    # (This is informational only - we proceed with user's explicit settings)
    # FIX: using_selected_donation_config is now determined by presence in saved_config_info
    using_selected_donation_config = 'donation_default' in saved_config_info
    _validate_income_mode_compatibility(effective_selected_decisions, unselected_decisions, using_selected_donation_config)
    
    # Create appropriate spinner message
    if len(effective_selected_decisions) == 0:
        spinner_msg = f"Running complete simulation: All {len(ALL_DECISIONS)} decisions with default values..."
    elif len(unselected_decisions) == 0:
        spinner_msg = f"Running complete simulation: All {len(effective_selected_decisions)} decisions with custom parameters..."
    else:
        spinner_msg = f"Running complete simulation: {len(effective_selected_decisions)} custom + {len(unselected_decisions)} default decisions..."
    
    with st.spinner(spinner_msg):
        try:
            # Store original selected decisions
            original_decisions = st.session_state.decision_params.selected_decisions.copy()
            
            # Set to run ALL decisions (this ensures complete simulation)
            st.session_state.decision_params.selected_decisions = ALL_DECISIONS
            
            # Store metadata about which decisions use custom vs default parameters
            # FIX: Use effective_selected_decisions which includes decisions with saved configs
            st.session_state.custom_decisions = effective_selected_decisions
            st.session_state.default_decisions = unselected_decisions
            
            # Run simulation with all decisions - check if Monte Carlo or Single Run mode
            if st.session_state.sim_params.simulation_mode == "Monte-Carlo Study":
                # Run Monte Carlo study
                from app.simulation import run_monte_carlo_study
                mc_summary, mc_detailed, output_log = run_monte_carlo_study()
                if mc_summary is not None:
                    st.session_state.mc_results = {
                        'summary': mc_summary,
                        'detailed': mc_detailed,
                        'log': output_log
                    }
                    st.session_state.simulation_results = None
                    st.success(f"✅ Monte Carlo complete simulation finished!")
                    
                    # Show Monte Carlo summary
                    st.info(f"📊 Completed {st.session_state.n_runs} Monte Carlo runs with {len(ALL_DECISIONS)} decisions")
                    
                    # Navigate to results page
                    st.session_state.page = 'results'
                    st.info("🔄 Redirecting to Results page to view full Monte Carlo analysis...")
                    
                    # Restore state before rerun
                    st.session_state.decision_params.selected_decisions = original_decisions
                    
                    st.rerun()
                else:
                    st.error("❌ Monte Carlo simulation returned no results")
            else:
                # Run single simulation
                run_simulation_from_sidebar()
            
            # Restore original selected decisions
            st.session_state.decision_params.selected_decisions = original_decisions
            
            # Show completion message
            if st.session_state.simulation_results:
                st.success(f"✅ Complete simulation finished!")
                
                # Provide clear messaging based on configuration
                # Use effective_selected_decisions to include decisions with saved configs
                if len(effective_selected_decisions) == 0:
                    st.info(f"🔧 **All {len(ALL_DECISIONS)} decisions** used default values")
                elif len(unselected_decisions) == 0:
                    st.info(f"📊 **All {len(effective_selected_decisions)} decisions** used your custom parameters")
                else:
                    st.info(f"📊 **{len(effective_selected_decisions)} decisions** used your custom parameters")
                    st.info(f"🔧 **{len(unselected_decisions)} decisions** used default values")
                
                # Show preview
                results = next(iter(st.session_state.simulation_results.values()))
                if results is not None and not results.empty:
                    st.metric("Total Agents Simulated", f"{len(results):,}")
                    
        except Exception as e:
            st.error(f"❌ Error running complete simulation: {str(e)}")
            import traceback
            st.text(traceback.format_exc())


# ==================== CONFIGURATION SELECTION SYSTEM ====================
# NOTE: These functions are kept for backwards compatibility.
# New code should use the unified save_decision_config() function from the
# UNIFIED DECISION CONFIGURATION SYSTEM section below.

def save_selected_configuration(result_key, result_df):
    """
    Save the selected donation configuration for later use in combined simulations.
    
    DEPRECATED: This is a wrapper for backwards compatibility.
    New code should use: save_decision_config('donation_default', result_key, result_df, params, metrics, extra_data)
    """
    # Extract configuration details from the result key
    config_details = extract_configuration_details(result_key)
    
    # Get current coefficient values from session state
    coefficients = get_current_coefficients()
    
    # Get current stochastic parameters
    stochastic_params = get_current_stochastic_params()
    
    # Calculate key metrics from the result
    metrics = calculate_result_metrics(result_df)
    
    # Build params dict for unified system
    params = {
        'coefficients': coefficients,
        'stochastic_params': stochastic_params,
        'income_mode': config_details['income_spec_mode']
    }
    
    extra_data = {
        'population_mode': config_details['population_mode'],
        'income_spec_mode': config_details['income_spec_mode']
    }
    
    # Use unified save function
    success, config, error_info = save_decision_config(
        'donation_default', result_key, result_df, params, metrics, extra_data
    )
    
    if success:
        return config
    else:
        # For backwards compatibility, still save to legacy storage even if seed mismatch
        # The UI will show the error, but existing code expecting a config will still work
        # Get seed used during the original run
        if st.session_state.sim_params.simulation_mode == "Single Run":
            original_seed = st.session_state.get('seed_input', st.session_state.seed)
        else:
            original_seed = st.session_state.get('base_seed_input', st.session_state.base_seed)
        
        legacy_config = {
            'result_key': result_key,
            'population_mode': config_details['population_mode'],
            'donation_income_mode': config_details['income_spec_mode'],
            'income_spec_mode': config_details['income_spec_mode'],
            'coefficients': coefficients,
            'stochastic_params': stochastic_params,
            'metrics': metrics,
            'selected_timestamp': datetime.now(),
            'total_agents': len(result_df),
            'source': 'individual_donation_run',
            'original_seed': original_seed,
            'original_n_agents': st.session_state.n_agents
        }
        
        st.session_state.selected_donation_config = legacy_config
        
        # Sync final_donation_rate_default_value
        mean_donation = metrics['mean_donation']
        st.session_state.final_donation_rate_default_value = mean_donation
        if '_persistent_defaults' not in st.session_state:
            st.session_state._persistent_defaults = {}
        st.session_state._persistent_defaults['final_donation_rate_default_value'] = mean_donation
        
        return legacy_config


def extract_configuration_details(result_key):
    """Extract population and income mode from result key"""
    
    # Population mode detection
    if 'copula' in result_key:
        population_mode = 'Copula (synthetic)'
    elif 'research_spec' in result_key or 'documentation' in result_key:
        population_mode = 'Research Specification'
    elif 'baseline' in result_key:
        population_mode = 'Research Baseline'
    else:
        # For single-mode results, use current session state
        population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
    
    # Income mode detection
    if 'categorical' in result_key:
        income_spec_mode = 'categorical only'
    elif 'continuous' in result_key:
        income_spec_mode = 'continuous only'
    else:
        # For single-mode results, use current session state
        income_spec_mode = st.session_state.get('income_spec_mode', 'categorical only')
    
    return {
        'population_mode': population_mode,
        'income_spec_mode': income_spec_mode
    }


def get_current_coefficients():
    """Collect all current coefficient values from session state
    
    IMPORTANT: Ensures coefficients are loaded from YAML first.
    YAML is the SINGLE source of truth - no fallback values.
    """
    # Ensure coefficients are loaded from YAML
    from app.models import load_donation_coefficients_from_yaml
    if 'donation_coeff_intercept' not in st.session_state:
        load_donation_coefficients_from_yaml()
    
    # Return coefficients from session state - NO FALLBACK VALUES
    return {
        'intercept': st.session_state.donation_coeff_intercept,
        'beta_group': {
            'MidSub': st.session_state.donation_coeff_midsub,
            'NoSub': st.session_state.donation_coeff_nosub,
            'FullSub': st.session_state.donation_coeff_fullsub
        },
        'beta_income_q': {
            'Q1': st.session_state.donation_coeff_q1,
            'Q2': st.session_state.donation_coeff_q2,
            'Q3': st.session_state.donation_coeff_q3,
            'Q4': st.session_state.get('donation_coeff_q4', 0.0),
            'Q5': st.session_state.get('donation_coeff_q5', st.session_state.get('donation_coeff_q45', 0.0))  # Support both Q5 and legacy Q4_Q5
        },
        'beta_income_linear': st.session_state.donation_coeff_linear,
        'beta_study': {
            'Incoming': st.session_state.donation_coeff_incoming,
            'Law5yr': st.session_state.donation_coeff_law,
            'UG3yr': st.session_state.donation_coeff_ug,
            'Grad2yr': st.session_state.donation_coeff_grad
        },
        'beta_hh': st.session_state.donation_coeff_hh
    }


def get_current_stochastic_params():
    """Collect current stochastic parameters from session state"""
    return {
        'stochastic': {
            'sigma_value': st.session_state.get('sigma_value_ui', 9.8995),
            'sigma_coefficient': st.session_state.get('sigma_coefficient', 1.0),
            'sigma_in_copula': st.session_state.get('sigma_in_copula', False),
            'sigma_in_research': st.session_state.get('sigma_in_research', True)
        },
        'anchor_weights': {
            'observed': st.session_state.get('anchor_observed_weight', 0.75),
            'predicted': 1 - st.session_state.get('anchor_observed_weight', 0.75)
        }
    }


def calculate_result_metrics(result_df):
    """Calculate key metrics from result DataFrame - always uses truncated donation_default"""
    
    # Always use truncated donation_default for consistency
    donation_col = 'donation_default'
    
    metrics = {
        'mean_donation': result_df[donation_col].mean(),
        'std_donation': result_df[donation_col].std(),
        'median_donation': result_df[donation_col].median(),
        'min_donation': result_df[donation_col].min(),
        'max_donation': result_df[donation_col].max(),
        'q25_donation': result_df[donation_col].quantile(0.25),
        'q75_donation': result_df[donation_col].quantile(0.75),
        'donation_column_used': donation_col
    }
    
    return metrics


def format_result_name(result_key):
    """Format result key into human-readable name"""
    
    name_mapping = {
        'copula_categorical': '🔬 Copula + Categorical Income',
        'copula_continuous': '🔬 Copula + Continuous Income',
        'research_spec_categorical': '📊 Research Spec + Categorical Income',
        'research_spec_continuous': '📊 Research Spec + Continuous Income',
        'research_baseline_categorical': '📈 Research Baseline + Categorical Income',
        'research_baseline_continuous': '📈 Research Baseline + Continuous Income',
        'categorical': '💰 Categorical Income Mode',
        'continuous': '📈 Continuous Income Mode',
        'copula': '🔬 Copula Population',
        'documentation': '📊 Research Specification',
        'baseline': '📈 Research Baseline'
    }
    
    return name_mapping.get(result_key, f"📊 {result_key.replace('_', ' ').title()}")


def is_configuration_selected(result_key):
    """Check if a specific configuration is currently selected"""
    
    if not hasattr(st.session_state, 'selected_donation_config'):
        return False
    
    return st.session_state.selected_donation_config.get('result_key') == result_key


def clear_selected_configuration():
    """Clear the currently selected configuration"""
    
    if hasattr(st.session_state, 'selected_donation_config'):
        del st.session_state.selected_donation_config
    
    # Reset final_donation_rate_default_value back to system default (10%)
    st.session_state.final_donation_rate_default_value = 0.10
    if '_persistent_defaults' in st.session_state:
        st.session_state._persistent_defaults['final_donation_rate_default_value'] = 0.10


# ==================== DISCLOSE INCOME CONFIGURATION SELECTION SYSTEM ====================
# NOTE: These functions are kept for backwards compatibility.
# New code should use the unified save_decision_config() function.

def save_disclose_income_configuration(result_key, result_df):
    """
    Save the selected disclose income configuration for later use in combined simulations.
    
    DEPRECATED: This is a wrapper for backwards compatibility.
    New code should use: save_decision_config('disclose_income', result_key, result_df, params, metrics, extra_data)
    """
    # Extract configuration details from the result key
    config_details = extract_disclose_income_configuration_details(result_key)
    
    # Get current disclose income parameters from session state
    params = get_current_disclose_income_params()
    
    # Calculate key metrics from the result
    metrics = calculate_disclose_income_metrics(result_df)
    
    extra_data = {
        'income_mode': config_details['income_mode']
    }
    
    # Use unified save function
    success, config, error_info = save_decision_config(
        'disclose_income', result_key, result_df, params, metrics, extra_data
    )
    
    if success:
        return config
    else:
        # For backwards compatibility, still save to legacy storage even if seed mismatch
        # Get seed used during the original run
        if st.session_state.sim_params.simulation_mode == "Single Run":
            original_seed = st.session_state.get('seed_input', st.session_state.seed)
        else:
            original_seed = st.session_state.get('base_seed_input', st.session_state.base_seed)
        
        legacy_config = {
            'result_key': result_key,
            'income_mode': config_details['income_mode'],
            'params': params,
            'metrics': metrics,
            'selected_timestamp': datetime.now(),
            'total_agents': len(result_df),
            'source': 'individual_disclose_income_run',
            'original_seed': original_seed,
            'original_n_agents': st.session_state.n_agents
        }
        
        # Store in legacy session state
        st.session_state.selected_disclose_income_config = legacy_config
        
        return legacy_config


def extract_disclose_income_configuration_details(result_key):
    """Extract income mode from result key for disclose income"""
    
    # Income mode detection
    if 'categorical' in result_key.lower():
        income_mode = 'Categorical only'
    elif 'continuous' in result_key.lower():
        income_mode = 'Continuous only'
    else:
        # Use current session state value
        income_mode = st.session_state.get('di_income_mode', 'Categorical only')
    
    return {
        'income_mode': income_mode
    }


def get_current_disclose_income_params():
    """Collect current disclose income parameters from session state"""
    return {
        'intercept': st.session_state.get('di_intercept', 0.75),
        'income_mode': st.session_state.get('di_income_mode', 'Categorical only'),
        'anchor_weights': {
            'observed_prosocial': st.session_state.get('di_wopb', 0.25),
            'prosocial_weight': st.session_state.get('di_wpb', 0.50)
        },
        'stochastic': {
            'sigma_enabled': st.session_state.get('di_sigma_enabled', False),
            'sigma_in_copula': st.session_state.get('di_sigma_in_copula', False),
            'scale_factor': st.session_state.get('di_scale_factor', 1.0),
            'sigma_strategy': st.session_state.get('di_sigma_strategy', 'overall'),
            'quintile_scale_factors': st.session_state.get('di_quintile_scale_factors', {})
        }
    }


def calculate_disclose_income_metrics(result_df):
    """Calculate key metrics from disclose income result DataFrame"""
    
    metrics = {}
    
    # Calculate Y/N rates
    if 'disclose_income' in result_df.columns:
        total = len(result_df)
        y_count = (result_df['disclose_income'] == 'Y').sum()
        n_count = (result_df['disclose_income'] == 'N').sum()
        metrics['y_rate'] = y_count / total if total > 0 else 0
        metrics['n_rate'] = n_count / total if total > 0 else 0
        metrics['y_count'] = int(y_count)
        metrics['n_count'] = int(n_count)
    
    # Calculate raw value statistics if available
    if 'disclose_income_raw' in result_df.columns:
        raw_values = result_df['disclose_income_raw'].dropna()
        metrics['raw_mean'] = float(raw_values.mean())
        metrics['raw_std'] = float(raw_values.std())
        metrics['raw_median'] = float(raw_values.median())
        metrics['raw_min'] = float(raw_values.min())
        metrics['raw_max'] = float(raw_values.max())
        metrics['raw_q25'] = float(raw_values.quantile(0.25))
        metrics['raw_q75'] = float(raw_values.quantile(0.75))
    
    # Calculate DI_i statistics if available (pre-stochastic value)
    if 'disclose_income_di' in result_df.columns:
        di_values = result_df['disclose_income_di'].dropna()
        metrics['di_mean'] = float(di_values.mean())
        metrics['di_std'] = float(di_values.std())
    
    return metrics


def is_disclose_income_configuration_selected(result_key):
    """Check if a specific disclose income configuration is currently selected"""
    
    if not hasattr(st.session_state, 'selected_disclose_income_config'):
        return False
    
    return st.session_state.selected_disclose_income_config.get('result_key') == result_key


def clear_disclose_income_configuration():
    """Clear the currently selected disclose income configuration"""
    
    if hasattr(st.session_state, 'selected_disclose_income_config'):
        del st.session_state.selected_disclose_income_config


# ==================== UNIFIED DECISION CONFIGURATION SYSTEM ====================
# This unified system replaces the fragmented per-decision config storage.
# All decision configs are now stored in a single dict: selected_decision_configs
# This enables seed matching validation and easy extensibility for new decisions.

def get_selected_decision_configs():
    """Get the unified decision configs dictionary, initializing if needed."""
    if 'selected_decision_configs' not in st.session_state:
        st.session_state.selected_decision_configs = {}
    return st.session_state.selected_decision_configs


def validate_seed_consistency(new_decision_name, new_seed, new_n_agents):
    """
    Check if new config's seed/n_agents matches existing configs.
    
    CRITICAL: Only validates against EXPLICITLY saved configs, not auto-implied ones.
    Auto-implied configs are for UI convenience only and should not block saving.
    
    Args:
        new_decision_name: Name of the decision being saved
        new_seed: Seed used for the new config
        new_n_agents: Number of agents used for the new config
    
    Returns:
        tuple: (is_valid, existing_seed, existing_n_agents, conflicting_decision)
            - is_valid: True if seed matches or no existing configs
            - existing_seed: The seed from existing configs (if any)
            - existing_n_agents: The n_agents from existing configs (if any)
            - conflicting_decision: Name of the decision with mismatched seed (if any)
    """
    configs = get_selected_decision_configs()
    
    for decision_name, config in configs.items():
        # Skip if checking against itself (re-saving same decision)
        if decision_name == new_decision_name:
            continue
        
        # Skip auto-implied configs - they're not real user selections
        # Only validate against explicitly saved configs
        if config.get('source') == 'auto_implied_single_config':
            continue
            
        existing_seed = config.get('original_seed')
        existing_n_agents = config.get('original_n_agents')
        
        if existing_seed != new_seed or existing_n_agents != new_n_agents:
            return (False, existing_seed, existing_n_agents, decision_name)
    
    return (True, new_seed, new_n_agents, None)


def save_decision_config(decision_name, result_key, result_df, params, metrics=None, extra_data=None):
    """
    Unified function to save a decision configuration.
    
    This is the standard way to save any decision config for use in combined simulations.
    All configs are validated for seed consistency before saving.
    
    Args:
        decision_name: Name of the decision (e.g., 'donation_default', 'disclose_income')
        result_key: Key identifying the result configuration
        result_df: DataFrame containing the simulation results
        params: Dict of decision-specific parameters (coefficients, weights, etc.)
        metrics: Optional dict of calculated metrics. If None, will use basic metrics.
        extra_data: Optional dict of additional data to store (e.g., for backwards compat)
    
    Returns:
        tuple: (success: bool, config: dict or None, error_info: dict or None)
            - success: True if config was saved successfully
            - config: The saved config dict (if successful)
            - error_info: Dict with seed mismatch details (if failed)
    """
    # Get seed used during the original run
    if hasattr(st.session_state, 'sim_params') and st.session_state.sim_params.simulation_mode == "Single Run":
        original_seed = st.session_state.get('seed_input', st.session_state.get('seed', 42))
    else:
        original_seed = st.session_state.get('base_seed_input', st.session_state.get('base_seed', 42))
    
    original_n_agents = st.session_state.get('n_agents', 1000)
    
    # Validate seed consistency with existing configs
    is_valid, existing_seed, existing_n_agents, conflicting_decision = validate_seed_consistency(
        decision_name, original_seed, original_n_agents
    )
    
    if not is_valid:
        return (False, None, {
            'new_seed': original_seed,
            'new_n_agents': original_n_agents,
            'existing_seed': existing_seed,
            'existing_n_agents': existing_n_agents,
            'conflicting_decision': conflicting_decision
        })
    
    # Build the config object
    config = {
        'result_key': result_key,
        'params': params,
        'metrics': metrics or {},
        'selected_timestamp': datetime.now(),
        'total_agents': len(result_df) if result_df is not None else 0,
        'source': f'individual_{decision_name}_run',
        'original_seed': original_seed,
        'original_n_agents': original_n_agents
    }
    
    # Merge in any extra data (for backwards compatibility)
    if extra_data:
        config.update(extra_data)
    
    # FIX: For disclose_income, ensure population_mode is stored (extract from result_key if needed)
    # This ensures the config can be properly synced when running full simulation
    if decision_name == 'disclose_income' and 'population_mode' not in config:
        config_details = extract_configuration_details(result_key)
        config['population_mode'] = config_details['population_mode']
    
    # Store in unified configs dict
    configs = get_selected_decision_configs()
    configs[decision_name] = config
    
    # Also maintain backwards compatibility with legacy storage
    _sync_to_legacy_storage(decision_name, config)
    
    return (True, config, None)


def _sync_to_legacy_storage(decision_name, config):
    """
    Sync unified config to legacy storage for backwards compatibility.
    
    This ensures existing code that reads from the old storage locations
    continues to work during the transition period.
    """
    if decision_name == 'donation_default':
        # Build legacy format donation config
        legacy_config = {
            'result_key': config['result_key'],
            'population_mode': config.get('population_mode', st.session_state.get('population_mode', 'Copula (synthetic)')),
            'donation_income_mode': config['params'].get('income_mode', config.get('income_spec_mode', 'categorical only')),
            'income_spec_mode': config['params'].get('income_mode', config.get('income_spec_mode', 'categorical only')),
            'coefficients': config['params'].get('coefficients', config.get('coefficients', {})),
            'stochastic_params': config['params'].get('stochastic_params', config.get('stochastic_params', {})),
            'metrics': config['metrics'],
            'selected_timestamp': config['selected_timestamp'],
            'total_agents': config['total_agents'],
            'source': config['source'],
            'original_seed': config['original_seed'],
            'original_n_agents': config['original_n_agents']
        }
        st.session_state.selected_donation_config = legacy_config
        
        # Sync final_donation_rate_default_value
        if 'mean_donation' in config['metrics']:
            mean_donation = config['metrics']['mean_donation']
            st.session_state.final_donation_rate_default_value = mean_donation
            if '_persistent_defaults' not in st.session_state:
                st.session_state._persistent_defaults = {}
            st.session_state._persistent_defaults['final_donation_rate_default_value'] = mean_donation
            
    elif decision_name == 'disclose_income':
        # Build legacy format disclose_income config
        # FIX: income_mode is in config directly (from extra_data), not in config['params']
        # FIX: Also extract and store population_mode from result_key for proper sync
        
        # Extract population_mode from result_key if not explicitly stored
        population_mode = config.get('population_mode')
        if not population_mode:
            config_details = extract_configuration_details(config['result_key'])
            population_mode = config_details['population_mode']
        
        legacy_config = {
            'result_key': config['result_key'],
            'population_mode': population_mode,  # ADD: Store population_mode for proper sync
            'income_mode': config.get('income_mode', config['params'].get('income_mode', 'Categorical only')),
            'params': config['params'],
            'metrics': config['metrics'],
            'selected_timestamp': config['selected_timestamp'],
            'total_agents': config['total_agents'],
            'source': config['source'],
            'original_seed': config['original_seed'],
            'original_n_agents': config['original_n_agents']
        }
        st.session_state.selected_disclose_income_config = legacy_config


def get_decision_config(decision_name):
    """
    Get the saved configuration for a specific decision.
    
    Args:
        decision_name: Name of the decision
    
    Returns:
        dict or None: The config dict if exists, None otherwise
    """
    configs = get_selected_decision_configs()
    return configs.get(decision_name)


def has_decision_config(decision_name):
    """
    Check if a decision has a saved configuration.
    
    Args:
        decision_name: Name of the decision
    
    Returns:
        bool: True if config exists
    """
    return get_decision_config(decision_name) is not None


def is_decision_config_selected(decision_name, result_key):
    """
    Check if a specific result_key is the currently selected config for a decision.
    
    Args:
        decision_name: Name of the decision
        result_key: The result key to check
    
    Returns:
        bool: True if this result_key is currently selected
    """
    config = get_decision_config(decision_name)
    if config is None:
        return False
    return config.get('result_key') == result_key


def clear_decision_config(decision_name):
    """
    Clear the saved configuration for a specific decision.
    
    Args:
        decision_name: Name of the decision to clear
    """
    configs = get_selected_decision_configs()
    if decision_name in configs:
        del configs[decision_name]
    
    # Also clear legacy storage
    if decision_name == 'donation_default':
        if hasattr(st.session_state, 'selected_donation_config'):
            del st.session_state.selected_donation_config
        # Reset final_donation_rate_default_value
        st.session_state.final_donation_rate_default_value = 0.10
        if '_persistent_defaults' in st.session_state:
            st.session_state._persistent_defaults['final_donation_rate_default_value'] = 0.10
    elif decision_name == 'disclose_income':
        if hasattr(st.session_state, 'selected_disclose_income_config'):
            del st.session_state.selected_disclose_income_config


def clear_all_decision_configs():
    """Clear all saved decision configurations."""
    st.session_state.selected_decision_configs = {}
    
    # Also clear legacy storage
    if hasattr(st.session_state, 'selected_donation_config'):
        del st.session_state.selected_donation_config
    if hasattr(st.session_state, 'selected_disclose_income_config'):
        del st.session_state.selected_disclose_income_config
    
    # Reset defaults
    st.session_state.final_donation_rate_default_value = 0.10
    if '_persistent_defaults' in st.session_state:
        st.session_state._persistent_defaults['final_donation_rate_default_value'] = 0.10


def get_simulation_seed_from_configs():
    """
    Get the seed to use for simulation from saved configs.
    
    If configs exist, returns their seed (all configs have matching seed).
    Otherwise returns the current session state seed.
    
    Returns:
        tuple: (seed, n_agents, source) where source is 'configs' or 'session_state'
    """
    configs = get_selected_decision_configs()
    
    if configs:
        # All configs have matching seed, so just get from first one
        first_config = next(iter(configs.values()))
        return (
            first_config['original_seed'],
            first_config['original_n_agents'],
            'configs'
        )
    
    # No saved configs, use session state
    if hasattr(st.session_state, 'sim_params') and st.session_state.sim_params.simulation_mode == "Single Run":
        seed = st.session_state.get('seed_input', st.session_state.get('seed', 42))
    else:
        seed = st.session_state.get('base_seed_input', st.session_state.get('base_seed', 42))
    
    n_agents = st.session_state.get('n_agents', 1000)
    
    return (seed, n_agents, 'session_state')


def get_all_saved_config_summary():
    """
    Get a summary of all saved decision configs.
    
    Returns:
        list of dicts: Summary info for each saved config
    """
    configs = get_selected_decision_configs()
    summaries = []
    
    for decision_name, config in configs.items():
        summaries.append({
            'decision_name': decision_name,
            'result_key': config.get('result_key', 'unknown'),
            'original_seed': config.get('original_seed'),
            'original_n_agents': config.get('original_n_agents'),
            'selected_timestamp': config.get('selected_timestamp'),
            'source': config.get('source')
        })
    
    return summaries


def migrate_legacy_configs_to_unified():
    """
    Migrate any existing legacy configs to the unified storage.
    
    This should be called at app startup to ensure backwards compatibility.
    Existing configs in the old format are copied to the new unified storage.
    """
    configs = get_selected_decision_configs()
    
    # Migrate donation_default if exists in legacy storage but not unified
    if 'donation_default' not in configs and hasattr(st.session_state, 'selected_donation_config'):
        legacy = st.session_state.selected_donation_config
        configs['donation_default'] = {
            'result_key': legacy.get('result_key', 'migrated'),
            'params': {
                'coefficients': legacy.get('coefficients', {}),
                'stochastic_params': legacy.get('stochastic_params', {}),
                'income_mode': legacy.get('donation_income_mode', legacy.get('income_spec_mode', 'categorical only'))
            },
            'metrics': legacy.get('metrics', {}),
            'selected_timestamp': legacy.get('selected_timestamp', datetime.now()),
            'total_agents': legacy.get('total_agents', 0),
            'source': legacy.get('source', 'migrated_legacy'),
            'original_seed': legacy.get('original_seed', st.session_state.get('seed', 42)),
            'original_n_agents': legacy.get('original_n_agents', st.session_state.get('n_agents', 1000)),
            # Keep extra fields for compatibility
            'population_mode': legacy.get('population_mode'),
            'income_spec_mode': legacy.get('income_spec_mode')
        }
    
    # Migrate disclose_income if exists in legacy storage but not unified
    if 'disclose_income' not in configs and hasattr(st.session_state, 'selected_disclose_income_config'):
        legacy = st.session_state.selected_disclose_income_config
        configs['disclose_income'] = {
            'result_key': legacy.get('result_key', 'migrated'),
            'params': legacy.get('params', {}),
            'metrics': legacy.get('metrics', {}),
            'selected_timestamp': legacy.get('selected_timestamp', datetime.now()),
            'total_agents': legacy.get('total_agents', 0),
            'source': legacy.get('source', 'migrated_legacy'),
            'original_seed': legacy.get('original_seed', st.session_state.get('seed', 42)),
            'original_n_agents': legacy.get('original_n_agents', st.session_state.get('n_agents', 1000)),
            # Keep extra fields for compatibility
            'income_mode': legacy.get('income_mode')
        }
