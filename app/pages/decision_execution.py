# app/pages/decision_execution.py
"""
Decision execution functions for running individual and combined simulations.
"""
import streamlit as st
import pandas as pd
from app.simulation import run_full_simulation
from app.models import ALL_DECISIONS, DONATION_SIGMA_OVERALL

# The 13 decisions' default values and default descriptions are pure contract
# data (src/contract/defaults.py). Re-exported here because the screens, the
# seam and the tests have always imported them from this module.
from src.contract.defaults import (  # noqa: F401
    DEFAULT_DECISION_DESCRIPTIONS,
    DEFAULT_DECISION_VALUES,
)

# The saved-decision-configuration store lives in app/state/saved_configs.py.
# Re-exported here so every existing caller keeps importing it from this module.
from app.state.saved_configs import (  # noqa: F401
    extract_disclose_income_configuration_details,
    get_current_disclose_income_params,
    calculate_disclose_income_metrics,
    extract_disclose_documents_configuration_details,
    get_current_disclose_documents_params,
    calculate_disclose_documents_metrics,
    RTD_CONFIG_MECHANISMS,
    extract_rejected_transaction_configuration_details,
    get_current_rejected_transaction_params,
    calculate_rejected_transaction_metrics,
    get_selected_decision_configs,
    validate_seed_consistency,
    get_decision_result_columns,
    hash_result_columns,
    save_decision_config,
    get_decision_config,
    has_decision_config,
    is_decision_config_selected,
    clear_decision_config,
    get_simulation_seed_from_configs,
    get_all_saved_config_summary,
)


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

    # Disclose documents income modes: "Compare both" generates 2, others generate 1
    dd_income_mode = st.session_state.get('dd_income_mode', 'Categorical only')
    dd_income_count = 2 if ('compare' in str(dd_income_mode).lower() or 'both' in str(dd_income_mode).lower()) else 1

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
        # Check if user has a saved disclose_income config
        if 'disclose_income' in configs:
            di_config = configs['disclose_income']
            has_disclose_income_config = True
            di_saved_mode = di_config.get('params', {}).get('income_mode', 
                di_config.get('income_mode', 'Unknown'))
        
        # If multiple configs and no saved selection, add to blocking issues
        if di_total_configs > 1 and not has_disclose_income_config:
            blocking_issues.append({
                'decision': 'disclose_income',
                'block_type': 'disclose_income',
                'config_count': di_total_configs,
                'reason': f"Disclose Income has {di_total_configs} configurations (population: {population_count}, income: {di_income_count}) - please run disclose_income only and select one"
            })
    
    # ========================================================================
    # CHECK DISCLOSE_DOCUMENTS (only if selected) - mirrors disclose_income
    # ========================================================================

    disclose_documents_selected = 'disclose_documents' in selected_decisions
    dd_total_configs = population_count * dd_income_count
    has_disclose_documents_config = False
    dd_saved_mode = None

    if disclose_documents_selected:
        if 'disclose_documents' in configs:
            ddc = configs['disclose_documents']
            has_disclose_documents_config = True
            dd_saved_mode = ddc.get('params', {}).get('income_mode',
                ddc.get('income_mode', 'Unknown'))

        if dd_total_configs > 1 and not has_disclose_documents_config:
            blocking_issues.append({
                'decision': 'disclose_documents',
                'block_type': 'disclose_documents',
                'config_count': dd_total_configs,
                'reason': f"Disclose Documents has {dd_total_configs} configurations (population: {population_count}, income: {dd_income_count}) - please run disclose_documents only and select one"
            })

    # ========================================================================
    # CHECK REJECTED_TRANSACTION_DEFAULTS (only if selected) - mirrors disclose_income
    # ========================================================================

    rtd_income_mode = st.session_state.get('rtd_income_mode', 'Continuous only')
    rtd_income_count = 2 if ('compare' in str(rtd_income_mode).lower() or 'both' in str(rtd_income_mode).lower()) else 1
    rtd_selected = 'rejected_transaction_defaults' in selected_decisions
    rtd_total_configs = population_count * rtd_income_count
    has_rtd_config = False
    rtd_saved_info = None

    if rtd_selected:
        if 'rejected_transaction_defaults' in configs:
            # R13: every stored config is an explicit "Use This Config" selection.
            rc = configs['rejected_transaction_defaults']
            has_rtd_config = True
            rtd_saved_info = (f"{rc.get('population_mode', 'Unknown')} + "
                              f"{rc.get('income_mode', rc.get('params', {}).get('income_mode', 'Unknown'))}")

        if rtd_total_configs > 1 and not has_rtd_config:
            blocking_issues.append({
                'decision': 'rejected_transaction_defaults',
                'block_type': 'rejected_transaction_defaults',
                'config_count': rtd_total_configs,
                'reason': f"Rejected Transaction Defaults has {rtd_total_configs} configurations (population: {population_count}, income: {rtd_income_count}) - please run rejected_transaction_defaults only and select one"
            })

    # ========================================================================
    # CHECK DONATION_DEFAULT (only if selected)
    # ========================================================================

    donation_default_selected = 'donation_default' in selected_decisions
    donation_total_configs = population_count * donation_income_count
    has_donation_config = False
    donation_saved_info = None
    
    if donation_default_selected:
        # Check if user has a saved donation_default config
        if 'donation_default' in configs:
            config = configs['donation_default']
            has_donation_config = True
            pop_mode = config.get('population_mode', st.session_state.get('population_mode', 'Unknown'))
            inc_mode = config.get('donation_income_mode', config.get('income_spec_mode', 'Unknown'))
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

    # Show disclose_documents config if selected and saved
    if has_disclose_documents_config and dd_saved_mode:
        config_parts.append(f"Disclose Documents: {dd_saved_mode}")

    # Show rejected_transaction_defaults config if selected and saved
    if has_rtd_config and rtd_saved_info:
        config_parts.append(f"Rejected Transaction Defaults: {rtd_saved_info}")

    # Determine total config count for display
    total_configs = max(di_total_configs if disclose_income_selected else 1,
                       donation_total_configs if donation_default_selected else 1,
                       dd_total_configs if disclose_documents_selected else 1,
                       rtd_total_configs if rtd_selected else 1)
    
    if config_parts:
        return (True, f"Using saved configuration(s): {', '.join(config_parts)}", total_configs, None, [])
    elif not disclose_income_selected and not donation_default_selected:
        return (True, "Using default values for all decisions", 1, None, [])
    else:
        return (True, "Single configuration", 1, None, [])


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
            # The single-issue wording follows the ONE blocking issue's own type (the
            # legacy block_type is the first issue's type; they agree, but the issue is
            # the source of truth). Each type has its own heading; the donation heading
            # is used for the donation block only, never as a catch-all.
            if len(blocking_issues) == 1:
                block_type = blocking_issues[0]['block_type']
            # Disabled button with explanation
            help_text = (f"{len(blocking_issues)} configuration issue(s) detected" if len(blocking_issues) > 1
                         else {"disclose_income": "Disclose Income is in Compare mode",
                               "disclose_documents": "Select a Disclose Documents configuration first",
                               "rejected_transaction_defaults": "Select a Rejected Transaction Defaults configuration first",
                               }.get(block_type, "Multiple configurations detected - select one first"))
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
                    elif issue['block_type'] == "disclose_documents":
                        st.warning(f"""
**Issue {i}: Disclose Documents**

{issue['reason']}

**Action Required:**
1. Go to the **Disclose Documents** tab
2. Run **disclose_documents only** and select one configuration
3. Or change to **"Categorical only"** or **"Continuous only"** mode
                        """)
                    elif issue['block_type'] == "rejected_transaction_defaults":
                        st.warning(f"""
**Issue {i}: Rejected Transaction Defaults**

{issue['reason']}

**Action Required:**
1. Go to the **Rejected Transaction Defaults** tab
2. Run **rejected_transaction_defaults only** and select one configuration ("Use This Config")
3. Or change its Income Specification to **"Categorical only"** or **"Continuous only"**
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
                elif block_type == "disclose_documents":
                    st.warning(f"""
⚠️ **Disclose Documents Configuration Required**

{reason}

**Action Required:**

1. Go to the **Disclose Documents** tab
2. Run **disclose_documents only** and select one configuration (or change "Income Specification for Disclosure Model" from "Compare both" to **"Categorical only"** or **"Continuous only"**)
3. Return here and click **Run Complete Simulation**

This ensures all decisions produce a single result set.
                    """)
                elif block_type == "rejected_transaction_defaults":
                    st.warning(f"""
⚠️ **Rejected Transaction Defaults Configuration Required**

{reason}

**Action Required:**

1. Go to the **Rejected Transaction Defaults** tab
2. Run **rejected_transaction_defaults only** and select one configuration ("Use This Config"),
   or change its Income Specification to **"Categorical only"** / **"Continuous only"**
3. Return here and click **Run Complete Simulation**

This ensures all decisions produce a single result set.
                    """)
                elif block_type == "donation_config":
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
                    # an unknown block type: say what is wrong without naming a decision
                    st.warning(f"""
⚠️ **Configuration Issue**

{reason}
                    """)
            
        else:
            # Enabled button - can proceed
            # Show info about the saved config if applicable
            if config_count > 1:
                dd_config = get_decision_config('donation_default')
                if dd_config:
                    st.success(f"✅ Using: {dd_config['population_mode']} + {dd_config.get('donation_income_mode', dd_config.get('income_spec_mode', 'unknown'))}")
            
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
            # This is needed because run_full_simulation() may call st.rerun()
            # BEFORE this function's restoration code can execute
            # NOTE: Only store selected_decisions - custom_decisions/default_decisions should
            # reflect the current run (needed for should_enable_selection())
            st.session_state._pending_decisions_restore = {
                'selected_decisions': original_decisions
            }
            
            # Clear any saved configuration to allow fresh run with current tab settings
            if decision_name in ("donation_default", "disclose_income", "disclose_documents"):
                if has_decision_config(decision_name):
                    st.info(f"🔄 Clearing saved {decision_name} configuration - will use current tab settings")
                    clear_decision_config(decision_name)
            
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
                # Run single simulation.
                # A successful run ends in st.rerun(), which raises a BaseException,
                # so nothing below runs on the success path - the state restoration
                # happens through _pending_decisions_restore instead.
                run_full_simulation()

            # Only reached when the run did not redirect: Monte Carlo returned no
            # results, or run_full_simulation() handled an error itself.
            # Restore ONLY selected_decisions - keep custom_decisions/default_decisions
            # as-is because they're needed for should_enable_selection() to show the
            # "Use This Config" button.
            st.session_state.decision_params.selected_decisions = original_decisions
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
    donation_config = get_decision_config('donation_default')
    if using_selected_donation_config and donation_config:
        donation_mode = donation_config.get('donation_income_mode', donation_config.get('income_spec_mode', 'categorical only'))
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
    # ADD ALL DECISIONS WITH SAVED CONFIGS TO EFFECTIVE LIST
    # Uses unified storage (selected_decision_configs) as single source of truth.
    # ========================================================================
    
    configs = get_selected_decision_configs()
    for decision_name, config in configs.items():
        if decision_name not in effective_selected_decisions:
            effective_selected_decisions.append(decision_name)
        
        if decision_name == 'disclose_income':
            saved_config_info['disclose_income'] = config.get('income_mode', 
                config.get('params', {}).get('income_mode', 'Unknown'))
        elif decision_name == 'donation_default':
            pop_mode = config.get('population_mode', 'Unknown')
            inc_mode = config.get('donation_income_mode', config.get('income_spec_mode', 'Unknown'))
            saved_config_info['donation_default'] = f"{pop_mode} + {inc_mode}"
        elif decision_name == 'rejected_transaction_defaults':
            pop_mode = config.get('population_mode', 'Unknown')
            inc_mode = config.get('income_mode', config.get('params', {}).get('income_mode', 'Unknown'))
            saved_config_info['rejected_transaction_defaults'] = f"{pop_mode} + {inc_mode}"
        else:
            saved_config_info[decision_name] = "custom config"
    
    unselected_decisions = [d for d in ALL_DECISIONS if d not in effective_selected_decisions]
    
    # ========================================================================
    # DISPLAY INFO ABOUT SAVED CONFIGS
    # ========================================================================
    
    # Show info about saved configs being used
    if 'donation_default' in saved_config_info:
        st.info(f"🎯 **Donation Default** will use saved config: {saved_config_info['donation_default']}")
    
    if 'disclose_income' in saved_config_info:
        di_saved_mode = saved_config_info['disclose_income']
        st.info(f"📋 **Disclose Income** will use saved config: {di_saved_mode}")
        # R28: the run does not write di_income_mode (or any other di_/dd_/rtd_ key)
        # back into session state - the saved config carries the mode the run uses.
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

            # R27: a successful run ends in st.rerun() (a BaseException), so the
            # restoration has to be handed to run_full_simulation() the same way
            # run_individual_decision() does it.
            st.session_state._pending_decisions_restore = {
                'selected_decisions': original_decisions
            }

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
                    if hasattr(st.session_state, '_pending_decisions_restore'):
                        del st.session_state._pending_decisions_restore

                    st.rerun()
                else:
                    st.error("❌ Monte Carlo simulation returned no results")
            else:
                # Run single simulation.
                # A successful run ends in st.rerun(), which raises a BaseException,
                # so nothing below runs on the success path - the restoration happens
                # through _pending_decisions_restore instead.
                run_full_simulation()

            # Only reached when the run did not redirect: Monte Carlo returned no
            # results, or run_full_simulation() handled an error itself.
            st.session_state.decision_params.selected_decisions = original_decisions
            if hasattr(st.session_state, '_pending_decisions_restore'):
                del st.session_state._pending_decisions_restore

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
    
    return config


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
            'sigma_value': st.session_state.get('sigma_value_ui', DONATION_SIGMA_OVERALL),
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
    """Check if a specific donation_default configuration is currently selected."""
    return is_decision_config_selected('donation_default', result_key)


def clear_selected_configuration():
    """Clear the currently selected donation_default configuration."""
    clear_decision_config('donation_default')
