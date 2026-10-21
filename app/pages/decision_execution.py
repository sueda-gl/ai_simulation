# app/pages/decision_execution.py
"""
Decision execution functions for running individual and combined simulations.
"""
import streamlit as st
import pandas as pd
from src.contract.plan import RunRequest
from app.simulation import run_full_simulation, run_monte_carlo_study
from app.models import ALL_DECISIONS, DONATION_SIGMA_OVERALL

# The 13 decisions' default values and default descriptions are pure contract
# data (src/contract/defaults.py). Re-exported here because the screens, the
# seam and the tests have always imported them from this module.
from src.contract.defaults import (  # noqa: F401
    DEFAULT_DECISION_DESCRIPTIONS,
    DEFAULT_DECISION_VALUES,
)

# The saved-decision-configuration store lives in app/state/saved_configs.py.
# The names a page or a test still imports from this module are re-exported
# here; new code imports them from app.state.saved_configs.
from app.state.saved_configs import (  # noqa: F401
    clear_decision_config,
    clear_selected_configuration,
    format_result_name,
    get_current_coefficients,
    get_current_stochastic_params,
    get_decision_config,
    get_selected_decision_configs,
    has_decision_config,
    save_selected_configuration,
)


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


def donation_income_mode_from_settings():
    """Donation Default's own income mode from the live settings.

    The global income_spec_mode can be contaminated when running disclose_income
    with "Compare both" (simulation.py syncs di_income_mode to income_spec_mode for
    results display), so the lookup is: 1) the donation tab persistence, 2) the
    page2_tab_income_spec_mode widget, 3) the global income_spec_mode. The gate
    (can_run_complete_simulation) and the Overview donation card both use it.
    """
    if (hasattr(st.session_state, 'donation_tab_persistence')
            and 'income_spec_mode' in st.session_state.donation_tab_persistence):
        return st.session_state.donation_tab_persistence['income_spec_mode']
    if 'page2_tab_income_spec_mode' in st.session_state:
        return st.session_state.page2_tab_income_spec_mode
    return st.session_state.get('income_spec_mode', 'categorical only')


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
    # (see donation_income_mode_from_settings)
    donation_income_mode = donation_income_mode_from_settings()
    
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
            # Every stored config is an explicit "Use This Config" selection.
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


# --------------------------------------------------------------------------- #
# The gate's blocking-issue warnings
# --------------------------------------------------------------------------- #
# Three screens show the "Run Complete Simulation" gate, each in its own wording:
# "overview" (the Page-2 Overview tab), "tab" (a decision tab's Simulation
# Options, render_simulation_buttons) and "results" (the results page,
# results/config_selection.py). Every text keeps the trailing whitespace of the
# screen's original; Streamlit strips it.

#: The decision a blocking issue names (by its block_type) on the results page, in
#: its warnings and in its disabled button's help.
def current_donation_configuration():
    """The donation configuration the CURRENT Page-1 / Donation Default settings imply.

    This is what the Overview donation card shows when nothing has been pinned
    into ``selected_decision_configs``. Returns None when the current settings do
    not imply exactly one donation configuration (no donation_default selected,
    or a Compare mode), which is when the card was not shown either.
    """
    selected_decisions = []
    if (hasattr(st.session_state, 'decision_params')
            and hasattr(st.session_state.decision_params, 'selected_decisions')):
        selected_decisions = st.session_state.decision_params.selected_decisions or []

    if 'donation_default' not in selected_decisions:
        return None

    population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
    # The gate's own lookup, so the Overview card never picks up a global
    # income_spec_mode that a disclose_income run contaminated.
    donation_income_mode = donation_income_mode_from_settings()

    is_compare_mode = (
        population_mode == "Compare all"
        or 'compare' in str(donation_income_mode).lower()
        or 'both' in str(donation_income_mode).lower()
    )
    if is_compare_mode:
        return None

    # Only one configuration may be implied, otherwise the user has to pick one.
    # Note: using *_ to handle the 5th return value (blocking_issues list)
    can_run, reason, config_count, block_type, *_ = can_run_complete_simulation()
    if config_count != 1:
        return None

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
                'sigma_value': DONATION_SIGMA_OVERALL,
                'sigma_coefficient': 1.0,
                'sigma_in_copula': False,
                'sigma_in_research': True
            },
            'anchor_weights': {
                'observed': 0.75,
                'predicted': 0.25
            }
        }

    return {
        'population_mode': population_mode,
        'donation_income_mode': donation_income_mode,
        # DEPRECATED alias the implied record carried; Decision 13's default block
        # ("Value synced from: ...") reads it
        'income_spec_mode': donation_income_mode,
        'coefficients': coefficients,
        'stochastic_params': stochastic_params,
        # No simulation has run: the linked final_donation_rate value is the
        # current default (what the implied record recorded at implication time)
        'metrics': {
            'mean_donation': st.session_state.get('final_donation_rate_default_value', 0.10),
        },
        'total_agents': st.session_state.get('n_agents', 1000)
    }


GATE_BLOCK_TYPE_TITLES = {
    "disclose_income": "Disclose Income",
    "donation_config": "Donation Default",
    "disclose_documents": "Disclose Documents",
    "rejected_transaction_defaults": "Rejected Transaction Defaults",
}


def blocking_issue_text(screen, i, block_type, reason):
    """The warning of issue `i` (its `block_type` and `reason`) on `screen` when
    SEVERAL issues block the complete run."""
    if screen == "results":
        # an issue of any other block type is the donation_config one
        title = GATE_BLOCK_TYPE_TITLES.get(block_type, "Donation Default")
        return f"""
**Issue {i}: {title}**

{reason}

**Action Required:**
1. Run **{title} Only** from the {title} tab
2. Click **"Use This Config"** on the result you want to use
                    """
    if block_type == "disclose_income":
        return f"""
**Issue {i}: Disclose Income**

{reason}

**Action Required:**
1. Go to the **Disclose Income** tab
2. Run **disclose_income only** and select one configuration
3. Or change to **"Categorical only"** or **"Continuous only"** mode
                        """
    if block_type == "rejected_transaction_defaults":
        return f"""
**Issue {i}: Rejected Transaction Defaults**

{reason}

**Action Required:**
1. Go to the **Rejected Transaction Defaults** tab
2. Run **rejected_transaction_defaults only** and select one configuration ("Use This Config")
3. Or change its Income Specification to **"Categorical only"** or **"Continuous only"**
                        """
    if block_type == "disclose_documents" and screen == "tab":
        return f"""
**Issue {i}: Disclose Documents**

{reason}

**Action Required:**
1. Go to the **Disclose Documents** tab
2. Run **disclose_documents only** and select one configuration
3. Or change to **"Categorical only"** or **"Continuous only"** mode
                        """
    if block_type == "disclose_documents":
        return f"""
**Issue {i}: Disclose Documents**

{reason}

**Action Required:**
1. Go to the **Disclose Documents** tab
2. Run **disclose_documents only** and select one configuration
                        """
    # donation_config block type
    return f"""
**Issue {i}: Donation Default**

{reason}

**Action Required:**
1. Go to the **Donation Default** tab
2. Run **donation_default only** and select one configuration
                        """


def configuration_required_text(screen, block_type, reason):
    """The warning on `screen` when exactly ONE issue blocks the complete run:
    `block_type` and `reason` are the gate's."""
    if screen == "results":
        if block_type in GATE_BLOCK_TYPE_TITLES:
            title = GATE_BLOCK_TYPE_TITLES[block_type]
            return f"""
⚠️ **{title} Configuration Required**

{reason}

**Action Required:**
1. Run **{title} Only** from the {title} tab
2. Click **"Use This Config"** on the result you want to use
3. Return here to run complete simulation
                """
        return f"""
⚠️ **Configuration Issue**

{reason}
                """
    if screen == "overview":
        if block_type == "disclose_income":
            return f"""
⚠️ **Disclose Income Configuration Required**

{reason}

**Action Required:**
1. Go to the **Disclose Income** tab
2. Change "Income Specification for Disclosure Model" from "Compare both" to either **"Categorical only"** or **"Continuous only"**
3. Return here to run complete simulation
                    """
        if block_type == "rejected_transaction_defaults":
            return f"""
⚠️ **Rejected Transaction Defaults Configuration Required**

{reason}

**Action Required:**
1. Go to the **Rejected Transaction Defaults** tab
2. Run **rejected_transaction_defaults only** and select one configuration ("Use This Config"),
   or change its Income Specification to **"Categorical only"** / **"Continuous only"**
3. Return here to run complete simulation
                    """
        if block_type == "disclose_documents":
            return f"""
⚠️ **Disclose Documents Configuration Required**

{reason}

**Action Required:**
1. Go to the **Disclose Documents** tab
2. Run **disclose_documents only** and select one configuration (or change "Income Specification for Disclosure Model" from "Compare both" to **"Categorical only"** or **"Continuous only"**)
3. Return here to run complete simulation
                    """
        if block_type == "donation_config":
            return f"""
⚠️ **Multiple Donation Configurations Detected**

{reason}

**Action Required:**
1. Go to the **Donation Default** tab
2. Run **donation_default only**
3. Select your preferred configuration from results
4. Return here to run complete simulation
                    """
    elif screen == "tab":
        if block_type == "disclose_income":
            return f"""
⚠️ **Disclose Income Configuration Required**

{reason}

**Action Required:**

1. Go to the **Disclose Income** tab
2. Change "Income Specification for Disclosure Model" from "Compare both" to either **"Categorical only"** or **"Continuous only"**
3. Return here and click **Run Complete Simulation**

This ensures all decisions produce a single result set.
                    """
        if block_type == "disclose_documents":
            return f"""
⚠️ **Disclose Documents Configuration Required**

{reason}

**Action Required:**

1. Go to the **Disclose Documents** tab
2. Run **disclose_documents only** and select one configuration (or change "Income Specification for Disclosure Model" from "Compare both" to **"Categorical only"** or **"Continuous only"**)
3. Return here and click **Run Complete Simulation**

This ensures all decisions produce a single result set.
                    """
        if block_type == "rejected_transaction_defaults":
            return f"""
⚠️ **Rejected Transaction Defaults Configuration Required**

{reason}

**Action Required:**

1. Go to the **Rejected Transaction Defaults** tab
2. Run **rejected_transaction_defaults only** and select one configuration ("Use This Config"),
   or change its Income Specification to **"Categorical only"** / **"Continuous only"**
3. Return here and click **Run Complete Simulation**

This ensures all decisions produce a single result set.
                    """
        if block_type == "donation_config":
            return f"""
⚠️ **Multiple Donation Configurations Detected**

{reason}

**Action Required:**

1. Go to the **Donation Default** tab
2. Run **donation_default only**
3. **Select one configuration** from the results
4. Return here and click **Run Complete Simulation**

This ensures all decisions use consistent settings.
                    """
    # an unknown block type: say what is wrong without naming a decision
    return f"""
⚠️ **Configuration Issue**

{reason}
                    """


def render_blocking_issues(screen, reason, block_type, blocking_issues):
    """Draw the gate's blocking-issue warnings on `screen` (see
    blocking_issue_text): with several issues an error saying how many, then one
    warning per issue; otherwise the one issue's warning. `reason`, `block_type`
    and `blocking_issues` are what can_run_complete_simulation() returned."""
    if blocking_issues and len(blocking_issues) > 1:
        st.error(f"⚠️ **{len(blocking_issues)} Configuration Issues Detected**")
        for i, issue in enumerate(blocking_issues, 1):
            st.warning(blocking_issue_text(screen, i, issue['block_type'], issue['reason']))
    else:
        st.warning(configuration_required_text(screen, block_type, reason))


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
            # block_type return value is the first issue's type; they agree, but the issue is
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
            
            # Show ALL blocking issues, not just the first one
            render_blocking_issues("tab", reason, block_type, blocking_issues)
            
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
        
        # Priority 2: Check for post-simulation adjustments (older keys; nothing writes them now)
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
            # Clear any saved configuration to allow fresh run with current tab settings
            if decision_name in ("donation_default", "disclose_income", "disclose_documents"):
                if has_decision_config(decision_name):
                    st.info(f"🔄 Clearing saved {decision_name} configuration - will use current tab settings")
                    clear_decision_config(decision_name)

            # Record which decisions this run executes with custom parameters: the
            # results page (and the tests) read these two keys after the run.
            st.session_state.custom_decisions = [decision_name]  # Only this decision was run with custom parameters
            st.session_state.default_decisions = []  # No decisions used default values (since only one was run)

            # The run request: only this decision runs; the Page-2 selection is
            # not touched (the plan builder reads the request, not the session).
            request = RunRequest(
                decisions_to_run=(decision_name,),
                custom_decisions=(decision_name,),
                default_decisions=(),
            )

            # Run simulation - check if Monte Carlo or Single Run mode
            if st.session_state.sim_params.simulation_mode == "Monte-Carlo Study":
                # Run Monte Carlo study
                mc_summary, mc_detailed, output_log = run_monte_carlo_study(request)
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

                    st.rerun()
                else:
                    st.error("❌ Monte Carlo simulation returned no results")
            else:
                # Run single simulation.  A successful run ends in st.rerun()
                # (a BaseException), so nothing below runs on the success path.
                run_full_simulation(request)

            # Only reached when the run did not redirect: Monte Carlo returned no
            # results, or run_full_simulation() handled an error itself.
            # custom_decisions / default_decisions stay: they describe this run.

        except Exception as e:
            # custom_decisions / default_decisions are not restored
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
        # The run does not write di_income_mode (or any other di_/dd_/rtd_ key)
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
            # Store metadata about which decisions use custom vs default parameters:
            # the results page (and the tests) read these two keys after the run.
            # FIX: Use effective_selected_decisions which includes decisions with saved configs
            st.session_state.custom_decisions = effective_selected_decisions
            st.session_state.default_decisions = unselected_decisions

            # The run request: ALL decisions run (a complete simulation); the
            # Page-2 selection is not touched (the plan builder reads the request).
            request = RunRequest(
                decisions_to_run=tuple(ALL_DECISIONS),
                custom_decisions=tuple(effective_selected_decisions),
                default_decisions=tuple(unselected_decisions),
            )

            # Run simulation with all decisions - check if Monte Carlo or Single Run mode
            if st.session_state.sim_params.simulation_mode == "Monte-Carlo Study":
                # Run Monte Carlo study
                mc_summary, mc_detailed, output_log = run_monte_carlo_study(request)
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

                    st.rerun()
                else:
                    st.error("❌ Monte Carlo simulation returned no results")
            else:
                # Run single simulation.  A successful run ends in st.rerun()
                # (a BaseException), so nothing below runs on the success path.
                run_full_simulation(request)

            # Only reached when the run did not redirect: Monte Carlo returned no
            # results, or run_full_simulation() handled an error itself.

        except Exception as e:
            st.error(f"❌ Error running complete simulation: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

