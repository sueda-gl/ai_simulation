# app/pages/decision_execution.py
"""
Decision execution functions for running individual and combined simulations.
"""
import traceback

import streamlit as st
from src.contract.plan import RunRequest
from app.simulation import run_full_simulation, run_monte_carlo_study
from app.models import ALL_DECISIONS, DONATION_SIGMA_OVERALL
# Used below; page1_common_params.py and the results pages also import some from here.
from app.state.saved_configs import (
    clear_decision_config,
    get_current_coefficients,
    get_current_stochastic_params,
    get_decision_config,
    get_selected_decision_configs,
    has_decision_config,
)

# Re-exported (unused here) for results/visualizations/viz_helpers.py and tests/conftest.py.
from src.contract.defaults import (  # noqa: F401
    DEFAULT_DECISION_DESCRIPTIONS,
    DEFAULT_DECISION_VALUES,
)
# Re-exported (unused here) for results/config_selection.py and visualizations/transaction_viz.py.
from app.state.saved_configs import (  # noqa: F401
    clear_selected_configuration,
    format_result_name,
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

    The lookup is: 1) the donation tab persistence, 2) the page2_tab_income_spec_mode
    widget, 3) the global income_spec_mode. The gate (can_run_complete_simulation)
    and the Overview donation card both use it.
    """
    if (hasattr(st.session_state, 'donation_tab_persistence')
            and 'income_spec_mode' in st.session_state.donation_tab_persistence):
        return st.session_state.donation_tab_persistence['income_spec_mode']
    if 'page2_tab_income_spec_mode' in st.session_state:
        return st.session_state.page2_tab_income_spec_mode
    return st.session_state.get('income_spec_mode', 'categorical only')


#: One row per decision the complete-run gate checks, in the order it checks them
#: (and the order of the blocking issues it reports). ``block_type`` is the name
#: the screens key their warnings on; ``income_mode`` reads the decision's live
#: income mode; ``saved_info`` is what a pinned record of the decision contributes
#: to the success message ("Using saved configuration(s): ...").
class GateDecision:
    __slots__ = ("name", "block_type", "income_mode", "saved_info")

    def __init__(self, name, block_type, income_mode, saved_info):
        self.name, self.block_type = name, block_type
        self.income_mode, self.saved_info = income_mode, saved_info


def _saved_income_mode(config):
    """A pinned Disclose Income / Disclose Documents record's income mode: the one
    under ``params`` first, then the top-level one."""
    return config.get('params', {}).get('income_mode', config.get('income_mode', 'Unknown'))


def _saved_rtd_info(config):
    """A pinned Decision 4 record: "<population> + <income mode>", the top-level
    income mode first (the reverse of _saved_income_mode's order)."""
    return (f"{config.get('population_mode', 'Unknown')} + "
            f"{config.get('income_mode', config.get('params', {}).get('income_mode', 'Unknown'))}")


def _saved_donation_info(config):
    """A pinned Donation record: "<population> + <income mode>"; a record without a
    population mode reports the session's."""
    pop_mode = config.get('population_mode', st.session_state.get('population_mode', 'Unknown'))
    inc_mode = config.get('donation_income_mode', config.get('income_spec_mode', 'Unknown'))
    return f"{pop_mode} + {inc_mode}"


GATE_DECISIONS = (
    GateDecision('disclose_income', 'disclose_income',
                 lambda: st.session_state.get('di_income_mode', 'Categorical only'),
                 _saved_income_mode),
    GateDecision('disclose_documents', 'disclose_documents',
                 lambda: st.session_state.get('dd_income_mode', 'Categorical only'),
                 _saved_income_mode),
    GateDecision('rejected_transaction_defaults', 'rejected_transaction_defaults',
                 lambda: st.session_state.get('rtd_income_mode', 'Continuous only'),
                 _saved_rtd_info),
    GateDecision('donation_default', 'donation_config',
                 donation_income_mode_from_settings,
                 _saved_donation_info),
)

#: The order of the parts of the success message: Donation Default comes second,
#: not last as in the checks.
GATE_SUCCESS_ORDER = ('disclose_income', 'donation_default', 'disclose_documents',
                      'rejected_transaction_defaults')


def can_run_complete_simulation():
    """
    Determine if complete simulation can run based on configuration state.
    
    This prevents running all decisions when:
    1. Any decision is in "Compare both" mode without a saved config selected
    2. Multiple configurations would be generated without an explicit selection
    
    IMPORTANT: Only check config requirements for decisions the user actually selected.
    Unselected decisions will use defaults and don't need explicit config selection.
    
    Accumulates ALL blocking issues instead of returning on the first one, so the
    UI can display all conflicts at once.

    One check per row of GATE_DECISIONS; tests/test_complete_run_gate.py pins
    every outcome.
    
    Returns:
        tuple: (can_run: bool, reason: str, config_count: int, block_type: str or None, blocking_issues: list)
            - can_run: Whether complete simulation is allowed
            - reason: Human-readable explanation (combined if multiple issues)
            - config_count: Number of configurations that would be generated
            - block_type: The first blocking issue's block type - "disclose_income", "disclose_documents",
              "rejected_transaction_defaults" or "donation_config" - or None
            - blocking_issues: List of all blocking issues with details
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
    
    # Population modes: "Compare all" generates 3, others generate 1
    population_mode = st.session_state.get('population_mode', 'Copula (synthetic)')
    population_count = 3 if population_mode == "Compare all" else 1

    # ========================================================================
    # CHECK EVERY GATED DECISION (only if selected)
    # ========================================================================

    checks = {}   # decision name -> (selected, total_configs, has_config, saved_info)
    for gate in GATE_DECISIONS:
        title = GATE_BLOCK_TYPE_TITLES[gate.block_type]
        # Income modes: "Compare both" generates 2, others generate 1
        income_mode = gate.income_mode()
        income_count = 2 if ('compare' in str(income_mode).lower() or 'both' in str(income_mode).lower()) else 1

        selected = gate.name in selected_decisions
        total_configs = population_count * income_count
        has_config = False
        saved_info = None

        if selected:
            # Every stored config is an explicit "Use This Config" selection.
            if gate.name in configs:
                has_config = True
                saved_info = gate.saved_info(configs[gate.name])

            # If multiple configs and no saved selection, add to blocking issues
            if total_configs > 1 and not has_config:
                blocking_issues.append({
                    'decision': gate.name,
                    'block_type': gate.block_type,
                    'config_count': total_configs,
                    'reason': f"{title} has {total_configs} configurations (population: {population_count}, income: {income_count}) - please run {gate.name} only and select one"
                })

        checks[gate.name] = (selected, total_configs, has_config, saved_info)
    
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
    
    # No blocking issues - build success message: one part per selected and saved
    # decision, in GATE_SUCCESS_ORDER
    titles = {gate.name: GATE_BLOCK_TYPE_TITLES[gate.block_type] for gate in GATE_DECISIONS}
    config_parts = []
    for name in GATE_SUCCESS_ORDER:
        selected, total_configs, has_config, saved_info = checks[name]
        if has_config and saved_info:
            config_parts.append(f"{titles[name]}: {saved_info}")

    # Determine total config count for display
    total_configs = max(total if selected else 1
                        for selected, total, _, _ in checks.values())
    
    if config_parts:
        return (True, f"Using saved configuration(s): {', '.join(config_parts)}", total_configs, None, [])
    elif not checks['disclose_income'][0] and not checks['donation_default'][0]:
        # neither Disclose Income nor Donation Default selected - a Disclose
        # Documents / Decision 4-only selection says this too
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
    # The gate's own lookup, so the card shows the income mode the gate counts.
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


#: The decision a blocking issue names (by its block_type) on the results page, in
#: its warnings and in its disabled button's help.
GATE_BLOCK_TYPE_TITLES = {
    "disclose_income": "Disclose Income",
    "donation_config": "Donation Default",
    "disclose_documents": "Disclose Documents",
    "rejected_transaction_defaults": "Rejected Transaction Defaults",
}


#: The help of the disabled "Run Complete Simulation" button when SEVERAL issues
#: block the run (a decision tab's button and the results page's button).
GATE_ISSUE_COUNT_HELP = "{} configuration issue(s) detected"


# The texts are ``str.format`` templates over ``{i}`` (the issue number),
# ``{title}`` (GATE_BLOCK_TYPE_TITLES) and ``{reason}`` (the gate's reason).
# Every text keeps the trailing whitespace of the screen's original; Streamlit
# strips it. tests/test_complete_run_gate.py pins each one.

_ISSUE_RESULTS = """
**Issue {i}: {title}**

{reason}

**Action Required:**
1. Run **{title} Only** from the {title} tab
2. Click **"Use This Config"** on the result you want to use
                    """

_ISSUE_DISCLOSE_INCOME = """
**Issue {i}: Disclose Income**

{reason}

**Action Required:**
1. Go to the **Disclose Income** tab
2. Run **disclose_income only** and select one configuration
3. Or change to **"Categorical only"** or **"Continuous only"** mode
                        """

_ISSUE_RTD = """
**Issue {i}: Rejected Transaction Defaults**

{reason}

**Action Required:**
1. Go to the **Rejected Transaction Defaults** tab
2. Run **rejected_transaction_defaults only** and select one configuration ("Use This Config")
3. Or change its Income Specification to **"Categorical only"** or **"Continuous only"**
                        """

_ISSUE_DISCLOSE_DOCUMENTS_TAB = """
**Issue {i}: Disclose Documents**

{reason}

**Action Required:**
1. Go to the **Disclose Documents** tab
2. Run **disclose_documents only** and select one configuration
3. Or change to **"Categorical only"** or **"Continuous only"** mode
                        """

_ISSUE_DISCLOSE_DOCUMENTS = """
**Issue {i}: Disclose Documents**

{reason}

**Action Required:**
1. Go to the **Disclose Documents** tab
2. Run **disclose_documents only** and select one configuration
                        """

_ISSUE_DONATION = """
**Issue {i}: Donation Default**

{reason}

**Action Required:**
1. Go to the **Donation Default** tab
2. Run **donation_default only** and select one configuration
                        """

_REQUIRED_RESULTS = """
⚠️ **{title} Configuration Required**

{reason}

**Action Required:**
1. Run **{title} Only** from the {title} tab
2. Click **"Use This Config"** on the result you want to use
3. Return here to run complete simulation
                """

_REQUIRED_RESULTS_UNKNOWN = """
⚠️ **Configuration Issue**

{reason}
                """

_REQUIRED_OVERVIEW_DISCLOSE_INCOME = """
⚠️ **Disclose Income Configuration Required**

{reason}

**Action Required:**
1. Go to the **Disclose Income** tab
2. Change "Income Specification for Disclosure Model" from "Compare both" to either **"Categorical only"** or **"Continuous only"**
3. Return here to run complete simulation
                    """

_REQUIRED_OVERVIEW_RTD = """
⚠️ **Rejected Transaction Defaults Configuration Required**

{reason}

**Action Required:**
1. Go to the **Rejected Transaction Defaults** tab
2. Run **rejected_transaction_defaults only** and select one configuration ("Use This Config"),
   or change its Income Specification to **"Categorical only"** / **"Continuous only"**
3. Return here to run complete simulation
                    """

_REQUIRED_OVERVIEW_DISCLOSE_DOCUMENTS = """
⚠️ **Disclose Documents Configuration Required**

{reason}

**Action Required:**
1. Go to the **Disclose Documents** tab
2. Run **disclose_documents only** and select one configuration (or change "Income Specification for Disclosure Model" from "Compare both" to **"Categorical only"** or **"Continuous only"**)
3. Return here to run complete simulation
                    """

_REQUIRED_OVERVIEW_DONATION = """
⚠️ **Multiple Donation Configurations Detected**

{reason}

**Action Required:**
1. Go to the **Donation Default** tab
2. Run **donation_default only**
3. Select your preferred configuration from results
4. Return here to run complete simulation
                    """

_REQUIRED_TAB_DISCLOSE_INCOME = """
⚠️ **Disclose Income Configuration Required**

{reason}

**Action Required:**

1. Go to the **Disclose Income** tab
2. Change "Income Specification for Disclosure Model" from "Compare both" to either **"Categorical only"** or **"Continuous only"**
3. Return here and click **Run Complete Simulation**

This ensures all decisions produce a single result set.
                    """

_REQUIRED_TAB_DISCLOSE_DOCUMENTS = """
⚠️ **Disclose Documents Configuration Required**

{reason}

**Action Required:**

1. Go to the **Disclose Documents** tab
2. Run **disclose_documents only** and select one configuration (or change "Income Specification for Disclosure Model" from "Compare both" to **"Categorical only"** or **"Continuous only"**)
3. Return here and click **Run Complete Simulation**

This ensures all decisions produce a single result set.
                    """

_REQUIRED_TAB_RTD = """
⚠️ **Rejected Transaction Defaults Configuration Required**

{reason}

**Action Required:**

1. Go to the **Rejected Transaction Defaults** tab
2. Run **rejected_transaction_defaults only** and select one configuration ("Use This Config"),
   or change its Income Specification to **"Categorical only"** / **"Continuous only"**
3. Return here and click **Run Complete Simulation**

This ensures all decisions produce a single result set.
                    """

_REQUIRED_TAB_DONATION = """
⚠️ **Multiple Donation Configurations Detected**

{reason}

**Action Required:**

1. Go to the **Donation Default** tab
2. Run **donation_default only**
3. **Select one configuration** from the results
4. Return here and click **Run Complete Simulation**

This ensures all decisions use consistent settings.
                    """

_REQUIRED_UNKNOWN = """
⚠️ **Configuration Issue**

{reason}
                    """

#: The warning of one of SEVERAL blocking issues: screen -> block type -> text.
#: A block type missing from a screen's row takes the row's ``None`` entry; a
#: screen missing here reads as "overview".
GATE_ISSUE_TEXTS = {
    "results": {None: _ISSUE_RESULTS},
    "overview": {"disclose_income": _ISSUE_DISCLOSE_INCOME,
                 "rejected_transaction_defaults": _ISSUE_RTD,
                 "disclose_documents": _ISSUE_DISCLOSE_DOCUMENTS,
                 None: _ISSUE_DONATION},
    "tab": {"disclose_income": _ISSUE_DISCLOSE_INCOME,
            "rejected_transaction_defaults": _ISSUE_RTD,
            "disclose_documents": _ISSUE_DISCLOSE_DOCUMENTS_TAB,
            None: _ISSUE_DONATION},
}

#: The warning when exactly ONE issue blocks the run: screen -> block type -> text.
#: A block type missing from a screen's row takes the row's ``None`` entry; a
#: screen missing here says what is wrong without naming a decision.
GATE_REQUIRED_TEXTS = {
    "results": {**{block_type: _REQUIRED_RESULTS for block_type in GATE_BLOCK_TYPE_TITLES},
                None: _REQUIRED_RESULTS_UNKNOWN},
    "overview": {"disclose_income": _REQUIRED_OVERVIEW_DISCLOSE_INCOME,
                 "rejected_transaction_defaults": _REQUIRED_OVERVIEW_RTD,
                 "disclose_documents": _REQUIRED_OVERVIEW_DISCLOSE_DOCUMENTS,
                 "donation_config": _REQUIRED_OVERVIEW_DONATION,
                 None: _REQUIRED_UNKNOWN},
    "tab": {"disclose_income": _REQUIRED_TAB_DISCLOSE_INCOME,
            "disclose_documents": _REQUIRED_TAB_DISCLOSE_DOCUMENTS,
            "rejected_transaction_defaults": _REQUIRED_TAB_RTD,
            "donation_config": _REQUIRED_TAB_DONATION,
            None: _REQUIRED_UNKNOWN},
}


def _gate_text(table, screen, block_type, fallback, **fields):
    row = table.get(screen, fallback)
    template = row.get(block_type, row[None])
    # an issue of any other block type is the donation_config one
    title = GATE_BLOCK_TYPE_TITLES.get(block_type, "Donation Default")
    return template.format(title=title, **fields)


def blocking_issue_text(screen, i, block_type, reason):
    """The warning of issue `i` (its `block_type` and `reason`) on `screen` when
    SEVERAL issues block the complete run (GATE_ISSUE_TEXTS)."""
    return _gate_text(GATE_ISSUE_TEXTS, screen, block_type, GATE_ISSUE_TEXTS["overview"],
                      i=i, reason=reason)


def configuration_required_text(screen, block_type, reason):
    """The warning on `screen` when exactly ONE issue blocks the complete run:
    `block_type` and `reason` are the gate's (GATE_REQUIRED_TEXTS)."""
    return _gate_text(GATE_REQUIRED_TEXTS, screen, block_type, {None: _REQUIRED_UNKNOWN},
                      reason=reason)


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
        can_run, reason, config_count, block_type, blocking_issues = can_run_complete_simulation()
        
        if not can_run:
            # The single-issue wording follows the ONE blocking issue's own type (the
            # block_type return value is the first issue's type; they agree, but the issue is
            # the source of truth). Each type has its own heading; the donation heading
            # is used for the donation block only, never as a catch-all.
            if len(blocking_issues) == 1:
                block_type = blocking_issues[0]['block_type']
            # Disabled button with explanation
            help_text = (GATE_ISSUE_COUNT_HELP.format(len(blocking_issues)) if len(blocking_issues) > 1
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

            # Record which decisions this run executes with custom parameters. The
            # results page reads the run's _run_metadata and falls back to these two
            # keys (results/run_context.py); the tests read them after the run.
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
    - donation_default uses its saved config's donation_income_mode (if saved) or income_spec_mode
    - disclose_income uses di_income_mode from its tab settings
    - Other decisions use their own respective settings
    
    The run does not write income_spec_mode.
    
    Every decision with a saved config is added to effective_selected_decisions (the
    run's custom decisions), not only the decisions selected on Page 2.
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
            # Store which decisions use custom vs default parameters (custom includes
            # every decision with a saved config). The results page reads the run's
            # _run_metadata and falls back to these two keys (results/run_context.py);
            # the tests read them after the run.
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
            st.text(traceback.format_exc())

