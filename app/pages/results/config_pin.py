# app/pages/results/config_pin.py
"""
The "Use This Config" buttons under a decision's results (donation_default,
disclose_income, disclose_documents and rejected_transaction_defaults): one
renderer, one spec per decision, and the seed-mismatch error shown when a pin is
refused. The selected-configuration banner and "Run Complete Simulation" are in
config_selection.py; the store is app/state/saved_configs.py.
"""
import streamlit as st


def render_seed_mismatch_error(decision_name, error_info):
    """
    Render a clear error message when seed mismatch is detected.
    
    Args:
        decision_name: Name of the decision being saved (e.g., 'disclose_income')
        error_info: Dict with keys: new_seed, new_n_agents, existing_seed, existing_n_agents, conflicting_decision
    """
    from app.state.saved_configs import clear_decision_config

    conflicting = error_info['conflicting_decision']
    conflicting_title = conflicting.replace('_', ' ').title()
    decision_title = decision_name.replace('_', ' ').title()
    
    st.error(f"""
**Seed Mismatch Detected**

Your **{decision_title}** configuration was run with:
- Seed: **{error_info['new_seed']}**
- Agents: **{error_info['new_n_agents']:,}**

But you already have a **{conflicting_title}** configuration saved with:
- Seed: **{error_info['existing_seed']}**
- Agents: **{error_info['existing_n_agents']:,}**

**Why this matters:** To ensure consistent results across decisions, all saved configurations must use the same seed and agent count.

**Action Required:** Re-run {decision_title} with seed={error_info['existing_seed']} and n_agents={error_info['existing_n_agents']}, then select that configuration.
    """)
    
    # Offer option to clear the conflicting config instead
    if st.button(
        f"🗑️ Clear {conflicting_title} config instead",
        key=f"clear_conflict_{decision_name}_{conflicting}",
        help=f"Remove the saved {conflicting_title} config so you can use this {decision_title} config"
    ):
        clear_decision_config(conflicting)
        st.success(f"Cleared {conflicting_title} configuration. Click 'Use This Config' again.")
        st.rerun()


# ----------------------------------------------------------------------------
# "Use This Config" buttons: one renderer, one spec per decision
# ----------------------------------------------------------------------------

def _donation_quick_summary(result_df):
    """Left-hand caption of the donation_default selection row."""
    # Show key metric for quick reference - check which columns are available
    donation_col = 'donation_default'
    if donation_col in result_df.columns:
        mean_donation = result_df[donation_col].mean()
        st.caption(f"📊 Quick Summary: {len(result_df):,} agents, avg donation {mean_donation:.2%}")
    elif 'disclose_income' in result_df.columns:
        y_rate = (result_df['disclose_income'] == 'Y').mean()
        st.caption(f"📊 Quick Summary: {len(result_df):,} agents, Y rate {y_rate:.2%}")
    else:
        st.caption(f"📊 Quick Summary: {len(result_df):,} agents")


def _disclose_income_quick_summary(result_df):
    """Left-hand caption of the disclose_income selection row."""
    # Show key metrics for quick reference
    raw_col = 'disclose_income_raw'
    if raw_col in result_df.columns:
        mean_raw = result_df[raw_col].mean()
        y_rate = (result_df[raw_col] > 0).mean() * 100
        st.caption(f"📊 Quick Summary: {len(result_df):,} agents, mean DI={mean_raw:.4f}, Y rate={y_rate:.2f}%")
    else:
        y_count = (result_df['disclose_income'] == 'Y').sum()
        y_rate = (y_count / len(result_df)) * 100 if len(result_df) > 0 else 0
        st.caption(f"📊 Quick Summary: {len(result_df):,} agents, Y rate={y_rate:.2f}%")


def _donation_save_args(result_key, result_df):
    """(params, metrics, extra_data) saved with a donation_default selection."""
    from app.state.saved_configs import (
        get_current_coefficients,
        get_current_stochastic_params,
        calculate_result_metrics,
        extract_configuration_details
    )
    # Get donation-specific parameters
    coefficients = get_current_coefficients()
    stochastic_params = get_current_stochastic_params()
    config_details = extract_configuration_details(result_key)
    
    params = {
        'coefficients': coefficients,
        'stochastic_params': stochastic_params,
        'income_mode': config_details['income_spec_mode']
    }
    
    metrics = calculate_result_metrics(result_df)
    
    extra_data = {
        'population_mode': config_details['population_mode'],
        'income_spec_mode': config_details['income_spec_mode']
    }
    return params, metrics, extra_data


def _disclose_income_save_args(result_key, result_df):
    """(params, metrics, extra_data) saved with a disclose_income selection."""
    from app.state.saved_configs import (
        get_current_disclose_income_params,
        calculate_disclose_income_metrics,
        extract_disclose_income_configuration_details
    )
    # Get disclose income specific parameters
    params = get_current_disclose_income_params()
    metrics = calculate_disclose_income_metrics(result_df)
    config_details = extract_disclose_income_configuration_details(result_key)
    
    extra_data = {
        'income_mode': config_details['income_mode']
    }
    return params, metrics, extra_data


def _disclose_documents_save_args(result_key, result_df):
    """(params, metrics, extra_data) saved with a disclose_documents selection."""
    from app.state.saved_configs import (
        get_current_disclose_documents_params,
        calculate_disclose_documents_metrics,
        extract_disclose_documents_configuration_details,
    )
    params = get_current_disclose_documents_params()
    metrics = calculate_disclose_documents_metrics(result_df)
    config_details = extract_disclose_documents_configuration_details(result_key)
    extra_data = {'income_mode': config_details['income_mode']}
    return params, metrics, extra_data


def _rejected_transaction_save_args(result_key, result_df):
    """(params, metrics, extra_data) saved with a Decision 4 selection."""
    from app.state.saved_configs import (
        get_current_rejected_transaction_params,
        calculate_rejected_transaction_metrics,
        extract_rejected_transaction_configuration_details,
    )
    # imported here: overview.py imports this module for its buttons
    from app.pages.results.overview import rtd_page_element
    params = get_current_rejected_transaction_params()
    metrics = calculate_rejected_transaction_metrics(result_df)
    details = extract_rejected_transaction_configuration_details(result_key)
    extra_data = {'income_mode': details['income_mode'],
                  'population_mode': details['population_mode'],
                  # the element this run showed (None = whole decision), so
                  # its summary lists only that element's metric
                  'run_element': rtd_page_element()}
    return params, metrics, extra_data


# Per decision: whether a divider precedes the row, the left-hand caption (None:
# the column stays empty), the button help, the message after a successful save,
# what is saved, and whether the selected state is a button that undoes the
# selection (Decision 4) rather than a "Selected" badge. The button keys are
# spelled out in _render_selection_button, where the session-key registry scan
# (tests/test_session_key_registry.py) can read them.
_SELECTION_BUTTONS = {
    'donation_default': dict(
        divider=False, summary=_donation_quick_summary,
        help="Select this configuration for combined simulations",
        saved="Configuration selected!", save_args=_donation_save_args, unselect=False),
    'disclose_income': dict(
        divider=True, summary=_disclose_income_quick_summary,
        help="Select this disclose income configuration for combined simulations",
        saved="Disclose Income configuration selected!", save_args=_disclose_income_save_args,
        unselect=False),
    'disclose_documents': dict(
        divider=True, summary=None,
        help="Select this disclose documents configuration for combined simulations",
        saved="Disclose Documents configuration selected!", save_args=_disclose_documents_save_args,
        unselect=False),
    # No "Quick Summary" caption for Decision 4 (it was unrelated to the element
    # presented, e.g. list lengths under the Flexibility results). The selection
    # can be undone right here (it must be possible to unselect at any point so
    # later runs present all alternatives).
    'rejected_transaction_defaults': dict(
        divider=True, summary=None,
        help="Select this Rejected Transaction Defaults configuration for combined simulations",
        saved="Rejected Transaction Defaults configuration selected!",
        save_args=_rejected_transaction_save_args, unselect=True),
}


def _render_selection_button(decision_name, result_key, result_df):
    """The "Use This Config" row under a result: a caption on the left, on the
    right the button (or the selected state) for `decision_name`."""
    # Import here to avoid circular imports
    from app.state.saved_configs import (
        save_decision_config,
        clear_decision_config,
        is_decision_config_selected,
    )
    spec = _SELECTION_BUTTONS[decision_name]
    select_key = (f"inline_select_{result_key}" if decision_name == 'donation_default'
                  else f"di_inline_select_{result_key}" if decision_name == 'disclose_income'
                  else f"dd_inline_select_{result_key}" if decision_name == 'disclose_documents'
                  else f"rtd_inline_select_{result_key}")

    is_selected = is_decision_config_selected(decision_name, result_key)
    
    if spec['divider']:
        st.markdown("---")
    
    # Create a compact selection interface
    col1, col2 = st.columns([2, 1])
    
    if spec['summary'] is not None:
        with col1:
            spec['summary'](result_df)
    
    with col2:
        # Selection button
        if is_selected:
            if not spec['unselect']:
                st.success("✅ Selected")
            else:
                if st.button("✅ Selected - click to unselect", key=f"rtd_inline_unselect_{result_key}",
                             use_container_width=True,
                             help="This configuration is selected for combined simulations. Click to "
                                  "unselect it; the next runs present all configurations again."):
                    clear_decision_config(decision_name)
                    st.rerun()
        else:
            if st.button(
                "🎯 Use This Config",
                key=select_key,
                type="primary",
                use_container_width=True,
                help=spec['help']
            ):
                params, metrics, extra_data = spec['save_args'](result_key, result_df)
                
                # Use unified save function with seed validation
                success, config, error_info = save_decision_config(
                    decision_name, result_key, result_df, params, metrics, extra_data
                )
                
                if success:
                    st.success(spec['saved'])
                    st.rerun()
                else:
                    # Show seed mismatch error
                    render_seed_mismatch_error(decision_name, error_info)


def render_inline_selection_button(result_key, result_df):
    """Render selection button directly under the chart (for donation_default)"""
    _render_selection_button('donation_default', result_key, result_df)


def render_disclose_income_selection_button(result_key, result_df):
    """Render selection button for disclose income configuration"""
    _render_selection_button('disclose_income', result_key, result_df)


def render_disclose_documents_selection_button(result_key, result_df):
    """Render the 'Use This Config' selection button for a disclose documents cell."""
    _render_selection_button('disclose_documents', result_key, result_df)


def render_rejected_transaction_selection_button(result_key, result_df):
    """Render the 'Use This Config' selection button for a Decision 4 (Rejected
    Transaction Defaults) result cell - mirrors render_disclose_income_selection_button."""
    _render_selection_button('rejected_transaction_defaults', result_key, result_df)
