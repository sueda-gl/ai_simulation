# app/pages/results/config_selection.py
"""
Configuration selection UI for decision results (donation_default, disclose_income,
disclose_documents and rejected_transaction_defaults): the selected-configuration
banner with its Clear button and the "Run Complete Simulation" section.
"""
import streamlit as st
import pandas as pd
from app.pages.decision_execution import (
    save_selected_configuration,
    format_result_name,
    clear_selected_configuration,
    run_combined_simulation,
    can_run_complete_simulation,
    get_selected_decision_configs,
    get_decision_config,
    clear_decision_config
)
from app.models import ALL_DECISIONS
from app.pages.results.run_context import RunContext
from app.components import rtd_page_element
from app.reports.rtd import rtd_summary_metric_keys


def render_configuration_selection_ui(results_dict, ctx=None):
    """Render configuration selection UI for donation decision results"""
    ctx = ctx if ctx is not None else RunContext.from_session()

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
    # (individual runs have empty default_decisions)
    is_individual_donation_run = ctx.is_individual_run('donation_default')

    if not is_individual_donation_run:
        return
    
    # Check if config is already selected
    dd_config = get_decision_config('donation_default')
    has_selected_config = dd_config is not None
    
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
            
            # Get population and income mode for display - the modes this run used
            population_mode = ctx.effective_population_mode
            income_spec_mode = ctx.effective_income_mode

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
        config = dd_config
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


def _di_dd_banner_title(label):
    """Banner title of Decisions 1 and 2: the saved configuration's income mode."""
    def title(config):
        income_mode = config.get('params', {}).get('income_mode', config.get('income_mode', 'Unknown'))
        return f"✅ **Selected {label} Configuration**: {income_mode}"
    return title


def _y_rate_caption(label):
    """Left-hand caption of the Decision 1 / 2 banner: the saved Y rate."""
    def caption(config, metrics):
        y_rate = metrics.get('y_rate', 0)
        st.caption(f"{label}: {y_rate:.2%}")
    return caption


def _rtd_banner_title(config):
    """Banner title of Decision 4: population mode + income mode."""
    income_mode = config.get('income_mode', config.get('params', {}).get('income_mode', 'Unknown'))
    population_mode = config.get('population_mode', st.session_state.get('population_mode', 'Unknown'))
    return (f"✅ **Selected Rejected Transaction Defaults Configuration**: "
            f"{population_mode} + {income_mode}")


def _rtd_banner_captions(config, metrics):
    """Left-hand captions of the Decision 4 banner: only the average that belongs
    to the element on this page."""
    shown = rtd_summary_metric_keys(rtd_page_element())
    if 'mean_choice_length' in metrics and 'mean_choice_length' in shown:
        st.caption(f"Avg options list length: {metrics['mean_choice_length']:.2f}")
    if 'mean_default_list_length' in metrics and 'mean_default_list_length' in shown:
        st.caption(f"Avg integrated default list length: {metrics['mean_default_list_length']:.2f}")


# Per decision: the column its results carry, the banner title, the left-hand
# caption(s), and whether the caption / timestamp / Clear row shows even when the
# saved configuration has no metrics. (The Clear buttons' keys are spelled out in
# _render_saved_config_banner, where the session-key registry scan reads them.)
_SAVED_CONFIG_BANNERS = {
    'disclose_income': dict(
        results_column='disclose_income', title=_di_dd_banner_title("Disclose Income"),
        captions=_y_rate_caption("Y Rate"), row_without_metrics=False),
    'disclose_documents': dict(
        results_column='disclose_documents', title=_di_dd_banner_title("Disclose Documents"),
        captions=_y_rate_caption("Y Rate (qualified)"), row_without_metrics=False),
    'rejected_transaction_defaults': dict(
        results_column='rtd_choice_length', title=_rtd_banner_title,
        captions=_rtd_banner_captions, row_without_metrics=True),
}


def _render_saved_config_banner(decision_name, results_dict, ctx):
    """Once a configuration of an individual `decision_name` run has been selected
    with "Use This Config": the selected configuration (with Clear) and the
    "Run Complete Simulation" section."""
    spec = _SAVED_CONFIG_BANNERS[decision_name]
    clear_key = ("clear_di_selection" if decision_name == 'disclose_income'
                 else "clear_dd_selection" if decision_name == 'disclose_documents'
                 else "clear_rtd_selection")
    ctx = ctx if ctx is not None else RunContext.from_session()

    if not results_dict:
        return

    has_results = any(
        spec['results_column'] in df.columns
        for df in results_dict.values()
        if isinstance(df, pd.DataFrame) and not df.empty
    )
    if not has_results:
        return

    # the run's own decision lists, not the live widgets
    if not ctx.is_individual_run(decision_name):
        return

    config = get_decision_config(decision_name)
    has_selected_config = config is not None

    if has_selected_config:
        st.markdown("---")
        title = spec['title'](config)

        with st.container():
            st.success(title)
            metrics = config.get('metrics', {}) or {}
            if metrics or spec['row_without_metrics']:
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    spec['captions'](config, metrics)
                with col2:
                    timestamp = config.get('selected_timestamp')
                    if timestamp:
                        st.caption(f"Selected at {timestamp.strftime('%H:%M:%S')}")
                with col3:
                    if st.button("🗑️ Clear", help="Clear the selected configuration", key=clear_key):
                        clear_decision_config(decision_name)
                        st.rerun()

        render_complete_simulation_section()


def render_disclose_income_config_selection_ui(results_dict, ctx=None):
    """Render configuration selection UI for disclose_income decision results"""
    _render_saved_config_banner('disclose_income', results_dict, ctx)


def render_disclose_documents_config_selection_ui(results_dict, ctx=None):
    """Render configuration selection UI for disclose_documents results (mirrors disclose_income)."""
    _render_saved_config_banner('disclose_documents', results_dict, ctx)


def render_rejected_transaction_config_selection_ui(results_dict, ctx=None):
    """Configuration selection UI for Decision 4 (Rejected Transaction Defaults)
    results - mirrors render_disclose_income_config_selection_ui: once a
    configuration of an individual Decision 4 run has been selected with
    "Use This Config", show the selected configuration (with Clear) and the
    "Run Complete Simulation" section."""
    _render_saved_config_banner('rejected_transaction_defaults', results_dict, ctx)


# The decision a blocking issue names (by its block_type) in the warnings and in
# the disabled button's help.
_BLOCK_TYPE_TITLES = {
    "disclose_income": "Disclose Income",
    "donation_config": "Donation Default",
    "disclose_documents": "Disclose Documents",
    "rejected_transaction_defaults": "Rejected Transaction Defaults",
}


def _issue_warning(i, title, reason):
    """Warning text of issue `i` when several issues block the complete run.
    (The trailing whitespace is the original's; Streamlit strips it.)"""
    return f"""
**Issue {i}: {title}**

{reason}

**Action Required:**
1. Run **{title} Only** from the {title} tab
2. Click **"Use This Config"** on the result you want to use
                    """


def _configuration_required_warning(title, reason):
    """Warning text when exactly one issue blocks the complete run.
    (The trailing whitespace is the original's; Streamlit strips it.)"""
    return f"""
⚠️ **{title} Configuration Required**

{reason}

**Action Required:**
1. Run **{title} Only** from the {title} tab
2. Click **"Use This Config"** on the result you want to use
3. Return here to run complete simulation
                """


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
                # an issue of any other block type is the donation_config one
                title = _BLOCK_TYPE_TITLES.get(issue['block_type'], "Donation Default")
                st.warning(_issue_warning(i, title, issue['reason']))
        else:
            # Single issue - show original format
            if block_type in _BLOCK_TYPE_TITLES:
                st.warning(_configuration_required_warning(_BLOCK_TYPE_TITLES[block_type], reason))
            else:
                st.warning(f"""
⚠️ **Configuration Issue**

{reason}
                """)
        
        # Disabled button
        help_text = (f"{len(blocking_issues)} configuration issue(s) detected" if len(blocking_issues) > 1
                     else f"Select a {_BLOCK_TYPE_TITLES[block_type]} config first" if block_type in _BLOCK_TYPE_TITLES
                     else "Configuration issue detected")
            
        st.button(
            "🚀 Run Complete Simulation",
            type="primary",
            use_container_width=True,
            disabled=True,
            key="run_complete_from_results_disabled",
            help=help_text
        )
    else:
        # Simulation CAN run - now show the saved configs that will be used
        explicit_configs = get_selected_decision_configs()
        
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
                elif decision_name == 'disclose_documents':
                    income_mode = config.get('params', {}).get('income_mode',
                        config.get('income_mode', 'Unknown'))
                    y_rate = config.get('metrics', {}).get('y_rate', 0)
                    st.caption(f"  ✅ {decision_title}: {income_mode} (Y rate qualified: {y_rate:.2%})")
                elif decision_name == 'rejected_transaction_defaults':
                    income_mode = config.get('income_mode', config.get('params', {}).get('income_mode', 'Unknown'))
                    population_mode = config.get('population_mode', 'Unknown')
                    rtd_metrics = config.get('metrics', {}) or {}
                    shown = rtd_summary_metric_keys(rtd_page_element())   # this page's element only
                    mean_len = rtd_metrics.get('mean_choice_length')
                    mean_default = rtd_metrics.get('mean_default_list_length')
                    if mean_len is not None and 'mean_choice_length' in shown:
                        extra = f" (avg options list length: {mean_len:.2f})"
                    elif (mean_default is not None and 'mean_default_list_length' in shown
                          and 'mean_choice_length' not in shown):
                        extra = f" (avg integrated default list length: {mean_default:.2f})"
                    else:
                        extra = ""
                    st.caption(f"  ✅ {decision_title}: {population_mode} + {income_mode}{extra}")
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
