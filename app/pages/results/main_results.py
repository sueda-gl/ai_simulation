# app/pages/results/main_results.py
"""
Main results page rendering for the Enhanced AI Agent Simulation.

Layout decisions (which grid, which sections, which frame feeds the export)
come from the run that produced the results - ``RunContext`` over
``st.session_state._run_metadata`` and the result frames (ruling R28) - never
from the live Page-1 / Page-2 mode keys.
"""
import streamlit as st
import pandas as pd
from app.pages.navigation import render_navigation
from app.components import show_overview, show_monte_carlo_results
from app.pages.results.comparisons import (
    render_all_modes_comparison,
    render_income_comparison
)
from app.pages.results.details import (
    render_individual_agent_details,
    render_export_section
)
from app.pages.decision_execution import (
    save_selected_configuration,
    format_result_name,
    is_configuration_selected,
    clear_selected_configuration,
    DEFAULT_DECISION_VALUES,
    DEFAULT_DECISION_DESCRIPTIONS,
    get_decision_config,
    has_decision_config
)
from app.models import ALL_DECISIONS
from app.pages.results.run_context import RunContext

# Import from new modules
from app.pages.results.decision_visualizations import (
    render_decision_results,
    DECISION_VISUALIZATIONS,
    get_dynamic_description
)
from app.pages.results.config_selection import (
    render_configuration_selection_ui,
    render_disclose_income_config_selection_ui,
    render_disclose_documents_config_selection_ui,
    render_rejected_transaction_config_selection_ui
)


def get_decision_config_display(decision_name, ctx=None):
    """Get the selected configuration info for a decision to display in results.

    Returns a dict with 'has_config', 'income_mode', 'source', 'is_saved' keys.

    Without a saved configuration the income mode is the one the run on screen
    actually used for this decision (``ctx``, from _run_metadata - e.g. Decision 4
    runs a complete simulation in its own tab mode, not the run's global mode);
    the decision's CURRENT tab setting is only the fallback for runs that did not
    record it.
    """
    result = {
        'has_config': False,
        'income_mode': None,
        'source': None,
        'is_saved': False
    }

    if decision_name == 'donation_default':
        config = get_decision_config('donation_default')
        if config:
            result['has_config'] = True
            result['income_mode'] = config.get('donation_income_mode', config.get('income_spec_mode', 'Unknown'))
            result['source'] = 'Saved Configuration'
            result['is_saved'] = True
        if not result['has_config']:
            result['has_config'] = True
            result['income_mode'] = st.session_state.get('income_spec_mode', 'categorical only')
            result['source'] = 'Page 2 Settings'

    elif decision_name == 'disclose_income':
        config = get_decision_config('disclose_income')
        if config:
            result['has_config'] = True
            result['income_mode'] = config.get('income_mode', config.get('params', {}).get('income_mode', 'Unknown'))
            result['source'] = 'Saved Configuration'
            result['is_saved'] = True
        if not result['has_config']:
            result['has_config'] = True
            result['income_mode'] = st.session_state.get('di_income_mode', 'Categorical only')
            result['source'] = 'Page 2 Settings'
    elif decision_name == 'disclose_documents':
        config = get_decision_config('disclose_documents')
        if config:
            result['has_config'] = True
            result['income_mode'] = config.get('income_mode', config.get('params', {}).get('income_mode', 'Unknown'))
            result['source'] = 'Saved Configuration'
            result['is_saved'] = True
        if not result['has_config']:
            result['has_config'] = True
            result['income_mode'] = st.session_state.get('dd_income_mode', 'Categorical only')
            result['source'] = 'Page 2 Settings'
    elif decision_name == 'rejected_transaction_defaults':
        config = get_decision_config('rejected_transaction_defaults')
        if config:
            result['has_config'] = True
            income_mode = config.get('income_mode', config.get('params', {}).get('income_mode', 'Unknown'))
            population_mode = config.get('population_mode')
            result['income_mode'] = f"{population_mode} + {income_mode}" if population_mode else income_mode
            result['source'] = 'Saved Configuration'
            result['is_saved'] = True
        if not result['has_config']:
            result['has_config'] = True
            result['income_mode'] = st.session_state.get('rtd_income_mode', 'Continuous only')
            result['source'] = 'Page 2 Settings'

    if result['has_config'] and not result['is_saved'] and ctx is not None:
        ran_with = ctx.decision_income_label(decision_name)
        if ran_with:
            result['income_mode'] = ran_with
            result['source'] = 'Run metadata'

    return result


RTD_COMPARE_BOTH_NOTE = (
    "ℹ️ The Decision 4 tab is set to **Compare both**, which a complete simulation "
    "cannot split: Decision 4 ran with **continuous** income in this run. Run "
    "Decision 4 on its own to compare both income specifications.")


def render_rtd_compare_both_note(decision_name, ctx):
    """Complete run with the Decision 4 tab on 'Compare both': say that it ran continuous."""
    if decision_name == 'rejected_transaction_defaults' and ctx is not None and ctx.rtd_compare_both_fallback:
        st.info(RTD_COMPARE_BOTH_NOTE)


def render_rtd_element_subtitle(decision_name, decision_number):
    """Subtitle under the Decision 4 title when only ONE element was run (Lavie
    2026-10), e.g. '4.1 Options List Length (Tendency to Plan)' - the element's
    sub-tab number and name. Same heading class as the decision title (no extra
    weight or size); nothing for whole-decision or complete runs."""
    if decision_name != 'rejected_transaction_defaults':
        return
    from app.components import rtd_page_element
    from app.reports.rtd import rtd_element_subtitle
    subtitle = rtd_element_subtitle(decision_number, rtd_page_element())
    if subtitle:
        st.markdown(f'<h5 class="subsection-header" style="margin-top:0">{subtitle}</h5>',
                    unsafe_allow_html=True)


def render_decision_config_badge(decision_name, ctx=None):
    """Render a compact badge showing the selected configuration for a decision."""
    config_info = get_decision_config_display(decision_name, ctx)

    if not config_info['has_config']:
        return

    # Create a compact display
    if config_info['is_saved']:
        icon = "🎯"
        label = "Saved Config"
    else:
        icon = "⚙️"
        label = "Current Settings"

    income_mode = config_info['income_mode']
    if income_mode:
        st.caption(f"{icon} **{label}:** {income_mode}")


def render_results_page():
    """Render the Results page"""
    st.markdown('<h2 class="page-header">Simulation Results</h2>', unsafe_allow_html=True)

    # Display single run results
    if st.session_state.simulation_results is not None:
        render_single_run_results()

    # Display Monte Carlo results
    elif st.session_state.mc_results is not None:
        show_monte_carlo_results(st.session_state.mc_results)

    # Show message if no results available
    else:
        st.info("🔍 No simulation results available yet.")
        st.write("Please configure your simulation parameters and click '🚀 Run Complete Simulation' on the Decisions page.")

    # Always show navigation
    render_navigation('results')


def _render_overview_section(results_dict, ctx, has_explicit_donation_config):
    """Overview / comparison summary block (per-mode grids or single-mode metrics).

    Rendered AFTER the Decision Results for most decisions, but BEFORE them
    (summary-first) for individual Decision 4 runs.
    """
    has_combined_simulation = ctx.has_combined_simulation

    # Show results based on mode (but only if donation_default is not being shown in dropdown)
    is_comparison_mode = ctx.is_comparison_mode
    has_donation_default_in_dropdown = (
        has_combined_simulation and
        "donation_default" in ctx.custom_decisions and
        is_comparison_mode
    )
    if has_donation_default_in_dropdown:
        return

    # Check if we're using a selected configuration (should not show overview)
    using_selected_config = has_explicit_donation_config

    # Check if this is a donation_default custom parameters run (should not show overview)
    is_donation_custom_only = 'donation_default' in ctx.custom_decisions

    # Check if results actually have "Compare all" keys before rendering comparison
    has_compare_all_results = ctx.has_compare_all_results

    # CRITICAL: For combined/full simulations, NEVER show compare-all view
    is_full_simulation = has_combined_simulation

    if ctx.is_compare_all and has_compare_all_results and not is_full_simulation:
        render_all_modes_comparison(results_dict, ctx)
    elif ctx.is_compare_both and not is_full_simulation:
        render_income_comparison(results_dict, ctx)
    elif not is_full_simulation and not using_selected_config and not is_donation_custom_only:
        # Single mode display - show high-level summary
        st.markdown('<h3 class="section-header">📊 Simulation Overview</h3>', unsafe_allow_html=True)
        df = next(iter(results_dict.values()))
        mode_name = next(iter(results_dict.keys()))

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.metric("Total Agents", f"{len(df):,}")
        with col2:
            donation_col = 'donation_default'
            if donation_col in df.columns:
                st.metric("Avg Donation Rate", f"{df[donation_col].mean():.2%}")
            elif 'rtd_choice_length' in df.columns:
                # Decision 4 model run: run-shape-aware headline metric (a per-element
                # run of a ranking element has none - professor 2026-09)
                from app.components import rtd_overview_metric
                rtd_label, rtd_value = rtd_overview_metric(df)
                if rtd_label:
                    st.metric(rtd_label, rtd_value)

        if 'rtd_choice_length' in df.columns and 'donation_default' not in df.columns:
            st.caption(f"📊 Mode: {mode_name.title()}")
        else:
            st.caption(f"📊 Mode: {mode_name.title()} | Anchor mix: {st.session_state.anchor_observed_weight:.2f} observed | {1 - st.session_state.anchor_observed_weight:.2f} predicted")

        # Check if we should enable selection for individual donation runs (Decision 4's
        # "Use This Config" renders under its detailed results, not in this overview)
        enable_selection = ctx.is_individual_run('donation_default')

        show_overview(
            df,
            f" ({mode_name.title()})",
            result_key=mode_name,
            enable_selection=enable_selection
        )
    # For full simulations, selected configs, or donation custom runs - skip overview display entirely


def render_single_run_results():
    """Render single run simulation results"""

    # What was actually run (R28): modes, result keys, custom / default decisions
    ctx = RunContext.from_session()

    # Show saved configuration info if donation_default has an explicitly saved config
    dd_saved_config = get_decision_config('donation_default')
    _has_explicit_donation_config = dd_saved_config is not None

    if _has_explicit_donation_config:
        donation_income_mode = dd_saved_config.get('donation_income_mode', dd_saved_config.get('income_spec_mode', 'categorical only'))
        donation_population_mode = dd_saved_config.get('population_mode', ctx.effective_population_mode)
        st.info(f"🎯 **Donation Default used saved configuration:** {donation_population_mode} + {donation_income_mode}")

        # Also show disclose_income mode if it was MANUALLY configured
        di_was_manually_configured = 'disclose_income' in ctx.custom_decisions
        di_saved_config = get_decision_config('disclose_income')
        di_has_saved_config = di_saved_config is not None

        if di_was_manually_configured or di_has_saved_config:
            di_mode = None
            di_population_mode = None
            if di_has_saved_config:
                di_mode = di_saved_config.get('income_mode', di_saved_config.get('params', {}).get('income_mode'))
                di_population_mode = di_saved_config.get('population_mode')
            if di_mode is None:
                # No saved DI config: the mode Decision 1 actually ran with (recorded
                # per decision since 2026-10-07), else the run's own income mode
                di_mode = ctx.decision_income_label('disclose_income') or ctx.effective_income_mode
            if di_population_mode is None:
                di_population_mode = ctx.effective_population_mode

            st.info(f"📋 **Disclose Income used:** {di_population_mode} + {di_mode}")

    # Show decision configuration summary when we have both custom and default decisions (combined simulation)
    # OR when in single mode (not comparison modes)
    is_comparison_mode = ctx.is_comparison_mode

    # Show if: (not comparison mode) OR (has both custom and default decisions from combined simulation)
    has_combined_simulation = ctx.has_combined_simulation  # Only show if there are actual default decisions

    # Decision 4 has no comparison grid of its own, so an individual Decision 4 run must
    # keep the Decision Results section even in comparison modes (it then renders its
    # model view once per population mode / result key).
    is_individual_rtd_run = ctx.is_individual_run('rejected_transaction_defaults')

    # Individual Decision 4 layout depends on the run shape:
    # - SINGLE-MODE runs (one result key): SUMMARY-FIRST - the overview renders at the
    #   TOP of the page, before the detailed per-element sections (professor feedback:
    #   the summary was landing at the end of a long page).
    # - COMPARISON runs (multiple result keys / Compare-all keys): the summary must be
    #   INTERLEAVED per income treatment (title -> overview row -> detail row), which
    #   render_rtd_comparison_results does inside the Decision Results section - so no
    #   separate summary block is rendered up front here.
    _rtd_results = ctx.results
    is_rtd_comparison_run = is_individual_rtd_run and (
        len(_rtd_results) > 1 or ctx.has_compare_all_results)
    if is_individual_rtd_run and _rtd_results and not is_rtd_comparison_run:
        _render_overview_section(_rtd_results, ctx, _has_explicit_donation_config)

    if not is_comparison_mode or has_combined_simulation or is_individual_rtd_run:
        st.markdown('<h3 class="section-header">📋 Decision Results</h3>', unsafe_allow_html=True)

        # Create individual dropdowns for each decision with full results
        results_dict = st.session_state.simulation_results
        df = next(iter(results_dict.values())) if results_dict else pd.DataFrame()

        # Only show decisions that were actually executed, in chronological order
        # Use ALL_DECISIONS order to maintain chronological sequence
        executed_decisions = ctx.executed_decisions

        for decision in executed_decisions:
            # Get decision number and format title
            decision_number = ALL_DECISIONS.index(decision) + 1 if decision in ALL_DECISIONS else None

            # Special handling for decision names
            if decision == "purchase_vs_bid":
                decision_title = "Purchase Now Vs Bid"
            elif decision == "purchasing_quantity":
                decision_title = "Purchase Request Quantity"
            elif decision == "purchasing_frequency":
                decision_title = "Purchase Request Frequency"
            else:
                decision_title = decision.replace('_', ' ').title()

            # Add number prefix
            if decision_number is not None:
                decision_title = f"{decision_number}. {decision_title}"

            # Determine if this decision was customized or uses defaults
            if decision in ctx.custom_decisions:
                # Custom decision - show green checkmark

                # FIX: Check if this decision has a saved config
                # If it does, we should show results even in "comparison mode"
                # because the user explicitly selected a specific configuration
                decision_has_saved_config = False
                if decision in ['donation_default', 'disclose_income', 'disclose_documents']:
                    config_info = get_decision_config_display(decision, ctx)
                    decision_has_saved_config = config_info.get('is_saved', False)

                # Check if results actually have compare-all keys (only true for individual decision runs in Compare all mode)
                # Full/combined simulations always run in single mode due to can_run_complete_simulation() blocking
                results_have_compare_all_keys = ctx.has_compare_all_results

                # For combined simulations, results are always single-mode, so never show comparison views
                is_individual_decision_run = not has_combined_simulation

                # Single decision - show content directly (better UX)
                st.markdown(f'<h4 class="subsection-header">✅ {decision_title} (Custom Parameters)</h4>', unsafe_allow_html=True)
                render_rtd_element_subtitle(decision, decision_number)
                st.success("This decision was configured with custom parameters")
                # Show selected config badge for relevant decisions
                if decision in ['donation_default', 'disclose_income', 'disclose_documents',
                                'rejected_transaction_defaults']:
                    render_decision_config_badge(decision, ctx)
                render_rtd_compare_both_note(decision, ctx)

                # Show decision-specific results if available
                if not df.empty and decision in df.columns:
                    # FIX: If decision has a saved config, always show results
                    # (user selected a specific config, so we're not in true "comparison" anymore)
                    if decision_has_saved_config:
                        render_decision_results(df, decision, decision_title)
                    elif is_comparison_mode and decision == "donation_default" and is_individual_decision_run and results_have_compare_all_keys:
                        # For donation_default in comparison mode - only show comparison grids for individual runs with actual compare-all results
                        if ctx.is_compare_all:
                            render_all_modes_comparison(results_dict, ctx)
                        elif ctx.is_compare_both:
                            render_income_comparison(results_dict, ctx)
                    elif is_comparison_mode and decision == "rejected_transaction_defaults" and is_individual_decision_run:
                        # Decision 4: one tab per configuration, mirroring the other decisions' comparison labels
                        from app.pages.results.visualizations.transaction_viz import render_rtd_comparison_results
                        if not render_rtd_comparison_results(results_dict, decision):
                            st.info("📊 Custom decision results are shown in the comparison grids below")
                    elif is_comparison_mode and not has_combined_simulation:
                        st.info("📊 Custom decision results are shown in the comparison grids below")
                    else:
                        render_decision_results(df, decision, decision_title)
                else:
                    st.info("Results data not available for this decision")

            else:
                # Default decision - show gear icon
                # Single decision - show content directly (better UX)
                st.markdown(f'<h4 class="subsection-header">🔧 {decision_title} (Default Values)</h4>', unsafe_allow_html=True)
                default_description = get_dynamic_description(decision)
                st.info("This decision used default values since it was not selected for customization")
                st.write(f"**Default Behavior:** {default_description}")
                render_rtd_compare_both_note(decision, ctx)

                # Show decision-specific results if available
                if not df.empty and decision in df.columns:
                    st.markdown("**📊 Results with Default Values:**")
                    render_decision_results(df, decision, decision_title)
                else:
                    st.caption("💡 To see results and customize this decision, select it on Page 2")

        st.markdown("---")

    # Show parameter summary
    with st.expander("📊 Simulation Parameters Summary", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Time & Market**")
            st.write(f"- Periods: {st.session_state.sim_params.periods}")
            st.write(f"- Duration: {st.session_state.sim_params.duration_hours} hours/period")
            st.write(f"- Vendors: {st.session_state.sim_params.num_vendors}")
            st.write(f"- Market Price: ${st.session_state.sim_params.market_price:.2f}")

        with col2:
            st.markdown("**Product & Pricing**")
            st.write(f"- Products/Vendor: {st.session_state.sim_params.products_per_vendor}")
            st.write(f"- Bidding %: {st.session_state.sim_params.bidding_percentage:.0%}")
            st.write(f"- Platform Markup: {st.session_state.sim_params.platform_markup:.0%}")
            st.write(f"- Price Range: ±{st.session_state.sim_params.price_range:.0%}")

        with col3:
            st.markdown("**Income & Agents**")
            st.write(f"- Distribution: {st.session_state.sim_params.income_distribution}")
            st.write(f"- Range: ${st.session_state.sim_params.income_min:.0f} - ${st.session_state.sim_params.income_max:.0f}")
            st.write(f"- {st.session_state.sim_params.income_avg_type.title()}: ${st.session_state.sim_params.income_avg:.0f}")
            st.write(f"- Discount Threshold: ${st.session_state.sim_params.discount_income_threshold:.0f}")
            st.write(f"- Agents: {st.session_state.n_agents}")
            # R27: the number of decisions THIS run executed (custom + default)
            st.write(f"- Decisions: {ctx.num_decisions}")


    # Show results based on comparison mode
    results_dict = st.session_state.simulation_results

    # For individual Decision 4 runs the overview/summary was already rendered above:
    # single-mode runs at the TOP of the page, comparison runs interleaved per income
    # treatment inside the Decision Results section - do not repeat it here.
    #
    # Likewise, an individual decision run in SINGLE mode (single population mode +
    # single income spec) already rendered that decision's full results once in the
    # Decision Results section above (metrics, distribution, statistics,
    # classification). The generic income-type overview ("📊 Simulation Overview" +
    # show_overview's "... Analysis (Continuous/Categorical)") would repeat the exact
    # same histogram/statistics/classification, so it is skipped for that run shape.
    # Comparison runs ("Compare all" population mode / "Compare both" income spec)
    # are NOT skipped: there the overview section renders the comparison grids,
    # which are the primary display.
    _single_mode_df = next(iter(results_dict.values())) if results_dict else pd.DataFrame()
    is_individual_single_mode_run_with_results = (
        not is_comparison_mode and
        ctx.is_individual_decision_run and
        not _single_mode_df.empty and
        ctx.custom_decisions[0] in _single_mode_df.columns
    )
    if results_dict and not is_individual_rtd_run and not is_individual_single_mode_run_with_results:
        _render_overview_section(results_dict, ctx, _has_explicit_donation_config)

    # Configuration selection UI - shows config cards and "Run Complete Simulation" button
    render_configuration_selection_ui(results_dict, ctx)

    # Disclose Income configuration selection UI
    render_disclose_income_config_selection_ui(results_dict, ctx)

    # Disclose Documents configuration selection UI
    render_disclose_documents_config_selection_ui(results_dict, ctx)

    # Decision 4 (Rejected Transaction Defaults) configuration selection UI: selected
    # configuration + "Run Complete Simulation", like the decisions above
    render_rejected_transaction_config_selection_ui(results_dict, ctx)

    # Get DataFrame for individual agent analysis: the frame the run's own shape
    # points at (compare-all / compare-both keys), falling back to any available one
    df = ctx.primary_df

    # Individual agent details
    if st.session_state.show_individual_agents and not df.empty:
        render_individual_agent_details(df)

    # Raw data download
    if not df.empty:
        # Detect if this is an individual decision run (single decision, no defaults).
        # For individual decision runs, saved config state should NOT filter the export --
        # all computed configs should always be available for download.
        _is_individual_decision_run = ctx.is_individual_decision_run

        if _is_individual_decision_run:
            # Individual decision runs: always pass full results_dict, never filter by saved config.
            # Saved configs are for the "Run Complete Simulation" workflow, not for export filtering.
            render_export_section(df, results_dict=results_dict, using_selected_config=False)
        else:
            # Combined/full simulation runs: apply donation-specific config logic
            using_selected_config_from_sim = _has_explicit_donation_config

            # Check if user has a selected config
            has_selected_config = has_decision_config('donation_default')

            # Determine which results to export
            if has_selected_config and not using_selected_config_from_sim:
                # User selected a config from current results - export only that config
                selected_key = dd_saved_config.get('result_key') if dd_saved_config else None

                # Filter results_dict to only include selected config AND update df to match
                if selected_key and selected_key in results_dict:
                    filtered_results = {selected_key: results_dict[selected_key]}
                    selected_df = results_dict[selected_key]  # Use the selected config's DataFrame
                    render_export_section(selected_df, results_dict=filtered_results, using_selected_config=True)
                else:
                    # Selected key not found, export all
                    render_export_section(df, results_dict=results_dict, using_selected_config=False)
            else:
                # Pass full results_dict for multi-config export (if not using selected config)
                render_export_section(df, results_dict=results_dict, using_selected_config=using_selected_config_from_sim)

    # No flag cleanup needed - we read directly from unified config storage
