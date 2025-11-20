# app/pages/results/main_results.py
"""
Main results page rendering for the Enhanced AI Agent Simulation.
"""
import streamlit as st
import pandas as pd
from app.pages.navigation import render_navigation
from app.components import show_overview, show_monte_carlo_results
from app.pages.results.comparisons import (
    render_all_modes_comparison,
    render_dependent_variable_results,
    render_income_comparison,
    render_population_comparison
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
    DEFAULT_DECISION_DESCRIPTIONS
)
from app.models import ALL_DECISIONS

# Import from new modules
from app.pages.results.decision_visualizations import (
    render_decision_results,
    DECISION_VISUALIZATIONS,
    get_dynamic_description
)
from app.pages.results.config_selection import (
    render_configuration_selection_ui,
    render_configuration_card,
    extract_configuration_details_from_key
)


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
        st.write("Please configure your simulation parameters and click '🚀 Run Simulation' in the sidebar.")
    
    # Always show navigation
    render_navigation('results')


def render_single_run_results():
    """Render single run simulation results"""
    
    # Check if we're using a selected configuration (don't clear flag yet - needed later)
    if hasattr(st.session_state, '_using_selected_config') and st.session_state._using_selected_config:
        config = st.session_state.selected_donation_config
        st.info(f"🎯 **Results using selected donation configuration:** {config['population_mode']} + {config['income_spec_mode']}")
    
    # Show decision configuration summary when we have both custom and default decisions (combined simulation)
    # OR when in single mode (not comparison modes)
    is_comparison_mode = (
        st.session_state.population_mode == "Compare all" or
        st.session_state.population_mode == "Dependent variable resampling" or
        st.session_state.income_spec_mode == "Compare both"
    )
    
    # Show if: (not comparison mode) OR (has both custom and default decisions from combined simulation)
    has_combined_simulation = (
        hasattr(st.session_state, 'custom_decisions') and 
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) > 0  # Only show if there are actual default decisions
    )
    
    if (not is_comparison_mode or has_combined_simulation) and hasattr(st.session_state, 'custom_decisions') and hasattr(st.session_state, 'default_decisions'):
        st.markdown('<h3 class="section-header">📋 Decision Results</h3>', unsafe_allow_html=True)
        
        # Create individual dropdowns for each decision with full results
        results_dict = st.session_state.simulation_results
        df = next(iter(results_dict.values())) if results_dict else pd.DataFrame()
        
        # Only show decisions that were actually executed, in chronological order
        # Use ALL_DECISIONS order to maintain chronological sequence
        all_executed = set(st.session_state.custom_decisions + st.session_state.default_decisions)
        executed_decisions = [d for d in ALL_DECISIONS if d in all_executed]
        
        # Don't use dropdown - show each decision individually
        use_dropdown = False
        
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
            if decision in st.session_state.custom_decisions:
                # Custom decision - show green checkmark
                if use_dropdown:
                    # Multiple decisions - use collapsible dropdown
                    with st.expander(f"✅ {decision_title} (Custom Parameters)", expanded=False):
                        st.success("This decision was configured with custom parameters on Page 2")
                        st.write("**Configuration Source:** Page 2 Decision Tab")
                        
                        # Show decision-specific results if available
                        if not df.empty and decision in df.columns:
                            if is_comparison_mode and decision == "donation_default":
                                # For donation_default in comparison mode, show the actual comparison grids
                                if st.session_state.population_mode == "Compare all":
                                    render_all_modes_comparison(results_dict)
                                elif st.session_state.income_spec_mode == "Compare both":
                                    render_income_comparison(results_dict)
                            elif is_comparison_mode:
                                st.info("📊 Custom decision results are shown in the comparison grids below")
                            else:
                                render_decision_results(df, decision, decision_title)
                        else:
                            st.info("Results data not available for this decision")
                else:
                    # Single decision - show content directly (better UX)
                    st.markdown(f'<h4 class="subsection-header">✅ {decision_title} (Custom Parameters)</h4>', unsafe_allow_html=True)
                    st.success("This decision was configured with custom parameters on Page 2")
                    st.write("**Configuration Source:** Page 2 Decision Tab")
                    
                    # Show decision-specific results if available
                    if not df.empty and decision in df.columns:
                        if is_comparison_mode and decision == "donation_default":
                            # For donation_default in comparison mode, show the actual comparison grids
                            if st.session_state.population_mode == "Compare all":
                                render_all_modes_comparison(results_dict)
                            elif st.session_state.income_spec_mode == "Compare both":
                                render_income_comparison(results_dict)
                        elif is_comparison_mode:
                            st.info("📊 Custom decision results are shown in the comparison grids below")
                        else:
                            render_decision_results(df, decision, decision_title)
                    else:
                        st.info("Results data not available for this decision")
                        
            else:
                # Default decision - show gear icon
                if use_dropdown:
                    # Multiple decisions - use collapsible dropdown
                    with st.expander(f"🔧 {decision_title} (Default Values)", expanded=False):
                        default_description = get_dynamic_description(decision)
                        st.info("This decision used default values since it was not selected for customization")
                        st.write(f"**Default Behavior:** {default_description}")
                        
                        # Show decision-specific results if available
                        if not df.empty and decision in df.columns:
                            render_decision_results(df, decision, decision_title)
                        else:
                            st.caption("💡 To see results and customize this decision, select it on Page 2")
                else:
                    # Single decision - show content directly (better UX)
                    st.markdown(f'<h4 class="subsection-header">🔧 {decision_title} (Default Values)</h4>', unsafe_allow_html=True)
                    default_description = get_dynamic_description(decision)
                    st.info("This decision used default values since it was not selected for customization")
                    st.write(f"**Default Behavior:** {default_description}")
                    
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
            st.write(f"- Decisions: {len(st.session_state.decision_params.selected_decisions)}")
    
    
    # Show results based on comparison mode
    results_dict = st.session_state.simulation_results
    
    if results_dict:
        # Show results based on mode (but only if donation_default is not being shown in dropdown)
        has_donation_default_in_dropdown = (
            has_combined_simulation and 
            "donation_default" in st.session_state.custom_decisions and
            is_comparison_mode
        )
        
        if not has_donation_default_in_dropdown:
            # Check if we're using a selected configuration (should not show overview)
            using_selected_config = hasattr(st.session_state, '_using_selected_config') and st.session_state._using_selected_config
            
            # Check if this is a donation_default custom parameters run (should not show overview)
            is_donation_custom_only = (
                hasattr(st.session_state, 'custom_decisions') and 
                'donation_default' in st.session_state.custom_decisions
            )
            
            if st.session_state.population_mode == "Compare all":
                render_all_modes_comparison(results_dict)
            elif st.session_state.population_mode == "Dependent variable resampling":
                render_dependent_variable_results(results_dict)
            elif st.session_state.income_spec_mode == "Compare both":
                render_income_comparison(results_dict)
            elif not using_selected_config and not is_donation_custom_only:
                # Single mode display - show high-level summary (but NOT for selected configurations or donation custom runs)
                st.markdown('<h3 class="section-header">📊 Simulation Overview</h3>', unsafe_allow_html=True)
                df = next(iter(results_dict.values()))
                mode_name = next(iter(results_dict.keys()))
                
                # Show high-level metrics only
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])
                
                with col1:
                    st.metric("Total Agents", f"{len(df):,}")
                
                with col2:
                    trait_cols = ['Assigned Allowance Level', 'Group_experiment', 'Honesty_Humility', 
                                 'Study Program', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
                    st.metric("Traits Available", len([c for c in trait_cols if c in df.columns]))
                
                with col3:
                    decision_cols = [c for c in df.columns if c not in trait_cols]
                    st.metric("Decisions Computed", len(decision_cols))
                
                with col4:
                    # Show overall donation rate if available - always use truncated
                    donation_col = 'donation_default'
                    if donation_col in df.columns:
                        st.metric("Avg Donation Rate", f"{df[donation_col].mean():.2%}")
                
                st.caption(f"📊 Mode: {mode_name.title()} | Anchor mix: {st.session_state.anchor_observed_weight:.2f} observed | {1 - st.session_state.anchor_observed_weight:.2f} predicted")
                
                # For single mode, also show the overview
                
                # Check if we should enable selection for individual donation runs
                enable_selection = (
                    hasattr(st.session_state, 'custom_decisions') and 
                    st.session_state.custom_decisions == ['donation_default'] and
                    hasattr(st.session_state, 'default_decisions') and
                    len(st.session_state.default_decisions) == 0
                )
                
                show_overview(
                    df, 
                    f" ({mode_name.title()})",
                    result_key=mode_name,
                    enable_selection=enable_selection
                )
            # If using_selected_config is True or is_donation_custom_only is True, we skip the overview display entirely
    
    # Configuration selection UI - shows config cards and "Run Complete Simulation" button
    render_configuration_selection_ui(results_dict)
    
    # Get DataFrame for individual agent analysis
    if st.session_state.population_mode == "Compare all":
        if st.session_state.income_spec_mode == "Compare both":
            df = next((results_dict[k] for k in ["copula_categorical", "research_spec_categorical", "research_baseline_categorical", "copula_continuous", "research_spec_continuous", "research_baseline_continuous"] if k in results_dict), pd.DataFrame())
        else:
            income_type = "continuous" if st.session_state.income_spec_mode == "continuous only" else "categorical"
            df = next((results_dict[k] for k in [f"copula_{income_type}", f"research_spec_{income_type}", f"research_baseline_{income_type}"] if k in results_dict), pd.DataFrame())
    elif st.session_state.income_spec_mode == "Compare both":
        df = next((results_dict[k] for k in ["categorical", "continuous"] if k in results_dict), pd.DataFrame())
    else:
        df = next(iter(results_dict.values()))
    
    # Individual agent details
    if st.session_state.show_individual_agents and not df.empty:
        render_individual_agent_details(df)
    
    # Raw data download
    if not df.empty:
        # Check if we're using a selected configuration (from new simulation)
        using_selected_config_from_sim = (
            hasattr(st.session_state, '_using_selected_config') and 
            st.session_state._using_selected_config
        )
        
        # ALSO check if user just selected a config from current results (without re-running)
        # In this case, filter results_dict to only show the selected config
        has_selected_config = hasattr(st.session_state, 'selected_donation_config')
        
        # Determine which results to export
        if has_selected_config and not using_selected_config_from_sim:
            # User selected a config from current results - export only that config
            selected_key = st.session_state.selected_donation_config.get('result_key')
            
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
    
    # Clear the selected config flag at the very end
    if hasattr(st.session_state, '_using_selected_config'):
        delattr(st.session_state, '_using_selected_config')
