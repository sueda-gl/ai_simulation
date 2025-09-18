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
    render_income_comparison
)
from app.pages.results.details import (
    render_individual_agent_details,
    render_export_section
)
import plotly.express as px


def render_decision_results(df, decision_name, decision_title):
    """Render detailed results for a specific decision"""
    if decision_name not in df.columns:
        st.warning(f"No results available for {decision_title}")
        return
    
    decision_data = df[decision_name]
    
    # Check if data is numeric
    try:
        # Try to convert to numeric, coercing errors to NaN
        numeric_data = pd.to_numeric(decision_data, errors='coerce')
        is_numeric = not numeric_data.isna().all()
    except:
        is_numeric = False
    
    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    if is_numeric:
        # Numeric decision - show statistical metrics
        with col2:
            st.metric("Mean", f"{numeric_data.mean():.1%}" if decision_name == "donation_default" else f"{numeric_data.mean():.2f}")
        
        with col3:
            st.metric("Std Dev", f"{numeric_data.std():.3f}")
        
        with col4:
            st.metric("Range", f"{numeric_data.min():.1%} - {numeric_data.max():.1%}" if decision_name == "donation_default" else f"{numeric_data.min():.2f} - {numeric_data.max():.2f}")
        
        # Show distribution plot and statistics
        col_plot, col_stats = st.columns([2, 1])
        
        with col_plot:
            # Create histogram for numeric data
            fig = px.histogram(
                df, 
                x=decision_name,
                nbins=30,
                title=f"Distribution of {decision_title}",
                labels={decision_name: decision_title, 'count': 'Number of Agents'},
                marginal="box"
            )
            
            # Format based on decision type
            if decision_name == "donation_default":
                fig.update_layout(xaxis_tickformat='.0%')
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
        
        with col_stats:
            st.markdown("**📈 Statistics**")
            stats = numeric_data.describe()
            
            # Format statistics based on decision type
            if decision_name == "donation_default":
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
                    'Value': [
                        f"{stats['mean']:.1%}",
                        f"{stats['std']:.3f}",
                        f"{stats['min']:.1%}",
                        f"{stats['max']:.1%}",
                        f"{stats['50%']:.1%}",
                        f"{stats['25%']:.1%}",
                        f"{stats['75%']:.1%}"
                    ]
                })
            else:
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
                    'Value': [
                        f"{stats['mean']:.2f}",
                        f"{stats['std']:.3f}",
                        f"{stats['min']:.2f}",
                        f"{stats['max']:.2f}",
                        f"{stats['50%']:.2f}",
                        f"{stats['25%']:.2f}",
                        f"{stats['75%']:.2f}"
                    ]
                })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    else:
        # Non-numeric decision - show categorical analysis
        with col2:
            # Handle unhashable types (like dictionaries) safely
            try:
                # Convert complex types to strings for counting
                string_data = decision_data.astype(str)
                unique_values = string_data.nunique()
                st.metric("Unique Values", f"{unique_values}")
            except Exception:
                st.metric("Unique Values", "N/A")
        
        with col3:
            try:
                # Convert to string for mode calculation
                string_data = decision_data.astype(str)
                most_common = string_data.mode().iloc[0] if len(string_data.mode()) > 0 else "N/A"
                st.metric("Most Common", str(most_common)[:20] + "..." if len(str(most_common)) > 20 else str(most_common))
            except Exception:
                st.metric("Most Common", "N/A")
        
        with col4:
            non_na_count = decision_data.notna().sum()
            st.metric("Valid Values", f"{non_na_count:,}")
        
        # Show value counts and distribution
        col_plot, col_stats = st.columns([2, 1])
        
        with col_plot:
            try:
                # Convert complex types to strings for value counting
                string_data = decision_data.astype(str)
                value_counts = string_data.value_counts().head(10)  # Top 10 values
                
                if len(value_counts) > 0:
                    fig = px.bar(
                        x=value_counts.index.astype(str),
                        y=value_counts.values,
                        title=f"Distribution of {decision_title}",
                        labels={'x': decision_title, 'y': 'Count'}
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No valid data to display")
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.info("Data contains complex types that cannot be visualized directly")
        
        with col_stats:
            st.markdown("**📈 Value Counts**")
            try:
                # Convert complex types to strings for value counting
                string_data = decision_data.astype(str)
                value_counts = string_data.value_counts().head(10)
                
                if len(value_counts) > 0:
                    # Show top values
                    counts_df = pd.DataFrame({
                        'Value': value_counts.index.astype(str),
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(decision_data) * 100).round(1)
                    })
                    st.dataframe(counts_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No valid data available")
            except Exception as e:
                st.error(f"Error calculating value counts: {str(e)}")
                
                # For dictionary data, show a special format
                if any(isinstance(val, dict) for val in decision_data.dropna()):
                    st.markdown("**Dictionary Data Detected:**")
                    sample_dict = next((val for val in decision_data.dropna() if isinstance(val, dict)), None)
                    if sample_dict:
                        st.json(sample_dict)


def render_results_page():
    """Render the Results page"""
    st.markdown('<h2 class="page-header">Simulation Results</h2>', unsafe_allow_html=True)
    
    # Debug info
    with st.expander("🔧 Debug: Session State", expanded=False):
        st.write(f"simulation_results: {'Yes' if st.session_state.simulation_results is not None else 'No'}")
        st.write(f"mc_results: {'Yes' if st.session_state.mc_results is not None else 'No'}")
        
        # Debug decision state variables
        st.write("**Decision State Variables:**")
        if hasattr(st.session_state, 'custom_decisions'):
            st.write(f"custom_decisions: {st.session_state.custom_decisions}")
        else:
            st.write("custom_decisions: NOT SET")
            
        if hasattr(st.session_state, 'default_decisions'):
            st.write(f"default_decisions: {st.session_state.default_decisions}")
        else:
            st.write("default_decisions: NOT SET")
            
        # Show actual results columns
        if st.session_state.simulation_results:
            results_dict = st.session_state.simulation_results
            df = next(iter(results_dict.values())) if results_dict else pd.DataFrame()
            st.write(f"**Actual Result Columns:** {list(df.columns) if not df.empty else 'No data'}")
            
        if st.session_state.mc_results is not None:
            st.write(f"mc_results keys: {list(st.session_state.mc_results.keys())}")
            st.write(f"summary shape: {st.session_state.mc_results['summary'].shape if st.session_state.mc_results['summary'] is not None else 'None'}")
            st.write(f"detailed shape: {st.session_state.mc_results['detailed'].shape if st.session_state.mc_results['detailed'] is not None else 'None'}")
            
        # Add clear session state button
        if st.button("🗑️ Clear Session State (Fix Stale Data)"):
            # Clear the problematic state variables
            if hasattr(st.session_state, 'custom_decisions'):
                delattr(st.session_state, 'custom_decisions')
            if hasattr(st.session_state, 'default_decisions'):
                delattr(st.session_state, 'default_decisions')
            st.success("✅ Cleared decision state variables. Try running your simulation again.")
            st.rerun()
    
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
    
    # Show decision configuration summary
    if hasattr(st.session_state, 'custom_decisions') and hasattr(st.session_state, 'default_decisions'):
        st.markdown('<h3 class="section-header">📋 Decision Configuration Summary</h3>', unsafe_allow_html=True)
        st.caption("Expand any decision to view its configuration details")
        
        from app.pages.decision_execution import DEFAULT_DECISION_VALUES, DEFAULT_DECISION_DESCRIPTIONS
        from app.models import ALL_DECISIONS
        
        # Create individual dropdowns for each decision with full results
        results_dict = st.session_state.simulation_results
        df = next(iter(results_dict.values())) if results_dict else pd.DataFrame()
        
        # Only show decisions that were actually executed
        executed_decisions = st.session_state.custom_decisions + st.session_state.default_decisions
        
        # Smart UI: Use dropdown only when there are multiple decisions
        use_dropdown = len(executed_decisions) > 1
        
        for decision in executed_decisions:
            decision_title = decision.replace('_', ' ').title()
            
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
                        render_decision_results(df, decision, decision_title)
                    else:
                        st.info("Results data not available for this decision")
                        
            else:
                # Default decision - show gear icon
                if use_dropdown:
                    # Multiple decisions - use collapsible dropdown
                    with st.expander(f"🔧 {decision_title} (Default Values)", expanded=False):
                        default_description = DEFAULT_DECISION_DESCRIPTIONS.get(decision, "Standard default behavior")
                        st.info("This decision used default values since it was not selected for customization")
                        st.write(f"**Default Behavior:** {default_description}")
                        
                        # Show decision-specific results if available
                        if not df.empty and decision in df.columns:
                            st.markdown("**📊 Results with Default Values:**")
                            render_decision_results(df, decision, decision_title)
                        else:
                            st.caption("💡 To see results and customize this decision, select it on Page 2")
                else:
                    # Single decision - show content directly (better UX)
                    st.markdown(f'<h4 class="subsection-header">🔧 {decision_title} (Default Values)</h4>', unsafe_allow_html=True)
                    default_description = DEFAULT_DECISION_DESCRIPTIONS.get(decision, "Standard default behavior")
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
    
    
    # Show high-level summary only (detailed results are in decision dropdowns above)
    results_dict = st.session_state.simulation_results
    
    if results_dict:
        st.markdown('<h3 class="section-header">📊 Simulation Overview</h3>', unsafe_allow_html=True)
        df = next(iter(results_dict.values()))
        mode_name = next(iter(results_dict.keys()))
        
        # Show high-level metrics only
        col1, col2, col3, col4 = st.columns(4)
        
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
            # Show overall donation rate if available
            donation_col = 'donation_default_raw' if 'donation_default_raw' in df.columns else 'donation_default'
            if donation_col in df.columns:
                st.metric("Avg Donation Rate", f"{df[donation_col].mean():.1%}")
        
        st.caption(f"📊 Mode: {mode_name.title()} | Anchor mix: {st.session_state.anchor_observed_weight:.2f} observed | {1 - st.session_state.anchor_observed_weight:.2f} predicted")
    
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
        render_export_section(df)
