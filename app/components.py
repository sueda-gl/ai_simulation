# app/components.py
"""
UI components and visualization functions for the Enhanced AI Agent Simulation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import yaml

from app.models import get_decision_global_parameters, get_all_global_parameters


def show_overview(df, title_suffix=""):
    """Helper function to show simulation overview for a DataFrame"""
    st.subheader(f"Simulation Overview{title_suffix}")
    
    # Check if this is dependent variable mode (only has donation_default column)
    is_depvar_mode = len(df.columns) == 1 and 'donation_default' in df.columns
    
    # Display anchor weights info (not for depvar mode)
    if not is_depvar_mode:
        st.caption(f"📊 Anchor mix: {st.session_state.anchor_observed_weight:.2f} observed | {1 - st.session_state.anchor_observed_weight:.2f} predicted")
    else:
        st.caption("📊 Resampling from empirical distribution of 280 original donation rates")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(df):,}")
    
    with col2:
        if not is_depvar_mode:
            trait_cols = ['Assigned Allowance Level', 'Group_experiment', 'Honesty_Humility', 
                         'Study Program', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
            st.metric("Traits Available", len([c for c in trait_cols if c in df.columns]))
        else:
            st.metric("Source", "280 participants")
    
    with col3:
        if not is_depvar_mode:
            decision_cols = [c for c in df.columns if c not in trait_cols]
            st.metric("Decisions Computed", len(decision_cols))
        else:
            st.metric("Method", "Bootstrap")
    
    with col4:
        # Determine which column to use for display
        donation_col = 'donation_default_raw' if 'donation_default_raw' in df.columns else 'donation_default'
        if donation_col in df.columns:
            avg_label = "Avg Donation Rate" + (" (raw)" if donation_col == 'donation_default_raw' else "")
            st.metric(avg_label, f"{df[donation_col].mean():.1%}")
    
    # Donation rate analysis (if available)
    donation_col = 'donation_default_raw' if 'donation_default_raw' in df.columns else 'donation_default'
    if donation_col in df.columns:
        raw_suffix = " (raw pre-truncation)" if donation_col == 'donation_default_raw' else ""
        st.subheader(f" Donation Rate Analysis{title_suffix}{raw_suffix}")
        
        # Distribution plot
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(
                df, 
                x=donation_col,
                nbins=30,
                title=f"Distribution of Donation Rates{title_suffix}{raw_suffix}",
                labels={donation_col: 'Donation Rate', 'count': 'Number of Agents'},
                marginal="box"
            )
            fig.update_layout(
                xaxis_tickformat='.0%',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📈 Statistics**")
            donation_stats = df[donation_col].describe()
            
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
                'Value': [
                    f"{donation_stats['mean']:.1%}",
                    f"{donation_stats['std']:.3f}",
                    f"{donation_stats['min']:.1%}",
                    f"{donation_stats['max']:.1%}",
                    f"{donation_stats['50%']:.1%}",
                    f"{donation_stats['25%']:.1%}",
                    f"{donation_stats['75%']:.1%}"
                ]
            })
            st.dataframe(stats_df, hide_index=True)


def show_parameter_applicability_analysis(selected_decisions):
    """Show parameter applicability analysis for selected decisions"""
    st.markdown('<h3 class="section-header">📋 Parameter Applicability Analysis</h3>', unsafe_allow_html=True)
    
    # Show overall summary
    total_applicable = get_decision_global_parameters(selected_decisions)
    all_global_params = get_all_global_parameters()
    total_not_applicable = all_global_params - total_applicable
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Parameters", len(all_global_params))
    with col2:
        st.metric("✅ Applicable", len(total_applicable))
    with col3:
        st.metric("❌ Not Applicable", len(total_not_applicable))
    
    # Show parameter breakdown
    with st.expander("🔍 Parameter Breakdown", expanded=False):
        col_app, col_not_app = st.columns(2)
        
        with col_app:
            st.markdown("### ✅ Applicable Parameters")
            if total_applicable:
                for param in sorted(total_applicable):
                    st.markdown(f"  • {param.replace('_', ' ').title()}")
            else:
                st.markdown("None")
        
        with col_not_app:
            st.markdown("### ❌ Not Applicable Parameters")
            if total_not_applicable:
                for param in sorted(total_not_applicable):
                    st.markdown(f"  • {param.replace('_', ' ').title()}")
            else:
                st.markdown("None")
    
    # Show decision-specific analysis
    with st.expander("📊 Decision-Specific Parameter Analysis", expanded=False):
        try:
            decisions_path = Path(__file__).resolve().parents[1] / "config" / "decisions.yaml"
            with open(decisions_path, 'r') as f:
                decisions_config = yaml.safe_load(f)
            
            for decision in selected_decisions:
                decision_config = decisions_config.get(decision, {})
                decision_global_params = set(decision_config.get('uses_global_parameters', []))
                all_params = get_all_global_parameters()
                not_used_params = all_params - decision_global_params
                
                st.markdown(f"**{decision.replace('_', ' ').title()}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Uses Global Params", len(decision_global_params))
                with col2:
                    st.metric("Doesn't Use", len(not_used_params))
                with col3:
                    efficiency = len(decision_global_params) / len(all_params) * 100 if all_params else 0
                    st.metric("Usage %", f"{efficiency:.0f}%")
                
                if decision_global_params:
                    st.markdown("✅ **Uses Global Parameters:**")
                    formatted_params = [p.replace('_', ' ').title() for p in sorted(decision_global_params)]
                    st.markdown(f"  {', '.join(formatted_params)}")
                else:
                    st.markdown("✅ **Uses Global Parameters:** None (trait-based decision)")
                
                st.markdown("---")
                
        except Exception as e:
            st.error(f"Error loading decision configurations: {e}")


def show_monte_carlo_results(mc_data):
    """Display Monte Carlo simulation results"""
    if mc_data['summary'] is not None:
        st.subheader("📈 Monte-Carlo Analysis Results")
        
        summary_df = mc_data['summary']
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
    
        if 'donation_default' in summary_df['decision'].values:
            donation_row = summary_df[summary_df['decision'] == 'donation_default'].iloc[0]
        
            with col1:
                st.metric("Mean Donation Rate", f"{donation_row['mean']:.1%}")
            
            with col2:
                st.metric("Standard Deviation", f"{donation_row['std']:.4f}")
            
            with col3:
                st.metric("95% CI Lower", f"{donation_row['p2.5']:.1%}")
            
            with col4:
                st.metric("95% CI Upper", f"{donation_row['p97.5']:.1%}")
    
        # Monte-Carlo convergence plot
        if mc_data['detailed'] is not None:
            detailed_df = mc_data['detailed']
            
            if 'donation_default_mean' in detailed_df.columns:
                st.subheader("📊 Monte-Carlo Convergence")
                
                # Calculate running average
                detailed_df['running_mean'] = detailed_df['donation_default_mean'].expanding().mean()
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Individual Run Results", "Running Average Convergence"),
                    vertical_spacing=0.1
                )
                
                # Individual runs
                fig.add_trace(
                    go.Scatter(
                        x=detailed_df['run'] + 1,
                        y=detailed_df['donation_default_mean'],
                        mode='markers+lines',
                        name='Individual Runs',
                        line=dict(color='lightblue', width=1),
                        marker=dict(size=4)
                    ),
                    row=1, col=1
                )
                
                # Running average
                fig.add_trace(
                    go.Scatter(
                        x=detailed_df['run'] + 1,
                        y=detailed_df['running_mean'],
                        mode='lines',
                        name='Running Average',
                        line=dict(color='red', width=2)
                    ),
                    row=2, col=1
                )
                
                # Add confidence interval
                if len(detailed_df) > 1:
                    final_mean = detailed_df['running_mean'].iloc[-1]
                    final_std = detailed_df['donation_default_mean'].std()
                    ci_upper = final_mean + 1.96 * final_std / np.sqrt(len(detailed_df))
                    ci_lower = final_mean - 1.96 * final_std / np.sqrt(len(detailed_df))
                    
                    fig.add_hline(y=ci_upper, line_dash="dash", line_color="gray", row=2, col=1)
                    fig.add_hline(y=ci_lower, line_dash="dash", line_color="gray", row=2, col=1)
                
                fig.update_layout(
                    height=600,
                    title="Monte-Carlo Study Results",
                    showlegend=True
                )
                fig.update_yaxes(tickformat='.1%')
                fig.update_xaxes(title="Run Number")
                
                st.plotly_chart(fig, use_container_width=True)
    
        # Summary statistics table
        st.subheader("📋 Summary Statistics")
    
        # Format the summary table for display
        display_summary = summary_df.copy()
        for col in ['mean', 'p2.5', 'p97.5']:
            if col in display_summary.columns:
                display_summary[col] = display_summary[col].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_summary, use_container_width=True)
    
        # Download Monte-Carlo results
        st.subheader("💾 Export Monte-Carlo Results")
        
        col1, col2, col3 = st.columns(3)
    
        with col1:
            if mc_data['summary'] is not None:
                summary_csv = mc_data['summary'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary",
                    data=summary_csv,
                    file_name=f"monte_carlo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if mc_data['detailed'] is not None:
                detailed_csv = mc_data['detailed'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Detailed",
                    data=detailed_csv,
                    file_name=f"monte_carlo_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🔄 Clear Monte-Carlo Results"):
                st.session_state.mc_results = None
                st.rerun()
    
        # Show log output
        if mc_data['log']:
            with st.expander("📋 Monte-Carlo Execution Log", expanded=False):
                st.text(mc_data['log'])


def show_dependent_variable_comparison(df):
    """Show comparison between original and resampled distributions for dependent variable mode"""
    try:
        from src.orchestrator_depvar import OrchestratorDepVar
        temp_orch = OrchestratorDepVar()
        # Set the same raw output mode as was used for simulation
        temp_orch.set_raw_output(st.session_state.raw_draw_mode)
        emp_stats = temp_orch.get_empirical_stats()
        original_donations = temp_orch.get_empirical_distribution()
        
        # Determine the column name based on raw mode
        donation_col = 'donation_default_raw' if st.session_state.raw_draw_mode else 'donation_default'
        
        # Create two columns for side-by-side comparison
        col_orig, col_resamp = st.columns(2)
        
        with col_orig:
            st.subheader("📊 Original 280 Participants")
            
            # Stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean", f"{emp_stats['mean']:.1%}")
                st.metric("Min", f"{emp_stats['min']:.1%}")
            with col2:
                st.metric("Std Dev", f"{emp_stats['std']:.4f}")
                st.metric("Max", f"{emp_stats['max']:.1%}")
            
            # Histogram
            fig_orig = px.histogram(
                pd.DataFrame({donation_col: original_donations}),
                x=donation_col,
                nbins=30,
                title="Original Distribution (n=280)",
                labels={donation_col: 'Donation Rate', 'count': 'Number of Participants'},
                marginal="box"
            )
            fig_orig.update_layout(
                xaxis_tickformat='.0%',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_orig, use_container_width=True)
        
        with col_resamp:
            st.subheader(f"📊 Resampled ({len(df):,} agents)")
            
            # Stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean", f"{df[donation_col].mean():.1%}")
                st.metric("Min", f"{df[donation_col].min():.1%}")
            with col2:
                st.metric("Std Dev", f"{df[donation_col].std():.4f}")
                st.metric("Max", f"{df[donation_col].max():.1%}")
            
            # Histogram
            fig_resamp = px.histogram(
                df,
                x=donation_col,
                nbins=30,
                title=f"Resampled Distribution (n={len(df):,})",
                labels={donation_col: 'Donation Rate', 'count': 'Number of Agents'},
                marginal="box"
            )
            fig_resamp.update_layout(
                xaxis_tickformat='.0%',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_resamp, use_container_width=True)
        
        # Additional info
        st.caption(f"The resampled distribution is created by bootstrap sampling with replacement from the {len(original_donations)} original donation rates.")
        
        # Combined comparison plot
        st.subheader("📊 Distribution Comparison")
        
        # Create combined dataframe for comparison
        orig_df = pd.DataFrame({
            donation_col: original_donations,
            'source': 'Original (n=280)'
        })
        resamp_df = pd.DataFrame({
            donation_col: df[donation_col].values,
            'source': f'Resampled (n={len(df):,})'
        })
        combined_df = pd.concat([orig_df, resamp_df])
        
        # Create overlaid histogram
        fig_combined = px.histogram(
            combined_df,
            x=donation_col,
            color='source',
            nbins=30,
            barmode='overlay',
            opacity=0.7,
            title="Original vs Resampled Distribution Overlay",
            labels={donation_col: 'Donation Rate', 'count': 'Count'}
        )
        fig_combined.update_layout(
            xaxis_tickformat='.0%',
            height=400
        )
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # Show unique values info
        st.markdown("### 📊 Distribution Details")
        col1, col2 = st.columns(2)
        with col1:
            unique_orig = len(np.unique(original_donations))
            st.metric("Unique values in original", unique_orig)
            st.caption(f"Maximum possible unique values: {len(original_donations)}")
        with col2:
            unique_resamp = len(np.unique(df[donation_col]))
            st.metric("Unique values in resampled", unique_resamp)
            st.caption(f"Limited by original {unique_orig} unique values")
        
    except Exception as e:
        st.caption(f"Error loading empirical distribution: {e}")
        show_overview(df)


def show_income_distribution_histogram(sim_params, n_samples: int = 1000):
    """Display income distribution histogram with discount threshold overlay"""
    try:
        # Generate sample data
        income_samples = sim_params.sample_income_distribution(n_samples)
        
        # Create histogram
        fig = px.histogram(
            x=income_samples,
            nbins=50,
            title=f"Income Distribution Preview ({sim_params.income_distribution.title()})",
            labels={'x': 'Income ($)', 'count': 'Number of Agents'},
            marginal="box"
        )
        
        # Add discount threshold line
        fig.add_vline(
            x=sim_params.discount_income_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Discount Threshold: ${sim_params.discount_income_threshold:,.0f}",
            annotation_position="top"
        )
        
        # Add average/median line
        avg_median_line_color = "blue" if sim_params.income_avg_type == "average" else "green"
        fig.add_vline(
            x=sim_params.income_avg,
            line_dash="solid",
            line_color=avg_median_line_color,
            line_width=2,
            annotation_text=f"{sim_params.income_avg_type.title()}: ${sim_params.income_avg:,.0f}",
            annotation_position="bottom"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_tickformat='$,.0f',
            showlegend=False,
            height=400
        )
        
        # Calculate and display statistics
        actual_mean = np.mean(income_samples)
        actual_median = np.median(income_samples)
        discount_rate = sim_params.get_discount_qualification_rate(n_samples)
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Show distribution statistics
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Actual Mean", f"${actual_mean:,.0f}")
        with col_stat2:
            st.metric("Actual Median", f"${actual_median:,.0f}")
        with col_stat3:
            st.metric("Discount Qualification", f"{discount_rate:.1%}")
        with col_stat4:
            st.metric("Sample Size", f"{n_samples:,}")
        
        # Show comparison with target
        target_value = sim_params.income_avg
        actual_value = actual_mean if sim_params.income_avg_type == "average" else actual_median
        deviation = abs(actual_value - target_value)
        deviation_pct = (deviation / target_value) * 100 if target_value > 0 else 0
        
        if deviation_pct < 5:
            st.success(f"✅ Distribution closely matches target {sim_params.income_avg_type} (deviation: {deviation_pct:.1f}%)")
        elif deviation_pct < 10:
            st.warning(f"⚠️ Moderate deviation from target {sim_params.income_avg_type} (deviation: {deviation_pct:.1f}%)")
        else:
            st.error(f"❌ Large deviation from target {sim_params.income_avg_type} (deviation: {deviation_pct:.1f}%)")
        
    except Exception as e:
        st.error(f"❌ Error generating income distribution histogram: {e}")
        st.caption("Please check your income distribution parameters.")


def get_css_styles():
    """Return the CSS styles for the application"""
    return """
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.page-header {
    font-size: 2rem;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 1.5rem;
}
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #34495e;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #3498db;
}
.subsection-header {
    font-size: 1.1rem;
    font-weight: 500;
    color: #2c3e50;
    margin-bottom: 0.8rem;
    margin-top: 1.5rem;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
}
.stAlert {
    margin-top: 1rem;
}
.navigation-buttons {
    margin-top: 2rem;
    margin-bottom: 2rem;
}
.param-group {
    margin-bottom: 2rem;
    padding: 1rem;
    border-left: 4px solid #3498db;
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border-radius: 0 0.5rem 0.5rem 0;
}

</style>
"""
