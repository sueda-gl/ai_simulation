# app/pages/results/mc_display.py
"""
The Monte-Carlo results screen: overview metrics, the convergence plot, the
summary table, the CSV downloads and the execution log. The numbers come from
app/reports/mc.py.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from app.reports import mc as mc_report


def show_monte_carlo_results(mc_data):
    """Display Monte Carlo simulation results"""
    if mc_data['summary'] is not None:
        st.subheader("📈 Monte-Carlo Analysis Results")
        
        summary_df = mc_data['summary']
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
    
        metrics = mc_report.overview_metrics(summary_df)
        if metrics is not None:
            with col1:
                st.metric("Mean Donation Rate", metrics["Mean Donation Rate"])

            with col2:
                st.metric("Standard Deviation", metrics["Standard Deviation"])

            with col3:
                st.metric("95% CI Lower", metrics["95% CI Lower"])

            with col4:
                st.metric("95% CI Upper", metrics["95% CI Upper"])
    
        # Monte-Carlo convergence plot
        if mc_data['detailed'] is not None:
            detailed_df = mc_data['detailed']
            
            if mc_report.has_convergence_series(detailed_df):
                st.subheader("📊 Monte-Carlo Convergence")

                # Calculate running average (written onto the session-state frame,
                # which is how running_mean reaches the detailed CSV below)
                mc_report.add_running_mean(detailed_df)

                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Test Run Results", "Running Average Convergence"),
                    vertical_spacing=0.1
                )
                
                # Test runs
                fig.add_trace(
                    go.Scatter(
                        x=detailed_df['run'] + 1,
                        y=detailed_df['donation_default_mean'],
                        mode='markers+lines',
                        name='Test Runs',
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
                interval = mc_report.convergence_interval(detailed_df)
                if interval is not None:
                    fig.add_hline(y=interval['ci_upper'], line_dash="dash", line_color="gray", row=2, col=1)
                    fig.add_hline(y=interval['ci_lower'], line_dash="dash", line_color="gray", row=2, col=1)
                
                fig.update_layout(
                    height=600,
                    title="Monte-Carlo Study Results",
                    showlegend=True
                )
                fig.update_yaxes(tickformat='.2%')
                fig.update_xaxes(title="Run Number")
                
                st.plotly_chart(fig, width="stretch")
    
        # Summary statistics table
        st.subheader("📋 Summary Statistics")
    
        # Format the summary table for display
        display_summary = mc_report.format_summary_for_display(summary_df)

        st.dataframe(display_summary, use_container_width=True)
    
        # Download Monte-Carlo results
        st.subheader("💾 Export Monte-Carlo Results")
        
        col1, col2, col3 = st.columns(3)
    
        with col1:
            if mc_data['summary'] is not None:
                summary_csv = mc_report.summary_csv(mc_data['summary'])
                st.download_button(
                    label="📥 Download Summary",
                    data=summary_csv,
                    file_name=f"monte_carlo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if mc_data['detailed'] is not None:
                detailed_csv = mc_report.detailed_csv(mc_data['detailed'])
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
