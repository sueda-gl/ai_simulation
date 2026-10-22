# app/pages/page1_income_histogram.py
"""
Page 1: the income distribution preview (histogram with the discount threshold,
then the distribution statistics) drawn under the income parameters.
"""
import streamlit as st
import numpy as np
import plotly.express as px


def show_income_distribution_histogram(sim_params, n_samples: int = 1000, seed: int = None):
    """Display income distribution histogram with discount threshold overlay"""
    try:
        # Use provided seed or default to 42
        preview_seed = seed if seed is not None else 42
        
        # Generate sample data
        income_samples = sim_params.sample_income_distribution(n_samples, seed=preview_seed)
        
        # Calculate percentiles and range for display
        p99_value = float(np.percentile(income_samples, 99))
        x_lower = float(max(0.0, np.min(income_samples)))
        x_upper = float(np.max(income_samples))
        
        # Always show full range - users want to see the complete distribution
        samples_for_display = income_samples
        display_range = [x_lower, x_upper]
        
        # Create histogram with all data
        fig = px.histogram(
            x=samples_for_display,
            nbins=50,
            title=f"Income Distribution Preview ({sim_params.income_distribution.title()})",
            labels={'x': 'Income ($)', 'count': 'Number of Agents'},
            marginal="box"
        )
        
        # Set explicit x-axis range to show full distribution
        fig.update_xaxes(range=display_range)
        
        # Add discount threshold line
        fig.add_vline(
            x=sim_params.discount_income_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Discount Threshold: ${sim_params.discount_income_threshold:,.0f}",
            annotation_position="top"
        )
        
        # Don't show any target line - distribution is based on user parameters
        
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
            st.metric("Mean", f"${actual_mean:,.0f}")
        with col_stat2:
            st.metric("Median", f"${actual_median:,.0f}")
        with col_stat3:
            st.metric("Discount Qualification", f"{discount_rate:.1%}",
                     help="Percentage of agents with income ≤ threshold (potential eligibility based on income alone). Actual discount customers will be lower, as they must also choose to disclose income AND documents in their decisions.")
        with col_stat4:
            st.metric("Sample Size", f"{n_samples:,}")
        
        # Show min/max and percentiles
        col_range1, col_range2, col_range3, col_range4 = st.columns(4)
        actual_min = np.min(income_samples)
        actual_max = np.max(income_samples)
        p95_value = np.percentile(income_samples, 95)
        
        with col_range1:
            st.metric("Minimum", f"${actual_min:,.0f}")
        with col_range2:
            st.metric("95th Percentile", f"${p95_value:,.0f}")
        with col_range3:
            st.metric("99th Percentile", f"${p99_value:,.0f}")
        with col_range4:
            st.metric("Maximum", f"${actual_max:,.0f}")
        
        # Show info about tail if there are significant outliers beyond p99
        num_outliers = int(np.sum(income_samples > p99_value))
        if num_outliers > 10:  # Only mention if substantial tail exists
            pct_outliers = (num_outliers / n_samples) * 100
            st.caption(f"ℹ️ Note: {pct_outliers:.1f}% of samples are beyond the 99th percentile (${p99_value:,.0f}), extending to ${actual_max:,.0f}.")
        
        # Distribution is based on user-specified parameters - no target comparison needed
        
    except Exception as e:
        st.error(f"❌ Error generating income distribution histogram: {e}")
        st.caption("Please check your income distribution parameters.")
