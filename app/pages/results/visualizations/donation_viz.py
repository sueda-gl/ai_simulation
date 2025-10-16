# app/pages/results/visualizations/donation_viz.py
"""
Donation-related visualization functions.
Handles donation_default and final_donation_rate decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px


def render_donation_default(df, decision_name, decision_title, decision_data):
    """Visualization for donation_default - placeholder until specialized view is added"""
    try:
        numeric_data = pd.to_numeric(decision_data, errors='coerce')
        if not numeric_data.isna().all():
            col1, col2, col3, col4 = st.columns([1, 1.1, 1.1, 1.2])
            with col1:
                st.metric("Total Agents", f"{len(decision_data):,}")
            with col2:
                st.metric("Mean", f"{numeric_data.mean():.2%}")
            with col3:
                st.metric("Std Dev", f"{numeric_data.std():.2%}")
            with col4:
                st.metric("Range", f"{numeric_data.min():.2%} - {numeric_data.max():.2%}")
            col_plot, col_stats = st.columns([2, 1])
            with col_plot:
                fig = px.histogram(
                    df,
                    x=decision_name,
                    nbins=30,
                    title=f"Distribution of {decision_title}",
                    labels={decision_name: decision_title, 'count': 'Number of Agents'}
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_tickformat='.0%'
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_stats:
                st.markdown("**📈 Statistics**")
                stats = numeric_data.describe()
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
                    'Value': [f"{stats[key]:.2%}" for key in ['mean', 'std', 'min', 'max', '50%', '25%', '75%']]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("Data not numeric; specialized visualization not available yet.")
    except Exception:
        st.info("Unable to render donation_default with placeholder visualization.")


def render_final_donation_rate(df, decision_name, decision_title, decision_data):
    """Visualization for final_donation_rate with 3-case logic for donation configs"""
    
    # CASE 3: Check if a donation configuration has been selected
    has_selected_config = hasattr(st.session_state, 'selected_donation_config')
    
    # CASE 1: Check if exactly one donation config exists (auto-use it)
    is_single_donation_run = (
        hasattr(st.session_state, 'custom_decisions') and 
        st.session_state.custom_decisions == ['donation_default'] and
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) == 0
    )
    
    # If this is a single donation run with exactly one result, treat it as "only config available"
    has_only_one_config = False
    if is_single_donation_run and hasattr(st.session_state, 'simulation_results'):
        results_dict = st.session_state.simulation_results
        if results_dict and len(results_dict) == 1:
            has_only_one_config = True
            only_config_key = list(results_dict.keys())[0]
            only_config_df = results_dict[only_config_key]
    
    # Decision logic: Use distribution if selected config OR only one config available
    use_distribution = (has_selected_config or has_only_one_config) and 'donation_default' in df.columns
    
    if use_distribution:
        # Show the actual donation distribution - distinguish between cases
        if has_selected_config:
            st.success("📊 **Using Distribution from Selected Donation Configuration**")
            st.caption("✅ The final_donation_rate values in your export match the donation_default distribution shown below")
        elif has_only_one_config:
            st.success("📊 **Using Distribution from Only Available Donation Configuration**")
            st.caption("✅ Only one donation configuration was generated - final_donation_rate values match donation_default")
        
        donation_data = df['donation_default']
        
        # Top section: Distribution statistics
        col1, col2, col3, col4 = st.columns([1, 1.1, 1.1, 1])
        
        with col1:
            st.metric("Total Agents", f"{len(donation_data):,}")
        
        with col2:
            st.metric("Mean Rate", f"{donation_data.mean():.2%}")
        
        with col3:
            st.metric("Median Rate", f"{donation_data.median():.2%}")
        
        with col4:
            st.metric("Std Dev", f"{donation_data.std():.2%}")
        
        # Distribution visualization
        st.markdown("---")
        st.markdown("**📊 Donation Rate Distribution:**")
        
        col_hist, col_stats = st.columns([2, 1])
        
        with col_hist:
            # Histogram showing the distribution - match overview chart settings for consistency
            fig = px.histogram(
                df,
                x='donation_default',
                title="Distribution of Donation Rates Across Agents",
                labels={'donation_default': 'Donation Rate', 'count': 'Number of Agents'},
                nbins=30,  # Match overview chart
                marginal="box"  # Match overview chart
            )
            fig.update_layout(
                xaxis_tickformat='.0%',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_stats:
            st.markdown("**📈 Distribution Stats:**")
            st.write(f"• **Min**: {donation_data.min():.2%}")
            st.write(f"• **25th %ile**: {donation_data.quantile(0.25):.2%}")
            st.write(f"• **50th %ile**: {donation_data.quantile(0.50):.2%}")
            st.write(f"• **75th %ile**: {donation_data.quantile(0.75):.2%}")
            st.write(f"• **Max**: {donation_data.max():.2%}")
            st.write(f"• **Range**: {donation_data.max() - donation_data.min():.2%}")
            
            st.markdown("---")
            st.markdown("**ℹ️ Source:**")
            if hasattr(st.session_state, 'selected_donation_config'):
                config = st.session_state.selected_donation_config
                st.caption(f"Population: {config['population_mode']}")
                st.caption(f"Income: {config['income_spec_mode']}")
    
    else:
        # Fall back to slider if no donation_default data available
        st.info("💡 **No donation configuration selected** - Using simple rate configuration")
        st.caption("Select a donation configuration on Page 2 to see the full distribution")
        
        # Use _default_value key (consistent with Page 2 for numeric defaults)
        slider_key = f"{decision_name}_default_value"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0.10  # 10% as default
        
        # Top section: Current settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Agents", f"{len(decision_data):,}")
        
        with col2:
            current_rate = st.session_state[slider_key]
            st.metric("Current Rate", f"{current_rate:.2%}")
        
        with col3:
            st.metric("Default", "10%")
        
        # Main configuration section - READ-ONLY DISPLAY
        st.markdown("---")
        st.markdown("**⚙️ Final Donation Rate (Read-Only):**")
        
        col_slider, col_info = st.columns([2, 1])
        
        with col_slider:
            # Get current donation rate from session state
            donation_rate = st.session_state.get(slider_key, 0.10)
            
            # Display as read-only metric
            st.metric(
                "Configured Donation Rate",
                f"{donation_rate:.2%}",
                help="Final donation rate configuration"
            )
            
            st.caption("💡 To modify this setting: Go to **Page 2 → Overview Tab**")
        
        with col_info:
            st.markdown("**📋 Rate Information:**")
            st.write(f"• **Selected**: {donation_rate:.2%}")
            st.write(f"• **Default**: 10%")
            st.write(f"• **Range**: 0% - 100%")
            
            if donation_rate == 0.10:
                st.success("✅ Using default rate")
            elif donation_rate < 0.10:
                st.info(f"📉 {abs(donation_rate - 0.10):.2%} below default")
            else:
                st.info(f"📈 {donation_rate - 0.10:.2%} above default")

