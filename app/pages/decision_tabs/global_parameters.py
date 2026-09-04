# app/pages/decision_tabs/global_parameters.py
"""
Global parameters display and configuration for decision tabs.
"""
import streamlit as st
from app.models import get_decision_global_parameters


def render_global_parameters_readonly(decision_name=None):
    """Render global parameters in read-only mode that exactly mirrors Page 1 structure"""
    st.markdown('<h3 class="section-header">🌐 Global Parameters (Read-Only)</h3>', unsafe_allow_html=True)
    st.caption("💡 These parameters are configured on Page 1: Common Simulation Parameters")
    
    # Show which parameters this specific decision uses if provided
    if decision_name:
        decision_params = get_decision_global_parameters([decision_name])
        if decision_params:
            st.info(f"✅ This decision uses: {', '.join([p.replace('_', ' ').title() for p in sorted(decision_params)])}")
        else:
            st.info("ℹ️ This is a trait-based decision (doesn't use global parameters)")
    
    # Create 4-column layout for better space utilization
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Simulation Execution Mode
        st.markdown('<h4 class="subsection-header">🎯 Simulation Execution Mode</h4>', unsafe_allow_html=True)
        execution_mode = "Live Simulation" if st.session_state.sim_params.simulation_execution_mode == "live" else "Snapshot"
        st.text(f"Execution Mode: {execution_mode}")
        
        # Simulation Mode
        st.markdown('<h4 class="subsection-header">🎲 Simulation Mode</h4>', unsafe_allow_html=True)
        st.text(f"Analysis Mode: {st.session_state.sim_params.simulation_mode}")
        
        # Simulation Settings
        st.markdown('<h4 class="subsection-header">⚙️ Simulation Settings</h4>', unsafe_allow_html=True)
        st.text(f"Number of Agents: {st.session_state.n_agents:,}")
        
        if st.session_state.sim_params.simulation_mode == "Single Run":
            st.text(f"Random Seed: {st.session_state.seed}")
        else:
            st.text(f"Number of Runs: {st.session_state.n_runs}")
            st.text(f"Base Seed: {st.session_state.base_seed}")
        
        st.text(f"Show Agent Details: {'Yes' if st.session_state.show_individual_agents else 'No'}")
        # COMMENTED OUT: Auto-save feature disabled
        # st.text(f"Save Results: {'Yes' if st.session_state.save_results else 'No'}")
    
    with col2:
        # Market Parameters
        st.markdown('<h4 class="subsection-header">🏪 Market Parameters</h4>', unsafe_allow_html=True)
        st.text(f"Platform Markup: {st.session_state.sim_params.platform_markup:.0%}")
        st.text(f"Price Range: ±{st.session_state.sim_params.price_range:.0%}")
        st.text(f"Bidding Percentage: {st.session_state.sim_params.bidding_percentage:.0%}")
        st.text(f"Price Grid Categories: {st.session_state.sim_params.price_grid}")
        
        # Income Categories
        st.markdown('<h4 class="subsection-header">📊 Income Categories</h4>', unsafe_allow_html=True)
        st.text(f"Discount Categories (NDIC): {st.session_state.sim_params.num_discount_categories}")
        st.text(f"Fixed Categories (NFIC): {st.session_state.sim_params.num_fixed_categories}")
        
        # Consumption Limits
        st.markdown('<h4 class="subsection-header">🛒 Consumption Limits</h4>', unsafe_allow_html=True)
        limits_status = "Enabled" if st.session_state.sim_params.apply_purchasing_limits else "Disabled"
        st.text(f"Apply Limits: {limits_status}")
        if st.session_state.sim_params.apply_purchasing_limits:
            limits_source = "Manual Entry" if st.session_state.sim_params.purchasing_limits_source == "manual" else "Upload CSV"
            st.text(f"Configuration Source: {limits_source}")
        else:
            # Show artificial limit when purchasing limits are disabled
            st.text(f"Artificial Limit: {st.session_state.sim_params.max_purchases_per_term} items/term")
    
    with col3:
        # Income Distribution
        st.markdown('<h4 class="subsection-header">💵 Income Distribution</h4>', unsafe_allow_html=True)
        st.text(f"Distribution Type: {st.session_state.sim_params.income_distribution.title()}")
        
        # Show distribution-specific parameters
        if st.session_state.sim_params.income_distribution == "lognormal":
            st.text(f"Lognormal μ: {st.session_state.sim_params.lognormal_mu:.1f}")
            st.text(f"Lognormal σ: {st.session_state.sim_params.lognormal_sigma:.1f}")
            st.text(f"Minimum Income: ${st.session_state.sim_params.lognormal_min:.0f}")
            if st.session_state.sim_params.lognormal_max is not None:
                st.text(f"Maximum Income: ${st.session_state.sim_params.lognormal_max:.0f}")
            else:
                st.text("Maximum Income: ∞ (no constraint)")
        elif st.session_state.sim_params.income_distribution == "pareto":
            st.text(f"Pareto x_m (minimum): ${st.session_state.sim_params.pareto_x_m:.0f}")
            st.text(f"Pareto α (shape): {st.session_state.sim_params.pareto_alpha:.1f}")
            if st.session_state.sim_params.pareto_max is not None:
                st.text(f"Maximum Income: ${st.session_state.sim_params.pareto_max:.0f}")
            else:
                st.text("Maximum Income: ∞ (no constraint)")
        elif st.session_state.sim_params.income_distribution == "weibull":
            st.text(f"Weibull k (shape): {st.session_state.sim_params.weibull_k:.1f}")
            st.text(f"Weibull λ (scale): ${st.session_state.sim_params.weibull_lambda:.0f}")
            st.text(f"Minimum Income: ${st.session_state.sim_params.weibull_min:.0f}")
            if st.session_state.sim_params.weibull_max is not None:
                st.text(f"Maximum Income: ${st.session_state.sim_params.weibull_max:.0f}")
            else:
                st.text("Maximum Income: ∞ (no constraint)")
        
        st.text(f"Discount Threshold: ${st.session_state.sim_params.discount_income_threshold:,.0f}")
        
        # Population Generation Mode
        st.markdown('<h4 class="subsection-header">🧬 Population Generation Mode</h4>', unsafe_allow_html=True)
        st.text(f"Population Mode: {st.session_state.population_mode}")
    
    with col4:
        # Vendor Configuration
        st.markdown('<h4 class="subsection-header">🏪 Vendor Configuration</h4>', unsafe_allow_html=True)
        st.text(f"Number of Vendors: {st.session_state.sim_params.num_vendors}")
        if st.session_state.sim_params.num_vendors == 1:
            st.text("Mode: Single Vendor (Simplified)")
            st.text(f"Product Price: ${st.session_state.sim_params.market_price:.2f}")
            st.text(f"Products Offered: {st.session_state.sim_params.vendor_products_avg}")
            # Carryover settings for single vendor
            if st.session_state.sim_params.override_carryover:
                carryover_status = "Enabled" if st.session_state.sim_params.global_carryover else "Disabled"
                st.text(f"Carryover: {carryover_status}")
            else:
                st.text(f"Carryover Probability: {st.session_state.sim_params.vendor_carryover_probability:.0%}")
        else:
            vendor_mode = "Generate Randomly" if st.session_state.sim_params.vendor_config_mode == "random" else "Upload CSV"
            st.text(f"Setup Mode: {vendor_mode}")
            if st.session_state.sim_params.vendor_config_mode == "random":
                st.text(f"Min Price: ${st.session_state.sim_params.vendor_price_min:.2f}")
                st.text(f"Max Price: ${st.session_state.sim_params.vendor_price_max:.2f}")
                st.text(f"Avg Price: ${st.session_state.sim_params.market_price:.2f}")
                st.text(f"Min Products: {st.session_state.sim_params.vendor_products_min}")
                st.text(f"Max Products: {st.session_state.sim_params.vendor_products_max}")
                st.text(f"Avg Products: {st.session_state.sim_params.vendor_products_avg}")
                # Carryover settings for multiple vendors
                if st.session_state.sim_params.override_carryover:
                    carryover_status = "Enabled" if st.session_state.sim_params.global_carryover else "Disabled"
                    st.text(f"Carryover: {carryover_status} (All vendors)")
                else:
                    st.text(f"Carryover Probability: {st.session_state.sim_params.vendor_carryover_probability:.0%}")
        
        # Time Parameters
        st.markdown('<h4 class="subsection-header">⏱️ Time Parameters</h4>', unsafe_allow_html=True)
        st.text(f"Number of Periods: {st.session_state.sim_params.periods}")
        st.text(f"Duration per Period: {st.session_state.sim_params.duration_hours:.0f} hours")
        st.text(f"Duration in Seconds: {st.session_state.sim_params.get_duration_seconds():.0f}")
    
    st.markdown("---")
    st.caption("💡 To modify these parameters, go to Page 1: Common Simulation Parameters")
