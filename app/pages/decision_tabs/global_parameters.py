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


def render_global_parameters_tab(selected_decisions):
    """Render global parameters that are applicable to selected decisions"""
    st.markdown('<h3 class="section-header">🌐 Global Parameters (Editable)</h3>', unsafe_allow_html=True)
    st.info("⚠️ These parameters were configured on Page 1. Changes here will override those settings.")
    
    all_applicable = get_decision_global_parameters(selected_decisions)
    
    if not all_applicable:
        st.info("The selected decisions don't use any global parameters (they are all trait-based).")
        return
    
    # Income Distribution Parameters
    income_params = ['income_distribution', 'income_min', 'income_max', 'income_avg', 'discount_income_threshold']
    if any(param in all_applicable for param in income_params):
        st.markdown('<h4 class="subsection-header">💵 Income Distribution</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'income_distribution' in all_applicable:
                income_dist_type = st.selectbox(
                    "Distribution Type",
                    ["lognormal", "pareto", "weibull"],
                    index=["lognormal", "pareto", "weibull"].index(st.session_state.sim_params.income_distribution),
                    key="tab_income_dist"
                )
                st.session_state.sim_params.income_distribution = income_dist_type
            
            if 'income_min' in all_applicable:
                income_min = st.number_input(
                    "Minimum Income",
                    min_value=0.0,
                    value=st.session_state.sim_params.income_min,
                    key="tab_income_min"
                )
                st.session_state.sim_params.income_min = income_min
            
            if 'income_avg' in all_applicable:
                current_type = st.session_state.sim_params.income_avg_type.title()
                income_avg = st.number_input(
                    f"{current_type} Income",
                    min_value=st.session_state.sim_params.income_min,
                    value=st.session_state.sim_params.income_avg,
                    key="tab_income_avg"
                )
                st.session_state.sim_params.income_avg = income_avg
        
        with col2:
            if 'income_max' in all_applicable:
                income_max = st.number_input(
                    "Maximum Income",
                    min_value=st.session_state.sim_params.income_avg,
                    value=st.session_state.sim_params.income_max,
                    key="tab_income_max"
                )
                st.session_state.sim_params.income_max = income_max
            
            if 'discount_income_threshold' in all_applicable:
                discount_threshold = st.number_input(
                    "Discount Threshold",
                    min_value=st.session_state.sim_params.income_min,
                    max_value=st.session_state.sim_params.income_max,
                    value=st.session_state.sim_params.discount_income_threshold,
                    key="tab_discount_threshold"
                )
                st.session_state.sim_params.discount_income_threshold = discount_threshold
    
    # Market Parameters
    market_params = ['num_vendors', 'market_price', 'vendor_price_min', 'vendor_price_max']
    if any(param in all_applicable for param in market_params):
        st.markdown('<h4 class="subsection-header">🏪 Market Parameters</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'num_vendors' in all_applicable:
                num_vendors = st.number_input(
                    "Number of Vendors",
                    min_value=1,
                    max_value=50,
                    value=st.session_state.sim_params.num_vendors,
                    key="tab_num_vendors"
                )
                st.session_state.sim_params.num_vendors = num_vendors
            
            if 'vendor_price_min' in all_applicable:
                vendor_price_min = st.number_input(
                    "Min Vendor Price",
                    min_value=0.01,
                    value=st.session_state.sim_params.vendor_price_min,
                    key="tab_vendor_price_min"
                )
                st.session_state.sim_params.vendor_price_min = vendor_price_min
        
        with col2:
            if 'market_price' in all_applicable:
                market_price = st.number_input(
                    "Average Market Price ($)",
                    min_value=1,
                    max_value=100000,
                    value=int(st.session_state.sim_params.market_price * 100),
                    key="tab_market_price",
                    help="Average market price (in cents, will be converted to dollars)"
                ) / 100.0
                st.session_state.sim_params.market_price = market_price
            
            if 'vendor_price_max' in all_applicable:
                vendor_price_max = st.number_input(
                    "Max Vendor Price",
                    min_value=st.session_state.sim_params.vendor_price_min,
                    value=st.session_state.sim_params.vendor_price_max,
                    key="tab_vendor_price_max"
                )
                st.session_state.sim_params.vendor_price_max = vendor_price_max
    
    # Pricing Parameters
    pricing_params = ['platform_markup', 'price_range', 'price_grid', 'bidding_percentage']
    if any(param in all_applicable for param in pricing_params):
        st.markdown('<h4 class="subsection-header">💰 Pricing Parameters</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'platform_markup' in all_applicable:
                platform_markup = st.slider(
                    "Platform Markup (m)",
                    min_value=0.0,
                    max_value=0.5,
                    value=st.session_state.sim_params.platform_markup,
                    step=0.01,
                    key="tab_platform_markup"
                )
                st.session_state.sim_params.platform_markup = platform_markup
            
            if 'price_range' in all_applicable:
                price_range = st.slider(
                    "Price Range (r)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.sim_params.price_range,
                    step=0.01,
                    key="tab_price_range"
                )
                st.session_state.sim_params.price_range = price_range
        
        with col2:
            if 'bidding_percentage' in all_applicable:
                bidding_percentage = st.slider(
                    "Bidding Percentage (bp)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.sim_params.bidding_percentage,
                    step=0.01,
                    key="tab_bidding_percentage"
                )
                st.session_state.sim_params.bidding_percentage = bidding_percentage
            
            if 'price_grid' in all_applicable:
                price_grid = st.number_input(
                    "Price Grid Categories (g)",
                    min_value=3,
                    max_value=21,
                    value=st.session_state.sim_params.price_grid,
                    step=2,
                    key="tab_price_grid"
                )
                # Ensure odd number
                if price_grid % 2 == 0:
                    price_grid += 1
                st.session_state.sim_params.price_grid = price_grid
