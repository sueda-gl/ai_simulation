# app/pages.py
"""
Page logic for the Enhanced AI Agent Simulation.
Handles Page 1 (Common Parameters), Page 2 (Decision Parameters), and Results.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from app.models import ALL_DECISIONS, get_decision_global_parameters, get_all_global_parameters
from app.components import (show_overview, 
                           show_monte_carlo_results, show_dependent_variable_comparison,
                           show_income_distribution_histogram)
from app.simulation import run_simulation_from_sidebar, run_monte_carlo_study


def go_to_page1():
    st.session_state.page = 'page1'


def go_to_page2():
    st.session_state.page = 'page2'


def go_to_results():
    st.session_state.page = 'results'


def render_navigation(current_page):
    """Render navigation buttons based on current page"""
    st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
    
    if current_page == 'page1':
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            st.button("Next: Decision Parameters →", type="primary", on_click=go_to_page2, use_container_width=True)
    
    elif current_page == 'page2':
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.button("← Back to Common Parameters", on_click=go_to_page1, use_container_width=True)
        with col3:
            # Show "Go to Results" button if results exist
            if st.session_state.simulation_results is not None or st.session_state.mc_results is not None:
                st.button("View Results →", type="primary", on_click=go_to_results, use_container_width=True)
    
    elif current_page == 'results':
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.button("← Back to Decision Parameters", on_click=go_to_page2, use_container_width=True)
        with col3:
            st.button("Back to Common Parameters →", on_click=go_to_page1, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_page1():
    """Render Page 1: Common Simulation Parameters"""
    st.markdown('<h2 class="page-header">Page 1: Common Simulation Parameters</h2>', unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulation Mode Selection
        st.markdown('<h3 class="section-header">🎯 Simulation Execution Mode</h3>', unsafe_allow_html=True)
        
        simulation_execution_mode = st.radio(
            "Execution Mode",
            ["Snapshot", "Live Simulation"],
            index=0 if st.session_state.sim_params.simulation_execution_mode == "snapshot" else 1,
            horizontal=True,
            help="Snapshot: Run simulation once with fixed parameters\nLive: Real-time simulation with dynamic updates (backend implementation pending)"
        )
        
        mode_value = "snapshot" if simulation_execution_mode == "Snapshot" else "live"
        st.session_state.sim_params.simulation_execution_mode = mode_value
        
        # Simulation Mode (Single vs Monte-Carlo)
        st.markdown('<h3 class="section-header">🎲 Simulation Mode</h3>', unsafe_allow_html=True)
        
        simulation_mode = st.radio(
            "Analysis Mode",
            ["Single Run", "Monte-Carlo Study"],
            index=0 if st.session_state.sim_params.simulation_mode == "Single Run" else 1,
            horizontal=True,
            help="Single Run: One simulation with specified parameters\nMonte-Carlo: Multiple runs for uncertainty analysis"
        )
        st.session_state.sim_params.simulation_mode = simulation_mode
        
        if simulation_mode == "Single Run":
            st.info("📊 Single Run: Execute one simulation with current parameters")
        else:
            st.info("🎯 Monte-Carlo: Execute multiple runs for statistical analysis")
        
        # Time Parameters Section
        st.markdown('<h3 class="section-header">⏱️ Time Parameters</h3>', unsafe_allow_html=True)
        
        periods = st.number_input(
            "Number of Periods",
            min_value=1,
            max_value=100,
            value=st.session_state.sim_params.periods,
            help="Number of periods for simulation run"
        )
        st.session_state.sim_params.periods = periods
        
        duration_hours = st.number_input(
            "Duration per Period (hours)",
            min_value=0.1,
            max_value=24.0,
            value=st.session_state.sim_params.duration_hours,
            step=0.1,
            help="Duration of each period in hours (will be converted to seconds for simulation)"
        )
        st.session_state.sim_params.duration_hours = duration_hours
        st.caption(f"Duration in seconds: {st.session_state.sim_params.get_duration_seconds():.0f}")
        
        # Market Parameters Section (simplified - vendor config moved to dedicated section)
        st.markdown('<h3 class="section-header">🏪 Market Parameters</h3>', unsafe_allow_html=True)
        
        st.info("ℹ️ Vendor configuration (number, prices, products, carryover) is now managed in the dedicated Vendor Configuration section below.")
        
        # Product Offering Section
        st.markdown('<h3 class="section-header">📦 Product Offering</h3>', unsafe_allow_html=True)
        
        products_per_vendor = st.number_input(
            "Products per Vendor (NV)",
            min_value=1,
            max_value=10000,
            value=st.session_state.sim_params.products_per_vendor,
            help="Number of products offered by each vendor at the beginning of each period"
        )
        st.session_state.sim_params.products_per_vendor = products_per_vendor
        
        carryover = st.checkbox(
            "Carryover Unsold Products",
            value=st.session_state.sim_params.carryover,
            help="If checked, unsold products carry over to the next period"
        )
        st.session_state.sim_params.carryover = carryover
        
        bidding_percentage = st.slider(
            "Bidding Percentage (bp)",
            min_value=0.0,
            max_value=1.0,  # Extended from 0.5 to 1.0 as requested
            value=st.session_state.sim_params.bidding_percentage,
            step=0.05,
            help="Proportion of products available for bidding (NA = bp × NV). Now supports up to 100%!"
        )
        st.session_state.sim_params.bidding_percentage = bidding_percentage
        st.caption(f"Products for auction per vendor: {st.session_state.sim_params.get_num_auction_products()}")
    
    with col2:
        # Pricing Parameters Section
        st.markdown('<h3 class="section-header">💰 Pricing Parameters</h3>', unsafe_allow_html=True)
        
        platform_markup = st.slider(
            "Platform Markup (m)",
            min_value=0.0,
            max_value=0.5,
            value=st.session_state.sim_params.platform_markup,
            step=0.01,
            help="Platform markup: Customer Price = (1+m) × Vendor Price"
        )
        st.session_state.sim_params.platform_markup = platform_markup
        
        price_range = st.slider(
            "Price Range (r)",
            min_value=0.0,
            max_value=1.0,  # Extended from 0.5 to 1.0 for simulation flexibility
            value=st.session_state.sim_params.price_range,
            step=0.05,
            help="Price range for Purchase Now and Minimum Bid prices. Extended to 1.0 for simulation flexibility."
        )
        st.session_state.sim_params.price_range = price_range
        
        price_grid = st.number_input(
            "Price Grid Categories (g)",
            min_value=3,
            max_value=21,
            value=st.session_state.sim_params.price_grid,
            step=2,
            help="Number of price categories (must be odd)"
        )
        st.session_state.sim_params.price_grid = price_grid
        
        # Ensure price grid is odd
        if st.session_state.sim_params.price_grid % 2 == 0:
            st.caption("Price grid adjusted to odd number: " + str(st.session_state.sim_params.price_grid + 1))
            st.session_state.sim_params.price_grid += 1
        
        # Show calculated prices example
        example_vendor_price = st.session_state.sim_params.market_price
        example_customer_price = example_vendor_price * (1 + st.session_state.sim_params.platform_markup)
        example_pn_price = st.session_state.sim_params.get_purchase_now_price(example_vendor_price)
        example_min_bid = st.session_state.sim_params.get_minimum_bid_price(example_vendor_price)
        

        
        # Income Distribution Section
        st.markdown('<h3 class="section-header">💵 Income Distribution</h3>', unsafe_allow_html=True)
        
        income_distribution = st.selectbox(
            "Income Distribution Type",
            ["lognormal", "pareto", "weibull"],
            index=["lognormal", "pareto", "weibull"].index(st.session_state.sim_params.income_distribution),
            help="Distribution function for generating agent incomes"
        )
        st.session_state.sim_params.income_distribution = income_distribution
        
        # Distribution-specific parameters
        if income_distribution == "lognormal":
            lognormal_sigma = st.slider(
                "Lognormal σ (Shape Parameter)",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.sim_params.lognormal_sigma,
                step=0.1,
                help="Standard deviation of the log-transformed values. Higher values = more right-skewed distribution"
            )
            st.session_state.sim_params.lognormal_sigma = lognormal_sigma
            
        elif income_distribution == "pareto":
            pareto_alpha = st.slider(
                "Pareto α (Shape Parameter)",
                min_value=1.1,
                max_value=5.0,
                value=st.session_state.sim_params.pareto_alpha,
                step=0.1,
                help="Shape parameter. Higher values = less inequality, more concentrated around lower incomes"
            )
            st.session_state.sim_params.pareto_alpha = pareto_alpha
            
        elif income_distribution == "weibull":
            weibull_shape = st.slider(
                "Weibull k (Shape Parameter)",
                min_value=0.5,
                max_value=5.0,
                value=st.session_state.sim_params.weibull_shape,
                step=0.1,
                help="Shape parameter. k=1: exponential, k=2: Rayleigh-like, k>3: bell-shaped"
            )
            st.session_state.sim_params.weibull_shape = weibull_shape
        
        col_inc1, col_inc2 = st.columns(2)
        with col_inc1:
            income_min = st.number_input(
                "Minimum Income ($)",
                min_value=0.0,
                max_value=1000000.0,
                value=st.session_state.sim_params.income_min,
                step=100.0
            )
            st.session_state.sim_params.income_min = income_min
            
            # Income average/median type selector
            income_avg_type = st.radio(
                "Central Tendency Type",
                ["Average", "Median"],
                index=0 if st.session_state.sim_params.income_avg_type == "average" else 1,
                horizontal=True,
                help="Specify whether the central value represents the average (mean) or median of the income distribution"
            )
            st.session_state.sim_params.income_avg_type = income_avg_type.lower()
            
            # Dynamic label based on selection
            central_label = f"{income_avg_type} Income ($)"
            income_avg = st.number_input(
                central_label,
                min_value=st.session_state.sim_params.income_min,
                max_value=1000000.0,
                value=st.session_state.sim_params.income_avg,
                step=100.0,
                help=f"The {income_avg_type.lower()} income value for the distribution"
            )
            st.session_state.sim_params.income_avg = income_avg
        
        with col_inc2:
            income_max = st.number_input(
                "Maximum Income ($)",
                min_value=st.session_state.sim_params.income_avg,
                max_value=10000000.0,
                value=st.session_state.sim_params.income_max,
                step=100.0
            )
            st.session_state.sim_params.income_max = income_max
            
            # Discount threshold
            discount_income_threshold = st.number_input(
                "Threshold Income for Discount ($)",
                min_value=st.session_state.sim_params.income_min,
                max_value=st.session_state.sim_params.income_max,
                value=st.session_state.sim_params.discount_income_threshold,
                step=100.0,
                help="Income threshold below which agents qualify for discounts (pending document disclosure)"
            )
            st.session_state.sim_params.discount_income_threshold = discount_income_threshold
            
            # Show threshold validation and info
            if st.session_state.sim_params.income_min <= discount_income_threshold <= st.session_state.sim_params.income_max:
                threshold_pct = ((discount_income_threshold - st.session_state.sim_params.income_min) / 
                               (st.session_state.sim_params.income_max - st.session_state.sim_params.income_min)) * 100
                st.caption(f"✅ Threshold at {threshold_pct:.1f}% of income range")
            else:
                st.error("❌ Threshold must be between minimum and maximum income!")
        
        # Income Distribution Preview
        st.markdown('<h4 class="subsection-header">📈 Distribution Preview</h4>', unsafe_allow_html=True)
        
        # Add toggle for showing histogram
        show_histogram = st.checkbox(
            "Show Income Distribution Histogram",
            value=False,
            help="Generate and display a preview of the income distribution with current parameters"
        )
        
        if show_histogram:
            show_income_distribution_histogram(st.session_state.sim_params)
        
        # Income Categories Section
        st.markdown('<h3 class="section-header">📊 Income Categories</h3>', unsafe_allow_html=True)
        
        col_cat1, col_cat2 = st.columns(2)
        with col_cat1:
            num_discount_categories = st.number_input(
                "Discount Income Categories (NDIC)",
                min_value=1,
                max_value=10,
                value=st.session_state.sim_params.num_discount_categories,
                help="Number of customer discount income categories"
            )
            st.session_state.sim_params.num_discount_categories = num_discount_categories
        
        with col_cat2:
            num_fixed_categories = st.number_input(
                "Fixed Income Categories (NFIC)",
                min_value=1,
                max_value=10,
                value=st.session_state.sim_params.num_fixed_categories,
                help="Number of customer fixed income categories"
            )
            st.session_state.sim_params.num_fixed_categories = num_fixed_categories
    
    # Vendor Configuration - Single Source of Truth
    st.markdown('<h3 class="section-header">🏪 Vendor Configuration</h3>', unsafe_allow_html=True)
    st.caption("Configure all vendor settings: number, prices, products, and carryover behavior")
    
    # Number of Vendors (moved from Market Parameters)
    num_vendors = st.number_input(
        "Number of Vendors (N)",
        min_value=1,
        max_value=50,
        value=st.session_state.sim_params.num_vendors,
        help="Total number of vendors operating on the platform"
    )
    st.session_state.sim_params.num_vendors = num_vendors
    
    # Vendor Setup Mode
    vendor_setup_mode = st.radio(
        "Vendor Setup Mode",
        ["Generate Randomly", "Upload Vendor Config File"],
        index=0 if st.session_state.sim_params.vendor_config_mode == "random" else 1,
        horizontal=True,
        help="Choose how to configure vendor properties (price, products, carryover)"
    )
    
    if vendor_setup_mode == "Generate Randomly":
        st.session_state.sim_params.vendor_config_mode = "random"
        
        # Create columns for organized layout
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Price Configuration
            st.markdown('<h4 class="subsection-header">💰 Price Configuration</h4>', unsafe_allow_html=True)
            
            vendor_price_min = st.number_input(
                "Min Price per Vendor ($)",
                min_value=0.01,
                max_value=1000.0,
                value=st.session_state.sim_params.vendor_price_min,
                step=0.01,
                help="Minimum price any vendor can have"
            )
            st.session_state.sim_params.vendor_price_min = vendor_price_min
            
            vendor_price_max = st.number_input(
                "Max Price per Vendor ($)",
                min_value=st.session_state.sim_params.vendor_price_min,
                max_value=1000.0,
                value=st.session_state.sim_params.vendor_price_max,
                step=0.01,
                help="Maximum price any vendor can have"
            )
            st.session_state.sim_params.vendor_price_max = vendor_price_max
            
            market_price = st.number_input(
                "Average Price per Vendor ($)",
                min_value=st.session_state.sim_params.vendor_price_min,
                max_value=st.session_state.sim_params.vendor_price_max,
                value=st.session_state.sim_params.market_price,
                step=0.01,
                help="Target average price across all vendors"
            )
            st.session_state.sim_params.market_price = market_price
            
            # Price validation
            price_total_min = num_vendors * vendor_price_min
            price_total_max = num_vendors * vendor_price_max
            price_total_avg = num_vendors * market_price
            price_valid = price_total_min <= price_total_avg <= price_total_max
            
 
        
        with col_right:
            # Products Configuration
            st.markdown('<h4 class="subsection-header">📦 Products Configuration</h4>', unsafe_allow_html=True)
            
            vendor_products_min = st.number_input(
                "Min Products per Vendor",
                min_value=1,
                max_value=10000,
                value=st.session_state.sim_params.vendor_products_min,
                help="Minimum products any vendor can offer per period"
            )
            st.session_state.sim_params.vendor_products_min = vendor_products_min
            
            vendor_products_max = st.number_input(
                "Max Products per Vendor",
                min_value=st.session_state.sim_params.vendor_products_min,
                max_value=10000,
                value=st.session_state.sim_params.vendor_products_max,
                help="Maximum products any vendor can offer per period"
            )
            st.session_state.sim_params.vendor_products_max = vendor_products_max
            
            vendor_products_avg = st.number_input(
                "Average Products per Vendor",
                min_value=st.session_state.sim_params.vendor_products_min,
                max_value=st.session_state.sim_params.vendor_products_max,
                value=st.session_state.sim_params.vendor_products_avg,
                help="Target average products per vendor"
            )
            st.session_state.sim_params.vendor_products_avg = vendor_products_avg
            
            # Products validation
            products_total_min = num_vendors * vendor_products_min
            products_total_max = num_vendors * vendor_products_max
            products_total_avg = num_vendors * vendor_products_avg
            products_valid = products_total_min <= products_total_avg <= products_total_max
        
        # Carryover Configuration (full width)
        st.markdown('<h4 class="subsection-header">🔄 Carryover Configuration</h4>', unsafe_allow_html=True)
        
        # Add carryover override switch
        override_carryover = st.checkbox(
            "Override per-vendor carryover",
            value=st.session_state.sim_params.override_carryover,
            help="If checked, apply the same carryover setting to all vendors instead of using probability"
        )
        st.session_state.sim_params.override_carryover = override_carryover
        
        if override_carryover:
            # Global carryover setting
            global_carryover = st.radio(
                "Apply carryover to all vendors",
                ["Yes", "No"],
                index=0 if st.session_state.sim_params.global_carryover else 1,
                horizontal=True,
                help="When override is enabled, all vendors will have the same carryover setting"
            )
            carryover_enabled = (global_carryover == "Yes")
            st.session_state.sim_params.global_carryover = carryover_enabled
            expected_carryover_vendors = num_vendors if carryover_enabled else 0
            st.info(f"🔧 Override active: All {num_vendors} vendors will have carryover {'ENABLED' if carryover_enabled else 'DISABLED'}")
        else:
            # Per-vendor probability
            vendor_carryover_probability = st.slider(
                "Carryover Probability (p)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.sim_params.vendor_carryover_probability,
                step=0.05,
                help="Probability that any given vendor will have carryover enabled (Bernoulli per vendor)"
            )
            st.session_state.sim_params.vendor_carryover_probability = vendor_carryover_probability
            expected_carryover_vendors = int(num_vendors * vendor_carryover_probability)
        
        # Summary chips
        st.markdown('<h4 class="subsection-header">📊 Configuration Summary</h4>', unsafe_allow_html=True)
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        with col_sum1:
            st.metric("Total Vendors", f"{num_vendors}")
        with col_sum2:
            st.metric("Expected Total Products", f"{products_total_avg:,}")
        with col_sum3:
            st.metric("Expected Total Revenue", f"${price_total_avg:.2f}")
        with col_sum4:
            st.metric("Expected Vendors with Carryover", f"{expected_carryover_vendors}")

    else:
        st.session_state.sim_params.vendor_config_mode = "upload"
        
        # Hide all random generation inputs when upload mode is selected
        st.markdown('<h4 class="subsection-header">📁 Upload Vendor Configuration</h4>', unsafe_allow_html=True)
        st.info("Upload a CSV file with complete vendor configuration. This will override all random generation settings.")
        
        # Show expected format
        with st.expander("📋 Expected CSV Format", expanded=False):
            st.code("""vendor_id,price,products_per_period,carryover
V1,8.50,120,1
V2,9.25,95,0
V3,10.00,80,1
V4,11.75,110,0
V5,12.00,100,1""")
            st.caption("Required columns: vendor_id, price, products_per_period, carryover (0=disabled, 1=enabled)")
        
        uploaded_file = st.file_uploader(
            "Upload Vendor Configuration CSV",
            type=['csv'],
            help="CSV file with complete vendor configuration"
        )
        
        if uploaded_file is not None:
            try:
                vendor_config_df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_columns = ['vendor_id', 'price', 'products_per_period', 'carryover']
                missing_columns = [col for col in required_columns if col not in vendor_config_df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                else:
                    # Convert to list of dictionaries and store
                    st.session_state.sim_params.vendor_config_data = vendor_config_df.to_dict('records')
                    
                    # Update num_vendors to match uploaded data
                    st.session_state.sim_params.num_vendors = len(vendor_config_df)
                    
                    # Show validation and summary
                    st.success(f"✅ Loaded configuration for {len(vendor_config_df)} vendors")
                    
                    # Summary metrics
                    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                    
                    with col_sum1:
                        st.metric("Vendors Loaded", len(vendor_config_df))
                    with col_sum2:
                        st.metric("Avg Price", f"${vendor_config_df['price'].mean():.2f}")
                    with col_sum3:
                        st.metric("Avg Products/Vendor", f"{vendor_config_df['products_per_period'].mean():.0f}")
                    with col_sum4:
                        st.metric("Vendors with Carryover", f"{vendor_config_df['carryover'].sum()}")
                    
                    # Show totals
                    total_products = vendor_config_df['products_per_period'].sum()
                    total_revenue = (vendor_config_df['price'] * vendor_config_df['products_per_period']).sum()
                    
                    col_total1, col_total2 = st.columns(2)
                    with col_total1:
                        st.metric("Total Products", f"{total_products:,}")
                    with col_total2:
                        st.metric("Expected Total Revenue", f"${total_revenue:,.2f}")
                    
                    # Show preview
                    with st.expander("👀 Preview Loaded Data", expanded=False):
                        st.dataframe(vendor_config_df, use_container_width=True)
                        
            except Exception as e:
                st.error(f"❌ Error loading vendor configuration: {e}")
                st.caption("Please check your CSV format and try again.")
    
    # Consumption Limits Configuration
    st.markdown('<h3 class="section-header">🛒 Consumption Limits</h3>', unsafe_allow_html=True)
    
    apply_limits = st.radio(
        "Apply Consumption Limits?",
        ["Yes", "No"],
        index=0 if st.session_state.sim_params.apply_consumption_limits else 1,
        horizontal=True,
        help="Choose whether to apply consumption limits per income category"
    )
    
    st.session_state.sim_params.apply_consumption_limits = (apply_limits == "Yes")
    
    if st.session_state.sim_params.apply_consumption_limits:
        st.caption("Set consumption limits per product for each income category per period")
        
        # Configuration source
        limits_source = st.radio(
            "Limits Configuration Source",
            ["Manual Entry", "Upload CSV"],
            index=0 if st.session_state.sim_params.consumption_limits_source == "manual" else 1,
            horizontal=True
        )
        
        if limits_source == "Manual Entry":
            st.session_state.sim_params.consumption_limits_source = "manual"
            
            # Create a simple interface for setting consumption limits
            total_categories = st.session_state.sim_params.num_fixed_categories
            
            consumption_limits = {}
            cols = st.columns(min(5, total_categories))
            for i in range(total_categories):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    limit = st.number_input(
                        f"Category {i+1} Limit",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.sim_params.consumption_limits.get(f"cat_{i+1}", 10),
                        key=f"consumption_limit_{i}"
                    )
                    consumption_limits[f"cat_{i+1}"] = limit
            
            st.session_state.sim_params.consumption_limits = consumption_limits
            
        else:
            st.session_state.sim_params.consumption_limits_source = "upload"
            
            st.info("Upload a CSV file with consumption limits. Required columns: `category_id`, `limit`")
            
            # Show expected format
            with st.expander("📋 Expected CSV Format", expanded=False):
                st.code("""category_id,limit
1,10
2,12
3,9
4,15
5,8""")
            
            limits_file = st.file_uploader(
                "Upload Consumption Limits CSV",
                type=['csv'],
                help="CSV file with consumption limits per category"
            )
            
            if limits_file is not None:
                try:
                    limits_df = pd.read_csv(limits_file)
                    
                    # Validate required columns
                    required_columns = ['category_id', 'limit']
                    missing_columns = [col for col in required_columns if col not in limits_df.columns]
                    
                    if missing_columns:
                        st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    else:
                        # Convert to dictionary format
                        consumption_limits = {}
                        for _, row in limits_df.iterrows():
                            consumption_limits[f"cat_{int(row['category_id'])}"] = float(row['limit'])
                        
                        st.session_state.sim_params.consumption_limits = consumption_limits
                        
                        # Show summary
                        st.success(f"✅ Loaded limits for {len(limits_df)} categories")
                        
                        # Show preview
                        with st.expander("👀 Preview Loaded Limits", expanded=False):
                            st.dataframe(limits_df, use_container_width=True)
                            
                except Exception as e:
                    st.error(f"❌ Error loading consumption limits: {e}")
                    st.caption("Please check your CSV format and try again.")
    
    else:
        st.info("ℹ️ Consumption limits are disabled. Agents will have no consumption restrictions.")
        # Clear consumption limits when disabled
        st.session_state.sim_params.consumption_limits = {}
    
    # Navigation
    render_navigation('page1')


# Tab-based helper functions for Page 2
def render_overview_tab(selected_decisions):
    """Render the overview tab with combined execution option"""
    # Add combined run button
    st.markdown('<h3 class="section-header">🚀 Combined Execution</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Run all {len(selected_decisions)} selected decisions together")
    with col2:
        if st.button("🚀 Run All Selected", type="primary", use_container_width=True):
            run_combined_simulation(selected_decisions)


def render_donation_default_tab():
    """Render donation_default specific configuration"""
    st.markdown('<h3 class="section-header"> Donation Default Configuration</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Population mode selector
        st.markdown('<h4 class="subsection-header">Population Generation</h4>', unsafe_allow_html=True)
        population_mode = st.radio(
            "Population Mode",
            ["Copula (synthetic)", "Research Specification", "Compare both"],
            index=["Copula (synthetic)", "Research Specification", "Compare both"].index(st.session_state.population_mode),
            help="Copula: Generate synthetic agents via fitted copula\nResearch: Use original participants with stochastic draws\nCompare both: Show both Copula and Research modes",
            key="tab_population_mode"
        )
        st.session_state.population_mode = population_mode
        
        # Income specification selector
        if population_mode != "Dependent variable resampling":
            st.markdown('<h4 class="subsection-header">Income Specification</h4>', unsafe_allow_html=True)
            income_spec_mode = st.radio(
                "Income Mode for Donation Model",
                ["categorical only", "continuous only", "Compare both"],
                index=["categorical only", "continuous only", "Compare both"].index(st.session_state.income_spec_mode) if st.session_state.income_spec_mode in ["categorical only", "continuous only", "Compare both", "compare both"] else (2 if st.session_state.income_spec_mode == "compare side-by-side" else 0),
                help="Choose income treatment: categorical (5 categories), continuous (linear), or Compare both",
                key="tab_income_spec_mode"
            )
            st.session_state.income_spec_mode = income_spec_mode
        else:
            st.session_state.income_spec_mode = "categorical only"
    
    with col2:
        # Stochastic component option
        st.markdown('<h4 class="subsection-header">Stochastic Component</h4>', unsafe_allow_html=True)
        # Copula toggle
        if population_mode in ["Copula (synthetic)", "Compare both"]:
            sigma_in_copula = st.checkbox(
                "Add Normal(anchor, σ) draw to Copula runs",
                value=getattr(st.session_state, "sigma_in_copula", True),
                key="tab_sigma_in_copula",
                help="When enabled, Copula mode will use the Normal(anchor, σ) stochastic draw"
            )
            st.session_state.sigma_in_copula = sigma_in_copula
        else:
            st.session_state.sigma_in_copula = False
        # Research toggle
        if population_mode in ["Research Specification", "Compare both"]:
            sigma_in_research = st.checkbox(
                "Add Normal(anchor, σ) draw to Research runs",
                value=getattr(st.session_state, "sigma_in_research", True),
                key="tab_sigma_in_research",
                help="Enable or disable stochastic component for Research mode runs"
            )
            st.session_state.sigma_in_research = sigma_in_research
        else:
            st.session_state.sigma_in_research = True
        # σ slider (always visible)
        sigma_value_ui = st.slider(
            "σ (standard deviation) on 0–112 scale",
            min_value=0.0,
            max_value=15.0,
            value=st.session_state.sigma_value_ui,
            step=0.1,
            key="tab_sigma_value",
            help="Controls the spread of the Normal(anchor, σ) draw. Set to 0 to disable variability."
        )
        st.session_state.sigma_value_ui = sigma_value_ui
        
        # Anchor weights
        if population_mode != "Dependent variable resampling":
            st.markdown('<h4 class="subsection-header">Anchor Mix</h4>', unsafe_allow_html=True)
            anchor_observed_weight = st.slider(
                "Weight for observed vs modeled prosocial behavior",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.anchor_observed_weight,
                step=0.05,
                help="Anchor = w × Observed + (1-w) × Predicted",
                key="tab_anchor_weight"
            )
            st.session_state.anchor_observed_weight = anchor_observed_weight
            st.caption(f"Predicted weight: {1 - anchor_observed_weight:.2f}")
        else:
            st.session_state.anchor_observed_weight = 0.75
    
    # Raw output option
    if population_mode != "Copula (synthetic)" or st.session_state.sigma_in_copula:
        st.markdown('<h4 class="subsection-header">Output Options</h4>', unsafe_allow_html=True)
        raw_draw_mode = st.checkbox(
            "Show pre-truncation (raw) donation rate",
            value=st.session_state.raw_draw_mode,
            help="Display the raw Normal(anchor, σ) draw before processing",
            key="tab_raw_draw_mode"
        )
        st.session_state.raw_draw_mode = raw_draw_mode
    else:
        st.session_state.raw_draw_mode = False
    
    # Display Global Parameters
    st.markdown("---")
    render_global_parameters_readonly("donation_default")
    
    # Simulation Settings
    st.markdown("---")
    st.markdown('<h3 class="section-header">⚙️ Simulation Settings</h3>', unsafe_allow_html=True)
    
    # Show current simulation mode
    st.info(f"📊 Mode: {st.session_state.sim_params.simulation_mode} (configured on Page 1)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_agents = st.number_input(
            "Number of Agents",
            min_value=10,
            max_value=50000,
            value=st.session_state.n_agents,
            step=100,
            key="donation_n_agents"
        )
        st.session_state.n_agents = n_agents
        
        if st.session_state.sim_params.simulation_mode == "Single Run":
            seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=2147483647,
                value=st.session_state.seed,
                key="donation_seed"
            )
            st.session_state.seed = seed
        else:
            n_runs = st.number_input(
                "Number of Runs",
                min_value=2,
                max_value=1000,
                value=st.session_state.n_runs,
                step=10,
                key="donation_n_runs"
            )
            st.session_state.n_runs = n_runs
            
            base_seed = st.number_input(
                "Base Seed",
                min_value=1,
                max_value=2147483647,
                value=st.session_state.base_seed,
                key="donation_base_seed"
            )
            st.session_state.base_seed = base_seed
    
    with col2:
        show_individual_agents = st.checkbox(
            "Show Individual Agent Details",
            value=st.session_state.show_individual_agents,
            key="donation_show_agents"
        )
        st.session_state.show_individual_agents = show_individual_agents
        
        save_results = st.checkbox(
            "Save Results to File",
            value=st.session_state.save_results,
            key="donation_save_results"
        )
        st.session_state.save_results = save_results
    
    # Individual run button
    st.markdown("---")
    if st.button("🚀 Run Donation Default Only", type="secondary", use_container_width=True):
        run_individual_decision("donation_default")


def render_global_parameters_readonly(decision_name=None):
    """Render global parameters in read-only mode for display in decision tabs"""
    st.markdown('<h3 class="section-header">🌐 Global Parameters (Read-Only)</h3>', unsafe_allow_html=True)
    
    # Show which parameters this specific decision uses if provided
    if decision_name:
        decision_params = get_decision_global_parameters([decision_name])
        if decision_params:
            st.info(f"✅ This decision uses: {', '.join([p.replace('_', ' ').title() for p in sorted(decision_params)])}")
        else:
            st.info("ℹ️ This is a trait-based decision (doesn't use global parameters)")
    
    # Income Distribution Parameters
    st.markdown('<h4 class="subsection-header">💵 Income Distribution</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.text(f"Distribution Type: {st.session_state.sim_params.income_distribution}")
        st.text(f"Minimum Income: ${st.session_state.sim_params.income_min:,.0f}")
        st.text(f"{st.session_state.sim_params.income_avg_type.title()} Income: ${st.session_state.sim_params.income_avg:,.0f}")
    
    with col2:
        st.text(f"Maximum Income: ${st.session_state.sim_params.income_max:,.0f}")
        st.text(f"Discount Threshold: ${st.session_state.sim_params.discount_income_threshold:,.0f}")
    
    # Market Parameters
    st.markdown('<h4 class="subsection-header">🏪 Market Parameters</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.text(f"Number of Vendors: {st.session_state.sim_params.num_vendors}")
        st.text(f"Min Vendor Price: ${st.session_state.sim_params.vendor_price_min:.2f}")
    
    with col2:
        st.text(f"Average Market Price: ${st.session_state.sim_params.market_price:.2f}")
        st.text(f"Max Vendor Price: ${st.session_state.sim_params.vendor_price_max:.2f}")
    
    # Pricing Parameters
    st.markdown('<h4 class="subsection-header">💰 Pricing Parameters</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.text(f"Platform Markup: {st.session_state.sim_params.platform_markup:.0%}")
        st.text(f"Price Range: ±{st.session_state.sim_params.price_range:.0%}")
    
    with col2:
        st.text(f"Bidding Percentage: {st.session_state.sim_params.bidding_percentage:.0%}")
        st.text(f"Price Grid Categories: {st.session_state.sim_params.price_grid}")
    
    st.caption("💡 To modify these parameters, go to Page 1: Common Simulation Parameters")


def render_decision_tab(decision_name):
    """Render configuration for a specific decision"""
    if decision_name == "donation_default":
        render_donation_default_tab()
    else:
        # Placeholder for other decisions
        st.markdown(f'<h3 class="section-header">🎯 {decision_name.replace("_", " ").title()} Configuration</h3>', unsafe_allow_html=True)
        
        st.info(f"Configuration for {decision_name} will be implemented here.")
        st.caption("This decision currently uses default values.")
        
        # Display Global Parameters
        st.markdown("---")
        render_global_parameters_readonly(decision_name)
        
        # Basic Simulation Settings
        st.markdown("---")
        st.markdown('<h4 class="subsection-header">⚙️ Simulation Settings</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_agents = st.number_input(
                "Number of Agents",
                min_value=10,
                max_value=50000,
                value=st.session_state.n_agents,
                step=100,
                key=f"{decision_name}_n_agents"
            )
            st.session_state.n_agents = n_agents
            
            if st.session_state.sim_params.simulation_mode == "Single Run":
                seed = st.number_input(
                    "Random Seed",
                    min_value=1,
                    max_value=2147483647,
                    value=st.session_state.seed,
                    key=f"{decision_name}_seed"
                )
                st.session_state.seed = seed
        
        with col2:
            save_results = st.checkbox(
                "Save Results to File",
                value=st.session_state.save_results,
                key=f"{decision_name}_save_results"
            )
            st.session_state.save_results = save_results
        
        # Individual run button
        st.markdown("---")
        if st.button(f"🚀 Run {decision_name.replace('_', ' ').title()} Only", type="secondary", use_container_width=True, key=f"run_{decision_name}"):
            run_individual_decision(decision_name)


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
                    min_value=0.01,
                    max_value=1000.0,
                    value=st.session_state.sim_params.market_price,
                    step=0.01,
                    key="tab_market_price"
                )
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
                    step=0.05,
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
                    step=0.05,
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


def run_individual_decision(decision_name):
    """Run a single decision simulation"""
    with st.spinner(f"Running {decision_name} simulation..."):
        try:
            # Import required functions
            from app.simulation import run_simulation_from_sidebar
            
            # Temporarily modify selected decisions
            original_decisions = st.session_state.decision_params.selected_decisions.copy()
            st.session_state.decision_params.selected_decisions = [decision_name]
            
            # Run simulation
            run_simulation_from_sidebar()
            
            # Store in individual results
            if st.session_state.simulation_results:
                if 'individual_results' not in st.session_state:
                    st.session_state.individual_results = {}
                
                st.session_state.individual_results[decision_name] = st.session_state.simulation_results
                st.success(f"✅ {decision_name} simulation complete!")
                
                # Show preview of results
                results = next(iter(st.session_state.simulation_results.values()))
                if results is not None and not results.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Agents Simulated", f"{len(results):,}")
                    with col2:
                        if decision_name == "donation_default":
                            donation_col = 'donation_default_raw' if 'donation_default_raw' in results.columns else 'donation_default'
                            if donation_col in results.columns:
                                st.metric("Average Donation Rate", f"{results[donation_col].mean():.1%}")
            
            # Restore original decisions
            st.session_state.decision_params.selected_decisions = original_decisions
            
        except Exception as e:
            st.error(f"❌ Error running {decision_name}: {str(e)}")
            import traceback
            st.text(traceback.format_exc())


def run_combined_simulation(selected_decisions):
    """Run all selected decisions together"""
    from app.simulation import run_simulation_from_sidebar
    
    # Ensure the selected decisions are set
    st.session_state.decision_params.selected_decisions = selected_decisions
    
    # Run simulation
    run_simulation_from_sidebar()


def render_page2():
    """Render Page 2: Decision-Specific Parameters"""
    st.markdown('<h2 class="page-header">Page 2: Decision-Specific Parameters</h2>', unsafe_allow_html=True)
    
    # Decision selection
    st.markdown('<h3 class="section-header">🎯 Decision Selection</h3>', unsafe_allow_html=True)
    
    # Multi-select with "Select All" functionality
    select_all = st.checkbox("Select All Decisions", value=False)
    
    if select_all:
        selected_decisions = st.multiselect(
            "Selected Decisions",
            ALL_DECISIONS,
            default=ALL_DECISIONS,
            help="All decisions are selected",
            disabled=True
        )
    else:
        # Use session state to preserve selections when navigating between pages
        # But default to empty list if nothing was previously selected
        default_selections = st.session_state.decision_params.selected_decisions if hasattr(st.session_state.decision_params, 'selected_decisions') and st.session_state.decision_params.selected_decisions else []
        
        selected_decisions = st.multiselect(
            "Select Decisions to Run",
            ALL_DECISIONS,
            default=default_selections,
            help="Select one or more decisions to run",
            placeholder="Choose decisions..."
        )
    
    # Store selected decisions
    st.session_state.decision_params.selected_decisions = selected_decisions
    
    if not selected_decisions:
        st.warning("Please select at least one decision to configure parameters")
        # Navigation
        render_navigation('page2')
        return
    
    # Create tabs
    tab_names = ["📊 Overview"] + [f"🎯 {d.replace('_', ' ').title()}" for d in selected_decisions]
    tabs = st.tabs(tab_names)
    
    # Overview Tab
    with tabs[0]:
        render_overview_tab(selected_decisions)
    
    # Decision-specific tabs
    for i, decision in enumerate(selected_decisions):
        with tabs[i + 1]:
            render_decision_tab(decision)
    
    # Navigation
    render_navigation('page2')


def configure_sidebar(selected_decisions):
    """Configure the sidebar based on selected decisions"""
    st.sidebar.title("⚙️ Decision Parameters")
    
    # Only show sidebar content if decisions are selected
    if not selected_decisions:
        st.sidebar.info("👈 Select decisions on the main page to see applicable parameters")
        # Store minimal settings in session state for no decisions selected
        st.session_state.n_agents = 1000
        st.session_state.simulation_mode = "Single Run"
        st.session_state.seed = 42
        st.session_state.show_individual_agents = False
        st.session_state.save_results = True
    else:
        # Show decision-specific parameters only for selected decisions
        
        # Only show donation-specific parameters if donation_default is selected
        if "donation_default" in selected_decisions:
            # Population mode selector
            st.sidebar.subheader("Population Generation")
            population_mode = st.sidebar.radio(
                "Population Mode",
                ["Copula (synthetic)", "Research Specification", "Compare both"],
                index=0,
                help="Copula: Generate synthetic agents via fitted copula\nResearch: Use original participants with stochastic draws\nCompare both: Show both Copula and Research modes"
            )
            
            # Income specification selector
            if population_mode != "Dependent variable resampling":
                st.sidebar.subheader("Income Specification")
                income_spec_mode = st.sidebar.radio(
                    "Income Mode for Donation Model",
                    ["categorical only", "continuous only", "Compare both"],
                    index=0,
                    help="Choose income treatment: categorical (5 categories), continuous (linear), or Compare both"
                )
            else:
                income_spec_mode = "categorical only"
            
            # Stochastic component option
            if population_mode == "Copula (synthetic)" or population_mode == "Compare both":
                st.sidebar.subheader("Stochastic Component")
                sigma_in_copula = st.sidebar.checkbox(
                    "Add Normal(anchor, σ) draw to Copula runs",
                    value=False,
                    help="When enabled, Copula mode will also use the stochastic component (Normal distribution draw) like Research mode"
                )
                sigma_value_ui = st.sidebar.slider(
                    "σ (standard deviation) on 0–112 scale",
                    min_value=0.0,
                    max_value=15.0,
                    value=9.0,
                    step=0.1,
                    help="Controls the spread of the Normal(anchor, σ) draw. Set to 0 to disable variability."
                )
            else:
                sigma_in_copula = False
                sigma_value_ui = st.sidebar.slider(
                    "σ (standard deviation) on 0–112 scale",
                    min_value=0.0,
                    max_value=15.0,
                    value=9.0,
                    step=0.1,
                    help="Controls the spread of the Normal(anchor, σ) draw. Set to 0 to disable variability."
                )
            
            # Anchor weights slider
            if population_mode != "Dependent variable resampling":
                st.sidebar.subheader("Anchor Mix")
                anchor_observed_weight = st.sidebar.slider(
                    "Weight for observed versus modeled prosocial behavior",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.75,
                    step=0.05,
                    help="Anchor = w × Observed + (1-w) × Predicted. Default is 0.75 observed + 0.25 predicted."
                )
                st.sidebar.caption(f"Predicted weight: {1 - anchor_observed_weight:.2f}")
            else:
                anchor_observed_weight = 0.75
            
            # Raw output option
            if population_mode != "Copula (synthetic)" or sigma_in_copula:
                st.sidebar.subheader("Output Options")
                raw_draw_mode = st.sidebar.checkbox(
                    "Show pre-truncation (raw) donation rate",
                    value=False,
                    help="Display the raw Normal(anchor, σ) draw before any processing. This shows negative values and the full range of the stochastic draw before flooring at 0 and rescaling by personal maximum."
                )
            else:
                raw_draw_mode = False
        else:
            # If donation_default is not selected, set default values
            population_mode = "Copula (synthetic)"
            income_spec_mode = "categorical only"
            sigma_in_copula = False
            sigma_in_research = True
            sigma_value_ui = 9.0
            anchor_observed_weight = 0.75
            raw_draw_mode = False

        # Show applicable global parameters dynamically
        all_applicable = get_decision_global_parameters(selected_decisions)
        
        # Only show income parameters if they are applicable
        if any(param in all_applicable for param in ['income_distribution', 'income_min', 'income_max', 'income_avg', 'discount_income_threshold']):
            st.sidebar.subheader("💵 Income Distribution")
            st.sidebar.caption("✅ Applicable for selected decisions")
            
            if 'income_distribution' in all_applicable:
                income_dist_type = st.sidebar.selectbox(
                    "Distribution Type",
                    ["lognormal", "pareto", "weibull"],
                    index=["lognormal", "pareto", "weibull"].index(st.session_state.sim_params.income_distribution),
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.income_distribution = income_dist_type
            
            if 'income_min' in all_applicable:
                income_min = st.sidebar.number_input(
                    "Minimum Income",
                    min_value=0.0,
                    value=st.session_state.sim_params.income_min,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.income_min = income_min
            
            if 'income_avg' in all_applicable:
                # Show current type and value
                current_type = st.session_state.sim_params.income_avg_type.title()
                income_avg = st.sidebar.number_input(
                    f"{current_type} Income",
                    min_value=st.session_state.sim_params.income_min,
                    value=st.session_state.sim_params.income_avg,
                    help=f"✅ Applicable for selected decisions (configured as {current_type.lower()} on Page 1)"
                )
                st.session_state.sim_params.income_avg = income_avg
            
            if 'income_max' in all_applicable:
                income_max = st.sidebar.number_input(
                    "Maximum Income",
                    min_value=st.session_state.sim_params.income_avg,
                    value=st.session_state.sim_params.income_max,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.income_max = income_max
            
            if 'discount_income_threshold' in all_applicable:
                discount_threshold = st.sidebar.number_input(
                    "Discount Threshold",
                    min_value=st.session_state.sim_params.income_min,
                    max_value=st.session_state.sim_params.income_max,
                    value=st.session_state.sim_params.discount_income_threshold,
                    help="✅ Income threshold for discount qualification"
                )
                st.session_state.sim_params.discount_income_threshold = discount_threshold

        # Only show market parameters if they are applicable
        market_params = ['num_vendors', 'market_price', 'vendor_price_min', 'vendor_price_max']
        if any(param in all_applicable for param in market_params):
            st.sidebar.subheader("🏪 Market Parameters")
            st.sidebar.caption("✅ Applicable for selected decisions")
            
            if 'num_vendors' in all_applicable:
                num_vendors = st.sidebar.number_input(
                    "Number of Vendors",
                    min_value=1,
                    max_value=50,
                    value=st.session_state.sim_params.num_vendors,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.num_vendors = num_vendors
            
            if 'market_price' in all_applicable:
                market_price = st.sidebar.number_input(
                    "Average Market Price ($)",
                    min_value=0.01,
                    max_value=1000.0,
                    value=st.session_state.sim_params.market_price,
                    step=0.01,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.market_price = market_price

        # Only show pricing parameters if they are applicable
        pricing_params = ['platform_markup', 'price_range', 'price_grid', 'bidding_percentage']
        if any(param in all_applicable for param in pricing_params):
            st.sidebar.subheader("💰 Pricing Parameters")
            st.sidebar.caption("✅ Applicable for selected decisions")
            
            if 'platform_markup' in all_applicable:
                platform_markup = st.sidebar.slider(
                    "Platform Markup (m)",
                    min_value=0.0,
                    max_value=0.5,
                    value=st.session_state.sim_params.platform_markup,
                    step=0.01,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.platform_markup = platform_markup
            
            if 'bidding_percentage' in all_applicable:
                bidding_percentage = st.sidebar.slider(
                    "Bidding Percentage (bp)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.sim_params.bidding_percentage,
                    step=0.05,
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.bidding_percentage = bidding_percentage
        
        # Simulation parameters (always show if decisions are selected)
        st.sidebar.subheader("Simulation Parameters")
        
        # Show current simulation mode (set on Page 1)
        st.sidebar.info(f"📊 Mode: {st.session_state.sim_params.simulation_mode} (configured on Page 1)")
        
        n_agents = st.sidebar.number_input(
            "Number of Agents",
            min_value=10,
            max_value=50000,
            value=1000,
            step=100,
            help="Number of synthetic agents to generate"
        )
        
        # Use simulation mode from Page 1 (stored in session state)
        simulation_mode = st.session_state.sim_params.simulation_mode
        
        if simulation_mode == "Single Run":
            seed = st.sidebar.number_input(
                "Random Seed",
                min_value=1,
                max_value=2147483647,
                value=42,
                help="Seed for reproducible results"
            )
        else:
            n_runs = st.sidebar.number_input(
                "Number of Runs",
                min_value=2,
                max_value=1000,
                value=10,
                step=10,
                help="Number of Monte-Carlo repetitions (Note: 100+ runs can take several minutes)"
            )
            base_seed = st.sidebar.number_input(
                "Base Seed",
                min_value=1,
                max_value=2147483647,
                value=42,
                help="Starting seed (subsequent runs use base_seed + i)"
            )
        
        # Advanced options
        st.sidebar.subheader("🔧 Advanced Options")

        show_individual_agents = st.sidebar.checkbox(
            "Show Individual Agent Details",
            value=False,
            help="Display detailed breakdown of individual agents"
        )
        
        save_results = st.sidebar.checkbox(
            "Save Results to File",
            value=True,
            help="Save simulation outputs to outputs/ directory"
        )
        
        # Store settings in session state
        st.session_state.population_mode = population_mode
        st.session_state.income_spec_mode = income_spec_mode
        st.session_state.sigma_in_copula = sigma_in_copula
        st.session_state.sigma_in_research = sigma_in_research
        st.session_state.sigma_value_ui = sigma_value_ui
        st.session_state.anchor_observed_weight = anchor_observed_weight
        st.session_state.raw_draw_mode = raw_draw_mode
        st.session_state.n_agents = n_agents
        # Note: simulation_mode now comes from Page 1 (st.session_state.sim_params.simulation_mode)
        if simulation_mode == "Single Run":
            st.session_state.seed = seed
        else:
            st.session_state.n_runs = n_runs
            st.session_state.base_seed = base_seed
        st.session_state.show_individual_agents = show_individual_agents
        st.session_state.save_results = save_results
        
        # Summary info
        if st.sidebar.button("📊 Show Parameter Summary"):
            total_params = len(get_all_global_parameters())
            applicable_count = len(all_applicable)
            st.sidebar.success(f"✅ {applicable_count}/{total_params} parameters applicable ({applicable_count/total_params:.0%})")
    
    # Run simulation button
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
        # Check if simulation is already running
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
            
        if st.session_state.simulation_running:
            st.warning("⚠️ A simulation is already running. Please wait for it to complete.")
        else:
            st.session_state.simulation_running = True
            try:
                if st.session_state.sim_params.simulation_mode == "Single Run":
                    run_simulation_from_sidebar()
                    st.session_state.mc_results = None
                else:
                    mc_summary, mc_detailed, output_log = run_monte_carlo_study()
                    if mc_summary is not None:
                        st.session_state.mc_results = {
                            'summary': mc_summary,
                            'detailed': mc_detailed,
                            'log': output_log
                        }
                        st.session_state.simulation_results = None
                        st.session_state.page = 'results'
                        st.success("✅ Monte Carlo results saved to session state. Redirecting to results page...")
                        st.rerun()
                    else:
                        st.error("❌ Monte Carlo simulation returned no results")
            finally:
                st.session_state.simulation_running = False


def render_results_page():
    """Render the Results page"""
    st.markdown('<h2 class="page-header">Simulation Results</h2>', unsafe_allow_html=True)
    
    # Debug info
    with st.expander("🔧 Debug: Session State", expanded=False):
        st.write(f"simulation_results: {'Yes' if st.session_state.simulation_results is not None else 'No'}")
        st.write(f"mc_results: {'Yes' if st.session_state.mc_results is not None else 'No'}")
        if st.session_state.mc_results is not None:
            st.write(f"mc_results keys: {list(st.session_state.mc_results.keys())}")
            st.write(f"summary shape: {st.session_state.mc_results['summary'].shape if st.session_state.mc_results['summary'] is not None else 'None'}")
            st.write(f"detailed shape: {st.session_state.mc_results['detailed'].shape if st.session_state.mc_results['detailed'] is not None else 'None'}")
    
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
    
    
    results_dict = st.session_state.simulation_results
    
    # Show results based on mode
    if st.session_state.population_mode == "Compare both":
        render_population_comparison(results_dict)
    elif st.session_state.population_mode == "Dependent variable resampling":
        render_dependent_variable_results(results_dict)
    elif st.session_state.income_spec_mode == "Compare both":
        render_income_comparison(results_dict)
    else:
        # Single mode display
        df = next(iter(results_dict.values()))
        mode_name = next(iter(results_dict.keys()))
        show_overview(df, f" ({mode_name.title()})")
    
    # Get DataFrame for individual agent analysis
    if st.session_state.population_mode == "Compare both":
        if st.session_state.income_spec_mode == "Compare both":
            df = next((results_dict[k] for k in ["copula_categorical", "doc_mode_categorical", "copula_continuous", "doc_mode_continuous"] if k in results_dict), pd.DataFrame())
        else:
            income_type = "continuous" if st.session_state.income_spec_mode == "continuous only" else "categorical"
            df = next((results_dict[k] for k in [f"copula_{income_type}", f"doc_mode_{income_type}"] if k in results_dict), pd.DataFrame())
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


def render_population_comparison(results_dict):
    """Render population mode comparison results"""
    st.markdown("### 🔬 Population Mode Comparison")
    
    if st.session_state.income_spec_mode == "Compare both":
        # 2x2 grid: copula vs doc_mode x categorical vs continuous
        st.markdown("#### Copula (Synthetic Agents)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Categorical Income**")
            if "copula_categorical" in results_dict:
                show_overview(results_dict["copula_categorical"], " (Copula, Cat)")
            else:
                st.caption("Categorical results not available")
        with col2:
            st.markdown("**Continuous Income**")
            if "copula_continuous" in results_dict:
                show_overview(results_dict["copula_continuous"], " (Copula, Cont)")
            else:
                st.caption("Continuous results not available")
        
        st.markdown("---")
        st.markdown("#### Research Mode (Original + Stochastic)")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Categorical Income**")
            if "doc_mode_categorical" in results_dict:
                show_overview(results_dict["doc_mode_categorical"], " (Research, Cat)")
            else:
                st.caption("Categorical results not available")
        with col4:
            st.markdown("**Continuous Income**")
            if "doc_mode_continuous" in results_dict:
                show_overview(results_dict["doc_mode_continuous"], " (Research, Cont)")
            else:
                st.caption("Continuous results not available")
    else:
        # Single income mode, compare population modes
        income_type = "continuous" if st.session_state.income_spec_mode == "continuous only" else "categorical"
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧬 Copula (Synthetic)")
            copula_key = f"copula_{income_type}"
            if copula_key in results_dict:
                show_overview(results_dict[copula_key], f" (Copula, {income_type.title()})")
            else:
                st.caption(f"Copula {income_type} results not available")
        
        with col2:
            st.markdown("#### 📄 Research Mode")
            doc_key = f"doc_mode_{income_type}"
            if doc_key in results_dict:
                show_overview(results_dict[doc_key], f" (Research, {income_type.title()})")
            else:
                st.caption(f"Research {income_type} results not available")


def render_dependent_variable_results(results_dict):
    """Render dependent variable resampling results"""
    raw_suffix = " (Raw Pre-truncation)" if st.session_state.raw_draw_mode else ""
    st.markdown(f"### 📊 Dependent Variable Resampling{raw_suffix}")
    if st.session_state.raw_draw_mode:
        st.caption("This mode resamples from the empirical distribution of RAW (pre-truncation) donation rates computed from the original 280 participants. These values represent the Normal(anchor, σ) draw before flooring at 0 and rescaling by personal maximum.")
    else:
        st.caption("This mode resamples from the empirical distribution of donation rates computed from the original 280 participants. No trait information is preserved.")
    
    df = results_dict["depvar"]
    show_dependent_variable_comparison(df)


def render_income_comparison(results_dict):
    """Render income specification comparison results"""
    st.markdown("### 📊 Income Specification Comparison")
    
    col_cat, col_cont = st.columns(2, gap="large")
    
    with col_cat:
        st.markdown("#### 📋 Categorical Income")
        if "categorical" in results_dict:
            show_overview(results_dict["categorical"], " (Categorical)")
        else:
            st.caption("Categorical results not available")
    
    with col_cont:
        st.markdown("#### 📈 Continuous Income") 
        if "continuous" in results_dict:
            show_overview(results_dict["continuous"], " (Continuous)")
        else:
            st.caption("Continuous results not available")


def render_parameter_applicability_summary():
    """Render parameter applicability summary for the run"""
    selected_decisions = st.session_state.decision_params.selected_decisions
    
    if selected_decisions:
        # Calculate overall applicability
        total_applicable = get_decision_global_parameters(selected_decisions)
        all_global_params = get_all_global_parameters()
        total_not_applicable = all_global_params - total_applicable
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Parameters", len(all_global_params))
        with col2:
            st.metric("✅ Applicable", len(total_applicable))
        with col3:
            st.metric("❌ Not Applicable", len(total_not_applicable))
        with col4:
            applicability_pct = len(total_applicable) / len(all_global_params) * 100 if all_global_params else 0
            st.metric("📈 Efficiency", f"{applicability_pct:.0f}%")
        
            # Show which parameters were actually used vs unused
        col_used, col_unused = st.columns(2)
        
        with col_used:
            st.markdown("### ✅ Parameters Used in This Simulation")
            if total_applicable:
                for param in sorted(total_applicable):
                    st.markdown(f"  • {param.replace('_', ' ').title()}")
            else:
                st.caption("No parameters were applicable for the selected decisions.")
        
        with col_unused:
            st.markdown("### ❌ Parameters Not Used in This Simulation")
            if total_not_applicable:
                for param in sorted(total_not_applicable):
                    st.markdown(f"  • {param.replace('_', ' ').title()}")
            else:
                st.caption("All parameters were used in this simulation.")
        
        # Show decision-specific breakdown
        st.markdown("### 📊 Parameter Usage by Decision")
        
        try:
            decisions_path = Path(__file__).resolve().parents[1] / "config" / "decisions.yaml"
            with open(decisions_path, 'r') as f:
                decisions_config = yaml.safe_load(f)
            
            for decision in selected_decisions:
                decision_config = decisions_config.get(decision, {})
                decision_params = set(decision_config.get('uses_global_parameters', []))
                not_used = all_global_params - decision_params
                efficiency = len(decision_params) / len(all_global_params) * 100 if all_global_params else 0
                
                with st.container():
                    col_title, col_metrics = st.columns([2, 3])
                    
                    with col_title:
                        st.markdown(f"**{decision.replace('_', ' ').title()}**")
                    
                    with col_metrics:
                        sub_col1, sub_col2, sub_col3 = st.columns(3)
                        with sub_col1:
                            st.metric("Uses", len(decision_params), label_visibility="collapsed")
                        with sub_col2:
                            st.metric("Doesn't Use", len(not_used), label_visibility="collapsed")
                        with sub_col3:
                            st.metric("Efficiency", f"{efficiency:.0f}%", label_visibility="collapsed")
                    
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error loading decision configurations: {e}")
    else:
        st.caption("No decisions were selected for this simulation.")


def render_individual_agent_details(df):
    """Render individual agent details section"""
    # Check if this is dependent variable mode
    is_depvar_mode = len(df.columns) == 1 and 'donation_default' in df.columns
    
    if not is_depvar_mode:
        st.subheader("🔍 Individual Agent Details")
        
        # Agent selection
        agent_id = st.selectbox(
            "Select Agent to Examine",
            options=range(len(df)),
            format_func=lambda x: f"Agent {x+1}"
        )
        
        agent_data = df.iloc[agent_id]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("** Agent Traits**")
            trait_data = {}
            for col in ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program', 
                       'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']:
                if col in agent_data:
                    trait_data[col] = agent_data[col]
            
            trait_df = pd.DataFrame(list(trait_data.items()), columns=['Trait', 'Value'])
            # Convert all values to strings to avoid PyArrow serialization issues
            trait_df['Value'] = trait_df['Value'].astype(str)
            st.dataframe(trait_df, hide_index=True)
        
        with col2:
            st.markdown("**🎯 Agent Decisions**")
            decision_data = {}
            for col in df.columns:
                if col not in ['Assigned Allowance Level', 'Group_experiment', 'Honesty_Humility', 
                              'Study Program', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']:
                    decision_data[col] = agent_data[col]
            
            decision_df = pd.DataFrame(list(decision_data.items()), columns=['Decision', 'Value'])
            if 'donation_default' in decision_data:
                decision_df.loc[decision_df['Decision'] == 'donation_default', 'Value'] = \
                    f"{decision_data['donation_default']:.1%}"
            if 'donation_default_raw' in decision_data:
                decision_df.loc[decision_df['Decision'] == 'donation_default_raw', 'Value'] = \
                    f"{decision_data['donation_default_raw']:.1%}"
            # Convert all values to strings to avoid PyArrow serialization issues
            decision_df['Value'] = decision_df['Value'].astype(str)
            st.dataframe(decision_df, hide_index=True)
    else:
        st.caption("Individual agent details not available in dependent variable resampling mode (no trait information)")


def render_export_section(df):
    """Render the export/download section"""
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"enhanced_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        if st.button("🔄 Clear Results"):
            st.session_state.simulation_results = None
            st.rerun()
