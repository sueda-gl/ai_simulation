# app/pages/page1_common_params.py
"""
Page 1: Common Simulation Parameters for the Enhanced AI Agent Simulation.
"""
import streamlit as st
import pandas as pd
from app.components import show_income_distribution_histogram
from app.pages.navigation import render_navigation


def render_page1():
    """Render Page 1: Common Simulation Parameters"""
    st.markdown('<h2 class="page-header">Page 1: Common Simulation Parameters</h2>', unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulation Mode Selection
        st.markdown('<h3 class="section-header">🎯 Simulation Execution Mode</h3>', unsafe_allow_html=True)
        
        # Initialize the widget key if it doesn't exist
        if "page1_simulation_execution_mode" not in st.session_state:
            st.session_state.page1_simulation_execution_mode = "Snapshot" if st.session_state.sim_params.simulation_execution_mode == "snapshot" else "Live Simulation"
        
        simulation_execution_mode = st.radio(
            "Execution Mode",
            ["Snapshot", "Live Simulation"],
            horizontal=True,
            help="Snapshot: Run simulation once with fixed parameters\nLive: Real-time simulation with dynamic updates (backend implementation pending)",
            key="page1_simulation_execution_mode"
        )
        
        # Sync the widget's state to sim_params
        st.session_state.sim_params.simulation_execution_mode = "snapshot" if st.session_state.page1_simulation_execution_mode == "Snapshot" else "live"
        
        # Simulation Mode (Single vs Monte-Carlo)
        st.markdown('<h3 class="section-header">🎲 Simulation Mode</h3>', unsafe_allow_html=True)
        
        # Initialize the widget key if it doesn't exist
        if "page1_simulation_mode" not in st.session_state:
            st.session_state.page1_simulation_mode = st.session_state.sim_params.simulation_mode
        
        simulation_mode = st.radio(
            "Analysis Mode",
            ["Single Run", "Monte-Carlo Study"],
            horizontal=True,
            help="Single Run: One simulation with specified parameters\nMonte-Carlo: Multiple runs for uncertainty analysis",
            key="page1_simulation_mode"
        )
        
        # Sync the widget's state to sim_params
        st.session_state.sim_params.simulation_mode = st.session_state.page1_simulation_mode
        
        if st.session_state.sim_params.simulation_mode == "Single Run":
            st.info("📊 Single Run: Execute one simulation with current parameters")
        else:
            st.info("🎯 Monte-Carlo: Execute multiple runs for statistical analysis")
        
        # Simulation Settings Section
        st.markdown('<h3 class="section-header">⚙️ Simulation Settings</h3>', unsafe_allow_html=True)
        
        n_agents = st.number_input(
            "Number of Agents",
            min_value=10,
            max_value=50000,
            value=st.session_state.n_agents,
            step=100,
            help="Number of agents to generate for the simulation",
            key="n_agents_input",
            on_change=lambda: setattr(st.session_state, 'n_agents', st.session_state.n_agents_input)
        )
        
        if st.session_state.sim_params.simulation_mode == "Single Run":
            seed = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=2147483647,
                value=st.session_state.seed,
                help="Seed for random number generation (for reproducible results)",
                key="seed_input",
                on_change=lambda: setattr(st.session_state, 'seed', st.session_state.seed_input)
            )
        else:
            n_runs = st.number_input(
                "Number of Runs",
                min_value=2,
                max_value=1000,
                value=st.session_state.n_runs,
                step=10,
                help="Number of Monte-Carlo runs to execute",
                key="n_runs_input",
                on_change=lambda: setattr(st.session_state, 'n_runs', st.session_state.n_runs_input)
            )
            
            base_seed = st.number_input(
                "Base Seed",
                min_value=1,
                max_value=2147483647,
                value=st.session_state.base_seed,
                help="Base seed for Monte-Carlo runs (each run uses base_seed + run_number)",
                key="base_seed_input",
                on_change=lambda: setattr(st.session_state, 'base_seed', st.session_state.base_seed_input)
            )
        
        show_individual_agents = st.checkbox(
            "Show Individual Agent Details",
            value=st.session_state.show_individual_agents,
            help="Display detailed information for each agent in results",
            key="show_individual_agents_checkbox",
            on_change=lambda: setattr(st.session_state, 'show_individual_agents', st.session_state.show_individual_agents_checkbox)
        )
        
        save_results = st.checkbox(
            "Save Results to File",
            value=st.session_state.save_results,
            help="Save simulation results to a file for later analysis",
            key="save_results_checkbox",
            on_change=lambda: setattr(st.session_state, 'save_results', st.session_state.save_results_checkbox)
        )
        
        # Time Parameters Section
        st.markdown('<h3 class="section-header">⏱️ Time Parameters</h3>', unsafe_allow_html=True)
        
        periods = st.number_input(
            "Number of Periods",
            min_value=1,
            max_value=100,
            value=st.session_state.sim_params.periods,
            help="Number of periods for simulation run",
            key="periods_input",
            on_change=lambda: setattr(st.session_state.sim_params, 'periods', st.session_state.periods_input)
        )
        
        duration_hours = st.number_input(
            "Duration per Period (hours)",
            min_value=1,
            max_value=24,
            value=int(st.session_state.sim_params.duration_hours),
            step=1,
            help="Duration of each period in hours (will be converted to seconds for simulation)",
            key="duration_hours_input",
            on_change=lambda: setattr(st.session_state.sim_params, 'duration_hours', float(st.session_state.duration_hours_input))
        )
        st.caption(f"Duration in seconds: {st.session_state.sim_params.get_duration_seconds():.0f}")
    
    with col2:
        # Market Parameters Section
        st.markdown('<h3 class="section-header">🏪 Market Parameters</h3>', unsafe_allow_html=True)
        
        platform_markup = st.slider(
            "Platform Markup (m)",
            min_value=0.0,
            max_value=0.5,
            value=st.session_state.sim_params.platform_markup,
            step=0.01,
            help="Platform markup: Customer Price = (1+m) × Vendor Price",
            key="platform_markup_slider",
            on_change=lambda: setattr(st.session_state.sim_params, 'platform_markup', st.session_state.platform_markup_slider)
        )
        
        price_range = st.slider(
            "Price Range (r)",
            min_value=0.0,
            max_value=1.0,  # Extended from 0.5 to 1.0 for simulation flexibility
            value=st.session_state.sim_params.price_range,
            step=0.05,
            help="Price range for Purchase Now and Minimum Bid prices. Extended to 1.0 for simulation flexibility.",
            key="price_range_slider",
            on_change=lambda: setattr(st.session_state.sim_params, 'price_range', st.session_state.price_range_slider)
        )
        
        bidding_percentage = st.slider(
            "Bidding Percentage (bp)",
            min_value=0.0,
            max_value=1.0,  # Extended from 0.5 to 1.0 as requested
            value=st.session_state.sim_params.bidding_percentage,
            step=0.05,
            help="Proportion of products available for bidding (NA = bp × NV). Now supports up to 100%!",
            key="bidding_percentage_slider",
            on_change=lambda: setattr(st.session_state.sim_params, 'bidding_percentage', st.session_state.bidding_percentage_slider)
        )
        st.caption(f"Products for auction per vendor: {st.session_state.sim_params.get_num_auction_products()}")
        
        def update_price_grid():
            # Ensure price grid is odd
            value = st.session_state.price_grid_input
            if value % 2 == 0:
                value += 1
            st.session_state.sim_params.price_grid = value
            # Reset NFIC tracking when price grid changes
            if "nfic_manually_set" in st.session_state:
                st.session_state.nfic_manually_set = False
        
        price_grid = st.number_input(
            "Price Grid Categories (g)",
            min_value=3,
            max_value=21,
            value=st.session_state.sim_params.price_grid,
            step=2,
            help="Number of price categories (must be odd)",
            key="price_grid_input",
            on_change=update_price_grid
        )
        
        # Show adjustment message if needed
        if st.session_state.sim_params.price_grid % 2 == 0:
            st.caption("Price grid adjusted to odd number: " + str(st.session_state.sim_params.price_grid + 1))
        elif st.session_state.price_grid_input != st.session_state.sim_params.price_grid:
            st.caption(f"Price grid adjusted to odd number: {st.session_state.sim_params.price_grid}")
        
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
            st.markdown("**Lognormal Distribution Parameters**")
            st.caption("X = a + Y, where Y ~ Lognormal(μ, σ)")
            
            # Add helpful tip about parameter values
            with st.expander("💡 Parameter Guidelines"):
                st.markdown("""
                **Understanding μ and σ:**
                - Y ~ Lognormal(μ, σ) means ln(Y) ~ Normal(μ, σ)
                - Mean of Y = exp(μ + σ²/2)
                - For realistic income distributions:
                  - μ typically ranges from 8 to 12
                  - σ typically ranges from 0.3 to 1.0
                - Example: μ=10, σ=0.5 gives Y with mean ≈ $25,000
                """)
            
            # Mu and Sigma parameters
            col_mu, col_sigma = st.columns(2)
            
            with col_mu:
                lognormal_mu = st.number_input(
                    "μ (mu) - Mean of ln(Y)",
                    min_value=0.0,
                    max_value=15.0,
                    value=st.session_state.sim_params.lognormal_mu,
                    step=0.1,
                    help="Mean parameter of the log-transformed values. For income distributions, typically 8-12.",
                    key="lognormal_mu_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'lognormal_mu', st.session_state.lognormal_mu_input)
                )
            
            with col_sigma:
                lognormal_sigma = st.number_input(
                    "σ (sigma) - Std Dev of ln(Y)",
                    min_value=0.1,
                    max_value=3.0,
                    value=st.session_state.sim_params.lognormal_sigma,
                    step=0.1,
                    help="Standard deviation parameter of the log-transformed values",
                    key="lognormal_sigma_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'lognormal_sigma', st.session_state.lognormal_sigma_input)
                )
            
            # Min and Max parameters
            col_min, col_max = st.columns(2)
            
            with col_min:
                lognormal_min = st.number_input(
                    "a - Minimum Value ($)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=st.session_state.sim_params.lognormal_min,
                    step=100.0,
                    help="Linear shift: all values will be at least this amount",
                    key="lognormal_min_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'lognormal_min', st.session_state.lognormal_min_input)
                )
            
            with col_max:
                # Checkbox for enabling max
                use_max = st.checkbox(
                    "Set Maximum Value",
                    value=st.session_state.sim_params.lognormal_max is not None,
                    help="Enable rejection sampling to enforce maximum bound",
                    key="use_lognormal_max"
                )
                
                if use_max:
                    if st.session_state.sim_params.lognormal_max is None:
                        st.session_state.sim_params.lognormal_max = st.session_state.sim_params.lognormal_min + 100000.0
                    
                    lognormal_max = st.number_input(
                        "b - Maximum Value ($)",
                        min_value=st.session_state.sim_params.lognormal_min + 1000.0,
                        max_value=1000000.0,
                        value=st.session_state.sim_params.lognormal_max,
                        step=1000.0,
                        help="Rejection sampling: values above this will be resampled",
                        key="lognormal_max_input",
                        on_change=lambda: setattr(st.session_state.sim_params, 'lognormal_max', st.session_state.lognormal_max_input)
                    )
                else:
                    st.session_state.sim_params.lognormal_max = None
            
            # Show income range info
            max_display = "∞" if st.session_state.sim_params.lognormal_max is None else f"${st.session_state.sim_params.lognormal_max:,.0f}"
            st.info(f"📊 Income Range: [${st.session_state.sim_params.lognormal_min:,.0f}, {max_display}]")
            
        elif income_distribution == "pareto":
            st.markdown("**Pareto Distribution Parameters**")
            # Parameters will be added here
            
        elif income_distribution == "weibull":
            st.markdown("**Weibull Distribution Parameters**")
            # Parameters will be added here
        
        # Always show distribution preview
        st.markdown("### 📊 Distribution Preview")
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
                help="Number of customer discount income categories",
                key="num_discount_categories_input",
                on_change=lambda: setattr(st.session_state.sim_params, 'num_discount_categories', st.session_state.num_discount_categories_input)
            )
        
        with col_cat2:
            # Set default NFIC to price_grid - 1
            default_nfic = st.session_state.sim_params.price_grid - 1
            
            # Initialize tracking for manual changes
            if "nfic_manually_set" not in st.session_state:
                st.session_state.nfic_manually_set = False
                # Set initial value to price_grid - 1
                st.session_state.sim_params.num_fixed_categories = default_nfic
            
            # Update to follow price_grid if not manually changed
            if not st.session_state.nfic_manually_set:
                st.session_state.sim_params.num_fixed_categories = default_nfic
            
            def update_nfic():
                st.session_state.sim_params.num_fixed_categories = st.session_state.num_fixed_categories_input
                st.session_state.nfic_manually_set = True
            
            num_fixed_categories = st.number_input(
                "Fixed Income Categories (NFIC)",
                min_value=1,
                max_value=50,
                value=st.session_state.sim_params.num_fixed_categories,
                help=f"Number of customer fixed income categories (Default: Price Grid - 1 = {default_nfic})",
                key="num_fixed_categories_input",
                on_change=update_nfic
            )
    
    # Vendor Configuration - Single Source of Truth
    st.markdown('<h3 class="section-header">🏪 Vendor Configuration</h3>', unsafe_allow_html=True)
    st.caption("Configure all vendor settings: number, prices, products, and carryover behavior")
    
    # Number of Vendors (moved from Market Parameters)
    num_vendors = st.number_input(
        "Number of Vendors (N)",
        min_value=1,
        max_value=50,
        value=st.session_state.sim_params.num_vendors,
        help="Total number of vendors operating on the platform",
        key="num_vendors_input",
        on_change=lambda: setattr(st.session_state.sim_params, 'num_vendors', st.session_state.num_vendors_input)
    )
    
    # Check if we have single vendor (simplified interface) or multiple vendors (full interface)
    if st.session_state.sim_params.num_vendors == 1:
        # Single Vendor Configuration - Simplified Interface

        
        col_single_left, col_single_right = st.columns(2)
        
        with col_single_left:
            # Single vendor price
            single_vendor_price = st.number_input(
                "Product Price ($)",
                min_value=1,
                max_value=100000,
                value=int(st.session_state.sim_params.market_price * 100),
                help="Price for the single vendor (in cents, will be converted to dollars)",
                key="single_vendor_price_input",
                on_change=lambda: setattr(st.session_state.sim_params, 'market_price', st.session_state.single_vendor_price_input / 100.0)
            )
            # Also set min and max to the same value for consistency
            st.session_state.sim_params.vendor_price_min = st.session_state.sim_params.market_price
            st.session_state.sim_params.vendor_price_max = st.session_state.sim_params.market_price
            
        with col_single_right:
            # Single vendor products
            single_vendor_products = st.number_input(
                "Products Offered",
                min_value=1,
                max_value=10000,
                value=st.session_state.sim_params.vendor_products_avg,
                help="Number of products offered by the vendor",
                key="single_vendor_products_input",
                on_change=lambda: setattr(st.session_state.sim_params, 'vendor_products_avg', st.session_state.single_vendor_products_input)
            )
            # Also set min and max to the same value for consistency
            st.session_state.sim_params.vendor_products_min = st.session_state.sim_params.vendor_products_avg
            st.session_state.sim_params.vendor_products_max = st.session_state.sim_params.vendor_products_avg
        
        # Single vendor carryover
        def update_single_vendor_carryover():
            st.session_state.sim_params.global_carryover = st.session_state.single_vendor_carryover
            st.session_state.sim_params.override_carryover = True  # Always override in single vendor mode
        
        single_vendor_carryover = st.checkbox(
            "Enable Carryover for Vendor",
            value=st.session_state.sim_params.global_carryover,
            help="If checked, unsold products carry over to the next period",
            key="single_vendor_carryover",
            on_change=update_single_vendor_carryover
        )
        
        # Summary for single vendor

            
    else:
        # Multiple Vendors Configuration - Full Interface
        st.info("🏢 **Multiple Vendors Mode**: Full configuration for multiple vendors")
        
        # Vendor Setup Mode
        # Initialize the widget key if it doesn't exist
        if "page1_vendor_setup_mode" not in st.session_state:
            st.session_state.page1_vendor_setup_mode = "Generate Randomly" if st.session_state.sim_params.vendor_config_mode == "random" else "Upload Vendor Config File"
        
        vendor_setup_mode = st.radio(
            "Vendor Setup Mode",
            ["Generate Randomly", "Upload Vendor Config File"],
            horizontal=True,
            help="Choose how to configure vendor properties (price, products, carryover)",
            key="page1_vendor_setup_mode"
        )
        
        # Sync the widget's state to sim_params
        st.session_state.sim_params.vendor_config_mode = "random" if st.session_state.page1_vendor_setup_mode == "Generate Randomly" else "upload"
        
        if st.session_state.page1_vendor_setup_mode == "Generate Randomly":
            
            # Create columns for organized layout
            col_left, col_right = st.columns(2)
        
            with col_left:
                # Price Configuration
                st.markdown('<h4 class="subsection-header">💰 Price Configuration</h4>', unsafe_allow_html=True)
                
                vendor_price_min = st.number_input(
                    "Min Price per Vendor ($)",
                    min_value=1,
                    max_value=100000,
                    value=int(st.session_state.sim_params.vendor_price_min * 100),
                    help="Minimum price any vendor can have (in cents, will be converted to dollars)",
                    key="vendor_price_min_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'vendor_price_min', st.session_state.vendor_price_min_input / 100.0)
                )
                
                vendor_price_max = st.number_input(
                    "Max Price per Vendor ($)",
                    min_value=int(st.session_state.sim_params.vendor_price_min * 100),
                    max_value=100000,
                    value=int(st.session_state.sim_params.vendor_price_max * 100),
                    help="Maximum price any vendor can have (in cents, will be converted to dollars)",
                    key="vendor_price_max_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'vendor_price_max', st.session_state.vendor_price_max_input / 100.0)
                )
                
                market_price = st.number_input(
                    "Average Price per Vendor ($)",
                    min_value=int(st.session_state.sim_params.vendor_price_min * 100),
                    max_value=int(st.session_state.sim_params.vendor_price_max * 100),
                    value=int(st.session_state.sim_params.market_price * 100),
                    help="Target average price across all vendors (in cents, will be converted to dollars)",
                    key="market_price_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'market_price', st.session_state.market_price_input / 100.0)
                )
                
                # Price validation
                price_total_min = st.session_state.sim_params.num_vendors * st.session_state.sim_params.vendor_price_min
                price_total_max = st.session_state.sim_params.num_vendors * st.session_state.sim_params.vendor_price_max
                price_total_avg = st.session_state.sim_params.num_vendors * st.session_state.sim_params.market_price
                price_valid = price_total_min <= price_total_avg <= price_total_max
            
 
        
            with col_right:
                # Products Configuration
                st.markdown('<h4 class="subsection-header">📦 Products Configuration</h4>', unsafe_allow_html=True)
                
                vendor_products_min = st.number_input(
                "Min Products per Vendor",
                min_value=1,
                max_value=10000,
                value=st.session_state.sim_params.vendor_products_min,
                help="Minimum products any vendor can offer per period",
                key="vendor_products_min_input",
                on_change=lambda: setattr(st.session_state.sim_params, 'vendor_products_min', st.session_state.vendor_products_min_input)
            )
                
                vendor_products_max = st.number_input(
                "Max Products per Vendor",
                min_value=st.session_state.sim_params.vendor_products_min,
                max_value=10000,
                value=st.session_state.sim_params.vendor_products_max,
                help="Maximum products any vendor can offer per period",
                key="vendor_products_max_input",
                on_change=lambda: setattr(st.session_state.sim_params, 'vendor_products_max', st.session_state.vendor_products_max_input)
            )
                
                vendor_products_avg = st.number_input(
                "Average Products per Vendor",
                min_value=st.session_state.sim_params.vendor_products_min,
                max_value=st.session_state.sim_params.vendor_products_max,
                value=st.session_state.sim_params.vendor_products_avg,
                help="Target average products per vendor",
                key="vendor_products_avg_input",
                on_change=lambda: setattr(st.session_state.sim_params, 'vendor_products_avg', st.session_state.vendor_products_avg_input)
            )
                
                # Products validation
                products_total_min = st.session_state.sim_params.num_vendors * st.session_state.sim_params.vendor_products_min
                products_total_max = st.session_state.sim_params.num_vendors * st.session_state.sim_params.vendor_products_max
                products_total_avg = st.session_state.sim_params.num_vendors * st.session_state.sim_params.vendor_products_avg
                products_valid = products_total_min <= products_total_avg <= products_total_max
        
            # Carryover Configuration (full width)
            st.markdown('<h4 class="subsection-header">🔄 Carryover Configuration</h4>', unsafe_allow_html=True)
            
            # Initialize the widget key if it doesn't exist
            if "page1_carryover_mode" not in st.session_state:
                # Determine initial mode based on current settings
                if st.session_state.sim_params.override_carryover:
                    if st.session_state.sim_params.global_carryover:
                        st.session_state.page1_carryover_mode = "All vendors have carryover"
                    else:
                        st.session_state.page1_carryover_mode = "No vendors have carryover"
                else:
                    st.session_state.page1_carryover_mode = "Use probability"
            
            carryover_mode = st.radio(
                "Carryover Mode",
                ["All vendors have carryover", "No vendors have carryover", "Use probability"],
                horizontal=True,
                help="Choose how to apply carryover settings to vendors",
                key="page1_carryover_mode"
            )
            
            # Apply the selected mode
            if st.session_state.page1_carryover_mode == "All vendors have carryover":
                st.session_state.sim_params.override_carryover = True
                st.session_state.sim_params.global_carryover = True
                expected_carryover_vendors = st.session_state.sim_params.num_vendors
                st.info(f"✅ All {st.session_state.sim_params.num_vendors} vendors will have carryover enabled")
                
            elif st.session_state.page1_carryover_mode == "No vendors have carryover":
                st.session_state.sim_params.override_carryover = True
                st.session_state.sim_params.global_carryover = False
                expected_carryover_vendors = 0
                st.info(f"❌ None of the {st.session_state.sim_params.num_vendors} vendors will have carryover")
                
            else:  # Use probability
                st.session_state.sim_params.override_carryover = False
                
                vendor_carryover_probability = st.slider(
                    "Carryover Probability (p)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.sim_params.vendor_carryover_probability,
                    step=0.05,
                    help="Probability that any given vendor will have carryover enabled (Bernoulli per vendor)",
                    key="vendor_carryover_probability_slider",
                    on_change=lambda: setattr(st.session_state.sim_params, 'vendor_carryover_probability', st.session_state.vendor_carryover_probability_slider)
                )
                expected_carryover_vendors = int(st.session_state.sim_params.num_vendors * st.session_state.sim_params.vendor_carryover_probability)
                st.info(f"🎲 Expected ~{expected_carryover_vendors} vendors (out of {st.session_state.sim_params.num_vendors}) to have carryover based on {st.session_state.sim_params.vendor_carryover_probability:.0%} probability")
            
            # Summary chips
            st.markdown('<h4 class="subsection-header">📊 Configuration Summary</h4>', unsafe_allow_html=True)
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            
            with col_sum1:
                st.metric("Total Vendors", f"{st.session_state.sim_params.num_vendors}")
            with col_sum2:
                st.metric("Expected Total Products", f"{products_total_avg:,}")
            with col_sum3:
                st.metric("Expected Total Revenue", f"${price_total_avg:.2f}")
            with col_sum4:
                st.metric("Expected Vendors with Carryover", f"{expected_carryover_vendors}")

        else:
            
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
    
    # Initialize the widget key if it doesn't exist
    if "page1_apply_limits" not in st.session_state:
        # Use the existing apply_consumption_limits value
        st.session_state.page1_apply_limits = "Yes" if st.session_state.sim_params.apply_consumption_limits else "No"
    
    apply_limits = st.radio(
        "Apply Consumption Limits?",
        ["Yes", "No"],
        horizontal=True,
        help="Choose whether to apply consumption limits per income category",
        key="page1_apply_limits"
    )
    
    # Sync the widget's state to the sim_params
    st.session_state.sim_params.apply_consumption_limits = (st.session_state.page1_apply_limits == "Yes")
    
    if st.session_state.sim_params.apply_consumption_limits:
        st.caption("Set consumption limits per product for each income category per period")
        
        # Configuration source
        # Initialize the widget key if it doesn't exist
        if "page1_limits_source" not in st.session_state:
            # Use the existing consumption_limits_source value
            st.session_state.page1_limits_source = "Manual Entry" if st.session_state.sim_params.consumption_limits_source == "manual" else "Upload CSV"
        
        limits_source = st.radio(
            "Limits Configuration Source",
            ["Manual Entry", "Upload CSV"],
            horizontal=True,
            key="page1_limits_source"
        )
        
        # Sync the widget's state to the sim_params
        st.session_state.sim_params.consumption_limits_source = "manual" if st.session_state.page1_limits_source == "Manual Entry" else "upload"
        
        if st.session_state.page1_limits_source == "Manual Entry":
            
            # Create a simple interface for setting consumption limits
            total_categories = st.session_state.sim_params.num_fixed_categories
            
            # Initialize consumption limits in session state if not exists
            if "consumption_limits_temp" not in st.session_state:
                st.session_state.consumption_limits_temp = st.session_state.sim_params.consumption_limits.copy()
            
            consumption_limits = {}
            cols = st.columns(min(5, total_categories))
            for i in range(total_categories):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    key = f"consumption_limit_{i}"
                    cat_key = f"cat_{i+1}"
                    
                    # Initialize widget key if it doesn't exist
                    if key not in st.session_state:
                        st.session_state[key] = st.session_state.sim_params.consumption_limits.get(cat_key, 10)
                    
                    limit = st.number_input(
                        f"Category {i+1} Limit",
                        min_value=0,
                        max_value=100,
                        value=st.session_state[key],
                        key=key,
                        on_change=lambda k=key, c=cat_key: st.session_state.consumption_limits_temp.update({c: st.session_state[k]})
                    )
                    consumption_limits[cat_key] = st.session_state[key]
            
            st.session_state.sim_params.consumption_limits = consumption_limits
            
        else:
            
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
    
    # Population Mode Selection (Global Parameter)
    st.markdown('<h3 class="section-header">🧬 Population Generation Mode</h3>', unsafe_allow_html=True)
    st.caption("This setting applies to all decisions and determines how agents are generated")
    
    # Initialize the widget key if it doesn't exist
    if "page1_population_mode" not in st.session_state:
        # Use the existing population_mode value or default to "Copula (synthetic)"
        st.session_state.page1_population_mode = getattr(st.session_state, "population_mode", "Copula (synthetic)")
    
    population_mode = st.radio(
        "Population Mode",
        ["Copula (synthetic)", "Research Specification", "Research Baseline", "Compare all"],
        horizontal=True,
        help="Copula: Generate synthetic agents via fitted copula\nResearch Specification: Use original participants with stochastic draws\nResearch Baseline: Use original participants with NO stochastic component (anchor values only)\nCompare all: Show all three modes side-by-side",
        key="page1_population_mode"
    )
    
    # Sync the widget's state to the main population_mode variable
    st.session_state.population_mode = st.session_state.page1_population_mode
    
    # Show description of selected mode
    if st.session_state.population_mode == "Copula (synthetic)":
        st.info("🧬 **Copula Mode**: Generates unlimited synthetic agents using fitted copula from 280 original participants. Preserves correlation structure.")
    elif st.session_state.population_mode == "Research Specification":
        st.info("📄 **Research Specification**: Uses original 280 participants with stochastic component (Normal draws). Follows research documentation methodology.")
    elif st.session_state.population_mode == "Research Baseline":
        st.info("⚖️ **Research Baseline**: Uses original 280 participants with NO stochastic component. Returns pure anchor values (deterministic).")
    else:  # Compare all
        st.info("🔬 **Compare All**: Runs all three population modes side-by-side for comprehensive comparison.")
    
    # Navigation
    render_navigation('page1')
