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
        
        # Create sub-columns for Simulation Settings and Time Parameters
        col1_left, col1_right = st.columns(2)
        
        with col1_left:
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
        
        with col1_right:
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
        
        # Vendor Configuration - Single Source of Truth
        st.markdown('<h3 class="section-header">🏪 Vendor Configuration</h3>', unsafe_allow_html=True)
        st.caption("Configure all vendor settings: number, prices, products, and carryover behavior")
        
        # Add reset to defaults button

        
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
                    min_value=0.01,
                    max_value=1000.0,
                    value=st.session_state.sim_params.market_price,
                    step=1.0,
                    help="Price for the single vendor",
                    key="single_vendor_price_input",
                    format="%.2f",
                    on_change=lambda: setattr(st.session_state.sim_params, 'market_price', st.session_state.single_vendor_price_input)
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
                "Enable Product Carryover to Next Period",
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
                # Show visual indicator for active mode
                st.info("🎲 **Random Generation Mode**: Configure parameters below to randomly generate vendor properties")
                
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
                        step=1.0,
                        help="Minimum price any vendor can have",
                        key="vendor_price_min_input",
                        format="%.2f",
                        on_change=lambda: setattr(st.session_state.sim_params, 'vendor_price_min', st.session_state.vendor_price_min_input)
                    )
                    
                    # Ensure vendor_price_max is at least vendor_price_min before creating the widget
                    min_for_max = st.session_state.sim_params.vendor_price_min
                    current_max = st.session_state.sim_params.vendor_price_max
                    
                    # Auto-adjust vendor_price_max if it's below vendor_price_min
                    if current_max < min_for_max:
                        st.session_state.sim_params.vendor_price_max = st.session_state.sim_params.vendor_price_min
                        current_max = min_for_max
                        st.warning(f"⚠️ Auto-adjusted: Max price was set to ${st.session_state.sim_params.vendor_price_min:.2f} (cannot be below min price)")
                    
                    vendor_price_max = st.number_input(
                        "Max Price per Vendor ($)",
                        min_value=min_for_max,
                        max_value=1000.0,
                        value=current_max,
                        step=1.0,
                        help="Maximum price any vendor can have",
                        key="vendor_price_max_input",
                        format="%.2f",
                        on_change=lambda: setattr(st.session_state.sim_params, 'vendor_price_max', st.session_state.vendor_price_max_input)
                    )
                    
                    # Ensure market_price is within bounds before creating the widget
                    min_price_dollars = st.session_state.sim_params.vendor_price_min
                    max_price_dollars = st.session_state.sim_params.vendor_price_max
                    current_price_dollars = st.session_state.sim_params.market_price
                    
                    # Auto-adjust market_price if it's outside the new bounds
                    if current_price_dollars < min_price_dollars:
                        st.session_state.sim_params.market_price = st.session_state.sim_params.vendor_price_min
                        current_price_dollars = min_price_dollars
                        st.warning(f"⚠️ Auto-adjusted: Average price was set to ${st.session_state.sim_params.vendor_price_min:.2f} (cannot be below min price)")
                    elif current_price_dollars > max_price_dollars:
                        st.session_state.sim_params.market_price = st.session_state.sim_params.vendor_price_max
                        current_price_dollars = max_price_dollars
                        st.warning(f"⚠️ Auto-adjusted: Average price was set to ${st.session_state.sim_params.vendor_price_max:.2f} (cannot be above max price)")
                    
                    market_price = st.number_input(
                        "Average Price per Vendor ($)",
                        min_value=min_price_dollars,
                        max_value=max_price_dollars,
                        value=current_price_dollars,
                        step=1.0,
                        help="Target average price across all vendors",
                        key="market_price_input",
                        format="%.2f",
                        on_change=lambda: setattr(st.session_state.sim_params, 'market_price', st.session_state.market_price_input)
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
                    
                    # Ensure vendor_products_max is at least vendor_products_min before creating the widget
                    min_for_max_products = st.session_state.sim_params.vendor_products_min
                    current_max_products = st.session_state.sim_params.vendor_products_max
                    
                    # Auto-adjust vendor_products_max if it's below vendor_products_min
                    if current_max_products < min_for_max_products:
                        st.session_state.sim_params.vendor_products_max = st.session_state.sim_params.vendor_products_min
                        current_max_products = min_for_max_products
                        st.warning(f"⚠️ Auto-adjusted: Max products was set to {st.session_state.sim_params.vendor_products_min} (cannot be below min products)")
                    
                    vendor_products_max = st.number_input(
                    "Max Products per Vendor",
                    min_value=min_for_max_products,
                    max_value=10000,
                    value=current_max_products,
                    help="Maximum products any vendor can offer per period",
                    key="vendor_products_max_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'vendor_products_max', st.session_state.vendor_products_max_input)
                )
                    
                    # Ensure vendor_products_avg is within bounds before creating the widget
                    min_products = st.session_state.sim_params.vendor_products_min
                    max_products = st.session_state.sim_params.vendor_products_max
                    current_avg_products = st.session_state.sim_params.vendor_products_avg
                    
                    # Auto-adjust vendor_products_avg if it's outside the new bounds
                    if current_avg_products < min_products:
                        st.session_state.sim_params.vendor_products_avg = st.session_state.sim_params.vendor_products_min
                        current_avg_products = min_products
                        st.warning(f"⚠️ Auto-adjusted: Average products was set to {st.session_state.sim_params.vendor_products_min} (cannot be below min products)")
                    elif current_avg_products > max_products:
                        st.session_state.sim_params.vendor_products_avg = st.session_state.sim_params.vendor_products_max
                        current_avg_products = max_products
                        st.warning(f"⚠️ Auto-adjusted: Average products was set to {st.session_state.sim_params.vendor_products_max} (cannot be above max products)")
                    
                    vendor_products_avg = st.number_input(
                    "Average Products per Vendor",
                    min_value=min_products,
                    max_value=max_products,
                    value=current_avg_products,
                    help="Target average products per vendor",
                    key="vendor_products_avg_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'vendor_products_avg', st.session_state.vendor_products_avg_input)
                )
                    
                    # Products validation
                    products_total_min = st.session_state.sim_params.num_vendors * st.session_state.sim_params.vendor_products_min
                    products_total_max = st.session_state.sim_params.num_vendors * st.session_state.sim_params.vendor_products_max
                    products_total_avg = st.session_state.sim_params.num_vendors * st.session_state.sim_params.vendor_products_avg
                    products_valid = products_total_min <= products_total_avg <= products_total_max
            
                # Carryover Configuration (full width) - ONLY SHOWN IN RANDOM MODE
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
                        step=0.01,
                        help="Probability that any given vendor will have carryover enabled (Bernoulli per vendor)",
                        key="vendor_carryover_probability_slider",
                        on_change=lambda: setattr(st.session_state.sim_params, 'vendor_carryover_probability', st.session_state.vendor_carryover_probability_slider)
                    )
                    expected_carryover_vendors = round(st.session_state.sim_params.num_vendors * st.session_state.sim_params.vendor_carryover_probability)
                    st.info(f"🎲 Expected ~{expected_carryover_vendors} vendors (out of {st.session_state.sim_params.num_vendors}) to have carryover based on {st.session_state.sim_params.vendor_carryover_probability:.0%} probability")
                
            else:  # Upload Vendor Config File Mode
                
                # Show visual indicator for upload mode
                st.warning("📁 **Upload Mode Active**: Random generation settings are disabled. Vendor configuration will be read from CSV file.")
                
                # Enhanced guidance section
                st.info("📁 Upload a CSV file with complete vendor configuration. This will override all random generation settings including carryover options.")
                
                # Comprehensive format guidance
                with st.expander("📋 Expected CSV Format & Field Specifications", expanded=False):
                    st.markdown("**Required columns and data types:**")
                    
                    # Field specifications
                    col_spec1, col_spec2 = st.columns(2)
                    with col_spec1:
                        st.markdown("""
                        - **vendor_id** (string): Unique vendor identifier
                          - Examples: `V1`, `V2`, `PREMIUM_VENDOR`, `BUDGET_SHOP`
                          - Must be unique across all rows
                        
                        - **price** (float): Product price per unit in dollars
                          - Examples: `8.50`, `12.75`, `100.00`
                          - Must be positive (> 0.00)
                        """)
                    
                    with col_spec2:
                        st.markdown("""
                        - **products_per_period** (integer): Products offered per period
                          - Examples: `50`, `120`, `200`
                          - Must be positive integer (≥ 1)
                        
                        - **carryover** (integer): Inventory carryover enabled
                          - Examples: `0` (disabled), `1` (enabled)
                          - Must be exactly 0 or 1
                        """)
                    
                    st.markdown("**Sample CSV Format:**")
                    st.code("""vendor_id,price,products_per_period,carryover
V1,8.50,120,1
V2,9.25,95,0
V3,10.00,80,1
PREMIUM_VENDOR,15.75,200,1
BUDGET_SHOP,5.25,50,0""")
                    
                    st.markdown("**Validation Rules:**")
                    st.markdown("""
                    ✅ All columns must be present with exact names  
                    ✅ `vendor_id` values must be unique  
                    ✅ `price` must be positive numbers  
                    ✅ `products_per_period` must be positive integers  
                    ✅ `carryover` must be exactly 0 or 1  
                    ✅ File must contain at least 1 vendor  
                    ✅ No empty cells allowed
                    """)
                    

            
            uploaded_file = st.file_uploader(
                "Upload Vendor Configuration CSV",
                type=['csv'],
                help="CSV file with complete vendor configuration"
            )
            
            if uploaded_file is not None:
                try:
                    vendor_config_df = pd.read_csv(uploaded_file)
                    
                    # Comprehensive validation
                    validation_errors = []
                    
                    # Check required columns
                    required_columns = ['vendor_id', 'price', 'products_per_period', 'carryover']
                    missing_columns = [col for col in required_columns if col not in vendor_config_df.columns]
                    
                    if missing_columns:
                        validation_errors.append(f"Missing required columns: {', '.join(missing_columns)}")
                    else:
                        # Check for empty dataframe
                        if len(vendor_config_df) == 0:
                            validation_errors.append("CSV file is empty - no vendor data found")
                        else:
                            # Check for duplicate vendor IDs
                            if vendor_config_df['vendor_id'].duplicated().any():
                                duplicate_ids = vendor_config_df[vendor_config_df['vendor_id'].duplicated()]['vendor_id'].tolist()
                                validation_errors.append(f"Duplicate vendor_id values found: {', '.join(map(str, duplicate_ids))}")
                            
                            # Check for empty cells
                            if vendor_config_df.isnull().any().any():
                                null_info = []
                                for col in vendor_config_df.columns:
                                    null_count = vendor_config_df[col].isnull().sum()
                                    if null_count > 0:
                                        null_info.append(f"{col}: {null_count} empty cells")
                                validation_errors.append(f"Empty cells found - {', '.join(null_info)}")
                            
                            # Validate price column
                            try:
                                prices = pd.to_numeric(vendor_config_df['price'], errors='coerce')
                                if prices.isnull().any():
                                    validation_errors.append("price column contains non-numeric values")
                                elif (prices <= 0).any():
                                    invalid_prices = vendor_config_df[prices <= 0]['vendor_id'].tolist()
                                    validation_errors.append(f"price must be positive - invalid vendors: {', '.join(map(str, invalid_prices))}")
                            except:
                                validation_errors.append("price column validation failed")
                            
                            # Validate products_per_period column
                            try:
                                products = pd.to_numeric(vendor_config_df['products_per_period'], errors='coerce')
                                if products.isnull().any():
                                    validation_errors.append("products_per_period column contains non-numeric values")
                                elif (products <= 0).any():
                                    invalid_products = vendor_config_df[products <= 0]['vendor_id'].tolist()
                                    validation_errors.append(f"products_per_period must be positive - invalid vendors: {', '.join(map(str, invalid_products))}")
                                elif (products != products.astype(int)).any():
                                    fractional_products = vendor_config_df[products != products.astype(int)]['vendor_id'].tolist()
                                    validation_errors.append(f"products_per_period must be integers - fractional values found for vendors: {', '.join(map(str, fractional_products))}")
                            except:
                                validation_errors.append("products_per_period column validation failed")
                            
                            # Validate carryover column
                            try:
                                carryover = pd.to_numeric(vendor_config_df['carryover'], errors='coerce')
                                if carryover.isnull().any():
                                    validation_errors.append("carryover column contains non-numeric values")
                                elif not carryover.isin([0, 1]).all():
                                    invalid_carryover = vendor_config_df[~carryover.isin([0, 1])]['vendor_id'].tolist()
                                    validation_errors.append(f"carryover must be 0 or 1 - invalid vendors: {', '.join(map(str, invalid_carryover))}")
                            except:
                                validation_errors.append("carryover column validation failed")
                    
                    # Display validation results
                    if validation_errors:
                        st.error("❌ **Validation Failed** - Please fix the following issues:")
                        for i, error in enumerate(validation_errors, 1):
                            st.error(f"{i}. {error}")
                        
                        st.markdown("**Troubleshooting suggestions:**")
                        st.markdown("""
                        - Ensure file is saved as CSV format (.csv extension)
                        - Check column names match exactly: `vendor_id`, `price`, `products_per_period`, `carryover`
                        - Verify all cells have values (no empty cells)
                        - Ensure vendor_id values are unique
                        - Check that prices are positive numbers
                        - Ensure products_per_period are positive integers (no decimals)
                        - Verify carryover values are exactly 0 or 1
                        """)
                    else:
                        # Validation passed - process the data
                        # Convert to list of dictionaries and store
                        st.session_state.sim_params.vendor_config_data = vendor_config_df.to_dict('records')
                        
                        # Update num_vendors to match uploaded data
                        st.session_state.sim_params.num_vendors = len(vendor_config_df)
                        
                        # Show success message with carryover note
                        carryover_count = vendor_config_df['carryover'].sum()
                        st.success(f"✅ **Successfully loaded configuration for {len(vendor_config_df)} vendors** ({carryover_count} with carryover enabled)")
                        
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
                    st.error(f"❌ **Failed to read CSV file**: {str(e)}")
                    st.markdown("**Common file issues:**")
                    st.markdown("""
                    - File might not be a valid CSV format
                    - File might be corrupted or empty
                    - Special characters in vendor names might cause parsing errors
                    - File encoding issues (try saving as UTF-8)
                    """)
                    st.caption("💡 Try opening the file in a text editor to check for obvious formatting issues.")
    
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
            step=0.01,
            help="Price range for Purchase Now and Minimum Bid prices. Extended to 1.0 for simulation flexibility.",
            key="price_range_slider",
            on_change=lambda: setattr(st.session_state.sim_params, 'price_range', st.session_state.price_range_slider)
        )
        
        bidding_percentage = st.slider(
            "Bidding Percentage (bp)",
            min_value=0.0,
            max_value=1.0,  # Extended from 0.5 to 1.0 as requested
            value=st.session_state.sim_params.bidding_percentage,
            step=0.01,
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
        st.markdown('<h3 class="section-header">💵 Annual Income Distribution</h3>', unsafe_allow_html=True)
        
        # Initialize the widget key if it doesn't exist
        if "page1_income_distribution" not in st.session_state:
            st.session_state.page1_income_distribution = st.session_state.sim_params.income_distribution if st.session_state.sim_params.income_distribution in ["lognormal", "generalised_gamma", "dagum"] else "lognormal"
        
        income_distribution = st.selectbox(
            "Income Distribution Type",
            ["lognormal", "generalised_gamma", "dagum"],
            index=["lognormal", "generalised_gamma", "dagum"].index(st.session_state.page1_income_distribution),
            help="Distribution function for generating agent incomes",
            key="page1_income_distribution"
        )
        
        # Sync the widget's state to sim_params
        st.session_state.sim_params.income_distribution = st.session_state.page1_income_distribution
        
        # Distribution-specific parameters
        if st.session_state.page1_income_distribution == "lognormal":
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
                # Single interactive text field that handles None values gracefully
                def update_lognormal_max():
                    input_value = st.session_state.lognormal_max_text_input.strip()
                    # Check if user entered "None", "none", empty string, or similar
                    if input_value.lower() in ['none', 'no maximum', 'unlimited', ''] or input_value == '0':
                        st.session_state.sim_params.lognormal_max = None
                    else:
                        try:
                            # Try to convert to float
                            value = float(input_value.replace(',', '').replace('$', ''))
                            if value > 0:
                                st.session_state.sim_params.lognormal_max = value
                            else:
                                st.session_state.sim_params.lognormal_max = None
                        except ValueError:
                            # If conversion fails, keep the current value
                            pass
                
                # Display "None" when no maximum, otherwise show the formatted value
                display_value = "None" if st.session_state.sim_params.lognormal_max is None else f"{st.session_state.sim_params.lognormal_max:,.0f}"
                
                lognormal_max = st.text_input(
                        "b - Maximum Value ($)",
                    value=display_value,
                    help="Maximum income value for rejection sampling. Enter 'None' for no maximum limit, or a numeric value.",
                    key="lognormal_max_text_input",
                    on_change=update_lognormal_max
                )
            
            # Show income range info
            max_display = "∞" if st.session_state.sim_params.lognormal_max is None else f"${st.session_state.sim_params.lognormal_max:,.0f}"
            st.info(f"📊 Income Range: [${st.session_state.sim_params.lognormal_min:,.0f}, {max_display}]")
            
        elif st.session_state.page1_income_distribution == "generalised_gamma":
            st.markdown("**Generalised Gamma Distribution Parameters**")
            st.caption("X = a + Y, where Y ~ GeneralisedGamma(k, c, λ)")
            
            # Add helpful tip about parameter values
            with st.expander("💡 Parameter Guidelines"):
                st.markdown("""
                **Understanding k, c, and λ:**
                - k (shape 1): Controls tail thickness
                  - k < 1: Heavy tail behavior
                  - k = 1: Exponential-like
                  - k > 1: Lighter tail, more bell-shaped
                  - Typical range: 0.3 to 3.0
                - c (shape 2): Controls skewness and body shape
                  - c < 1: Right skewed
                  - c = 1: Special case (Gamma distribution)
                  - c > 1: More symmetric
                  - Typical range: 0.5 to 5.0
                - λ (scale): Sets the overall scale/spread
                  - Larger λ → wider distribution
                  - Roughly corresponds to median income
                - **Special cases:**
                  - k=1, c=1: Exponential distribution
                  - c→∞: Approaches Weibull distribution
                  - k=c: Approaches lognormal as both increase
                
                ---
                
                **📚 Key References:**
                
                - **McDonald, J. B., & Xu, Y. J. (1995).** A generalization of the beta distribution with applications. *Journal of Econometrics*, 66(1–2), 133–152.
                
                - **Bourguignon, F., Fournier, M., & Gurgand, M. (2007).** Selection bias corrections based on the multinomial logit model: Monte Carlo comparisons. *Journal of Economic Surveys*, 21(1), 174–205.
                """)
            
            # k, c, and lambda parameters
            col_k, col_c, col_lambda = st.columns(3)
            
            with col_k:
                gg_k = st.number_input(
                    "k - Shape 1 (tail)",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state.sim_params.gg_k,
                    step=0.1,
                    help="Shape parameter controlling tail thickness (0.3-3.0 typical)",
                    key="gg_k_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'gg_k', st.session_state.gg_k_input)
                )
            
            with col_c:
                gg_c = st.number_input(
                    "c - Shape 2 (skew)",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state.sim_params.gg_c,
                    step=0.1,
                    help="Shape parameter controlling skewness (0.5-5.0 typical)",
                    key="gg_c_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'gg_c', st.session_state.gg_c_input)
                )
            
            with col_lambda:
                gg_lambda = st.number_input(
                    "λ - Scale ($)",
                    min_value=100.0,
                    max_value=1000000.0,
                    value=st.session_state.sim_params.gg_lambda,
                    step=1000.0,
                    help="Scale parameter: sets overall income scale",
                    key="gg_lambda_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'gg_lambda', st.session_state.gg_lambda_input)
                )
            
            # Min and Max parameters
            col_min, col_max = st.columns(2)
            
            with col_min:
                gg_min = st.number_input(
                    "a - Minimum Value ($)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=st.session_state.sim_params.gg_min,
                    step=100.0,
                    help="Linear shift: all values will be at least this amount",
                    key="gg_min_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'gg_min', st.session_state.gg_min_input)
                )
            
            with col_max:
                # Single interactive text field that handles None values gracefully
                def update_gg_max():
                    input_value = st.session_state.gg_max_text_input.strip()
                    # Check if user entered "None", "none", empty string, or similar
                    if input_value.lower() in ['none', 'no maximum', 'unlimited', ''] or input_value == '0':
                        st.session_state.sim_params.gg_max = None
                    else:
                        try:
                            # Try to convert to float
                            value = float(input_value.replace(',', '').replace('$', ''))
                            if value > 0:
                                st.session_state.sim_params.gg_max = value
                            else:
                                st.session_state.sim_params.gg_max = None
                        except ValueError:
                            # If conversion fails, keep the current value
                            pass
                
                # Display "None" when no maximum, otherwise show the formatted value
                display_value = "None" if st.session_state.sim_params.gg_max is None else f"{st.session_state.sim_params.gg_max:,.0f}"
                
                gg_max = st.text_input(
                    "b - Maximum Value ($)",
                    value=display_value,
                    help="Maximum income value for rejection sampling. Enter 'None' for no maximum limit, or a numeric value.",
                    key="gg_max_text_input",
                    on_change=update_gg_max
                )
            
            # Show income range info
            max_display = "∞" if st.session_state.sim_params.gg_max is None else f"${st.session_state.sim_params.gg_max:,.0f}"
            st.info(f"📊 Income Range: [${st.session_state.sim_params.gg_min:,.0f}, {max_display}]")
            
        elif st.session_state.page1_income_distribution == "dagum":
            st.markdown("**Dagum (Type I) Distribution Parameters**")
            st.caption("X = min + Y, where Y ~ Dagum(a, p, b)")
            
            # Add helpful tip about parameter values
            with st.expander("💡 Parameter Guidelines"):
                st.markdown("""
                **Understanding a, p, and b:**
                - a (tail shape): Controls tail heaviness
                  - a > 1: Finite mean (required for income modeling)
                  - a > 2: Finite variance
                  - Typical range: 1.5 to 3.0 for income data
                  - Smaller a → heavier tail (more inequality)
                - p (body shape): Controls the distribution body
                  - p < 1: Very peaked near minimum
                  - p = 1: Moderate peakedness
                  - p > 1: More spread out, bell-shaped
                  - Typical range: 0.5 to 5.0
                - b (scale): Sets the median-like scale
                  - Roughly corresponds to median income
                  - Adjust based on your income range
                - **Inequality properties:**
                  - Lower a → higher inequality (heavier tail)
                  - Higher p → lower inequality (more concentrated)
                  - Dagum often fits income data better than Pareto
                
                ---
                
                **📚 Key References:**
                
                - **Pérez, C. G. (2011).** Using the Dagum model to explain changes in personal income distribution in Spain, 1995–2005. *Applied Economics*, 43(17), 2149–2157.
                
                - **Kleiber, C., & Kotz, S. (2003).** *Statistical Size Distributions in Economics and Actuarial Sciences.* Wiley Series in Probability and Statistics.
                """)
            
            # a, p, and b parameters
            col_a, col_p, col_b = st.columns(3)
            
            with col_a:
                dagum_a = st.number_input(
                    "a - Shape (tail)",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state.sim_params.dagum_a,
                    step=0.1,
                    help="Tail thickness: smaller values = heavier tail (>1 for finite mean)",
                    key="dagum_a_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'dagum_a', st.session_state.dagum_a_input)
                )
            
            with col_p:
                dagum_p = st.number_input(
                    "p - Shape (body)",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state.sim_params.dagum_p,
                    step=0.1,
                    help="Body shape: controls concentration around median",
                    key="dagum_p_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'dagum_p', st.session_state.dagum_p_input)
                )
            
            with col_b:
                dagum_b = st.number_input(
                    "b - Scale ($)",
                    min_value=100.0,
                    max_value=1000000.0,
                    value=st.session_state.sim_params.dagum_b,
                    step=1000.0,
                    help="Scale parameter: sets median income level",
                    key="dagum_b_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'dagum_b', st.session_state.dagum_b_input)
                )
            
            # Min and Max parameters
            col_min, col_max = st.columns(2)
            
            with col_min:
                dagum_min = st.number_input(
                    "Minimum Value ($)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=st.session_state.sim_params.dagum_min,
                    step=100.0,
                    help="Linear shift: all values will be at least this amount",
                    key="dagum_min_input",
                    on_change=lambda: setattr(st.session_state.sim_params, 'dagum_min', st.session_state.dagum_min_input)
                )
            
            with col_max:
                # Single interactive text field that handles None values gracefully
                def update_dagum_max():
                    input_value = st.session_state.dagum_max_text_input.strip()
                    # Check if user entered "None", "none", empty string, or similar
                    if input_value.lower() in ['none', 'no maximum', 'unlimited', ''] or input_value == '0':
                        st.session_state.sim_params.dagum_max = None
                    else:
                        try:
                            # Try to convert to float
                            value = float(input_value.replace(',', '').replace('$', ''))
                            if value > 0:
                                st.session_state.sim_params.dagum_max = value
                            else:
                                st.session_state.sim_params.dagum_max = None
                        except ValueError:
                            # If conversion fails, keep the current value
                            pass
                
                # Display "None" when no maximum, otherwise show the formatted value
                display_value = "None" if st.session_state.sim_params.dagum_max is None else f"{st.session_state.sim_params.dagum_max:,.0f}"
                
                dagum_max = st.text_input(
                    "Maximum Value ($)",
                    value=display_value,
                    help="Maximum income value for rejection sampling. Enter 'None' for no maximum limit, or a numeric value.",
                    key="dagum_max_text_input",
                    on_change=update_dagum_max
                )
            
            # Show income range info
            max_display = "∞" if st.session_state.sim_params.dagum_max is None else f"${st.session_state.sim_params.dagum_max:,.0f}"
            st.info(f"📊 Income Range: [${st.session_state.sim_params.dagum_min:,.0f}, {max_display}]")
        
        # Discount Threshold Configuration
        st.markdown("### 💰 Discount Threshold")
        
        # Calculate dynamic min/max based on income distribution
        income_min = st.session_state.sim_params.income_min
        income_max = st.session_state.sim_params.income_max
        
        # For distributions without explicit min/max, use reasonable defaults
        if st.session_state.page1_income_distribution == "lognormal":
            income_min = st.session_state.sim_params.lognormal_min
            income_max = st.session_state.sim_params.lognormal_max if st.session_state.sim_params.lognormal_max else income_min + 100000
        elif st.session_state.page1_income_distribution == "generalised_gamma":
            income_min = st.session_state.sim_params.gg_min
            income_max = st.session_state.sim_params.gg_max if st.session_state.sim_params.gg_max else income_min + 200000
        elif st.session_state.page1_income_distribution == "dagum":
            income_min = st.session_state.sim_params.dagum_min
            income_max = st.session_state.sim_params.dagum_max if st.session_state.sim_params.dagum_max else income_min + 150000
        
        # Ensure discount threshold is within bounds before creating the widget
        current_threshold = st.session_state.sim_params.discount_income_threshold
        
        # Auto-adjust discount threshold if it's outside the income distribution bounds
        if current_threshold < income_min:
            st.session_state.sim_params.discount_income_threshold = income_min
            current_threshold = income_min
            st.warning(f"⚠️ Auto-adjusted: Discount threshold was set to ${income_min:,.0f} (cannot be below income minimum)")
        elif current_threshold > income_max:
            st.session_state.sim_params.discount_income_threshold = income_max
            current_threshold = income_max
            st.warning(f"⚠️ Auto-adjusted: Discount threshold was set to ${income_max:,.0f} (cannot be above income maximum)")
        
        discount_income_threshold = st.number_input(
            "Threshold Income for Discount ($)",
            min_value=income_min,
            max_value=income_max,
            value=current_threshold,
            step=100.0,
            help="Income threshold below which agents qualify for discounts (pending document disclosure)",
            key="discount_threshold_input",
            on_change=lambda: setattr(st.session_state.sim_params, 'discount_income_threshold', st.session_state.discount_threshold_input)
        )
        
        # Show threshold validation and info
        if income_min <= discount_income_threshold <= income_max:
            threshold_pct = ((discount_income_threshold - income_min) / (income_max - income_min)) * 100
            st.caption(f"✅ Threshold at {threshold_pct:.1f}% of income range")
        else:
            st.error("❌ Threshold must be between minimum and maximum income!")
        
        # Always show distribution preview
        st.markdown("### 📊 Distribution Preview")
        show_income_distribution_histogram(st.session_state.sim_params)
        
        # Income Categories Section
        st.markdown('<h3 class="section-header">📊 Income Categories</h3>', unsafe_allow_html=True)
        st.caption("Categories determine customer status (discount/fixed) and consumption limits")
        
        # Use narrower columns with gaps to keep buttons close while maintaining same level
        col_cat1, col_gap, col_cat2 = st.columns([1.2, 0.6, 1.2])
        with col_cat1:
            num_discount_categories = st.number_input(
                "Discount Income Categories (NDIC)",
                min_value=1,
                max_value=10,
                value=st.session_state.sim_params.num_discount_categories,
                help="Number of customer discount income categories (lowest income levels)",
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
                help=f"Number of customer fixed income categories (higher income levels). Used for consumption limits. Default: Price Grid - 1 = {default_nfic}",
                key="num_fixed_categories_input",
                on_change=update_nfic
            )
    
    # Create columns for Consumption Limits and Population Mode side by side
    col_limits, col_population = st.columns(2)
    
    with col_limits:
        # Consumption Limits Configuration
        st.markdown('<h3 class="section-header">🛒 Consumption Limits</h3>', unsafe_allow_html=True)
        st.caption("📊 Limits by **Fixed Income Categories** (Cat 1 = lowest income, applies to discount customers)")

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
            # Calculate term duration for display
            term_periods = st.session_state.sim_params.periods
            term_hours = st.session_state.sim_params.periods * st.session_state.sim_params.duration_hours
            
            st.caption(f"Set consumption limits per product for each **fixed income category** **per term**")
            st.info(f"📅 **Term Definition**: Number of Periods × Length of Period = {term_periods} period(s) × {int(st.session_state.sim_params.duration_hours)}h = {term_hours}h total")
            st.caption("💡 **Income Order**: Category 1 = Lowest Income (discount customers) → Higher Categories = Higher Income")

            # Create a simple interface for setting consumption limits
            total_categories = st.session_state.sim_params.num_fixed_categories
            
            st.markdown("**Fixed Income Categories** (ordered from lowest to highest income)")

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
                    
                    # Label for first category (discount customers)
                    if i == 0:
                        label = f"Cat 1 "
                    else:
                        label = f"Cat {i+1}"

                    # Initialize widget key if it doesn't exist
                    if key not in st.session_state:
                        st.session_state[key] = st.session_state.sim_params.consumption_limits.get(cat_key, 10)

                    limit = st.number_input(
                        label,
                        min_value=0,
                        max_value=100,
                        value=st.session_state[key],
                        key=key,
                        help=f"Max consumption for fixed income category {i+1} over entire term ({term_hours}h total). Cat 1 = lowest income (discount customers).",
                        on_change=lambda k=key, c=cat_key: st.session_state.consumption_limits_temp.update({c: st.session_state[k]})
                    )
                    consumption_limits[cat_key] = st.session_state[key]

            st.session_state.sim_params.consumption_limits = consumption_limits

        else:
            st.info("ℹ️ Consumption limits are disabled. All agents will use a single maximum purchase amount.")
            # Clear consumption limits when disabled
            st.session_state.sim_params.consumption_limits = {}
            
            # Calculate term duration for display
            term_periods = st.session_state.sim_params.periods
            term_hours = st.session_state.sim_params.periods * st.session_state.sim_params.duration_hours
            
            # Show fallback maximum input (professor's requirement)
            st.caption(f"🛒 Set the **maximum purchase amount** for the entire term ({term_periods} period(s) × {int(st.session_state.sim_params.duration_hours)}h = {term_hours}h total)")
            
            max_purchases_per_term = st.number_input(
                "Maximum Purchases per Term",
                min_value=0,
                max_value=1000,
                value=st.session_state.sim_params.max_purchases_per_term,
                help=f"Maximum number of items any agent can purchase during the entire term ({term_hours}h total). This applies to ALL agents when category-specific limits are disabled.",
                key="max_purchases_per_term_input"
            )
            st.session_state.sim_params.max_purchases_per_term = max_purchases_per_term
    
    with col_population:
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
