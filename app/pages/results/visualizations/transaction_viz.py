# app/pages/results/visualizations/transaction_viz.py
"""
Transaction and purchase-related visualization functions.
Handles purchase_vs_bid, rejected_transaction_defaults, and rejected_transaction_option decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter


def render_purchase_vs_bid(df, decision_name, decision_title, decision_data):
    """Visualization for purchase_vs_bid - per-request Purchase Now/bid choices"""
    
    # Show that decisions are now made PER REQUEST
    st.info("⚠️ **Note**: Decisions are made **per purchase request**, not per agent. A single agent can choose differently for each purchase.")
    
    # Analyze customer type distribution at agent level
    if 'customer_type' in df.columns:
        from src.decisions.income_utils import analyze_customer_types
        customer_stats = analyze_customer_types(df)
        
        # Show customer type breakdown
        type_col1, type_col2, type_col3, type_col4 = st.columns(4)
        
        with type_col1:
            st.metric("Total Agents", f"{customer_stats['total']:,}")
        with type_col2:
            st.metric("Regular Customers", 
                     f"{customer_stats['regular']['count']:,}",
                     f"{customer_stats['regular']['percentage']:.1f}%",
                     help="Only these customers make Purchase Now vs Bid choice")
        with type_col3:
            st.metric("Fixed Customers", 
                     f"{customer_stats['fixed']['count']:,}",
                     f"{customer_stats['fixed']['percentage']:.1f}%",
                     help="Use fixed pricing only (NA)")
        with type_col4:
            st.metric("Discount Customers", 
                     f"{customer_stats['discount']['count']:,}",
                     f"{customer_stats['discount']['percentage']:.1f}%",
                     help="Use discount pricing (NA)")
    
    # Extract REQUEST-LEVEL data from purchase_requests
    st.markdown("---")
    st.markdown("### 🛒 Purchase Decisions per Request")
    
    if 'purchase_requests' in df.columns:
        # Collect all purchase decisions from all requests
        all_platform_prices = []
        regular_requests = []
        
        for idx, row in df.iterrows():
            requests = row.get('purchase_requests', [])
            if isinstance(requests, list):
                for req in requests:
                    if isinstance(req, dict):
                        platform_price = req.get('platformPrice')
                        all_platform_prices.append(platform_price)
                        
                        # Count only PN and BID for regular customers
                        if platform_price in ['PN', 'BID']:
                            regular_requests.append(platform_price)
        
        # Count platform prices
        all_counts = Counter(all_platform_prices)
        regular_counts = Counter(regular_requests)
        
        total_requests = len(all_platform_prices)
        total_regular_requests = len(regular_requests)
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", f"{total_requests:,}", help="All purchase requests across all agents")
        with col2:
            discount_count = all_counts.get('DISCOUNT', 0)
            st.metric("Discount Requests", f"{discount_count:,} ({discount_count/total_requests*100 if total_requests > 0 else 0:.1f}%)")
        with col3:
            fixed_count = all_counts.get('FIXED', 0)
            st.metric("Fixed Requests", f"{fixed_count:,} ({fixed_count/total_requests*100 if total_requests > 0 else 0:.1f}%)")
        with col4:
            st.metric("Regular Requests", f"{total_regular_requests:,} ({total_regular_requests/total_requests*100 if total_requests > 0 else 0:.1f}%)")
        
        # Regular customer requests (PN vs BID)
        st.markdown("---")
        st.markdown("### 🎯 Regular Customer Requests: Purchase Now vs Bid")
        
        if total_regular_requests > 0:
            pn_count = regular_counts.get('PN', 0)
            bid_count = regular_counts.get('BID', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Regular Requests", f"{total_regular_requests:,}", 
                         help="Purchase requests from regular customers only")
            with col2:
                st.metric("Purchase Now (PN)", f"{pn_count:,} ({pn_count/total_regular_requests*100:.1f}%)")
            with col3:
                st.metric("Bid (BID)", f"{bid_count:,} ({bid_count/total_regular_requests*100:.1f}%)")
            with col4:
                st.metric("Purchase Now Rate", f"{pn_count/total_regular_requests*100:.1f}%")
            
            # Purchase Now vs Bid visualization - donut chart
            col_plot, col_stats = st.columns([2, 1])
            
            with col_plot:
                fig = px.pie(
                    values=[pn_count, bid_count],
                    names=['Purchase Now (PN)', 'Bid (BID)'],
                    title=f"Purchase Decisions Distribution ({total_regular_requests:,} requests)",
                    hole=0.4,  # Donut chart
                    color_discrete_map={
                        'Purchase Now (PN)': '#4CAF50',  # Green
                        'Bid (BID)': '#FF9800'  # Orange
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_stats:
                st.markdown("**🛒 Request-Level Choices**")
                st.caption("(Regular customers only)")
                breakdown_df = pd.DataFrame({
                    'Choice': ['Purchase Now (PN)', 'Bid (BID)'],
                    'Requests': [pn_count, bid_count],
                    'Percentage': [
                        f"{pn_count/total_regular_requests*100:.1f}%",
                        f"{bid_count/total_regular_requests*100:.1f}%"
                    ]
                })
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        else:
            st.info("No regular customer purchase requests found")
    else:
        st.warning("No purchase_requests data available")


def render_rejected_transaction_defaults(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_transaction_defaults with interactive radio buttons"""
    
    # Define the 5 options
    options = [
        ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
        ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
        ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
        ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
        ("forgo_transaction", "Option 5: Forgo the purchase request")
    ]
    
    option_names = dict(options)
    
    # Check the actual simulation execution mode from session state
    simulation_mode = "unknown"
    if hasattr(st.session_state, 'sim_params') and hasattr(st.session_state.sim_params, 'simulation_execution_mode'):
        simulation_mode = st.session_state.sim_params.simulation_execution_mode
    
    # Get current default from results or session state
    value_counts = decision_data.value_counts()
    current_default = value_counts.index[0] if len(value_counts) > 0 else "forgo_transaction"
    
    # Use _default_selection key (same as Page 2 Overview tab) for consistency
    radio_key = f"{decision_name}_default_selection"
    if radio_key not in st.session_state:
        # Initialize with what was actually used in the simulation
        st.session_state[radio_key] = current_default
    
    # Top section: Current results display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        st.metric("Simulation Mode", simulation_mode.title())
    
    with col3:
        # Show what's configured for NEXT simulation (from session state)
        configured_option = st.session_state.get(radio_key, current_default)
        display_name = option_names.get(configured_option, configured_option)
        st.metric("Configured Default", display_name.split(":")[0])  # Just "Option X"
    
    with col4:
        # Show consistency of CURRENT results
        if len(value_counts) > 0:
            percentage = (value_counts.iloc[0] / len(decision_data)) * 100
            st.metric("Results Consistency", f"{percentage:.1f}%")
        else:
            st.metric("Results Consistency", "N/A")
    
    # Main configuration section - READ-ONLY DISPLAY
    st.markdown("---")
    st.markdown("**⚙️ Default Behavior Configuration (Read-Only):**")
    
    col_radio, col_viz = st.columns([1, 1])
    
    with col_radio:
        # Get current selection from session state
        current_selection = st.session_state.get(radio_key, current_default)
        
        # Display selected option as read-only
        st.success(f"✅ **Selected Default:**\n\n{option_names.get(current_selection, current_selection)}")
        
        # Show description based on simulation mode
        if simulation_mode == "snapshot":
            st.caption(f"**Snapshot Mode**: All agents use this option when transactions are rejected")
        elif simulation_mode == "live":
            st.caption(f"**Live Mode**: This is the fallback default (agents asked in real-time)")
        
        st.caption("💡 To modify this setting: Go to **Page 2 → Overview Tab**")
    
    with col_viz:
        # Show current results visualization
        if len(value_counts) > 0:
            # Create readable labels for the chart
            readable_labels = [option_names.get(opt, opt) for opt in value_counts.index]
            
            fig = px.pie(
                values=value_counts.values,
                names=readable_labels,
                title="Current Simulation Results",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True, key="rejected_transaction_defaults_chart")
        else:
            st.info("No simulation data available")


def render_rejected_transaction_option(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_transaction_option with interactive radio buttons"""
    
    # Define the 5 options
    options = [
        ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
        ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
        ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
        ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
        ("forgo_transaction", "Option 5: Forgo the purchase request")
    ]
    
    option_names = dict(options)
    
    # Get current option from results or session state
    value_counts = decision_data.value_counts()
    current_option = value_counts.index[0] if len(value_counts) > 0 else "forgo_transaction"
    
    # Use _default_selection key (same as Page 2 Overview tab) for consistency
    radio_key = f"{decision_name}_default_selection"
    if radio_key not in st.session_state:
        # Initialize with what was actually used in the simulation
        st.session_state[radio_key] = current_option
    
    # Top section: Current results display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        # Show what's configured for NEXT simulation (from session state)
        configured_option = st.session_state.get(radio_key, current_option)
        display_name = option_names.get(configured_option, configured_option)
        st.metric("Configured Option", display_name.split(":")[0])  # Just "Option X"
    
    with col3:
        # Show most common from CURRENT results
        most_common_option = option_names.get(current_option, current_option)
        st.metric("Most Common Result", most_common_option.split(":")[0])
    
    with col4:
        if len(value_counts) > 0:
            percentage = (value_counts.iloc[0] / len(decision_data)) * 100
            st.metric("Result Frequency", f"{percentage:.1f}%")
    
    # Main configuration section - READ-ONLY DISPLAY
    st.markdown("---")
    st.markdown("**⚙️ Specific Option Configuration (Read-Only):**")
    
    col_radio, col_viz = st.columns([1, 1])
    
    with col_radio:
        # Get current selection from session state
        current_selection = st.session_state.get(radio_key, current_option)
        
        # Display selected option as read-only
        st.success(f"✅ **Selected Option:**\n\n{option_names.get(current_selection, current_selection)}")
        
        st.caption("💡 To modify this setting: Go to **Page 2 → Overview Tab**")
    
    with col_viz:
        # Show current results visualization
        if len(value_counts) > 0:
            # Create readable labels for the chart
            readable_labels = [option_names.get(opt, opt) for opt in value_counts.index]
            
            fig = px.pie(
                values=value_counts.values,
                names=readable_labels,
                title="Current Simulation Results",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True, key="rejected_transaction_option_chart")
        else:
            st.info("No simulation data available")


def render_rejected_bid_value(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_bid_value"""
    st.markdown("**Not relevant given choice of Option 5**")

