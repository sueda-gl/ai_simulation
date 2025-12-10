# app/pages/results/visualizations/transaction_viz.py
"""
Transaction and purchase-related visualization functions.
Handles purchase_vs_bid, rejected_transaction_defaults, and rejected_transaction_option decisions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter


def _build_purchase_vs_bid_export(df):
    """
    Build transaction-level export data for regular customers showing purchase vs bid decisions.
    
    Returns a list of transaction records with fields:
    - Agent ID
    - Assigned Allowance Level
    - Group_experiment
    - Customer Type (Regular, Fixed, Discount)
    - Income Category
    - Purchase Request Type (PN/Bid)
    - timestamp (DD/MM/YYYY HH:MM format)
    - Period
    - Customer Price (based on PN price or bid value, using vendor's actual price)
    
    Records are sorted by timestamp in chronological order.
    """
    from datetime import datetime, timedelta
    
    transaction_records = []
    
    if 'purchase_requests' not in df.columns:
        return transaction_records
    
    # Get pricing parameters from session state or use defaults
    platform_markup = 0.1
    price_range = 0.25
    if hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        platform_markup = sim_params.get('platform_markup', 0.1)
        price_range = sim_params.get('price_range', 0.25)
    elif hasattr(st.session_state, 'sim_params'):
        platform_markup = getattr(st.session_state.sim_params, 'platform_markup', 0.1)
        price_range = getattr(st.session_state.sim_params, 'price_range', 0.25)
    
    # Get vendor data for price lookup
    vendors_data = None
    
    # Check session state locations in order of likelihood
    if hasattr(st.session_state, 'vendors'):
        vendors_data = st.session_state.vendors
    elif hasattr(st.session_state, 'vendors_data'):
        vendors_data = st.session_state.vendors_data
    elif hasattr(st.session_state, 'simulation_results') and isinstance(st.session_state.simulation_results, dict):
        vendors_data = st.session_state.simulation_results.get('vendors_data', None)
    
    # Build vendor lookup dictionary for quick access
    vendor_lookup = {}
    if vendors_data:
        for vendor in vendors_data:
            vendor_id = vendor.get('vendor_id')
            if vendor_id is not None:
                # Store with both int and string keys to ensure lookup works
                vendor_lookup[vendor_id] = vendor
                vendor_lookup[str(vendor_id)] = vendor
    
    # Base date for timestamp conversion (current date when simulation is run)
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for idx, row in df.iterrows():
        # Get agent information
        agent_id = row.get('agent_id', idx + 1)
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        group_experiment = row.get('Group_experiment', '')
        income_category = row.get('income_category', np.nan)
        
        # Get purchase requests
        purchase_requests = row.get('purchase_requests', [])
        if not isinstance(purchase_requests, list):
            continue
        
        # Process each purchase request
        for req_idx, request in enumerate(purchase_requests):
            if not isinstance(request, dict):
                continue
            
            # Get customer type from request
            customer_type = request.get('customer_type', request.get('customerType', 'regular'))
            if isinstance(customer_type, str):
                customer_type_display = customer_type.capitalize()
            else:
                customer_type_display = 'Regular'
            
            # Only include regular customers for this export
            if customer_type.lower() != 'regular':
                continue
            
            # Get timestamp and convert to Period and formatted timestamp
            timestamp_hours = request.get('timestamp_hours', np.nan)
            if not pd.isna(timestamp_hours):
                # Get periods and duration from session state or use defaults
                periods = 1
                duration_hours = 1.0
                if hasattr(st.session_state, 'simulation_params'):
                    sim_params = st.session_state.simulation_params.get('simulation', {})
                    periods = sim_params.get('periods', 1)
                    duration_hours = sim_params.get('duration_hours', 1.0)
                elif hasattr(st.session_state, 'sim_params'):
                    periods = getattr(st.session_state.sim_params, 'periods', 1)
                    duration_hours = getattr(st.session_state.sim_params, 'duration_hours', 1.0)
                
                # Calculate period (each period has duration_hours hours)
                period = int(timestamp_hours // duration_hours) + 1 if timestamp_hours >= 0 else 1
                
                # Convert timestamp_hours to datetime format
                timestamp_dt = base_date + timedelta(hours=float(timestamp_hours))
                timestamp_str = timestamp_dt.strftime('%d/%m/%Y %H:%M')
                sort_key = timestamp_hours  # Use numeric timestamp for sorting
            else:
                # Fallback if timestamp_hours not available
                timestamp_str = base_date.strftime('%d/%m/%Y %H:%M')
                period = request.get('period', 1)
                sort_key = 0.0
            
            # Determine Purchase Request Type and Customer Price
            platform_price = request.get('platformPrice', request.get('platform_price', ''))
            bid_value = request.get('bid_value', 'N/A')
            
            # Get vendor price for this request's vendor
            vendor_id = request.get('vendorID', request.get('vendor_id'))
            
            # Normalize vendor_id for lookup (handle float 1.0 -> int 1)
            lookup_key = vendor_id
            if isinstance(vendor_id, float) and vendor_id.is_integer():
                lookup_key = int(vendor_id)
                
            vendor_price = None
            if lookup_key is not None:
                # Try direct lookup first
                if lookup_key in vendor_lookup:
                    vendor_price = vendor_lookup[lookup_key].get('price')
                # Try string lookup if not found
                elif str(lookup_key) in vendor_lookup:
                    vendor_price = vendor_lookup[str(lookup_key)].get('price')
            
            # Get Transaction ID (pre-assigned by central system)
            transaction_id = request.get('transaction_id')
            
            # Calculate customer price based on vendor's actual price
            # Formula: Customer Price (PN) = (1 + price_range) × (1 + platform_markup) × vendor_price
            if vendor_price is not None:
                baseline_price = (1 + platform_markup) * vendor_price
                pn_price = (1 + price_range) * baseline_price
            else:
                # Fallback to market_price if vendor price not available
                market_price = 100.0
                if hasattr(st.session_state, 'simulation_params'):
                    sim_params = st.session_state.simulation_params.get('simulation', {})
                    market_price = sim_params.get('market_price', 100.0)
                elif hasattr(st.session_state, 'sim_params'):
                    market_price = getattr(st.session_state.sim_params, 'market_price', 100.0)
                baseline_price = (1 + platform_markup) * market_price
                pn_price = (1 + price_range) * baseline_price
            
            # Only include PN and BID for regular customers
            if platform_price == 'PN':
                purchase_request_type = 'PN'
                customer_price = pn_price  # PN uses calculated price based on vendor
            elif platform_price == 'BID' and bid_value != 'N/A':
                purchase_request_type = 'Bid'
                try:
                    customer_price = float(bid_value)
                except (ValueError, TypeError):
                    customer_price = pn_price
            else:
                # Skip if not PN or BID
                continue
            
            # Show price for both PN and BID customers
            # Format to 2 decimal places for display
            display_customer_price = float(f"{customer_price:.2f}")
            
            # Build record
            record = {
                'Transaction ID': transaction_id,
                'Agent ID': agent_id,
                'Assigned Allowance Level': allowance_level,
                'Group_experiment': group_experiment,
                'Customer Type': customer_type_display,
                'Income Category': income_category,
                'Purchase Request Type': purchase_request_type,
                'Vendor': vendor_id,
                'Vendor Price': vendor_price,
                'timestamp': timestamp_str,
                'Period': period,
                'Customer Price': display_customer_price,
                '_sort_key': sort_key  # Hidden column for sorting
            }
            
            transaction_records.append(record)
    
    # Sort all records by timestamp in chronological order
    if transaction_records:
        transaction_records.sort(key=lambda x: x.get('_sort_key', 0.0))
        
        # Remove the hidden sorting column before returning
        for record in transaction_records:
            record.pop('_sort_key', None)
            
            # If Transaction ID is missing (e.g. old simulation run), generate a placeholder or sequence
            # But since we're filtering, we can't easily regenerate global sequence here.
            # We trust the central system. If None, it will show as empty in Excel.
    
    return transaction_records


def render_purchase_vs_bid(df, decision_name, decision_title, decision_data):
    """Visualization for purchase_vs_bid - per-request Purchase Now/bid choices"""
    
    # Show that decisions are now made PER REQUEST
    st.info("⚠️ **Note**: Decisions are made **per purchase request**, not per agent. A single agent can choose differently for each purchase.")
    
    # Reference to Decision 2 for customer type definitions
    st.info("💡 **Customer Type Information**: This decision only applies to **Regular Customers**. For detailed customer type definitions and distribution, see **Decision 2: Disclose Documents**.")
    
    # Extract REQUEST-LEVEL data from purchase_requests
    st.markdown("---")
    st.markdown("### 🎯 Purchase Now vs Bid Decisions")
    st.caption("📊 Decisions for **Regular Customers only** - For full customer type breakdown, see **Decision 2: Disclose Documents**")
    
    if 'purchase_requests' in df.columns:
        # Collect all purchase decisions from all requests
        regular_requests = []
        
        for idx, row in df.iterrows():
            requests = row.get('purchase_requests', [])
            if isinstance(requests, list):
                for req in requests:
                    if isinstance(req, dict):
                        platform_price = req.get('platformPrice')
                        
                        # Count only PN and BID for regular customers
                        if platform_price in ['PN', 'BID']:
                            regular_requests.append(platform_price)
        
        # Count regular customer choices
        regular_counts = Counter(regular_requests)
        total_regular_requests = len(regular_requests)
        
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
                st.markdown(f"**Purchase Decisions Distribution ({total_regular_requests:,} requests)**")
                fig = px.pie(
                    values=[pn_count, bid_count],
                    names=['Purchase Now (PN)', 'Bid (BID)'],
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
        
        # Excel Export Section
        st.markdown("---")
        st.markdown("**📥 Export Purchase vs Bid Decision Data**")
        st.caption("Download detailed request-level data for all regular customers with pricing and transaction information")
        
        # Build transaction records
        transaction_records = _build_purchase_vs_bid_export(df)
        
        if transaction_records and len(transaction_records) > 0:
            try:
                from io import BytesIO
                from datetime import datetime
                
                # Create DataFrame (already sorted by the build function)
                export_df = pd.DataFrame(transaction_records)
                
                # Create Excel with multiple sheets
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    # Sheet 1: Total (all data)
                    export_df.to_excel(writer, index=False, sheet_name='Total')
                    
                    # Additional sheets by Period
                    if 'Period' in export_df.columns:
                        periods = sorted(export_df['Period'].dropna().unique())
                        for period in periods:
                            period_df = export_df[export_df['Period'] == period]
                            sheet_name = f'Period {int(period)}'
                            period_df.to_excel(writer, index=False, sheet_name=sheet_name)
                
                col_download, col_info = st.columns([1, 2])
                
                with col_download:
                    st.download_button(
                        label="📊 Download Purchase vs Bid Excel",
                        data=buffer.getvalue(),
                        file_name=f"purchase_vs_bid_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download request-level data for regular customers with purchase decisions"
                    )
                
                with col_info:
                    num_sheets = 1 + len(export_df['Period'].dropna().unique()) if 'Period' in export_df.columns else 1
                    st.caption(f"📋 Export includes {len(export_df):,} requests across {num_sheets} sheets")
                    st.caption(f"✅ Fields: Transaction ID, Agent ID, Allowance Level, Group, Customer Type, Income Category, Purchase Type, Vendor, Vendor Price, timestamp, Period, Customer Price")
                    st.caption(f"🔄 Sorted by: timestamp (chronological order)")
            
            except ImportError:
                st.warning("⚠️ Excel export requires openpyxl package")
            except Exception as e:
                st.error(f"❌ Error creating Excel file: {str(e)}")
        else:
            st.info("ℹ️ No purchase request data available for export")
    else:
        st.warning("No purchase_requests data available")


def render_rejected_transaction_defaults(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_transaction_defaults - prioritized options per agent"""
    
    # Define the 5 options
    options = [
        ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
        ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
        ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
        ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
        ("forgo_transaction", "Option 5: Forgo the purchase request")
    ]
    
    option_names = dict(options)
    option_numbers = {
        "higher_price_category": "Option 1",
        "lower_pn_vendor": "Option 2",
        "current_vendor_pn": "Option 3",
        "place_bid": "Option 4",
        "forgo_transaction": "Option 5"
    }
    
    # Check the actual simulation execution mode from session state
    simulation_mode = "unknown"
    if hasattr(st.session_state, 'sim_params') and hasattr(st.session_state.sim_params, 'simulation_execution_mode'):
        simulation_mode = st.session_state.sim_params.simulation_execution_mode
    
    # Analyze the prioritized lists from agent data
    st.info("ℹ️ **Note**: Each agent has a prioritized list of default options (1-5 options). The list shows their order of preference when transactions are rejected.")
    
    # Top section: Current results display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        st.metric("Simulation Mode", simulation_mode.title())
    
    with col3:
        # Count how many agents have lists vs single values
        list_count = decision_data.apply(lambda x: isinstance(x, list)).sum()
        st.metric("Agents with Priority Lists", f"{list_count:,}")
    
    # Show configured priority template
    st.markdown("---")
    st.markdown("**📊 Configured Priority Options**")
    
    # Get configured priority template for display
    priority_key = f"{decision_name}_priority_template"
    configured_template = st.session_state.get(priority_key, ["forgo_transaction"])
    
    col_template, col_chart = st.columns([1, 2])
    
    with col_template:
        st.markdown("**Priority Template:**")
        
        # Display configured priority list
        if isinstance(configured_template, list):
            for i, opt in enumerate(configured_template, 1):
                option_label = option_numbers.get(opt, opt)
                option_desc = option_names.get(opt, opt)
                st.markdown(f"**Priority {i}.** {option_label}")
                st.caption(f"   {option_desc}")
        else:
            st.caption("No priority template configured")
        
        st.caption(f"💡 {len(configured_template)} option(s) configured")
        st.caption("All agents use this priority list")
    
    with col_chart:
        # Count which options appear in agents' priority lists
        from collections import Counter
        
        total_agents = len(decision_data)
        
        # Count how many agents have each option in their priority list
        option_agent_counts = Counter()
        for agent_list in decision_data:
            if isinstance(agent_list, list):
                # Count unique options per agent (not duplicates)
                for opt in set(agent_list):
                    option_agent_counts[opt] += 1
            else:
                option_agent_counts[agent_list] += 1
        
        # Create individual charts for each option
        if len(option_agent_counts) > 0:
            # Sort by the order in configured_template if possible
            if isinstance(configured_template, list):
                sorted_options = [opt for opt in configured_template if opt in option_agent_counts]
            else:
                sorted_options = list(option_agent_counts.keys())
            
            st.markdown("**Options in Priority Lists**")
            st.caption("Each chart shows the percentage of agents that have this option in their priority list")
            
            # Create a row of small donut charts - one for each option
            if len(sorted_options) <= 3:
                chart_cols = st.columns(len(sorted_options))
            else:
                chart_cols = st.columns(3)
            
            # Color palette for consistency
            colors = px.colors.qualitative.Set3
            
            for idx, opt in enumerate(sorted_options):
                col_idx = idx % len(chart_cols)
                with chart_cols[col_idx]:
                    agent_count = option_agent_counts[opt]
                    percentage = (agent_count / total_agents) * 100
                    option_label = option_numbers.get(opt, opt)
                    
                    # Create individual donut chart for this option
                    fig = px.pie(
                        values=[percentage, 100 - percentage],
                        names=[option_label, ""],
                        hole=0.6,
                        color_discrete_sequence=[colors[idx % len(colors)], "#f0f0f0"]
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent',
                        hovertemplate=f'<b>{option_label}</b><br>{agent_count:,} agents<br>%{{percent}}<extra></extra>',
                        showlegend=False
                    )
                    # Add center text showing percentage
                    fig.update_layout(
                        showlegend=False,
                        height=200,
                        margin=dict(t=30, b=10, l=10, r=10),
                        annotations=[dict(
                            text=f'{percentage:.0f}%',
                            x=0.5, y=0.5,
                            font=dict(size=20, weight='bold'),
                            showarrow=False
                        )]
                    )
                    # Hide the empty slice from tooltip
                    fig.data[0].hoverinfo = 'skip'
                    
                    # Display title as markdown
                    st.markdown(f"**{option_label}**")
                    st.plotly_chart(fig, use_container_width=True, key=f"{decision_name}_option_{idx}_chart")
                    st.caption(f"{agent_count:,} agents")
            
            # Show detailed breakdown
            with st.expander("📋 Detailed Breakdown"):
                for opt in sorted_options:
                    count = option_agent_counts[opt]
                    percentage = (count / total_agents) * 100
                    st.caption(f"• {option_numbers.get(opt, opt)}: {count:,} agents ({percentage:.0f}%)")
    
    # Summary statistics
    st.markdown("---")
    st.markdown("**📈 Summary Statistics**")
    
    col_summary1, col_summary2 = st.columns(2)
    
    with col_summary1:
        st.markdown("**Priority List Lengths:**")
        list_lengths = decision_data.apply(lambda x: len(x) if isinstance(x, list) else 1)
        length_counts = list_lengths.value_counts().sort_index()
        
        for length, count in length_counts.items():
            percentage = (count / len(decision_data)) * 100
            st.caption(f"• {int(length)} option(s): {count:,} agents ({percentage:.1f}%)")
    
    with col_summary2:
        st.markdown("**Most Common 1st Choice:**")
        first_choices = decision_data.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
        first_choice_counts = first_choices.value_counts()
        
        for i, (choice, count) in enumerate(first_choice_counts.head(3).items(), 1):
            percentage = (count / len(decision_data)) * 100
            st.caption(f"{i}. {option_numbers.get(choice, choice)}: {count:,} agents ({percentage:.1f}%)")
    
    # Download section
    st.markdown("---")
    st.markdown("**📥 Download Priority Lists**")
    
    # Prepare export data
    export_df = _prepare_priority_lists_export(df, decision_data)
    
    if export_df is not None and not export_df.empty:
        # Create Excel file
        from io import BytesIO
        from datetime import datetime
        
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Priority Lists')
        
        st.download_button(
            label="📊 Download Priority Lists Excel",
            data=buffer.getvalue(),
            file_name=f"rejected_transaction_priorities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download detailed priority lists with Agent ID, Allowance Level, Group, and Priority 1-5 columns"
        )
        
        # Show preview
        with st.expander("📋 Preview Export Data (first 50 rows)"):
            st.dataframe(export_df.head(50), use_container_width=True)
            st.caption(f"Total rows: {len(export_df):,}")
    else:
        st.warning("Unable to prepare export data")


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
            
            st.markdown("### Current Simulation Results")
            fig = px.pie(
                values=value_counts.values,
                names=readable_labels,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True, key="rejected_transaction_option_chart")
        else:
            st.info("No simulation data available")


def _prepare_priority_lists_export(df: pd.DataFrame, decision_data) -> pd.DataFrame:
    """
    Prepare rejected transaction defaults priority lists for Excel export.
    
    Creates columns: Agent ID, Assigned Allowance Level, Group_experiment, 
    Priority 1, Priority 2, Priority 3, Priority 4, Priority 5
    
    Priority columns contain option numbers (1, 2, 3, 4, 5) or blank.
    
    Args:
        df: Full results DataFrame with agent data
        decision_data: Series containing priority lists
        
    Returns:
        DataFrame formatted for Excel export
    """
    # Map option codes to numbers
    option_to_number = {
        "higher_price_category": 1,
        "lower_pn_vendor": 2,
        "current_vendor_pn": 3,
        "place_bid": 4,
        "forgo_transaction": 5
    }
    
    # Create export dataframe
    export_df = pd.DataFrame()
    
    # Agent ID
    if 'agent_id' in df.columns:
        export_df['Agent ID'] = df['agent_id']
    elif 'index' in df.columns:
        export_df['Agent ID'] = df['index'] + 1  # Convert 0-based to 1-based
    else:
        export_df['Agent ID'] = range(1, len(df) + 1)
    
    # Assigned Allowance Level
    if 'Assigned Allowance Level' in df.columns:
        export_df['Assigned Allowance Level'] = df['Assigned Allowance Level']
    elif 'income_category' in df.columns:
        export_df['Assigned Allowance Level'] = df['income_category']
    else:
        export_df['Assigned Allowance Level'] = ''
    
    # Group_experiment
    if 'Group_experiment' in df.columns:
        export_df['Group_experiment'] = df['Group_experiment']
    elif 'group' in df.columns:
        export_df['Group_experiment'] = df['group']
    else:
        export_df['Group_experiment'] = ''
    
    # Priority columns (1-5)
    for priority_pos in range(1, 6):
        column_name = f'Priority {priority_pos}'
        priority_values = []
        
        for agent_list in decision_data:
            if isinstance(agent_list, list):
                # Check if agent has this priority position
                if len(agent_list) >= priority_pos:
                    option_code = agent_list[priority_pos - 1]
                    option_number = option_to_number.get(option_code, '')
                    priority_values.append(option_number)
                else:
                    # Agent doesn't have this many priorities
                    priority_values.append('')
            else:
                # Single value (legacy format) - only for priority 1
                if priority_pos == 1:
                    option_number = option_to_number.get(agent_list, '')
                    priority_values.append(option_number)
                else:
                    priority_values.append('')
        
        export_df[column_name] = priority_values
    
    return export_df


def render_rejected_bid_value(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_bid_value"""
    st.markdown("**Not relevant given choice of Option 5**")

