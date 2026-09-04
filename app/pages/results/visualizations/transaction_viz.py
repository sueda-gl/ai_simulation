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
from app.utils.timestamp_utils import TimestampConverter


def _apply_price_formatting_transaction(writer, sheet_name: str, df: pd.DataFrame):
    """
    Apply Excel number formatting to price-related columns to display 2 decimal places.
    """
    price_columns = [
        'Customer Price', 'customer_price', 'Bid Value', 'bid_value',
        'Final Donation Rate', 'final_donation_rate',
    ]
    
    workbook = writer.book
    worksheet = workbook[sheet_name]
    
    for col_idx, col_name in enumerate(df.columns, start=1):
        if col_name in price_columns:
            for row_idx in range(2, len(df) + 2):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                if isinstance(cell.value, (int, float)) and cell.value is not None:
                    cell.number_format = '0.00'


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
    
    # Use centralized timestamp converter for consistent handling
    ts_converter = TimestampConverter()
    
    for idx, row in df.iterrows():
        # Get agent information
        agent_id = row.get('agent_id', idx + 1)
        
        # Agent Traits (matching disclose income export)
        # Honesty_Humility
        honesty_humility = ''
        if 'Honesty_Humility' in row and pd.notna(row['Honesty_Humility']):
            honesty_humility = round(row['Honesty_Humility'], 2)
        
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        
        # Study Program
        study_program = row.get('Study Program', '')
        
        # Group_experiment (with fallbacks)
        group_experiment = ''
        if 'Group_experiment' in row and pd.notna(row['Group_experiment']):
            group_experiment = row['Group_experiment']
        elif 'group' in row and pd.notna(row['group']):
            group_experiment = row['group']
        elif 'group_experiment' in row and pd.notna(row['group_experiment']):
            group_experiment = row['group_experiment']
        
        # TWT+Sospeso
        twt_sospeso = ''
        if 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}' in row and pd.notna(row['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']):
            twt_sospeso = round(row['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'], 2)
        
        # Income
        income = ''
        if 'income' in row and pd.notna(row['income']):
            income = round(row['income'], 2)
        elif 'actual_allowance' in row and pd.notna(row['actual_allowance']):
            income = round(row['actual_allowance'], 2)
        
        # Income category - Regular customers don't have this (assigned in Decision 6 only for Discount/Fixed)
        income_category_raw = row.get('income_category', np.nan)
        # Use 'N/A' for empty/missing income_category (e.g., regular customers who didn't disclose income)
        if pd.isna(income_category_raw) or income_category_raw == '' or income_category_raw is None:
            income_category = 'N/A'
        else:
            income_category = income_category_raw
        
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
            
            # Get timestamp and convert using centralized utilities
            timestamp_hours = request.get('timestamp_hours', np.nan)
            ts_result = ts_converter.convert(timestamp_hours)
            
            period = ts_result['period']
            timestamp_str = ts_result['formatted']
            sort_key = ts_result['timestamp_hours'] if not pd.isna(ts_result['timestamp_hours']) else 0.0
            
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
                'Purchase Request ID': transaction_id,  # Placeholder, will be updated after sorting
                'Agent ID': agent_id,
                'Honesty_Humility': honesty_humility,
                'Assigned Allowance Level': allowance_level,
                'Study Program': study_program,
                'Group_experiment': group_experiment,
                'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': twt_sospeso,
                'income': income,
                'Customer Type': customer_type_display,
                'Income Category': income_category,
                'Purchase Request Type': purchase_request_type,
                'Vendor': vendor_id,
                'Vendor Price': vendor_price,
                'Purchase Timestamp': timestamp_str,
                'Period': period,
                'Customer Price': display_customer_price,
                '_sort_key': sort_key  # Hidden column for sorting
            }
            
            transaction_records.append(record)
    
    # Sort all records by timestamp in chronological order
    if transaction_records:
        transaction_records.sort(key=lambda x: x.get('_sort_key', 0.0))
        
        # Assign unique Purchase Request IDs based on sorted order
        for idx, record in enumerate(transaction_records):
            record['Purchase Request ID'] = idx + 1
            record.pop('_sort_key', None)
    
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
        st.markdown("**📥 Export Purchase Now vs Bid Decision Data**")
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
                    # Apply 2-decimal formatting
                    _apply_price_formatting_transaction(writer, 'Total', export_df)
                    
                    # Additional sheets by Period
                    if 'Period' in export_df.columns:
                        periods = sorted(export_df['Period'].dropna().unique())
                        for period in periods:
                            period_df = export_df[export_df['Period'] == period]
                            sheet_name = f'Period {int(period)}'
                            period_df.to_excel(writer, index=False, sheet_name=sheet_name)
                            # Apply 2-decimal formatting to each period sheet
                            _apply_price_formatting_transaction(writer, sheet_name, period_df)
                
                col_download, col_info = st.columns([1, 2])
                
                with col_download:
                    st.download_button(
                        label="📊 Download Purchase Now vs Bid Excel",
                        data=buffer.getvalue(),
                        file_name=f"purchase_now_vs_bid_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download request-level data for regular customers with purchase decisions"
                    )
                
                with col_info:
                    num_sheets = 1 + len(export_df['Period'].dropna().unique()) if 'Period' in export_df.columns else 1
                    st.caption(f"📋 Export includes {len(export_df):,} requests across {num_sheets} sheets")
                    st.caption(f"✅ Fields: Purchase Request ID, Agent ID, Honesty_Humility, Assigned Allowance Level, Study Program, Group_experiment, TWT+Sospeso, income, Customer Type, Income Category, Purchase Type, Vendor, Vendor Price, timestamp, Period, Customer Price")
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
    """Visualization for rejected_transaction_defaults - prioritized options per agent.

    Two display modes:
    - MODEL run (Decision 4 selected): the four trait-based sub-decision mechanisms
      (TTP list length + Loyalty/WTP/Risk-Taking rankings) and the Section-6 rank
      aggregation's integrated default list -> _render_rtd_model_results.
    - DEFAULT run (unselected): the legacy priority-template view below.
    """
    if 'rtd_choice_length' in df.columns:
        _render_rtd_model_results(df, decision_name, result_key=_rtd_result_key_for(df))
        return

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
        
        st.caption(f"💡 {len(configured_template)} options configured")
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
                    st.caption(f"{option_numbers.get(opt, opt)}: {count:,} agents ({percentage:.0f}%)")
    
    # Summary statistics
    st.markdown("---")
    st.markdown("**📈 Summary Statistics**")
    
    col_summary1, col_summary2 = st.columns(2)
    
    with col_summary1:
        st.markdown("**Priority List Lengths:**")
        list_lengths = decision_data.apply(lambda x: len(x) if isinstance(x, list) else 1)
        length_counts = list_lengths.value_counts().sort_index()
        
        breakdown_lines = [
            f"{int(length)} options: {count:,} agents ({(count / len(decision_data)) * 100:.1f}%)"
            for length, count in length_counts.items()]
        st.caption("  \n".join(breakdown_lines))
    
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
            # Apply 2-decimal formatting (for any numeric columns that may exist)
            _apply_price_formatting_transaction(writer, 'Priority Lists', export_df)
        
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


_RTD_OPTION_SHORT = {
    1: "Opt 1: higher price, same vendor",
    2: "Opt 2: other vendor, lower PN",
    3: "Opt 3: current vendor at PN",
    4: "Opt 4: place a bid",
    5: "Opt 5: forgo transaction",
}

# Priority sequences (most-likely option first) taken from the model at runtime -
# the same constants the decision function maps segments with (mirrored in
# config/decisions.yaml priority_sequences) - so chart orderings derived from
# them can never drift from the run's actual rankings.
try:
    from src.decisions.rejected_transaction_defaults import (
        PRIORITY_SEQUENCES as _RTD_PRIORITY_SEQUENCES)
except Exception:   # pragma: no cover - defensive fallback, values identical
    _RTD_PRIORITY_SEQUENCES = {'loyalty': [3, 1, 4, 5, 2], 'wtp': [3, 2, 1, 4, 5],
                               'risk_taking': [4, 2, 1, 3, 5], 'flexibility': [2, 4, 3, 1, 5]}

_RTD_MECHS = [
    ('loyalty', 'loyalty', 'Loyalty', _RTD_PRIORITY_SEQUENCES['loyalty']),
    ('wtp', 'wtp', 'Willingness-to-Pay', _RTD_PRIORITY_SEQUENCES['wtp']),
    ('risk_taking', 'rt', 'Risk-Taking', _RTD_PRIORITY_SEQUENCES['risk_taking']),
    ('flexibility', 'flex', 'Cognitive Flexibility', _RTD_PRIORITY_SEQUENCES['flexibility']),
]

# Per-element sheet / section names for the Decision 4 exports.
_RTD_ELEMENT_SHEETS = {
    'ttp': 'Options List Length',
    'loyalty': 'Loyalty',
    'wtp': 'Willingness-to-Pay',
    'risk_taking': 'Risk-Taking',
    'flexibility': 'Cognitive Flexibility',
}
# Section-6 rank aggregation (integrated default list) sheet / section name.
_RTD_AGG_SHEET = 'Integrated Default List'
_RTD_STAGE_LABELS = {
    'kemeny': 'Kemeny alone (unique full ranking)',
    'schulze': 'Schulze',
    'copeland': 'Copeland',
    'footrule': 'Footrule',
    'random': 'Last resort (random)',
}
_RTD_TRUNCATION_LABELS = {
    'none': 'Not truncated (full ranking kept)',
    'length': 'Options list length',
    'option5': 'Option 5 stop rule',
    'both': 'Both rules bind at the same position',
}

# Independent variables per element (Stata-aligned trait column names, in each
# element's equation order). WTP and Risk-Taking additionally get 'Assigned
# Allowance Level' when the frame was computed with categorical income.
_RTD_ELEMENT_INPUTS = {
    'ttp': ['ExtraversionBig5', 'Agreeable', 'NeuroticismBig5',
            'ConscientiousnessBig5', 'Education'],
    'loyalty': ['ExtraversionBig5', 'OpennessBig5', 'Agreeable'],
    'wtp': ['ExtraversionBig5', 'Agreeable', 'income'],
    'risk_taking': ['ExtraversionBig5', 'OpennessBig5', 'Agreeable',
                    'ConscientiousnessBig5', 'NeuroticismBig5', 'income'],
    # Cognitive Flexibility: Big 5 (IVW equation order) + the observed anchor stdactions
    'flexibility': ['ExtraversionBig5', 'OpennessBig5', 'NeuroticismBig5', 'Agreeable',
                    'ConscientiousnessBig5', 'stdactions'],
}


def _rtd_frame_income_mode(df):
    """Income specification the frame was computed with ('categorical'/'continuous')."""
    if 'rtd_income_mode' in df.columns and len(df) > 0:
        first = df['rtd_income_mode'].dropna()
        if len(first) > 0 and str(first.iloc[0]) == 'categorical':
            return 'categorical'
    return 'continuous'


def _rtd_active_element():
    """The element selected via a per-element Run button on the Decision 4 tab
    ('ttp' | 'loyalty' | 'wtp' | 'risk_taking' | 'flexibility'), or None when the
    whole decision was run. Only individual Decision 4 runs are filtered -
    combined/complete simulations always show all five elements."""
    if (getattr(st.session_state, 'custom_decisions', None) == ['rejected_transaction_defaults']
            and not getattr(st.session_state, 'default_decisions', [])):
        element = st.session_state.get('rtd_run_element')
        if element in _RTD_ELEMENT_SHEETS:
            return element
    return None

# Compact options-numbering key shown under each allocation chart (plain caption
# lines, no bullets; the trailing two spaces force markdown line breaks).
_RTD_OPTION_NUMBERING = (
    "Option 1: higher price category, same vendor · Option 2: other vendor, lower PN price  \n"
    "Option 3: current vendor at PN price · Option 4: place a bid  \n"
    "Option 5: forgo the transaction"
)


def _rtd_density_hist(series, title, x_title, chart_key):
    """Histogram of a continuous score, normalised so the bar heights sum to 1
    (each bar is the proportion of observations falling in that bin). The bin
    count is HALF of Stata's default rule - k = min(sqrt(N), 10*log10(N))
    equal-width bins spanning min..max - per professor feedback (2026-08: "cut
    the number of bins by half so will be closer to the Stata graphs"). The
    series mean is marked with a vertical red line."""
    import plotly.graph_objects as go
    s = pd.Series(series).dropna().astype(float)
    n = len(s)
    vmin, vmax = float(s.min()), float(s.max())
    k_stata = max(1, int(min(np.sqrt(n), 10 * np.log10(n)))) if n > 1 else 1
    k = max(1, k_stata // 2)
    size = (vmax - vmin) / k if vmax > vmin else 1.0
    fig = go.Figure(go.Histogram(
        x=s, histnorm='probability',
        xbins=dict(start=vmin, end=vmax + size * 1e-9, size=size),
        marker_color='steelblue'))
    fig.add_vline(x=float(s.mean()), line_color='red', line_width=2)
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title='Proportion', height=320,
                      margin=dict(t=40, b=10), showlegend=False, bargap=0.05)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def _rtd_fraction_bar(x_labels, fractions, title, x_title, chart_key):
    """Discrete allocation chart: fraction of agents per category, in the given order."""
    fig = px.bar(x=x_labels, y=fractions, title=title)
    fig.update_traces(marker_color='steelblue')
    fig.update_layout(xaxis_title=x_title, yaxis_title='% of agents', height=320,
                      margin=dict(t=40, b=10), yaxis_tickformat='.0%',
                      xaxis=dict(type='category', categoryorder='array',
                                 categoryarray=list(x_labels)))
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def _rtd_score_stats_caption(series):
    """Summary line matching Stata's `summarize` output for the score variable."""
    s = pd.Series(series).astype(float)
    st.caption(f"Mean {s.mean():.4f} · SD {s.std(ddof=1):.4f} · "
               f"Min {s.min():.4f} · Max {s.max():.4f} · N {s.notna().sum():,}")


def render_rtd_comparison_results(results_dict, decision_name):
    """Comparison-mode rendering for Decision 4, grouped by income treatment.

    Each income-treatment group renders as: group title ("Categorical Income
    Treatment" / "Continuous Income Treatment"; omitted when only one group
    exists), then a row of per-population-mode overview cells (Simulation
    Overview + the D4 headline metrics, mirroring the donation-era comparison
    grids), then the detailed per-element sections for the SAME population-mode
    columns underneath - i.e. title -> summary -> details per income treatment,
    instead of all summaries first and all details after.

    Returns True if anything was rendered.
    """
    from app.components import show_overview
    from app.pages.decision_execution import format_result_name

    keys = [k for k, df_ in results_dict.items()
            if hasattr(df_, 'columns') and 'rtd_choice_length' in df_.columns]
    if not keys:
        return False
    # A COMBINED run's frame carries every decision; show_overview would then render
    # the other decisions' analyses (donation rate, disclose income, ...) inside the
    # Decision 4 section. Such frames get the Decision 4 headline metric only.
    combined_frame = any(
        c in results_dict[k].columns for k in keys
        for c in ('donation_default', 'disclose_income', 'disclose_documents'))

    # Group by income treatment, categorical first (matches the donation-era
    # grids' section order). Covers both the Compare-all key style
    # (copula_categorical, ...) and the plain single-population Compare-both
    # keys (categorical / continuous).
    cat_keys = [k for k in keys if k.endswith('categorical')]
    cont_keys = [k for k in keys if k.endswith('continuous')]
    other_keys = [k for k in keys if k not in cat_keys and k not in cont_keys]
    groups = [(title, group_keys) for title, group_keys in (
        ("Categorical Income Treatment", cat_keys),
        ("Continuous Income Treatment", cont_keys),
        (None, other_keys)) if group_keys]

    mode_labels = {
        'copula': '🧬 Copula (Synthetic)',
        'research_spec': '📄 Research Specification',
        'research_baseline': '⚖️ Research Baseline',
    }
    mode_short = {
        'copula': 'Copula',
        'research_spec': 'Research Spec',
        'research_baseline': 'Research Baseline',
    }

    def label_for(key):
        for prefix, label in mode_labels.items():
            if key.startswith(prefix):
                return label
        return format_result_name(key)

    def suffix_for(key):
        """show_overview title suffix, mirroring the donation-era comparison
        grids (' (Copula, Cat)', ' (Categorical)', ...)."""
        for prefix, short in mode_short.items():
            if key.startswith(prefix):
                income = 'Cat' if key.endswith('categorical') else 'Cont'
                return f" ({short}, {income})"
        return f" ({key.replace('_', ' ').title()})"

    show_group_titles = len(groups) > 1
    for group_idx, (group_title, group_keys) in enumerate(groups):
        if group_idx:
            st.markdown("---")
        if show_group_titles and group_title:
            st.markdown(f"#### {group_title}")

        group_labels = [label_for(k) for k in group_keys]
        if len(set(group_labels)) != len(group_labels):
            # e.g. the same population mode appearing twice within a group
            group_labels = [format_result_name(k) for k in group_keys]

        # Rows of up to 3 population-mode columns: the overview cells (summary)
        # first, then the detailed per-element sections for the same keys.
        for start in range(0, len(group_keys), 3):
            row_keys = group_keys[start:start + 3]
            row_labels = group_labels[start:start + 3]
            if start:
                st.markdown("---")

            overview_cols = st.columns(len(row_keys))
            for col, key, label in zip(overview_cols, row_keys, row_labels):
                with col:
                    st.markdown(f"**{label}**")
                    # summary cell; the per-cell "Use This Config" button sits under the
                    # detail cell below (_render_rtd_model_results)
                    if combined_frame:
                        from app.components import rtd_overview_metric
                        st.subheader(f"Simulation Overview{suffix_for(key)}")
                        rtd_label, rtd_value = rtd_overview_metric(results_dict[key])
                        st.metric("Total Agents", f"{len(results_dict[key]):,}")
                        st.metric(rtd_label, rtd_value)
                    else:
                        show_overview(results_dict[key], suffix_for(key), result_key=key)

            detail_cols = st.columns(len(row_keys))
            for col, key, label in zip(detail_cols, row_keys, row_labels):
                with col:
                    st.markdown(f"**{label}**")
                    _render_rtd_model_results(results_dict[key], decision_name,
                                              chart_suffix=f"_{key}", compact=True,
                                              result_key=key)
    return True


def _rtd_result_key_for(df):
    """The simulation_results key this Decision 4 frame belongs to (None if unknown).

    Single-mode runs have one key; otherwise the frame is matched by identity."""
    results = st.session_state.get('simulation_results') or {}
    if not isinstance(results, dict) or not results:
        return None
    for key, frame in results.items():
        if frame is df:
            return key
    return next(iter(results.keys())) if len(results) == 1 else None


def _render_rtd_selection_button(df, result_key):
    """'Use This Config' for an individual Decision 4 run, rendered under the
    decision's results (same placement as the other decisions' buttons)."""
    if not result_key:
        return
    from app.pages.results.comparisons import should_enable_selection
    if not should_enable_selection():
        return
    from app.components import render_rejected_transaction_selection_button
    render_rejected_transaction_selection_button(result_key, df)


def _render_rtd_model_results(df, decision_name, chart_suffix='', compact=False, result_key=None):
    """Model-run results for Decision 4.

    Display rule (professor, 2026-09):
    - a per-element Run button (st.session_state.rtd_run_element set on an individual
      Decision 4 run) renders ONLY that element's section and its Excel;
    - "Run Rejected Transaction Defaults Only" (whole decision) and the complete
      simulation render ONLY the integrated ranking (Section 6): the elements'
      scores, distributions and rankings stay available in the downloadable Excel.
      The element sections are shown for a whole run only when no integrated list
      exists (rank aggregation disabled on the tab).

    chart_suffix disambiguates Streamlit element keys when this view is rendered
    once per result_key (comparison modes). compact=True stacks each section
    vertically for use inside a per-mode comparison column.
    """
    n = len(df)
    active = _rtd_active_element()
    has_integrated = 'rtd_default_list' in df.columns
    if active is not None:
        elements_to_show = {active}
    elif has_integrated:
        elements_to_show = set()
    else:
        elements_to_show = {'ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility'}

    def _element_section(chart_a, chart_b):
        """Chart A (score distribution) and Chart B (allocation) side by side, or
        stacked in compact comparison columns."""
        if compact:
            chart_a()
            chart_b()
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                chart_a()
            with col_b:
                chart_b()

    def _element_download(mech, label):
        """Per-element Excel: Agent ID, the element's independent variables, its
        score and the resulting option sequence per customer."""
        export_df = _prepare_rtd_element_export(df, mech)
        if export_df is None or export_df.empty:
            return
        from io import BytesIO
        from datetime import datetime
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name=_RTD_ELEMENT_SHEETS[mech])
        # Human-readable filename slugs ('ttp' reads too much like 'wtp')
        fname_slug = {'ttp': 'options_list_length', 'loyalty': 'loyalty',
                      'wtp': 'willingness_to_pay', 'risk_taking': 'risk_taking',
                      'flexibility': 'cognitive_flexibility'}[mech]
        st.download_button(
            label=f"📊 Download {label} Excel",
            data=buffer.getvalue(),
            file_name=f"rejected_transaction_{fname_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help=f"Per-agent {label} results: independent variables, score and "
                 "resulting option sequence per customer",
            key=f"rtd_dl_{mech}{chart_suffix}",
        )

    # ---- Element 1: Options List Length (Tendency to Plan) ----
    if 'ttp' in elements_to_show:
        st.markdown("---")
        st.markdown("**1️⃣ Options List Length (Tendency to Plan)**")
        st.markdown("presents how many default options each customer pre-selects (0-5)")
        _rtd_score_stats_caption(df['rtd_weighted_ttp'])

        def _ttp_score_chart():
            _rtd_density_hist(df['rtd_weighted_ttp'], "Tendency to Plan score",
                              "Tendency to Plan score",
                              f"{decision_name}_rtd_ttp_score{chart_suffix}")

        def _ttp_alloc_chart():
            counts = df['rtd_choice_length'].astype(int).value_counts()
            # Bars ALWAYS in natural 0..5 order (professor 2026-08: "always start
            # with 0 and end with 5, presenting bars in order rather than from
            # low to high") - no sort-by-frequency for this chart.
            lengths = list(range(0, 6))
            fractions = [counts.get(l, 0) / n for l in lengths]
            _rtd_fraction_bar([str(l) for l in lengths], fractions,
                              "% of pre-selected options",
                              "Number of pre-selected options",
                              f"{decision_name}_rtd_length_chart{chart_suffix}")
            # Companion table: number of pre-selected options (0-5) -> % of agents
            st.dataframe(pd.DataFrame({
                'Number of pre-selected options': lengths,
                '% of agents': [f"{counts.get(l, 0) / n * 100:.1f}%" for l in lengths],
            }), hide_index=True, use_container_width=True)
            st.caption(_RTD_OPTION_NUMBERING)

        _element_section(_ttp_score_chart, _ttp_alloc_chart)
        _element_download('ttp', "Options List Length")

    # ---- Elements 2-4: priority rankings ----
    # All three score charts plot the STANDARDIZED score (professor 2026-08:
    # "present the standardized loyalty graph rather than the one before
    # standardization"): rtd_loyalty_z / rtd_wtp_z / rtd_rt_z, matching the doc's
    # `histogram weighted_loyalty` after `egen weighted_loyalty = std(...)`.
    score_specs = {
        'loyalty': ('rtd_loyalty_z', "Loyalty score"),
        'wtp': ('rtd_wtp_z', "Willingness-to-Pay score"),
        'risk_taking': ('rtd_rt_z', "Risk-Taking score"),
        # z_anchored_flexibility (doc: `histogram z_anchored_flexibility`)
        'flexibility': ('rtd_flex_z', "Cognitive Flexibility score"),
    }
    for idx, (mech, col_key, label, seq) in enumerate(_RTD_MECHS, start=2):
        seg_col = f'rtd_{col_key}_segment'
        if seg_col not in df.columns or mech not in elements_to_show:
            continue
        st.markdown("---")
        st.markdown(f"**{idx}️⃣ {label} Ranking**")
        st.markdown("Priority sequence " + " > ".join(f"Option {o}" for o in seq))

        score_col, score_title = score_specs[mech]
        _rtd_score_stats_caption(df[score_col])

        def _score_chart(sc=score_col, ti=score_title, ck=col_key):
            _rtd_density_hist(df[sc], ti, ti,
                              f"{decision_name}_rtd_{ck}_score{chart_suffix}")

        def _alloc_chart(sq=seq, sgc=seg_col, ck=col_key, lb=label,
                         per_element=(active == mech)):
            # STATA direction: segment s -> first choice sq[s - 1] (segment 1 gets the
            # top option of the priority sequence; mirrors _ranking_for_segment).
            first_choice = df[sgc].astype(int).map(lambda s: sq[s - 1])
            counts = first_choice.value_counts()
            if per_element:
                # Per-element run (professor 2026-08): categories in the element's
                # priority sequence REVERSED - least likely option on the left,
                # most likely on the right (derived from the runtime sequence).
                order = list(reversed(sq))
            else:
                # Whole-Decision-4 run keeps the least-popular -> most-popular
                # presentation (ascending observed share, ties by option number),
                # as the professor asked to retain for the integrated view.
                order = sorted(sq, key=lambda o: (counts.get(o, 0), o))
            fractions = [counts.get(o, 0) / n for o in order]
            _rtd_fraction_bar([f"Option {o}" for o in order], fractions,
                              f"% of {lb.lower()}-based options ranking",
                              "Selected option",
                              f"{decision_name}_rtd_{ck}_seg_chart{chart_suffix}")
            # Companion table: options 1-5 (natural order) -> first-ranked % of agents
            st.dataframe(pd.DataFrame({
                'Option': [f"Option {o}" for o in range(1, 6)],
                '% of agents': [f"{counts.get(o, 0) / n * 100:.1f}%" for o in range(1, 6)],
            }), hide_index=True, use_container_width=True)
            st.caption(_RTD_OPTION_NUMBERING)

        _element_section(_score_chart, _alloc_chart)
        _element_download(mech, f"{label} Ranking")

    # ---- Integrated default list (Section-6 rank aggregation): the ONLY section of a
    # whole-decision / complete-simulation run ----
    if active is None and has_integrated:
        _render_rtd_aggregation_section(df, decision_name, chart_suffix, _element_section)
    elif active is None:
        st.info("The rank aggregation is disabled on the Decision 4 tab, so no integrated "
                "default list was produced; the individual elements are shown instead.")

    # ---- Whole-decision per-agent Excel export (only for whole-decision runs;
    # per-element runs already have their element's download in the section above) ----
    if active is None:
        st.markdown("---")
        st.markdown("**📥 Download Decision 4 Model Results**")
        sheets = _prepare_rtd_model_export(df)
        if sheets:
            from io import BytesIO
            from datetime import datetime
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                for sheet_name, sheet_df in sheets.items():
                    sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)
            st.caption("The workbook holds the final integrated ranking first, then all "
                       "elements side by side (inputs, scores, intermediate distributions "
                       "and rankings per customer), then one sheet per element.")
            st.download_button(
                label="📊 Download Decision 4 Excel (all elements)",
                data=buffer.getvalue(),
                file_name=f"rejected_transaction_mechanisms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Final integrated ranking + every element's independent variables, "
                     "scores, intermediate distributions and option sequences per customer",
                key=f"rtd_model_download{chart_suffix}",
            )
            with st.expander("📋 Preview Export Data (first 10 rows per sheet)"):
                for sheet_name, sheet_df in sheets.items():
                    st.markdown(f"**{sheet_name} Sheet:**")
                    st.dataframe(sheet_df.head(10).astype(str), use_container_width=True)
                    st.caption(f"Rows: {len(sheet_df):,} · Columns: {', '.join(sheet_df.columns)}")

    # ---- "Use This Config" (individual Decision 4 runs): UNDER the decision's results,
    # where the other decisions place their selection buttons ----
    _render_rtd_selection_button(df, result_key)


def _rtd_list_str(lst):
    """'a > b > c' rendering of an option-number list ('(empty)' for no options)."""
    if isinstance(lst, (list, tuple)):
        return ' > '.join(str(o) for o in lst) if len(lst) else '(empty)'
    return 'N/A'


def _render_rtd_aggregation_section(df, decision_name, chart_suffix, element_section):
    """Section 6 of the Decision 4 results: the integrated default list produced by
    the Section-6 rank aggregation (Kemeny-Young + tie-break hierarchy, truncated to
    the options list length and at Option 5)."""
    n = len(df)
    lists = df['rtd_default_list']
    inputs = df['rtd_consensus_inputs'].iloc[0] if 'rtd_consensus_inputs' in df.columns else []
    input_labels = {'loyalty': 'Loyalty', 'wtp': 'Willingness-to-Pay',
                    'risk_taking': 'Risk-Taking', 'flexibility': 'Cognitive Flexibility'}
    inputs_txt = ', '.join(input_labels.get(m, m) for m in inputs) if isinstance(inputs, list) else ''

    st.markdown("---")
    st.markdown("**6️⃣ Integrated Default List (Rank Aggregation)**")
    st.markdown(
        f"presents each customer's final pre-selected default list: the Kemeny-Young "
        f"consensus of the {inputs_txt} rankings (ties broken Schulze → Copeland → "
        f"footrule → random), cut to the customer's Options List Length. Option 5 "
        f"(forgo the transaction) is the final option of the produced list: any option "
        f"ranked after it is discarded."
    )
    st.caption("The individual elements' results are not shown for a whole-decision run; "
               "their scores, distributions and rankings are in the Excel below, and each "
               "element has its own Run button on the Decision 4 tab.")

    def _lists_chart():
        # The ranking itself: share of agents per integrated default list (the most
        # common lists; the list length is element 1's result and is not repeated here)
        common = lists.apply(_rtd_list_str).value_counts()
        top = common.head(8)
        _rtd_fraction_bar(list(top.index), [c / n for c in top.values],
                          "% of agents per integrated default list",
                          "Integrated default list (options in order)",
                          f"{decision_name}_rtd_agg_lists_chart{chart_suffix}")
        rows = [{'Integrated default list': lst, '% of agents': f"{c / n * 100:.1f}%"}
                for lst, c in common.head(10).items()]
        if len(common) > 10:
            rest = common.iloc[10:].sum()
            rows.append({'Integrated default list': f"other ({len(common) - 10} lists)",
                         '% of agents': f"{rest / n * 100:.1f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    def _first_choice_chart():
        firsts = lists.apply(lambda l: l[0] if isinstance(l, list) and len(l) else 0)
        counts = firsts.value_counts()
        order = list(range(1, 6))
        fractions = [counts.get(o, 0) / n for o in order]
        _rtd_fraction_bar([f"Option {o}" for o in order], fractions,
                          "% of first integrated default option",
                          "First option in the integrated default list",
                          f"{decision_name}_rtd_agg_first_chart{chart_suffix}")
        rows = [{'Option': f"Option {o}", '% of agents': f"{counts.get(o, 0) / n * 100:.1f}%"}
                for o in order]
        rows.append({'Option': 'No default options (list length 0)',
                     '% of agents': f"{counts.get(0, 0) / n * 100:.1f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(_RTD_OPTION_NUMBERING)

    element_section(_lists_chart, _first_choice_chart)

    # Tie-break diagnostics + the full consensus rankings (before the cut)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Tie-breaking stage that settled the consensus ranking**")
        settled = df['rtd_consensus_settled_by'].value_counts()
        st.dataframe(pd.DataFrame({
            'Settled by': [_RTD_STAGE_LABELS[s] for s in _RTD_STAGE_LABELS],
            '% of agents': [f"{settled.get(s, 0) / n * 100:.1f}%" for s in _RTD_STAGE_LABELS],
        }), hide_index=True, use_container_width=True)
        if 'rtd_consensus_is_kemeny_optimal' in df.columns:
            opt_share = df['rtd_consensus_is_kemeny_optimal'].astype(bool).mean() * 100
            st.caption(f"Final consensus ranking is Kemeny-optimal for {opt_share:.1f}% of agents.")
    with col_b:
        st.markdown("**Most common consensus rankings (full ranking before the cut)**")
        consensus = df['rtd_consensus_ranking'].apply(_rtd_list_str).value_counts().head(10)
        st.dataframe(pd.DataFrame({
            'Consensus ranking': consensus.index,
            '% of agents': [f"{c / n * 100:.1f}%" for c in consensus.values],
        }), hide_index=True, use_container_width=True)

    export_df = _prepare_rtd_aggregation_export(df)
    if export_df is not None and not export_df.empty:
        from io import BytesIO
        from datetime import datetime
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name=_RTD_AGG_SHEET)
        st.download_button(
            label="📊 Download Integrated Default List Excel",
            data=buffer.getvalue(),
            file_name=f"rejected_transaction_integrated_default_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Per-agent mechanism lists, consensus ranking, tie-break diagnostics and "
                 "the integrated default list",
            key=f"rtd_dl_aggregation{chart_suffix}",
        )


def _prepare_rtd_aggregation_export(df):
    """Integrated Default List sheet: Agent ID, choice length, the mechanism lists
    that entered the aggregation, the consensus ranking with its tie-break
    diagnostics, and the truncated default list as choice1..choice5."""
    try:
        out = pd.DataFrame(index=df.index)
        out['Agent ID'] = _rtd_agent_id_series(df)
        out['choice_length'] = df['rtd_choice_length']
        for col_key, stata in (('loyalty', 'loyalty'), ('wtp', 'WTP'), ('rt', 'RT'), ('flex', 'Flexibility')):
            if f'rtd_{col_key}_ranking' in df.columns:
                out[f'{stata}_list'] = df[f'rtd_{col_key}_ranking'].apply(_rtd_list_str)
        out['consensus_ranking'] = df['rtd_consensus_ranking'].apply(_rtd_list_str)
        out['kemeny_status'] = df['rtd_consensus_kemeny_status']
        out['n_kemeny_optimal'] = df['rtd_consensus_n_kemeny_optimal']
        out['settled_by'] = df['rtd_consensus_settled_by']
        out['is_kemeny_optimal'] = df['rtd_consensus_is_kemeny_optimal']
        out['truncated_by'] = df['rtd_consensus_truncated_by']
        out['default_list_length'] = df['rtd_default_list_length']
        _rtd_choice_columns(out, df['rtd_default_list'])
        return out
    except Exception as e:
        st.error(f"Error preparing Decision 4 {_RTD_AGG_SHEET} export: {e}")
        return None


def _rtd_agent_id_series(df):
    if 'agent_id' in df.columns:
        return df['agent_id']
    return pd.Series(range(1, len(df) + 1), index=df.index)


def _rtd_element_inputs_frame(df, mech):
    """Agent ID + the element's OWN independent variables (Stata-aligned names).
    Categorical income adds 'Assigned Allowance Level' for wtp / risk_taking."""
    out = pd.DataFrame(index=df.index)
    out['Agent ID'] = _rtd_agent_id_series(df)
    inputs = list(_RTD_ELEMENT_INPUTS[mech])
    if mech in ('wtp', 'risk_taking') and _rtd_frame_income_mode(df) == 'categorical':
        inputs.append('Assigned Allowance Level')
    for col in inputs:
        if col in df.columns:
            out[col] = df[col]
    return out


def _rtd_choice_columns(out, rankings):
    """choice1..choice5 columns (option numbers; blank beyond the list length)."""
    for pos in range(1, 6):
        out[f'choice{pos}'] = rankings.apply(
            lambda lst, p=pos: lst[p - 1] if isinstance(lst, list) and len(lst) >= p else np.nan)


_RTD_STATA_NAMES = {'loyalty': ('loyalty', 'loyalty'), 'wtp': ('wtp', 'WTP'),
                    'risk_taking': ('rt', 'RT'), 'flexibility': ('flex', 'Flexibility')}


def _rtd_flex_intermediates(out, df):
    """Cognitive Flexibility intermediates in Stata naming: the calculated IVW score,
    its population z, and z_stdactions (the anchored score / its z follow as
    Flexibility_score / z_Flexibility)."""
    for src, dst in (('rtd_flex_ivw', 'Flexibility_calculated_ivw'),
                     ('rtd_flex_z_ivw', 'z_Flexibility_calculated_ivw'),
                     ('rtd_z_stdactions', 'z_stdactions')):
        if src in df.columns:
            out[dst] = df[src]


def _prepare_rtd_element_export(df, mech):
    """Per-element Excel frame: Agent ID, ONLY this element's independent variables,
    its score, the segment (or list length for TTP) and the resulting option
    sequence per customer as choice1..choice5. Stata-aligned column names."""
    try:
        out = _rtd_element_inputs_frame(df, mech)
        if mech == 'ttp':
            out['weighted_ttp'] = df['rtd_weighted_ttp']
            out['choice_length'] = df['rtd_choice_length']
            return out
        col_key, stata = _RTD_STATA_NAMES[mech]
        if mech == 'flexibility':
            _rtd_flex_intermediates(out, df)
        out[f'{stata}_score'] = df[f'rtd_{col_key}_score']
        if f'rtd_{col_key}_z' in df.columns:
            out[f'z_{stata}'] = df[f'rtd_{col_key}_z']
        out[f'{stata}_segment'] = df[f'rtd_{col_key}_segment']
        _rtd_choice_columns(out, df[f'rtd_{col_key}_ranking'])
        return out
    except Exception as e:
        st.error(f"Error preparing Decision 4 {_RTD_ELEMENT_SHEETS.get(mech, mech)} export: {e}")
        return None


def _prepare_rtd_model_export(df):
    """Whole-decision Decision 4 workbook, organized as one self-contained sheet per
    element ('Options List Length', 'Loyalty', 'Willingness-to-Pay', 'Risk-Taking',
    'Cognitive Flexibility') plus the 'Integrated Default List' aggregation sheet.

    Each sheet mirrors the per-element file (Agent ID + the element's own
    independent variables + choice1..choice5) and additionally carries the
    intermediate distributions: score, z (where present), deterministic and final
    segment / list length, and the sigma used. Stata-aligned column names.

    Returns an ordered {sheet_name: DataFrame} dict, or None on error.
    """
    try:
        sheets = {}

        # ---- 1. Final integrated ranking (Section-6 rank aggregation) first ----
        if 'rtd_default_list' in df.columns:
            agg = _prepare_rtd_aggregation_export(df)
            if agg is not None:
                sheets[_RTD_AGG_SHEET] = agg

        # ---- 2. All elements side by side (inputs -> element by element -> final) ----
        all_elements = _prepare_rtd_all_elements_export(df)
        if all_elements is not None:
            sheets[_RTD_ALL_ELEMENTS_SHEET] = all_elements

        # ---- 3. One sheet per element ----
        # ---- Options List Length (TTP) ----
        ttp = _rtd_element_inputs_frame(df, 'ttp')
        for src, dst in [('rtd_weighted_ttp', 'weighted_ttp'),
                         ('rtd_weighted_ttp06', 'weighted_ttp06'),
                         ('rtd_choice_length_deterministic', 'choice_length_deterministic'),
                         ('rtd_choice_length', 'choice_length'),
                         ('rtd_sigma_used_ttp', 'sigma_used_ttp')]:
            if src in df.columns:
                ttp[dst] = df[src]
        sheets[_RTD_ELEMENT_SHEETS['ttp']] = ttp

        # ---- Rankings: Loyalty / Willingness-to-Pay / Risk-Taking ----
        for mech, (col_key, stata) in _RTD_STATA_NAMES.items():
            if f'rtd_{col_key}_score' not in df.columns:
                continue
            sheet = _rtd_element_inputs_frame(df, mech)
            if mech == 'flexibility':
                _rtd_flex_intermediates(sheet, df)
            sheet[f'{stata}_score'] = df[f'rtd_{col_key}_score']
            if f'rtd_{col_key}_z' in df.columns:
                sheet[f'z_{stata}'] = df[f'rtd_{col_key}_z']
            if f'rtd_{col_key}_segment_deterministic' in df.columns:
                sheet[f'{stata}_segment_deterministic'] = df[f'rtd_{col_key}_segment_deterministic']
            sheet[f'{stata}_segment'] = df[f'rtd_{col_key}_segment']
            _rtd_choice_columns(sheet, df[f'rtd_{col_key}_ranking'])
            if f'rtd_sigma_used_{col_key}' in df.columns:
                sheet[f'sigma_used_{stata}'] = df[f'rtd_sigma_used_{col_key}']
            sheets[_RTD_ELEMENT_SHEETS[mech]] = sheet
        return sheets
    except Exception as e:
        st.error(f"Error preparing Decision 4 export: {e}")
        return None


_RTD_ALL_ELEMENTS_SHEET = 'All Elements'


def _prepare_rtd_all_elements_export(df):
    """One row per customer with every element side by side, columns grouped in
    the order of the decision: Agent ID -> inputs (traits, education, income /
    allowance level, stdactions) -> 1 Options List Length -> 2 Loyalty -> 3 WTP ->
    4 Risk-Taking -> 5 Cognitive Flexibility (each: score, z, segment, option list)
    -> 6 Integrated default list (consensus ranking, tie-break stage, final options).
    Stata-aligned column names, prefixed per element."""
    try:
        out = pd.DataFrame(index=df.index)
        out['Agent ID'] = _rtd_agent_id_series(df)
        inputs = ['ExtraversionBig5', 'Agreeable', 'NeuroticismBig5', 'ConscientiousnessBig5',
                  'OpennessBig5', 'Education', 'income']
        if _rtd_frame_income_mode(df) == 'categorical':
            inputs.append('Assigned Allowance Level')
        inputs.append('stdactions')
        for col in inputs:
            if col in df.columns:
                out[col] = df[col]

        # 1. Options List Length
        for src, dst in [('rtd_weighted_ttp', 'weighted_ttp'),
                         ('rtd_weighted_ttp06', 'weighted_ttp06'),
                         ('rtd_choice_length_deterministic', 'choice_length_deterministic'),
                         ('rtd_choice_length', 'choice_length')]:
            if src in df.columns:
                out[dst] = df[src]

        # 2-5. Ranking elements
        for mech, (col_key, stata) in _RTD_STATA_NAMES.items():
            if f'rtd_{col_key}_score' not in df.columns:
                continue
            if mech == 'flexibility':
                _rtd_flex_intermediates(out, df)
            out[f'{stata}_score'] = df[f'rtd_{col_key}_score']
            if f'rtd_{col_key}_z' in df.columns:
                out[f'z_{stata}'] = df[f'rtd_{col_key}_z']
            if f'rtd_{col_key}_segment_deterministic' in df.columns:
                out[f'{stata}_segment_deterministic'] = df[f'rtd_{col_key}_segment_deterministic']
            out[f'{stata}_segment'] = df[f'rtd_{col_key}_segment']
            out[f'{stata}_list'] = df[f'rtd_{col_key}_ranking'].apply(_rtd_list_str)

        # 6. Integrated default list
        if 'rtd_default_list' in df.columns:
            out['consensus_ranking'] = df['rtd_consensus_ranking'].apply(_rtd_list_str)
            out['settled_by'] = df['rtd_consensus_settled_by']
            out['truncated_by'] = df['rtd_consensus_truncated_by']
            out['default_list_length'] = df['rtd_default_list_length']
            for pos in range(1, 6):
                out[f'final_choice{pos}'] = df['rtd_default_list'].apply(
                    lambda lst, p=pos: lst[p - 1] if isinstance(lst, list) and len(lst) >= p else np.nan)
        return out
    except Exception as e:
        st.error(f"Error preparing Decision 4 {_RTD_ALL_ELEMENTS_SHEET} export: {e}")
        return None


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
    
    Creates columns: Agent ID, Honesty_Humility, Assigned Allowance Level, Study Program,
    Group_experiment, TWT+Sospeso [=AW2+AX2]{Periods 1+2}, income,
    Priority 1, Priority 2, Priority 3, Priority 4, Priority 5
    
    Priority columns contain option numbers (1, 2, 3, 4, 5) or N/A if not selected.
    
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
    
    # Honesty_Humility
    if 'Honesty_Humility' in df.columns:
        export_df['Honesty_Humility'] = df['Honesty_Humility'].round(2)
    else:
        export_df['Honesty_Humility'] = ''
    
    # Assigned Allowance Level
    if 'Assigned Allowance Level' in df.columns:
        export_df['Assigned Allowance Level'] = df['Assigned Allowance Level']
    elif 'income_category' in df.columns:
        export_df['Assigned Allowance Level'] = df['income_category']
    else:
        export_df['Assigned Allowance Level'] = ''
    
    # Study Program
    if 'Study Program' in df.columns:
        export_df['Study Program'] = df['Study Program']
    else:
        export_df['Study Program'] = ''
    
    # Group_experiment
    if 'Group_experiment' in df.columns:
        export_df['Group_experiment'] = df['Group_experiment']
    elif 'group' in df.columns:
        export_df['Group_experiment'] = df['group']
    elif 'group_experiment' in df.columns:
        export_df['Group_experiment'] = df['group_experiment']
    else:
        export_df['Group_experiment'] = ''
    
    # TWT+Sospeso
    if 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}' in df.columns:
        export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'].round(2)
    else:
        export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = ''
    
    # Income
    if 'income' in df.columns:
        export_df['income'] = df['income'].round(2)
    elif 'actual_allowance' in df.columns:
        export_df['income'] = df['actual_allowance'].round(2)
    else:
        export_df['income'] = ''
    
    # Priority columns (1-5)
    for priority_pos in range(1, 6):
        column_name = f'Priority {priority_pos}'
        priority_values = []
        
        for agent_list in decision_data:
            if isinstance(agent_list, list):
                # Check if agent has this priority position
                if len(agent_list) >= priority_pos:
                    option_code = agent_list[priority_pos - 1]
                    option_number = option_to_number.get(option_code, 'N/A')
                    priority_values.append(option_number)
                else:
                    # Agent doesn't have this many priorities
                    priority_values.append('N/A')
            else:
                # Single value (legacy format) - only for priority 1
                if priority_pos == 1:
                    option_number = option_to_number.get(agent_list, 'N/A')
                    priority_values.append(option_number)
                else:
                    priority_values.append('N/A')
        
        export_df[column_name] = priority_values
    
    return export_df


def render_rejected_bid_value(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_bid_value"""
    st.markdown("**Not relevant given choice of Option 5**")

