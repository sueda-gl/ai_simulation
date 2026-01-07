# app/pages/results/visualizations/bidding_viz.py
"""
Bidding-related visualization functions.
Handles bid_value decision.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from io import BytesIO
from app.utils.timestamp_utils import TimestampConverter


def _apply_price_formatting_bid(writer, sheet_name: str, df: pd.DataFrame):
    """
    Apply Excel number formatting to price-related columns to display 2 decimal places.
    """
    from openpyxl.styles import numbers
    
    price_columns = [
        'Honesty_Humility', 'income', 'Vendor Price', 'Bid Value',
        'TWT+Sospeso [=AW2+AX2]{Periods 1+2}'
    ]
    
    workbook = writer.book
    worksheet = workbook[sheet_name]
    
    for col_idx, col_name in enumerate(df.columns, start=1):
        if col_name in price_columns:
            for row_idx in range(2, len(df) + 2):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                if isinstance(cell.value, (int, float)) and cell.value is not None:
                    cell.number_format = '0.00'


def _build_bid_value_export(df):
    """
    Build transaction-level export data for bid transactions only.
    
    Returns a list of transaction records with fields:
    - Transaction ID
    - Agent ID
    - Honesty_Humility
    - Assigned Allowance Level
    - Study Program
    - Group_experiment
    - TWT+Sospeso [=AW2+AX2]{Periods 1+2}
    - income
    - Vendor ID
    - Vendor Price
    - Bid Value
    - Purchase Timestamp
    - Period
    
    Note: This export only includes BID transactions (typically from Regular customers
    who chose to bid rather than use Purchase Now).
    """
    bid_records = []
    
    if 'purchase_requests' not in df.columns:
        return bid_records
    
    # Use centralized timestamp converter
    ts_converter = TimestampConverter()
    
    # Get vendor data for price lookup
    vendors_data = None
    if hasattr(st.session_state, 'vendors'):
        vendors_data = st.session_state.vendors
    elif hasattr(st.session_state, 'vendors_data'):
        vendors_data = st.session_state.vendors_data
    elif hasattr(st.session_state, 'simulation_results') and isinstance(st.session_state.simulation_results, dict):
        vendors_data = st.session_state.simulation_results.get('vendors_data', None)
    
    # Build vendor lookup
    vendor_lookup = {}
    if vendors_data:
        for vendor in vendors_data:
            vendor_id = vendor.get('vendor_id')
            if vendor_id is not None:
                vendor_lookup[vendor_id] = vendor
                vendor_lookup[str(vendor_id)] = vendor
    
    for idx, row in df.iterrows():
        # Get agent information
        agent_id = row.get('agent_id', idx + 1)
        
        # Agent traits
        honesty_humility = row.get('Honesty_Humility', np.nan)
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        study_program = row.get('Study Program', '')
        
        # Group_experiment with fallbacks
        group_experiment = row.get('Group_experiment', '')
        if group_experiment == '' or pd.isna(group_experiment):
            group_experiment = row.get('group', '')
        if group_experiment == '' or pd.isna(group_experiment):
            group_experiment = row.get('group_experiment', '')
        if pd.isna(group_experiment):
            group_experiment = ''
        
        twt_sospeso = row.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', np.nan)
        
        # Income
        income = np.nan
        if 'income' in row and pd.notna(row['income']):
            income = round(row['income'], 2)
        elif 'actual_allowance' in row and pd.notna(row['actual_allowance']):
            income = round(row['actual_allowance'], 2)
        
        # Get purchase requests
        purchase_requests = row.get('purchase_requests', [])
        if not isinstance(purchase_requests, list):
            continue
        
        # Process each purchase request - only include BID transactions
        for req_idx, request in enumerate(purchase_requests):
            if not isinstance(request, dict):
                continue
            
            # Check if this is a BID transaction
            platform_price = request.get('platformPrice', request.get('platform_price', ''))
            bid_value = request.get('bid_value', 'N/A')
            
            # Only include actual BID transactions with valid bid values
            if platform_price != 'BID' or bid_value == 'N/A' or bid_value is None:
                continue
            
            try:
                bid_value_numeric = float(bid_value)
            except (ValueError, TypeError):
                continue
            
            # Get transaction ID
            transaction_id = request.get('transaction_id', f"A{agent_id}_R{req_idx+1}")
            
            # Get vendor info
            vendor_id = request.get('vendorID', request.get('vendor_id'))
            vendor_price = np.nan
            
            if vendor_id is not None:
                lookup_key = vendor_id
                if isinstance(vendor_id, float) and vendor_id.is_integer():
                    lookup_key = int(vendor_id)
                
                if lookup_key in vendor_lookup:
                    vendor_price = vendor_lookup[lookup_key].get('price', np.nan)
                elif str(lookup_key) in vendor_lookup:
                    vendor_price = vendor_lookup[str(lookup_key)].get('price', np.nan)
            
            # Get timestamp
            timestamp_hours = request.get('timestamp_hours', np.nan)
            ts_result = ts_converter.convert(timestamp_hours)
            period = ts_result['period']
            request_datetime = ts_result['datetime']
            
            # Build record
            record = {
                'Transaction ID': transaction_id,
                'Agent ID': agent_id,
                'Honesty_Humility': honesty_humility,
                'Assigned Allowance Level': allowance_level,
                'Study Program': study_program,
                'Group_experiment': group_experiment,
                'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': twt_sospeso,
                'income': income,
                'Vendor ID': f"Vendor {int(vendor_id)}" if vendor_id is not None and not pd.isna(vendor_id) else '',
                'Vendor Price': vendor_price,
                'Bid Value': bid_value_numeric,
                'Purchase Timestamp': ts_result['formatted'],
                'Period': period,
                '_sort_datetime': request_datetime  # Hidden for sorting
            }
            
            bid_records.append(record)
    
    # Sort by timestamp
    if bid_records:
        bid_records.sort(key=lambda x: x['_sort_datetime'] if isinstance(x['_sort_datetime'], datetime) else datetime.min)
        
        # Remove hidden sort column
        for record in bid_records:
            record.pop('_sort_datetime', None)
    
    return bid_records


def render_bid_value(df, decision_name, decision_title, decision_data):
    """Visualization for bid_value with bidding price range formula"""
    
    # Get parameters from session state (from Page 1)
    if hasattr(st.session_state, 'sim_params'):
        platform_markup = getattr(st.session_state.sim_params, 'platform_markup', 0.1)  # Default 10%
        price_range = getattr(st.session_state.sim_params, 'price_range', 0.25)  # Default 25%
    else:
        # Fallback defaults
        platform_markup = 0.1
        price_range = 0.25
    
    # ============================================================================
    # SECTION 1: ACTUAL SIMULATION RESULTS (Request-Level)
    # ============================================================================
    st.markdown("### 📊 Bid Values")
    st.caption("Each bid request gets a unique random bid value based on the vendor's price")
    
    if 'purchase_requests' in df.columns:
        # Extract all bid values from all requests
        all_bids = []
        
        for idx, row in df.iterrows():
            requests = row.get('purchase_requests', [])
            if isinstance(requests, list):
                for req in requests:
                    if isinstance(req, dict):
                        bid_val = req.get('bid_value')
                        # Only include actual numeric bid values (not "N/A")
                        if bid_val != 'N/A' and bid_val is not None:
                            try:
                                all_bids.append(float(bid_val))
                            except (ValueError, TypeError):
                                pass
        
        if len(all_bids) > 0:
            # Summary metrics
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            
            with col_stats1:
                st.metric("Total Bid Requests", f"{len(all_bids):,}", 
                         help="Number of BID requests across all agents")
            
            with col_stats2:
                st.metric("Mean Bid", f"€{np.mean(all_bids):.2f}")
            
            with col_stats3:
                st.metric("Min Bid", f"€{min(all_bids):.2f}")
            
            with col_stats4:
                st.metric("Max Bid", f"€{max(all_bids):.2f}")
            
            # Distribution visualization
            st.markdown("**📈 Distribution of Bid Values:**")
            
            col_hist, col_info = st.columns([2, 1])
            
            with col_hist:
                st.markdown(f"**Distribution of {len(all_bids):,} Bid Values Across All Requests**")
                fig = px.histogram(
                    x=all_bids,
                    nbins=30,
                    labels={'x': 'Bid Amount (€)', 'count': 'Number of Bids'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_info:
                st.markdown("**📊 Statistics**")
                stats_df = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{len(all_bids):,}",
                        f"€{np.mean(all_bids):.2f}",
                        f"€{np.median(all_bids):.2f}",
                        f"€{np.std(all_bids):.2f}",
                        f"€{min(all_bids):.2f}",
                        f"€{max(all_bids):.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Show unique count
                unique_bids = len(set(all_bids))
                st.caption(f"✅ {unique_bids:,} unique bid values")
                if unique_bids == len(all_bids):
                    st.success("🎯 All bids are unique!")
            
            # ========================================================================
            # EXCEL EXPORT SECTION
            # ========================================================================
            st.markdown("---")
            st.markdown("**💾 Export Bid Transaction Data**")
            st.caption("Download detailed data for each bid transaction with agent traits and vendor information")
            
            try:
                # Build bid records
                bid_records = _build_bid_value_export(df)
                
                if len(bid_records) > 0:
                    bid_df = pd.DataFrame(bid_records)
                    
                    # Create Excel with multiple sheets
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        # Sheet 1: Total (all periods)
                        bid_df.to_excel(writer, index=False, sheet_name='Total')
                        _apply_price_formatting_bid(writer, 'Total', bid_df)
                        
                        # Additional sheets by Period
                        if 'Period' in bid_df.columns:
                            periods = sorted(bid_df['Period'].dropna().unique())
                            for period_val in periods:
                                period_df = bid_df[bid_df['Period'] == period_val]
                                sheet_name = f'Period {int(period_val)}'
                                period_df.to_excel(writer, index=False, sheet_name=sheet_name)
                                _apply_price_formatting_bid(writer, sheet_name, period_df)
                    
                    col_download, col_info = st.columns([1, 2])
                    
                    with col_download:
                        st.download_button(
                            label="📊 Download Bid Values Excel",
                            data=buffer.getvalue(),
                            file_name=f"bid_values_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Download bid transaction data with agent traits and vendor prices"
                        )
                    
                    with col_info:
                        num_agents = bid_df['Agent ID'].nunique()
                        st.caption(f"📋 {len(bid_df):,} bid transactions from {num_agents:,} agents")
                        st.caption("✅ Fields: Transaction ID, Agent ID, Agent Traits, Vendor ID, Vendor Price, Bid Value, Timestamp, Period")
                else:
                    st.info("ℹ️ No bid transactions to export")
            
            except ImportError:
                st.warning("⚠️ Excel export requires openpyxl package")
            except Exception as e:
                st.error(f"❌ Error creating Excel file: {str(e)}")
        
        else:
            st.info("ℹ️ No bid requests found (no agents chose to bid in this simulation)")
    else:
        st.warning("⚠️ No purchase_requests data available in results")
    
    # ============================================================================
    # SECTION 2: FORMULA EXPLANATION (Educational)
    # ============================================================================
    st.markdown("---")
    st.markdown("### 📚 How Bid Values Are Calculated")
    st.info("**Note**: The following is an illustration of how the bidding range formula works. Each vendor in the simulation has its own price, so bid ranges vary by vendor.")
    
    # Use example vendor price for illustration
    example_vendor_price = 100.0
    
    # Calculate example bidding range using the formula
    baseline_price = (1 + platform_markup) * example_vendor_price  # Pc = (1+m) × vendor_price
    min_bid_price = (1 - price_range) * baseline_price      # Pmb = (1-r) × Pc
    max_bid_price = (1 + price_range) * baseline_price      # Ppn = (1+r) × Pc
    
    # Display formula parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Simulation Parameters:**")
        st.write(f"• **Platform Markup (m)**: {platform_markup:.1%}")
        st.write(f"• **Price Range Parameter (r)**: {price_range:.1%}")
    
    with col2:
        st.markdown("**🎯 Bidding Behavior:**")
        st.write("Each agent selects a **random bid value** within the range calculated for their chosen vendor")
    
    st.markdown("---")
    
    # Formula breakdown with example
    col_formula, col_example = st.columns([1, 1])
    
    with col_formula:
        st.markdown("**📐 Formula:**")
        st.code("""
Given a vendor price V:
  
1. Baseline Price (Pc):
   Pc = (1 + m) × V
   
2. Minimum Bid (Pmb):
   Pmb = (1 - r) × Pc
   
3. Maximum Bid (Ppn):
   Ppn = (1 + r) × Pc
   
4. Actual Bid:
   Random value in [Pmb, Ppn)
        """, language="text")
    
    with col_example:
        st.markdown(f"**💡 Example Calculation:**")
        st.markdown(f"*Assuming vendor price = €{example_vendor_price:.2f}*")
        st.write("")
        st.write(f"**Step 1: Baseline Price**")
        st.write(f"Pc = (1 + {platform_markup:.1%}) × €{example_vendor_price:.2f}")
        st.write(f"Pc = €{baseline_price:.2f}")
        st.write("")
        st.write(f"**Step 2: Minimum Bid**")
        st.write(f"Pmb = (1 - {price_range:.1%}) × €{baseline_price:.2f}")
        st.write(f"Pmb = €{min_bid_price:.2f}")
        st.write("")
        st.write(f"**Step 3: Maximum Bid**")
        st.write(f"Ppn = (1 + {price_range:.1%}) × €{baseline_price:.2f}")
        st.write(f"Ppn = €{max_bid_price:.2f}")
        st.write("")
        st.success(f"**Bidding Range**: [€{min_bid_price:.2f}, €{max_bid_price:.2f})")
        
        # Show sample bids from this example
        import random
        st.write("")
        st.write("**Example random bids:**")
        example_bids = []
        for i in range(5):
            random_bid = random.uniform(min_bid_price, max_bid_price)
            example_bids.append(f"€{random_bid:.2f}")
        st.caption(", ".join(example_bids))
    
    st.caption("💡 **Remember**: In the actual simulation, each of the 6 vendors has a different price, so the bidding range varies per vendor.")

