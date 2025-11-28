import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
from app.models import initialize_session_state


def _build_agent_level_dataframe(df, vendors_data=None):
    """
    Build agent-level DataFrame with one row per agent.
    
    Includes:
    - Agent ID and traits
    - All agent-level decisions
    - Summary statistics from transactions
    - Vendor proximity scores (expanded)
    
    Args:
        df: Original simulation results DataFrame
        vendors_data: List of vendor dictionaries (optional)
        
    Returns:
        pd.DataFrame: Agent-level data
    """
    agent_records = []
    
    for idx, row in df.iterrows():
        agent_id = row.get('agent_id', idx + 1)
        
        # Start with agent ID and traits
        agent_record = {
            'Agent ID': agent_id,
            'Honesty_Humility': row.get('Honesty_Humility', np.nan),
            'Assigned Allowance Level': row.get('Assigned Allowance Level', np.nan),
            'Study Program': row.get('Study Program', ''),
            'Group_experiment': row.get('Group_experiment', ''),
            'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': row.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', np.nan),
        }
        
        # Decision 1: Disclose Income
        agent_record['disclose_income'] = row.get('disclose_income', '')
        
        # Decision 2: Disclose Documents & Customer Type
        agent_record['disclose_documents'] = row.get('disclose_documents', '')
        agent_record['customer_type'] = row.get('customer_type', '')
        
        # Decision 3: Donation Default
        agent_record['donation_default'] = row.get('donation_default', np.nan)
        
        # Decision 4: Rejected Transaction Defaults
        agent_record['rejected_transaction_defaults'] = row.get('rejected_transaction_defaults', '')
        
        # Decision 5: Vendor Choice Weights (flatten dict to columns)
        vendor_weights = row.get('vendor_choice_weights', {})
        if isinstance(vendor_weights, dict):
            agent_record['weight_price'] = vendor_weights.get('price', np.nan)
            agent_record['weight_quality'] = vendor_weights.get('quality', np.nan)
            agent_record['weight_proximity'] = vendor_weights.get('proximity', np.nan)
            agent_record['weight_sustainability'] = vendor_weights.get('sustainability', np.nan)
        else:
            agent_record['weight_price'] = np.nan
            agent_record['weight_quality'] = np.nan
            agent_record['weight_proximity'] = np.nan
            agent_record['weight_sustainability'] = np.nan
        
        # Decision 6: Purchasing Quantity (agent-level)
        agent_record['income'] = row.get('income', np.nan)
        agent_record['income_category'] = row.get('income_category', np.nan)
        agent_record['purchasing_quantity'] = row.get('purchasing_quantity', 0)
        
        # Decision 7: Purchasing Frequency
        agent_record['purchasing_frequency'] = row.get('purchasing_frequency', np.nan)
        
        # Decision 8: Vendor Selection (agent-level)
        agent_record['preferred_vendor'] = row.get('preferred_vendor', np.nan)
        
        # Vendor proximity scores (expand to columns)
        proximity_scores = row.get('vendor_proximity_scores', {})
        if isinstance(proximity_scores, dict):
            # Get number of vendors
            num_vendors = len(vendors_data) if vendors_data else 0
            if num_vendors == 0:
                # Infer from proximity_scores keys
                vendor_ids = [int(vid) for vid in proximity_scores.keys() if vid.isdigit()]
                num_vendors = max(vendor_ids) if vendor_ids else 0
            
            for vendor_id in range(1, num_vendors + 1):
                agent_record[f'proximity_v{vendor_id}'] = proximity_scores.get(str(vendor_id), np.nan)
        
        # Decision 9: Purchase vs Bid (summary statistics from transactions)
        purchase_requests = row.get('purchase_requests', [])
        if isinstance(purchase_requests, list):
            total_requests = len(purchase_requests)
            pn_count = sum(1 for req in purchase_requests 
                          if isinstance(req, dict) and req.get('platformPrice') == 'PN')
            bid_count = sum(1 for req in purchase_requests 
                           if isinstance(req, dict) and req.get('platformPrice') == 'BID')
            
            agent_record['total_purchase_requests'] = total_requests
            agent_record['pn_requests_count'] = pn_count
            agent_record['bid_requests_count'] = bid_count
            agent_record['pct_purchase_now'] = (pn_count / total_requests * 100) if total_requests > 0 else np.nan
        else:
            agent_record['total_purchase_requests'] = 0
            agent_record['pn_requests_count'] = 0
            agent_record['bid_requests_count'] = 0
            agent_record['pct_purchase_now'] = np.nan
        
        # Decision 11: Rejected Transaction Option
        agent_record['rejected_transaction_option'] = row.get('rejected_transaction_option', '')
        
        # Decision 13: Final Donation Rate
        agent_record['final_donation_rate'] = row.get('final_donation_rate', np.nan)
        
        agent_records.append(agent_record)
    
    return pd.DataFrame(agent_records)


def _build_transaction_level_dataframe(df, vendors_data=None, simulation_params=None):
    """
    Build transaction-level DataFrame with one row per purchase request.
    
    Includes:
    - Transaction ID and timing
    - Agent reference (ID and traits)
    - Vendor selection and attributes
    - Purchase decision (PN/BID)
    - Pricing and donation information
    
    Args:
        df: Original simulation results DataFrame
        vendors_data: List of vendor dictionaries (optional)
        simulation_params: Simulation parameters for pricing calculations
        
    Returns:
        pd.DataFrame: Transaction-level data
    """
    transaction_records = []
    
    # Get pricing parameters
    market_price = 100.0
    platform_markup = 0.1
    price_range = 0.25
    duration_hours = 1.0
    
    if simulation_params:
        sim_params = simulation_params.get('simulation', {})
        market_price = sim_params.get('market_price', 100.0)
        platform_markup = sim_params.get('platform_markup', 0.1)
        price_range = sim_params.get('price_range', 0.25)
        duration_hours = sim_params.get('duration_hours', 1.0)
    elif hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        market_price = sim_params.get('market_price', 100.0)
        platform_markup = sim_params.get('platform_markup', 0.1)
        price_range = sim_params.get('price_range', 0.25)
        duration_hours = sim_params.get('duration_hours', 1.0)
    elif hasattr(st.session_state, 'sim_params'):
        market_price = getattr(st.session_state.sim_params, 'market_price', 100.0)
        platform_markup = getattr(st.session_state.sim_params, 'platform_markup', 0.1)
        price_range = getattr(st.session_state.sim_params, 'price_range', 0.25)
        duration_hours = getattr(st.session_state.sim_params, 'duration_hours', 1.0)
    
    # Calculate standard prices
    baseline_price = (1 + platform_markup) * market_price
    pn_price = (1 + price_range) * baseline_price
    discount_price = market_price * 0.7
    fixed_price = market_price
    
    # Build vendor lookup
    vendor_lookup = {}
    if vendors_data:
        for vendor in vendors_data:
            vendor_id = vendor.get('vendor_id')
            vendor_lookup[vendor_id] = vendor
    
    # Simulation start time for timestamp conversion
    simulation_start_time = datetime.now()
    
    for idx, row in df.iterrows():
        # Get agent-level data
        agent_id = row.get('agent_id', idx + 1)
        honesty_humility = row.get('Honesty_Humility', np.nan)
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        group_experiment = row.get('Group_experiment', '')
        customer_type = row.get('customer_type', '')
        income_category = row.get('income_category', np.nan)
        agent_donation_default = row.get('donation_default', np.nan)
        
        # Get vendor proximity scores for this agent
        proximity_scores = row.get('vendor_proximity_scores', {})
        if not isinstance(proximity_scores, dict):
            proximity_scores = {}
        
        # Get vendor choice weights for score calculation
        vendor_weights = row.get('vendor_choice_weights', {})
        if not isinstance(vendor_weights, dict):
            vendor_weights = {
                'price': 0.25,
                'quality': 0.25,
                'proximity': 0.25,
                'sustainability': 0.25
            }
        
        # Get purchase requests
        purchase_requests = row.get('purchase_requests', [])
        if not isinstance(purchase_requests, list):
            continue
        
        # Process each purchase request
        for req_idx, request in enumerate(purchase_requests):
            if not isinstance(request, dict):
                continue
            
            # Transaction identification
            request_id = request.get('request_id', req_idx + 1)
            transaction_id = f"A{agent_id}_R{request_id}"
            
            # Timing
            timestamp_hours = request.get('timestamp_hours', np.nan)
            if not pd.isna(timestamp_hours):
                period = int(timestamp_hours // duration_hours) + 1 if timestamp_hours >= 0 else 1
                request_datetime = simulation_start_time + timedelta(hours=float(timestamp_hours))
                purchase_date = request_datetime.date()
                purchase_time = request_datetime.time()
            else:
                period = request.get('period', 1)
                request_datetime = simulation_start_time
                purchase_date = request_datetime.date()
                purchase_time = request_datetime.time()
            
            # Vendor information
            vendor_id = request.get('vendorID', np.nan)
            vendor_price = np.nan
            vendor_quality = np.nan
            vendor_sustainability = np.nan
            vendor_proximity = np.nan
            vendor_integrated_score = np.nan
            
            if not pd.isna(vendor_id) and vendor_id in vendor_lookup:
                vendor = vendor_lookup[vendor_id]
                vendor_price = vendor.get('price', np.nan)
                vendor_quality = vendor.get('quality', np.nan)
                vendor_sustainability = vendor.get('sustainability', np.nan)
                vendor_proximity = proximity_scores.get(str(int(vendor_id)), np.nan)
                
                # Calculate vendor integrated score
                if not pd.isna(vendor_price) and not pd.isna(vendor_quality) and \
                   not pd.isna(vendor_sustainability) and not pd.isna(vendor_proximity):
                    vendor_integrated_score = _calculate_vendor_composite_score(
                        vendor, vendor_weights, vendor_proximity, vendors_data
                    )
            
            # Purchase decision and pricing
            platform_price = request.get('platformPrice', '')
            bid_value = request.get('bid_value', 'N/A')
            
            # Determine purchase request type and customer price
            if platform_price == 'DISCOUNT':
                purchase_request_type = 'Discount'
                customer_price = discount_price
            elif platform_price == 'FIXED':
                purchase_request_type = 'Fixed'
                customer_price = fixed_price
            elif platform_price == 'PN':
                purchase_request_type = 'Purchase Now'
                customer_price = pn_price
            elif platform_price == 'BID':
                purchase_request_type = 'Bid'
                try:
                    customer_price = float(bid_value) if bid_value != 'N/A' else pn_price
                except (ValueError, TypeError):
                    customer_price = pn_price
            else:
                purchase_request_type = customer_type.capitalize() if customer_type else 'Regular'
                customer_price = pn_price
            
            # Display customer price only for PN, show N/A for others
            display_customer_price = customer_price if purchase_request_type == 'Purchase Now' else 'N/A'
            
            # Donation information
            # Priority: request-level > agent-level
            final_donation_rate = request.get('final_donation_rate', agent_donation_default)
            try:
                final_donation_rate = float(final_donation_rate) if not pd.isna(final_donation_rate) else 0.0
            except (ValueError, TypeError):
                final_donation_rate = 0.0
            
            donation_paid = customer_price * final_donation_rate if not pd.isna(customer_price) else np.nan
            total_paid = customer_price + donation_paid if not pd.isna(customer_price) and not pd.isna(donation_paid) else np.nan
            
            # Build transaction record
            transaction_record = {
                # Identification
                'Transaction ID': transaction_id,
                'Agent ID': agent_id,
                'Request ID': request_id,
                
                # Agent traits (for reference)
                'Honesty_Humility': honesty_humility,
                'Assigned Allowance Level': allowance_level,
                'Group_experiment': group_experiment,
                'Customer Type': customer_type.capitalize() if customer_type else '',
                'Income Category': income_category,
                
                # Timing
                'Timestamp (hours)': timestamp_hours,
                'Period': period,
                'Purchase Date': purchase_date,
                'Purchase Time': purchase_time,
                
                # Vendor
                'Vendor ID': f"Vendor {int(vendor_id)}" if not pd.isna(vendor_id) else '',
                'Vendor Price': vendor_price,
                'Vendor Quality': vendor_quality,
                'Vendor Sustainability': vendor_sustainability,
                'Vendor Proximity': vendor_proximity,
                'Vendor Integrated Score': vendor_integrated_score,
                
                # Purchase Decision
                'Platform Price': platform_price,
                'Purchase Request Type': purchase_request_type,
                
                # Pricing
                'Bid Value': bid_value if purchase_request_type == 'Bid' else 'N/A',
                'Customer Price': display_customer_price,  # Only show for PN, N/A for others
                
                # Donation
                'Agent Donation Default': agent_donation_default,
                'Final Donation Rate': final_donation_rate,
                'Donation Paid': donation_paid,
                'Total Paid': total_paid,
            }
            
            transaction_records.append(transaction_record)
    
    # Sort by timestamp
    if transaction_records:
        transaction_records.sort(key=lambda x: (x['Agent ID'], x.get('Timestamp (hours)', 0.0)))
    
    return pd.DataFrame(transaction_records)


def _calculate_vendor_composite_score(vendor, weights, proximity, all_vendors):
    """
    Calculate vendor composite score.
    
    Args:
        vendor: Vendor dict
        weights: Dict of attribute weights
        proximity: Proximity score (0-100)
        all_vendors: List of all vendors
        
    Returns:
        float: Composite score
    """
    price = vendor.get('price', 0)
    quality = vendor.get('quality', 3)
    sustainability = vendor.get('sustainability', 3)
    
    # Normalize price (inverted: lower price = higher score)
    if all_vendors and len(all_vendors) > 0:
        prices = [v.get('price', 0) for v in all_vendors]
        min_price = min(prices)
        max_price = max(prices)
        if max_price > min_price:
            norm_price = 1 - ((price - min_price) / (max_price - min_price))
        else:
            norm_price = 0.5
    else:
        norm_price = 0.5
    
    # Normalize other attributes
    norm_quality = (quality - 1) / 4 if quality >= 1 else 0
    norm_sustainability = (sustainability - 1) / 4 if sustainability >= 1 else 0
    norm_proximity = proximity / 100 if not pd.isna(proximity) else 0
    
    # Calculate weighted score
    score = (
        weights.get('price', 0.25) * norm_price +
        weights.get('quality', 0.25) * norm_quality +
        weights.get('proximity', 0.25) * norm_proximity +
        weights.get('sustainability', 0.25) * norm_sustainability
    )
    
    return score


def render_export_section(df, results_dict=None, using_selected_config=False):
    """Render the export/download section (simplified)"""
    # Remove 'raw', 'index', 'consumption_frequency', 'actual_allowance', 'income', 'customer_type', and 'enriched_requests_count' columns before any processing
    # Use exact column name matching to avoid filtering out 'disclose_income' when we only want to exclude 'income'
    columns_to_exclude = ['raw', 'index', 'consumption_frequency', 'actual_allowance', 'income', 'customer_type', 'enriched_requests_count']
    
    if df is not None:
        df = df[[col for col in df.columns if col not in columns_to_exclude]]
    if results_dict is not None:
        results_dict = {
            key: config_df[[col for col in config_df.columns if col not in columns_to_exclude]]
            for key, config_df in results_dict.items()
        }

    st.subheader("💾 Export Results")
    
    st.markdown("""
    **Two export options are available:**
    - **Agent-Level Excel**: One row per agent with all agent-level decisions and summary statistics
    - **Transaction-Level Excel**: One row per purchase request with detailed transaction information
    """)
    
    # Get vendor data if available
    vendors_data = None
    if hasattr(df, 'attrs') and 'vendors' in df.attrs:
        vendors_data = df.attrs['vendors']
    
    # Get simulation parameters if available
    simulation_params = None
    if hasattr(st.session_state, 'simulation_params'):
        simulation_params = st.session_state.simulation_params
    
    try:
        # Build agent-level and transaction-level DataFrames
        agent_df = _build_agent_level_dataframe(df, vendors_data=vendors_data)
        transaction_df = _build_transaction_level_dataframe(df, vendors_data=vendors_data, simulation_params=simulation_params)
        
        # Create Excel with both sheets
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Agent-level sheet
            agent_df.to_excel(writer, index=False, sheet_name='Agent Level')
            
            # Transaction-level sheet
            transaction_df.to_excel(writer, index=False, sheet_name='Transaction Level')
        
        # Show summary statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Agents", len(agent_df))
            st.caption("Rows in Agent-Level sheet")
        with col2:
            st.metric("Total Transactions", len(transaction_df))
            st.caption("Rows in Transaction-Level sheet")
        
        # Download button for combined Excel
        excel_filename = f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        st.download_button(
            label="📊 Download Complete Excel (Both Levels)",
            data=buffer.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Excel file with two sheets: Agent Level (one row per agent) and Transaction Level (one row per purchase request)"
        )
        
        # Show preview of what's in each sheet
        with st.expander("📋 Preview Agent-Level Data (first 5 rows)"):
            st.dataframe(agent_df.head(), use_container_width=True)
            st.caption(f"**Columns ({len(agent_df.columns)})**: {', '.join(agent_df.columns[:10])}{'...' if len(agent_df.columns) > 10 else ''}")
        
        with st.expander("📋 Preview Transaction-Level Data (first 5 rows)"):
            st.dataframe(transaction_df.head(), use_container_width=True)
            st.caption(f"**Columns ({len(transaction_df.columns)})**: {', '.join(transaction_df.columns[:10])}{'...' if len(transaction_df.columns) > 10 else ''}")
        
    except Exception as e:
        st.error(f"Error creating Excel export: {str(e)}")
        st.caption("⚠️ Please ensure all required data is available. If the problem persists, contact support.")
        
        # Fallback: show raw data
        with st.expander("🔍 View Raw Data (for debugging)"):
            st.dataframe(df, use_container_width=True)

    if st.button("🔄 Clear Results"):
        # Clear all session state to reset the entire application
        keys_to_delete = [key for key in st.session_state.keys()]
        for key in keys_to_delete:
            del st.session_state[key]
        
        # Reinitialize session state with default values
        initialize_session_state()
        
        # Stay on results page to show "no results" message
        st.session_state.page = 'results'
        
        # Force page reload
        st.rerun()
