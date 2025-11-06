import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
from app.models import initialize_session_state


def _build_regular_customer_transaction_export(df):
    """
    Build transaction-level data for Regular Customers only.
    
    Returns a list of transaction records with fields:
    - Agent ID
    - Assigned Allowance Level
    - Group_experiment
    - Customer Type (Regular, Fixed, Discount)
    - Income Category
    - Purchase Request Type (PN/Bid)
    - Date/Time of Purchase Request
    - Period
    - Customer Price (based on PN price or bid value)
    - Transaction Completed (0/1)
    """
    transaction_records = []
    
    if 'purchase_requests' not in df.columns:
        return transaction_records
    
    for idx, row in df.iterrows():
        # Get agent information
        agent_id = row.get('agent_id', idx)
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
                customer_type = customer_type.capitalize()
            
            # FILTER: Only include Regular customers
            if customer_type != 'Regular':
                continue
            
            # Get timestamp and convert to Period
            timestamp_hours = request.get('timestamp_hours', np.nan)
            if not pd.isna(timestamp_hours):
                # Convert hours to period (assuming 24 hours per period)
                period = int(timestamp_hours // 24) + 1 if timestamp_hours >= 0 else np.nan
                # Format as "Period X, Hour Y"
                hour_in_period = timestamp_hours % 24 if timestamp_hours >= 0 else 0
                request_datetime = f"Period {period}, Hour {hour_in_period:.1f}"
            else:
                request_datetime = request.get('requestDateTime', '')
                period = request.get('period', np.nan)
            
            # Determine Purchase Request Type (PN or Bid)
            platform_price = request.get('platformPrice', request.get('platform_price', ''))
            bid_value = request.get('bid_value', 'N/A')
            
            if platform_price == 'PN' or (platform_price != 'BID' and bid_value == 'N/A'):
                purchase_request_type = 'PN'
            elif platform_price == 'BID' or (bid_value != 'N/A' and bid_value is not None):
                purchase_request_type = 'Bid'
            else:
                purchase_request_type = 'Unknown'
            
            # Get Customer Price
            # For PN: use platform_price value or vendor price
            # For Bid: use bid_value
            customer_price = request.get('price_paid', 
                                        request.get('customer_paid_price', 
                                        request.get('price', np.nan)))
            
            # If not found, try to determine from bid_value or platformPrice
            if pd.isna(customer_price) or customer_price == '':
                if purchase_request_type == 'Bid' and bid_value != 'N/A':
                    try:
                        customer_price = float(bid_value)
                    except (ValueError, TypeError):
                        customer_price = np.nan
                else:
                    # Try to get vendor price
                    customer_price = request.get('vendor_price', 
                                                request.get('vendorPrice', np.nan))
            
            # Transaction Completed (0/1)
            transaction_completed = request.get('transaction_completed', 
                                               request.get('transactionCompleted', 1))
            # Ensure it's 0 or 1
            if transaction_completed not in [0, 1]:
                transaction_completed = 1 if transaction_completed else 0
            
            # Build record
            record = {
                'Agent ID': agent_id,
                'Assigned Allowance Level': allowance_level,
                'Group_experiment': group_experiment,
                'Customer Type': customer_type,
                'Income Category': income_category,
                'Purchase Request Type': purchase_request_type,
                'Date/Time of Purchase Request': request_datetime,
                'Period': period,
                'Customer Price': customer_price,
                'Transaction Completed': transaction_completed
            }
            
            transaction_records.append(record)
    
    return transaction_records


def render_export_section(df, results_dict=None, using_selected_config=False):
    """Render the export/download section (simplified)"""
    # Remove 'raw', 'index', 'consumption_frequency', 'actual_allowance', 'income', 'customer_type', and 'enriched_requests_count' columns before any processing
    columns_to_exclude = ['raw', 'index', 'consumption_frequency', 'actual_allowance', 'income', 'customer_type', 'enriched_requests_count']
    
    if df is not None:
        df = df[[col for col in df.columns if not any(excl in col.lower() for excl in columns_to_exclude)]]
    if results_dict is not None:
        results_dict = {
            key: config_df[[col for col in config_df.columns if not any(excl in col.lower() for excl in columns_to_exclude)]]
            for key, config_df in results_dict.items()
        }

    st.subheader("💾 Export Results")

    trait_columns = ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program', 
                     'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    
    is_donation_only_run = (
        hasattr(st.session_state, 'custom_decisions') and 
        st.session_state.custom_decisions == ['donation_default'] and
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) == 0
    )
    
    if is_donation_only_run:
        columns_to_keep = [col for col in df.columns if 'donation' in col.lower() or col in trait_columns or col == 'agent_id']
        df = df[columns_to_keep]
        if results_dict:
            results_dict = {
                key: config_df[[col for col in config_df.columns if 'donation' in col.lower() or col in trait_columns or col == 'agent_id']]
                for key, config_df in results_dict.items()
            }

    try:
        buffer = BytesIO()
        export_all_configs = results_dict is not None and len(results_dict) > 1 and not using_selected_config

        if export_all_configs:
            first_config_df = next(iter(results_dict.values()))
            available_traits = [col for col in trait_columns if col in first_config_df.columns]
            combined_df = first_config_df[available_traits].copy()
            
            # Add agent_id if it exists
            if 'agent_id' in first_config_df.columns:
                combined_df['Agent ID'] = first_config_df['agent_id'].values

            green_columns = []
            
            if not is_donation_only_run:
                decision_cols_first = [col for col in first_config_df.columns if col not in trait_columns and col != 'agent_id']
                for col in decision_cols_first:
                    if 'donation_default' not in col:
                        combined_df[col] = first_config_df[col].values
            
            for config_key, config_df in results_dict.items():
                if not config_df.empty:
                    decision_cols = [col for col in config_df.columns if col not in trait_columns and col != 'agent_id']
                    for col in decision_cols:
                        if 'donation_default' in col:
                            config_suffix = config_key.replace('_', ' ').title().replace(' ', '_')
                            new_col_name = f"{col}_{config_suffix}"
                            combined_df[new_col_name] = config_df[col].values
                            green_columns.append(new_col_name)
            
            # Reorder columns to put Agent ID first
            if 'Agent ID' in combined_df.columns:
                cols = ['Agent ID'] + [col for col in combined_df.columns if col != 'Agent ID']
                combined_df = combined_df[cols]

            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                combined_df.to_excel(writer, index=False, sheet_name='All Configurations')
                from openpyxl.styles import PatternFill
                worksheet = writer.sheets['All Configurations']
                green_fill = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')
                header_row = list(combined_df.columns)
                for col_name in green_columns:
                    if col_name in header_row:
                        col_idx = header_row.index(col_name) + 1
                        for row_idx in range(1, len(combined_df) + 2):
                            worksheet.cell(row=row_idx, column=col_idx).fill = green_fill
            
            excel_label = f"📊 Download Excel (All {len(results_dict)} Configs)"
            excel_filename = f"enhanced_simulation_all_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        else:
            df_export = df.copy()
            # Rename agent_id to 'Agent ID' for clarity
            if 'agent_id' in df_export.columns:
                df_export = df_export.rename(columns={'agent_id': 'Agent ID'})
            
            # Reorder columns to put Agent ID first
            if 'Agent ID' in df_export.columns:
                cols = ['Agent ID'] + [col for col in df_export.columns if col != 'Agent ID']
                df_export = df_export[cols]
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Results')
            excel_label = "📊 Download Excel"
            excel_filename = f"enhanced_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        st.download_button(
            label=excel_label,
            data=buffer.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except ImportError:
        st.caption("⚠️ Excel export requires openpyxl")
    
    # Transaction-Level Export for Regular Customers
    st.markdown("---")
    st.markdown("**📋 Transaction-Level Export (Regular Customers)**")
    st.caption("Download detailed transaction data for Regular customers, organized by period")
    
    try:
        # Build transaction records
        transaction_records = _build_regular_customer_transaction_export(df)
        
        if len(transaction_records) > 0:
            # Create DataFrame from records
            transactions_df = pd.DataFrame(transaction_records)
            
            # Sort by Period and Agent ID
            transactions_df = transactions_df.sort_values(['Period', 'Agent ID'])
            
            # Show summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", f"{len(transactions_df):,}")
            with col2:
                num_agents = transactions_df['Agent ID'].nunique()
                st.metric("Regular Customers", f"{num_agents:,}")
            with col3:
                num_periods = transactions_df['Period'].nunique()
                st.metric("Periods", f"{int(num_periods)}")
            
            # Create Excel with multiple sheets
            buffer_transactions = BytesIO()
            with pd.ExcelWriter(buffer_transactions, engine='openpyxl') as writer:
                # Sheet 1: Total (all periods combined)
                transactions_df.to_excel(writer, index=False, sheet_name='Total')
                
                # Additional sheets: One per Period
                periods = sorted(transactions_df['Period'].dropna().unique())
                for period in periods:
                    period_df = transactions_df[transactions_df['Period'] == period]
                    sheet_name = f'Period {int(period)}'
                    period_df.to_excel(writer, index=False, sheet_name=sheet_name)
            
            # Download button
            transaction_filename = f"regular_customers_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            st.download_button(
                label="📥 Download Regular Customer Transactions Excel",
                data=buffer_transactions.getvalue(),
                file_name=transaction_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Downloads transaction-level data with Total sheet + one sheet per period"
            )
        else:
            st.info("ℹ️ No Regular customer transactions found in this simulation")
    
    except Exception as e:
        st.error(f"⚠️ Error creating transaction export: {str(e)}")

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
