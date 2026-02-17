# app/pages/results/visualizations/donation_viz.py
"""
Donation-related visualization functions.
Handles donation_default and final_donation_rate decisions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from io import BytesIO
from app.utils.timestamp_utils import TimestampConverter


def _apply_price_formatting_donation(writer, sheet_name: str, df: pd.DataFrame):
    """
    Apply Excel number formatting to price-related columns to display 2 decimal places.
    
    This formats the DISPLAY only - the underlying values retain full precision.
    """
    from openpyxl.styles import numbers
    
    # Define columns that should display with 2 decimal places
    # Note: 'Donation Paid' and 'Total Paid by Customer' columns were removed
    # because actual price is unknown for Fixed/Discount customers
    price_columns = [
        'Customer Price', 'customer_price',
        'Default Donation Rate', 'Final Donation Rate',
        'donation_default', 'final_donation_rate',
        'Honesty_Humility', 'income', 'Income',
    ]
    
    workbook = writer.book
    worksheet = workbook[sheet_name]
    
    # Get column indices for price columns
    for col_idx, col_name in enumerate(df.columns, start=1):
        if col_name in price_columns:
            # Apply number format to entire column (skip header row)
            for row_idx in range(2, len(df) + 2):  # Start from row 2 (after header)
                cell = worksheet.cell(row=row_idx, column=col_idx)
                if isinstance(cell.value, (int, float)) and cell.value is not None:
                    cell.number_format = '0.00'


def _build_donation_transaction_export(df, simulation_config=None):
    """
    Build transaction-level data for all customer types with donation information.
    
    Returns a list of transaction records with fields:
    - Transaction ID
    - Agent ID
    - Honesty_Humility (agent trait)
    - Assigned Allowance Level
    - Study Program (agent trait)
    - Group_experiment
    - TWT+Sospeso [=AW2+AX2]{Periods 1+2} (agent trait)
    - income (agent trait)
    - Customer Type (Regular, Fixed, Discount)
    - Income Category
    - Purchase Request Type (PN/Bid/Fixed/Discount)
    - Purchase Timestamp (DD/MM/YYYY HH:MM format)
    - Period
    - Customer Price (PN/Bid only, N/A for Fixed/Discount since actual price is unknown)
    - Default Donation Rate
    - Final Donation Rate
    
    Note: 'Donation Paid' and 'Total Paid by Customer' columns were removed because
    the actual price is unknown for Fixed/Discount customers, making these calculations
    misleading. The donation rate (percentage) is the meaningful decision output.
    """
    transaction_records = []
    
    if 'purchase_requests' not in df.columns:
        return transaction_records
    
    # Get pricing parameters from session state or use defaults
    market_price = 100.0
    platform_markup = 0.1
    price_range = 0.25
    if hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        market_price = sim_params.get('market_price', 100.0)
        platform_markup = sim_params.get('platform_markup', 0.1)
        price_range = sim_params.get('price_range', 0.25)
    
    # Get vendor data for price lookup (consistent with transaction_viz.py)
    vendors_data = None
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
                vendor_lookup[vendor_id] = vendor
                vendor_lookup[str(vendor_id)] = vendor

    # Calculate standard prices (legacy fallback using market_price)
    baseline_price = (1 + platform_markup) * market_price
    pn_price = (1 + price_range) * baseline_price  # PN price = max bid price
    discount_price = market_price * 0.7  # Assume 30% discount
    fixed_price = market_price  # Fixed price = market price
    
    # Use centralized timestamp converter for consistent handling
    ts_converter = TimestampConverter()
    
    for idx, row in df.iterrows():
        # Get agent information
        agent_id = row.get('agent_id', idx + 1)
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        
        # ====================================================================
        # AGENT TRAITS: Extract standard trait columns (consistent with Disclose Income)
        # ====================================================================
        honesty_humility = row.get('Honesty_Humility', np.nan)
        study_program = row.get('Study Program', np.nan)
        twt_sospeso = row.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', np.nan)
        income_value = row.get('income', np.nan)
        
        # Group_experiment with fallbacks (handle various column naming conventions)
        group_experiment = row.get('Group_experiment', '')
        if group_experiment == '' or pd.isna(group_experiment):
            group_experiment = row.get('group', '')
        if group_experiment == '' or pd.isna(group_experiment):
            group_experiment = row.get('group_experiment', '')
        if pd.isna(group_experiment):
            group_experiment = ''
        # ====================================================================
        
        income_category_raw = row.get('income_category', np.nan)
        # Use 'N/A' for empty/missing income_category (e.g., regular customers who didn't disclose income)
        if pd.isna(income_category_raw) or income_category_raw == '' or income_category_raw is None:
            income_category = 'N/A'
        else:
            income_category = income_category_raw
        
        # Get AGENT-LEVEL donation rates (used as fallback only)
        agent_default_rate = row.get('donation_default', np.nan)
        agent_final_rate = row.get('final_donation_rate', agent_default_rate)
        
        # Convert to numeric for fallback
        try:
            agent_default_rate = float(agent_default_rate) if not pd.isna(agent_default_rate) else 0.10
        except (ValueError, TypeError):
            agent_default_rate = 0.10
        
        try:
            agent_final_rate = float(agent_final_rate) if not pd.isna(agent_final_rate) else agent_default_rate
        except (ValueError, TypeError):
            agent_final_rate = agent_default_rate
        
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
            
            # Get timestamp and convert using centralized utilities
            timestamp_hours = request.get('timestamp_hours', np.nan)
            ts_result = ts_converter.convert(timestamp_hours)
            
            period = ts_result['period']
            request_datetime = ts_result['datetime']
            purchase_date = ts_result['date']
            purchase_time = ts_result['time']
            
            # Determine Purchase Request Type and Customer Price
            platform_price = request.get('platformPrice', request.get('platform_price', ''))
            bid_value = request.get('bid_value', 'N/A')
            transaction_id = request.get('transaction_id')
            
            # Lookup specific vendor price
            vendor_id = request.get('vendorID', request.get('vendor_id'))
            
            # Normalize vendor_id for lookup (handle float 1.0 -> int 1)
            lookup_key = vendor_id
            if isinstance(vendor_id, float) and vendor_id.is_integer():
                lookup_key = int(vendor_id)
                
            vendor_price = None
            if lookup_key is not None:
                if lookup_key in vendor_lookup:
                    vendor_price = vendor_lookup[lookup_key].get('price')
                elif str(lookup_key) in vendor_lookup:
                    vendor_price = vendor_lookup[str(lookup_key)].get('price')
            
            # Recalculate PN price based on actual vendor price if available
            current_pn_price = pn_price  # Default to market-based
            current_fixed_price = fixed_price
            current_discount_price = discount_price
            
            if vendor_price is not None:
                v_baseline = (1 + platform_markup) * vendor_price
                current_pn_price = (1 + price_range) * v_baseline
                current_fixed_price = vendor_price
                current_discount_price = vendor_price * 0.7
            
            if platform_price == 'DISCOUNT' or customer_type.lower() == 'discount':
                purchase_request_type = 'Discount'
                customer_price = current_discount_price
            elif platform_price == 'FIXED' or customer_type.lower() == 'fixed':
                purchase_request_type = 'Fixed'
                customer_price = current_fixed_price
            elif platform_price == 'PN':
                purchase_request_type = 'PN'
                customer_price = current_pn_price
            elif platform_price == 'BID' and bid_value != 'N/A':
                purchase_request_type = 'Bid'
                try:
                    customer_price = float(bid_value)
                except (ValueError, TypeError):
                    customer_price = current_pn_price
            else:
                # Default to PN for regular customers
                purchase_request_type = 'PN' if customer_type.lower() == 'regular' else customer_type_display
                customer_price = current_pn_price
            
            # ====================================================================
            # NEW: Get REQUEST-SPECIFIC donation rate (priority over agent-level)
            # ====================================================================
            # Check if this request has its own donation rate
            request_donation_rate = request.get('final_donation_rate', None)
            
            # Use request-level if available, otherwise fall back to agent-level
            if request_donation_rate is not None:
                try:
                    final_donation_rate = float(request_donation_rate)
                except (ValueError, TypeError):
                    final_donation_rate = agent_final_rate
            else:
                # No request-level rate, use agent-level fallback
                final_donation_rate = agent_final_rate
            
            # Ensure valid range
            final_donation_rate = np.clip(final_donation_rate, 0.0, 1.0) if not pd.isna(final_donation_rate) else agent_final_rate
            # ====================================================================
            
            # Only show price for PN and BID customers (we don't know actual price for Fixed/Discount)
            # Excel formatting will handle 2-decimal display (no rounding of actual values)
            # Use np.nan instead of 'N/A' string to avoid mixed types in the column (Arrow compatibility)
            if purchase_request_type in ['PN', 'Bid']:
                display_customer_price = customer_price
            else:
                display_customer_price = np.nan
            
            # Build record with standardized timestamp column
            # NOTE: Removed 'Donation Paid' and 'Total Paid by Customer' columns because:
            # - For Fixed/Discount customers, we don't know the actual price paid
            # - Donation rate (percentage) is the meaningful decision output
            # - Actual monetary values would be misleading without knowing the true price
            record = {
                'Transaction ID': transaction_id,
                'Agent ID': agent_id,
                'Honesty_Humility': honesty_humility,
                'Assigned Allowance Level': allowance_level,
                'Study Program': study_program,
                'Group_experiment': group_experiment,
                'TWT+Sospeso [=AW2+AX2]{Periods 1+2}': twt_sospeso,
                'income': income_value,
                'Customer Type': customer_type_display,
                'Income Category': income_category,
                'Purchase Request Type': purchase_request_type,
                'Purchase Timestamp': ts_result['formatted'],
                'Period': period,
                'Customer Price': display_customer_price,
                'Default Donation Rate': agent_default_rate,
                'Final Donation Rate': final_donation_rate,
                '_sort_datetime': request_datetime  # Hidden column for sorting
            }
            
            transaction_records.append(record)
    
    # Sort all records by timestamp in chronological order
    if transaction_records:
        transaction_records.sort(key=lambda x: x['_sort_datetime'] if isinstance(x['_sort_datetime'], datetime) else datetime.min)
        
        # Remove the hidden sorting column before returning
        for record in transaction_records:
            record.pop('_sort_datetime', None)
    
    return transaction_records


def render_donation_default(df, decision_name, decision_title, decision_data):
    """Visualization for donation_default - placeholder until specialized view is added"""
    try:
        numeric_data = pd.to_numeric(decision_data, errors='coerce')
        if not numeric_data.isna().all():
            col1, col2, col3, col4 = st.columns([1, 1.1, 1.1, 1.2])
            with col1:
                st.metric("Total Agents", f"{len(decision_data):,}")
            with col2:
                st.metric("Mean", f"{numeric_data.mean():.2%}")
            with col3:
                st.metric("Std Dev", f"{numeric_data.std():.2%}")
            with col4:
                st.metric("Range", f"{numeric_data.min():.2%} - {numeric_data.max():.2%}")
            col_plot, col_stats = st.columns([2, 1])
            with col_plot:
                st.markdown(f"**Distribution of {decision_title}**")
                fig = px.histogram(
                    df,
                    x=decision_name,
                    nbins=30,
                    labels={decision_name: decision_title, 'count': 'Number of Agents'}
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_tickformat='.0%'
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_stats:
                st.markdown("**📈 Statistics**")
                stats = numeric_data.describe()
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
                    'Value': [f"{stats[key]:.2%}" for key in ['mean', 'std', 'min', 'max', '50%', '25%', '75%']]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Add Excel export for donation_default when using custom parameters
            # Check if this is a custom parameters run (not default values)
            is_custom_parameters = (
                hasattr(st.session_state, 'custom_decisions') and 
                'donation_default' in st.session_state.custom_decisions
            )
            
            if is_custom_parameters:
                st.markdown("---")
                st.markdown("**💾 Export Donation Results**")
                
                # Define the specific columns to export
                trait_columns = [
                    'Assigned Allowance Level',
                    'Group_experiment',
                    'Honesty_Humility',
                    'Study Program',
                    'TWT+Sospeso [=AW2+AX2]{Periods 1+2}'
                ]
                
                # Build export dataframe with requested columns
                export_columns = []
                
                # Add Agent ID first
                if 'agent_id' in df.columns:
                    export_columns.append('agent_id')
                
                # Add trait columns that exist in the dataframe
                for col in trait_columns:
                    if col in df.columns:
                        export_columns.append(col)
                
                # Add donation_default column
                if decision_name in df.columns:
                    export_columns.append(decision_name)
                
                if export_columns:
                    export_df = df[export_columns].copy()
                    
                    # Rename agent_id to 'Agent ID' for clarity
                    if 'agent_id' in export_df.columns:
                        export_df = export_df.rename(columns={'agent_id': 'Agent ID'})
                    
                    # Create Excel file
                    try:
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            export_df.to_excel(writer, index=False, sheet_name='Donation Results')
                            # Apply 2-decimal formatting to price/rate columns
                            _apply_price_formatting_donation(writer, 'Donation Results', export_df)
                        
                        col_download, col_info = st.columns([1, 2])
                        
                        with col_download:
                            st.download_button(
                                label="📊 Download Donation Excel",
                                data=buffer.getvalue(),
                                file_name=f"donation_default_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Download donation results with agent traits and donation rates"
                            )
                        
                        with col_info:
                            st.caption(f"📋 Export includes {len(export_df):,} agents with {len(export_columns)} columns")
                            st.caption(f"✅ Fields: Agent ID, traits, and {decision_name}")
                    
                    except ImportError:
                        st.warning("⚠️ Excel export requires openpyxl package")
                    except Exception as e:
                        st.error(f"❌ Error creating Excel file: {str(e)}")
                else:
                    st.warning("⚠️ No valid columns found for export")
        else:
            st.info("Data not numeric; specialized visualization not available yet.")
    except Exception:
        st.info("Unable to render donation_default with placeholder visualization.")


def render_final_donation_rate(df, decision_name, decision_title, decision_data):
    """Visualization for final_donation_rate with 3-case logic for donation configs"""
    
    # CASE 3: Check if a donation configuration has been selected
    from app.pages.decision_execution import get_decision_config
    _donation_config = get_decision_config('donation_default')
    has_selected_config = _donation_config is not None
    
    # CASE 1: Check if exactly one donation config exists (auto-use it)
    is_single_donation_run = (
        hasattr(st.session_state, 'custom_decisions') and 
        st.session_state.custom_decisions == ['donation_default'] and
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) == 0
    )
    
    # If this is a single donation run with exactly one result, treat it as "only config available"
    has_only_one_config = False
    if is_single_donation_run and hasattr(st.session_state, 'simulation_results'):
        results_dict = st.session_state.simulation_results
        if results_dict and len(results_dict) == 1:
            has_only_one_config = True
            only_config_key = list(results_dict.keys())[0]
            only_config_df = results_dict[only_config_key]
    
    # Decision logic: Use distribution if selected config OR only one config available
    use_distribution = (has_selected_config or has_only_one_config) and 'donation_default' in df.columns
    
    if use_distribution:
        # Show the actual donation distribution - distinguish between cases
        if has_selected_config:
            st.success("📊 **Using Distribution from Selected Donation Configuration**")
            st.caption("✅ The final_donation_rate values in your export match the donation_default distribution shown below")
        elif has_only_one_config:
            st.success("📊 **Using Distribution from Only Available Donation Configuration**")
            st.caption("✅ Only one donation configuration was generated - final_donation_rate values match donation_default")
        
        donation_data = df['donation_default']
        
        # Top section: Distribution statistics
        col1, col2, col3, col4 = st.columns([1, 1.1, 1.1, 1])
        
        with col1:
            st.metric("Total Agents", f"{len(donation_data):,}")
        
        with col2:
            st.metric("Mean Rate", f"{donation_data.mean():.2%}")
        
        with col3:
            st.metric("Median Rate", f"{donation_data.median():.2%}")
        
        with col4:
            st.metric("Std Dev", f"{donation_data.std():.2%}")
        
        # Distribution visualization
        st.markdown("---")
        st.markdown("**📊 Donation Rate Distribution:**")
        
        col_hist, col_stats = st.columns([2, 1])
        
        with col_hist:
            # Histogram showing the distribution - match overview chart settings for consistency
            st.markdown("**Distribution of Donation Rates Across Agents**")
            fig = px.histogram(
                df,
                x='donation_default',
                labels={'donation_default': 'Donation Rate', 'count': 'Number of Agents'},
                nbins=30,  # Match overview chart
                marginal="box"  # Match overview chart
            )
            fig.update_layout(
                xaxis_tickformat='.0%',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_stats:
            st.markdown("**📈 Distribution Stats:**")
            st.write(f"• **Min**: {donation_data.min():.2%}")
            st.write(f"• **25th %ile**: {donation_data.quantile(0.25):.2%}")
            st.write(f"• **50th %ile**: {donation_data.quantile(0.50):.2%}")
            st.write(f"• **75th %ile**: {donation_data.quantile(0.75):.2%}")
            st.write(f"• **Max**: {donation_data.max():.2%}")
            st.write(f"• **Range**: {donation_data.max() - donation_data.min():.2%}")
            
            st.markdown("---")
            st.markdown("**ℹ️ Source:**")
            if _donation_config:
                st.caption(f"Population: {_donation_config.get('population_mode', 'Unknown')}")
                st.caption(f"Income: {_donation_config.get('donation_income_mode', _donation_config.get('income_spec_mode', 'Unknown'))}")
    
    else:
        # Fall back to slider if no donation_default data available
        st.info("💡 **No donation configuration selected** - Using simple rate configuration")
        st.caption("Select a donation configuration on Page 2 to see the full distribution")
        
        # Use _default_value key (consistent with Page 2 for numeric defaults)
        slider_key = f"{decision_name}_default_value"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0.10  # 10% as default
        
        # Top section: Current settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Agents", f"{len(decision_data):,}")
        
        with col2:
            # Calculate average of actual final donation rates
            avg_final_rate = pd.to_numeric(decision_data, errors='coerce').mean()
            if pd.isna(avg_final_rate):
                avg_final_rate = st.session_state[slider_key]
            st.metric("Final Donation Rate", f"{avg_final_rate:.2%}")
        
        with col3:
            # Default donation rate with 2 decimal points
            default_rate = 0.10
            st.metric("Default Donation Rate", f"{default_rate:.2%}")
    
    # ========================================================================
    # NEW: Transaction-Level Export (ALWAYS AVAILABLE if purchase_requests exist)
    # ========================================================================
    # This section appears REGARDLESS of whether donation_default was selected
    # It will use request-level rates if available, or agent-level fallback
    
    if 'purchase_requests' in df.columns:
        st.markdown("---")
        st.markdown("**💾 Transaction-Level Export**")
        st.caption("Download detailed purchase request data with donation rates (one row per request)")
        
        try:
            # Build transaction records
            transaction_records = _build_donation_transaction_export(df)
            
            if len(transaction_records) > 0:
                # Create DataFrame from records
                transactions_df = pd.DataFrame(transaction_records)
                
                # Sort by Period and Agent ID
                transactions_df = transactions_df.sort_values(['Period', 'Agent ID'])
                
                # Show summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Requests", f"{len(transactions_df):,}")
                with col2:
                    num_agents = transactions_df['Agent ID'].nunique()
                    st.metric("Total Agents", f"{num_agents:,}")
                with col3:
                    num_periods = transactions_df['Period'].nunique()
                    st.metric("Periods", f"{int(num_periods)}")
                with col4:
                    avg_donation = transactions_df['Final Donation Rate'].mean()
                    if not pd.isna(avg_donation):
                        st.metric("Avg Donation Rate", f"{avg_donation:.2%}")
                    else:
                        st.metric("Avg Donation Rate", "N/A")
                
                # Create Excel with multiple sheets
                buffer_transactions = BytesIO()
                with pd.ExcelWriter(buffer_transactions, engine='openpyxl') as writer:
                    # Sheet 1: Total (all periods combined)
                    transactions_df.to_excel(writer, index=False, sheet_name='Total')
                    # Apply 2-decimal formatting
                    _apply_price_formatting_donation(writer, 'Total', transactions_df)
                    
                    # Additional sheets: One per Period
                    periods = sorted(transactions_df['Period'].dropna().unique())
                    for period_val in periods:
                        period_df = transactions_df[transactions_df['Period'] == period_val]
                        sheet_name = f'Period {int(period_val)}'
                        period_df.to_excel(writer, index=False, sheet_name=sheet_name)
                        # Apply 2-decimal formatting to each period sheet
                        _apply_price_formatting_donation(writer, sheet_name, period_df)
                
                # Download button
                transaction_filename = f"donation_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                st.download_button(
                    label="📥 Download Transaction-Level Excel",
                    data=buffer_transactions.getvalue(),
                    file_name=transaction_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Downloads purchase request-level data with donation rates (Total sheet + one sheet per period)"
                )
                
                # Show preview of data
                with st.expander("📊 Preview Transaction Data"):
                    st.dataframe(transactions_df.head(20), use_container_width=True)
            else:
                st.info("ℹ️ No purchase request data found in this simulation")
        
        except Exception as e:
            st.error(f"⚠️ Error creating transaction export: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

