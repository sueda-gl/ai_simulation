# app/pages/results/visualizations/donation_viz.py
"""
Donation-related visualization functions.
Handles donation_default and final_donation_rate decisions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from io import BytesIO


def _build_donation_transaction_export(df, simulation_config=None):
    """
    Build transaction-level data for all customer types with donation information.
    
    Returns a list of transaction records with fields:
    - Agent ID
    - Assigned Allowance Level
    - Group_experiment
    - Customer Type (Regular, Fixed, Discount)
    - Income Category
    - Purchase Request Type (PN/Bid/Fixed/Discount)
    - Date/Time of Purchase Request
    - Period
    - Customer Price (based on PN price, bid value, Fixed price or Discount price)
    - Transaction Completed (0/1)
    - Default donation rate
    - Final donation rate
    - Donation paid
    - Total paid by customer
    """
    transaction_records = []
    
    if 'purchase_requests' not in df.columns:
        return transaction_records
    
    # Get pricing parameters from session state or use defaults
    market_price = 100.0
    platform_markup = 0.1
    if hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        market_price = sim_params.get('market_price', 100.0)
        platform_markup = sim_params.get('platform_markup', 0.1)
    
    # Calculate standard prices
    baseline_price = (1 + platform_markup) * market_price  # PN price
    discount_price = market_price * 0.7  # Assume 30% discount
    fixed_price = market_price  # Fixed price = market price
    
    for idx, row in df.iterrows():
        # Get agent information
        agent_id = row.get('agent_id', idx + 1)
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        group_experiment = row.get('Group_experiment', '')
        income_category = row.get('income_category', np.nan)
        
        # Get donation rates
        default_donation_rate = row.get('donation_default', np.nan)
        final_donation_rate = row.get('final_donation_rate', default_donation_rate)
        
        # Convert donation rates to numeric
        try:
            default_donation_rate = float(default_donation_rate) if not pd.isna(default_donation_rate) else np.nan
        except (ValueError, TypeError):
            default_donation_rate = np.nan
        
        try:
            final_donation_rate = float(final_donation_rate) if not pd.isna(final_donation_rate) else default_donation_rate
        except (ValueError, TypeError):
            final_donation_rate = default_donation_rate
        
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
            
            # Get timestamp and convert to Period
            timestamp_hours = request.get('timestamp_hours', np.nan)
            if not pd.isna(timestamp_hours):
                # Get periods and duration from session state or use defaults
                periods = 1
                duration_hours = 1.0
                if hasattr(st.session_state, 'simulation_params'):
                    sim_params = st.session_state.simulation_params.get('simulation', {})
                    periods = sim_params.get('periods', 1)
                    duration_hours = sim_params.get('duration_hours', 1.0)
                
                # Calculate period (each period has duration_hours hours)
                period = int(timestamp_hours // duration_hours) + 1 if timestamp_hours >= 0 else 1
                # Format as "Period X, Hour Y"
                hour_in_period = timestamp_hours % duration_hours if timestamp_hours >= 0 else 0
                request_datetime = f"Period {period}, Hour {hour_in_period:.1f}"
            else:
                request_datetime = request.get('requestDateTime', '')
                period = request.get('period', 1)
            
            # Determine Purchase Request Type and Customer Price
            platform_price = request.get('platformPrice', request.get('platform_price', ''))
            bid_value = request.get('bid_value', 'N/A')
            
            if platform_price == 'DISCOUNT' or customer_type.lower() == 'discount':
                purchase_request_type = 'Discount'
                customer_price = discount_price
            elif platform_price == 'FIXED' or customer_type.lower() == 'fixed':
                purchase_request_type = 'Fixed'
                customer_price = fixed_price
            elif platform_price == 'PN':
                purchase_request_type = 'PN'
                customer_price = baseline_price
            elif platform_price == 'BID' and bid_value != 'N/A':
                purchase_request_type = 'Bid'
                try:
                    customer_price = float(bid_value)
                except (ValueError, TypeError):
                    customer_price = baseline_price
            else:
                # Default to PN for regular customers
                purchase_request_type = 'PN' if customer_type.lower() == 'regular' else customer_type_display
                customer_price = baseline_price
            
            # Transaction Completed (0/1)
            transaction_completed = request.get('transaction_completed', 
                                               request.get('transactionCompleted', 1))
            # Ensure it's 0 or 1
            if transaction_completed not in [0, 1]:
                transaction_completed = 1 if transaction_completed else 0
            
            # Calculate donation and total paid
            if not pd.isna(final_donation_rate) and not pd.isna(customer_price):
                donation_paid = customer_price * final_donation_rate
                total_paid = customer_price + donation_paid
            else:
                donation_paid = np.nan
                total_paid = customer_price if not pd.isna(customer_price) else np.nan
            
            # Build record
            record = {
                'Agent ID': agent_id,
                'Assigned Allowance Level': allowance_level,
                'Group_experiment': group_experiment,
                'Customer Type': customer_type_display,
                'Income Category': income_category,
                'Purchase Request Type': purchase_request_type,
                'Date/Time of Purchase Request': request_datetime,
                'Period': period,
                'Customer Price': customer_price,
                'Transaction Completed': transaction_completed,
                'Default Donation Rate': default_donation_rate,
                'Final Donation Rate': final_donation_rate,
                'Donation Paid': donation_paid,
                'Total Paid by Customer': total_paid
            }
            
            transaction_records.append(record)
    
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
                fig = px.histogram(
                    df,
                    x=decision_name,
                    nbins=30,
                    title=f"Distribution of {decision_title}",
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
    has_selected_config = hasattr(st.session_state, 'selected_donation_config')
    
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
            fig = px.histogram(
                df,
                x='donation_default',
                title="Distribution of Donation Rates Across Agents",
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
            if hasattr(st.session_state, 'selected_donation_config'):
                config = st.session_state.selected_donation_config
                st.caption(f"Population: {config['population_mode']}")
                st.caption(f"Income: {config['income_spec_mode']}")
        
        # Transaction-Level Export Section
        st.markdown("---")
        st.markdown("**💾 Transaction-Level Export (All Customer Types)**")
        st.caption("Download detailed transaction data with donation information, organized by period")
        
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
                    st.metric("Total Transactions", f"{len(transactions_df):,}")
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
                    
                    # Additional sheets: One per Period
                    periods = sorted(transactions_df['Period'].dropna().unique())
                    for period_val in periods:
                        period_df = transactions_df[transactions_df['Period'] == period_val]
                        sheet_name = f'Period {int(period_val)}'
                        period_df.to_excel(writer, index=False, sheet_name=sheet_name)
                
                # Download button
                transaction_filename = f"donation_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.download_button(
                    label="📥 Download Transaction-Level Excel",
                    data=buffer_transactions.getvalue(),
                    file_name=transaction_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Downloads transaction-level data with Total sheet + one sheet per period"
                )
                
                # Show preview of data
                with st.expander("📊 Preview Transaction Data"):
                    st.dataframe(transactions_df.head(20), use_container_width=True)
            else:
                st.info("ℹ️ No transaction data found in this simulation")
        
        except Exception as e:
            st.error(f"⚠️ Error creating transaction export: {str(e)}")
    
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

