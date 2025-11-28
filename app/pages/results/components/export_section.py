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
    - Average vendor proximity (not per-vendor)
    
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
        
        # Decision 3: Donation Default (exclude raw/intermediate columns)
        agent_record['donation_default'] = row.get('donation_default', np.nan)
        # NOTE: donation_default_raw_pos is intentionally excluded
        
        # Decision 4: Rejected Transaction Defaults - Split list into 5 priority columns
        rejected_defaults = row.get('rejected_transaction_defaults', '')
        
        # Parse if it's a string representation of a list
        if isinstance(rejected_defaults, str) and rejected_defaults.startswith('['):
            try:
                import ast
                rejected_defaults = ast.literal_eval(rejected_defaults)
            except:
                rejected_defaults = []
        elif not isinstance(rejected_defaults, list):
            rejected_defaults = [rejected_defaults] if rejected_defaults else []
        
        # Create 5 priority columns
        for priority_num in range(1, 6):
            if isinstance(rejected_defaults, list) and len(rejected_defaults) >= priority_num:
                agent_record[f'rejected_transaction_{priority_num}_choice'] = rejected_defaults[priority_num - 1]
            else:
                agent_record[f'rejected_transaction_{priority_num}_choice'] = ''
        
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
        # Income: Try to get from multiple sources
        income = row.get('income', np.nan)
        # If income is NaN or not present, try actual_allowance as fallback
        if pd.isna(income) or income is None:
            income = row.get('actual_allowance', np.nan)
        
        agent_record['income'] = income
        agent_record['income_category'] = row.get('income_category', np.nan)
        
        # Split purchasing_quantity into two columns
        total_requests = row.get('purchasing_quantity', 0)
        agent_record['purchase_requests'] = total_requests  # Count of requests made
        agent_record['completed_transactions'] = 0  # Placeholder for external algorithm
        
        # Decision 7: Purchasing Frequency
        agent_record['purchasing_frequency'] = row.get('purchasing_frequency', np.nan)
        
        # Decision 8: Vendor Selection (agent-level)
        # Note: This represents the highest scored vendor on average, not a fixed choice
        # In reality, vendor selection varies by product/request
        agent_record['highest_scored_vendor_avg'] = row.get('preferred_vendor', np.nan)
        
        # Vendor proximity scores - Calculate AVERAGE instead of per-vendor columns
        proximity_scores = row.get('vendor_proximity_scores', {})
        if isinstance(proximity_scores, dict) and proximity_scores:
            # Calculate average proximity across all vendors
            proximities = [float(v) for v in proximity_scores.values() if not pd.isna(v)]
            agent_record['avg_vendor_proximity'] = np.mean(proximities) if proximities else np.nan
        else:
            agent_record['avg_vendor_proximity'] = np.nan
        
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
    # Get configured price bounds for consistent normalization
    price_min_config = 50.0
    price_max_config = 150.0
    
    if simulation_params:
        sim_params = simulation_params.get('simulation', {})
        market_price = sim_params.get('market_price', 100.0)
        platform_markup = sim_params.get('platform_markup', 0.1)
        price_range = sim_params.get('price_range', 0.25)
        duration_hours = sim_params.get('duration_hours', 1.0)
        price_min_config = sim_params.get('vendor_price_min', 50.0)
        price_max_config = sim_params.get('vendor_price_max', 150.0)
    elif hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        market_price = sim_params.get('market_price', 100.0)
        platform_markup = sim_params.get('platform_markup', 0.1)
        price_range = sim_params.get('price_range', 0.25)
        duration_hours = sim_params.get('duration_hours', 1.0)
        price_min_config = sim_params.get('vendor_price_min', 50.0)
        price_max_config = sim_params.get('vendor_price_max', 150.0)
    elif hasattr(st.session_state, 'sim_params'):
        market_price = getattr(st.session_state.sim_params, 'market_price', 100.0)
        platform_markup = getattr(st.session_state.sim_params, 'platform_markup', 0.1)
        price_min_config = getattr(st.session_state.sim_params, 'vendor_price_min', 50.0)
        price_max_config = getattr(st.session_state.sim_params, 'vendor_price_max', 150.0)
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
            vendor_price_score = np.nan
            vendor_quality_score = np.nan
            vendor_sustainability_score = np.nan
            vendor_proximity_score = np.nan
            vendor_integrated_score = np.nan
            
            if not pd.isna(vendor_id) and vendor_id in vendor_lookup:
                vendor = vendor_lookup[vendor_id]
                vendor_price = vendor.get('price', np.nan)
                vendor_quality = vendor.get('quality', np.nan)
                vendor_sustainability = vendor.get('sustainability', np.nan)
                vendor_proximity = proximity_scores.get(str(int(vendor_id)), np.nan)
                
                # Calculate normalized scores (0-1 scale)
                if not pd.isna(vendor_price) and not pd.isna(vendor_quality) and \
                   not pd.isna(vendor_sustainability) and not pd.isna(vendor_proximity):
                    
                    # Normalize price (inverted: lower price = higher score)
                    # Use FIXED reference bounds from configuration for consistent normalization
                    # This ensures price has the same discriminatory power as other attributes
                    if price_max_config > price_min_config:
                        # Clamp price to configured bounds
                        clamped_price = max(price_min_config, min(vendor_price, price_max_config))
                        vendor_price_score = 1 - ((clamped_price - price_min_config) / (price_max_config - price_min_config))
                    else:
                        vendor_price_score = 0.5
                    
                    # Normalize quality (1-5 scale to 0-1)
                    vendor_quality_score = (vendor_quality - 1) / 4 if vendor_quality >= 1 else 0
                    
                    # Normalize sustainability (1-5 scale to 0-1)
                    vendor_sustainability_score = (vendor_sustainability - 1) / 4 if vendor_sustainability >= 1 else 0
                    
                    # Normalize proximity (0-100 scale to 0-1)
                    vendor_proximity_score = vendor_proximity / 100 if not pd.isna(vendor_proximity) else 0
                    
                    # Calculate integrated score (weighted average)
                    vendor_integrated_score = (
                        vendor_weights.get('price', 0.25) * vendor_price_score +
                        vendor_weights.get('quality', 0.25) * vendor_quality_score +
                        vendor_weights.get('proximity', 0.25) * vendor_proximity_score +
                        vendor_weights.get('sustainability', 0.25) * vendor_sustainability_score
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
                
                # Vendor - All scores normalized to 0-1 scale
                'Vendor ID': f"Vendor {int(vendor_id)}" if not pd.isna(vendor_id) else '',
                'Vendor Price Score': vendor_price_score,
                'Vendor Quality Score': vendor_quality_score,
                'Vendor Sustainability Score': vendor_sustainability_score,
                'Vendor Proximity Score': vendor_proximity_score,
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
            }
            
            transaction_records.append(transaction_record)
    
    # Sort by timestamp
    if transaction_records:
        transaction_records.sort(key=lambda x: (x['Agent ID'], x.get('Timestamp (hours)', 0.0)))
    
    return pd.DataFrame(transaction_records)


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
    
    # Check if this is a donation-only run (special simplified export)
    trait_columns = ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program', 
                     'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    
    is_donation_only_run = (
        hasattr(st.session_state, 'custom_decisions') and 
        st.session_state.custom_decisions == ['donation_default'] and
        hasattr(st.session_state, 'default_decisions') and
        len(st.session_state.default_decisions) == 0
    )
    
    if is_donation_only_run:
        # DONATION-ONLY EXPORT: Simplified version with just donation and traits
        # Filter main df to only include donation columns
        columns_to_keep = [col for col in df.columns 
                          if (col == 'donation_default' or col in trait_columns or col == 'agent_id')]
        df = df[columns_to_keep]
        
        # Filter results_dict to only include donation columns
        if results_dict is not None:
            results_dict = {
                key: config_df[[col for col in config_df.columns 
                               if (col == 'donation_default' or col in trait_columns or col == 'agent_id')]]
                for key, config_df in results_dict.items()
            }
        
        # Check if we have multiple configurations to compare
        export_all_configs = results_dict is not None and len(results_dict) > 1 and not using_selected_config
        
        if export_all_configs:
            st.markdown(f"""
            **Donation Default Results Export (All {len(results_dict)} Configurations):**
            - Agent ID, trait columns, and donation_default rate for each configuration
            - All configurations combined in one file for comparison
            """)
        else:
            st.markdown("""
            **Donation Default Results Export:**
            - Agent ID, trait columns, and donation_default rate
            """)
        
        try:
            buffer = BytesIO()
            
            if export_all_configs:
                # MULTI-CONFIG: Combine all configurations in one Excel for comparison
                first_config_df = next(iter(results_dict.values()))
                available_traits = [col for col in trait_columns if col in first_config_df.columns]
                combined_df = first_config_df[available_traits].copy()
                
                # Add agent_id if it exists
                if 'agent_id' in first_config_df.columns:
                    combined_df['Agent ID'] = first_config_df['agent_id'].values
                
                # Add donation_default from each configuration
                for config_key, config_df in results_dict.items():
                    if not config_df.empty and 'donation_default' in config_df.columns:
                        config_suffix = config_key.replace('_', ' ').title().replace(' ', '_')
                        new_col_name = f"donation_default_{config_suffix}"
                        combined_df[new_col_name] = config_df['donation_default'].values
                
                # Reorder columns to put Agent ID first
                if 'Agent ID' in combined_df.columns:
                    cols = ['Agent ID'] + [col for col in combined_df.columns if col != 'Agent ID']
                    combined_df = combined_df[cols]
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    combined_df.to_excel(writer, index=False, sheet_name='All Configurations')
                
                st.metric("Total Agents", len(combined_df))
                
                excel_label = f"📊 Download Donation Excel (All {len(results_dict)} Configs)"
                excel_filename = f"donation_all_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                st.download_button(
                    label=excel_label,
                    data=buffer.getvalue(),
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Donation results with {len(results_dict)} configurations for comparison"
                )
                
                # Show preview
                with st.expander("📋 Preview Donation Data (first 5 rows)"):
                    st.dataframe(combined_df.head(), use_container_width=True)
                    st.caption(f"**Columns ({len(combined_df.columns)})**: {', '.join(combined_df.columns[:10])}{'...' if len(combined_df.columns) > 10 else ''}")
            
            else:
                # SINGLE CONFIG: Simple export with just one configuration
                df_export = df.copy()
                
                # Rename agent_id to 'Agent ID' for clarity
                if 'agent_id' in df_export.columns:
                    df_export = df_export.rename(columns={'agent_id': 'Agent ID'})
                
                # Reorder columns to put Agent ID first
                if 'Agent ID' in df_export.columns:
                    cols = ['Agent ID'] + [col for col in df_export.columns if col != 'Agent ID']
                    df_export = df_export[cols]
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_export.to_excel(writer, index=False, sheet_name='Donation Results')
                
                st.metric("Total Agents", len(df_export))
                
                excel_filename = f"donation_default_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.download_button(
                    label="📊 Download Donation Default Excel",
                    data=buffer.getvalue(),
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Donation results with agent traits and donation rates"
                )
                
                # Show preview
                with st.expander("📋 Preview Donation Data (first 5 rows)"):
                    st.dataframe(df_export.head(), use_container_width=True)
                    st.caption(f"**Columns ({len(df_export.columns)})**: {', '.join(df_export.columns)}")
        
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")
    
    else:
        # FULL TWO-LEVEL EXPORT: For all other simulations
        st.markdown("""
        **Two separate Excel files are available for download:**
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
            
            # Show summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Agents", len(agent_df))
                st.caption("Rows in Agent-Level file")
            with col2:
                st.metric("Total Transactions", len(transaction_df))
                st.caption("Rows in Transaction-Level file")
            
            # Create two separate Excel files
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Agent-Level Excel
            agent_buffer = BytesIO()
            with pd.ExcelWriter(agent_buffer, engine='openpyxl') as writer:
                agent_df.to_excel(writer, index=False, sheet_name='Agent Level')
            
            # Transaction-Level Excel
            transaction_buffer = BytesIO()
            with pd.ExcelWriter(transaction_buffer, engine='openpyxl') as writer:
                transaction_df.to_excel(writer, index=False, sheet_name='Transaction Level')
            
            # Download buttons for separate files
            st.markdown("### 📥 Download Files")
            col1, col2 = st.columns(2)
            
            with col1:
                agent_filename = f"simulation_agent_level_{timestamp}.xlsx"
                st.download_button(
                    label="📊 Download Agent-Level Excel",
                    data=agent_buffer.getvalue(),
                    file_name=agent_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Agent-level data: {len(agent_df)} agents × {len(agent_df.columns)} columns"
                )
            
            with col2:
                transaction_filename = f"simulation_transaction_level_{timestamp}.xlsx"
                st.download_button(
                    label="📊 Download Transaction-Level Excel",
                    data=transaction_buffer.getvalue(),
                    file_name=transaction_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Transaction-level data: {len(transaction_df)} transactions × {len(transaction_df.columns)} columns"
                )
            
            # Show preview of what's in each file
            st.markdown("### 📋 Data Preview")
            
            with st.expander("👥 Preview Agent-Level Data (first 5 rows)"):
                st.dataframe(agent_df.head(), use_container_width=True)
                st.caption(f"**Columns ({len(agent_df.columns)})**: {', '.join(agent_df.columns[:15])}{'...' if len(agent_df.columns) > 15 else ''}")
            
            with st.expander("🔄 Preview Transaction-Level Data (first 5 rows)"):
                st.dataframe(transaction_df.head(), use_container_width=True)
                st.caption(f"**Columns ({len(transaction_df.columns)})**: {', '.join(transaction_df.columns[:15])}{'...' if len(transaction_df.columns) > 15 else ''}")
            
        except Exception as e:
            st.error(f"Error creating Excel export: {str(e)}")
            st.caption("⚠️ Please ensure all required data is available. If the problem persists, contact support.")
            import traceback
            st.caption(f"Error details: {traceback.format_exc()}")
            
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
