# app/pages/results/visualizations/vendor_viz.py
"""
Vendor-related visualization functions.
Handles vendor_choice_weights and vendor_selection decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from io import BytesIO
import numpy as np
from app.utils.timestamp_utils import TimestampConverter
from src.vendor_attribute_generator import calculate_vendor_score_with_breakdown


def _apply_price_formatting_vendor(writer, sheet_name: str, df: pd.DataFrame):
    """
    Apply Excel number formatting to price-related columns to display 2 decimal places.
    """
    price_columns = [
        'Vendor Price', 'price', 'Customer Paid Price', 'Customer Price',
        'Bid Value', 'bid_value', 'Vendor Score', 'Integrated Score',
        'Price Score', 'Quality Score', 'Sustainability Score', 'Proximity Score',
        'Price (Normalized)', 'Quality (Normalized)', 'Sustainability (Normalized)', 'Proximity (Normalized)',
        'Weight_Price', 'Weight_Quality', 'Weight_Proximity', 'Weight_Sustainability',
        'weight_price', 'weight_quality', 'weight_proximity', 'weight_sustainability',
        'Proximity', 'Quality', 'Sustainability',
    ]
    
    workbook = writer.book
    worksheet = workbook[sheet_name]
    
    for col_idx, col_name in enumerate(df.columns, start=1):
        if col_name in price_columns:
            for row_idx in range(2, len(df) + 2):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                if isinstance(cell.value, (int, float)) and cell.value is not None:
                    cell.number_format = '0.00'


def _build_purchase_request_export(df, vendors_data, price_min_config=None, price_max_config=None):
    """
    Build purchase request-level export data from simulation results.
    
    Args:
        df: DataFrame with simulation results
        vendors_data: List of vendor dictionaries (or None if not available)
        price_min_config: Configured minimum price bound (from vendor_price_min)
        price_max_config: Configured maximum price bound (from vendor_price_max)
    
    Returns:
        List of dicts with purchase request level data
    """
    purchase_request_records = []
    
    # Check if we have purchase_requests column
    if 'purchase_requests' not in df.columns:
        return []
    
    # Use centralized timestamp converter for consistent handling
    ts_converter = TimestampConverter()
    
    # Build vendor lookup dictionary for quick access
    vendor_lookup = {}
    if vendors_data:
        for vendor in vendors_data:
            vendor_id = vendor.get('vendor_id')
            vendor_lookup[vendor_id] = vendor
    
    # Get pricing parameters from session state for customer price calculation
    platform_markup = 0.1
    price_range = 0.25
    if hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        platform_markup = sim_params.get('platform_markup', 0.1)
        price_range = sim_params.get('price_range', 0.25)
    elif hasattr(st.session_state, 'sim_params'):
        platform_markup = getattr(st.session_state.sim_params, 'platform_markup', 0.1)
        price_range = getattr(st.session_state.sim_params, 'price_range', 0.25)
    
    # Iterate through each agent
    for idx, row in df.iterrows():
        # Get agent-level data
        agent_id = row.get('agent_id', idx + 1)
        allowance_level = row.get('Assigned Allowance Level', np.nan)
        group_experiment = row.get('Group_experiment', np.nan)
        
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
        
        # Get purchase requests for this agent
        purchase_requests = row.get('purchase_requests', [])
        if not isinstance(purchase_requests, list):
            continue
        
        # Process each purchase request
        for req_idx, request in enumerate(purchase_requests):
            if not isinstance(request, dict):
                continue
            
            # Extract request data
            # Use global transaction_id assigned by simulation.py (snake_case), with fallback
            transaction_id = request.get('transaction_id', request.get('transactionID', request.get('request_id', f"T{agent_id}_{req_idx+1}")))
            vendor_id = request.get('vendorID', np.nan)
            
            # Get timestamp and convert using centralized utilities
            timestamp_hours = request.get('timestamp_hours', np.nan)
            ts_result = ts_converter.convert(timestamp_hours)
            
            period = ts_result['period']
            request_datetime = ts_result['datetime']
            
            # Determine customer type from request or agent
            customer_type = request.get('customer_type', request.get('customerType', 'Regular'))
            # Capitalize first letter
            if isinstance(customer_type, str):
                customer_type = customer_type.capitalize()
            
            # Get vendor attributes
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
                
                # Get proximity for this agent-vendor pair
                vendor_proximity = proximity_scores.get(str(int(vendor_id)), np.nan)
                
                # Calculate vendor integrated score
                if not pd.isna(vendor_price) and not pd.isna(vendor_quality) and \
                   not pd.isna(vendor_sustainability) and not pd.isna(vendor_proximity):
                    vendor_integrated_score = _calculate_vendor_score(
                        vendor, vendor_weights, vendor_proximity, vendors_data,
                        price_min_config=price_min_config,
                        price_max_config=price_max_config
                    )
            
            # Get platform price type for this request
            platform_price = request.get('platformPrice', request.get('platform_price', ''))
            bid_value = request.get('bid_value', 'N/A')
            
            # Calculate customer paid price based on vendor's actual price and pricing formula
            # Formula: Customer Price (PN) = (1 + price_range) × (1 + platform_markup) × vendor_price
            customer_paid_price = np.nan
            if not pd.isna(vendor_price):
                if platform_price == 'PN':
                    # PN price: apply both platform markup and price range
                    baseline_price = (1 + platform_markup) * vendor_price
                    customer_paid_price = (1 + price_range) * baseline_price
                elif platform_price == 'BID' and bid_value != 'N/A':
                    # BID: customer pays their bid value
                    try:
                        customer_paid_price = float(bid_value)
                    except (ValueError, TypeError):
                        customer_paid_price = np.nan
            
            # Format for display - show 2 decimal places for both PN and BID
            # Use np.nan instead of 'N/A' for Arrow compatibility in dataframe display
            if (platform_price == 'PN' or platform_price == 'BID') and not pd.isna(customer_paid_price):
                display_customer_paid_price = float(f"{customer_paid_price:.2f}")
            else:
                display_customer_paid_price = np.nan
            
            # Determine Purchase Type (PN, Bid, Fixed, Discount)
            if platform_price == 'PN':
                purchase_type = 'PN'
            elif platform_price == 'BID':
                purchase_type = 'Bid'
            elif platform_price == 'FIXED' or customer_type.lower() == 'fixed':
                purchase_type = 'Fixed'
            elif platform_price == 'DISCOUNT' or customer_type.lower() == 'discount':
                purchase_type = 'Discount'
            else:
                # Fallback based on customer type
                purchase_type = customer_type if customer_type else 'Unknown'
            
            # Build record (include hidden sort key)
            record = {
                'Purchase Request ID': transaction_id,  # Will be reassigned after sorting
                'Agent ID': agent_id,
                'Assigned Allowance Level': allowance_level,
                'Group_experiment': group_experiment,
                'Customer Type': customer_type,
                'Purchase Type': purchase_type,
                'Purchase Timestamp': ts_result['formatted'],
                'Period': period,
                'Selected Vendor': f"Vendor {int(vendor_id)}" if not pd.isna(vendor_id) else np.nan,
                'Vendor Price': vendor_price,
                'Quality': vendor_quality,
                'Sustainability': vendor_sustainability,
                'Proximity': vendor_proximity,
                'Vendor Integrated Score': vendor_integrated_score,
                'Customer Paid Price': display_customer_paid_price,
                '_sort_datetime': request_datetime  # Hidden sort key
            }
            
            purchase_request_records.append(record)
    
    # Sort records by timestamp in chronological order
    if purchase_request_records:
        purchase_request_records.sort(key=lambda x: x.get('_sort_datetime', datetime.min))
        
        # Assign unique Purchase Request IDs (1, 2, 3, ...) based on chronological order
        # This ensures each request has a unique ID that reflects its position in the timeline
        for idx, record in enumerate(purchase_request_records):
            record['Purchase Request ID'] = idx + 1
            record.pop('_sort_datetime', None)  # Remove temporary sorting column
    
    return purchase_request_records


def _calculate_vendor_score(vendor, weights, proximity, all_vendors, 
                            price_min_config=None, price_max_config=None):
    """
    Calculate vendor integrated composite score.
    
    This is a thin wrapper around the centralized calculate_vendor_score_with_breakdown()
    function from vendor_attribute_generator.py. All scoring logic is maintained in one place.
    
    Args:
        vendor: Vendor dict with attributes
        weights: Dict of weights for each attribute
        proximity: Proximity score for this agent-vendor pair
        all_vendors: List of all vendors (for fallback price normalization)
        price_min_config: Configured minimum price bound (from vendor_price_min)
        price_max_config: Configured maximum price bound (from vendor_price_max)
    
    Returns:
        float: Composite score
    """
    result = calculate_vendor_score_with_breakdown(
        vendor=vendor,
        weights=weights,
        proximity=proximity,
        all_vendors=all_vendors,
        price_min_config=price_min_config,
        price_max_config=price_max_config
    )
    return result['integrated_score']


def render_vendor_choice_weights(df, decision_name, decision_title, decision_data):
    """Visualization for vendor_choice_weights with interactive parameter selection"""
    
    # Define the 4 vendor choice parameters
    parameters = [
        ("price", "Price", "the product price offered to the customer"),
        ("quality", "Quality", "product quality based on customer ratings"),
        ("proximity", "Proximity", "the proximity of vendor to customer"),
        ("sustainability", "Sustainability", "vendor sustainability rating")
    ]
    
    param_names = {param[0]: param[1] for param in parameters}
    param_descriptions = {param[0]: param[2] for param in parameters}
    
    # Use _default_ key (same as Page 2 Overview tab) for consistency
    selection_key = f"{decision_name}_default_params"
    
    # Initialize if not exists (try to infer from actual results)
    if selection_key not in st.session_state:
        # Try to infer selection from the actual weights in the data
        if not decision_data.empty and isinstance(decision_data.iloc[0], dict):
            # The data contains weight dictionaries
            sample_weights = decision_data.iloc[0]
            # Find which parameters have non-zero weights
            inferred_selection = [key for key, weight in sample_weights.items() if weight > 0]
            if inferred_selection:
                st.session_state[selection_key] = inferred_selection
            else:
                # Default to all if no inference possible
                st.session_state[selection_key] = ["price", "quality", "proximity", "sustainability"]
        else:
            # Default to all if data format doesn't match
            st.session_state[selection_key] = ["price", "quality", "proximity", "sustainability"]
    
    # Top section: Current results display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        # Show number of selected parameters
        num_selected = len(st.session_state[selection_key])
        st.metric("Active Parameters", f"{num_selected}/4")
    
    with col3:
        # Show current weight per parameter
        if num_selected > 0:
            weight_per_param = 100 / num_selected
            st.metric("Weight Each", f"{weight_per_param:.1f}%")
        else:
            st.metric("Weight Each", "0%")
    
    with col4:
        # Show current configuration
        if num_selected == 4:
            st.metric("Configuration", "All Factors")
        elif num_selected == 1:
            st.metric("Configuration", "Single Factor")
        else:
            st.metric("Configuration", f"{num_selected} Factors")
    
    # Main configuration section
    st.markdown("---")
    st.markdown("**🎛️ Configure Vendor Choice Parameters:**")
    
    col_selection, col_viz = st.columns([1, 1])
    
    with col_selection:
        st.markdown("**⚙️ Selected Parameters (Read-Only):**")
        
        # Get selected parameters from session state
        selected_params = st.session_state.get(selection_key, [])
        
        # Display active parameters
        if len(selected_params) > 0:
            st.success(f"✅ **Active Parameters:**")
            for param_key in selected_params:
                st.write(f"• {param_names[param_key]} - {param_descriptions[param_key]}")
        else:
            st.warning("⚠️ No parameters selected")
        
        # Show excluded parameters if any
        excluded_params = [param for param, _, _ in parameters if param not in selected_params]
        if excluded_params:
            st.markdown("**Excluded:**")
            for param_key in excluded_params:
                st.caption(f"• {param_names[param_key]}")
        
        # Calculate and display weights
        if len(selected_params) > 0:
            weight_per_param = 1.0 / len(selected_params)
            
            st.markdown("**📊 Calculated Weights:**")
            
            # Show weight distribution
            weight_data = []
            for param_key in selected_params:
                weight_data.append({
                    'Parameter': param_names[param_key],
                    'Weight': f"{weight_per_param:.1%}",
                    'Decimal': f"{weight_per_param:.3f}"
                })
            
            if weight_data:
                weight_df = pd.DataFrame(weight_data)
                st.dataframe(weight_df, use_container_width=True, hide_index=True)
        
        # Show helpful message
        st.caption("💡 To modify these settings: Go to **Page 2 → Overview Tab**")
    
    with col_viz:
        # Show current weights visualization
        if len(selected_params) > 0:
            # Create pie chart showing weight distribution
            weight_per_param = 1.0 / len(selected_params)
            
            st.markdown("### Vendor Choice Weight Distribution")
            fig = px.pie(
                values=[weight_per_param] * len(selected_params),
                names=[param_names[param] for param in selected_params],
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True, key="vendor_choice_weights_chart", config={'displayModeBar': True, 'displaylogo': False})
            
            # Show summary
            st.markdown("**📋 Weight Summary:**")
            summary_text = []
            for param_key in selected_params:
                summary_text.append(f"• {param_names[param_key]}: {weight_per_param:.1%}")
            
            if len(selected_params) < 4:
                summary_text.append("")
                summary_text.append("**Excluded:**")
                for param_key, param_name, _ in parameters:
                    if param_key not in selected_params:
                        summary_text.append(f"• {param_name}: 0%")
            
            st.markdown("\n".join(summary_text))
        else:
            st.info("Select parameters to see weight distribution")
    
    # Excel Export Section
    st.markdown("---")
    st.markdown("**💾 Export Vendor Choice Weights**")
    
    # Build export dataframe
    export_data = []
    
    for idx, row in df.iterrows():
        # Start with basic info
        row_data = {}
        
        # Add Agent ID
        if 'agent_id' in df.columns:
            row_data['Agent ID'] = row['agent_id']
        
        # Add Assigned Allowance Level
        if 'Assigned Allowance Level' in df.columns:
            row_data['Assigned Allowance Level'] = row['Assigned Allowance Level']
        
        # Add Group_experiment
        if 'Group_experiment' in df.columns:
            row_data['Group_experiment'] = row['Group_experiment']
        
        # Extract weights from the decision_data (which is a dict)
        weights = decision_data.iloc[idx]
        
        if isinstance(weights, dict):
            # Add each weight as a numeric value (e.g., 0.25 instead of "25%")
            row_data['Price'] = weights.get('price', 0.0)
            row_data['Quality'] = weights.get('quality', 0.0)
            row_data['Proximity'] = weights.get('proximity', 0.0)
            row_data['Sustainability'] = weights.get('sustainability', 0.0)
        else:
            # Fallback if weights aren't in expected format
            row_data['Price'] = 0.0
            row_data['Quality'] = 0.0
            row_data['Proximity'] = 0.0
            row_data['Sustainability'] = 0.0
        
        export_data.append(row_data)
    
    if export_data:
        export_df = pd.DataFrame(export_data)
        
        # Create Excel file
        try:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Vendor Choice Weights')
                # Apply 2-decimal formatting
                _apply_price_formatting_vendor(writer, 'Vendor Choice Weights', export_df)
            
            col_download, col_info = st.columns([1, 2])
            
            with col_download:
                st.download_button(
                    label="📊 Download Vendor Weights Excel",
                    data=buffer.getvalue(),
                    file_name=f"vendor_choice_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download vendor choice weights with agent info and weight percentages (numeric format)"
                )
            
            with col_info:
                st.caption(f"📋 Export includes {len(export_df):,} agents with {len(export_df.columns)} columns")
                st.caption(f"✅ Fields: Agent ID, Assigned Allowance Level, Group_experiment, Price, Quality, Proximity, Sustainability")
        
        except ImportError:
            st.warning("⚠️ Excel export requires openpyxl package")
        except Exception as e:
            st.error(f"❌ Error creating Excel file: {str(e)}")
    else:
        st.warning("⚠️ No data available for export")


def render_vendor_selection(df, decision_name, decision_title, decision_data):
    """Visualization for vendor_selection - shows vendor distribution and selection logic"""
    
    # Get vendor data to determine total vendors available
    vendors_data = None
    total_vendors_available = 0
    
    if hasattr(st.session_state, 'vendors') and st.session_state.vendors:
        vendors_data = st.session_state.vendors
        total_vendors_available = len(vendors_data)
    elif 'simulation_results' in st.session_state:
        results = st.session_state.simulation_results
        if isinstance(results, dict):
            vendors_data = results.get('vendors') or results.get('config', {}).get('vendors')
            if vendors_data:
                total_vendors_available = len(vendors_data)
    
    # Get configured price bounds for consistent normalization
    # (Used in vendor score calculations throughout this function)
    price_min_config = None
    price_max_config = None
    if hasattr(st.session_state, 'sim_params'):
        price_min_config = getattr(st.session_state.sim_params, 'vendor_price_min', 50.0)
        price_max_config = getattr(st.session_state.sim_params, 'vendor_price_max', 150.0)
    elif hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        price_min_config = sim_params.get('vendor_price_min', 50.0)
        price_max_config = sim_params.get('vendor_price_max', 150.0)
    
    # Count unique vendors selected (excluding NaN)
    vendor_counts = decision_data.dropna().value_counts()
    num_vendors_selected = len(vendor_counts)
    
    # Calculate purchase request and transaction shares per vendor
    total_purchase_requests = 0
    total_transactions_completed = 0
    vendor_pr_counts = {}  # Purchase requests per vendor
    vendor_tx_counts = {}  # Transactions per vendor
    
    if 'purchase_requests' in df.columns:
        for idx, row in df.iterrows():
            requests = row.get('purchase_requests', [])
            if isinstance(requests, list):
                total_purchase_requests += len(requests)
                # Count completed transactions per vendor
                for req in requests:
                    if isinstance(req, dict):
                        vendor_id = req.get('vendorID')
                        if not pd.isna(vendor_id):
                            # Count purchase request per vendor
                            vendor_pr_counts[vendor_id] = vendor_pr_counts.get(vendor_id, 0) + 1
                            
                            # Count transaction if completed
                            completed = req.get('transactionCompleted', req.get('completed', req.get('transaction_completed', True)))
                            if completed or completed == 1:
                                total_transactions_completed += 1
                                vendor_tx_counts[vendor_id] = vendor_tx_counts.get(vendor_id, 0) + 1
    
    # Calculate ACTUAL dominant share (maximum share held by any single vendor)
    agents_with_selection = decision_data.notna().sum()
    
    # Find dominant vendor for agents
    if len(vendor_counts) > 0 and agents_with_selection > 0:
        max_agent_count = vendor_counts.max()
        max_agent_share = (max_agent_count / agents_with_selection) * 100
        dominant_agent_vendor = vendor_counts.idxmax()
    else:
        max_agent_share = 0
        dominant_agent_vendor = None
    
    # Find dominant vendor for purchase requests
    if vendor_pr_counts and total_purchase_requests > 0:
        max_pr_count = max(vendor_pr_counts.values())
        max_pr_share = (max_pr_count / total_purchase_requests) * 100
    else:
        max_pr_share = 0
    
    # Find dominant vendor for transactions
    if vendor_tx_counts and total_transactions_completed > 0:
        max_tx_count = max(vendor_tx_counts.values())
        max_tx_share = (max_tx_count / total_transactions_completed) * 100
    else:
        max_tx_share = 0
    
    # Overview metrics - 6 columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        st.metric("Vendors Available", f"{total_vendors_available}", 
                 help="Total number of vendors configured in the simulation")
    
    with col3:
        st.metric("Vendors Selected", f"{num_vendors_selected}",
                 help="Number of vendors that were actually chosen by at least one agent")
    
    with col4:
        # Show actual dominant share instead of theoretical average
        dominant_label = f"Vendor {int(dominant_agent_vendor)}" if dominant_agent_vendor is not None else "N/A"
        st.metric("Max Agent Share", f"{max_agent_share:.1f}%",
                 help=f"Highest share of agents selecting any single vendor ({dominant_label})")
    
    with col5:
        st.metric("Max Request Share", f"{max_pr_share:.1f}%",
                 help="Highest share of purchase requests going to any single vendor")
    
    with col6:
        st.metric("Max Transaction Share", f"{max_tx_share:.1f}%",
                 help="Highest share of completed transactions at any single vendor")
    
    # Check if only 1 vendor exists
    if num_vendors_selected == 1 and len(vendor_counts) == 1:
        st.info(f"""
        ℹ️ **Single Vendor Simulation**: Only 1 vendor was configured on Page 1, so all agents select that vendor.
        
        💡 **To see vendor selection in action**: 
        1. Go to **Page 1 → Market & Vendor Configuration**
        2. Change **Number of Vendors (N)** from 1 to 3 or 5
        3. Re-run the simulation
        
        With multiple vendors, agents will select different vendors based on weighted composite scores.
        """)
    
    # Vendor distribution visualization
    st.markdown("---")
    st.markdown("**📊 Vendor Selection Distribution:**")
    
    # Check if we have any data to show (selections or configured vendors)
    has_vendor_data = len(vendor_counts) > 0 or (vendors_data and len(vendors_data) > 0)
    
    if has_vendor_data:
        # Sort vendor_counts by vendor ID (index) instead of by count
        if len(vendor_counts) > 0:
            vendor_counts_sorted = vendor_counts.sort_index()
        
        # Count purchase requests and transactions per vendor
        vendor_purchase_requests = {}
        vendor_transactions = {}
        
        if 'purchase_requests' in df.columns:
            for idx, row in df.iterrows():
                requests = row.get('purchase_requests', [])
                if isinstance(requests, list):
                    for req in requests:
                        if isinstance(req, dict):
                            vendor_id = req.get('vendorID')
                            if not pd.isna(vendor_id):
                                # Count purchase request
                                vendor_purchase_requests[vendor_id] = vendor_purchase_requests.get(vendor_id, 0) + 1
                                
                                # Count transaction if completed
                                completed = req.get('transactionCompleted', req.get('completed', req.get('transaction_completed', True)))
                                if completed or completed == 1:
                                    vendor_transactions[vendor_id] = vendor_transactions.get(vendor_id, 0) + 1
        
        # Calculate totals for percentages
        total_vendor_purchase_requests = sum(vendor_purchase_requests.values()) if vendor_purchase_requests else 0
        total_vendor_transactions = sum(vendor_transactions.values()) if vendor_transactions else 0
        
        # Get all relevant vendor IDs
        all_vendor_ids = set()
        if vendors_data:
            all_vendor_ids = {int(v.get('vendor_id')) for v in vendors_data}
        
        # Add any vendors that were selected (in case configuration doesn't match results)
        if len(vendor_counts) > 0:
            all_vendor_ids.update([int(vid) for vid in vendor_counts.index])
        
        sorted_vendor_ids = sorted(list(all_vendor_ids))
        
        # Bar chart showing vendor distribution
        if len(vendor_counts) > 0:
            st.markdown("**Number of Agents Selecting Each Vendor**")
            
            # Prepare data for chart (include 0s for unselected vendors)
            chart_x = []
            chart_y = []
            
            for vid in sorted_vendor_ids:
                chart_x.append(f"Vendor {vid}")
                
                # Get count
                count = 0
                if vid in vendor_counts.index:
                    count = vendor_counts[vid]
                elif float(vid) in vendor_counts.index:
                    count = vendor_counts[float(vid)]
                chart_y.append(count)
                
            fig = px.bar(
                x=chart_x,
                y=chart_y,
                labels={'x': 'Vendor', 'y': 'Number of Agents'}
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Vendor",
                yaxis_title="Number of Agents"
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        
        # Selection breakdown table below the graph
        st.markdown("**📈 Selection Breakdown:**")
        
        # Build detailed breakdown data
        breakdown_data = []
        for vid in sorted_vendor_ids:
            # Get agent count
            agent_count = 0
            if vid in vendor_counts.index:
                agent_count = vendor_counts[vid]
            elif float(vid) in vendor_counts.index:
                agent_count = vendor_counts[float(vid)]
            
            # Get request count
            pr_count = vendor_purchase_requests.get(vid, 0)
            if pr_count == 0:
                pr_count = vendor_purchase_requests.get(float(vid), 0)
                
            # Get transaction count
            tx_count = vendor_transactions.get(vid, 0)
            if tx_count == 0:
                tx_count = vendor_transactions.get(float(vid), 0)
            
            # Calculate completion rate for this vendor (% of requests completed)
            completion_rate = f"{(tx_count/pr_count)*100:.1f}%" if pr_count > 0 else "0.0%"
            
            breakdown_data.append({
                'Vendor': f"Vendor {int(vid)}",
                'Agents': int(agent_count),
                '% Agents': f"{(agent_count/agents_with_selection)*100:.1f}%" if agents_with_selection > 0 else "0.0%",
                'Purchase Requests': int(pr_count),
                '% Requests': f"{(pr_count/total_vendor_purchase_requests)*100:.1f}%" if total_vendor_purchase_requests > 0 else "0.0%",
                'Transactions': int(tx_count),
                '% Completed': completion_rate
            })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    else:
        st.info("No vendor selections found (agents may have 0 purchases)")
    
    # Breakdown by Period
    st.markdown("---")
    st.markdown("**📅 Vendor Selection Breakdown by Period:**")
    
    # Breakdown by Period
    st.markdown("---")
    st.markdown("**📅 Vendor Selection Breakdown by Period:**")
    
    # Check if we have data (requests + vendors)
    has_purchase_requests = 'purchase_requests' in df.columns
    has_any_vendor_data = (len(vendor_counts) > 0) or (vendors_data and len(vendors_data) > 0)
    
    if has_purchase_requests and has_any_vendor_data:
        # Get duration_hours using centralized utility
        from app.utils.timestamp_utils import get_duration_hours
        duration_hours = get_duration_hours()
        
        # Collect data by period
        period_data = {}  # {period: {vendor_id: {'agents': set(), 'requests': count, 'transactions': count}}}
        
        for idx, row in df.iterrows():
            requests = row.get('purchase_requests', [])
            agent_id = row.get('agent_id', idx + 1)
            
            if isinstance(requests, list):
                for req in requests:
                    if isinstance(req, dict):
                        vendor_id = req.get('vendorID')
                        
                        # Get period from timestamp_hours or period field
                        timestamp_hours = req.get('timestamp_hours', np.nan)
                        if not pd.isna(timestamp_hours):
                            # FIXED: Use actual duration_hours instead of hardcoded 24
                            period = int(timestamp_hours // duration_hours) + 1 if timestamp_hours >= 0 else np.nan
                        else:
                            period = req.get('period', np.nan)
                        
                        if not pd.isna(vendor_id) and not pd.isna(period):
                            # Normalize vendor_id and period
                            try:
                                vendor_id_key = int(vendor_id)
                                period_key = int(period)
                            except (ValueError, TypeError):
                                continue
                                
                            # Initialize period if not exists
                            if period_key not in period_data:
                                period_data[period_key] = {}
                            
                            # Initialize vendor if not exists for this period
                            if vendor_id_key not in period_data[period_key]:
                                period_data[period_key][vendor_id_key] = {
                                    'agents': set(),
                                    'requests': 0,
                                    'transactions': 0
                                }
                            
                            # Add agent to set (for unique count)
                            period_data[period_key][vendor_id_key]['agents'].add(agent_id)
                            
                            # Count purchase request
                            period_data[period_key][vendor_id_key]['requests'] += 1
                            
                            # Count transaction if completed
                            completed = req.get('transactionCompleted', req.get('completed', req.get('transaction_completed', True)))
                            if completed or completed == 1:
                                period_data[period_key][vendor_id_key]['transactions'] += 1
        
        if period_data:
            # Sort periods
            sorted_periods = sorted(period_data.keys())
            
            # Calculate totals across ALL periods
            all_agents_across_periods = set()
            total_requests_all_periods = 0
            total_transactions_all_periods = 0
            
            for period in sorted_periods:
                period_vendors = period_data[period]
                all_agents_across_periods.update(set().union(*[v['agents'] for v in period_vendors.values()]))
                total_requests_all_periods += sum(v['requests'] for v in period_vendors.values())
                total_transactions_all_periods += sum(v['transactions'] for v in period_vendors.values())
            
            # Show summary metrics for ALL periods
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Agents", f"{len(all_agents_across_periods):,}")
            with col2:
                st.metric("Total Purchase Requests", f"{total_requests_all_periods:,}")
            with col3:
                st.metric("Total Transactions", f"{total_transactions_all_periods:,}")
            
            st.markdown("---")
            
            # Build combined breakdown data for all periods with Period column
            all_periods_breakdown_data = []
            
            # Get all relevant vendor IDs
            all_vendor_ids = set()
            if vendors_data:
                all_vendor_ids = {int(v.get('vendor_id')) for v in vendors_data}
            if len(vendor_counts) > 0:
                all_vendor_ids.update([int(vid) for vid in vendor_counts.index])
            sorted_vendor_ids = sorted(list(all_vendor_ids))
            
            for period in sorted_periods:
                # Get vendors for this period (already normalized to int keys)
                period_vendors = period_data[period]
                
                # Calculate totals for this period (for percentage calculations)
                # Only sum up ACTUAL data from period_vendors
                total_agents_period = len(set().union(*[v['agents'] for v in period_vendors.values()]))
                total_requests_period = sum(v['requests'] for v in period_vendors.values())
                
                # Iterate through ALL vendors (including those with 0 selections)
                for vid in sorted_vendor_ids:
                    # Check if vendor exists in this period's data
                    if vid in period_vendors:
                        vendor_stats = period_vendors[vid]
                        agent_count = len(vendor_stats['agents'])
                        request_count = vendor_stats['requests']
                        transaction_count = vendor_stats['transactions']
                    else:
                        # Zero values for unselected vendor
                        agent_count = 0
                        request_count = 0
                        transaction_count = 0
                    
                    # Calculate completion rate for this vendor in this period (% of requests completed)
                    period_completion_rate = f"{(transaction_count/request_count)*100:.1f}%" if request_count > 0 else "0.0%"
                    
                    all_periods_breakdown_data.append({
                        'Period': int(period),
                        'Vendor': f"Vendor {vid}",
                        'Agents': int(agent_count),
                        '% Agents': f"{(agent_count/total_agents_period)*100:.1f}%" if total_agents_period > 0 else "0.0%",
                        'Purchase Requests': int(request_count),
                        '% Requests': f"{(request_count/total_requests_period)*100:.1f}%" if total_requests_period > 0 else "0.0%",
                        'Transactions': int(transaction_count),
                        '% Completed': period_completion_rate
                    })
            
            # Create combined DataFrame
            combined_breakdown_df = pd.DataFrame(all_periods_breakdown_data)
            
            # Display combined table
            st.dataframe(combined_breakdown_df, use_container_width=True, hide_index=True)
            
            # EXCEL EXPORT BUTTON
            try:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    combined_breakdown_df.to_excel(writer, index=False, sheet_name='Vendor Breakdown')
                    # Apply 2-decimal formatting
                    _apply_price_formatting_vendor(writer, 'Vendor Breakdown', combined_breakdown_df)
                
                st.download_button(
                    label="📥 Download Period Breakdown Excel",
                    data=buffer.getvalue(),
                    file_name=f"vendor_selection_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download the vendor selection breakdown by period table"
                )
            except ImportError:
                st.warning("⚠️ Excel export requires openpyxl package")
            except Exception as e:
                st.error(f"❌ Error creating Excel file: {str(e)}")
        else:
            st.info("No period data available in purchase requests")
    else:
        st.info("No purchase request data available for period breakdown")
    
    # Purchase Request Level Export
    st.markdown("---")
    st.markdown("**📊 Purchase Request Level Data Export**")
    st.caption("Download detailed data for each purchase request with vendor attributes and transaction outcomes")
    
    st.info("ℹ️ **Note on Customer Paid Price**: The 'Customer Paid Price' column currently shows vendor base prices as placeholder values. Final customer prices will be calculated based on customer type (Discount/Fixed/Regular), platform price type (PN/BID), and pricing parameters once the pricing algorithm integration is completed.")
    
    # Try to get vendor data from multiple sources
    vendors_for_export = None
    if hasattr(st.session_state, 'vendors') and st.session_state.vendors:
        vendors_for_export = st.session_state.vendors
    elif 'simulation_results' in st.session_state:
        results = st.session_state.simulation_results
        if isinstance(results, dict):
            vendors_for_export = results.get('vendors') or results.get('config', {}).get('vendors')
    
    # Get configured price bounds for consistent normalization
    price_min_config = None
    price_max_config = None
    if hasattr(st.session_state, 'sim_params'):
        price_min_config = getattr(st.session_state.sim_params, 'vendor_price_min', 50.0)
        price_max_config = getattr(st.session_state.sim_params, 'vendor_price_max', 150.0)
    elif hasattr(st.session_state, 'simulation_params'):
        sim_params = st.session_state.simulation_params.get('simulation', {})
        price_min_config = sim_params.get('vendor_price_min', 50.0)
        price_max_config = sim_params.get('vendor_price_max', 150.0)
    
    # Build purchase request level data
    purchase_request_data = _build_purchase_request_export(
        df, vendors_for_export, 
        price_min_config=price_min_config,
        price_max_config=price_max_config
    )
    
    if purchase_request_data and len(purchase_request_data) > 0:
        try:
            # Create DataFrame
            pr_df = pd.DataFrame(purchase_request_data)
            
            # Create Excel with multiple sheets
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Sheet 1: Total (all data)
                pr_df.to_excel(writer, index=False, sheet_name='Total')
                # Apply 2-decimal formatting
                _apply_price_formatting_vendor(writer, 'Total', pr_df)
                
                # Additional sheets by Period
                if 'Period' in pr_df.columns:
                    periods = sorted(pr_df['Period'].unique())
                    for period in periods:
                        period_df = pr_df[pr_df['Period'] == period]
                        sheet_name = f'Period {period}'
                        period_df.to_excel(writer, index=False, sheet_name=sheet_name)
                        # Apply 2-decimal formatting to each period sheet
                        _apply_price_formatting_vendor(writer, sheet_name, period_df)
            
            col_download, col_info = st.columns([1, 2])
            
            with col_download:
                st.download_button(
                    label="📥 Download Purchase Requests Excel",
                    data=buffer.getvalue(),
                    file_name=f"purchase_requests_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download purchase request-level data with vendor attributes and outcomes"
                )
            
            with col_info:
                num_sheets = 1 + len(pr_df['Period'].unique()) if 'Period' in pr_df.columns else 1
                st.caption(f"📋 Export includes {len(pr_df):,} purchase requests across {num_sheets} sheets")
                st.caption(f"✅ Sheets: Total + {len(pr_df['Period'].unique())} Period sheets" if 'Period' in pr_df.columns else "✅ Sheet: Total")
        
        except ImportError:
            st.warning("⚠️ Excel export requires openpyxl package")
        except Exception as e:
            st.error(f"❌ Error creating Excel file: {str(e)}")
    else:
        st.info("ℹ️ No purchase request data available for export")
    
    # Vendor Data Section (only for multiple vendors)
    # FIXED: Show vendor table if multiple vendors were GENERATED (not just selected)
    if total_vendors_available > 1:
        st.markdown("---")
        st.markdown("**🏪 Vendor Data & Selection Analysis:**")
        st.caption("Understanding why certain vendors were selected or not selected")
        
        # Use vendors_data already retrieved at the beginning of the function
        # If not available, try to get from DataFrame metadata
        if not vendors_data and hasattr(df, 'attrs') and 'vendors' in df.attrs:
            vendors_data = df.attrs['vendors']
        
        if vendors_data and isinstance(vendors_data, list) and len(vendors_data) > 0:
            # Calculate proximity statistics from all agents' proximity data
            avg_proximity_per_vendor = {}
            min_proximity_per_vendor = {}
            max_proximity_per_vendor = {}
            std_proximity_per_vendor = {}
            
            if 'vendor_proximity_scores' in df.columns:
                # Extract all proximity scores
                all_proximity_scores = df['vendor_proximity_scores'].dropna()
                
                if len(all_proximity_scores) > 0:
                    # Initialize accumulators
                    proximity_lists = {}
                    
                    for proximity_dict in all_proximity_scores:
                        if isinstance(proximity_dict, dict):
                            for vendor_key, proximity_value in proximity_dict.items():
                                # vendor_key is a string like "1", "2", etc.
                                if vendor_key not in proximity_lists:
                                    proximity_lists[vendor_key] = []
                                proximity_lists[vendor_key].append(float(proximity_value))
                    
                    # Calculate statistics
                    for vendor_key in proximity_lists:
                        scores = proximity_lists[vendor_key]
                        if len(scores) > 0:
                            vendor_id = int(vendor_key)
                            avg_proximity_per_vendor[vendor_id] = np.mean(scores)
                            min_proximity_per_vendor[vendor_id] = np.min(scores)
                            max_proximity_per_vendor[vendor_id] = np.max(scores)
                            std_proximity_per_vendor[vendor_id] = np.std(scores)
            
            # Calculate integrated scores for each vendor (average across all agents)
            vendor_integrated_scores = {}
            
            if 'vendor_choice_weights' in df.columns and 'vendor_proximity_scores' in df.columns:
                for vendor in vendors_data:
                    vendor_id = vendor.get('vendor_id')
                    scores = []
                    
                    # Calculate score for each agent
                    for idx, row in df.iterrows():
                        weights = row.get('vendor_choice_weights', {})
                        proximity_scores = row.get('vendor_proximity_scores', {})
                        
                        if isinstance(weights, dict) and isinstance(proximity_scores, dict):
                            proximity = proximity_scores.get(str(vendor_id), 50.0)
                            score = _calculate_vendor_score(
                                vendor, weights, proximity, vendors_data,
                                price_min_config=price_min_config,
                                price_max_config=price_max_config
                            )
                            scores.append(score)
                    
                    # Average score across all agents
                    if scores:
                        vendor_integrated_scores[vendor_id] = np.mean(scores)
            
            # Create vendor comparison table
            vendor_table_data = []
            
            for idx, vendor in enumerate(vendors_data, 1):
                vendor_id = vendor.get('vendor_id', idx)
                
                # Get counts for this vendor (from earlier calculations)
                agent_count = 0
                if vendor_id in vendor_counts.index:
                    agent_count = int(vendor_counts[vendor_id])
                
                pr_count = vendor_purchase_requests.get(vendor_id, 0)
                tx_count = vendor_transactions.get(vendor_id, 0)
                
                # Get proximity statistics for this vendor
                avg_proximity = avg_proximity_per_vendor.get(vendor_id, None)
                proximity_avg_display = f"{avg_proximity:.1f}" if avg_proximity is not None else "N/A"
                
                # Get integrated score
                integrated_score = vendor_integrated_scores.get(vendor_id, None)
                integrated_score_display = f"{integrated_score:.3f}" if integrated_score is not None else "N/A"
                
                # Get quantity information - check if per-period data exists
                quantity_per_period = vendor.get('quantity_offered_per_period', {})
                if quantity_per_period and isinstance(quantity_per_period, dict):
                    # Calculate total quantity across all periods
                    total_quantity = sum(quantity_per_period.values())
                    # Show average quantity (calculated from per-period values)
                    avg_quantity = vendor.get('quantity_offered', 100)
                    # Show ONLY the average quantity (clean display)
                    quantity_display = str(avg_quantity)
                    total_quantity_display = total_quantity
                else:
                    # Legacy: single quantity value (assume 1 period)
                    quantity_display = str(vendor.get('quantity_offered', 100))
                    total_quantity_display = vendor.get('quantity_offered', 100)
                
                vendor_table_data.append({
                    'Vendor ID': f"Vendor {vendor_id}",
                    'Price ($)': f"${vendor.get('price', 0):.2f}",
                    'Average Quantity Per Period': quantity_display,
                    'Total Quantity': total_quantity_display,
                    'Quality': vendor.get('quality', 'N/A'),
                    'Sustainability': vendor.get('sustainability', 'N/A'),
                    'Average Proximity': proximity_avg_display,
                    'Integrated Score': integrated_score_display,
                    'Agents': agent_count,
                    '% Agents': f"{(agent_count / agents_with_selection * 100) if agents_with_selection > 0 else 0:.1f}%",
                    'Purchase Requests': pr_count,
                    '% Purchase Requests': f"{(pr_count / total_vendor_purchase_requests * 100) if total_vendor_purchase_requests > 0 else 0:.1f}%",
                    'Transactions': tx_count,
                    '% Transactions': f"{(tx_count / total_vendor_transactions * 100) if total_vendor_transactions > 0 else 0:.1f}%"
                })
            
            vendor_df = pd.DataFrame(vendor_table_data)
            
            st.markdown("**📋 Vendor Attributes & Selection Results:**")
            st.dataframe(vendor_df, use_container_width=True, hide_index=True)

            # NEW: Excel Export for Period-Level Vendor Data
            vendor_period_data = []
            
            for vendor in vendors_data:
                vendor_id = vendor.get('vendor_id')
                price = vendor.get('price', 0)
                quality = vendor.get('quality', 0)
                sustainability = vendor.get('sustainability', 0)
                
                # Get metrics
                avg_prox = avg_proximity_per_vendor.get(vendor_id, np.nan)
                int_score = vendor_integrated_scores.get(vendor_id, np.nan)
                
                # Get periods
                quantity_per_period = vendor.get('quantity_offered_per_period', {})
                if quantity_per_period and isinstance(quantity_per_period, dict):
                    for period, qty in sorted(quantity_per_period.items()):
                        vendor_period_data.append({
                            'Vendor ID': f"Vendor {vendor_id}",
                            'Period': int(period),
                            'Quantity Offered': qty,
                            'Price': price,
                            'Quality': quality,
                            'Sustainability': sustainability,
                            'Average Proximity': avg_prox,
                            'Integrated Score': int_score
                        })
                else:
                    # Single period default (Period 1)
                    qty = vendor.get('quantity_offered', 100)
                    vendor_period_data.append({
                        'Vendor ID': f"Vendor {vendor_id}",
                        'Period': 1,
                        'Quantity Offered': qty,
                        'Price': price,
                        'Quality': quality,
                        'Sustainability': sustainability,
                        'Average Proximity': avg_prox,
                        'Integrated Score': int_score
                    })
            
            if vendor_period_data:
                try:
                    vendor_period_df = pd.DataFrame(vendor_period_data)
                    
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        vendor_period_df.to_excel(writer, index=False, sheet_name='Vendor Details Per Period')
                        # Apply 2-decimal formatting
                        _apply_price_formatting_vendor(writer, 'Vendor Details Per Period', vendor_period_df)
                    
                    st.download_button(
                        label="📥 Download Vendor Details (Per Period)",
                        data=buffer.getvalue(),
                        file_name=f"vendor_details_per_period_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download detailed vendor attributes including quantity offered for each period"
                    )
                except Exception as e:
                    # Silent fail or log if needed, but keeping it simple as per style
                    pass
            
            # Score comparison visualization
            st.markdown("**📊 Vendor Attribute Comparison:**")
            
            col_price, col_quality = st.columns(2)
            col_sust, col_integrated = st.columns(2)
            
            with col_price:
                # Price comparison (inverted - lower is better)
                # Normalize to 0-100 scale
                prices = [float(p.replace('$', '').replace(',', '')) for p in vendor_df['Price ($)']]
                min_price = min(prices) if prices else 0
                max_price = max(prices) if prices else 100
                
                # Inverted normalization: lower price = higher score
                if max_price > min_price:
                    price_scores = [100 * (1 - (p - min_price) / (max_price - min_price)) for p in prices]
                else:
                    price_scores = [50.0] * len(prices)  # All same price
                
                st.markdown("**Price Score (0-100) (Higher = Lower Price)**")
                price_fig = px.bar(
                    vendor_df,
                    x='Vendor ID',
                    y=price_scores,
                    labels={'y': 'Score', 'x': ''}
                )
                price_fig.update_layout(showlegend=False, height=250, yaxis=dict(range=[0, 100]))
                st.plotly_chart(price_fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
            
            with col_quality:
                # Quality comparison
                quality_vals = [v if isinstance(v, int) else 0 for v in vendor_df['Quality']]
                st.markdown("**Quality Score (1-5)**")
                qual_fig = px.bar(
                    vendor_df,
                    x='Vendor ID',
                    y=quality_vals,
                    labels={'y': 'Quality', 'x': ''}
                )
                qual_fig.update_layout(showlegend=False, height=250, yaxis=dict(range=[1, 5], dtick=1))
                st.plotly_chart(qual_fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
            
            with col_sust:
                # Sustainability comparison
                sust_vals = [v if isinstance(v, int) else 0 for v in vendor_df['Sustainability']]
                st.markdown("**Sustainability Score (1-5)**")
                sust_fig = px.bar(
                    vendor_df,
                    x='Vendor ID',
                    y=sust_vals,
                    labels={'y': 'Sustainability', 'x': ''}
                )
                sust_fig.update_layout(showlegend=False, height=250, yaxis=dict(range=[1, 5], dtick=1))
                st.plotly_chart(sust_fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
            
            with col_integrated:
                # Integrated Score comparison
                integrated_vals = []
                for val in vendor_df['Integrated Score']:
                    if val != "N/A":
                        integrated_vals.append(float(val))
                    else:
                        integrated_vals.append(0.0)
                
                st.markdown("**Integrated Score (0-1)**")
                int_fig = px.bar(
                    vendor_df,
                    x='Vendor ID',
                    y=integrated_vals,
                    labels={'y': 'Score', 'x': ''}
                )
                int_fig.update_layout(showlegend=False, height=250, yaxis=dict(range=[0, 1]))
                st.plotly_chart(int_fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
            
            # Third row: Proximity chart
            col_proximity, col_spacer = st.columns(2)
            
            with col_proximity:
                # Proximity comparison (average across all agents)
                proximity_vals = []
                for val in vendor_df['Average Proximity']:
                    if val != "N/A":
                        proximity_vals.append(float(val))
                    else:
                        proximity_vals.append(0.0)
                
                st.markdown("**Average Proximity Score (0-100) (Higher = Closer)**")
                prox_fig = px.bar(
                    vendor_df,
                    x='Vendor ID',
                    y=proximity_vals,
                    labels={'y': 'Proximity', 'x': ''}
                )
                prox_fig.update_layout(showlegend=False, height=250, yaxis=dict(range=[0, 100]))
                st.plotly_chart(prox_fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
            
            with col_spacer:
                # Empty space to maintain layout balance
                pass
            
            # Vendor Score Breakdown Table (NEW: Show transparent scoring)
            st.markdown("---")
            st.markdown("**🔍 Vendor Score Breakdown (Average Across All Agents)**")
            st.caption("Shows how each vendor's score is calculated from normalized attributes and weights")
            
            # Build score breakdown table using centralized scoring function
            if 'vendor_choice_weights' in df.columns and 'vendor_proximity_scores' in df.columns:
                score_breakdown_data = []
                
                # Get average weights across all agents
                all_weights = df['vendor_choice_weights'].dropna()
                if len(all_weights) > 0:
                    avg_weights = {
                        'price': np.mean([w.get('price', 0) for w in all_weights if isinstance(w, dict)]),
                        'quality': np.mean([w.get('quality', 0) for w in all_weights if isinstance(w, dict)]),
                        'proximity': np.mean([w.get('proximity', 0) for w in all_weights if isinstance(w, dict)]),
                        'sustainability': np.mean([w.get('sustainability', 0) for w in all_weights if isinstance(w, dict)])
                    }
                    
                    for vendor in vendors_data:
                        vendor_id = vendor.get('vendor_id')
                        price = vendor.get('price', 0)
                        quality = vendor.get('quality', 3)
                        sustainability = vendor.get('sustainability', 3)
                        avg_proximity = avg_proximity_per_vendor.get(vendor_id, 50.0)
                        
                        # Use centralized scoring function for consistency
                        score_result = calculate_vendor_score_with_breakdown(
                            vendor=vendor,
                            weights=avg_weights,
                            proximity=avg_proximity,
                            all_vendors=vendors_data,
                            price_min_config=price_min_config,
                            price_max_config=price_max_config
                        )
                        
                        score_breakdown_data.append({
                            'Vendor': f"Vendor {vendor_id}",
                            'Price ($)': f"${price:.2f}",
                            'Norm Price': f"{score_result['norm_price']:.3f}",
                            'Price Weight': f"{score_result['weight_price']:.2f}",
                            'Price Component': f"{score_result['weighted_price']:.3f}",
                            'Quality (1-5)': quality,
                            'Norm Quality': f"{score_result['norm_quality']:.3f}",
                            'Quality Weight': f"{score_result['weight_quality']:.2f}",
                            'Quality Component': f"{score_result['weighted_quality']:.3f}",
                            'Sustainability (1-5)': sustainability,
                            'Norm Sustain': f"{score_result['norm_sustainability']:.3f}",
                            'Sustain Weight': f"{score_result['weight_sustainability']:.2f}",
                            'Sustain Component': f"{score_result['weighted_sustainability']:.3f}",
                            'Avg Proximity': f"{avg_proximity:.1f}",
                            'Norm Proximity': f"{score_result['norm_proximity']:.3f}",
                            'Proximity Weight': f"{score_result['weight_proximity']:.2f}",
                            'Proximity Component': f"{score_result['weighted_proximity']:.3f}",
                            'Final Score': f"{score_result['integrated_score']:.3f}"
                        })
                    
                    if score_breakdown_data:
                        score_df = pd.DataFrame(score_breakdown_data)
                        st.dataframe(score_df, use_container_width=True, hide_index=True, height=min(400, 35 + 35 * len(score_breakdown_data)))
                        
                        st.caption("💡 **Formula**: Final Score = (Price Weight × Norm Price) + (Quality Weight × Norm Quality) + (Proximity Weight × Norm Proximity) + (Sustainability Weight × Norm Sustainability)")
                        st.caption("📊 **Normalization**: Price is inverted (lower=better), others are scaled to [0,1]. Proximity averaged across all agents.")
                        
                        # Excel export for Vendor Score Breakdown
                        try:
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                score_df.to_excel(writer, index=False, sheet_name='Vendor Score Breakdown')
                                # Apply 2-decimal formatting
                                _apply_price_formatting_vendor(writer, 'Vendor Score Breakdown', score_df)
                            
                            st.download_button(
                                label="📥 Download Vendor Score Breakdown Excel",
                                data=buffer.getvalue(),
                                file_name=f"vendor_score_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Download the vendor score breakdown table showing normalized attributes and score calculation"
                            )
                        except ImportError:
                            st.warning("⚠️ Excel export requires openpyxl package")
                        except Exception as e:
                            st.error(f"❌ Error creating Excel file: {str(e)}")
            
            # Agent-Vendor Proximity Matrix (NEW: Display table + Download)
            st.markdown("---")
            st.markdown("**🔍 Agent-Vendor Proximity Score Matrix**")
            st.caption("View and download the complete matrix showing each agent's proximity to each vendor")
            
            if 'vendor_proximity_scores' in df.columns:
                # Build complete proximity matrix
                proximity_matrix_data = []
                for idx in range(len(df)):
                    row_data = {}
                    
                    # Add Agent ID
                    if 'agent_id' in df.columns:
                        row_data['Agent ID'] = df.iloc[idx]['agent_id']
                    else:
                        row_data['Agent ID'] = idx + 1
                    
                    # Add proximity scores for each vendor
                    scores = df.iloc[idx]['vendor_proximity_scores']
                    if isinstance(scores, dict):
                        for v_id in sorted(scores.keys(), key=lambda x: int(x)):
                            row_data[f'Vendor {v_id} Proximity'] = scores[v_id]
                    
                    proximity_matrix_data.append(row_data)
                
                if proximity_matrix_data:
                    proximity_df = pd.DataFrame(proximity_matrix_data)
                    
                    # Display proximity matrix table (with option to show all or just sample)
                    with st.expander("📊 View Proximity Matrix Table", expanded=False):
                        show_all_agents = st.checkbox(
                            "Show all agents", 
                            value=False, 
                            key="show_all_proximity_matrix",
                            help="Display proximity matrix for all agents (can be large). Default shows first 20 agents."
                        )
                        
                        if show_all_agents:
                            st.dataframe(proximity_df, use_container_width=True, height=min(600, 35 + 35 * len(proximity_df)))
                            st.caption(f"Showing all {len(proximity_df)} agents")
                        else:
                            # Show first 20 agents
                            display_df = proximity_df.head(20)
                            st.dataframe(display_df, use_container_width=True, height=min(600, 35 + 35 * len(display_df)))
                            st.caption(f"Showing first 20 agents (out of {len(proximity_df)} total). Check 'Show all agents' to view complete matrix.")
                    
                    # Create Excel file for proximity matrix
                    try:
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            proximity_df.to_excel(writer, index=False, sheet_name='Agent-Vendor Proximity')
                            # Apply 2-decimal formatting
                            _apply_price_formatting_vendor(writer, 'Agent-Vendor Proximity', proximity_df)
                        
                        col_download, col_info = st.columns([1, 2])
                        
                        with col_download:
                            st.download_button(
                                label="📊 Download Proximity Matrix Excel",
                                data=buffer.getvalue(),
                                file_name=f"agent_vendor_proximity_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Download complete Agent-Vendor proximity score matrix"
                            )
                        
                        with col_info:
                            st.caption(f"📋 Matrix includes {len(proximity_df):,} agents × {len(vendors_data)} vendors = {len(proximity_df) * len(vendors_data):,} proximity scores")
                    
                    except ImportError:
                        st.warning("⚠️ Excel export requires openpyxl package")
                    except Exception as e:
                        st.error(f"❌ Error creating Excel file: {str(e)}")
        else:
            st.info("ℹ️ Vendor attribute data not available. This section shows detailed vendor data in multi-vendor simulations.")
    
    # Explanation of how vendor selection works
    with st.expander("ℹ️ How Vendor Selection Works (Default Behavior)", expanded=False):
        st.markdown("""
        **Vendor Selection Default Logic:**
        
        For each agent:
        1. **Get Vendor Pool**: Vendors have attributes:
           - **Price**: Randomized within [vendor_price_min, vendor_price_max] from Page 1 configuration
           - **Quantity Offered**: Random integer in [vendor_products_min, vendor_products_max] per period
           - **Quality**: Random integer in [1, 5] (generated once per vendor)
           - **Sustainability**: Random integer in [1, 5] (generated once per vendor)
           - **Proximity**: Random score [0, 100] per customer-vendor dyad
             - Uniformly distributed in [0, 100] range
             - Each agent-vendor pair gets a unique proximity value (fixed per dyad)
             - Different agents have different proximities to the same vendor
             - No predefined vendor location types (purely random)
        
        2. **Get Weights**: From vendor_choice_weights decision (configured on Page 2 Overview)
           - Example: {price: 0.5, quality: 0.5, proximity: 0.0, sustainability: 0.0}
        
        3. **Standardize Attributes** to [0, 1]:
           - Price: Normalized using **min-max normalization** where best price = 1, worst price = 0
             `norm_price = 1.0 - (price - min_price) / (max_price - min_price)`
           - Quality: (value - 1) / 4
           - Sustainability: (value - 1) / 4
           - Proximity: value / 100
        
        4. **Calculate Composite Score** for each vendor:
           ```
           score = w_price × norm_price + w_quality × norm_quality + 
                   w_proximity × norm_proximity + w_sustainability × norm_sustainability
           ```
        
        5. **Select Best Vendor**: Vendor with highest composite score
        
        6. **Apply to All Requests**: All purchase requests from the same agent get the same vendorID
        
        **Result**: Deterministic selection based on weighted preferences
        
        **Note**: Quantity offered represents vendor capacity per period and can be used for supply constraints in future implementations.
        """)
    
    # Show configured weights (read-only)
    if 'vendor_choice_weights' in df.columns:
        st.markdown("---")
        st.markdown("**⚙️ Configured Vendor Choice Weights (Read-Only):**")
        
        # Get weights from first agent (all should have same weights)
        sample_weights = df['vendor_choice_weights'].iloc[0]
        
        if isinstance(sample_weights, dict):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show active weights
                active_weights = {k: v for k, v in sample_weights.items() if v > 0}
                
                if active_weights:
                    st.success("✅ **Active Parameters:**")
                    for param, weight in active_weights.items():
                        st.write(f"• {param.title()}: {weight:.2%}")
                else:
                    st.warning("No parameters selected")
            
            with col2:
                # Show pie chart if multiple weights
                if len(active_weights) > 1:
                    st.markdown("### Weight Distribution")
                    fig = px.pie(
                        values=list(active_weights.values()),
                        names=[k.title() for k in active_weights.keys()]
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
                elif len(active_weights) == 1:
                    st.info(f"Single factor: {list(active_weights.keys())[0].title()}")
        
        st.caption("💡 To modify weights: Go to **Page 2 → Overview Tab → Vendor Choice Weights**")

