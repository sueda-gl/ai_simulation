# app/pages/results/visualizations/vendor_viz.py
"""
Vendor-related visualization functions.
Handles vendor_choice_weights and vendor_selection decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from app.reports.vendor import (
    average_vendor_scores,
    build_period_breakdown,
    build_proximity_matrix,
    build_purchase_request_export,
    build_selection_breakdown,
    build_vendor_attributes_table,
    build_vendor_choice_weights_export,
    build_vendor_period_details,
    build_vendor_score_breakdown,
    collect_period_data,
    count_requests_per_vendor,
    count_vendor_requests,
    period_totals,
    proximity_matrix_xlsx,
    proximity_statistics,
    purchase_requests_xlsx,
    sorted_vendor_ids as vendor_ids_sorted,
    vendor_choice_weights_xlsx,
    vendor_period_details_xlsx,
    vendor_score_breakdown_xlsx,
    vendor_selection_breakdown_xlsx,
)
from app.utils.timestamp_utils import (
    get_duration_hours,
    get_periods,
    get_simulation_base_time,
)


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
    # (read-only here: the key is initialised at app start by app.models)
    selection_key = f"{decision_name}_default_params"

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
    export_data = build_vendor_choice_weights_export(df, decision_data)
    
    if export_data:
        export_df = pd.DataFrame(export_data)
        
        # Create Excel file
        try:
            xlsx_bytes = vendor_choice_weights_xlsx(export_df)
            
            col_download, col_info = st.columns([1, 2])
            
            with col_download:
                st.download_button(
                    label="📊 Download Vendor Weights Excel",
                    data=xlsx_bytes,
                    file_name=f"vendor_choice_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download vendor choice weights with agent info and weight percentages (numeric format)"
                )
            
            with col_info:
                st.caption(f"📋 Export includes {len(export_df):,} agents with {len(export_df.columns)} columns")
                st.caption(f"✅ Fields: Agent ID, Honesty_Humility, Assigned Allowance Level, Study Program, Group_experiment, TWT+Sospeso, income, Price, Quality, Proximity, Sustainability")
        
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

    # Get configured price bounds for consistent normalization
    # (Used in vendor score calculations throughout this function)
    price_min_config = None
    price_max_config = None
    if hasattr(st.session_state, 'sim_params'):
        price_min_config = getattr(st.session_state.sim_params, 'vendor_price_min', 50.0)
        price_max_config = getattr(st.session_state.sim_params, 'vendor_price_max', 150.0)

    # Count unique vendors selected (excluding NaN)
    vendor_counts = decision_data.dropna().value_counts()
    num_vendors_selected = len(vendor_counts)
    
    # Calculate purchase request and transaction shares per vendor
    (total_purchase_requests, total_transactions_completed,
     vendor_pr_counts, vendor_tx_counts) = count_vendor_requests(df)
    
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
        vendor_purchase_requests, vendor_transactions = count_requests_per_vendor(df)
        
        # Calculate totals for percentages
        total_vendor_purchase_requests = sum(vendor_purchase_requests.values()) if vendor_purchase_requests else 0
        total_vendor_transactions = sum(vendor_transactions.values()) if vendor_transactions else 0
        
        # Get all relevant vendor IDs
        sorted_vendor_ids = vendor_ids_sorted(vendors_data, vendor_counts)
        
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
        breakdown_data = build_selection_breakdown(
            sorted_vendor_ids, vendor_counts, vendor_purchase_requests,
            vendor_transactions, agents_with_selection, total_vendor_purchase_requests
        )
        
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
        duration_hours = get_duration_hours()
        
        # Collect data by period
        period_data = collect_period_data(df, duration_hours)
        
        if period_data:
            # Sort periods
            sorted_periods = sorted(period_data.keys())
            
            # Calculate totals across ALL periods
            (all_agents_across_periods, total_requests_all_periods,
             total_transactions_all_periods) = period_totals(period_data, sorted_periods)
            
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
            # Get all relevant vendor IDs
            sorted_vendor_ids = vendor_ids_sorted(vendors_data, vendor_counts)
            
            all_periods_breakdown_data = build_period_breakdown(
                period_data, sorted_periods, sorted_vendor_ids
            )
            
            # Create combined DataFrame
            combined_breakdown_df = pd.DataFrame(all_periods_breakdown_data)
            
            # Display combined table
            st.dataframe(combined_breakdown_df, use_container_width=True, hide_index=True)
            
            # EXCEL EXPORT BUTTON
            try:
                xlsx_bytes = vendor_selection_breakdown_xlsx(combined_breakdown_df)
                
                st.download_button(
                    label="📥 Download Period Breakdown Excel",
                    data=xlsx_bytes,
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

    # Get the pricing parameters the customer price formula uses
    platform_markup = 0.1
    price_range = 0.25
    if hasattr(st.session_state, 'sim_params'):
        platform_markup = getattr(st.session_state.sim_params, 'platform_markup', 0.1)
        price_range = getattr(st.session_state.sim_params, 'price_range', 0.25)

    # Build purchase request level data
    purchase_request_data = build_purchase_request_export(
        df, vendors_for_export,
        get_simulation_base_time(), get_duration_hours(), get_periods(),
        platform_markup=platform_markup,
        price_range=price_range,
        price_min_config=price_min_config,
        price_max_config=price_max_config
    )
    
    if purchase_request_data and len(purchase_request_data) > 0:
        try:
            # Create DataFrame
            pr_df = pd.DataFrame(purchase_request_data)
            
            # Create Excel with multiple sheets
            xlsx_bytes = purchase_requests_xlsx(pr_df)
            
            col_download, col_info = st.columns([1, 2])
            
            with col_download:
                st.download_button(
                    label="📥 Download Purchase Requests Excel",
                    data=xlsx_bytes,
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
            (avg_proximity_per_vendor, min_proximity_per_vendor,
             max_proximity_per_vendor, std_proximity_per_vendor) = proximity_statistics(df)
            
            # Calculate integrated scores for each vendor (average across all agents)
            vendor_integrated_scores = average_vendor_scores(
                df, vendors_data,
                price_min_config=price_min_config,
                price_max_config=price_max_config
            )
            
            # Create vendor comparison table
            vendor_table_data = build_vendor_attributes_table(
                vendors_data, vendor_counts, vendor_purchase_requests,
                vendor_transactions, agents_with_selection,
                total_vendor_purchase_requests, total_vendor_transactions,
                avg_proximity_per_vendor, vendor_integrated_scores
            )
            
            vendor_df = pd.DataFrame(vendor_table_data)
            
            st.markdown("**📋 Vendor Attributes & Selection Results:**")
            st.dataframe(vendor_df, use_container_width=True, hide_index=True)

            # NEW: Excel Export for Period-Level Vendor Data
            vendor_period_data = build_vendor_period_details(
                vendors_data, avg_proximity_per_vendor, vendor_integrated_scores
            )
            
            if vendor_period_data:
                try:
                    vendor_period_df = pd.DataFrame(vendor_period_data)
                    
                    xlsx_bytes = vendor_period_details_xlsx(vendor_period_df)
                    
                    st.download_button(
                        label="📥 Download Vendor Details (Per Period)",
                        data=xlsx_bytes,
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
                score_breakdown_data = build_vendor_score_breakdown(
                    df, vendors_data, avg_proximity_per_vendor,
                    price_min_config=price_min_config,
                    price_max_config=price_max_config
                )
                
                if score_breakdown_data:
                    score_df = pd.DataFrame(score_breakdown_data)
                    st.dataframe(score_df, use_container_width=True, hide_index=True, height=min(400, 35 + 35 * len(score_breakdown_data)))
                    
                    st.caption("💡 **Formula**: Final Score = (Price Weight × Norm Price) + (Quality Weight × Norm Quality) + (Proximity Weight × Norm Proximity) + (Sustainability Weight × Norm Sustainability)")
                    st.caption("📊 **Normalization**: Price is inverted (lower=better), others are scaled to [0,1]. Proximity averaged across all agents.")
                    
                    # Excel export for Vendor Score Breakdown
                    try:
                        xlsx_bytes = vendor_score_breakdown_xlsx(score_df)
                        
                        st.download_button(
                            label="📥 Download Vendor Score Breakdown Excel",
                            data=xlsx_bytes,
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
                proximity_matrix_data = build_proximity_matrix(df)
                
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
                        xlsx_bytes = proximity_matrix_xlsx(proximity_df)
                        
                        col_download, col_info = st.columns([1, 2])
                        
                        with col_download:
                            st.download_button(
                                label="📊 Download Proximity Matrix Excel",
                                data=xlsx_bytes,
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

