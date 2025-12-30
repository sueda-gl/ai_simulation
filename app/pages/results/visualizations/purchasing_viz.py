# app/pages/results/visualizations/purchasing_viz.py
"""
Purchasing-related visualization functions.
Handles purchasing_quantity and purchasing_frequency decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from datetime import datetime, timedelta
from app.utils.timestamp_utils import TimestampConverter, get_duration_hours, get_periods


def _apply_price_formatting_purchasing(writer, sheet_name: str, df: pd.DataFrame):
    """
    Apply Excel number formatting to price-related columns to display 2 decimal places.
    """
    price_columns = [
        'purchasing_frequency', 'Purchasing Frequency',
        'Customer Price', 'customer_price',
    ]
    
    workbook = writer.book
    worksheet = workbook[sheet_name]
    
    for col_idx, col_name in enumerate(df.columns, start=1):
        if col_name in price_columns:
            for row_idx in range(2, len(df) + 2):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                if isinstance(cell.value, (int, float)) and cell.value is not None:
                    cell.number_format = '0.00'


def render_purchasing_quantity(df, decision_name, decision_title, decision_data):
    """Visualization for purchasing_quantity - quantity analysis with purchase requests"""
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        mean_qty = decision_data.mean()
        st.metric("Mean Quantity", f"{mean_qty:.1f}")
    
    with col3:
        total_purchases = decision_data.sum()
        st.metric("Total Purchase Requests", f"{int(total_purchases):,}")
    
    with col4:
        agents_with_purchases = (decision_data > 0).sum()
        pct_with_purchases = agents_with_purchases / len(decision_data) * 100
        st.metric("Agents w/ Purchase Requests", f"{pct_with_purchases:.1f}%")
    
    # Distribution plot and statistics
    col_plot, col_stats = st.columns([2, 1])
    
    with col_plot:
        # Histogram of purchase quantities
        st.markdown("**Distribution of Purchase Requests**")
        fig = px.histogram(
            df,
            x=decision_name,
            nbins=min(30, int(decision_data.max()) + 1),
            labels={decision_name: 'Purchase Requests per Period', 'count': 'Number of Agents'}
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Purchase Requests per Period",
            yaxis_title="Number of Agents"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
        st.markdown("**📈 Statistics**")
        stats = decision_data.describe()
        
        # Get number of periods and duration from simulation config
        if hasattr(st.session_state, 'sim_params'):
            periods = st.session_state.sim_params.periods
            duration_hours = st.session_state.sim_params.duration_hours
        else:
            periods = 15  # default
            duration_hours = 2.0
        
        # Calculate ACTUAL per-period statistics (not just dividing term stats by periods)
        # This properly computes statistics over actual per-agent-per-period purchase counts
        per_period_counts = []
        
        if 'purchase_requests' in df.columns:
            for idx, row in df.iterrows():
                purchase_requests = row.get('purchase_requests', [])
                if isinstance(purchase_requests, list):
                    # Initialize period counts for this agent
                    agent_period_counts = {i: 0 for i in range(periods)}
                    
                    for req in purchase_requests:
                        if isinstance(req, dict) and 'timestamp_hours' in req:
                            timestamp = req['timestamp_hours']
                            period_idx = int(timestamp // duration_hours)
                            if 0 <= period_idx < periods:
                                agent_period_counts[period_idx] += 1
                    
                    # Add all period counts for this agent to the list
                    per_period_counts.extend(agent_period_counts.values())
        
        # Compute statistics on actual per-period values
        if per_period_counts:
            per_period_series = pd.Series(per_period_counts)
            per_period_stats = per_period_series.describe()
        else:
            # Fallback: divide by periods if no purchase_requests data available
            per_period_stats = pd.Series({
                'mean': stats['mean'] / periods,
                'std': stats['std'] / periods,
                'min': stats['min'] / periods,
                'max': stats['max'] / periods,
                '25%': stats['25%'] / periods,
                '50%': stats['50%'] / periods,
                '75%': stats['75%'] / periods
            })
        
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
            'Purchase Requests per Term': [
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{int(stats['min'])}",
                f"{int(stats['max'])}",
                f"{stats['50%']:.2f}",
                f"{stats['25%']:.2f}",
                f"{stats['75%']:.2f}"
            ],
            'Purchase Requests per Period': [
                f"{per_period_stats['mean']:.2f}",
                f"{per_period_stats['std']:.2f}",
                f"{int(per_period_stats['min'])}",
                f"{int(per_period_stats['max'])}",
                f"{per_period_stats['50%']:.2f}",
                f"{per_period_stats['25%']:.2f}",
                f"{per_period_stats['75%']:.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Customer Type Breakdown - Purchase Requests by customer type
    if 'purchase_requests' in df.columns:
        st.markdown("---")
        st.markdown("**🎯 Purchase Requests by Customer Type**")
        st.caption("Distribution of total purchase requests across Regular, Fixed, and Discount customers")
        
        # Extract customer type from purchase_requests
        customer_type_counts = {'Regular': 0, 'Fixed': 0, 'Discount': 0}
        
        for idx, row in df.iterrows():
            purchase_requests = row.get('purchase_requests', [])
            if isinstance(purchase_requests, list):
                for req in purchase_requests:
                    if isinstance(req, dict):
                        customer_type = req.get('customer_type', 'regular')
                        # Normalize to title case
                        if isinstance(customer_type, str):
                            customer_type = customer_type.capitalize()
                        
                        if customer_type in customer_type_counts:
                            customer_type_counts[customer_type] += 1
        
        total_purchases_by_type = sum(customer_type_counts.values())
        
        if total_purchases_by_type > 0:
            col_pie, col_stats_table = st.columns([2, 1])
            
            with col_pie:
                # Create pie chart
                pie_data = pd.DataFrame({
                    'Customer Type': list(customer_type_counts.keys()),
                    'Purchase Requests': list(customer_type_counts.values())
                })
                
                # Filter out zero values for cleaner pie chart
                pie_data = pie_data[pie_data['Purchase Requests'] > 0]
                
                st.markdown("### Purchase Request Distribution by Customer Type")
                fig_pie = px.pie(
                    pie_data,
                    values='Purchase Requests',
                    names='Customer Type',
                    color='Customer Type',
                    color_discrete_map={
                        'Regular': '#1f77b4',
                        'Fixed': '#ff7f0e',
                        'Discount': '#2ca02c'
                    }
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_stats_table:
                st.markdown("**📊 Statistics**")
                
                # Create statistics table
                type_stats = []
                for ctype, count in customer_type_counts.items():
                    percentage = (count / total_purchases_by_type * 100) if total_purchases_by_type > 0 else 0
                    type_stats.append({
                        'Customer Type': ctype,
                        'Purchase Requests': f"{count:,}",
                        'Percentage': f"{percentage:.1f}%"
                    })
                
                # Add total row
                type_stats.append({
                    'Customer Type': 'TOTAL',
                    'Purchase Requests': f"{total_purchases_by_type:,}",
                    'Percentage': '100.0%'
                })
                
                type_stats_df = pd.DataFrame(type_stats)
                st.dataframe(type_stats_df, use_container_width=True, hide_index=True)
            
            # Now create three sub-sections, one for each customer type
            st.markdown("---")
            st.markdown("**📊 Detailed Analysis by Customer Type**")
            st.caption("Purchase request quantity distribution and statistics for each customer type")
            
            # Get number of periods for per-period calculations
            if hasattr(st.session_state, 'sim_params'):
                periods = st.session_state.sim_params.periods
                duration_hours = st.session_state.sim_params.duration_hours
            else:
                periods = 15  # default
                duration_hours = 2.0
            
            # Helper function to get purchasing quantities for a specific customer type
            def get_quantities_by_customer_type(df, target_type):
                """Extract purchasing quantities for agents of a specific customer type"""
                quantities = []
                for idx, row in df.iterrows():
                    # Get customer type - try direct column first, then purchase_requests as fallback
                    customer_type = ''
                    
                    # Priority 1: Check if customer_type is directly available in the dataframe
                    if 'customer_type' in row and pd.notna(row['customer_type']) and str(row['customer_type']).strip():
                        customer_type = str(row['customer_type']).capitalize()
                    else:
                        # Priority 2: Extract from purchase_requests if available
                        purchase_requests = row.get('purchase_requests', [])
                        if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
                            # Check customer type from first request (all requests have same customer type)
                            first_req = purchase_requests[0]
                            if isinstance(first_req, dict):
                                customer_type = first_req.get('customer_type', 'regular')
                                if isinstance(customer_type, str):
                                    customer_type = customer_type.capitalize()
                    
                    # If this agent matches the target customer type, include their quantity
                    if customer_type == target_type:
                        qty = row.get('purchasing_quantity', 0)
                        quantities.append(qty)
                
                return pd.Series(quantities) if quantities else pd.Series([0])
            
            # Helper function to compute ACTUAL per-period statistics for a customer type
            def get_per_period_stats_by_customer_type(df, target_type, periods, duration_hours):
                """
                Calculate actual per-period statistics for a specific customer type.
                Returns statistics computed on actual per-agent-per-period purchase counts.
                """
                per_period_counts = []
                
                for idx, row in df.iterrows():
                    # Get customer type for this agent
                    customer_type = ''
                    
                    if 'customer_type' in row and pd.notna(row['customer_type']) and str(row['customer_type']).strip():
                        customer_type = str(row['customer_type']).capitalize()
                    else:
                        purchase_requests = row.get('purchase_requests', [])
                        if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
                            first_req = purchase_requests[0]
                            if isinstance(first_req, dict):
                                customer_type = first_req.get('customer_type', 'regular')
                                if isinstance(customer_type, str):
                                    customer_type = customer_type.capitalize()
                    
                    # Only process agents of the target customer type
                    if customer_type == target_type:
                        purchase_requests = row.get('purchase_requests', [])
                        if isinstance(purchase_requests, list):
                            # Initialize period counts for this agent
                            agent_period_counts = {i: 0 for i in range(periods)}
                            
                            for req in purchase_requests:
                                if isinstance(req, dict) and 'timestamp_hours' in req:
                                    timestamp = req['timestamp_hours']
                                    period_idx = int(timestamp // duration_hours)
                                    if 0 <= period_idx < periods:
                                        agent_period_counts[period_idx] += 1
                            
                            # Add all period counts for this agent
                            per_period_counts.extend(agent_period_counts.values())
                
                # Compute statistics on actual per-period values
                if per_period_counts:
                    per_period_series = pd.Series(per_period_counts)
                    return per_period_series.describe()
                else:
                    # Return zero stats if no data
                    return pd.Series({
                        'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                        '25%': 0, '50%': 0, '75%': 0
                    })
            
            # Create three sub-sections
            customer_types_to_analyze = ['Regular', 'Fixed', 'Discount']
            icons = {'Regular': '🔵', 'Fixed': '🟠', 'Discount': '🟢'}
            
            for ctype in customer_types_to_analyze:
                if customer_type_counts.get(ctype, 0) > 0:  # Only show if there are customers of this type
                    st.markdown(f"### {icons[ctype]} {ctype} Customers")
                    
                    # Get quantities for this customer type
                    type_quantities = get_quantities_by_customer_type(df, ctype)
                    
                    if len(type_quantities) > 0 and type_quantities.sum() > 0:
                        # Create two columns: plot and stats
                        col_plot_type, col_stats_type = st.columns([2, 1])
                        
                        with col_plot_type:
                            # Create histogram for this customer type
                            st.markdown(f"**Distribution of Purchase Requests - {ctype} Customers**")
                            type_df = pd.DataFrame({decision_name: type_quantities})
                            
                            fig_type = px.histogram(
                                type_df,
                                x=decision_name,
                                nbins=min(30, int(type_quantities.max()) + 1),
                                labels={decision_name: 'Items per Period', 'count': 'Number of Agents'}
                            )
                            fig_type.update_layout(
                                showlegend=False,
                                xaxis_title="Purchase Requests per Period",
                                yaxis_title="Number of Agents"
                            )
                            st.plotly_chart(fig_type, use_container_width=True)
                        
                        with col_stats_type:
                            st.markdown("**📈 Statistics**")
                            type_stats_desc = type_quantities.describe()
                            
                            # Get ACTUAL per-period statistics for this customer type
                            type_per_period_stats = get_per_period_stats_by_customer_type(
                                df, ctype, periods, duration_hours
                            )
                            
                            type_stats_table = pd.DataFrame({
                                'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
                                'Purchase Requests per Term': [
                                    f"{type_stats_desc['mean']:.2f}",
                                    f"{type_stats_desc['std']:.2f}",
                                    f"{int(type_stats_desc['min'])}",
                                    f"{int(type_stats_desc['max'])}",
                                    f"{type_stats_desc['50%']:.2f}",
                                    f"{type_stats_desc['25%']:.2f}",
                                    f"{type_stats_desc['75%']:.2f}"
                                ],
                                'Purchase Requests per Period': [
                                    f"{type_per_period_stats['mean']:.2f}",
                                    f"{type_per_period_stats['std']:.2f}",
                                    f"{int(type_per_period_stats['min'])}",
                                    f"{int(type_per_period_stats['max'])}",
                                    f"{type_per_period_stats['50%']:.2f}",
                                    f"{type_per_period_stats['25%']:.2f}",
                                    f"{type_per_period_stats['75%']:.2f}"
                                ]
                            })
                            st.dataframe(type_stats_table, use_container_width=True, hide_index=True)
                        
                        # Add agent count
                        st.caption(f"📊 {len(type_quantities)} {ctype.lower()} customers with {int(type_quantities.sum()):,} total purchase requests")
                    else:
                        st.info(f"No purchase data for {ctype} customers")
        else:
            st.info("No purchase data available by customer type")
    
    # Income category analysis if available
    if 'income_category' in df.columns:
        st.markdown("---")
        st.markdown("**📊 Requests by Income Category**")
        
        # Clarification about income category assignment
        st.info(
            "ℹ️ **Note:** Income categories are assigned only to **Discount and Fixed customers** "
            "(who disclosed their income). Regular customers (who did not disclose income) are not "
            "assigned to income categories and instead use the maximum consumption limit.\n\n"
            "**Category Order:** Category 1 = Lowest Income → Higher Categories = Higher Income"
        )
        
        # Filter out rows with None/NaN income_category (Regular customers)
        df_with_category = df[df['income_category'].notna()].copy()
        
        if len(df_with_category) > 0:
            category_stats = df_with_category.groupby('income_category')['purchasing_quantity'].agg([
                ('count', 'count'),
                ('mean', 'mean'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max')
            ]).reset_index()
            
            # Sort by category number (ascending = lowest income first)
            category_stats = category_stats.sort_values('income_category')
            
            category_stats.columns = ['Category', 'Agents', 'Mean Qty', 'Std Dev', 'Min', 'Max']
            category_stats['Mean Qty'] = category_stats['Mean Qty'].round(2)
            category_stats['Std Dev'] = category_stats['Std Dev'].round(2)
            
            # Show count of agents with/without income categories
            agents_with_category = len(df_with_category)
            agents_without_category = len(df) - agents_with_category
            st.caption(f"📊 {agents_with_category} agents with income categories (Discount + Fixed), {agents_without_category} Regular customers (no income category)")
            
            col_table, col_chart = st.columns([1, 2])
            
            with col_table:
                st.dataframe(category_stats, use_container_width=True, hide_index=True)
            
            with col_chart:
                # Box plot by category (sorted properly - ascending = lowest income first)
                df_sorted = df_with_category.sort_values('income_category')
                
                st.markdown("### Purchase Requests Distribution by Income Category")
                st.caption("Category 1 = Lowest Income, Higher Categories = Higher Income")
                fig_box = px.box(
                    df_sorted,
                    x='income_category',
                    y='purchasing_quantity',
                    labels={
                        'income_category': 'Income Category (1=Lowest Income)',
                        'purchasing_quantity': 'Items per Term'
                    },
                    category_orders={"income_category": sorted(df_with_category['income_category'].unique())}
                )
                
                # Ensure all income categories are shown on X axis
                fig_box.update_layout(
                    xaxis=dict(
                        tickmode='linear',
                        dtick=1
                    )
                )
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No agents with income categories found. This can happen if all agents are Regular customers.")
    
    # Purchase request timing analysis if available
    if 'purchase_requests' in df.columns:
        st.markdown("---")
        st.markdown("**⏱️ Purchase Requests and Completed Transactions**")
        
        # Important note about completed transaction data availability
        st.info(
            "ℹ️ **Important Note about Consumption Limits:**\n\n"
            "**Current Default Behavior:** Purchase requests are generated up to the consumption limit "
            "(assuming 100% completion rate). This is simplified behavior when transaction outcomes are not simulated.\n\n"
            "**Reality:** Consumption limits apply to COMPLETED TRANSACTIONS, not to purchase requests. "
            "Agents could make MORE requests than the limit, anticipating some rejections. "
            "For example: if limit=50, an agent could make 100 requests with 50% completion rate = 50 completed transactions (within limit).\n\n"
            "This will be revisited in the future."
        )
        
        # Extract all timestamps and prepare data
        all_timestamps = []
        agent_timelines = []
        
        for idx, requests in enumerate(df['purchase_requests']):
            if isinstance(requests, list) and len(requests) > 0:
                agent_id = df.iloc[idx].get('agent_id', idx + 1)
                for req in requests:
                    if isinstance(req, dict) and 'timestamp_hours' in req:
                        timestamp = req['timestamp_hours']
                        all_timestamps.append(timestamp)
                        agent_timelines.append({
                            'agent_id': agent_id,
                            'timestamp': timestamp
                        })
        
        if len(all_timestamps) > 0:
            # Get simulation parameters for period breakdown
            if hasattr(st.session_state, 'sim_params'):
                periods = st.session_state.sim_params.periods
                duration_hours = st.session_state.sim_params.duration_hours
                term_duration = periods * duration_hours
            else:
                term_duration = max(all_timestamps) if all_timestamps else 30
                periods = 15  # default
                duration_hours = term_duration / periods
            
            # 1. PURCHASES PER PERIOD (Most important visualization)
            st.markdown("**📊 Purchase Volume by Period**")
            st.caption("Shows purchase requests and completed transactions per period")
            
            # Create period bins
            period_bins = []
            period_labels = []
            for i in range(periods):
                start = i * duration_hours
                end = (i + 1) * duration_hours
                period_labels.append(f"P{i+1}")
                period_bins.append(start)
            period_bins.append(term_duration)
            
            # Count purchases per period
            period_counts = pd.cut(all_timestamps, bins=period_bins, labels=period_labels, include_lowest=True)
            purchase_requests_per_period = [sum(period_counts == label) for label in period_labels]
            
            # For now, all purchase requests are completed (100% completion rate)
            # In future versions, this could be different based on rejection logic
            purchases_completed_per_period = purchase_requests_per_period.copy()
            
            # Create DataFrame for the chart
            period_df = pd.DataFrame({
                'Period': period_labels,
                'Purchase Requests': purchase_requests_per_period,
                'Purchases Completed': purchases_completed_per_period
            })
            
            # Grouped bar chart showing both metrics side by side
            st.markdown("### Purchase Requests and Completed Transactions per Period")
            fig_periods = px.bar(
                period_df,
                x='Period',
                y=['Purchase Requests', 'Purchases Completed'],
                labels={'value': 'Count', 'Period': 'Period', 'variable': 'Type'},
                barmode='group',
                color_discrete_sequence=['#1f77b4', '#2ca02c']
            )
            fig_periods.update_layout(
                xaxis_title="Period",
                yaxis_title="Number of Transactions",
                legend_title_text="Transaction Type"
            )
            st.plotly_chart(fig_periods, use_container_width=True)
            
            # Period Details table below the graph
            st.markdown("**Period Details**")
            
            # Create statistics table with Purchase Requests, Purchases Completed, and % Completed
            stats_rows = []
            for i, label in enumerate(period_labels):
                requests = purchase_requests_per_period[i]
                completed = purchases_completed_per_period[i]
                pct_completed = (completed / requests * 100) if requests > 0 else 100.0
                
                stats_rows.append({
                    'Period': label,
                    'Purchase Requests': requests,
                    'Purchases Completed': completed,
                    '% Completed': f"{pct_completed:.1f}%"
                })
            
            # Add TOTAL row
            total_requests = sum(purchase_requests_per_period)
            total_completed = sum(purchases_completed_per_period)
            total_pct = (total_completed / total_requests * 100) if total_requests > 0 else 100.0
            
            stats_rows.append({
                'Period': 'TOTAL',
                'Purchase Requests': total_requests,
                'Purchases Completed': total_completed,
                '% Completed': f"{total_pct:.1f}%"
            })
            
            stats_df = pd.DataFrame(stats_rows)
            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Breakdown by Customer Type
            st.markdown("---")
            st.markdown("**📊 Purchase Requests and Completed Transactions by Customer Type**")
            st.caption("Detailed breakdown for Regular, Fixed, and Discount customers")
            
            st.info(
                "ℹ️ **Note:** Completed transaction data will be extracted from the algorithm once available. "
                "Currently displaying all requests as completed (100%)."
            )
            
            # Extract timestamps by customer type from the customer_type field
            timestamps_by_type = {'Regular': [], 'Fixed': [], 'Discount': []}
            
            for idx, requests in enumerate(df['purchase_requests']):
                if isinstance(requests, list) and len(requests) > 0:
                    for req in requests:
                        if isinstance(req, dict) and 'timestamp_hours' in req:
                            # Get customer_type from request (lowercase: discount, fixed, regular)
                            customer_type = req.get('customer_type', 'regular')
                            if isinstance(customer_type, str):
                                # Normalize to title case for grouping
                                customer_type = customer_type.capitalize()
                            
                            if customer_type in timestamps_by_type:
                                timestamps_by_type[customer_type].append(req['timestamp_hours'])
            
            # Create sub-sections for each customer type
            customer_types_order = ['Regular', 'Fixed', 'Discount']
            icons = {'Regular': '🔵', 'Fixed': '🟠', 'Discount': '🟢'}
            
            for ctype in customer_types_order:
                type_timestamps = timestamps_by_type[ctype]
                
                if len(type_timestamps) > 0:
                    st.markdown(f"### {icons[ctype]} {ctype} Customers")
                    
                    # Count requests per period for this customer type
                    type_period_counts = pd.cut(type_timestamps, bins=period_bins, labels=period_labels, include_lowest=True)
                    type_requests_per_period = [sum(type_period_counts == label) for label in period_labels]
                    
                    # All requests are completed (100% completion rate)
                    type_completed_per_period = type_requests_per_period.copy()
                    
                    # Create DataFrame for the chart
                    type_period_df = pd.DataFrame({
                        'Period': period_labels,
                        'Purchase Requests': type_requests_per_period,
                        'Purchases Completed': type_completed_per_period
                    })
                    
                    # Grouped bar chart for this customer type
                    st.markdown(f"### Purchase Requests and Completed Transactions - {ctype} Customers")
                    fig_type_periods = px.bar(
                        type_period_df,
                        x='Period',
                        y=['Purchase Requests', 'Purchases Completed'],
                        labels={'value': 'Count', 'Period': 'Period', 'variable': 'Type'},
                        barmode='group',
                        color_discrete_sequence=['#1f77b4', '#2ca02c']
                    )
                    fig_type_periods.update_layout(
                        xaxis_title="Period",
                        yaxis_title="Number of Transactions",
                        legend_title_text="Transaction Type"
                    )
                    st.plotly_chart(fig_type_periods, use_container_width=True)
                    
                    # Period Details table below the graph
                    st.markdown("**Period Details**")
                    
                    # Create statistics table
                    type_stats_rows = []
                    for i, label in enumerate(period_labels):
                        requests = type_requests_per_period[i]
                        completed = type_completed_per_period[i]
                        pct_completed = (completed / requests * 100) if requests > 0 else 100.0
                        
                        type_stats_rows.append({
                            'Period': label,
                            'Purchase Requests': requests,
                            'Purchases Completed': completed,
                            '% Completed': f"{pct_completed:.1f}%"
                        })
                    
                    # Add TOTAL row
                    type_total_requests = sum(type_requests_per_period)
                    type_total_completed = sum(type_completed_per_period)
                    type_total_pct = (type_total_completed / type_total_requests * 100) if type_total_requests > 0 else 100.0
                    
                    type_stats_rows.append({
                        'Period': 'TOTAL',
                        'Purchase Requests': type_total_requests,
                        'Purchases Completed': type_total_completed,
                        '% Completed': f"{type_total_pct:.1f}%"
                    })
                    
                    type_stats_df = pd.DataFrame(type_stats_rows)
                    st.dataframe(
                        type_stats_df,
                        use_container_width=True,
                        hide_index=True
                    )
        
        else:
            st.info("No purchase requests found in the data")
    
    # Default behavior explanation
    with st.expander("ℹ️ How This Decision Works (Default Behavior)", expanded=False):
        st.markdown("""
        **Purchasing Quantity Default Logic:**
        
        1. **Income Category Assignment**: 
           - The income range is split into NFIC equal intervals
           - **Category 1 = Lowest Income** → **Category N = Highest Income**
           - **Discount and Fixed customers** (who disclosed income) are assigned to categories based on their income level
           - **Regular customers** (who did not disclose income) are NOT assigned to income categories
           - Example: If NFIC=10 and range is [$0-$100k]:
             - Category 1 = [$0-$10k] (Lowest income)
             - Category 2 = [$10k-$20k]
             - ...
             - Category 10 = [$90k-$100k] (Highest income)
        
        2. **Purchasing Limit**: 
           - **Discount customers**: Use purchasing limit from **Category 1** (lowest income category)
           - **Regular customers**: Use purchasing limit from **Category N** (highest income category)
           - **Fixed customers**: Use purchasing limit from their actual income category
           - If limits disabled: Uses `max_purchases_per_term` fallback
        
        3. **Total Quantity**: Random integer uniformly distributed in [0, limit]
           - ⚠️ **Important**: In default mode, this limits REQUESTS (assuming 100% completion)
           - **Reality**: The limit should apply to COMPLETED TRANSACTIONS, not requests
           - **Example**: Agent could make 100 requests with 50% completion = 50 transactions (within 50 limit)
           - This will be revisited in the future
        
        4. **Purchase Requests**: 
           - Number of requests = total quantity
           - Each request = 1 item (for defaults)
           - Timestamps randomly distributed across term duration
        
        **Professor's Specification**: 
        "The income range is split into equal intervals. Discount and Fixed customers (who 
        disclosed their income) are assigned to income categories based on their income level. 
        Regular customers (who did not disclose income) are not assigned to income categories 
        and instead use the maximum consumption limit by default. The total quantity is a 
        random number between 0 and the purchasing limit, with each purchase order for 1 item 
        by default, randomly spread during the term."
        """)
    
    # Export section for purchasing quantity / transactions
    if 'purchase_requests' in df.columns:
        st.markdown("---")
        st.markdown("**📥 Export Transaction Data**")
        
        try:
            # Flatten purchase_requests to transaction-level DataFrame
            transactions = []
            # Use centralized timestamp converter for consistent handling
            ts_converter = TimestampConverter()
            
            for idx, row in df.iterrows():
                purchase_requests = row.get('purchase_requests', [])
                if isinstance(purchase_requests, list):
                    for req in purchase_requests:
                        if isinstance(req, dict):
                            # Get timestamp_hours and convert using centralized utilities
                            timestamp_hours = req.get('timestamp_hours', 0.0)
                            ts_result = ts_converter.convert(timestamp_hours)
                            
                            period = ts_result['period']
                            timestamp_str = ts_result['formatted']
                            
                            transactions.append({
                                'transaction_id': req.get('transaction_id'),
                                'customer_id': req.get('customer_id', idx + 1),
                                'vendorID': req.get('vendorID', 1),
                                'platformProductID': req.get('platformProductID', 1),
                                'purchase type': req.get('platformPrice', 'N/A'),
                                'purchase_bid_value': req.get('bid_value', 'N/A'),
                                'Purchase Timestamp': timestamp_str,
                                'Period': period,
                                'timestamp_hours': timestamp_hours  # Keep for sorting
                            })
            
            if len(transactions) > 0:
                transactions_df = pd.DataFrame(transactions)
                
                # CRITICAL: Sort by timestamp across ALL customers
                transactions_df['timestamp_hours'] = pd.to_numeric(transactions_df['timestamp_hours'], errors='coerce')
                transactions_df = transactions_df.sort_values(
                    by='timestamp_hours', 
                    ascending=True,
                    na_position='last'
                ).reset_index(drop=True).copy()
                
                # Handle transaction_id
                # If IDs were pre-assigned (central system), use them. Otherwise generate them.
                if 'transaction_id' in transactions_df.columns and not transactions_df['transaction_id'].isnull().all():
                    # Move transaction_id to first column
                    cols = ['transaction_id'] + [c for c in transactions_df.columns if c != 'transaction_id']
                    transactions_df = transactions_df[cols]
                else:
                    # Fallback: Generate sequential IDs if missing
                    if 'transaction_id' in transactions_df.columns:
                        transactions_df = transactions_df.drop(columns=['transaction_id'])
                    transactions_df.insert(0, 'transaction_id', range(1, len(transactions_df) + 1))
                
                # Drop timestamp_hours column before display/export
                transactions_df = transactions_df.drop(columns=['timestamp_hours'])
                
                col_export, col_preview = st.columns([1, 2])
                
                with col_export:
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        transactions_df.to_excel(writer, index=False, sheet_name='Transactions')
                        # Apply 2-decimal formatting
                        _apply_price_formatting_purchasing(writer, 'Transactions', transactions_df)
                    
                    st.download_button(
                        label="📊 Download Transactions Excel",
                        data=buffer.getvalue(),
                        file_name=f"purchasing_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download transaction-level data with one row per purchase request"
                    )
                    
                    st.caption(f"📋 {len(transactions_df):,} transactions from {len(df):,} agents")
                
                with col_preview:
                    with st.expander("📋 Preview Transaction Data", expanded=False):
                        st.dataframe(transactions_df.head(20), use_container_width=True)
                        st.caption(f"Showing first 20 of {len(transactions_df):,} total transactions")
            else:
                st.info("No transactions to export")
        
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")
        
        # Agent-Level Export
        st.markdown("---")
        st.markdown("**📊 Export Agent-Level Purchasing Information**")
        st.caption("Download aggregated purchasing data at the agent level with breakdown by period")
        
        try:
            # Get simulation parameters for period breakdown
            if hasattr(st.session_state, 'sim_params'):
                periods = st.session_state.sim_params.periods
                duration_hours = st.session_state.sim_params.duration_hours
            else:
                periods = 15
                duration_hours = 2.0
            
            # Build agent-level data
            agent_level_data = []
            
            for idx, row in df.iterrows():
                agent_id = row.get('agent_id', idx + 1)
                allowance_level = row.get('Assigned Allowance Level', '')
                group_experiment = row.get('Group_experiment', '')
                income_category = row.get('income_category', '')
                # Handle None/NaN - display as 'N/A' for Regular customers who don't have income categories
                if pd.isna(income_category) or income_category is None:
                    income_category = 'N/A'
                
                # Get customer type - try direct column first, then purchase_requests as fallback
                customer_type = ''
                
                # Priority 1: Check if customer_type is directly available in the dataframe
                if 'customer_type' in row and pd.notna(row['customer_type']) and str(row['customer_type']).strip():
                    customer_type = str(row['customer_type']).capitalize()
                else:
                    # Priority 2: Extract from purchase_requests if available
                    purchase_requests = row.get('purchase_requests', [])
                    if isinstance(purchase_requests, list) and len(purchase_requests) > 0:
                        first_req = purchase_requests[0]
                        if isinstance(first_req, dict):
                            customer_type = first_req.get('customer_type', '')
                            if isinstance(customer_type, str):
                                customer_type = customer_type.capitalize()
                
                # Get purchase_requests for counting
                purchase_requests = row.get('purchase_requests', [])
                
                # Total counts
                total_requests = len(purchase_requests) if isinstance(purchase_requests, list) else 0
                total_completed = total_requests  # All requests are completed (100%)
                pct_completed = 100.0 if total_requests > 0 else 0.0
                
                # Add overall record
                agent_level_data.append({
                    'Agent ID': agent_id,
                    'Assigned Allowance Level': allowance_level,
                    'Group_experiment': group_experiment,
                    'Customer Type': customer_type,
                    'Income Category': income_category,
                    'Count of Purchase Requests': total_requests,
                    'Count of Completed Transactions': total_completed,
                    '% Completed Transactions': f"{pct_completed:.1f}%",
                    'Period': 'Total'
                })
                
                # Breakdown by period
                if isinstance(purchase_requests, list):
                    # Count requests per period for this agent
                    period_counts = {f"P{i+1}": 0 for i in range(periods)}
                    
                    for req in purchase_requests:
                        if isinstance(req, dict) and 'timestamp_hours' in req:
                            timestamp = req['timestamp_hours']
                            # Determine which period this request belongs to
                            period_idx = int(timestamp // duration_hours)
                            if 0 <= period_idx < periods:
                                period_label = f"P{period_idx + 1}"
                                period_counts[period_label] += 1
                    
                    # Add one record per period for this agent
                    for period_label, count in period_counts.items():
                        completed = count  # All requests are completed
                        pct = 100.0 if count > 0 else 0.0
                        
                        agent_level_data.append({
                            'Agent ID': agent_id,
                            'Assigned Allowance Level': allowance_level,
                            'Group_experiment': group_experiment,
                            'Customer Type': customer_type,
                            'Income Category': income_category,
                            'Count of Purchase Requests': count,
                            'Count of Completed Transactions': completed,
                            '% Completed Transactions': f"{pct:.1f}%",
                            'Period': period_label
                        })
            
            if len(agent_level_data) > 0:
                agent_df = pd.DataFrame(agent_level_data)
                
                # Create multi-sheet Excel
                buffer_agent = BytesIO()
                with pd.ExcelWriter(buffer_agent, engine='openpyxl') as writer:
                    # Sheet 1: Total (all agents, total across all periods)
                    total_df = agent_df[agent_df['Period'] == 'Total'].drop(columns=['Period'])
                    total_df.to_excel(writer, index=False, sheet_name='Total')
                    # Apply 2-decimal formatting
                    _apply_price_formatting_purchasing(writer, 'Total', total_df)
                    
                    # Additional sheets: One per Period
                    period_labels = [f"P{i+1}" for i in range(periods)]
                    for period_label in period_labels:
                        period_df = agent_df[agent_df['Period'] == period_label].drop(columns=['Period'])
                        if len(period_df) > 0:
                            period_df.to_excel(writer, index=False, sheet_name=period_label)
                            # Apply 2-decimal formatting to each period sheet
                            _apply_price_formatting_purchasing(writer, period_label, period_df)
                
                col_download_agent, col_info_agent = st.columns([1, 2])
                
                with col_download_agent:
                    st.download_button(
                        label="📥 Download Agent-Level Excel",
                        data=buffer_agent.getvalue(),
                        file_name=f"agent_level_purchases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download agent-level purchasing information with Total + Period breakdown"
                    )
                
                with col_info_agent:
                    num_sheets = 1 + periods
                    st.caption(f"📋 {len(df):,} agents across {num_sheets} sheets (Total + {periods} Periods)")
                    st.caption("✅ Each sheet contains: Agent ID, Allowance Level, Group, Customer Type, Income Category, Requests, Completed, % Completed")
            else:
                st.info("No agent data to export")
        
        except ImportError:
            st.caption("⚠️ Excel export requires openpyxl")
        except Exception as e:
            st.error(f"⚠️ Error creating agent-level export: {str(e)}")


def render_purchasing_frequency(df, decision_name, decision_title, decision_data):
    """Visualization for purchasing_frequency - shows WHEN purchases occur (timing/frequency)"""
    
    # Check if purchase_requests data is available
    if 'purchase_requests' not in df.columns:
        st.warning("No purchase_requests data available for frequency visualization")
        return
    
    # Get simulation parameters
    if hasattr(st.session_state, 'sim_params'):
        periods = st.session_state.sim_params.periods
        duration_hours = st.session_state.sim_params.duration_hours
        term_duration = periods * duration_hours
    else:
        term_duration = 30  # Default
        periods = 15
        duration_hours = 2.0
    
    # Extract all timestamps from purchase_requests for analysis
    all_timestamps = []
    
    for idx, row in df.iterrows():
        requests = row.get('purchase_requests', [])
        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict):
                    if 'timestamp_hours' in req:
                        all_timestamps.append(req['timestamp_hours'])
    
    if len(all_timestamps) == 0:
        st.info("No purchase requests found")
        return
    
    # Display Purchase Decisions per Request breakdown
    st.markdown("### 🛒 Purchase Requests per Type")
    st.caption("Breakdown of all purchase requests by customer type and pricing model")
    
    # Count by customer_type field directly (not platformPrice which may not exist)
    from collections import Counter
    customer_type_counts = Counter()
    
    for idx, row in df.iterrows():
        requests = row.get('purchase_requests', [])
        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict):
                    # Get customer_type from request (lowercase: discount, fixed, regular)
                    customer_type = req.get('customer_type', 'regular')
                    if isinstance(customer_type, str):
                        # Normalize to lowercase for counting
                        customer_type = customer_type.lower()
                        customer_type_counts[customer_type] += 1
    
    total_requests = sum(customer_type_counts.values())
    discount_count = customer_type_counts.get('discount', 0)
    fixed_count = customer_type_counts.get('fixed', 0)
    regular_count = customer_type_counts.get('regular', 0)
    
    # Overall metrics showing breakdown by customer type
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", f"{total_requests:,}", help="All purchase requests across all agents")
    with col2:
        discount_pct = (discount_count/total_requests*100) if total_requests > 0 else 0
        st.metric("Discount Requests", 
                 f"{discount_count:,}",
                 f"↗ {discount_pct:.1f}%")
    with col3:
        fixed_pct = (fixed_count/total_requests*100) if total_requests > 0 else 0
        st.metric("Fixed Requests", 
                 f"{fixed_count:,}",
                 f"↗ {fixed_pct:.1f}%")
    with col4:
        regular_pct = (regular_count/total_requests*100) if total_requests > 0 else 0
        st.metric("Regular Requests", 
                 f"{regular_count:,}",
                 f"↗ {regular_pct:.1f}%")
    
    # # Add completed transactions metrics (all requests are considered completed transactions)
    # st.markdown("---")
    # st.markdown("### ✅ Completed Transactions")
    # st.caption("All purchase requests result in completed transactions in this simulation")
    # 
    # col1, col2, col3, col4 = st.columns(4)
    # 
    # with col1:
    #     st.metric("Total Transactions", f"{total_requests:,}", 
    #              help="Total completed transactions (same as total requests)")
    # with col2:
    #     st.metric("Discount Transactions", 
    #              f"{discount_count:,}",
    #              f"↗ {discount_pct:.1f}%")
    # with col3:
    #     st.metric("Fixed Transactions", 
    #              f"{fixed_count:,}",
    #              f"↗ {fixed_pct:.1f}%")
    # with col4:
    #     st.metric("Regular Transactions", 
    #              f"{regular_count:,}",
    #              f"↗ {regular_pct:.1f}%")
    
    # MAIN VISUALIZATION: Sample Agent Purchase Schedules (Timeline)
    st.markdown("---")
    st.markdown("**👥 Sample Agent Purchase Schedules**")
    st.caption("Individual agent timelines showing random distribution of their purchases")
    
    # Select up to 20 agents with most purchases for visualization
    agent_purchase_counts = df.groupby(df.index)['purchasing_quantity'].first().sort_values(ascending=False)
    sample_agents = agent_purchase_counts.head(20).index.tolist()
    
    timeline_data = []
    for idx in sample_agents:
        requests = df.iloc[idx]['purchase_requests']
        agent_id = df.iloc[idx].get('agent_id', idx + 1)
        quantity = df.iloc[idx].get('purchasing_quantity', 0)
        
        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict) and 'timestamp_hours' in req:
                    timeline_data.append({
                        'Agent': f"Agent {agent_id} ({quantity} items)",
                        'Time': req['timestamp_hours'],
                        'Purchase': 1
                    })
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        
        st.markdown(f"### Purchase Timing for Top {len(sample_agents)} Agents (by quantity)")
        fig_timeline = px.scatter(
            timeline_df,
            x='Time',
            y='Agent',
            labels={'Time': 'Time (hours)', 'Agent': 'Agent ID'},
            color_discrete_sequence=['#1f77b4']
        )
        
        # Add period markers
        for i in range(1, periods):
            fig_timeline.add_vline(
                x=i * duration_hours,
                line_dash="dot",
                line_color="gray",
                opacity=0.3
            )
        
        fig_timeline.update_traces(marker=dict(size=8, symbol='line-ns-open'))
        fig_timeline.update_layout(
            xaxis_title="Time (hours from term start)",
            yaxis_title="",
            height=max(400, len(sample_agents) * 25),
            showlegend=False
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.markdown("""
    **What this shows:**
    - Each horizontal line represents one agent
    - Each vertical tick mark is a purchase request
    - Purchases are randomly distributed across the term duration (not evenly spaced)
    - Different agents have different frequencies based on their purchasing quantity
    """)

