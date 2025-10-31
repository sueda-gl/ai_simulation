# app/pages/results/visualizations/consumption_viz.py
"""
Consumption-related visualization functions.
Handles consumption_quantity and consumption_frequency decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from datetime import datetime, timedelta


def render_consumption_quantity(df, decision_name, decision_title, decision_data):
    """Visualization for consumption_quantity - quantity analysis with purchase requests"""
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        mean_qty = decision_data.mean()
        st.metric("Mean Quantity", f"{mean_qty:.1f}")
    
    with col3:
        total_purchases = decision_data.sum()
        st.metric("Total Purchases", f"{int(total_purchases):,}")
    
    with col4:
        agents_with_purchases = (decision_data > 0).sum()
        pct_with_purchases = agents_with_purchases / len(decision_data) * 100
        st.metric("Agents w/ Purchases", f"{pct_with_purchases:.1f}%")
    
    # Distribution plot and statistics
    col_plot, col_stats = st.columns([2, 1])
    
    with col_plot:
        # Histogram of consumption quantities
        fig = px.histogram(
            df,
            x=decision_name,
            nbins=min(30, int(decision_data.max()) + 1),
            title="Distribution of Consumption Quantities",
            labels={decision_name: 'Items per Term', 'count': 'Number of Agents'}
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Items Purchased per Term",
            yaxis_title="Number of Agents"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
        st.markdown("**📈 Statistics**")
        stats = decision_data.describe()
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
            'Value': [
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{int(stats['min'])}",
                f"{int(stats['max'])}",
                f"{stats['50%']:.2f}",
                f"{stats['25%']:.2f}",
                f"{stats['75%']:.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Income category analysis if available
    if 'income_category' in df.columns:
        st.markdown("---")
        st.markdown("**📊 Quantity by Income Category**")
        
        category_stats = df.groupby('income_category')['consumption_quantity'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max')
        ]).reset_index()
        
        category_stats.columns = ['Category', 'Agents', 'Mean Qty', 'Std Dev', 'Min', 'Max']
        category_stats['Mean Qty'] = category_stats['Mean Qty'].round(2)
        category_stats['Std Dev'] = category_stats['Std Dev'].round(2)
        
        col_table, col_chart = st.columns([1, 2])
        
        with col_table:
            st.dataframe(category_stats, use_container_width=True, hide_index=True)
        
        with col_chart:
            # Box plot by category (sorted properly)
            # Sort DataFrame by income_category to ensure proper ordering
            df_sorted = df.sort_values('income_category')
            
            fig_box = px.box(
                df_sorted,
                x='income_category',
                y='consumption_quantity',
                title="Quantity Distribution by Income Category",
                labels={
                    'income_category': 'Income Category',
                    'consumption_quantity': 'Items per Term'
                },
                category_orders={"income_category": sorted(df['income_category'].unique())}
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Purchase request timing analysis if available
    if 'purchase_requests' in df.columns:
        st.markdown("---")
        st.markdown("**⏱️ Purchase Timing & Frequency Analysis**")
        
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
            st.caption("Shows how many purchases occur in each period - demonstrates random distribution")
            
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
            period_df = pd.DataFrame({
                'Period': period_labels,
                'Purchases': [sum(period_counts == label) for label in period_labels],
                'Hours': [f"{i*duration_hours:.0f}-{(i+1)*duration_hours:.0f}" for i in range(periods)]
            })
            
            col_period1, col_period2 = st.columns([3, 1])
            
            with col_period1:
                # Bar chart by period
                fig_periods = px.bar(
                    period_df,
                    x='Period',
                    y='Purchases',
                    title=f"Purchase Requests per Period (Total: {len(all_timestamps):,} requests)",
                    labels={'Purchases': 'Number of Requests', 'Period': 'Period'},
                    text='Purchases'
                )
                fig_periods.update_traces(textposition='outside')
                fig_periods.update_layout(
                    xaxis_title="Period (Time Window)",
                    yaxis_title="Number of Purchase Requests",
                    showlegend=False
                )
                st.plotly_chart(fig_periods, use_container_width=True)
            
            with col_period2:
                st.markdown("**Period Details**")
                st.dataframe(
                    period_df.rename(columns={'Hours': 'Time Range'}),
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            
            # 2. CUMULATIVE PURCHASES OVER TIME
            st.markdown("---")
            st.markdown("**📈 Cumulative Purchases Over Time**")
            st.caption("Shows how total purchases accumulate throughout the term")
            
            # Sort timestamps and create cumulative count
            sorted_times = sorted(all_timestamps)
            cumulative_counts = list(range(1, len(sorted_times) + 1))
            
            cumulative_df = pd.DataFrame({
                'Time (hours)': sorted_times,
                'Cumulative Purchases': cumulative_counts
            })
            
            fig_cumulative = px.line(
                cumulative_df,
                x='Time (hours)',
                y='Cumulative Purchases',
                title="Cumulative Purchase Requests Over Time"
            )
            
            # Add period markers
            for i in range(1, periods):
                fig_cumulative.add_vline(
                    x=i * duration_hours,
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text=f"P{i+1}",
                    annotation_position="top"
                )
            
            fig_cumulative.update_layout(
                    xaxis_title="Time (hours from term start)",
                yaxis_title="Total Purchases",
                showlegend=False
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)
            
            st.info("💡 To see **individual agent purchase schedules** (frequency visualization), view the **Consumption Frequency** decision.")
        
        else:
            st.info("No purchase requests found in the data")
    
    # Default behavior explanation
    with st.expander("ℹ️ How This Decision Works (Default Behavior)", expanded=False):
        st.markdown("""
        **Consumption Quantity Default Logic:**
        
        1. **Income Category Assignment**: Each agent is assigned to an income category (1 to NFIC) based on:
           - The income range is split into NFIC equal intervals
           - All customers (discount, fixed, regular) are assigned to categories based on their income
           - **No distinction by customer type** during category assignment
           - Example: If NFIC=10 and range is [$0-$100k], Category 1 = [$0-$10k], Category 2 = [$10k-$20k], etc.
        
        2. **Consumption Limit**: 
           - **Discount customers**: Use consumption limit from Category 1 (lowest)
           - **Regular customers**: Use consumption limit from Category 10 (highest)
           - **Fixed customers**: Use consumption limit from their actual income category
           - If limits disabled: Uses `max_purchases_per_term` fallback
        
        3. **Total Quantity**: Random integer uniformly distributed in [0, limit]
        
        4. **Purchase Requests**: 
           - Number of requests = total quantity
           - Each request = 1 item (for defaults)
           - Timestamps randomly distributed across term duration
        
        **Professor's Specification**: 
        "The income range is split into equal intervals. All customers with income within 
        the corresponding interval are assigned to it irrespective of their type. The total 
        quantity is a random number between 0 and the consumption limit, with each purchase 
        order for 1 item by default, randomly spread during the term."
        """)
    
    # Export section for consumption quantity / transactions
    if 'purchase_requests' in df.columns:
        st.markdown("---")
        st.markdown("**📥 Export Transaction Data**")
        
        try:
            # Flatten purchase_requests to transaction-level DataFrame
            transactions = []
            # Base date for timestamp conversion (current date when simulation is run)
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            for idx, row in df.iterrows():
                purchase_requests = row.get('purchase_requests', [])
                if isinstance(purchase_requests, list):
                    for req in purchase_requests:
                        if isinstance(req, dict):
                            # Convert timestamp_hours to datetime format
                            timestamp_hours = req.get('timestamp_hours', 0.0)
                            timestamp_dt = base_date + timedelta(hours=float(timestamp_hours))
                            timestamp_str = timestamp_dt.strftime('%d/%m/%Y %H:%M')
                            
                            transactions.append({
                                'customer_id': req.get('customer_id', idx + 1),
                                'vendorID': req.get('vendorID', 1),
                                'platformProductID': req.get('platformProductID', 1),
                                'platformPrice': req.get('platformPrice', 'N/A'),
                                'purchase_bid_value': req.get('bid_value', 'N/A'),
                                'timestamp': timestamp_str,
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
                
                # Add transaction_id AFTER sorting
                transactions_df.insert(0, 'transaction_id', range(1, len(transactions_df) + 1))
                
                # Drop timestamp_hours column before display/export
                transactions_df = transactions_df.drop(columns=['timestamp_hours'])
                
                col_export, col_preview = st.columns([1, 2])
                
                with col_export:
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        transactions_df.to_excel(writer, index=False, sheet_name='Transactions')
                    
                    st.download_button(
                        label="📊 Download Transactions Excel",
                        data=buffer.getvalue(),
                        file_name=f"consumption_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
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


def render_consumption_frequency(df, decision_name, decision_title, decision_data):
    """Visualization for consumption_frequency - shows WHEN purchases occur (timing/frequency)"""
    
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
    
    # Extract all timestamps from purchase_requests
    all_timestamps = []
    for idx, row in df.iterrows():
        requests = row.get('purchase_requests', [])
        if isinstance(requests, list):
            for req in requests:
                if isinstance(req, dict) and 'timestamp_hours' in req:
                    all_timestamps.append(req['timestamp_hours'])
    
    if len(all_timestamps) == 0:
        st.info("No purchase requests found")
        return
    
    # MAIN VISUALIZATION: Sample Agent Purchase Schedules (Timeline)
    st.markdown("---")
    st.markdown("**👥 Sample Agent Purchase Schedules**")
    st.caption("Individual agent timelines showing random distribution of their purchases")
    
    # Select up to 20 agents with most purchases for visualization
    agent_purchase_counts = df.groupby(df.index)['consumption_quantity'].first().sort_values(ascending=False)
    sample_agents = agent_purchase_counts.head(20).index.tolist()
    
    timeline_data = []
    for idx in sample_agents:
        requests = df.iloc[idx]['purchase_requests']
        agent_id = df.iloc[idx].get('agent_id', idx + 1)
        quantity = df.iloc[idx].get('consumption_quantity', 0)
        
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
        
        fig_timeline = px.scatter(
            timeline_df,
            x='Time',
            y='Agent',
            title=f"Purchase Timing for Top {len(sample_agents)} Agents (by quantity)",
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
    - Different agents have different frequencies based on their consumption quantity
    """)

