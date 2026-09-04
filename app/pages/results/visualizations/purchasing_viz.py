# app/pages/results/visualizations/purchasing_viz.py
"""
Purchasing-related visualization functions.
Handles purchasing_quantity and purchasing_frequency decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from app.reports.purchasing import (
    agent_level_purchases_xlsx,
    build_agent_level_purchases,
    build_agent_timeline,
    build_transaction_export,
    collect_purchase_timestamps,
    collect_timestamps,
    count_requests_by_customer_type,
    count_requests_by_customer_type_lower,
    counts_per_period,
    customer_type_quantity_stats_frame,
    customer_type_stats_frame,
    income_category_stats_frame,
    period_bins_and_labels,
    period_details_frame,
    purchasing_transactions_xlsx,
    quantities_by_customer_type,
    quantity_stats_frame,
    timestamps_by_customer_type,
    top_agents_by_quantity,
)
from app.utils.timestamp_utils import (
    get_duration_hours,
    get_periods,
    get_simulation_base_time,
)


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
        # Get number of periods from simulation config
        if hasattr(st.session_state, 'sim_params'):
            periods = st.session_state.sim_params.periods
        else:
            periods = 15  # default
        
        stats_df = quantity_stats_frame(decision_data, periods)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Customer Type Breakdown - Purchase Requests by customer type
    if 'purchase_requests' in df.columns:
        st.markdown("---")
        st.markdown("**🎯 Purchase Requests by Customer Type**")
        st.caption("Distribution of total purchase requests across Regular, Fixed, and Discount customers")
        
        # Extract customer type from purchase_requests
        customer_type_counts = count_requests_by_customer_type(df)
        
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
                type_stats_df = customer_type_stats_frame(customer_type_counts, total_purchases_by_type)
                st.dataframe(type_stats_df, use_container_width=True, hide_index=True)
            
            # Now create three sub-sections, one for each customer type
            st.markdown("---")
            st.markdown("**📊 Detailed Analysis by Customer Type**")
            st.caption("Purchase request quantity distribution and statistics for each customer type")
            
            # Get number of periods for per-period calculations
            if hasattr(st.session_state, 'sim_params'):
                periods = st.session_state.sim_params.periods
            else:
                periods = 15  # default
            
            
            # Create three sub-sections
            customer_types_to_analyze = ['Regular', 'Fixed', 'Discount']
            icons = {'Regular': '🔵', 'Fixed': '🟠', 'Discount': '🟢'}
            
            for ctype in customer_types_to_analyze:
                if customer_type_counts.get(ctype, 0) > 0:  # Only show if there are customers of this type
                    st.markdown(f"### {icons[ctype]} {ctype} Customers")
                    
                    # Get quantities for this customer type
                    type_quantities = quantities_by_customer_type(df, ctype)
                    
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
                            type_stats_table = customer_type_quantity_stats_frame(type_quantities, periods)
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
            category_stats = income_category_stats_frame(df_with_category)
            
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
                
                st.markdown("### Quantity Distribution by Income Category")
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
        all_timestamps, agent_timelines = collect_purchase_timestamps(df)
        
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
            period_bins, period_labels = period_bins_and_labels(periods, duration_hours, term_duration)
            
            # Count purchases per period
            purchase_requests_per_period = counts_per_period(all_timestamps, period_bins, period_labels)
            
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
            stats_df = period_details_frame(
                period_labels, purchase_requests_per_period, purchases_completed_per_period
            )
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
            timestamps_by_type = timestamps_by_customer_type(df)
            
            # Create sub-sections for each customer type
            customer_types_order = ['Regular', 'Fixed', 'Discount']
            icons = {'Regular': '🔵', 'Fixed': '🟠', 'Discount': '🟢'}
            
            for ctype in customer_types_order:
                type_timestamps = timestamps_by_type[ctype]
                
                if len(type_timestamps) > 0:
                    st.markdown(f"### {icons[ctype]} {ctype} Customers")
                    
                    # Count requests per period for this customer type
                    type_requests_per_period = counts_per_period(type_timestamps, period_bins, period_labels)
                    
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
                    type_stats_df = period_details_frame(
                        period_labels, type_requests_per_period, type_completed_per_period
                    )
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
            # Flatten purchase_requests to a transaction-level DataFrame
            transactions_df = build_transaction_export(
                df, get_simulation_base_time(), get_duration_hours(), get_periods()
            )
            
            if transactions_df is not None:
                
                col_export, col_preview = st.columns([1, 2])
                
                with col_export:
                    xlsx_bytes = purchasing_transactions_xlsx(transactions_df)
                    
                    st.download_button(
                        label="📊 Download Transactions Excel",
                        data=xlsx_bytes,
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
            agent_level_data = build_agent_level_purchases(df, periods, duration_hours)
            
            if len(agent_level_data) > 0:
                agent_df = pd.DataFrame(agent_level_data)
                
                # Create multi-sheet Excel
                xlsx_bytes = agent_level_purchases_xlsx(agent_df, periods)
                
                col_download_agent, col_info_agent = st.columns([1, 2])
                
                with col_download_agent:
                    st.download_button(
                        label="📥 Download Agent-Level Excel",
                        data=xlsx_bytes,
                        file_name=f"agent_level_purchases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download agent-level purchasing information with Total + Period breakdown"
                    )
                
                with col_info_agent:
                    num_sheets = 1 + periods
                    st.caption(f"📋 {len(df):,} agents across {num_sheets} sheets (Total + {periods} Periods)")
                    st.caption("✅ Each sheet contains: Agent ID, Honesty_Humility, Assigned Allowance Level, Study Program, Group_experiment, TWT+Sospeso, income, Customer Type, Income Category, Requests, Completed, % Completed")
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
    all_timestamps = collect_timestamps(df)
    
    if len(all_timestamps) == 0:
        st.info("No purchase requests found")
        return
    
    # Display Purchase Decisions per Request breakdown
    st.markdown("### 🛒 Purchase Requests per Type")
    st.caption("Breakdown of all purchase requests by customer type and pricing model")
    
    # Count by customer_type field directly (not platformPrice which may not exist)
    customer_type_counts = count_requests_by_customer_type_lower(df)
    
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
    sample_agents = top_agents_by_quantity(df, 20)
    
    timeline_data = build_agent_timeline(df, sample_agents)
    
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

