# app/pages/results/visualizations/transaction_viz.py
"""
Transaction and purchase-related visualization functions.
Handles purchase_vs_bid, rejected_transaction_defaults, and rejected_transaction_option decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter


def render_purchase_vs_bid(df, decision_name, decision_title, decision_data):
    """Visualization for purchase_vs_bid - per-request Purchase Now/bid choices"""
    
    # Show that decisions are now made PER REQUEST
    st.info("⚠️ **Note**: Decisions are made **per purchase request**, not per agent. A single agent can choose differently for each purchase.")
    
    # Analyze customer type distribution at agent level
    if 'customer_type' in df.columns:
        from src.decisions.income_utils import analyze_customer_types
        customer_stats = analyze_customer_types(df)
        
        # Show customer type breakdown
        type_col1, type_col2, type_col3, type_col4 = st.columns(4)
        
        with type_col1:
            st.metric("Total Agents", f"{customer_stats['total']:,}")
        with type_col2:
            st.metric("Regular Customers", 
                     f"{customer_stats['regular']['count']:,} ({customer_stats['regular']['percentage']:.1f}%)",
                     help="Only these customers make Purchase Now vs Bid choice")
        with type_col3:
            st.metric("Fixed Customers", 
                     f"{customer_stats['fixed']['count']:,} ({customer_stats['fixed']['percentage']:.1f}%)",
                     help="Use fixed pricing only (NA)")
        with type_col4:
            st.metric("Discount Customers", 
                     f"{customer_stats['discount']['count']:,} ({customer_stats['discount']['percentage']:.1f}%)",
                     help="Use discount pricing (NA)")
        
        # Add pie chart showing customer type distribution
        st.markdown("---")
        st.markdown("### 👥 Customer Type Distribution")
        
        col_pie, col_table = st.columns([2, 1])
        
        with col_pie:
            # Create pie chart for customer types
            customer_types_data = {
                'Customer Type': ['Regular Customers', 'Fixed Customers', 'Discount Customers'],
                'Count': [
                    customer_stats['regular']['count'],
                    customer_stats['fixed']['count'],
                    customer_stats['discount']['count']
                ]
            }
            
            fig = px.pie(
                values=customer_types_data['Count'],
                names=customer_types_data['Customer Type'],
                title=f"Customer Type Breakdown ({customer_stats['total']:,} total agents)",
                hole=0.4,  # Donut chart
                color_discrete_map={
                    'Regular Customers': '#2196F3',  # Blue
                    'Fixed Customers': '#9C27B0',     # Purple
                    'Discount Customers': '#FF5722'   # Red
                }
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>%{value:,} agents<br>%{percent}<extra></extra>'
            )
            fig.update_layout(
                showlegend=True,
                height=400,
                margin=dict(t=60, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_table:
            st.markdown("**📊 Customer Type Summary**")
            st.caption("Breakdown by pricing model")
            
            # Create summary table
            summary_df = pd.DataFrame({
                'Type': ['Regular', 'Fixed', 'Discount'],
                'Agents': [
                    f"{customer_stats['regular']['count']:,}",
                    f"{customer_stats['fixed']['count']:,}",
                    f"{customer_stats['discount']['count']:,}"
                ],
                'Share': [
                    f"{customer_stats['regular']['percentage']:.1f}%",
                    f"{customer_stats['fixed']['percentage']:.1f}%",
                    f"{customer_stats['discount']['percentage']:.1f}%"
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            st.caption("💡 Only **Regular Customers** participate in Purchase Now vs Bid decisions")
    
    # Extract REQUEST-LEVEL data from purchase_requests
    st.markdown("---")
    st.markdown("### 🎯 Purchase Now vs Bid Decisions")
    st.caption("📊 For request-level breakdown by customer type, see **Decision 7: Consumption Frequency**")
    
    if 'purchase_requests' in df.columns:
        # Collect all purchase decisions from all requests
        regular_requests = []
        
        for idx, row in df.iterrows():
            requests = row.get('purchase_requests', [])
            if isinstance(requests, list):
                for req in requests:
                    if isinstance(req, dict):
                        platform_price = req.get('platformPrice')
                        
                        # Count only PN and BID for regular customers
                        if platform_price in ['PN', 'BID']:
                            regular_requests.append(platform_price)
        
        # Count regular customer choices
        regular_counts = Counter(regular_requests)
        total_regular_requests = len(regular_requests)
        
        if total_regular_requests > 0:
            pn_count = regular_counts.get('PN', 0)
            bid_count = regular_counts.get('BID', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Regular Requests", f"{total_regular_requests:,}", 
                         help="Purchase requests from regular customers only")
            with col2:
                st.metric("Purchase Now (PN)", f"{pn_count:,} ({pn_count/total_regular_requests*100:.1f}%)")
            with col3:
                st.metric("Bid (BID)", f"{bid_count:,} ({bid_count/total_regular_requests*100:.1f}%)")
            with col4:
                st.metric("Purchase Now Rate", f"{pn_count/total_regular_requests*100:.1f}%")
            
            # Purchase Now vs Bid visualization - donut chart
            col_plot, col_stats = st.columns([2, 1])
            
            with col_plot:
                st.markdown(f"**Purchase Decisions Distribution ({total_regular_requests:,} requests)**")
                fig = px.pie(
                    values=[pn_count, bid_count],
                    names=['Purchase Now (PN)', 'Bid (BID)'],
                    hole=0.4,  # Donut chart
                    color_discrete_map={
                        'Purchase Now (PN)': '#4CAF50',  # Green
                        'Bid (BID)': '#FF9800'  # Orange
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_stats:
                st.markdown("**🛒 Request-Level Choices**")
                st.caption("(Regular customers only)")
                breakdown_df = pd.DataFrame({
                    'Choice': ['Purchase Now (PN)', 'Bid (BID)'],
                    'Requests': [pn_count, bid_count],
                    'Percentage': [
                        f"{pn_count/total_regular_requests*100:.1f}%",
                        f"{bid_count/total_regular_requests*100:.1f}%"
                    ]
                })
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        else:
            st.info("No regular customer purchase requests found")
    else:
        st.warning("No purchase_requests data available")


def render_rejected_transaction_defaults(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_transaction_defaults - prioritized options per agent"""
    
    # Define the 5 options
    options = [
        ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
        ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
        ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
        ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
        ("forgo_transaction", "Option 5: Forgo the purchase request")
    ]
    
    option_names = dict(options)
    option_numbers = {
        "higher_price_category": "Option 1",
        "lower_pn_vendor": "Option 2",
        "current_vendor_pn": "Option 3",
        "place_bid": "Option 4",
        "forgo_transaction": "Option 5"
    }
    
    # Check the actual simulation execution mode from session state
    simulation_mode = "unknown"
    if hasattr(st.session_state, 'sim_params') and hasattr(st.session_state.sim_params, 'simulation_execution_mode'):
        simulation_mode = st.session_state.sim_params.simulation_execution_mode
    
    # Analyze the prioritized lists from agent data
    st.info("ℹ️ **Note**: Each agent has a prioritized list of default options (1-5 options). The list shows their order of preference when transactions are rejected.")
    
    # Top section: Current results display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        st.metric("Simulation Mode", simulation_mode.title())
    
    with col3:
        # Count how many agents have lists vs single values
        list_count = decision_data.apply(lambda x: isinstance(x, list)).sum()
        st.metric("Agents with Priority Lists", f"{list_count:,}")
    
    # Show configured priority template
    st.markdown("---")
    st.markdown("**📊 Configured Priority Options**")
    
    # Get configured priority template for display
    priority_key = f"{decision_name}_priority_template"
    configured_template = st.session_state.get(priority_key, ["forgo_transaction"])
    
    col_template, col_chart = st.columns([1, 2])
    
    with col_template:
        st.markdown("**Priority Template:**")
        
        # Display configured priority list
        if isinstance(configured_template, list):
            for i, opt in enumerate(configured_template, 1):
                option_label = option_numbers.get(opt, opt)
                option_desc = option_names.get(opt, opt)
                st.markdown(f"**{i}.** {option_label}")
                st.caption(f"   {option_desc}")
        else:
            st.caption("No priority template configured")
        
        st.caption(f"💡 {len(configured_template)} option(s) configured")
        st.caption("All agents use this priority list")
    
    with col_chart:
        # Count which options appear in agents' priority lists
        from collections import Counter
        all_options_used = []
        
        for agent_list in decision_data:
            if isinstance(agent_list, list):
                all_options_used.extend(agent_list)
            else:
                all_options_used.append(agent_list)
        
        option_counts = Counter(all_options_used)
        
        # Create pie chart showing which options are in the priority lists
        if len(option_counts) > 0:
            # Sort by the order in configured_template if possible
            if isinstance(configured_template, list):
                sorted_options = [opt for opt in configured_template if opt in option_counts]
            else:
                sorted_options = list(option_counts.keys())
            
            labels = [option_numbers.get(opt, opt) for opt in sorted_options]
            values = [option_counts[opt] for opt in sorted_options]
            
            # Create pie chart
            fig = px.pie(
                values=values,
                names=labels,
                title="Options in Priority Lists",
                hole=0.4,  # Donut chart
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>%{value:,} times in lists<br>%{percent}<extra></extra>'
            )
            fig.update_layout(
                showlegend=True, 
                height=450,
                margin=dict(t=60, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{decision_name}_options_chart")
            
            # Show detailed breakdown
            with st.expander("📋 Detailed Breakdown"):
                for opt in sorted_options:
                    count = option_counts[opt]
                    percentage = (count / len(all_options_used)) * 100
                    st.caption(f"• {option_numbers.get(opt, opt)}: appears {count:,} times ({percentage:.1f}%)")
    
    # Summary statistics
    st.markdown("---")
    st.markdown("**📈 Summary Statistics**")
    
    col_summary1, col_summary2 = st.columns(2)
    
    with col_summary1:
        st.markdown("**Priority List Lengths:**")
        list_lengths = decision_data.apply(lambda x: len(x) if isinstance(x, list) else 1)
        length_counts = list_lengths.value_counts().sort_index()
        
        for length, count in length_counts.items():
            percentage = (count / len(decision_data)) * 100
            st.caption(f"• {int(length)} option(s): {count:,} agents ({percentage:.1f}%)")
    
    with col_summary2:
        st.markdown("**Most Common 1st Choice:**")
        first_choices = decision_data.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
        first_choice_counts = first_choices.value_counts()
        
        for i, (choice, count) in enumerate(first_choice_counts.head(3).items(), 1):
            percentage = (count / len(decision_data)) * 100
            st.caption(f"{i}. {option_numbers.get(choice, choice)}: {count:,} agents ({percentage:.1f}%)")
    
    # Download section
    st.markdown("---")
    st.markdown("**📥 Download Priority Lists**")
    
    # Prepare export data
    export_df = _prepare_priority_lists_export(df, decision_data)
    
    if export_df is not None and not export_df.empty:
        # Create Excel file
        from io import BytesIO
        from datetime import datetime
        
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Priority Lists')
        
        st.download_button(
            label="📊 Download Priority Lists Excel",
            data=buffer.getvalue(),
            file_name=f"rejected_transaction_priorities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download detailed priority lists with Agent ID, Allowance Level, Group, and Priority 1-5 columns"
        )
        
        # Show preview
        with st.expander("📋 Preview Export Data (first 50 rows)"):
            st.dataframe(export_df.head(50), use_container_width=True)
            st.caption(f"Total rows: {len(export_df):,}")
    else:
        st.warning("Unable to prepare export data")


def render_rejected_transaction_option(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_transaction_option with interactive radio buttons"""
    
    # Define the 5 options
    options = [
        ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
        ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
        ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
        ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
        ("forgo_transaction", "Option 5: Forgo the purchase request")
    ]
    
    option_names = dict(options)
    
    # Get current option from results or session state
    value_counts = decision_data.value_counts()
    current_option = value_counts.index[0] if len(value_counts) > 0 else "forgo_transaction"
    
    # Use _default_selection key (same as Page 2 Overview tab) for consistency
    radio_key = f"{decision_name}_default_selection"
    if radio_key not in st.session_state:
        # Initialize with what was actually used in the simulation
        st.session_state[radio_key] = current_option
    
    # Top section: Current results display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        # Show what's configured for NEXT simulation (from session state)
        configured_option = st.session_state.get(radio_key, current_option)
        display_name = option_names.get(configured_option, configured_option)
        st.metric("Configured Option", display_name.split(":")[0])  # Just "Option X"
    
    with col3:
        # Show most common from CURRENT results
        most_common_option = option_names.get(current_option, current_option)
        st.metric("Most Common Result", most_common_option.split(":")[0])
    
    with col4:
        if len(value_counts) > 0:
            percentage = (value_counts.iloc[0] / len(decision_data)) * 100
            st.metric("Result Frequency", f"{percentage:.1f}%")
    
    # Main configuration section - READ-ONLY DISPLAY
    st.markdown("---")
    st.markdown("**⚙️ Specific Option Configuration (Read-Only):**")
    
    col_radio, col_viz = st.columns([1, 1])
    
    with col_radio:
        # Get current selection from session state
        current_selection = st.session_state.get(radio_key, current_option)
        
        # Display selected option as read-only
        st.success(f"✅ **Selected Option:**\n\n{option_names.get(current_selection, current_selection)}")
        
        st.caption("💡 To modify this setting: Go to **Page 2 → Overview Tab**")
    
    with col_viz:
        # Show current results visualization
        if len(value_counts) > 0:
            # Create readable labels for the chart
            readable_labels = [option_names.get(opt, opt) for opt in value_counts.index]
            
            fig = px.pie(
                values=value_counts.values,
                names=readable_labels,
                title="Current Simulation Results",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True, key="rejected_transaction_option_chart")
        else:
            st.info("No simulation data available")


def _prepare_priority_lists_export(df: pd.DataFrame, decision_data) -> pd.DataFrame:
    """
    Prepare rejected transaction defaults priority lists for Excel export.
    
    Creates columns: Agent ID, Assigned Allowance Level, Group_experiment, 
    Priority 1, Priority 2, Priority 3, Priority 4, Priority 5
    
    Priority columns contain option numbers (1, 2, 3, 4, 5) or blank.
    
    Args:
        df: Full results DataFrame with agent data
        decision_data: Series containing priority lists
        
    Returns:
        DataFrame formatted for Excel export
    """
    # Map option codes to numbers
    option_to_number = {
        "higher_price_category": 1,
        "lower_pn_vendor": 2,
        "current_vendor_pn": 3,
        "place_bid": 4,
        "forgo_transaction": 5
    }
    
    # Create export dataframe
    export_df = pd.DataFrame()
    
    # Agent ID
    if 'agent_id' in df.columns:
        export_df['Agent ID'] = df['agent_id']
    elif 'index' in df.columns:
        export_df['Agent ID'] = df['index'] + 1  # Convert 0-based to 1-based
    else:
        export_df['Agent ID'] = range(1, len(df) + 1)
    
    # Assigned Allowance Level
    if 'Assigned Allowance Level' in df.columns:
        export_df['Assigned Allowance Level'] = df['Assigned Allowance Level']
    elif 'income_category' in df.columns:
        export_df['Assigned Allowance Level'] = df['income_category']
    else:
        export_df['Assigned Allowance Level'] = ''
    
    # Group_experiment
    if 'Group_experiment' in df.columns:
        export_df['Group_experiment'] = df['Group_experiment']
    elif 'group' in df.columns:
        export_df['Group_experiment'] = df['group']
    else:
        export_df['Group_experiment'] = ''
    
    # Priority columns (1-5)
    for priority_pos in range(1, 6):
        column_name = f'Priority {priority_pos}'
        priority_values = []
        
        for agent_list in decision_data:
            if isinstance(agent_list, list):
                # Check if agent has this priority position
                if len(agent_list) >= priority_pos:
                    option_code = agent_list[priority_pos - 1]
                    option_number = option_to_number.get(option_code, '')
                    priority_values.append(option_number)
                else:
                    # Agent doesn't have this many priorities
                    priority_values.append('')
            else:
                # Single value (legacy format) - only for priority 1
                if priority_pos == 1:
                    option_number = option_to_number.get(agent_list, '')
                    priority_values.append(option_number)
                else:
                    priority_values.append('')
        
        export_df[column_name] = priority_values
    
    return export_df


def render_rejected_bid_value(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_bid_value"""
    st.markdown("**Not relevant given choice of Option 5**")

