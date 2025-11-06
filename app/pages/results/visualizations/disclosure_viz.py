# app/pages/results/visualizations/disclosure_viz.py
"""
Disclosure-related visualization functions.
Handles disclose_income and disclose_documents decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px


def render_disclose_income(df, decision_name, decision_title, decision_data):
    """Visualization for disclose_income - binary Y/N choice"""
    
    # Binary choice metrics
    col1, col2, col3, col4 = st.columns(4)
    
    value_counts = decision_data.value_counts()
    total = len(decision_data)
    
    with col1:
        st.metric("Total Agents", f"{total:,}")
    with col2:
        yes_count = value_counts.get('Y', 0)
        pct_yes = (yes_count/total)*100
        st.metric("Disclosed (Y)", f"{yes_count:,} ({pct_yes:.1f}%)")
    with col3:
        no_count = value_counts.get('N', 0)
        pct_no = (no_count/total)*100
        st.metric("Not Disclosed (N)", f"{no_count:,} ({pct_no:.1f}%)")
    with col4:
        disclosure_rate = (yes_count/total)*100
        st.metric("Disclosure Rate", f"{disclosure_rate:.1f}%")
    
    # Binary choice visualization - pie chart
    col_plot, col_stats = st.columns([2, 1])
    
    with col_plot:
        if len(value_counts) > 0:
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"{decision_title} Distribution",
                color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}  # Green for Yes, Red for No
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
        st.markdown("**📊 Choice Breakdown**")
        # Ensure Y appears before N in the breakdown
        ordered_choices = ['Y', 'N']
        ordered_data = []
        for choice in ordered_choices:
            if choice in value_counts.index:
                count = value_counts[choice]
                ordered_data.append({
                    'Choice': choice,
                    'Count': count,
                    'Percentage': f"{(count/total)*100:.1f}%"
                })
        
        breakdown_df = pd.DataFrame(ordered_data)
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)


def render_disclose_documents(df, decision_name, decision_title, decision_data):
    """Visualization for disclose_documents - binary Y/N choice with NA handling
    
    This decision only applies to agents qualified for discount (income < threshold).
    Agents not qualified will have "NA" value.
    """
    
    # Separate NA (not applicable) from Y/N choices
    value_counts = decision_data.value_counts()
    total_agents = len(decision_data)
    
    na_count = value_counts.get('NA', 0)
    qualified_agents = total_agents - na_count
    
    # Show overall metrics
    st.markdown("### Eligibility & Application")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Agents", f"{total_agents:,}")
    with col2:
        st.metric("Qualified for Discount", f"{qualified_agents:,}", 
                  help="Agents with income < discount threshold")
    with col3:
        st.metric("Not Qualified (NA)", f"{na_count:,}", 
                  help="Agents with income ≥ discount threshold (decision does not apply)")
    
    # If there are qualified agents, show their Y/N choices
    if qualified_agents > 0:
        st.markdown("### Qualified Agents' Choices")
        st.caption(f"📊 Among the {qualified_agents:,} agents qualified for discount (income < threshold)")
        
        # Binary choice metrics for qualified agents only
        col1, col2, col3, col4 = st.columns(4)
        
        yes_count = value_counts.get('Y', 0)
        no_count = value_counts.get('N', 0)
        
        with col1:
            st.metric("Qualified Agents", f"{qualified_agents:,}")
        with col2:
            pct_yes = (yes_count/qualified_agents)*100
            st.metric("Disclosed (Y)", f"{yes_count:,} ({pct_yes:.1f}%)")
        with col3:
            pct_no = (no_count/qualified_agents)*100
            st.metric("Not Disclosed (N)", f"{no_count:,} ({pct_no:.1f}%)")
        with col4:
            disclosure_rate = (yes_count/qualified_agents)*100
            st.metric("Disclosure Rate", f"{disclosure_rate:.1f}%",
                      help="Percentage of qualified agents who disclosed documents")
        
        # Binary choice visualization - pie chart (only Y/N, excluding NA)
        col_plot, col_stats = st.columns([2, 1])
        
        with col_plot:
            # Filter out NA for the pie chart
            qualified_counts = {k: v for k, v in value_counts.items() if k != 'NA'}
            if len(qualified_counts) > 0:
                fig = px.pie(
                    values=list(qualified_counts.values()),
                    names=list(qualified_counts.keys()),
                    title=f"{decision_title} - Qualified Agents Only",
                    color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}  # Green for Yes, Red for No
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col_stats:
            st.markdown("**📊 Choice Breakdown (Qualified)**")
            # Ensure Y appears before N in the breakdown
            ordered_choices = ['Y', 'N']
            ordered_data = []
            for choice in ordered_choices:
                if choice in qualified_counts:
                    count = qualified_counts[choice]
                    ordered_data.append({
                        'Choice': choice,
                        'Count': count,
                        'Percentage': f"{(count/qualified_agents)*100:.1f}%"
                    })
            
            qualified_breakdown = pd.DataFrame(ordered_data)
            st.dataframe(qualified_breakdown, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No agents qualified for discount (all agents have income ≥ threshold)")
    
    # Show full breakdown including NA
    st.markdown("### Complete Breakdown (All Agents)")
    full_breakdown = pd.DataFrame({
        'Status': value_counts.index,
        'Count': value_counts.values,
        'Percentage': [f"{(count/total_agents)*100:.1f}%" for count in value_counts.values]
    })
    st.dataframe(full_breakdown, use_container_width=True, hide_index=True)
    
    # NEW: Customer Type Breakdown
    st.markdown("### Customer Type Distribution")
    st.caption("📊 Based on disclosure decisions and income threshold")
    
    # Check if customer_type column exists in the dataframe
    if 'customer_type' in df.columns:
        customer_type_counts = df['customer_type'].value_counts()
        
        # Define customer type mapping with descriptions
        customer_type_labels = {
            'regular': 'Regular customers (did not disclose income)',
            'fixed': 'Fixed price customers (disclosed income but not documents)',
            'discount': 'Discount customers (income below threshold, disclosed both income and documents)'
        }
        
        # Calculate metrics
        total_agents_ct = len(df)
        regular_count = customer_type_counts.get('regular', 0)
        fixed_count = customer_type_counts.get('fixed', 0)
        discount_count = customer_type_counts.get('discount', 0)
        
        # Show metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Agents", f"{total_agents_ct:,}")
        with col2:
            pct_regular = (regular_count/total_agents_ct)*100
            st.metric("Regular", f"{regular_count:,} ({pct_regular:.1f}%)",
                      help="Agents who did not disclose income")
        with col3:
            pct_fixed = (fixed_count/total_agents_ct)*100
            st.metric("Fixed Price", f"{fixed_count:,} ({pct_fixed:.1f}%)",
                      help="Agents who disclosed income but not documents")
        with col4:
            pct_discount = (discount_count/total_agents_ct)*100
            st.metric("Discount", f"{discount_count:,} ({pct_discount:.1f}%)",
                      help="Agents below threshold who disclosed both income and documents")
        
        # Visualization: Pie chart and breakdown table
        col_plot, col_stats = st.columns([2, 1])
        
        with col_plot:
            if len(customer_type_counts) > 0:
                # Create labeled data for pie chart
                pie_labels = []
                pie_values = []
                
                # Ensure consistent order: Regular, Fixed, Discount
                for ct_key in ['regular', 'fixed', 'discount']:
                    if ct_key in customer_type_counts.index:
                        count = customer_type_counts[ct_key]
                        if ct_key == 'regular':
                            pie_labels.append('Regular')
                        elif ct_key == 'fixed':
                            pie_labels.append('Fixed Price')
                        else:  # discount
                            pie_labels.append('Discount')
                        pie_values.append(count)
                
                fig = px.pie(
                    values=pie_values,
                    names=pie_labels,
                    title="Customer Type Distribution",
                    color_discrete_map={
                        'Regular': '#4169E1',      # Royal Blue
                        'Fixed Price': '#FF8C00',  # Dark Orange
                        'Discount': '#32CD32'      # Lime Green
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col_stats:
            st.markdown("**📊 Customer Type Breakdown**")
            
            # Create breakdown table with consistent order
            breakdown_data = []
            for ct_key in ['regular', 'fixed', 'discount']:
                if ct_key in customer_type_counts.index:
                    count = customer_type_counts[ct_key]
                    if ct_key == 'regular':
                        label = 'Regular'
                    elif ct_key == 'fixed':
                        label = 'Fixed Price'
                    else:  # discount
                        label = 'Discount'
                    
                    breakdown_data.append({
                        'Customer Type': label,
                        'Count': count,
                        'Percentage': f"{(count/total_agents_ct)*100:.1f}%"
                    })
            
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        
        # Excel download section
        st.markdown("---")
        
        # Prepare Excel data
        excel_data = _prepare_disclosure_excel_data(df)
        
        if excel_data is not None:
            # Convert to Excel bytes
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                excel_data.to_excel(writer, index=False, sheet_name='Agent Disclosure Data')
            excel_bytes = output.getvalue()
            
            # Download button
            st.download_button(
                label="📥 Download Agent Disclosure Data (Excel)",
                data=excel_bytes,
                file_name="agent_disclosure_customer_types.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download detailed agent data including disclosure decisions and customer types"
            )
        else:
            st.warning("⚠️ Unable to prepare Excel data. Some required columns may be missing.")
    else:
        st.warning("⚠️ Customer type information not available in results data")


def _prepare_disclosure_excel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare disclosure and customer type data for Excel export.
    
    Converts Y/N/NA values to 1/0 format and creates customer type indicator columns.
    
    Args:
        df: Results dataframe with agent data
        
    Returns:
        DataFrame formatted for Excel export, or None if required columns missing
    """
    # Check required columns
    required_cols = ['disclose_income', 'disclose_documents', 'customer_type']
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Create export dataframe
    export_df = pd.DataFrame()
    
    # Agent ID - try multiple possible column names
    if 'agent_id' in df.columns:
        export_df['Agent ID'] = df['agent_id']
    elif 'index' in df.columns:
        export_df['Agent ID'] = df['index'] + 1  # Convert 0-based to 1-based
    else:
        export_df['Agent ID'] = range(1, len(df) + 1)
    
    # Assigned Allowance Level (income category)
    if 'income_category' in df.columns:
        export_df['Assigned Allowance Level'] = df['income_category']
    else:
        export_df['Assigned Allowance Level'] = ''
    
    # Group_experiment
    if 'group' in df.columns:
        export_df['Group_experiment'] = df['group']
    elif 'group_experiment' in df.columns:
        export_df['Group_experiment'] = df['group_experiment']
    else:
        export_df['Group_experiment'] = ''
    
    # Assigned income from the distribution
    if 'income' in df.columns:
        export_df['Assigned income from the distribution'] = df['income'].round(2)
    else:
        export_df['Assigned income from the distribution'] = ''
    
    # Disclosed income (Y/N to 1/0)
    if 'disclose_income' in df.columns:
        export_df['Disclosed income'] = df['disclose_income'].apply(
            lambda x: 1 if x == 'Y' else (0 if x == 'N' else '')
        )
    else:
        export_df['Disclosed income'] = ''
    
    # Disclosed documents (Y/N/NA to 1/0/blank)
    if 'disclose_documents' in df.columns:
        export_df['Disclosed documents'] = df['disclose_documents'].apply(
            lambda x: 1 if x == 'Y' else (0 if x == 'N' else '')
        )
    else:
        export_df['Disclosed documents'] = ''
    
    # Customer type indicator columns
    if 'customer_type' in df.columns:
        export_df['Regular'] = (df['customer_type'] == 'regular').astype(int)
        export_df['Fixed'] = (df['customer_type'] == 'fixed').astype(int)
        export_df['Discount'] = (df['customer_type'] == 'discount').astype(int)
    else:
        export_df['Regular'] = ''
        export_df['Fixed'] = ''
        export_df['Discount'] = ''
    
    return export_df

