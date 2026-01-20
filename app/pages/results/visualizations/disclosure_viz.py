# app/pages/results/visualizations/disclosure_viz.py
"""
Disclosure-related visualization functions.
Handles disclose_income and disclose_documents decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px


def _apply_price_formatting_disclosure(writer, sheet_name: str, df: pd.DataFrame):
    """
    Apply Excel number formatting to price-related columns to display 2 decimal places.
    """
    price_columns = [
        'income', 'Income', 'Honesty_Humility',
        'TWT+Sospeso [=AW2+AX2]{Periods 1+2}',
        'Assigned income from the distribution',
    ]
    
    workbook = writer.book
    worksheet = workbook[sheet_name]
    
    for col_idx, col_name in enumerate(df.columns, start=1):
        if col_name in price_columns:
            for row_idx in range(2, len(df) + 2):
                cell = worksheet.cell(row=row_idx, column=col_idx)
                if isinstance(cell.value, (int, float)) and cell.value is not None:
                    cell.number_format = '0.00'


def render_disclose_income(df, decision_name, decision_title, decision_data):
    """Visualization for disclose_income - binary Y/N choice with optional raw value histogram"""
    import plotly.graph_objects as go
    
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
    
    # Check if raw DI values are available for detailed analysis
    has_raw_values = 'disclose_income_raw' in df.columns
    
    if has_raw_values:
        # Show raw DI value histogram (distribution before Y/N classification)
        st.markdown("### 📊 Raw Disclosure Intention Distribution")
        st.caption("Distribution of raw DI values before classification (>0 → Y, ≤0 → N)")
        
        raw_values = df['disclose_income_raw'].dropna()
        mean_val = raw_values.mean()
        
        # Create histogram with vertical line at 0
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=raw_values,
            nbinsx=40,
            name='DI Raw Values',
            marker_color='steelblue',
            opacity=0.7
        ))
        
        # Add vertical line at 0 (decision boundary)
        fig.add_vline(
            x=0,
            line_dash="solid",
            line_color="red",
            line_width=3,
            annotation_text="Boundary (0)",
            annotation_position="top",
            annotation_font_color="red"
        )
        
        # Add vertical line at mean
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Mean: {mean_val:.3f}",
            annotation_position="bottom",
            annotation_font_color="green"
        )
        
        fig.update_layout(
            title="Raw DI Value Distribution",
            xaxis_title="Raw DI Value (>0 → Y, ≤0 → N)",
            yaxis_title="Number of Agents",
            showlegend=False,
            height=350,
            xaxis=dict(zeroline=True, zerolinecolor='red', zerolinewidth=2)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics and pie chart side by side
        col_stats, col_pie = st.columns(2)
        
        with col_stats:
            st.markdown("**📈 Raw Value Statistics**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'Median', 'Min', 'Max'],
                'Value': [
                    f"{mean_val:.4f}",
                    f"{raw_values.std():.4f}",
                    f"{raw_values.median():.4f}",
                    f"{raw_values.min():.4f}",
                    f"{raw_values.max():.4f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            # Show insight
            if mean_val > 0:
                st.success(f"✅ Mean ({mean_val:.4f}) > 0: Distribution favors disclosure")
            elif mean_val < 0:
                st.warning(f"⚠️ Mean ({mean_val:.4f}) < 0: Distribution favors non-disclosure")
            else:
                st.info("ℹ️ Mean ≈ 0: Distribution is balanced")
        
        with col_pie:
            st.markdown("**📊 Final Classification (Pie Chart)**")
            if len(value_counts) > 0:
                fig_pie = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}
                )
                fig_pie.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        # Basic visualization - pie chart only (no raw values available)
        col_plot, col_stats = st.columns([2, 1])
        
        with col_plot:
            if len(value_counts) > 0:
                st.markdown(f"**{decision_title} Distribution**")
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col_stats:
            st.markdown("**📊 Choice Breakdown**")
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
    
    # Excel download section
    st.markdown("---")
    st.markdown("### 📥 Download Agent Disclose Income Data")
    
    # Prepare Excel data
    excel_data = _prepare_disclose_income_excel_data(df)
    
    if excel_data is not None:
        # Convert to Excel bytes
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            excel_data.to_excel(writer, index=False, sheet_name='Agent Disclose Income Data')
            # Apply 2-decimal formatting
            _apply_price_formatting_disclosure(writer, 'Agent Disclose Income Data', excel_data)
        excel_bytes = output.getvalue()
        
        # Download button
        st.download_button(
            label="📥 Download Agent Disclose Income Data (Excel)",
            data=excel_bytes,
            file_name="agent_disclose_income_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download detailed agent data including traits and disclose income decision"
        )
        
        # Show preview of the Excel data
        with st.expander("📋 Preview Excel Data (first 10 rows)"):
            st.dataframe(excel_data.head(10), use_container_width=True)
            st.caption(f"**Columns**: {', '.join(excel_data.columns)}")
    else:
        st.warning("⚠️ Unable to prepare Excel data. Some required columns may be missing.")


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
        st.metric("Eligible to Disclose Documents", f"{qualified_agents:,}", 
                  help="Agents with income < threshold who disclosed income. These agents are asked if they want to disclose documents for discount eligibility.")
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
                st.markdown(f"### {decision_title} - Qualified Agents Only")
                fig = px.pie(
                    values=list(qualified_counts.values()),
                    names=list(qualified_counts.keys()),
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
    
    # CUSTOMER TYPE DISTRIBUTION - Comprehensive visualization
    st.markdown("---")
    st.markdown("### 👥 Customer Type Distribution")
    st.info("💡 **Customer types** are determined by disclosure decisions and affect pricing and purchasing behavior throughout the simulation.")
    
    # Check if customer_type column exists in the dataframe
    if 'customer_type' in df.columns:
        from src.decisions.income_utils import analyze_customer_types
        customer_stats = analyze_customer_types(df)
        
        # Show customer type breakdown with detailed metrics
        type_col1, type_col2, type_col3, type_col4 = st.columns(4)
        
        with type_col1:
            st.metric("Total Agents", f"{customer_stats['total']:,}")
        with type_col2:
            st.metric("Regular Customers", 
                     f"{customer_stats['regular']['count']:,} ({customer_stats['regular']['percentage']:.1f}%)",
                     help="Did not disclose income → Pay regular Purchase Now (PN) prices or place bids (BID)")
        with type_col3:
            st.metric("Fixed Customers", 
                     f"{customer_stats['fixed']['count']:,} ({customer_stats['fixed']['percentage']:.1f}%)",
                     help="Disclosed income but not documents → Use fixed pricing only (FIXED)")
        with type_col4:
            st.metric("Discount Customers", 
                     f"{customer_stats['discount']['count']:,} ({customer_stats['discount']['percentage']:.1f}%)",
                     help="Income < threshold, disclosed both → Get discount pricing (DISCOUNT)")
        
        # Detailed explanation expander
        with st.expander("📖 Customer Type Definitions & Impact"):
            st.markdown("""
            **Customer types are determined by agents' disclosure decisions and income level:**
            
            **🔵 Regular Customers**
            - **How assigned**: Did not disclose income (Decision 1: disclose_income = "N")
            - **Pricing**: Pay regular Purchase Now (PN) prices or can place bids (BID)
            - **Purchase decisions**: Choose between Purchase Now and Bid (Decision 9)
            - **Platform price label**: PN or BID
            
            **🟣 Fixed Customers**
            - **How assigned**: Disclosed income (Decision 1: disclose_income = "Y") AND (income above threshold OR (income below threshold but did NOT disclose documents (Decision 2: disclose_documents = "N" or "NA")))
            - **Pricing**: Use fixed pricing only (FIXED)
            - **Purchase decisions**: Do not participate in Purchase Now vs Bid decisions (Decision 9 = "NA_fixed")
            - **Platform price label**: FIXED
            
            **🔴 Discount Customers**
            - **How assigned**: Income below threshold AND disclosed income (Decision 1: "Y") AND disclosed documents (Decision 2: "Y")
            - **Pricing**: Get discounted prices (DISCOUNT)
            - **Purchase decisions**: Do not participate in Purchase Now vs Bid decisions (Decision 9 = "NA_discount")
            - **Platform price label**: DISCOUNT
            
            💡 **Note**: Customer types are used throughout the simulation to determine pricing, purchase options, and vendor selection behavior.
            """)
        
        # Visualization: Donut chart and breakdown table
        col_pie, col_table = st.columns([2, 1])
        
        with col_pie:
            # Create donut chart for customer types
            customer_types_data = {
                'Customer Type': ['Regular Customers', 'Fixed Customers', 'Discount Customers'],
                'Count': [
                    customer_stats['regular']['count'],
                    customer_stats['fixed']['count'],
                    customer_stats['discount']['count']
                ]
            }
            
            st.markdown(f"### Customer Type Breakdown ({customer_stats['total']:,} total agents)")
            fig = px.pie(
                values=customer_types_data['Count'],
                names=customer_types_data['Customer Type'],
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
            
            st.caption("💡 Only **Regular Customers** participate in Purchase Now vs Bid decisions (Decision 9)")
        
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
                # Apply 2-decimal formatting
                _apply_price_formatting_disclosure(writer, 'Agent Disclosure Data', excel_data)
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


def _prepare_disclose_income_excel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare disclose income data for Excel export.
    
    Includes all agent traits and the disclose income decision indicator.
    
    Args:
        df: Results dataframe with agent data
        
    Returns:
        DataFrame formatted for Excel export, or None if required columns missing
    """
    # Check required column
    if 'disclose_income' not in df.columns:
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
    
    # Agent Traits
    # Honesty_Humility
    if 'Honesty_Humility' in df.columns:
        export_df['Honesty_Humility'] = df['Honesty_Humility'].round(2)
    else:
        export_df['Honesty_Humility'] = ''
    
    # Assigned Allowance Level - use the actual allowance column which exists for ALL agents
    # Priority: 'Assigned Allowance Level' > 'actual_allowance' > 'income' (not income_category which is only for Discount/Fixed)
    if 'Assigned Allowance Level' in df.columns:
        export_df['Assigned Allowance Level'] = df['Assigned Allowance Level']
    elif 'actual_allowance' in df.columns:
        export_df['Assigned Allowance Level'] = df['actual_allowance']
    elif 'income' in df.columns:
        export_df['Assigned Allowance Level'] = df['income']
    else:
        export_df['Assigned Allowance Level'] = ''
    
    # Study Program
    if 'Study Program' in df.columns:
        export_df['Study Program'] = df['Study Program']
    else:
        export_df['Study Program'] = ''
    
    # Group_experiment
    if 'Group_experiment' in df.columns:
        export_df['Group_experiment'] = df['Group_experiment']
    elif 'group' in df.columns:
        export_df['Group_experiment'] = df['group']
    elif 'group_experiment' in df.columns:
        export_df['Group_experiment'] = df['group_experiment']
    else:
        export_df['Group_experiment'] = ''
    
    # TWT+Sospeso
    if 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}' in df.columns:
        export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'].round(2)
    else:
        export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = ''
    
    # Income
    if 'income' in df.columns:
        export_df['income'] = df['income'].round(2)
    elif 'actual_allowance' in df.columns:
        export_df['income'] = df['actual_allowance'].round(2)
    else:
        export_df['income'] = ''
    
    # disclose_income (Y/N to 1/0)
    export_df['disclose_income'] = df['disclose_income'].apply(
        lambda x: 1 if x == 'Y' else (0 if x == 'N' else '')
    )
    
    return export_df


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
    
    # ====================================================================
    # AGENT TRAITS: Full agent trait columns (consistent with Disclose Income)
    # ====================================================================
    
    # Honesty_Humility
    if 'Honesty_Humility' in df.columns:
        export_df['Honesty_Humility'] = df['Honesty_Humility'].round(2)
    else:
        export_df['Honesty_Humility'] = ''
    
    # Assigned Allowance Level - use the actual allowance column which exists for ALL agents
    # Priority: 'Assigned Allowance Level' > 'actual_allowance' > 'income' (not income_category which is only for Discount/Fixed)
    if 'Assigned Allowance Level' in df.columns:
        export_df['Assigned Allowance Level'] = df['Assigned Allowance Level']
    elif 'actual_allowance' in df.columns:
        export_df['Assigned Allowance Level'] = df['actual_allowance']
    elif 'income' in df.columns:
        export_df['Assigned Allowance Level'] = df['income']
    else:
        export_df['Assigned Allowance Level'] = ''
    
    # Study Program
    if 'Study Program' in df.columns:
        export_df['Study Program'] = df['Study Program']
    else:
        export_df['Study Program'] = ''
    
    # Group_experiment (check for various possible column names, case-insensitive)
    if 'Group_experiment' in df.columns:
        export_df['Group_experiment'] = df['Group_experiment']
    elif 'group' in df.columns:
        export_df['Group_experiment'] = df['group']
    elif 'group_experiment' in df.columns:
        export_df['Group_experiment'] = df['group_experiment']
    else:
        export_df['Group_experiment'] = ''
    
    # TWT+Sospeso
    if 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}' in df.columns:
        export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'].round(2)
    else:
        export_df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}'] = ''
    
    # Income
    if 'income' in df.columns:
        export_df['income'] = df['income'].round(2)
    elif 'actual_allowance' in df.columns:
        export_df['income'] = df['actual_allowance'].round(2)
    else:
        export_df['income'] = ''
    
    # ====================================================================
    
    # disclose_income (Y/N to 1/0)
    if 'disclose_income' in df.columns:
        export_df['disclose_income'] = df['disclose_income'].apply(
            lambda x: 1 if x == 'Y' else (0 if x == 'N' else '')
        )
    else:
        export_df['disclose_income'] = ''
    
    # disclose_documents (Y/N/NA to 1/0/N/A)
    if 'disclose_documents' in df.columns:
        export_df['disclose_documents'] = df['disclose_documents'].apply(
            lambda x: 1 if x == 'Y' else (0 if x == 'N' else 'N/A')
        )
    else:
        export_df['disclose_documents'] = ''
    
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

