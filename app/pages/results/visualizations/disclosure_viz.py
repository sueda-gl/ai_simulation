# app/pages/results/visualizations/disclosure_viz.py
"""
Disclosure-related visualization functions.
Handles disclose_income and disclose_documents decisions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _apply_price_formatting_disclosure(writer, sheet_name: str, df: pd.DataFrame):
    """
    Apply Excel number formatting to numeric columns.
    
    Uses 'General' format to preserve original decimal precision from the data.
    No rounding or truncation is applied - values display exactly as stored.
    """
    # All numeric columns use General format to preserve original precision
    numeric_columns = [
        'Agreeable', 'Openness', 'Honesty_Humility', 'Extraversion', 'Neuroticism',
        'Religious', 'TWT+Sospeso', 'calc_PB', 'WOPB', 'WPB', 'Intercept', 'PB_i', 'Disclosure Income',
        'income', 'Income', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}',
        'Assigned income from the distribution',
    ]
    
    workbook = writer.book
    worksheet = workbook[sheet_name]
    
    for col_idx, col_name in enumerate(df.columns, start=1):
        if col_name not in numeric_columns:
            continue
        
        # Apply General format to preserve original precision
        for row_idx in range(2, len(df) + 2):
            cell = worksheet.cell(row=row_idx, column=col_idx)
            if isinstance(cell.value, (int, float)) and cell.value is not None:
                cell.number_format = 'General'


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
        st.metric("Disclosed (Y)", f"{yes_count:,} ({pct_yes:.2f}%)")
    with col3:
        no_count = value_counts.get('N', 0)
        pct_no = (no_count/total)*100
        st.metric("Not Disclosed (N)", f"{no_count:,} ({pct_no:.2f}%)")
    with col4:
        disclosure_rate = (yes_count/total)*100
        st.metric("Disclosure Rate", f"{disclosure_rate:.2f}%")
    
    # Check if raw DI values are available for histogram
    has_raw_values = 'disclose_income_raw' in df.columns
    
    if has_raw_values:
        # TWO-COLUMN LAYOUT: Pie chart on left, Histogram on right
        col_pie, col_hist = st.columns(2)
        
        with col_pie:
            st.markdown(f"**1. {decision_title} Distribution**")
            if len(value_counts) > 0:
                fig_pie = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}  # Green for Yes, Red for No
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Choice breakdown table below pie chart
            st.markdown("**📊 Choice Breakdown**")
            ordered_choices = ['Y', 'N']
            ordered_data = []
            for choice in ordered_choices:
                if choice in value_counts.index:
                    count = value_counts[choice]
                    ordered_data.append({
                        'Choice': choice,
                        'Count': count,
                        'Percentage': f"{(count/total)*100:.2f}%"
                    })
            breakdown_df = pd.DataFrame(ordered_data)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        
        with col_hist:
            # Get raw values for histogram
            raw_values = df['disclose_income_raw'].dropna()
            
            if len(raw_values) > 0:
                # Calculate statistics
                mean_val = raw_values.mean()
                std_val = raw_values.std()
                median_val = raw_values.median()
                min_val = raw_values.min()
                max_val = raw_values.max()
                
                # Calculate Y/N split based on threshold
                y_count_raw = (raw_values > 0).sum()
                n_count_raw = (raw_values <= 0).sum()
                total_raw = len(raw_values)
                y_pct_raw = (y_count_raw / total_raw) * 100 if total_raw > 0 else 0
                n_pct_raw = (n_count_raw / total_raw) * 100 if total_raw > 0 else 0
                
                # Get income mode for title
                income_mode = st.session_state.get('di_income_mode', 'Categorical')
                if 'categorical' in str(income_mode).lower():
                    mode_suffix = " (Categorical)"
                elif 'continuous' in str(income_mode).lower():
                    mode_suffix = " (Continuous)"
                else:
                    mode_suffix = ""
                
                st.markdown(f"**📈 Raw Disclose Income Distribution{mode_suffix}**")
                
                # Create histogram with vertical line at 0
                fig_hist = go.Figure()
                
                # Add histogram
                fig_hist.add_trace(go.Histogram(
                    x=raw_values,
                    nbinsx=40,
                    name='DI Raw Values',
                    marker_color='steelblue',
                    opacity=0.7
                ))
                
                # Add vertical line at 0 (decision boundary)
                fig_hist.add_vline(
                    x=0,
                    line_dash="solid",
                    line_color="red",
                    line_width=3,
                    annotation_text="Threshold (0)",
                    annotation_position="top",
                    annotation_font_color="red"
                )
                
                # Add vertical line at mean
                fig_hist.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color="green",
                    line_width=2,
                    annotation_text=f"Mean: {mean_val:.3f}",
                    annotation_position="bottom",
                    annotation_font_color="green"
                )
                
                # Update layout
                fig_hist.update_layout(
                    xaxis_title="Raw DI Value (>0 → Y, ≤0 → N)",
                    yaxis_title="Agents",
                    showlegend=False,
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis=dict(
                        zeroline=True,
                        zerolinecolor='red',
                        zerolinewidth=2
                    )
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Statistics and Classification side by side below histogram
                col_stats, col_class = st.columns(2)
                
                with col_stats:
                    st.markdown("**📈 Statistics**")
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Std Dev', 'Median', 'Min', 'Max'],
                        'Value': [
                            f"{mean_val:.4f}",
                            f"{std_val:.4f}",
                            f"{median_val:.4f}",
                            f"{min_val:.4f}",
                            f"{max_val:.4f}"
                        ]
                    })
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                with col_class:
                    st.markdown("**📊 Classification**")
                    classification_df = pd.DataFrame({
                        'Choice': ['Y (disclose)', 'N (not disclose)'],
                        'Count': [y_count_raw, n_count_raw],
                        '%': [f"{y_pct_raw:.2f}%", f"{n_pct_raw:.2f}%"]
                    })
                    st.dataframe(classification_df, hide_index=True, use_container_width=True)
            else:
                st.warning("No raw DI values available for histogram")
    
    else:
        # ORIGINAL LAYOUT: Pie chart + Choice breakdown (no raw values available)
        col_plot, col_stats = st.columns([2, 1])
        
        with col_plot:
            if len(value_counts) > 0:
                st.markdown(f"**{decision_title} Distribution**")
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
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
                        'Percentage': f"{(count/total)*100:.2f}%"
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
            st.metric("Disclosed (Y)", f"{yes_count:,} ({pct_yes:.2f}%)")
        with col3:
            pct_no = (no_count/qualified_agents)*100
            st.metric("Not Disclosed (N)", f"{no_count:,} ({pct_no:.2f}%)")
        with col4:
            disclosure_rate = (yes_count/qualified_agents)*100
            st.metric("Disclosure Rate", f"{disclosure_rate:.2f}%",
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
                        'Percentage': f"{(count/qualified_agents)*100:.2f}%"
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
                     f"{customer_stats['regular']['count']:,} ({customer_stats['regular']['percentage']:.2f}%)",
                     help="Did not disclose income → Pay regular Purchase Now (PN) prices or place bids (BID)")
        with type_col3:
            st.metric("Fixed Customers", 
                     f"{customer_stats['fixed']['count']:,} ({customer_stats['fixed']['percentage']:.2f}%)",
                     help="Disclosed income but not documents → Use fixed pricing only (FIXED)")
        with type_col4:
            st.metric("Discount Customers", 
                     f"{customer_stats['discount']['count']:,} ({customer_stats['discount']['percentage']:.2f}%)",
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
                    f"{customer_stats['regular']['percentage']:.2f}%",
                    f"{customer_stats['fixed']['percentage']:.2f}%",
                    f"{customer_stats['discount']['percentage']:.2f}%"
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
    
    Includes all variables used in the disclose income calculation:
    - Raw trait values (non-standardized): Agreeable, Openness, Honesty_Humility, 
      Extraversion, Neuroticism, ReligiousAffiliation, ReligiousService, Religious composite
    - Income information: Assigned Allowance Level, I-High indicator
    - Observed prosocial behavior: TWT+Sospeso
    - Configuration values: WOPB, WPB, Intercept
    - Calculated values: PB_i (anchored prosocial behavior), DI_i (continuous value)
    - Income (actual income value)
    - Final decision: disclose_income (1/0)
    
    IMPORTANT: Columns after disclose_income are intentionally excluded.
    
    Args:
        df: Results dataframe with agent data
        
    Returns:
        DataFrame formatted for Excel export with 19 columns, or None if required columns missing
    """
    # Check required column
    if 'disclose_income' not in df.columns:
        return None
    
    # Create export dataframe
    export_df = pd.DataFrame()
    
    # ========================================================================
    # 1. Agent ID
    # ========================================================================
    if 'agent_id' in df.columns:
        export_df['Agent ID'] = df['agent_id']
    elif 'index' in df.columns:
        export_df['Agent ID'] = df['index'] + 1  # Convert 0-based to 1-based
    else:
        export_df['Agent ID'] = range(1, len(df) + 1)
    
    # ========================================================================
    # 2-6. Raw Personality Trait Values (non-standardized)
    # ========================================================================
    
    # 2. Agreeable
    if 'Agreeable' in df.columns:
        export_df['Agreeable'] = df['Agreeable']
    else:
        export_df['Agreeable'] = ''
    
    # 3. Openness (from OpennessBig5)
    if 'OpennessBig5' in df.columns:
        export_df['Openness'] = df['OpennessBig5']
    else:
        export_df['Openness'] = ''
    
    # 4. Honesty_Humility
    if 'Honesty_Humility' in df.columns:
        export_df['Honesty_Humility'] = df['Honesty_Humility']
    else:
        export_df['Honesty_Humility'] = ''
    
    # 5. Extraversion (from ExtraversionBig5)
    if 'ExtraversionBig5' in df.columns:
        export_df['Extraversion'] = df['ExtraversionBig5']
    else:
        export_df['Extraversion'] = ''
    
    # 6. Neuroticism (from NeuroticismBig5)
    if 'NeuroticismBig5' in df.columns:
        export_df['Neuroticism'] = df['NeuroticismBig5']
    else:
        export_df['Neuroticism'] = ''
    
    # ========================================================================
    # 7-9. Religious Components (raw values + computed composite)
    # ========================================================================
    
    # 7. ReligiousAffiliation (raw binary 0/1)
    if 'ReligiousAffiliation' in df.columns:
        export_df['ReligiousAffiliation'] = df['ReligiousAffiliation']
    else:
        export_df['ReligiousAffiliation'] = ''
    
    # 8. ReligiousService (raw ordinal)
    if 'ReligiousService' in df.columns:
        export_df['ReligiousService'] = df['ReligiousService']
    else:
        export_df['ReligiousService'] = ''
    
    # 9. Religious composite (computed, non-standardized)
    # This comes from the decision function output
    if 'disclose_income_religious_composite' in df.columns:
        export_df['Religious'] = df['disclose_income_religious_composite']
    else:
        # Fallback: compute it here if not available
        # Religious = (ReligiousAffiliation + scaled_ReligiousService) / 2
        # where scaled_ReligiousService = ReligiousService / 4 (assuming max=4)
        if 'ReligiousAffiliation' in df.columns and 'ReligiousService' in df.columns:
            rs_scaled = df['ReligiousService'] / 4.0  # Scale to 0-1
            export_df['Religious'] = (df['ReligiousAffiliation'] + rs_scaled) / 2
        else:
            export_df['Religious'] = ''
    
    # ========================================================================
    # 10-12. Income Information
    # ========================================================================
    
    # 10. Assigned Allowance Level
    if 'Assigned Allowance Level' in df.columns:
        export_df['Assigned Allowance Level'] = df['Assigned Allowance Level']
    elif 'actual_allowance' in df.columns:
        export_df['Assigned Allowance Level'] = df['actual_allowance']
    else:
        export_df['Assigned Allowance Level'] = ''
    
    # 11. Income (right after Assigned Allowance Level)
    if 'income' in df.columns:
        export_df['income'] = df['income']
    elif 'actual_allowance' in df.columns:
        export_df['income'] = df['actual_allowance']
    else:
        export_df['income'] = ''
    
    # 12. I-High (income_high indicator: 1 if level > 3, else 0)
    if 'disclose_income_income_high' in df.columns:
        export_df['I-High'] = df['disclose_income_income_high']
    else:
        # Fallback: compute from Assigned Allowance Level
        if 'Assigned Allowance Level' in df.columns:
            export_df['I-High'] = (df['Assigned Allowance Level'] > 3).astype(int)
        else:
            export_df['I-High'] = ''
    
    # ========================================================================
    # 13. Observed Prosocial Behavior
    # ========================================================================
    
    # TWT+Sospeso (observed prosocial behavior)
    if 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}' in df.columns:
        export_df['TWT+Sospeso'] = df['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    else:
        export_df['TWT+Sospeso'] = ''
    
    # ========================================================================
    # 14. Predicted Prosocial Behavior (trait-based)
    # ========================================================================
    
    # calc_PB (weighted_prosocial from traits, before anchoring)
    if 'disclose_income_weighted_prosocial' in df.columns:
        export_df['calc_PB'] = df['disclose_income_weighted_prosocial']
    else:
        export_df['calc_PB'] = ''
    
    # ========================================================================
    # 15-17. Configuration Values (Weights and Intercept)
    # ========================================================================
    
    # 14. WOPB (Observed Prosocial Behavior Weight)
    if 'disclose_income_wopb' in df.columns:
        export_df['WOPB'] = df['disclose_income_wopb']
    else:
        # Default value from config
        export_df['WOPB'] = 0.25
    
    # 15. WPB (Prosocial Behavior Weight in final equation)
    if 'disclose_income_wpb' in df.columns:
        export_df['WPB'] = df['disclose_income_wpb']
    else:
        # Default value from config
        export_df['WPB'] = 0.50
    
    # 16. Intercept (β₀)
    if 'disclose_income_intercept' in df.columns:
        export_df['Intercept'] = df['disclose_income_intercept']
    else:
        # Default value from config
        export_df['Intercept'] = 0.75
    
    # ========================================================================
    # 17-18. Calculated Values
    # ========================================================================
    
    # 17. PB_i (Anchored Prosocial Behavior)
    if 'disclose_income_anchored_pb' in df.columns:
        export_df['PB_i'] = df['disclose_income_anchored_pb']
    else:
        export_df['PB_i'] = ''
    
    # 18. Disclosure Income (Continuous value before Y/N classification)
    if 'disclose_income_di' in df.columns:
        export_df['Disclosure Income'] = df['disclose_income_di']
    elif 'disclose_income_raw' in df.columns:
        export_df['Disclosure Income'] = df['disclose_income_raw']
    else:
        export_df['Disclosure Income'] = ''
    
    # ========================================================================
    # 19. Final Decision (LAST COLUMN - nothing after this)
    # ========================================================================
    
    # disclose_income (Y/N to 1/0)
    export_df['Disclose Income (Y=1)'] = df['disclose_income'].apply(
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


def render_disclose_income_comparison_excel(results_dict, mode="compare_all"):
    """
    Render Excel export section for disclose_income comparison modes.
    
    Handles two modes:
    - "compare_all": Multiple sheets (one per population mode: Copula, ResSpec, ResBase)
    - "compare_both": Single sheet with separate columns for Categorical and Continuous
    
    Args:
        results_dict: Dictionary of DataFrames keyed by configuration name
        mode: Either "compare_all" or "compare_both"
    """
    from io import BytesIO
    from datetime import datetime
    import numpy as np
    
    st.markdown("### 📥 Download Disclose Income Comparison Data")
    
    if mode == "compare_all":
        # COMPARE ALL MODE: Each population mode gets its own sheet
        st.markdown(f"""
        **Disclose Income Results Export (Compare All Mode - {len(results_dict)} Configurations):**
        - Each population mode (Copula, Research Spec, Research Baseline) has its own sheet
        - Each sheet contains all 19 columns for disclose income analysis
        - Includes both Categorical and Continuous income mode results where available
        """)
        
        population_modes = [
            ('copula', 'Copula'),
            ('research_spec', 'ResSpec'),
            ('research_baseline', 'ResBase')
        ]
        
        sheets_data = {}
        
        for pop_key, pop_prefix in population_modes:
            # Find the DataFrames for this population mode
            cat_key = f"{pop_key}_categorical"
            cont_key = f"{pop_key}_continuous"
            
            cat_df = results_dict.get(cat_key)
            cont_df = results_dict.get(cont_key)
            
            # Use whichever DataFrame is available for base data
            base_df = cat_df if cat_df is not None and not cat_df.empty else cont_df
            
            if base_df is None or base_df.empty:
                continue
            
            # Prepare the export DataFrame using the standard function
            sheet_df = _prepare_disclose_income_excel_data(base_df)
            
            if sheet_df is None:
                continue
            
            # If we have both income modes, add a suffix to the disclose_income column
            # and add the other mode's disclose_income
            if cat_df is not None and cont_df is not None and not cat_df.empty and not cont_df.empty:
                # Rename the existing disclose_income to specify it's from base
                if base_df is cat_df:
                    sheet_df = sheet_df.rename(columns={'disclose_income': 'disclose_income_Categorical'})
                    # Add continuous disclose_income
                    cont_di = cont_df['disclose_income'].apply(
                        lambda x: 1 if x == 'Y' else (0 if x == 'N' else '')
                    )
                    sheet_df['disclose_income_Continuous'] = cont_di.values
                else:
                    sheet_df = sheet_df.rename(columns={'disclose_income': 'disclose_income_Continuous'})
                    # Add categorical disclose_income
                    cat_di = cat_df['disclose_income'].apply(
                        lambda x: 1 if x == 'Y' else (0 if x == 'N' else '')
                    )
                    # Insert categorical before continuous
                    cols = list(sheet_df.columns)
                    cont_idx = cols.index('disclose_income_Continuous')
                    sheet_df.insert(cont_idx, 'disclose_income_Categorical', cat_di.values)
            
            sheets_data[pop_prefix] = sheet_df
        
        if not sheets_data:
            st.warning("⚠️ No data available for export")
            return
        
        # Create Excel file with multiple sheets
        try:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                for sheet_name, sheet_df in sheets_data.items():
                    sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)
                    _apply_price_formatting_disclosure(writer, sheet_name, sheet_df)
            
            # Show metrics
            total_sheets = len(sheets_data)
            first_sheet_df = next(iter(sheets_data.values()))
            n_agents = len(first_sheet_df)
            n_columns = len(first_sheet_df.columns)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sheets", total_sheets)
            with col2:
                st.metric("Agents per Sheet", n_agents)
            with col3:
                st.metric("Columns per Sheet", n_columns)
            
            excel_label = f"📊 Download Disclose Income Excel ({total_sheets} Sheets)"
            excel_filename = f"disclose_income_compare_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            st.download_button(
                label=excel_label,
                data=buffer.getvalue(),
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Each population mode has its own sheet with all disclose income columns"
            )
            
            # Show preview
            with st.expander("📋 Preview Disclose Income Data (first 5 rows per sheet)"):
                st.info("""
                **Sheet Structure:**
                - **Copula**: Synthetic agents generated from copula
                - **ResSpec**: Original 280 participants with stochastic draws
                - **ResBase**: Original 280 participants without stochastic draws
                
                Each sheet contains all 19 disclose income columns plus income mode columns if both are available.
                """)
                
                for sheet_name, sheet_df in sheets_data.items():
                    st.markdown(f"**{sheet_name} Sheet:**")
                    st.dataframe(sheet_df.head(), use_container_width=True)
                    st.caption(f"Columns: {', '.join(sheet_df.columns[:10])}{'...' if len(sheet_df.columns) > 10 else ''}")
        
        except Exception as e:
            st.error(f"Error creating Excel export: {str(e)}")
    
    elif mode == "compare_both":
        # COMPARE BOTH MODE: Single sheet with separate columns for each income mode
        st.markdown(f"""
        **Disclose Income Results Export (Compare Both Income Modes):**
        - Single sheet with all trait columns
        - Separate disclose_income columns for Categorical and Continuous modes
        """)
        
        # Get the DataFrames
        cat_df = results_dict.get("categorical")
        cont_df = results_dict.get("continuous")
        
        if (cat_df is None or cat_df.empty) and (cont_df is None or cont_df.empty):
            st.warning("⚠️ No data available for export")
            return
        
        # Use categorical as base if available, otherwise continuous
        base_df = cat_df if cat_df is not None and not cat_df.empty else cont_df
        
        # Prepare the export DataFrame
        export_df = _prepare_disclose_income_excel_data(base_df)
        
        if export_df is None:
            st.warning("⚠️ Unable to prepare Excel data")
            return
        
        # Handle the disclose_income columns based on what's available
        if cat_df is not None and cont_df is not None and not cat_df.empty and not cont_df.empty:
            # Both modes available - create separate columns
            # Remove the original disclose_income column
            export_df = export_df.drop(columns=['disclose_income'])
            
            # Add categorical disclose_income
            cat_di = cat_df['disclose_income'].apply(
                lambda x: 1 if x == 'Y' else (0 if x == 'N' else '')
            )
            export_df['disclose_income_Categorical'] = cat_di.values
            
            # Add continuous disclose_income
            cont_di = cont_df['disclose_income'].apply(
                lambda x: 1 if x == 'Y' else (0 if x == 'N' else '')
            )
            export_df['disclose_income_Continuous'] = cont_di.values
        elif cat_df is not None and not cat_df.empty:
            # Only categorical
            export_df = export_df.rename(columns={'disclose_income': 'disclose_income_Categorical'})
        else:
            # Only continuous
            export_df = export_df.rename(columns={'disclose_income': 'disclose_income_Continuous'})
        
        try:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Disclose Income Comparison')
                _apply_price_formatting_disclosure(writer, 'Disclose Income Comparison', export_df)
            
            # Show metrics
            st.metric("Total Agents", len(export_df))
            
            excel_label = "📊 Download Disclose Income Excel (Both Income Modes)"
            excel_filename = f"disclose_income_compare_both_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            st.download_button(
                label=excel_label,
                data=buffer.getvalue(),
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Disclose income results with separate columns for Categorical and Continuous income modes"
            )
            
            # Show preview
            with st.expander("📋 Preview Disclose Income Data (first 5 rows)"):
                st.dataframe(export_df.head(), use_container_width=True)
                st.caption(f"**Columns ({len(export_df.columns)})**: {', '.join(export_df.columns[:10])}{'...' if len(export_df.columns) > 10 else ''}")
        
        except Exception as e:
            st.error(f"Error creating Excel export: {str(e)}")

