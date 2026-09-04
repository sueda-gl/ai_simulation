# app/pages/results/visualizations/disclosure_viz.py
"""
Disclosure-related visualization functions.
Handles disclose_income and disclose_documents decisions.

The Excel sheet builders (`prepare_*`), their number formatting and the
statistics tables live in `app/reports/disclosure.py` (pure pandas/openpyxl, no
Streamlit); this module renders the charts, metrics and download buttons.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.reports.disclosure import (
    build_disclose_documents_xlsx,
    build_disclose_income_xlsx,
    build_disclosure_xlsx,
    customer_type_summary_frame,
    prepare_disclose_documents_excel_data,
    prepare_disclose_income_excel_data,
    prepare_disclosure_excel_data,
    raw_stats_frame,
    raw_value_stats,
)
from app.pages.results.run_context import RunContext


def _income_mode_suffix():
    """Title suffix of the raw DI/DD histograms - the RUN's income mode (ruling R28).

    The three-way test below is the one the screen has always used, character for
    character; only its INPUT changed.  It used to read the live
    ``di_income_mode`` / ``dd_income_mode`` tab keys, which describe the NEXT run
    the user is configuring, so a continuous run was captioned "(Categorical)"
    whenever the tab still sat on its default.  It now reads the executed run's
    own effective income mode off the run metadata (``RunContext``).

    Keeping the original branch - rather than collapsing it to a two-way
    categorical/continuous test - matters for a "Compare both" run: that string
    contains neither word, so the title carries NO suffix, exactly as before.
    """
    income_mode = RunContext.from_session().effective_income_mode
    if 'categorical' in str(income_mode).lower():
        return " (Categorical)"
    elif 'continuous' in str(income_mode).lower():
        return " (Continuous)"
    else:
        return ""


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
        st.metric("Disclosed income (Y)", f"{yes_count:,} ({pct_yes:.2f}%)")
    with col3:
        no_count = value_counts.get('N', 0)
        pct_no = (no_count/total)*100
        st.metric("Not disclosed income (N)", f"{no_count:,} ({pct_no:.2f}%)")
    with col4:
        disclosure_rate = (yes_count/total)*100
        st.metric("Disclosure Rate", f"{disclosure_rate:.2f}%")

    # Check if raw DI values are available for histogram
    has_raw_values = 'disclose_income_raw' in df.columns

    if has_raw_values:
        # TWO-COLUMN LAYOUT: Pie chart on left, Histogram on right
        col_pie, col_hist = st.columns(2)

        with col_pie:
            st.markdown(f"**{decision_title} Distribution**")
            if len(value_counts) > 0:
                fig_pie = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}  # Green for Yes, Red for No
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_hist:
            # Get raw values for histogram
            raw_values = df['disclose_income_raw'].dropna()

            if len(raw_values) > 0:
                # Calculate statistics
                stats = raw_value_stats(raw_values)
                mean_val = stats['mean']

                # Calculate Y/N split based on threshold
                y_count_raw = (raw_values > 0).sum()
                n_count_raw = (raw_values <= 0).sum()
                total_raw = len(raw_values)
                y_pct_raw = (y_count_raw / total_raw) * 100 if total_raw > 0 else 0
                n_pct_raw = (n_count_raw / total_raw) * 100 if total_raw > 0 else 0

                mode_suffix = _income_mode_suffix()

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

                # Statistics below histogram
                col_stats, _ = st.columns(2)

                with col_stats:
                    st.markdown("**📈 Statistics**")
                    stats_df = raw_stats_frame(stats, 'Raw Disclose Income Value')
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
            else:
                st.warning("No raw DI values available for histogram")

    else:
        # No raw values available - show pie chart only
        if len(value_counts) > 0:
            st.markdown(f"**{decision_title} Distribution**")
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}  # Green for Yes, Red for No
            )
            st.plotly_chart(fig, use_container_width=True)

    # Excel download section
    st.markdown("---")
    st.markdown("### 📥 Download Agent Disclose Income Data")

    # Prepare Excel data
    excel_data = prepare_disclose_income_excel_data(df)

    if excel_data is not None:
        # Convert to Excel bytes
        excel_bytes = build_disclose_income_xlsx(excel_data)

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
            st.markdown(f"**Columns**: {', '.join(excel_data.columns)}")
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
        st.markdown(f"📊 Among the {qualified_agents:,} agents qualified for discount (income < threshold)")

        # Binary choice metrics for qualified agents only
        col1, col2, col3, col4 = st.columns(4)

        yes_count = value_counts.get('Y', 0)
        no_count = value_counts.get('N', 0)

        with col1:
            st.metric("Qualified Agents", f"{qualified_agents:,}")
        with col2:
            pct_yes = (yes_count/qualified_agents)*100
            st.metric("Disclosed documents (Y)", f"{yes_count:,} ({pct_yes:.2f}%)")
        with col3:
            pct_no = (no_count/qualified_agents)*100
            st.metric("Not disclosed documents (N)", f"{no_count:,} ({pct_no:.2f}%)")
        with col4:
            disclosure_rate = (yes_count/qualified_agents)*100
            st.metric("Disclosure Rate", f"{disclosure_rate:.2f}%",
                      help="Percentage of qualified agents who disclosed documents")

        # TWO-COLUMN LAYOUT: Pie chart on left, raw histogram on right (mirrors Disclose Income)
        col_pie, col_hist = st.columns(2)

        with col_pie:
            # Filter out NA for the pie chart
            qualified_counts = {k: v for k, v in value_counts.items() if k != 'NA'}
            if len(qualified_counts) > 0:
                st.markdown(f"**{decision_title} Distribution**")
                fig = px.pie(
                    values=list(qualified_counts.values()),
                    names=list(qualified_counts.keys()),
                    color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}  # Green for Yes, Red for No
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_hist:
            # Raw DD distribution for qualified agents (mirrors Disclose Income)
            if 'disclose_documents_raw' in df.columns:
                raw_values = df.loc[df['disclose_documents'] != 'NA', 'disclose_documents_raw'].dropna()
            else:
                raw_values = pd.Series([], dtype=float)

            if len(raw_values) > 0:
                stats = raw_value_stats(raw_values)
                mean_val = stats['mean']

                mode_suffix = _income_mode_suffix()

                st.markdown(f"**📈 Raw Disclose Documents Distribution{mode_suffix}**")

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=raw_values,
                    nbinsx=40,
                    name='DD Raw Values',
                    marker_color='steelblue',
                    opacity=0.7
                ))
                fig_hist.add_vline(
                    x=0, line_dash="solid", line_color="red", line_width=3,
                    annotation_text="Threshold (0)", annotation_position="top",
                    annotation_font_color="red"
                )
                fig_hist.add_vline(
                    x=mean_val, line_dash="dash", line_color="green", line_width=2,
                    annotation_text=f"Mean: {mean_val:.3f}", annotation_position="bottom",
                    annotation_font_color="green"
                )
                fig_hist.update_layout(
                    xaxis_title="Raw DD Value (>0 → Y, ≤0 → N)",
                    yaxis_title="Agents",
                    showlegend=False,
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis=dict(zeroline=True, zerolinecolor='red', zerolinewidth=2)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # Statistics below histogram
                col_stats, _ = st.columns(2)
                with col_stats:
                    st.markdown("**📈 Statistics**")
                    stats_df = raw_stats_frame(stats, 'Raw Disclose Documents Value')
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
            else:
                st.info("Raw DD values not available for this run.")
    else:
        st.warning("⚠️ No agents qualified for discount (all agents have income ≥ threshold)")

    # Excel download section — same agent-level file (same columns) as the individual-decision export
    st.markdown("---")
    st.markdown("### 📥 Download Agent Disclose Documents Data")
    dd_excel_data = prepare_disclose_documents_excel_data(df)
    if dd_excel_data is not None:
        dd_excel_bytes = build_disclose_documents_xlsx(dd_excel_data)
        st.download_button(
            label="📥 Download Agent Disclose Documents Data (Excel)",
            data=dd_excel_bytes,
            file_name="agent_disclose_documents_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download detailed agent data including traits and disclose documents decision"
        )
        with st.expander("📋 Preview Excel Data (first 10 rows)"):
            st.dataframe(dd_excel_data.head(10), use_container_width=True)
            st.markdown(f"**Columns**: {', '.join(dd_excel_data.columns)}")
    else:
        st.warning("⚠️ Unable to prepare Excel data. Some required columns may be missing.")

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
            st.markdown("Breakdown by pricing model")

            # Create summary table
            summary_df = customer_type_summary_frame(customer_stats)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.markdown("💡 Only **Regular Customers** participate in Purchase Now vs Bid decisions (Decision 9)")

        # Excel download section
        st.markdown("---")

        # Prepare Excel data
        excel_data = prepare_disclosure_excel_data(df)

        if excel_data is not None:
            # Convert to Excel bytes
            excel_bytes = build_disclosure_xlsx(excel_data)

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
