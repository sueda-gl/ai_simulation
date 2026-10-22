# app/pages/results/overview.py
"""
The results overview of one result frame (``show_overview``): the headline
metrics, then the donation-rate, disclose-income and disclose-documents
analyses with their "Use This Config" buttons, and the Decision 4 run-shape
helpers the results page shares (``rtd_page_element``, ``rtd_overview_metric``).
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.reports.donation import donation_default_stats_frame
from app.pages.results.visualizations.viz_helpers import yes_no_pie
from app.pages.results.config_pin import (
    render_disclose_documents_selection_button,
    render_disclose_income_selection_button,
    render_inline_selection_button,
)


# Per-element runs of the four RANKING elements have no meaningful headline metric:
# there is deliberately no "Mean <element> score" metric, so those runs show the
# agent count alone.
_RTD_NO_HEADLINE_ELEMENTS = ('loyalty', 'wtp', 'risk_taking', 'flexibility')


def rtd_page_element():
    """The Decision 4 element the results page shows: the rtd_run_element flag of an
    individual Decision 4 run ('ttp' ... 'flexibility', or 'aggregation' for "Run
    Integrated Default List Only"); None for whole-decision and combined runs."""
    from app.pages.results.run_context import RunContext
    from app.reports.rtd import RTD_AGGREGATION_ELEMENT, RTD_ALL_ELEMENTS

    ctx = RunContext.from_session()
    if not ctx.is_individual_run('rejected_transaction_defaults'):
        return None
    element = ctx.rtd_run_element      # the run's flag, not the live tab's
    if element in RTD_ALL_ELEMENTS or element == RTD_AGGREGATION_ELEMENT:
        return element
    return None


def rtd_overview_metric(df):
    """Headline metric for a Decision 4 model run, run-shape aware.

    Returns (label, formatted_value), or (None, None) when the run shape has no
    element-specific headline metric:
      - Loyalty / WTP / Risk-Taking / Flexibility per-element run -> (None, None);
      - "Run Integrated Default List Only"  -> average integrated default list length;
      - Options List Length per-element run, whole-decision runs and combined runs
        -> average options list length (the caller adds its min/max).

    The element is rtd_page_element()'s: the rtd_run_element flag of an individual
    Decision 4 run (RunContext, which describes the run that happened).
    """
    element = rtd_page_element()

    if element in _RTD_NO_HEADLINE_ELEMENTS:
        return None, None
    if element == 'aggregation':
        if 'rtd_default_list_length' in df.columns:
            return ("Avg. integrated default list length",
                    f"{df['rtd_default_list_length'].mean():.2f}")
        return None, None
    return "Avg. Options List Length", f"{df['rtd_choice_length'].mean():.2f}"


def show_overview(df, title_suffix="", result_key=None, enable_selection=False):
    """Helper function to show simulation overview for a DataFrame
    
    Args:
        df: Results DataFrame
        title_suffix: Additional text for titles
        result_key: Unique key for this result (needed for selection)
        enable_selection: Whether to show inline selection button
    """
    st.subheader(f"Simulation Overview{title_suffix}")
    
    # Disclose-Documents-only run: the disclose_income column is present solely to drive the
    # eligibility gate, so none of its metrics/graphs/analysis should appear on the Disclose
    # Documents results page. Detected via the DD-only run flag (the run's own
    # custom-decision list, i.e. what actually ran).
    from app.pages.results.run_context import RunContext
    ctx = RunContext.from_session()
    is_dd_focus = ctx.custom_decisions == ('disclose_documents',)
    
    # Display anchor weights info (donation-specific; hidden on Decision 4 model runs):
    # the mix the RUN used, not the live slider
    if 'rtd_choice_length' not in df.columns or 'donation_default' in df.columns:
        st.caption(f"📊 Anchor mix: {ctx.anchor_observed_weight:.2f} observed | {ctx.anchor_predicted_weight:.2f} predicted")

    # Headline decision drives the metric layout. "Traits Available" / "Decisions Computed"
    # are deliberately not shown. Priority mirrors the old headline metric:
    # donation > income > documents; is_dd_focus forces the documents layout on DD-only runs.
    donation_col = 'donation_default'
    has_donation = donation_col in df.columns
    has_income = 'disclose_income' in df.columns and not is_dd_focus
    has_documents = 'disclose_documents' in df.columns
    has_rtd = 'rtd_choice_length' in df.columns

    if has_rtd and not has_donation and not has_income and not has_documents:
        # Decision 4 (Rejected Transaction Defaults) model run: no donation /
        # disclosure metrics exist - show the D4-relevant headline metrics instead.
        # The metric depends on the run shape (see rtd_overview_metric): a per-element
        # run of a RANKING element shows no element metric at all.
        label, value = rtd_overview_metric(df)
        if label is None:
            st.metric("Total Agents", f"{len(df):,}")
        elif label == "Avg. Options List Length":
            # Options List Length element / whole-decision run: min and max Options
            # List Length shown next to the average.
            lengths = df['rtd_choice_length'].astype(int)
            col1, col2, col3, col4 = st.columns([1, 1.2, 1, 1])
            col1.metric("Total Agents", f"{len(df):,}")
            col2.metric(label, value)
            col3.metric("Min Options List Length", f"{int(lengths.min())}")
            col4.metric("Max Options List Length", f"{int(lengths.max())}")
        else:
            col1, col2 = st.columns([1, 1.2])
            col1.metric("Total Agents", f"{len(df):,}")
            col2.metric(label, value)
    elif has_documents and not has_donation and not has_income:
        # Decision 2 (Disclose Documents): the realized all-agent disclosure rate
        # (agents who actually disclosed ÷ total agents — only qualified agents can disclose,
        # everyone else is 'NA'/'N'), the qualified-agent count, and the rate among qualified only.
        qualified = df.loc[df['disclose_documents'] != 'NA', 'disclose_documents']
        qualified_rate = qualified.eq('Y').mean() if len(qualified) > 0 else 0
        overall_rate = (df['disclose_documents'] == 'Y').mean() if len(df) > 0 else 0
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])
        col1.metric("Total Agents", f"{len(df):,}")
        col2.metric("Disclosure Rate (all agents)", f"{overall_rate:.2%}")
        col3.metric("Qualified Agents", f"{len(qualified):,}")
        col4.metric("Disclose Documents Rate (qualified)", f"{qualified_rate:.2%}")
    else:
        # Decisions 1 & 3 (and any other headline): Total Agents + the headline rate only.
        col1, col2 = st.columns([1, 1.2])
        col1.metric("Total Agents", f"{len(df):,}")
        if has_donation:
            col2.metric("Avg Donation Rate", f"{df[donation_col].mean():.2%}")
        elif has_income:
            col2.metric("Disclose Income (Y)", f"{(df['disclose_income'] == 'Y').mean():.2%}")
    
    # Donation rate analysis (if available) - always use truncated
    donation_col = 'donation_default'
    if donation_col in df.columns:
        st.subheader(f"📊 Donation Rate Analysis{title_suffix}")
        
        # Distribution plot
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(
                df, 
                x=donation_col,
                nbins=30,
                title=f"Distribution of Donation Rates{title_suffix}",
                labels={donation_col: 'Donation Rate', 'count': 'Number of Agents'},
                marginal="box"
            )
            fig.update_layout(
                xaxis_tickformat='.0%',
                showlegend=False
            )
            chart_key = f"donation_hist_{result_key}" if result_key else f"donation_hist_{title_suffix}"
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        
        with col2:
            st.markdown("**📈 Statistics**")
            stats_df = donation_default_stats_frame(df[donation_col])
            st.dataframe(stats_df, hide_index=True)
    
    # Disclose Income analysis (if available; hidden on Disclose-Documents-only runs, where the
    # disclose_income column exists only to drive the eligibility gate)
    if 'disclose_income' in df.columns and not is_dd_focus:
        st.subheader(f"📊 Disclose Income Analysis{title_suffix}")
        
        # Check if we have raw DI values for detailed analysis
        has_raw_values = 'disclose_income_raw' in df.columns
        
        if has_raw_values:
            # DETAILED ANALYSIS: Show histogram of raw values before Y/N classification
            show_disclose_income_rate_analysis(df, title_suffix, result_key, enable_selection)
        else:
            # BASIC ANALYSIS: Only show pie chart of Y/N outcomes (no raw values available)
            value_counts = df['disclose_income'].value_counts()
            total = len(df)
            
            # Pie chart (full width)
            if len(value_counts) > 0:
                fig = yes_no_pie(value_counts)
                chart_key = f"disclose_income_pie_{result_key}" if result_key else f"disclose_income_pie_{title_suffix}"
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
            
            # Choice breakdown table below the chart
            st.markdown("**📊 Choice Breakdown**")
            # Ensure Y appears before N in the breakdown
            ordered_choices = ['Y', 'N']
            choice_labels = {'Y': 'Y (disclose income)', 'N': 'N (not disclose income)'}
            ordered_data = []
            for choice in ordered_choices:
                if choice in value_counts.index:
                    count = value_counts[choice]
                    ordered_data.append({
                        'Choice': choice_labels[choice],
                        'Count': count,
                        'Percentage': f"{(count/total)*100:.2f}%"
                    })

            breakdown_df = pd.DataFrame(ordered_data)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # Disclose Documents analysis (if available)
    if 'disclose_documents' in df.columns:
        st.subheader(f"📊 Disclose Documents Analysis{title_suffix}")
        if 'disclose_documents_raw' in df.columns:
            # DETAILED ANALYSIS: histogram of raw scores (qualified agents only)
            show_disclose_documents_rate_analysis(df, title_suffix, result_key, enable_selection)
        else:
            # BASIC ANALYSIS: Y/N pie over qualified (non-NA) agents only
            qualified = df.loc[df['disclose_documents'] != 'NA', 'disclose_documents']
            value_counts = qualified.value_counts()
            if len(value_counts) > 0:
                fig = yes_no_pie(value_counts)
                chart_key = f"disclose_documents_pie_{result_key}" if result_key else f"disclose_documents_pie_{title_suffix}"
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
            st.markdown("**📊 Choice Breakdown (qualified agents)**")
            total_q = len(qualified)
            choice_labels = {'Y': 'Y (disclose documents)', 'N': 'N (not disclose documents)'}
            ordered_data = []
            for choice in ['Y', 'N']:
                if choice in value_counts.index:
                    count = value_counts[choice]
                    ordered_data.append({
                        'Choice': choice_labels[choice], 'Count': int(count),
                        'Percentage': f"{(count/total_q)*100:.2f}%" if total_q else "0.00%"
                    })
            st.dataframe(pd.DataFrame(ordered_data), use_container_width=True, hide_index=True)
            if enable_selection and result_key:
                render_disclose_documents_selection_button(result_key, df)

    # Add inline selection button if enabled (only for donation_default results)
    # Disclose income/documents have their own selection buttons in their rate-analysis helpers
    if enable_selection and result_key and 'donation_default' in df.columns:
        render_inline_selection_button(result_key, df)

    # Decision 4 (Rejected Transaction Defaults): its "Use This Config" button renders
    # UNDER the decision's detailed results (transaction_viz._render_rtd_model_results),
    # like the other decisions' buttons sit under their result charts - not here.


def show_disclose_income_rate_analysis(df, title_suffix="", result_key=None, enable_selection=False):
    """
    Display Disclose Income Rate Analysis with histogram of raw DI values.
    
    Shows the distribution of raw disclose income values BEFORE classification
    (i.e., before the >0 → Y, ≤0 → N threshold is applied).
    
    The histogram includes a clear vertical line at 0 to show the decision boundary.
    
    Args:
        df: DataFrame with 'disclose_income_raw' column containing raw DI values
        title_suffix: Additional text for titles
        result_key: Unique key for this result (needed for selection buttons)
        enable_selection: Whether to show "Use This Config" button
    """
    raw_col = 'disclose_income_raw'
    
    if raw_col not in df.columns:
        st.warning("Raw DI values not available - showing basic Y/N analysis only")
        return
    
    # Get the raw values
    raw_values = df[raw_col].dropna()
    
    if len(raw_values) == 0:
        st.warning("No raw DI values found in results")
        return
    
    # Calculate statistics
    mean_val = raw_values.mean()
    std_val = raw_values.std()
    min_val = raw_values.min()
    max_val = raw_values.max()
    median_val = raw_values.median()
    
    # Calculate Y/N split based on threshold
    y_count = (raw_values > 0).sum()
    n_count = (raw_values <= 0).sum()
    total = len(raw_values)
    y_pct = (y_count / total) * 100 if total > 0 else 0
    n_pct = (n_count / total) * 100 if total > 0 else 0
    
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
        annotation_text="Threshold (0)",
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
    
    # Update layout
    fig.update_layout(
        title=f"Raw Disclose Income Distribution{title_suffix}",
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
    
    chart_key = f"di_raw_hist_{result_key}" if result_key else f"di_raw_hist_{title_suffix}"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    
    # Statistics + classification panels BELOW the graph
    # Stacked full-width tables (Classification under Statistics) so all numbers show fully.
    st.markdown("**📈 Statistics**")
    stats_df = pd.DataFrame({
        'Statistic': ['Mean', 'Std Dev', 'Median', 'Min', 'Max'],
        'Raw Disclose Income Value': [
            f"{mean_val:.4f}",
            f"{std_val:.4f}",
            f"{median_val:.4f}",
            f"{min_val:.4f}",
            f"{max_val:.4f}"
        ]
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)
    st.markdown("**📊 Classification**")
    st.dataframe(pd.DataFrame({
        'Choice': ['Y (disclose income)', 'N (not disclose income)', 'Total'],
        'Count': [int(y_count), int(n_count), int(total)],
        '%': [f"{y_pct:.2f}%", f"{n_pct:.2f}%", "100.00%"]
    }), hide_index=True, use_container_width=True)

    # Show key insight about distribution relative to threshold
    if mean_val > 0:
        st.success(f"✅ Mean ({mean_val:.4f}) > 0: Distribution favors disclosure")
    elif mean_val < 0:
        st.warning(f"⚠️ Mean ({mean_val:.4f}) < 0: Distribution favors non-disclosure")
    else:
        st.info("ℹ️ Mean ≈ 0: Distribution is balanced")
    
    # Show "Use This Config" button if enabled
    if enable_selection and result_key:
        render_disclose_income_selection_button(result_key, df)


def show_disclose_documents_rate_analysis(df, title_suffix="", result_key=None, enable_selection=False):
    """
    Display Disclose Documents Rate Analysis: histogram of raw DD scores BEFORE the
    >0 -> Y, <=0 -> N threshold. NA (ineligible) agents have no raw value and are dropped.
    Mirrors show_disclose_income_rate_analysis but with DD keys and NA handling.
    """
    raw_col = 'disclose_documents_raw'
    if raw_col not in df.columns:
        st.warning("Raw DD values not available - showing basic Y/N analysis only")
        return

    # Qualified subgroup = agents the gate did NOT send to NA (disclosed income AND income <
    # threshold). Identify them by the DECISION column, NOT by raw being NaN: the model now
    # emits a raw score for EVERY agent (including gated NAs) for export, so a bare
    # raw.dropna() would wrongly include ineligible agents (showing the all-agent model rate
    # instead of the qualified subgroup).
    if 'disclose_documents' in df.columns:
        raw_values = df.loc[df['disclose_documents'] != 'NA', raw_col].dropna()
    else:
        raw_values = df[raw_col].dropna()
    if len(raw_values) == 0:
        st.warning("No qualified agents (all income >= discount threshold) - no raw DD values")
        return

    mean_val = raw_values.mean(); std_val = raw_values.std()
    min_val = raw_values.min(); max_val = raw_values.max(); median_val = raw_values.median()
    y_count = int((raw_values > 0).sum()); n_count = int((raw_values <= 0).sum())
    total = len(raw_values)
    y_pct = (y_count / total) * 100 if total > 0 else 0
    n_pct = (n_count / total) * 100 if total > 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=raw_values, nbinsx=40, name='DD Raw Values',
                               marker_color='steelblue', opacity=0.7))
    fig.add_vline(x=0, line_dash="solid", line_color="red", line_width=3,
                  annotation_text="Threshold (0)", annotation_position="top", annotation_font_color="red")
    fig.add_vline(x=mean_val, line_dash="dash", line_color="green", line_width=2,
                  annotation_text=f"Mean: {mean_val:.3f}", annotation_position="bottom", annotation_font_color="green")
    fig.update_layout(
        title=f"Raw Disclose Documents Distribution{title_suffix}",
        xaxis_title="Raw DD Value (>0 → Y, ≤0 → N)", yaxis_title="Qualified Agents",
        showlegend=False, height=300, margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(zeroline=True, zerolinecolor='red', zerolinewidth=2)
    )
    chart_key = f"dd_raw_hist_{result_key}" if result_key else f"dd_raw_hist_{title_suffix}"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


    # Stacked full-width tables (Classification under Statistics) so all numbers show fully.
    st.markdown("**📈 Statistics** (qualified agents)")
    st.dataframe(pd.DataFrame({
        'Statistic': ['Mean', 'Std Dev', 'Median', 'Min', 'Max'],
        'Raw Disclose Documents Value': [f"{mean_val:.4f}", f"{std_val:.4f}", f"{median_val:.4f}", f"{min_val:.4f}", f"{max_val:.4f}"]
    }), hide_index=True, use_container_width=True)
    st.markdown("**📊 Classification** (qualified agents)")
    st.dataframe(pd.DataFrame({
        'Choice': ['Y (disclose documents)', 'N (not disclose documents)', 'Total'],
        'Count': [y_count, n_count, total],
        '%': [f"{y_pct:.2f}%", f"{n_pct:.2f}%", "100.00%"]
    }), hide_index=True, use_container_width=True)

    if mean_val > 0:
        st.success(f"✅ Mean ({mean_val:.4f}) > 0: Distribution favors disclosure")
    elif mean_val < 0:
        st.warning(f"⚠️ Mean ({mean_val:.4f}) < 0: Distribution favors non-disclosure")
    else:
        st.info("ℹ️ Mean ≈ 0: Distribution is balanced")

    if enable_selection and result_key:
        render_disclose_documents_selection_button(result_key, df)
