# app/components.py
"""
UI components and visualization functions for the Enhanced AI Agent Simulation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import yaml

from app.models import get_decision_global_parameters, get_all_global_parameters


def rtd_overview_metric(df):
    """Headline metric for a Decision 4 model run, element-aware.

    On a per-element run (rtd_run_element set for an individual Decision 4 run) the
    metric is the run element's own mean score; otherwise the whole-decision metric
    (average options list length). Returns (label, formatted_value).
    """
    element = None
    if (getattr(st.session_state, 'custom_decisions', None) == ['rejected_transaction_defaults']
            and not getattr(st.session_state, 'default_decisions', [])):
        element = st.session_state.get('rtd_run_element')

    specs = {
        # All three use the STANDARDIZED score (matches the results charts;
        # professor 2026-08: all elements present standardized results).
        'loyalty': ("Mean Loyalty score", 'rtd_loyalty_z', "{:.4f}"),
        'wtp': ("Mean Willingness-to-Pay score", 'rtd_wtp_z', "{:.4f}"),
        'risk_taking': ("Mean Risk-Taking score", 'rtd_rt_z', "{:.4f}"),
    }
    if element in specs:
        label, col, fmt = specs[element]
        if col in df.columns:
            return label, fmt.format(df[col].mean())
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
    # Documents results page (per professor feedback). Detected via the DD-only run flag.
    is_dd_focus = (
        hasattr(st.session_state, 'custom_decisions')
        and st.session_state.custom_decisions == ['disclose_documents']
    )
    
    # Display anchor weights info (donation-specific; hidden on Decision 4 model runs)
    if 'rtd_choice_length' not in df.columns or 'donation_default' in df.columns:
        st.caption(f"📊 Anchor mix: {st.session_state.anchor_observed_weight:.2f} observed | {1 - st.session_state.anchor_observed_weight:.2f} predicted")

    # Headline decision drives the metric layout. "Traits Available" / "Decisions Computed"
    # removed per professor feedback. Priority mirrors the old headline metric:
    # donation > income > documents; is_dd_focus forces the documents layout on DD-only runs.
    donation_col = 'donation_default'
    has_donation = donation_col in df.columns
    has_income = 'disclose_income' in df.columns and not is_dd_focus
    has_documents = 'disclose_documents' in df.columns
    has_rtd = 'rtd_choice_length' in df.columns

    if has_rtd and not has_donation and not has_income and not has_documents:
        # Decision 4 (Rejected Transaction Defaults) model run: no donation /
        # disclosure metrics exist - show the D4-relevant headline metrics instead.
        # On a per-ELEMENT run, the headline metric must be the run element's own
        # (a Loyalty-only run must not show the Options List Length metric).
        label, value = rtd_overview_metric(df)
        if label == "Avg. Options List Length":
            # Whole-decision run: min and max Options List Length shown next to
            # the average (professor 2026-08 request).
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
            donation_stats = df[donation_col].describe()
            
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
                'Donation Rate': [
                    f"{donation_stats['mean']:.2%}",
                    f"{donation_stats['std']:.2%}",
                    f"{donation_stats['min']:.2%}",
                    f"{donation_stats['max']:.2%}",
                    f"{donation_stats['50%']:.2%}",
                    f"{donation_stats['25%']:.2%}",
                    f"{donation_stats['75%']:.2%}"
                ]
            })
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
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}  # Green for Yes, Red for No
                )
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
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}
                )
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


def render_seed_mismatch_error(decision_name, error_info):
    """
    Render a clear error message when seed mismatch is detected.
    
    Args:
        decision_name: Name of the decision being saved (e.g., 'disclose_income')
        error_info: Dict with keys: new_seed, new_n_agents, existing_seed, existing_n_agents, conflicting_decision
    """
    from app.pages.decision_execution import clear_decision_config
    
    conflicting = error_info['conflicting_decision']
    conflicting_title = conflicting.replace('_', ' ').title()
    decision_title = decision_name.replace('_', ' ').title()
    
    st.error(f"""
**Seed Mismatch Detected**

Your **{decision_title}** configuration was run with:
- Seed: **{error_info['new_seed']}**
- Agents: **{error_info['new_n_agents']:,}**

But you already have a **{conflicting_title}** configuration saved with:
- Seed: **{error_info['existing_seed']}**
- Agents: **{error_info['existing_n_agents']:,}**

**Why this matters:** To ensure consistent results across decisions, all saved configurations must use the same seed and agent count.

**Action Required:** Re-run {decision_title} with seed={error_info['existing_seed']} and n_agents={error_info['existing_n_agents']}, then select that configuration.
    """)
    
    # Offer option to clear the conflicting config instead
    if st.button(
        f"🗑️ Clear {conflicting_title} config instead",
        key=f"clear_conflict_{decision_name}_{conflicting}",
        help=f"Remove the saved {conflicting_title} config so you can use this {decision_title} config"
    ):
        clear_decision_config(conflicting)
        st.success(f"Cleared {conflicting_title} configuration. Click 'Use This Config' again.")
        st.rerun()


def render_inline_selection_button(result_key, result_df):
    """Render selection button directly under the chart (for donation_default)"""
    
    # Import here to avoid circular imports
    from app.pages.decision_execution import (
        save_decision_config,
        is_decision_config_selected,
        get_current_coefficients,
        get_current_stochastic_params,
        calculate_result_metrics,
        extract_configuration_details
    )
    
    is_selected = is_decision_config_selected('donation_default', result_key)
    
    # Create a compact selection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show key metric for quick reference - check which columns are available
        donation_col = 'donation_default'
        if donation_col in result_df.columns:
            mean_donation = result_df[donation_col].mean()
            st.caption(f"📊 Quick Summary: {len(result_df):,} agents, avg donation {mean_donation:.2%}")
        elif 'disclose_income' in result_df.columns:
            y_rate = (result_df['disclose_income'] == 'Y').mean()
            st.caption(f"📊 Quick Summary: {len(result_df):,} agents, Y rate {y_rate:.2%}")
        else:
            st.caption(f"📊 Quick Summary: {len(result_df):,} agents")
    
    with col2:
        # Selection button
        if is_selected:
            st.success("✅ Selected")
        else:
            if st.button(
                "🎯 Use This Config",
                key=f"inline_select_{result_key}",
                type="primary",
                use_container_width=True,
                help="Select this configuration for combined simulations"
            ):
                # Get donation-specific parameters
                coefficients = get_current_coefficients()
                stochastic_params = get_current_stochastic_params()
                config_details = extract_configuration_details(result_key)
                
                params = {
                    'coefficients': coefficients,
                    'stochastic_params': stochastic_params,
                    'income_mode': config_details['income_spec_mode']
                }
                
                metrics = calculate_result_metrics(result_df)
                
                extra_data = {
                    'population_mode': config_details['population_mode'],
                    'income_spec_mode': config_details['income_spec_mode']
                }
                
                # Use unified save function with seed validation
                success, config, error_info = save_decision_config(
                    'donation_default', result_key, result_df, params, metrics, extra_data
                )
                
                if success:
                    st.success("Configuration selected!")
                    st.rerun()
                else:
                    # Show seed mismatch error
                    render_seed_mismatch_error('donation_default', error_info)


def show_disclose_income_overview(df, title_suffix="", result_key=None, enable_selection=False):
    """
    Display compact Disclose Income overview for comparison views.
    
    Shows key metrics: Total Agents, Y count, N count, Disclosure Rate.
    Suitable for side-by-side comparison grids.
    
    Args:
        df: DataFrame with 'disclose_income' column
        title_suffix: Additional text for titles
        result_key: Unique key for this result (needed for selection buttons)
        enable_selection: Whether to show "Use This Config" button
    """
    st.subheader(f"Disclose Income Overview{title_suffix}")
    
    if 'disclose_income' not in df.columns:
        st.warning("disclose_income column not found in results")
        return
    
    # Calculate metrics
    total = len(df)
    y_count = (df['disclose_income'] == 'Y').sum()
    n_count = (df['disclose_income'] == 'N').sum()
    y_rate = (y_count / total) * 100 if total > 0 else 0
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Agents", f"{total:,}")
    with col2:
        st.metric("Disclosed (Y)", f"{y_count:,}")
    with col3:
        st.metric("Not Disclosed (N)", f"{n_count:,}")
    
    # Disclosure rate metric
    st.metric("Disclosure Rate", f"{y_rate:.2f}%")
    
    # Pie chart (compact)
    import plotly.express as px
    value_counts = df['disclose_income'].value_counts()
    if len(value_counts) > 0:
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            color_discrete_map={'Y': '#2E8B57', 'N': '#DC143C'}
        )
        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=200
        )
        chart_key = f"di_pie_{result_key}" if result_key else f"di_pie_{title_suffix.replace(' ', '_')}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    
    # Add inline selection button if enabled
    if enable_selection and result_key:
        from app.pages.decision_execution import (
            save_disclose_income_config_from_results, 
            validate_seed_for_config_save
        )
        
        # Get current simulation parameters
        current_seed = st.session_state.get('seed', 42)
        current_n_agents = st.session_state.get('n_agents', 1000)
        
        # Validate seed consistency before showing the button
        is_valid, error_info = validate_seed_for_config_save(
            'disclose_income', current_seed, current_n_agents
        )
        
        if is_valid:
            button_key = f"select_di_{result_key}"
            if st.button(f"✅ Use This Config", key=button_key, help="Select this configuration for complete simulation"):
                success = save_disclose_income_config_from_results(result_key, df)
                if success:
                    st.success(f"✅ Configuration saved: {result_key}")
                    st.rerun()
        else:
            # Show seed mismatch error
            render_seed_mismatch_error('disclose_income', error_info)


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
    q25_val = raw_values.quantile(0.25)
    q75_val = raw_values.quantile(0.75)
    
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


def render_disclose_income_selection_button(result_key, result_df):
    """Render selection button for disclose income configuration"""
    
    # Import here to avoid circular imports
    from app.pages.decision_execution import (
        save_decision_config,
        is_decision_config_selected,
        get_current_disclose_income_params,
        calculate_disclose_income_metrics,
        extract_disclose_income_configuration_details
    )
    
    is_selected = is_decision_config_selected('disclose_income', result_key)
    
    st.markdown("---")
    
    # Create a compact selection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show key metrics for quick reference
        raw_col = 'disclose_income_raw'
        if raw_col in result_df.columns:
            mean_raw = result_df[raw_col].mean()
            y_rate = (result_df[raw_col] > 0).mean() * 100
            st.caption(f"📊 Quick Summary: {len(result_df):,} agents, mean DI={mean_raw:.4f}, Y rate={y_rate:.2f}%")
        else:
            y_count = (result_df['disclose_income'] == 'Y').sum()
            y_rate = (y_count / len(result_df)) * 100 if len(result_df) > 0 else 0
            st.caption(f"📊 Quick Summary: {len(result_df):,} agents, Y rate={y_rate:.2f}%")
    
    with col2:
        # Selection button
        if is_selected:
            st.success("✅ Selected")
        else:
            if st.button(
                "🎯 Use This Config",
                key=f"di_inline_select_{result_key}",
                type="primary",
                use_container_width=True,
                help="Select this disclose income configuration for combined simulations"
            ):
                # Get disclose income specific parameters
                params = get_current_disclose_income_params()
                metrics = calculate_disclose_income_metrics(result_df)
                config_details = extract_disclose_income_configuration_details(result_key)
                
                extra_data = {
                    'income_mode': config_details['income_mode']
                }
                
                # Use unified save function with seed validation
                success, config, error_info = save_decision_config(
                    'disclose_income', result_key, result_df, params, metrics, extra_data
                )
                
                if success:
                    st.success("Disclose Income configuration selected!")
                    st.rerun()
                else:
                    # Show seed mismatch error
                    render_seed_mismatch_error('disclose_income', error_info)


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


def render_disclose_documents_selection_button(result_key, result_df):
    """Render the 'Use This Config' selection button for a disclose documents cell."""
    from app.pages.decision_execution import (
        save_decision_config,
        is_decision_config_selected,
        get_current_disclose_documents_params,
        calculate_disclose_documents_metrics,
        extract_disclose_documents_configuration_details,
    )

    is_selected = is_decision_config_selected('disclose_documents', result_key)
    st.markdown("---")
    _, col2 = st.columns([2, 1])

    with col2:
        if is_selected:
            st.success("✅ Selected")
        else:
            if st.button("🎯 Use This Config", key=f"dd_inline_select_{result_key}", type="primary",
                         use_container_width=True,
                         help="Select this disclose documents configuration for combined simulations"):
                params = get_current_disclose_documents_params()
                metrics = calculate_disclose_documents_metrics(result_df)
                config_details = extract_disclose_documents_configuration_details(result_key)
                extra_data = {'income_mode': config_details['income_mode']}
                success, config, error_info = save_decision_config(
                    'disclose_documents', result_key, result_df, params, metrics, extra_data
                )
                if success:
                    st.success("Disclose Documents configuration selected!")
                    st.rerun()
                else:
                    render_seed_mismatch_error('disclose_documents', error_info)


def show_parameter_applicability_analysis(selected_decisions):
    """Show parameter applicability analysis for selected decisions"""
    st.markdown('<h3 class="section-header">📋 Parameter Applicability Analysis</h3>', unsafe_allow_html=True)
    
    # Show overall summary
    total_applicable = get_decision_global_parameters(selected_decisions)
    all_global_params = get_all_global_parameters()
    total_not_applicable = all_global_params - total_applicable
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Parameters", len(all_global_params))
    with col2:
        st.metric("✅ Applicable", len(total_applicable))
    with col3:
        st.metric("❌ Not Applicable", len(total_not_applicable))
    
    # Show parameter breakdown
    with st.expander("🔍 Parameter Breakdown", expanded=False):
        col_app, col_not_app = st.columns(2)
        
        with col_app:
            st.markdown("### ✅ Applicable Parameters")
            if total_applicable:
                for param in sorted(total_applicable):
                    st.markdown(f"  • {param.replace('_', ' ').title()}")
            else:
                st.markdown("None")
        
        with col_not_app:
            st.markdown("### ❌ Not Applicable Parameters")
            if total_not_applicable:
                for param in sorted(total_not_applicable):
                    st.markdown(f"  • {param.replace('_', ' ').title()}")
            else:
                st.markdown("None")
    
    # Show decision-specific analysis
    with st.expander("📊 Decision-Specific Parameter Analysis", expanded=False):
        try:
            decisions_path = Path(__file__).resolve().parents[1] / "config" / "decisions.yaml"
            with open(decisions_path, 'r') as f:
                decisions_config = yaml.safe_load(f)
            
            for decision in selected_decisions:
                decision_config = decisions_config.get(decision, {})
                decision_global_params = set(decision_config.get('uses_global_parameters', []))
                all_params = get_all_global_parameters()
                not_used_params = all_params - decision_global_params
                
                st.markdown(f"**{decision.replace('_', ' ').title()}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Uses Global Params", len(decision_global_params))
                with col2:
                    st.metric("Doesn't Use", len(not_used_params))
                with col3:
                    efficiency = len(decision_global_params) / len(all_params) * 100 if all_params else 0
                    st.metric("Usage %", f"{efficiency:.0f}%")
                
                if decision_global_params:
                    st.markdown("✅ **Uses Global Parameters:**")
                    formatted_params = [p.replace('_', ' ').title() for p in sorted(decision_global_params)]
                    st.markdown(f"  {', '.join(formatted_params)}")
                else:
                    st.markdown("✅ **Uses Global Parameters:** None (trait-based decision)")
                
                st.markdown("---")
                
        except Exception as e:
            st.error(f"Error loading decision configurations: {e}")


def show_monte_carlo_results(mc_data):
    """Display Monte Carlo simulation results"""
    if mc_data['summary'] is not None:
        st.subheader("📈 Monte-Carlo Analysis Results")
        
        summary_df = mc_data['summary']
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
    
        if 'donation_default' in summary_df['decision'].values:
            donation_row = summary_df[summary_df['decision'] == 'donation_default'].iloc[0]
        
            with col1:
                st.metric("Mean Donation Rate", f"{donation_row['mean']:.2%}")
            
            with col2:
                st.metric("Standard Deviation", f"{donation_row['std']:.2%}")
            
            with col3:
                st.metric("95% CI Lower", f"{donation_row['p2.5']:.2%}")
            
            with col4:
                st.metric("95% CI Upper", f"{donation_row['p97.5']:.2%}")
    
        # Monte-Carlo convergence plot
        if mc_data['detailed'] is not None:
            detailed_df = mc_data['detailed']
            
            if 'donation_default_mean' in detailed_df.columns:
                st.subheader("📊 Monte-Carlo Convergence")
                
                # Calculate running average
                detailed_df['running_mean'] = detailed_df['donation_default_mean'].expanding().mean()
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Test Run Results", "Running Average Convergence"),
                    vertical_spacing=0.1
                )
                
                # Test runs
                fig.add_trace(
                    go.Scatter(
                        x=detailed_df['run'] + 1,
                        y=detailed_df['donation_default_mean'],
                        mode='markers+lines',
                        name='Test Runs',
                        line=dict(color='lightblue', width=1),
                        marker=dict(size=4)
                    ),
                    row=1, col=1
                )
                
                # Running average
                fig.add_trace(
                    go.Scatter(
                        x=detailed_df['run'] + 1,
                        y=detailed_df['running_mean'],
                        mode='lines',
                        name='Running Average',
                        line=dict(color='red', width=2)
                    ),
                    row=2, col=1
                )
                
                # Add confidence interval
                if len(detailed_df) > 1:
                    final_mean = detailed_df['running_mean'].iloc[-1]
                    final_std = detailed_df['donation_default_mean'].std()
                    ci_upper = final_mean + 1.96 * final_std / np.sqrt(len(detailed_df))
                    ci_lower = final_mean - 1.96 * final_std / np.sqrt(len(detailed_df))
                    
                    fig.add_hline(y=ci_upper, line_dash="dash", line_color="gray", row=2, col=1)
                    fig.add_hline(y=ci_lower, line_dash="dash", line_color="gray", row=2, col=1)
                
                fig.update_layout(
                    height=600,
                    title="Monte-Carlo Study Results",
                    showlegend=True
                )
                fig.update_yaxes(tickformat='.2%')
                fig.update_xaxes(title="Run Number")
                
                st.plotly_chart(fig, width="stretch")
    
        # Summary statistics table
        st.subheader("📋 Summary Statistics")
    
        # Format the summary table for display
        display_summary = summary_df.copy()
        for col in ['mean', 'p2.5', 'p97.5']:
            if col in display_summary.columns:
                display_summary[col] = display_summary[col].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(display_summary, use_container_width=True)
    
        # Download Monte-Carlo results
        st.subheader("💾 Export Monte-Carlo Results")
        
        col1, col2, col3 = st.columns(3)
    
        with col1:
            if mc_data['summary'] is not None:
                summary_csv = mc_data['summary'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary",
                    data=summary_csv,
                    file_name=f"monte_carlo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if mc_data['detailed'] is not None:
                detailed_csv = mc_data['detailed'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Detailed",
                    data=detailed_csv,
                    file_name=f"monte_carlo_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🔄 Clear Monte-Carlo Results"):
                st.session_state.mc_results = None
                st.rerun()
    
        # Show log output
        if mc_data['log']:
            with st.expander("📋 Monte-Carlo Execution Log", expanded=False):
                st.text(mc_data['log'])


def show_income_distribution_histogram(sim_params, n_samples: int = 1000, seed: int = None):
    """Display income distribution histogram with discount threshold overlay"""
    try:
        # Use provided seed or default to 42
        preview_seed = seed if seed is not None else 42
        
        # Generate sample data
        income_samples = sim_params.sample_income_distribution(n_samples, seed=preview_seed)
        
        # Calculate percentiles and range for display
        p99_value = float(np.percentile(income_samples, 99))
        x_lower = float(max(0.0, np.min(income_samples)))
        x_upper = float(np.max(income_samples))
        
        # Always show full range - users want to see the complete distribution
        samples_for_display = income_samples
        display_range = [x_lower, x_upper]
        
        # Create histogram with all data
        fig = px.histogram(
            x=samples_for_display,
            nbins=50,
            title=f"Income Distribution Preview ({sim_params.income_distribution.title()})",
            labels={'x': 'Income ($)', 'count': 'Number of Agents'},
            marginal="box"
        )
        
        # Set explicit x-axis range to show full distribution
        fig.update_xaxes(range=display_range)
        
        # Add discount threshold line
        fig.add_vline(
            x=sim_params.discount_income_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Discount Threshold: ${sim_params.discount_income_threshold:,.0f}",
            annotation_position="top"
        )
        
        # Don't show any target line - distribution is based on user parameters
        
        # Update layout
        fig.update_layout(
            xaxis_tickformat='$,.0f',
            showlegend=False,
            height=400
        )
        
        # Calculate and display statistics
        actual_mean = np.mean(income_samples)
        actual_median = np.median(income_samples)
        discount_rate = sim_params.get_discount_qualification_rate(n_samples)
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Show distribution statistics
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Mean", f"${actual_mean:,.0f}")
        with col_stat2:
            st.metric("Median", f"${actual_median:,.0f}")
        with col_stat3:
            st.metric("Discount Qualification", f"{discount_rate:.1%}",
                     help="Percentage of agents with income ≤ threshold (potential eligibility based on income alone). Actual discount customers will be lower, as they must also choose to disclose income AND documents in their decisions.")
        with col_stat4:
            st.metric("Sample Size", f"{n_samples:,}")
        
        # Show min/max and percentiles
        col_range1, col_range2, col_range3, col_range4 = st.columns(4)
        actual_min = np.min(income_samples)
        actual_max = np.max(income_samples)
        p95_value = np.percentile(income_samples, 95)
        
        with col_range1:
            st.metric("Minimum", f"${actual_min:,.0f}")
        with col_range2:
            st.metric("95th Percentile", f"${p95_value:,.0f}")
        with col_range3:
            st.metric("99th Percentile", f"${p99_value:,.0f}")
        with col_range4:
            st.metric("Maximum", f"${actual_max:,.0f}")
        
        # Show info about tail if there are significant outliers beyond p99
        num_outliers = int(np.sum(income_samples > p99_value))
        if num_outliers > 10:  # Only mention if substantial tail exists
            pct_outliers = (num_outliers / n_samples) * 100
            st.caption(f"ℹ️ Note: {pct_outliers:.1f}% of samples are beyond the 99th percentile (${p99_value:,.0f}), extending to ${actual_max:,.0f}.")
        
        # Distribution is based on user-specified parameters - no target comparison needed
        
    except Exception as e:
        st.error(f"❌ Error generating income distribution histogram: {e}")
        st.caption("Please check your income distribution parameters.")


def get_css_styles():
    """Return the CSS styles for the application"""
    return """
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.page-header {
    font-size: 2rem;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 1.5rem;
}
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #34495e;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #3498db;
}
.subsection-header {
    font-size: 1.1rem;
    font-weight: 500;
    color: #2c3e50;
    margin-bottom: 0.8rem;
    margin-top: 1.5rem;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
}
.stAlert {
    margin-top: 1rem;
}
.navigation-buttons {
    margin-top: 2rem;
    margin-bottom: 2rem;
}
.param-group {
    margin-bottom: 2rem;
    padding: 1rem;
    border-left: 4px solid #3498db;
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border-radius: 0 0.5rem 0.5rem 0;
}

</style>
"""
