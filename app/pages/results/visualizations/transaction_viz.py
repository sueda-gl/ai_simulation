# app/pages/results/visualizations/transaction_viz.py
"""
Transaction and purchase-related visualization functions.
Handles purchase_vs_bid, rejected_transaction_defaults, and rejected_transaction_option decisions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from app.reports.purchase import (
    build_purchase_vs_bid_export,
    prepare_priority_lists_export,
    priority_first_choice_counts,
    priority_length_breakdown_lines,
    priority_list_agent_count,
    priority_lists_xlsx_bytes,
    priority_option_agent_counts,
    purchase_vs_bid_breakdown_frame,
    purchase_vs_bid_request_counts,
    purchase_vs_bid_xlsx_bytes,
    rejected_option_value_counts,
)
from app.reports.rtd import (
    RTD_ELEMENT_FILE_SLUGS,
    RTD_ELEMENT_SHEETS,
    prepare_rtd_element_export,
    prepare_rtd_model_export,
    rtd_element_xlsx_bytes,
    rtd_model_xlsx_bytes,
    rtd_score_stats_caption,
)
from app.utils.timestamp_utils import TimestampConverter

# The pure builders now live in app/reports/{purchase,rtd}.py. This one alias is
# kept because tests/test_rtd_batch4_ui.py imports the sheet-name map from this
# module and `_rtd_active_element` below reads it.
_RTD_ELEMENT_SHEETS = RTD_ELEMENT_SHEETS


def render_purchase_vs_bid(df, decision_name, decision_title, decision_data):
    """Visualization for purchase_vs_bid - per-request Purchase Now/bid choices"""
    
    # Show that decisions are now made PER REQUEST
    st.info("⚠️ **Note**: Decisions are made **per purchase request**, not per agent. A single agent can choose differently for each purchase.")
    
    # Reference to Decision 2 for customer type definitions
    st.info("💡 **Customer Type Information**: This decision only applies to **Regular Customers**. For detailed customer type definitions and distribution, see **Decision 2: Disclose Documents**.")
    
    # Extract REQUEST-LEVEL data from purchase_requests
    st.markdown("---")
    st.markdown("### 🎯 Purchase Now vs Bid Decisions")
    st.caption("📊 Decisions for **Regular Customers only** - For full customer type breakdown, see **Decision 2: Disclose Documents**")
    
    if 'purchase_requests' in df.columns:
        # Count regular customer choices (PN / BID, per request)
        regular_counts, total_regular_requests = purchase_vs_bid_request_counts(df)
        
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
                breakdown_df = purchase_vs_bid_breakdown_frame(
                    pn_count, bid_count, total_regular_requests)
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        else:
            st.info("No regular customer purchase requests found")
        
        # Excel Export Section
        st.markdown("---")
        st.markdown("**📥 Export Purchase Now vs Bid Decision Data**")
        st.caption("Download detailed request-level data for all regular customers with pricing and transaction information")
        
        # Build transaction records (the pricing parameters and the vendor list
        # the builder needs come off the session; the defaults match the
        # fallbacks the builder used when it read them itself)
        sim_params = getattr(st.session_state, 'sim_params', None)
        transaction_records = build_purchase_vs_bid_export(
            df,
            # TimestampConverter takes its base time, period duration and period
            # count off session state, so the page builds it
            ts_converter=TimestampConverter(),
            market_price=getattr(sim_params, 'market_price', 100.0),
            platform_markup=getattr(sim_params, 'platform_markup', 0.1),
            price_range=getattr(sim_params, 'price_range', 0.25),
            vendors=getattr(st.session_state, 'vendors', None),
        )
        
        if transaction_records and len(transaction_records) > 0:
            try:
                from datetime import datetime

                # Create DataFrame (already sorted by the build function)
                export_df = pd.DataFrame(transaction_records)

                # Create Excel with multiple sheets
                xlsx_bytes = purchase_vs_bid_xlsx_bytes(export_df)
                
                col_download, col_info = st.columns([1, 2])
                
                with col_download:
                    st.download_button(
                        label="📊 Download Purchase Now vs Bid Excel",
                        data=xlsx_bytes,
                        file_name=f"purchase_now_vs_bid_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download request-level data for regular customers with purchase decisions"
                    )
                
                with col_info:
                    num_sheets = 1 + len(export_df['Period'].dropna().unique()) if 'Period' in export_df.columns else 1
                    st.caption(f"📋 Export includes {len(export_df):,} requests across {num_sheets} sheets")
                    st.caption(f"✅ Fields: Purchase Request ID, Agent ID, Honesty_Humility, Assigned Allowance Level, Study Program, Group_experiment, TWT+Sospeso, income, Customer Type, Income Category, Purchase Type, Vendor, Vendor Price, timestamp, Period, Customer Price")
                    st.caption(f"🔄 Sorted by: timestamp (chronological order)")
            
            except ImportError:
                st.warning("⚠️ Excel export requires openpyxl package")
            except Exception as e:
                st.error(f"❌ Error creating Excel file: {str(e)}")
        else:
            st.info("ℹ️ No purchase request data available for export")
    else:
        st.warning("No purchase_requests data available")


def render_rejected_transaction_defaults(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_transaction_defaults - prioritized options per agent.

    Two display modes:
    - MODEL run (Decision 4 selected): the four trait-based sub-decision mechanisms
      (TTP list length + Loyalty/WTP/Risk-Taking rankings) -> _render_rtd_model_results.
    - DEFAULT run (unselected): the legacy priority-template view below.
    """
    if 'rtd_choice_length' in df.columns:
        _render_rtd_model_results(df, decision_name)
        return

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
        list_count = priority_list_agent_count(decision_data)
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
                st.markdown(f"**Priority {i}.** {option_label}")
                st.caption(f"   {option_desc}")
        else:
            st.caption("No priority template configured")
        
        st.caption(f"💡 {len(configured_template)} options configured")
        st.caption("All agents use this priority list")
    
    with col_chart:
        total_agents = len(decision_data)

        # Count how many agents have each option in their priority list
        option_agent_counts = priority_option_agent_counts(decision_data)
        
        # Create individual charts for each option
        if len(option_agent_counts) > 0:
            # Sort by the order in configured_template if possible
            if isinstance(configured_template, list):
                sorted_options = [opt for opt in configured_template if opt in option_agent_counts]
            else:
                sorted_options = list(option_agent_counts.keys())
            
            st.markdown("**Options in Priority Lists**")
            st.caption("Each chart shows the percentage of agents that have this option in their priority list")
            
            # Create a row of small donut charts - one for each option
            if len(sorted_options) <= 3:
                chart_cols = st.columns(len(sorted_options))
            else:
                chart_cols = st.columns(3)
            
            # Color palette for consistency
            colors = px.colors.qualitative.Set3
            
            for idx, opt in enumerate(sorted_options):
                col_idx = idx % len(chart_cols)
                with chart_cols[col_idx]:
                    agent_count = option_agent_counts[opt]
                    percentage = (agent_count / total_agents) * 100
                    option_label = option_numbers.get(opt, opt)
                    
                    # Create individual donut chart for this option
                    fig = px.pie(
                        values=[percentage, 100 - percentage],
                        names=[option_label, ""],
                        hole=0.6,
                        color_discrete_sequence=[colors[idx % len(colors)], "#f0f0f0"]
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent',
                        hovertemplate=f'<b>{option_label}</b><br>{agent_count:,} agents<br>%{{percent}}<extra></extra>',
                        showlegend=False
                    )
                    # Add center text showing percentage
                    fig.update_layout(
                        showlegend=False,
                        height=200,
                        margin=dict(t=30, b=10, l=10, r=10),
                        annotations=[dict(
                            text=f'{percentage:.0f}%',
                            x=0.5, y=0.5,
                            font=dict(size=20, weight='bold'),
                            showarrow=False
                        )]
                    )
                    # Hide the empty slice from tooltip
                    fig.data[0].hoverinfo = 'skip'
                    
                    # Display title as markdown
                    st.markdown(f"**{option_label}**")
                    st.plotly_chart(fig, use_container_width=True, key=f"{decision_name}_option_{idx}_chart")
                    st.caption(f"{agent_count:,} agents")
            
            # Show detailed breakdown
            with st.expander("📋 Detailed Breakdown"):
                for opt in sorted_options:
                    count = option_agent_counts[opt]
                    percentage = (count / total_agents) * 100
                    st.caption(f"{option_numbers.get(opt, opt)}: {count:,} agents ({percentage:.0f}%)")
    
    # Summary statistics
    st.markdown("---")
    st.markdown("**📈 Summary Statistics**")
    
    col_summary1, col_summary2 = st.columns(2)
    
    with col_summary1:
        st.markdown("**Priority List Lengths:**")
        breakdown_lines = priority_length_breakdown_lines(decision_data)
        st.caption("  \n".join(breakdown_lines))
    
    with col_summary2:
        st.markdown("**Most Common 1st Choice:**")
        first_choice_counts = priority_first_choice_counts(decision_data)
        
        for i, (choice, count) in enumerate(first_choice_counts.head(3).items(), 1):
            percentage = (count / len(decision_data)) * 100
            st.caption(f"{i}. {option_numbers.get(choice, choice)}: {count:,} agents ({percentage:.1f}%)")
    
    # Download section
    st.markdown("---")
    st.markdown("**📥 Download Priority Lists**")
    
    # Prepare export data
    export_df = prepare_priority_lists_export(df, decision_data)
    
    if export_df is not None and not export_df.empty:
        # Create Excel file
        from datetime import datetime

        xlsx_bytes = priority_lists_xlsx_bytes(export_df)

        st.download_button(
            label="📊 Download Priority Lists Excel",
            data=xlsx_bytes,
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


_RTD_OPTION_SHORT = {
    1: "Opt 1: higher price, same vendor",
    2: "Opt 2: other vendor, lower PN",
    3: "Opt 3: current vendor at PN",
    4: "Opt 4: place a bid",
    5: "Opt 5: forgo transaction",
}

# Priority sequences (most-likely option first) taken from the model at runtime -
# the same constants the decision function maps segments with (mirrored in
# config/decisions.yaml priority_sequences) - so chart orderings derived from
# them can never drift from the run's actual rankings.
try:
    from src.decisions.rejected_transaction_defaults import (
        PRIORITY_SEQUENCES as _RTD_PRIORITY_SEQUENCES)
except Exception:   # pragma: no cover - defensive fallback, values identical
    _RTD_PRIORITY_SEQUENCES = {'loyalty': [3, 1, 4, 5, 2], 'wtp': [3, 2, 1, 4, 5],
                               'risk_taking': [4, 2, 1, 3, 5]}

_RTD_MECHS = [
    ('loyalty', 'loyalty', 'Loyalty', _RTD_PRIORITY_SEQUENCES['loyalty']),
    ('wtp', 'wtp', 'Willingness-to-Pay', _RTD_PRIORITY_SEQUENCES['wtp']),
    ('risk_taking', 'rt', 'Risk-Taking', _RTD_PRIORITY_SEQUENCES['risk_taking']),
]


def _rtd_active_element():
    """The element selected via a per-element Run button on the Decision 4 tab
    ('ttp' | 'loyalty' | 'wtp' | 'risk_taking'), or None when the whole decision
    was run. Only individual Decision 4 runs are filtered - combined/complete
    simulations always show all four elements."""
    from app.pages.results.run_context import RunContext
    if RunContext.from_session().is_individual_run('rejected_transaction_defaults'):
        element = st.session_state.get('rtd_run_element')
        if element in _RTD_ELEMENT_SHEETS:
            return element
    return None

# Compact options-numbering key shown under each allocation chart (plain caption
# lines, no bullets; the trailing two spaces force markdown line breaks).
_RTD_OPTION_NUMBERING = (
    "Option 1: higher price category, same vendor · Option 2: other vendor, lower PN price  \n"
    "Option 3: current vendor at PN price · Option 4: place a bid  \n"
    "Option 5: forgo the transaction"
)


def _rtd_density_hist(series, title, x_title, chart_key):
    """Histogram of a continuous score, normalised so the bar heights sum to 1
    (each bar is the proportion of observations falling in that bin). The bin
    count is HALF of Stata's default rule - k = min(sqrt(N), 10*log10(N))
    equal-width bins spanning min..max - per professor feedback (2026-08: "cut
    the number of bins by half so will be closer to the Stata graphs"). The
    series mean is marked with a vertical red line."""
    import plotly.graph_objects as go
    s = pd.Series(series).dropna().astype(float)
    n = len(s)
    vmin, vmax = float(s.min()), float(s.max())
    k_stata = max(1, int(min(np.sqrt(n), 10 * np.log10(n)))) if n > 1 else 1
    k = max(1, k_stata // 2)
    size = (vmax - vmin) / k if vmax > vmin else 1.0
    fig = go.Figure(go.Histogram(
        x=s, histnorm='probability',
        xbins=dict(start=vmin, end=vmax + size * 1e-9, size=size),
        marker_color='steelblue'))
    fig.add_vline(x=float(s.mean()), line_color='red', line_width=2)
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title='Proportion', height=320,
                      margin=dict(t=40, b=10), showlegend=False, bargap=0.05)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def _rtd_fraction_bar(x_labels, fractions, title, x_title, chart_key):
    """Discrete allocation chart: fraction of agents per category, in the given order."""
    fig = px.bar(x=x_labels, y=fractions, title=title)
    fig.update_traces(marker_color='steelblue')
    fig.update_layout(xaxis_title=x_title, yaxis_title='% of agents', height=320,
                      margin=dict(t=40, b=10), yaxis_tickformat='.0%',
                      xaxis=dict(type='category', categoryorder='array',
                                 categoryarray=list(x_labels)))
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def _rtd_score_stats_caption(series):
    """Summary line matching Stata's `summarize` output for the score variable."""
    st.caption(rtd_score_stats_caption(series))


def render_rtd_comparison_results(results_dict, decision_name):
    """Comparison-mode rendering for Decision 4, grouped by income treatment.

    Each income-treatment group renders as: group title ("Categorical Income
    Treatment" / "Continuous Income Treatment"; omitted when only one group
    exists), then a row of per-population-mode overview cells (Simulation
    Overview + the D4 headline metrics, mirroring the donation-era comparison
    grids), then the detailed per-element sections for the SAME population-mode
    columns underneath - i.e. title -> summary -> details per income treatment,
    instead of all summaries first and all details after.

    Returns True if anything was rendered.
    """
    from app.components import show_overview
    from app.pages.decision_execution import format_result_name

    keys = [k for k, df_ in results_dict.items()
            if hasattr(df_, 'columns') and 'rtd_choice_length' in df_.columns]
    if not keys:
        return False

    # Group by income treatment, categorical first (matches the donation-era
    # grids' section order). Covers both the Compare-all key style
    # (copula_categorical, ...) and the plain single-population Compare-both
    # keys (categorical / continuous).
    cat_keys = [k for k in keys if k.endswith('categorical')]
    cont_keys = [k for k in keys if k.endswith('continuous')]
    other_keys = [k for k in keys if k not in cat_keys and k not in cont_keys]
    groups = [(title, group_keys) for title, group_keys in (
        ("Categorical Income Treatment", cat_keys),
        ("Continuous Income Treatment", cont_keys),
        (None, other_keys)) if group_keys]

    mode_labels = {
        'copula': '🧬 Copula (Synthetic)',
        'research_spec': '📄 Research Specification',
        'research_baseline': '⚖️ Research Baseline',
    }
    mode_short = {
        'copula': 'Copula',
        'research_spec': 'Research Spec',
        'research_baseline': 'Research Baseline',
    }

    def label_for(key):
        for prefix, label in mode_labels.items():
            if key.startswith(prefix):
                return label
        return format_result_name(key)

    def suffix_for(key):
        """show_overview title suffix, mirroring the donation-era comparison
        grids (' (Copula, Cat)', ' (Categorical)', ...)."""
        for prefix, short in mode_short.items():
            if key.startswith(prefix):
                income = 'Cat' if key.endswith('categorical') else 'Cont'
                return f" ({short}, {income})"
        return f" ({key.replace('_', ' ').title()})"

    show_group_titles = len(groups) > 1
    for group_idx, (group_title, group_keys) in enumerate(groups):
        if group_idx:
            st.markdown("---")
        if show_group_titles and group_title:
            st.markdown(f"#### {group_title}")

        group_labels = [label_for(k) for k in group_keys]
        if len(set(group_labels)) != len(group_labels):
            # e.g. the same population mode appearing twice within a group
            group_labels = [format_result_name(k) for k in group_keys]

        # Rows of up to 3 population-mode columns: the overview cells (summary)
        # first, then the detailed per-element sections for the same keys.
        for start in range(0, len(group_keys), 3):
            row_keys = group_keys[start:start + 3]
            row_labels = group_labels[start:start + 3]
            if start:
                st.markdown("---")

            overview_cols = st.columns(len(row_keys))
            for col, key, label in zip(overview_cols, row_keys, row_labels):
                with col:
                    st.markdown(f"**{label}**")
                    show_overview(results_dict[key], suffix_for(key),
                                  result_key=key, enable_selection=False)

            detail_cols = st.columns(len(row_keys))
            for col, key, label in zip(detail_cols, row_keys, row_labels):
                with col:
                    st.markdown(f"**{label}**")
                    _render_rtd_model_results(results_dict[key], decision_name,
                                              chart_suffix=f"_{key}", compact=True)
    return True


def _render_rtd_model_results(df, decision_name, chart_suffix='', compact=False):
    """Model-run results for Decision 4: four sub-decision mechanisms per agent.

    When a per-element Run button was used (st.session_state.rtd_run_element set on
    an individual Decision 4 run), ONLY that element's section is rendered; a
    whole-decision run renders all four.

    chart_suffix disambiguates Streamlit element keys when this view is rendered
    once per result_key (comparison modes). compact=True stacks each section
    vertically for use inside a per-mode comparison column.
    """
    n = len(df)
    active = _rtd_active_element()

    def _element_section(chart_a, chart_b):
        """Chart A (score distribution) and Chart B (allocation) side by side, or
        stacked in compact comparison columns."""
        if compact:
            chart_a()
            chart_b()
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                chart_a()
            with col_b:
                chart_b()

    def _element_download(mech, label):
        """Per-element Excel: Agent ID, the element's independent variables, its
        score and the resulting option sequence per customer."""
        export_df = _prepare_rtd_element_export(df, mech)
        if export_df is None or export_df.empty:
            return
        from datetime import datetime
        xlsx_bytes = rtd_element_xlsx_bytes(export_df, mech)
        # Human-readable filename slugs ('ttp' reads too much like 'wtp')
        fname_slug = RTD_ELEMENT_FILE_SLUGS[mech]
        st.download_button(
            label=f"📊 Download {label} Excel",
            data=xlsx_bytes,
            file_name=f"rejected_transaction_{fname_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help=f"Per-agent {label} results: independent variables, score and "
                 "resulting option sequence per customer",
            key=f"rtd_dl_{mech}{chart_suffix}",
        )

    # ---- Element 1: Options List Length (Tendency to Plan) ----
    if active in (None, 'ttp'):
        st.markdown("---")
        st.markdown("**1️⃣ Options List Length (Tendency to Plan)**")
        st.markdown("presents how many default options each customer pre-selects (0-5)")
        _rtd_score_stats_caption(df['rtd_weighted_ttp'])

        def _ttp_score_chart():
            _rtd_density_hist(df['rtd_weighted_ttp'], "Tendency to Plan score",
                              "Tendency to Plan score",
                              f"{decision_name}_rtd_ttp_score{chart_suffix}")

        def _ttp_alloc_chart():
            counts = df['rtd_choice_length'].astype(int).value_counts()
            # Bars ALWAYS in natural 0..5 order (professor 2026-08: "always start
            # with 0 and end with 5, presenting bars in order rather than from
            # low to high") - no sort-by-frequency for this chart.
            lengths = list(range(0, 6))
            fractions = [counts.get(l, 0) / n for l in lengths]
            _rtd_fraction_bar([str(l) for l in lengths], fractions,
                              "% of pre-selected options",
                              "Number of pre-selected options",
                              f"{decision_name}_rtd_length_chart{chart_suffix}")
            # Companion table: number of pre-selected options (0-5) -> % of agents
            st.dataframe(pd.DataFrame({
                'Number of pre-selected options': lengths,
                '% of agents': [f"{counts.get(l, 0) / n * 100:.1f}%" for l in lengths],
            }), hide_index=True, use_container_width=True)
            st.caption(_RTD_OPTION_NUMBERING)

        _element_section(_ttp_score_chart, _ttp_alloc_chart)
        _element_download('ttp', "Options List Length")

    # ---- Elements 2-4: priority rankings ----
    # All three score charts plot the STANDARDIZED score (professor 2026-08:
    # "present the standardized loyalty graph rather than the one before
    # standardization"): rtd_loyalty_z / rtd_wtp_z / rtd_rt_z, matching the doc's
    # `histogram weighted_loyalty` after `egen weighted_loyalty = std(...)`.
    score_specs = {
        'loyalty': ('rtd_loyalty_z', "Loyalty score"),
        'wtp': ('rtd_wtp_z', "Willingness-to-Pay score"),
        'risk_taking': ('rtd_rt_z', "Risk-Taking score"),
    }
    for idx, (mech, col_key, label, seq) in enumerate(_RTD_MECHS, start=2):
        seg_col = f'rtd_{col_key}_segment'
        if seg_col not in df.columns or (active is not None and active != mech):
            continue
        st.markdown("---")
        st.markdown(f"**{idx}️⃣ {label} Ranking**")
        st.markdown("Priority sequence " + " > ".join(f"Option {o}" for o in seq))

        score_col, score_title = score_specs[mech]
        _rtd_score_stats_caption(df[score_col])

        def _score_chart(sc=score_col, ti=score_title, ck=col_key):
            _rtd_density_hist(df[sc], ti, ti,
                              f"{decision_name}_rtd_{ck}_score{chart_suffix}")

        def _alloc_chart(sq=seq, sgc=seg_col, ck=col_key, lb=label,
                         per_element=(active == mech)):
            # Mirrored mapping: segment s -> first choice sq[5 - s] (highest segment
            # gets the top option of the priority sequence).
            first_choice = df[sgc].astype(int).map(lambda s: sq[5 - s])
            counts = first_choice.value_counts()
            if per_element:
                # Per-element run (professor 2026-08): categories in the element's
                # priority sequence REVERSED - least likely option on the left,
                # most likely on the right (derived from the runtime sequence).
                order = list(reversed(sq))
            else:
                # Whole-Decision-4 run keeps the least-popular -> most-popular
                # presentation (ascending observed share, ties by option number),
                # as the professor asked to retain for the integrated view.
                order = sorted(sq, key=lambda o: (counts.get(o, 0), o))
            fractions = [counts.get(o, 0) / n for o in order]
            _rtd_fraction_bar([f"Option {o}" for o in order], fractions,
                              f"% of {lb.lower()}-based options ranking",
                              "Selected option",
                              f"{decision_name}_rtd_{ck}_seg_chart{chart_suffix}")
            # Companion table: options 1-5 (natural order) -> first-ranked % of agents
            st.dataframe(pd.DataFrame({
                'Option': [f"Option {o}" for o in range(1, 6)],
                '% of agents': [f"{counts.get(o, 0) / n * 100:.1f}%" for o in range(1, 6)],
            }), hide_index=True, use_container_width=True)
            st.caption(_RTD_OPTION_NUMBERING)

        _element_section(_score_chart, _alloc_chart)
        _element_download(mech, f"{label} Ranking")

    # ---- Whole-decision per-agent Excel export (only for whole-decision runs;
    # per-element runs already have their element's download in the section above) ----
    if active is None:
        st.markdown("---")
        st.markdown("**📥 Download Decision 4 Model Results**")
        sheets = _prepare_rtd_model_export(df)
        if sheets:
            from datetime import datetime
            xlsx_bytes = rtd_model_xlsx_bytes(sheets)
            st.download_button(
                label="📊 Download Decision 4 Excel (all elements)",
                data=xlsx_bytes,
                file_name=f"rejected_transaction_mechanisms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="One self-contained sheet per element: independent variables, "
                     "scores, intermediate distributions and the resulting option "
                     "sequence per customer",
                key=f"rtd_model_download{chart_suffix}",
            )
            with st.expander("📋 Preview Export Data (first 10 rows per sheet)"):
                for sheet_name, sheet_df in sheets.items():
                    st.markdown(f"**{sheet_name} Sheet:**")
                    st.dataframe(sheet_df.head(10).astype(str), use_container_width=True)
                    st.caption(f"Rows: {len(sheet_df):,} · Columns: {', '.join(sheet_df.columns)}")


def _prepare_rtd_element_export(df, mech):
    """Page wrapper over `app.reports.rtd.prepare_rtd_element_export`, keeping the
    inline error the section showed when a frame could not be exported."""
    try:
        return prepare_rtd_element_export(df, mech)
    except Exception as e:
        st.error(f"Error preparing Decision 4 {RTD_ELEMENT_SHEETS.get(mech, mech)} export: {e}")
        return None


def _prepare_rtd_model_export(df):
    """Page wrapper over `app.reports.rtd.prepare_rtd_model_export`, keeping the
    inline error the download section showed."""
    try:
        return prepare_rtd_model_export(df)
    except Exception as e:
        st.error(f"Error preparing Decision 4 export: {e}")
        return None


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
    value_counts, current_option = rejected_option_value_counts(decision_data)
    
    # Use _default_selection key (same as Page 2 Overview tab) for consistency
    # (read-only here: the key is initialised at app start by app.models)
    radio_key = f"{decision_name}_default_selection"

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
            
            st.markdown("### Current Simulation Results")
            fig = px.pie(
                values=value_counts.values,
                names=readable_labels,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True, key="rejected_transaction_option_chart")
        else:
            st.info("No simulation data available")


def render_rejected_bid_value(df, decision_name, decision_title, decision_data):
    """Visualization for rejected_bid_value"""
    st.markdown("**Not relevant given choice of Option 5**")

