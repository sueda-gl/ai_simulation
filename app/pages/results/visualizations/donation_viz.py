# app/pages/results/visualizations/donation_viz.py
"""
Donation-related visualization functions.
Handles donation_default and final_donation_rate decisions.

The Excel building and the statistics tables live in `app/reports/donation.py`
(pure pandas/openpyxl, no Streamlit); this module reads the session values,
calls those builders and renders charts, metrics and download buttons.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from app.utils.timestamp_utils import TimestampConverter
from app.reports.donation import (
    build_donation_default_export,
    build_donation_default_xlsx,
    build_donation_transaction_export,
    build_donation_transactions_xlsx,
    donation_default_export_columns,
    donation_default_stats_frame,
    donation_transactions_frame,
)


def render_donation_default(df, decision_name, decision_title, decision_data):
    """Visualization for donation_default - placeholder until specialized view is added"""
    try:
        numeric_data = pd.to_numeric(decision_data, errors='coerce')
        if not numeric_data.isna().all():
            col1, col2, col3, col4 = st.columns([1, 1.1, 1.1, 1.2])
            with col1:
                st.metric("Total Agents", f"{len(decision_data):,}")
            with col2:
                st.metric("Mean", f"{numeric_data.mean():.2%}")
            with col3:
                st.metric("Std Dev", f"{numeric_data.std():.2%}")
            with col4:
                st.metric("Range", f"{numeric_data.min():.2%} - {numeric_data.max():.2%}")
            col_plot, col_stats = st.columns([2, 1])
            with col_plot:
                st.markdown(f"**Distribution of {decision_title}**")
                fig = px.histogram(
                    df,
                    x=decision_name,
                    nbins=30,
                    labels={decision_name: decision_title, 'count': 'Number of Agents'}
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_tickformat='.0%'
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_stats:
                st.markdown("**📈 Statistics**")
                stats_df = donation_default_stats_frame(numeric_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            # Add Excel export for donation_default when using custom parameters
            # Check if this is a custom parameters run (not default values)
            from app.pages.results.run_context import RunContext
            is_custom_parameters = 'donation_default' in RunContext.from_session().custom_decisions

            if is_custom_parameters:
                st.markdown("---")
                st.markdown("**💾 Export Donation Results**")

                # Columns to export: Agent ID, the trait columns that exist, the decision
                export_columns = donation_default_export_columns(df, decision_name)

                if export_columns:
                    export_df = build_donation_default_export(df, export_columns)

                    # Create Excel file
                    try:
                        excel_bytes = build_donation_default_xlsx(export_df)

                        col_download, col_info = st.columns([1, 2])

                        with col_download:
                            st.download_button(
                                label="📊 Download Donation Excel",
                                data=excel_bytes,
                                file_name=f"donation_default_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Download donation results with agent traits and donation rates"
                            )

                        with col_info:
                            st.markdown(f"📋 Export includes {len(export_df):,} agents with {len(export_columns)} columns")
                            st.markdown(f"✅ Fields: Agent ID, traits, and {decision_name}")

                    except ImportError:
                        st.warning("⚠️ Excel export requires openpyxl package")
                    except Exception as e:
                        st.error(f"❌ Error creating Excel file: {str(e)}")
                else:
                    st.warning("⚠️ No valid columns found for export")
        else:
            st.info("Data not numeric; specialized visualization not available yet.")
    except Exception:
        st.info("Unable to render donation_default with placeholder visualization.")


def render_final_donation_rate(df, decision_name, decision_title, decision_data):
    """Visualization for final_donation_rate with 3-case logic for donation configs"""

    from app.pages.decision_execution import get_decision_config
    from app.pages.results.run_context import RunContext
    ctx = RunContext.from_session()

    # CASE 3: Check if a donation configuration has been selected - an explicitly
    # pinned record, or (R13) donation_default run as a custom decision of THIS run,
    # whose donation_default column IS the selected distribution
    _donation_config = get_decision_config('donation_default')
    donation_was_custom = 'donation_default' in ctx.custom_decisions
    has_selected_config = _donation_config is not None or donation_was_custom

    # CASE 1: Check if exactly one donation config exists (auto-use it)
    is_single_donation_run = ctx.is_individual_run('donation_default')

    # If this is a single donation run with exactly one result, treat it as "only config available"
    has_only_one_config = False
    if is_single_donation_run:
        results_dict = ctx.results
        if results_dict and len(results_dict) == 1:
            has_only_one_config = True
            only_config_key = list(results_dict.keys())[0]
            only_config_df = results_dict[only_config_key]

    # Decision logic: Use distribution if selected config OR only one config available
    use_distribution = (has_selected_config or has_only_one_config) and 'donation_default' in df.columns

    if use_distribution:
        # Show the actual donation distribution - distinguish between cases
        if has_selected_config:
            st.success("📊 **Using Distribution from Selected Donation Configuration**")
            st.markdown("✅ The final_donation_rate values in your export match the donation_default distribution shown below")
        elif has_only_one_config:
            st.success("📊 **Using Distribution from Only Available Donation Configuration**")
            st.markdown("✅ Only one donation configuration was generated - final_donation_rate values match donation_default")

        donation_data = df['donation_default']

        # Top section: Distribution statistics
        col1, col2, col3, col4 = st.columns([1, 1.1, 1.1, 1])

        with col1:
            st.metric("Total Agents", f"{len(donation_data):,}")

        with col2:
            st.metric("Mean Rate", f"{donation_data.mean():.2%}")

        with col3:
            st.metric("Median Rate", f"{donation_data.median():.2%}")

        with col4:
            st.metric("Std Dev", f"{donation_data.std():.2%}")

        # Distribution visualization
        st.markdown("---")
        st.markdown("**📊 Donation Rate Distribution:**")

        col_hist, col_stats = st.columns([2, 1])

        with col_hist:
            # Histogram showing the distribution - match overview chart settings for consistency
            st.markdown("**Distribution of Donation Rates Across Agents**")
            fig = px.histogram(
                df,
                x='donation_default',
                labels={'donation_default': 'Donation Rate', 'count': 'Number of Agents'},
                nbins=30,  # Match overview chart
                marginal="box"  # Match overview chart
            )
            fig.update_layout(
                xaxis_tickformat='.0%',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            st.markdown("**📈 Distribution Stats:**")
            st.write(f"• **Min**: {donation_data.min():.2%}")
            st.write(f"• **25th %ile**: {donation_data.quantile(0.25):.2%}")
            st.write(f"• **50th %ile**: {donation_data.quantile(0.50):.2%}")
            st.write(f"• **75th %ile**: {donation_data.quantile(0.75):.2%}")
            st.write(f"• **Max**: {donation_data.max():.2%}")
            st.write(f"• **Range**: {donation_data.max() - donation_data.min():.2%}")

            st.markdown("---")
            st.markdown("**ℹ️ Source:**")
            if _donation_config:
                st.markdown(f"Population: {_donation_config.get('population_mode', 'Unknown')}")
                st.markdown(f"Income: {_donation_config.get('donation_income_mode', _donation_config.get('income_spec_mode', 'Unknown'))}")
            elif donation_was_custom:
                # R13: the modes this run computed donation_default with
                st.markdown(f"Population: {ctx.effective_population_mode}")
                st.markdown(f"Income: {ctx.effective_income_mode}")

    else:
        # Fall back to slider if no donation_default data available
        st.info("💡 **No donation configuration selected** - Using simple rate configuration")
        st.markdown("Select a donation configuration on Page 2 to see the full distribution")

        # Use _default_value key (consistent with Page 2 for numeric defaults;
        # read-only here - the key is initialised at app start by app.models)
        slider_key = f"{decision_name}_default_value"

        # Top section: Current settings
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Agents", f"{len(decision_data):,}")

        with col2:
            # Calculate average of actual final donation rates
            avg_final_rate = pd.to_numeric(decision_data, errors='coerce').mean()
            if pd.isna(avg_final_rate):
                avg_final_rate = st.session_state.get(slider_key, 0.10)  # 10% as default
            st.metric("Final Donation Rate", f"{avg_final_rate:.2%}")

        with col3:
            # Default donation rate with 2 decimal points
            default_rate = 0.10
            st.metric("Default Donation Rate", f"{default_rate:.2%}")

    # ========================================================================
    # NEW: Transaction-Level Export (ALWAYS AVAILABLE if purchase_requests exist)
    # ========================================================================
    # This section appears REGARDLESS of whether donation_default was selected
    # It will use request-level rates if available, or agent-level fallback

    if 'purchase_requests' in df.columns:
        st.markdown("---")
        st.markdown("**💾 Transaction-Level Export**")
        st.markdown("Download detailed purchase request data with donation rates (one row per request)")

        try:
            # Session values the pure builder needs. The pricing constants it uses
            # (market_price / platform_markup / price_range) stay its own defaults:
            # this export never read the Page-1 parameters.
            vendors_data = None
            if hasattr(st.session_state, 'vendors'):
                vendors_data = st.session_state.vendors

            # Use centralized timestamp converter for consistent handling
            ts_converter = TimestampConverter()

            # Build transaction records
            transaction_records = build_donation_transaction_export(
                df, vendors=vendors_data, ts_converter=ts_converter
            )

            if len(transaction_records) > 0:
                # Create DataFrame from records, sorted by Period and Agent ID
                transactions_df = donation_transactions_frame(transaction_records)

                # Show summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Requests", f"{len(transactions_df):,}")
                with col2:
                    num_agents = transactions_df['Agent ID'].nunique()
                    st.metric("Total Agents", f"{num_agents:,}")
                with col3:
                    num_periods = transactions_df['Period'].nunique()
                    st.metric("Periods", f"{int(num_periods)}")
                with col4:
                    avg_donation = transactions_df['Final Donation Rate'].mean()
                    if not pd.isna(avg_donation):
                        st.metric("Avg Donation Rate", f"{avg_donation:.2%}")
                    else:
                        st.metric("Avg Donation Rate", "N/A")

                # Create Excel with multiple sheets (Total + one per Period)
                transactions_bytes = build_donation_transactions_xlsx(transactions_df)

                # Download button
                transaction_filename = f"donation_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

                st.download_button(
                    label="📥 Download Transaction-Level Excel",
                    data=transactions_bytes,
                    file_name=transaction_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Downloads purchase request-level data with donation rates (Total sheet + one sheet per period)"
                )

                # Show preview of data
                with st.expander("📊 Preview Transaction Data"):
                    st.dataframe(transactions_df.head(20), use_container_width=True)
            else:
                st.info("ℹ️ No purchase request data found in this simulation")

        except Exception as e:
            st.error(f"⚠️ Error creating transaction export: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
