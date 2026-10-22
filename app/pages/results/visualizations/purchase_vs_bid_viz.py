# app/pages/results/visualizations/purchase_vs_bid_viz.py
"""
Decision 9 (purchase_vs_bid) visualization: the per-request Purchase Now / Bid
choices of the regular customers and their request-level Excel.

The request statistics and the Excel builder live in `app.reports.transactions`
(pure, Streamlit-free); this module renders the metrics, the donut chart and the
download button.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

from app.reports.transactions import (
    build_purchase_vs_bid_export,
    purchase_vs_bid_breakdown_frame,
    purchase_vs_bid_request_counts,
    purchase_vs_bid_xlsx_bytes,
)
from app.reports.timestamps import TimestampConverter
from app.pages.results.run_context import RunContext


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
        
        # Build transaction records: the pricing parameters are the RUN's
        # (RunContext, not the live Page 1 widgets); the vendor list is the run's
        ctx = RunContext.from_session()
        transaction_records = build_purchase_vs_bid_export(
            df,
            # The converter's base time (the run's start), period duration and
            # period count are the run's, so the page builds it
            ts_converter=TimestampConverter(
                ctx.base_time,
                ctx.duration_hours,
                ctx.periods,
            ),
            market_price=ctx.market_price,
            platform_markup=ctx.platform_markup,
            price_range=ctx.price_range,
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
