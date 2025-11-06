# app/pages/results/visualizations/bidding_viz.py
"""
Bidding-related visualization functions.
Handles bid_value decision.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def render_bid_value(df, decision_name, decision_title, decision_data):
    """Visualization for bid_value with bidding price range formula"""
    
    # Get parameters from session state (from Page 1)
    if hasattr(st.session_state, 'sim_params'):
        platform_markup = getattr(st.session_state.sim_params, 'platform_markup', 0.1)  # Default 10%
        price_range = getattr(st.session_state.sim_params, 'price_range', 0.25)  # Default 25%
    else:
        # Fallback defaults
        platform_markup = 0.1
        price_range = 0.25
    
    # ============================================================================
    # SECTION 1: ACTUAL SIMULATION RESULTS (Request-Level)
    # ============================================================================
    st.markdown("### 📊 Actual Bid Values from Simulation")
    st.caption("Each bid request gets a unique random bid value based on the vendor's price")
    
    if 'purchase_requests' in df.columns:
        # Extract all bid values from all requests
        all_bids = []
        
        for idx, row in df.iterrows():
            requests = row.get('purchase_requests', [])
            if isinstance(requests, list):
                for req in requests:
                    if isinstance(req, dict):
                        bid_val = req.get('bid_value')
                        # Only include actual numeric bid values (not "N/A")
                        if bid_val != 'N/A' and bid_val is not None:
                            try:
                                all_bids.append(float(bid_val))
                            except (ValueError, TypeError):
                                pass
        
        if len(all_bids) > 0:
            # Summary metrics
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            
            with col_stats1:
                st.metric("Total Bid Requests", f"{len(all_bids):,}", 
                         help="Number of BID requests across all agents")
            
            with col_stats2:
                st.metric("Mean Bid", f"€{np.mean(all_bids):.2f}")
            
            with col_stats3:
                st.metric("Min Bid", f"€{min(all_bids):.2f}")
            
            with col_stats4:
                st.metric("Max Bid", f"€{max(all_bids):.2f}")
            
            # Distribution visualization
            st.markdown("**📈 Distribution of Bid Values:**")
            
            col_hist, col_info = st.columns([2, 1])
            
            with col_hist:
                fig = px.histogram(
                    x=all_bids,
                    nbins=30,
                    title=f"Distribution of {len(all_bids):,} Bid Values Across All Requests",
                    labels={'x': 'Bid Amount (€)', 'count': 'Number of Bids'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_info:
                st.markdown("**📊 Statistics**")
                stats_df = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{len(all_bids):,}",
                        f"€{np.mean(all_bids):.2f}",
                        f"€{np.median(all_bids):.2f}",
                        f"€{np.std(all_bids):.2f}",
                        f"€{min(all_bids):.2f}",
                        f"€{max(all_bids):.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Show unique count
                unique_bids = len(set(all_bids))
                st.caption(f"✅ {unique_bids:,} unique bid values")
                if unique_bids == len(all_bids):
                    st.success("🎯 All bids are unique!")
        else:
            st.info("ℹ️ No bid requests found (no agents chose to bid in this simulation)")
    else:
        st.warning("⚠️ No purchase_requests data available in results")
    
    # ============================================================================
    # SECTION 2: FORMULA EXPLANATION (Educational)
    # ============================================================================
    st.markdown("---")
    st.markdown("### 📚 How Bid Values Are Calculated")
    st.info("**Note**: The following is an illustration of how the bidding range formula works. Each vendor in the simulation has its own price, so bid ranges vary by vendor.")
    
    # Use example vendor price for illustration
    example_vendor_price = 100.0
    
    # Calculate example bidding range using the formula
    baseline_price = (1 + platform_markup) * example_vendor_price  # Pc = (1+m) × vendor_price
    min_bid_price = (1 - price_range) * baseline_price      # Pmb = (1-r) × Pc
    max_bid_price = (1 + price_range) * baseline_price      # Ppn = (1+r) × Pc
    
    # Display formula parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Simulation Parameters:**")
        st.write(f"• **Platform Markup (m)**: {platform_markup:.1%}")
        st.write(f"• **Price Range Parameter (r)**: {price_range:.1%}")
    
    with col2:
        st.markdown("**🎯 Bidding Behavior:**")
        st.write("Each agent selects a **random bid value** within the range calculated for their chosen vendor")
    
    st.markdown("---")
    
    # Formula breakdown with example
    col_formula, col_example = st.columns([1, 1])
    
    with col_formula:
        st.markdown("**📐 Formula:**")
        st.code("""
Given a vendor price V:
  
1. Baseline Price (Pc):
   Pc = (1 + m) × V
   
2. Minimum Bid (Pmb):
   Pmb = (1 - r) × Pc
   
3. Maximum Bid (Ppn):
   Ppn = (1 + r) × Pc
   
4. Actual Bid:
   Random value in [Pmb, Ppn)
        """, language="text")
    
    with col_example:
        st.markdown(f"**💡 Example Calculation:**")
        st.markdown(f"*Assuming vendor price = €{example_vendor_price:.2f}*")
        st.write("")
        st.write(f"**Step 1: Baseline Price**")
        st.write(f"Pc = (1 + {platform_markup:.1%}) × €{example_vendor_price:.2f}")
        st.write(f"Pc = €{baseline_price:.2f}")
        st.write("")
        st.write(f"**Step 2: Minimum Bid**")
        st.write(f"Pmb = (1 - {price_range:.1%}) × €{baseline_price:.2f}")
        st.write(f"Pmb = €{min_bid_price:.2f}")
        st.write("")
        st.write(f"**Step 3: Maximum Bid**")
        st.write(f"Ppn = (1 + {price_range:.1%}) × €{baseline_price:.2f}")
        st.write(f"Ppn = €{max_bid_price:.2f}")
        st.write("")
        st.success(f"**Bidding Range**: [€{min_bid_price:.2f}, €{max_bid_price:.2f})")
        
        # Show sample bids from this example
        import random
        st.write("")
        st.write("**Example random bids:**")
        example_bids = []
        for i in range(5):
            random_bid = random.uniform(min_bid_price, max_bid_price)
            example_bids.append(f"€{random_bid:.2f}")
        st.caption(", ".join(example_bids))
    
    st.caption("💡 **Remember**: In the actual simulation, each of the 6 vendors has a different price, so the bidding range varies per vendor.")

