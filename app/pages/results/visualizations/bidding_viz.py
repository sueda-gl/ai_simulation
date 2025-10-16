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
        vendor_price = getattr(st.session_state.sim_params, 'market_price', 100.0)  # Default €100
        platform_markup = getattr(st.session_state.sim_params, 'platform_markup', 0.1)  # Default 10%
        price_range = getattr(st.session_state.sim_params, 'price_range', 0.25)  # Default 25%
    else:
        # Fallback defaults
        vendor_price = 100.0
        platform_markup = 0.1
        price_range = 0.25
    
    # Calculate bidding range using the formula
    baseline_price = (1 + platform_markup) * vendor_price  # Pc = (1+m) × vendor_price
    min_bid_price = (1 - price_range) * baseline_price      # Pmb = (1-r) × Pc
    max_bid_price = (1 + price_range) * baseline_price      # Ppn = (1+r) × Pc
    
    # Top section: Current parameters and calculated range
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(decision_data):,}")
    
    with col2:
        st.metric("Vendor Price", f"€{vendor_price:.2f}")
    
    with col3:
        st.metric("Baseline Price (Pc)", f"€{baseline_price:.2f}")
    
    with col4:
        st.metric("Range Parameter (r)", f"{price_range:.1%}")
    
    # Bidding range display
    st.markdown("---")
    st.markdown("**📊 Bidding Range Formula**")
    
    col_formula, col_range = st.columns([1, 1])
    
    with col_formula:
        st.markdown("**Formula Components:**")
        st.write(f"• **Vendor Price**: €{vendor_price:.2f}")
        st.write(f"• **Platform Markup (m)**: {platform_markup:.1%}")
        st.write(f"• **Range Parameter (r)**: {price_range:.1%}")
        st.write("")
        st.markdown("**Calculations:**")
        st.write(f"• **Baseline Price (Pc)**: (1 + {platform_markup:.1%}) × €{vendor_price:.2f} = €{baseline_price:.2f}")
        st.write(f"• **Min Bid (Pmb)**: (1 - {price_range:.1%}) × €{baseline_price:.2f} = €{min_bid_price:.2f}")
        st.write(f"• **Max Bid (Ppn)**: (1 + {price_range:.1%}) × €{baseline_price:.2f} = €{max_bid_price:.2f}")
    
    with col_range:
        st.markdown("**📈 Bidding Range:**")
        
        # Visual range display
        range_width = max_bid_price - min_bid_price
        
        # Create metrics for the range
        col_min, col_max = st.columns(2)
        with col_min:
            st.metric("Minimum Bid", f"€{min_bid_price:.2f}")
        with col_max:
            st.metric("Maximum Bid", f"€{max_bid_price:.2f}")
        
        st.metric("Range Width", f"€{range_width:.2f}")
        
        # Show the range notation
        st.success(f"**Bidding Range**: [€{min_bid_price:.2f}, €{max_bid_price:.2f})")
        st.caption("Range notation: [minimum, maximum)")
    
    # Configuration section
    st.markdown("---")
    st.markdown("**🎛️ Bidding Behavior:**")
    
    st.info("**Default Behavior**: Random bid amount within the calculated range")
    st.caption("💡 Agents will select random bid values between the minimum and maximum bid prices")
    
    # Show example bids
    if st.button("🎲 Show Example Bids", help="Generate sample bid values within the range"):
        import random
        st.markdown("**🎯 Example Bid Values:**")
        
        # Generate 5 random example bids
        example_bids = []
        for i in range(5):
            random_bid = random.uniform(min_bid_price, max_bid_price)
            example_bids.append(f"€{random_bid:.2f}")
        
        st.write(f"Sample bids: {', '.join(example_bids)}")
        st.caption(f"All values fall within [€{min_bid_price:.2f}, €{max_bid_price:.2f})")
    
    # Current simulation results summary - REQUEST LEVEL
    st.markdown("---")
    st.markdown("**📊 Actual Bid Values from Simulation (Request-Level)**")
    st.caption("Each bid request gets a unique random bid value")
    
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
            
            # Histogram of bid values
            st.markdown("**📈 Distribution of Actual Bid Values:**")
            
            col_hist, col_info = st.columns([2, 1])
            
            with col_hist:
                fig = px.histogram(
                    x=all_bids,
                    nbins=30,
                    title=f"Distribution of {len(all_bids):,} Bid Values",
                    labels={'x': 'Bid Amount (€)', 'count': 'Number of Bids'}
                )
                
                # Add vertical lines for theoretical range
                fig.add_vline(x=min_bid_price, line_dash="dash", line_color="red", 
                             annotation_text=f"Min €{min_bid_price:.2f}")
                fig.add_vline(x=max_bid_price, line_dash="dash", line_color="red",
                             annotation_text=f"Max €{max_bid_price:.2f}")
                
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
            st.info("No bid requests found (no agents chose to bid)")
    else:
        st.caption("No purchase_requests data available")

