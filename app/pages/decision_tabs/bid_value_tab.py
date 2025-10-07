# app/pages/decision_tabs/bid_value_tab.py
"""
Bid Value decision tab configuration.
Shows the bidding range formula and current parameters.
"""
import streamlit as st
from app.pages.decision_execution import render_simulation_buttons


def render_bid_value_tab():
    """Render bid_value specific configuration"""
    st.markdown('<h3 class="section-header">🎯 Bid Value Configuration</h3>', unsafe_allow_html=True)
    
    st.info("""
    **Bid Value Decision**: When an agent chooses to bid (rather than purchase immediately), 
    this decision determines the bid amount they submit.
    """)
    
    # Show formula section
    st.markdown("---")
    st.markdown('<h4 class="subsection-header">📐 Bidding Range Formula</h4>', unsafe_allow_html=True)
    
    # Get parameters from session state
    if hasattr(st.session_state, 'sim_params'):
        vendor_price = st.session_state.sim_params.market_price
        platform_markup = st.session_state.sim_params.platform_markup
        price_range = st.session_state.sim_params.price_range
    else:
        vendor_price = 100.0
        platform_markup = 0.1
        price_range = 0.25
    
    # Calculate range
    baseline_price = (1 + platform_markup) * vendor_price
    min_bid = (1 - price_range) * baseline_price
    max_bid = (1 + price_range) * baseline_price
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Mathematical Formula:**")
        
        # Display formulas using LaTeX
        st.latex(r"P_c = (1 + m) \times P_v")
        st.caption("Baseline customer price")
        
        st.latex(r"P_{mb} = (1 - r) \times P_c")
        st.caption("Minimum bid price")
        
        st.latex(r"P_{pn} = (1 + r) \times P_c")
        st.caption("Maximum bid price (Purchase Now)")
        
        st.markdown("**Where:**")
        st.markdown("""
        - **Pv** = Vendor price (market_price)
        - **m** = Platform markup
        - **r** = Price range parameter
        - **Pc** = Customer baseline price
        - **Pmb** = Minimum bid price
        - **Ppn** = Maximum bid price
        """)
    
    with col2:
        st.markdown("**Current Parameters:**")
        
        # Display current parameter values
        st.metric("Vendor Price (Pv)", f"€{vendor_price:.2f}")
        st.metric("Platform Markup (m)", f"{platform_markup:.1%}")
        st.metric("Price Range (r)", f"{price_range:.1%}")
        
        st.markdown("---")
        st.markdown("**Calculated Bidding Range:**")
        
        # Show baseline price
        st.metric("Baseline Price (Pc)", f"€{baseline_price:.2f}")
        
        # Show bidding range
        range_col1, range_col2 = st.columns(2)
        with range_col1:
            st.metric("Min Bid (Pmb)", f"€{min_bid:.2f}")
        with range_col2:
            st.metric("Max Bid (Ppn)", f"€{max_bid:.2f}")
        
        # Highlight the range
        st.success(f"**Bidding Range**: [€{min_bid:.2f}, €{max_bid:.2f}]")
        st.caption("Agents will select random bids uniformly within this range")
    
    # Example calculation section
    st.markdown("---")
    st.markdown('<h4 class="subsection-header">🧮 Example Calculation</h4>', unsafe_allow_html=True)
    
    with st.expander("📊 View Step-by-Step Calculation", expanded=False):
        st.markdown("**Step 1: Calculate Baseline Price**")
        st.code(f"Pc = (1 + {platform_markup}) × €{vendor_price:.2f} = €{baseline_price:.2f}")
        
        st.markdown("**Step 2: Calculate Minimum Bid**")
        st.code(f"Pmb = (1 - {price_range}) × €{baseline_price:.2f} = €{min_bid:.2f}")
        
        st.markdown("**Step 3: Calculate Maximum Bid**")
        st.code(f"Ppn = (1 + {price_range}) × €{baseline_price:.2f} = €{max_bid:.2f}")
        
        st.markdown("**Step 4: Generate Random Bid**")
        st.code(f"bid_value = uniform_random({min_bid:.2f}, {max_bid:.2f})")
        
        st.info(f"""
        **Example**: If an agent chooses to bid, their bid amount will be randomly selected 
        from the range [€{min_bid:.2f}, €{max_bid:.2f}] using a uniform distribution.
        
        For instance, possible bid values could be:
        - €{min_bid + (max_bid - min_bid) * 0.25:.2f}
        - €{min_bid + (max_bid - min_bid) * 0.50:.2f}
        - €{min_bid + (max_bid - min_bid) * 0.75:.2f}
        """)
    
    # Behavior configuration section
    st.markdown("---")
    st.markdown('<h4 class="subsection-header">⚙️ Bid Selection Behavior</h4>', unsafe_allow_html=True)
    
    col_behavior1, col_behavior2 = st.columns([2, 1])
    
    with col_behavior1:
        st.info("""
        **Current Implementation**: Uniform Random Distribution
        
        When an agent decides to bid (via the `purchase_vs_bid` decision), 
        their bid amount is selected uniformly at random from the calculated 
        bidding range [Pmb, Ppn].
        
        **Key Features:**
        - ✅ Only generates bid if agent chose "bid" option
        - ✅ Returns NaN (empty/missing value) if agent chose "Purchase Now" option
        - ✅ Uses agent-specific RNG for reproducibility
        - ✅ Rounds to 2 decimal places (currency standard)
        """)
    
    with col_behavior2:
        st.markdown("**Distribution Type**")
        st.success("Uniform")
        st.caption("Equal probability for any value in the range")
        
        st.markdown("**Dependencies**")
        st.info("Requires `purchase_vs_bid` decision result")
        st.caption("If agent chose 'Purchase Now', bid_value = NaN (empty)")
    
    # Global parameters info
    st.markdown("---")
    st.markdown('<h4 class="subsection-header">🌐 Global Parameters Used</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    This decision uses the following global parameters from **Page 1: Common Simulation Parameters**:
    """)
    
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        st.metric("Market Price", f"€{vendor_price:.2f}", help="Vendor price from Page 1")
    
    with param_col2:
        st.metric("Platform Markup", f"{platform_markup:.1%}", help="Platform markup from Page 1")
    
    with param_col3:
        st.metric("Price Range", f"{price_range:.1%}", help="Price range from Page 1")
    
    st.caption("""
    💡 **Note**: To change these parameters, go to **Page 1: Common Simulation Parameters** 
    and adjust the Market Parameters section.
    """)
    
    # Technical details in expander
    st.markdown("---")
    with st.expander("🔧 Technical Implementation Details", expanded=False):
        st.markdown("**Function:** `bid_value(agent_state, params, rng, simulation_config)`")
        st.markdown("**Location:** `src/decisions/bid_value.py`")
        
        st.markdown("**Logic Flow:**")
        st.code("""
1. Check if agent chose 'bid' (from purchase_vs_bid decision)
   → If 'Purchase Now', return {"bid_value": np.nan}
   
2. Extract pricing parameters from simulation_config:
   - vendor_price (market_price)
   - platform_markup (m)
   - price_range (r)
   
3. Calculate bidding range:
   - baseline_price = (1 + m) × vendor_price
   - min_bid_price = (1 - r) × baseline_price
   - max_bid_price = (1 + r) × baseline_price
   
4. Generate random bid using agent's RNG:
   - bid_amount = rng.uniform(min_bid_price, max_bid_price)
   
5. Round to 2 decimal places and return
        """, language="text")
        
        st.markdown("**Random Number Generation:**")
        st.markdown("""
        - Uses agent-specific RNG (passed as `rng` parameter)
        - Ensures reproducibility with same seed
        - Different agents get different bid values
        - `rng.uniform(low, high)` generates from [low, high)
        """)
    
    # Render simulation buttons
    render_simulation_buttons(
        decision_name="bid_value",
        selected_decisions=st.session_state.decision_params.selected_decisions
    )

