# Enhanced AI Agent Simulation Dashboard with Two-Page Interface
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.orchestrator import Orchestrator
from src.trait_engine import TraitEngine

# Page configuration
st.set_page_config(
    page_title="Enhanced AI Agent Simulation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
""", unsafe_allow_html=True)

# Data classes for parameter management
@dataclass
class SimulationParameters:
    """Common simulation parameters (Page 1)"""
    # Time parameters
    periods: int = 1
    duration_hours: float = 1.0  # Duration in hours (converted to seconds for simulation)
    
    # Market parameters
    num_vendors: int = 5
    market_price: float = 10.0
    vendor_price_min: float = 8.0
    vendor_price_max: float = 12.0
    
    # Product offering
    products_per_vendor: int = 100  # NV
    carryover: bool = False
    
    # Pricing parameters
    bidding_percentage: float = 0.5  # bp (proportion available for bidding)
    platform_markup: float = 0.1  # m
    price_range: float = 0.25  # r
    price_grid: int = 11  # g (must be odd)
    
    # Income distribution parameters
    income_min: float = 1000.0
    income_max: float = 10000.0
    income_avg: float = 5000.0
    income_distribution: str = "lognormal"  # lognormal, pareto, weibull
    
    # Income categories
    num_discount_categories: int = 3  # NDIC
    num_fixed_categories: int = 5  # NFIC
    
    # Consumption limits
    consumption_limits: Dict[str, float] = field(default_factory=dict)
    
    # Vendor prices (either generated or from file)
    vendor_prices: Optional[List[float]] = None
    vendor_price_source: str = "random"  # "random" or "file"
    
    def get_duration_seconds(self) -> float:
        """Convert duration from hours to seconds"""
        return self.duration_hours * 3600
    
    def get_purchase_now_price(self, base_price: float) -> float:
        """Calculate Purchase Now price from base price"""
        customer_price = base_price * (1 + self.platform_markup)
        return customer_price * (1 + self.price_range)
    
    def get_minimum_bid_price(self, base_price: float) -> float:
        """Calculate minimum bid price from base price"""
        customer_price = base_price * (1 + self.platform_markup)
        return customer_price * (1 - self.price_range)
    
    def get_num_auction_products(self) -> int:
        """Calculate number of products available for auction per vendor"""
        return int(self.products_per_vendor * self.bidding_percentage)

@dataclass
class DecisionParameters:
    """Decision-specific parameters (Page 2)"""
    selected_decisions: List[str] = field(default_factory=list)
    decision_configs: Dict[str, Dict] = field(default_factory=dict)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'page1'
if 'sim_params' not in st.session_state:
    st.session_state.sim_params = SimulationParameters()
if 'decision_params' not in st.session_state:
    st.session_state.decision_params = DecisionParameters()
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# Navigation functions
def go_to_page2():
    st.session_state.page = 'page2'

def go_to_page1():
    st.session_state.page = 'page1'

def _show_overview(df, title_suffix=""):
    """Helper function to show simulation overview for a DataFrame"""
    st.subheader(f"Simulation Overview{title_suffix}")
    
    # Check if this is dependent variable mode (only has donation_default column)
    is_depvar_mode = len(df.columns) == 1 and 'donation_default' in df.columns
    
    # Display anchor weights info (not for depvar mode)
    if not is_depvar_mode:
        st.caption(f"📊 Anchor mix: {st.session_state.anchor_observed_weight:.2f} observed | {1 - st.session_state.anchor_observed_weight:.2f} predicted")
    else:
        st.caption("📊 Resampling from empirical distribution of 280 original donation rates")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(df):,}")
    
    with col2:
        if not is_depvar_mode:
            trait_cols = ['Assigned Allowance Level', 'Group_experiment', 'Honesty_Humility', 
                         'Study Program', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
            st.metric("Traits Available", len([c for c in trait_cols if c in df.columns]))
        else:
            st.metric("Source", "280 participants")
    
    with col3:
        if not is_depvar_mode:
            decision_cols = [c for c in df.columns if c not in trait_cols]
            st.metric("Decisions Computed", len(decision_cols))
        else:
            st.metric("Method", "Bootstrap")
    
    with col4:
        # Determine which column to use for display
        donation_col = 'donation_default_raw' if 'donation_default_raw' in df.columns else 'donation_default'
        if donation_col in df.columns:
            avg_label = "Avg Donation Rate" + (" (raw)" if donation_col == 'donation_default_raw' else "")
            st.metric(avg_label, f"{df[donation_col].mean():.1%}")
    
    # Donation rate analysis (if available)
    donation_col = 'donation_default_raw' if 'donation_default_raw' in df.columns else 'donation_default'
    if donation_col in df.columns:
        raw_suffix = " (raw pre-truncation)" if donation_col == 'donation_default_raw' else ""
        st.subheader(f" Donation Rate Analysis{title_suffix}{raw_suffix}")
        
        # Distribution plot
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(
                df, 
                x=donation_col,
                nbins=30,
                title=f"Distribution of Donation Rates{title_suffix}{raw_suffix}",
                labels={donation_col: 'Donation Rate', 'count': 'Number of Agents'},
                marginal="box"
            )
            fig.update_layout(
                xaxis_tickformat='.0%',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📈 Statistics**")
            donation_stats = df[donation_col].describe()
            
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
                'Value': [
                    f"{donation_stats['mean']:.1%}",
                    f"{donation_stats['std']:.3f}",
                    f"{donation_stats['min']:.1%}",
                    f"{donation_stats['max']:.1%}",
                    f"{donation_stats['50%']:.1%}",
                    f"{donation_stats['25%']:.1%}",
                    f"{donation_stats['75%']:.1%}"
                ]
            })
            st.dataframe(stats_df, hide_index=True)

def run_simulation():
    """Run simulation with current parameters"""
    st.session_state.page = 'results'
    # Simulation will be triggered on results page

# Main title
st.markdown('<h1 class="main-header">Enhanced AI Agent Simulation</h1>', unsafe_allow_html=True)

# Page routing
if st.session_state.page == 'page1':
    # Page 1: Common Simulation Parameters
    st.markdown('<h2 class="page-header">Page 1: Common Simulation Parameters</h2>', unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Time Parameters Section
        st.markdown('<h3 class="section-header">⏱️ Time Parameters</h3>', unsafe_allow_html=True)
        
        st.session_state.sim_params.periods = st.number_input(
            "Number of Periods",
            min_value=1,
            max_value=100,
            value=st.session_state.sim_params.periods,
            help="Number of periods for simulation run"
        )
        
        st.session_state.sim_params.duration_hours = st.number_input(
            "Duration per Period (hours)",
            min_value=0.1,
            max_value=24.0,
            value=st.session_state.sim_params.duration_hours,
            step=0.1,
            help="Duration of each period in hours (will be converted to seconds for simulation)"
        )
        st.caption(f"Duration in seconds: {st.session_state.sim_params.get_duration_seconds():.0f}")
        
        # Market Parameters Section
        st.markdown('<h3 class="section-header">🏪 Market Parameters</h3>', unsafe_allow_html=True)
        
        st.session_state.sim_params.num_vendors = st.number_input(
            "Number of Vendors",
            min_value=1,
            max_value=50,
            value=st.session_state.sim_params.num_vendors,
            help="Number of vendors operating on the platform"
        )
        
        st.session_state.sim_params.market_price = st.number_input(
            "Average Market Price ($)",
            min_value=0.01,
            max_value=1000.0,
            value=st.session_state.sim_params.market_price,
            step=0.01,
            help="The average market price of the product"
        )
        
        col_min, col_max = st.columns(2)
        with col_min:
            st.session_state.sim_params.vendor_price_min = st.number_input(
                "Min Vendor Price ($)",
                min_value=0.01,
                max_value=st.session_state.sim_params.market_price,
                value=st.session_state.sim_params.vendor_price_min,
                step=0.01,
                help="Minimum vendor price"
            )
        
        with col_max:
            st.session_state.sim_params.vendor_price_max = st.number_input(
                "Max Vendor Price ($)",
                min_value=st.session_state.sim_params.market_price,
                max_value=1000.0,
                value=st.session_state.sim_params.vendor_price_max,
                step=0.01,
                help="Maximum vendor price"
            )
        
        # Product Offering Section
        st.markdown('<h3 class="section-header">📦 Product Offering</h3>', unsafe_allow_html=True)
        
        st.session_state.sim_params.products_per_vendor = st.number_input(
            "Products per Vendor (NV)",
            min_value=1,
            max_value=10000,
            value=st.session_state.sim_params.products_per_vendor,
            help="Number of products offered by each vendor at the beginning of each period"
        )
        
        st.session_state.sim_params.carryover = st.checkbox(
            "Carryover Unsold Products",
            value=st.session_state.sim_params.carryover,
            help="If checked, unsold products carry over to the next period"
        )
        
        st.session_state.sim_params.bidding_percentage = st.slider(
            "Bidding Percentage (bp)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.sim_params.bidding_percentage,
            step=0.05,
            help="Proportion of products available for bidding (NA = bp × NV)"
        )
        st.caption(f"Products for auction per vendor: {st.session_state.sim_params.get_num_auction_products()}")
    
    with col2:
        # Pricing Parameters Section
        st.markdown('<h3 class="section-header">💰 Pricing Parameters</h3>', unsafe_allow_html=True)
        
        st.session_state.sim_params.platform_markup = st.slider(
            "Platform Markup (m)",
            min_value=0.0,
            max_value=0.5,
            value=st.session_state.sim_params.platform_markup,
            step=0.01,
            help="Platform markup: Customer Price = (1+m) × Vendor Price"
        )
        
        st.session_state.sim_params.price_range = st.slider(
            "Price Range (r)",
            min_value=0.0,
            max_value=0.5,
            value=st.session_state.sim_params.price_range,
            step=0.05,
            help="Price range for Purchase Now and Minimum Bid prices"
        )
        
        st.session_state.sim_params.price_grid = st.number_input(
            "Price Grid Categories (g)",
            min_value=3,
            max_value=21,
            value=st.session_state.sim_params.price_grid,
            step=2,
            help="Number of price categories (must be odd)"
        )
        
        # Ensure price grid is odd
        if st.session_state.sim_params.price_grid % 2 == 0:
            st.caption("Price grid adjusted to odd number: " + str(st.session_state.sim_params.price_grid + 1))
            st.session_state.sim_params.price_grid += 1
        
        # Show calculated prices example
        example_vendor_price = st.session_state.sim_params.market_price
        example_customer_price = example_vendor_price * (1 + st.session_state.sim_params.platform_markup)
        example_pn_price = st.session_state.sim_params.get_purchase_now_price(example_vendor_price)
        example_min_bid = st.session_state.sim_params.get_minimum_bid_price(example_vendor_price)
        

        
        # Income Distribution Section
        st.markdown('<h3 class="section-header">💵 Income Distribution</h3>', unsafe_allow_html=True)
        
        st.session_state.sim_params.income_distribution = st.selectbox(
            "Income Distribution Type",
            ["lognormal", "pareto", "weibull"],
            index=["lognormal", "pareto", "weibull"].index(st.session_state.sim_params.income_distribution),
            help="Distribution function for generating agent incomes"
        )
        
        col_inc1, col_inc2 = st.columns(2)
        with col_inc1:
            st.session_state.sim_params.income_min = st.number_input(
                "Minimum Income ($)",
                min_value=0.0,
                max_value=1000000.0,
                value=st.session_state.sim_params.income_min,
                step=100.0
            )
            
            st.session_state.sim_params.income_avg = st.number_input(
                "Average/Median Income ($)",
                min_value=st.session_state.sim_params.income_min,
                max_value=1000000.0,
                value=st.session_state.sim_params.income_avg,
                step=100.0
            )
        
        with col_inc2:
            st.session_state.sim_params.income_max = st.number_input(
                "Maximum Income ($)",
                min_value=st.session_state.sim_params.income_avg,
                max_value=10000000.0,
                value=st.session_state.sim_params.income_max,
                step=100.0
            )
        
        # Income Categories Section
        st.markdown('<h3 class="section-header">📊 Income Categories</h3>', unsafe_allow_html=True)
        
        col_cat1, col_cat2 = st.columns(2)
        with col_cat1:
            st.session_state.sim_params.num_discount_categories = st.number_input(
                "Discount Income Categories (NDIC)",
                min_value=1,
                max_value=10,
                value=st.session_state.sim_params.num_discount_categories,
                help="Number of customer discount income categories"
            )
        
        with col_cat2:
            st.session_state.sim_params.num_fixed_categories = st.number_input(
                "Fixed Income Categories (NFIC)",
                min_value=1,
                max_value=10,
                value=st.session_state.sim_params.num_fixed_categories,
                help="Number of customer fixed income categories"
            )
    
    # Vendor Price Configuration
    st.markdown('<h3 class="section-header">🏷️ Vendor Price Configuration</h3>', unsafe_allow_html=True)
    
    price_source = st.radio(
        "Vendor Price Source",
        ["Generate Randomly", "Upload Price File"],
        index=0 if st.session_state.sim_params.vendor_price_source == "random" else 1,
        horizontal=True
    )
    
    if price_source == "Generate Randomly":
        st.session_state.sim_params.vendor_price_source = "random"
        st.caption(f"Vendor prices will be randomly generated within the range ${st.session_state.sim_params.vendor_price_min:.2f} - ${st.session_state.sim_params.vendor_price_max:.2f} with an average of ${st.session_state.sim_params.market_price:.2f}")
    else:
        st.session_state.sim_params.vendor_price_source = "file"
        uploaded_file = st.file_uploader(
            "Upload Vendor Prices CSV",
            type=['csv'],
            help="CSV file with vendor prices. Should have columns: vendor_id, price"
        )
        if uploaded_file is not None:
            try:
                prices_df = pd.read_csv(uploaded_file)
                st.session_state.sim_params.vendor_prices = prices_df['price'].tolist()
                st.caption(f"Loaded {len(st.session_state.sim_params.vendor_prices)} vendor prices")
            except Exception as e:
                st.caption(f"Error loading price file: {e}")
    
    # Consumption Limits Configuration
    st.markdown('<h3 class="section-header">🛒 Consumption Limits</h3>', unsafe_allow_html=True)
    
    st.caption("Set consumption limits per product for each income category per period")
    
    # Create a simple interface for setting consumption limits
    total_categories = st.session_state.sim_params.num_fixed_categories
    
    consumption_limits = {}
    cols = st.columns(min(5, total_categories))
    for i in range(total_categories):
        col_idx = i % len(cols)
        with cols[col_idx]:
            limit = st.number_input(
                f"Category {i+1} Limit",
                min_value=0,
                max_value=100,
                value=st.session_state.sim_params.consumption_limits.get(f"cat_{i+1}", 10),
                key=f"consumption_limit_{i}"
            )
            consumption_limits[f"cat_{i+1}"] = limit
    
    st.session_state.sim_params.consumption_limits = consumption_limits
    
    # Navigation
    st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        st.button("Next: Decision Parameters →", type="primary", on_click=go_to_page2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'page2' or st.session_state.page == 'results':
    # Page 2: Decision-Specific Parameters (also shown on results page)
    if st.session_state.page == 'page2':
        st.markdown('<h2 class="page-header">Page 2: Decision-Specific Parameters</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="page-header">Simulation Results</h2>', unsafe_allow_html=True)
    
    # Decision selection (like in original app.py)
    if st.session_state.page == 'page2':
        st.markdown('<h3 class="section-header">🎯 Decision Selection</h3>', unsafe_allow_html=True)
    
    all_decisions = [
        "donation_default",
        "disclose_income", 
        "disclose_documents",
        "rejected_transaction_defaults",
        "vendor_choice_weights",
        "consumption_quantity",
        "consumption_frequency", 
        "vendor_selection",
        "purchase_vs_bid",
        "bid_value",
        "rejected_transaction_option",
        "rejected_bid_value",
        "final_donation_rate"
    ]
    
    # Multi-select with "Select All" functionality (like original app.py)
    if st.session_state.page == 'page2':
        select_all = st.checkbox("Select All Decisions", value=True)
        
        if select_all:
            selected_decisions = st.multiselect(
                "Selected Decisions",
                all_decisions,
                default=all_decisions,
                help="Select one or more decisions to run",
                disabled=True
            )
        else:
            selected_decisions = st.multiselect(
                "Select Decisions to Run",
                all_decisions,
                default=st.session_state.decision_params.selected_decisions or ["donation_default"],
                help="Select one or more decisions to run"
            )
        
        # Ensure at least one decision is selected
        if not selected_decisions:
            st.caption("Please select at least one decision")
            selected_decisions = ["donation_default"]
        
        st.session_state.decision_params.selected_decisions = selected_decisions
    
    # Sidebar configuration (like original app.py) - DYNAMIC BASED ON SELECTED DECISIONS
    st.sidebar.title("⚙️ Decision Parameters")
    
    # DYNAMIC PARAMETERS BASED ON SELECTED DECISIONS
    selected_decisions = st.session_state.decision_params.selected_decisions
    
    # Only show donation-specific parameters if donation_default is selected
    if "donation_default" in selected_decisions:
        # Population mode selector (like original app.py) - ONLY for donation_default
        st.sidebar.subheader("Population Generation")
        population_mode = st.sidebar.radio(
            "Population Mode",
            ["Copula (synthetic)", "Research Specification", "Compare both"],
            index=0,
            help="Copula: Generate synthetic agents via fitted copula\nDocumentation: Use original participants with stochastic draws\nCompare both: Show Copula vs Documentation side-by-side"
        )
        
        # Income specification selector (like original app.py)
        if population_mode != "Dependent variable resampling":
            st.sidebar.subheader("Income Specification")
            income_spec_mode = st.sidebar.radio(
                "Income Mode for Donation Model",
                ["categorical only", "continuous only", "compare side-by-side"],
                index=0,
                help="Choose income treatment: categorical (5 categories), continuous (linear), or compare both side-by-side"
            )
        else:
            income_spec_mode = "categorical only"  # Default for dependent variable mode
        
        # Stochastic component option (like original app.py)
        if population_mode == "Copula (synthetic)" or population_mode == "Compare both":
            st.sidebar.subheader("Stochastic Component")
            sigma_in_copula = st.sidebar.checkbox(
                "Add Normal(anchor, σ) draw to Copula runs",
                value=False,
                help="When enabled, Copula mode will also use the stochastic component (Normal distribution draw) like Documentation mode"
            )
            # Slider to adjust sigma (original SD ≈ 9 on 0–112 scale).
            sigma_value_ui = st.sidebar.slider(
                "σ (standard deviation) on 0–112 scale",
                min_value=0.0,
                max_value=15.0,
                value=9.0,
                step=0.1,
                help="Controls the spread of the Normal(anchor, σ) draw. Set to 0 to disable variability."
            )
        else:
            sigma_in_copula = False  # Default for other modes
            # Provide sigma slider for documentation & compare-both modes as well
            sigma_value_ui = st.sidebar.slider(
                "σ (standard deviation) on 0–112 scale",
                min_value=0.0,
                max_value=15.0,
                value=9.0,
                step=0.1,
                help="Controls the spread of the Normal(anchor, σ) draw. Set to 0 to disable variability."
            )
        
        # Anchor weights slider (like original app.py)
        if population_mode != "Dependent variable resampling":
            st.sidebar.subheader("Anchor Mix")
            anchor_observed_weight = st.sidebar.slider(
                "Weight for observed versus modeled prosocial behavior",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                help="Anchor = w × Observed + (1-w) × Predicted. Default is 0.75 observed + 0.25 predicted."
            )
            st.sidebar.caption(f"Predicted weight: {1 - anchor_observed_weight:.2f}")
        else:
            anchor_observed_weight = 0.75  # Default value used in pre-computation
        
        # Raw output option (like original app.py)
        if population_mode != "Copula (synthetic)" or sigma_in_copula:
            st.sidebar.subheader("Output Options")
            raw_draw_mode = st.sidebar.checkbox(
                "Show pre-truncation (raw) donation rate",
                value=False,
                help="Display the raw Normal(anchor, σ) draw before any processing. This shows negative values and the full range of the stochastic draw before flooring at 0 and rescaling by personal maximum."
            )
        else:
            raw_draw_mode = False  # Not applicable for deterministic copula mode
    else:
        # If donation_default is not selected, set default values and show minimal sidebar
        population_mode = "Copula (synthetic)"  # Default population mode
        income_spec_mode = "categorical only"
        sigma_in_copula = False
        sigma_value_ui = 9.0
        anchor_observed_weight = 0.75
        raw_draw_mode = False
    
    # Simulation parameters (like original app.py)
    st.sidebar.subheader("Simulation Parameters")
    
    n_agents = st.sidebar.number_input(
        "Number of Agents",
        min_value=10,
        max_value=50000,
        value=1000,
        step=100,
        help="Number of synthetic agents to generate"
    )
    
    simulation_mode = st.sidebar.radio(
        "Simulation Mode",
        ["Single Run", "Monte-Carlo Study"],
        index=0,
        help="Single Run: One simulation with specified parameters\nMonte-Carlo: Multiple runs for uncertainty analysis"
    )
    
    if simulation_mode == "Single Run":
        seed = st.sidebar.number_input(
            "Random Seed",
            min_value=1,
            max_value=2147483647,
            value=42,
            help="Seed for reproducible results"
        )
    else:
        n_runs = st.sidebar.number_input(
            "Number of Runs",
            min_value=2,
            max_value=1000,
            value=100,
            step=10,
            help="Number of Monte-Carlo repetitions"
        )
        base_seed = st.sidebar.number_input(
            "Base Seed",
            min_value=1,
            max_value=2147483647,
            value=42,
            help="Starting seed (subsequent runs use base_seed + i)"
        )
    
    # Advanced options (like original app.py)
    st.sidebar.subheader("🔧 Advanced Options")

    # --- NEW: Global Income Distribution controls (visible on Page 2/Results) ---
    st.sidebar.subheader("💵 Income Distribution (Global)")
    income_dist_type = st.sidebar.selectbox(
        "Distribution Type",
        ["lognormal", "pareto", "weibull"],
        index=["lognormal", "pareto", "weibull"].index(st.session_state.sim_params.income_distribution)
    )
    st.session_state.sim_params.income_distribution = income_dist_type

    st.sidebar.caption("Set bounds and central tendency for generated incomes ($)")
    st.session_state.sim_params.income_min = st.sidebar.number_input(
        "Minimum Income",
        min_value=0.0,
        value=st.session_state.sim_params.income_min
    )
    st.session_state.sim_params.income_avg = st.sidebar.number_input(
        "Average / Median Income",
        min_value=st.session_state.sim_params.income_min,
        value=st.session_state.sim_params.income_avg
    )
    st.session_state.sim_params.income_max = st.sidebar.number_input(
        "Maximum Income",
        min_value=st.session_state.sim_params.income_avg,
        value=st.session_state.sim_params.income_max
    )
    # --- END NEW CONTROLS ---

    show_individual_agents = st.sidebar.checkbox(
        "Show Individual Agent Details",
        value=False,
        help="Display detailed breakdown of individual agents"
    )
    
    save_results = st.sidebar.checkbox(
        "Save Results to File",
        value=True,
        help="Save simulation outputs to outputs/ directory"
    )
    
    # Store settings in session state
    st.session_state.population_mode = population_mode
    st.session_state.income_spec_mode = income_spec_mode
    st.session_state.sigma_in_copula = sigma_in_copula
    st.session_state.sigma_value_ui = sigma_value_ui
    st.session_state.anchor_observed_weight = anchor_observed_weight
    st.session_state.raw_draw_mode = raw_draw_mode
    st.session_state.n_agents = n_agents
    st.session_state.simulation_mode = simulation_mode
    if simulation_mode == "Single Run":
        st.session_state.seed = seed
    else:
        st.session_state.n_runs = n_runs
        st.session_state.base_seed = base_seed
    st.session_state.show_individual_agents = show_individual_agents
    st.session_state.save_results = save_results
    
    # Main content area - show selected decisions info (only on page2)
    if st.session_state.page == 'page2':
        st.markdown('<h3 class="section-header">📋 Selected Decisions</h3>', unsafe_allow_html=True)
        
        if selected_decisions:
            for i, decision in enumerate(selected_decisions):
                st.markdown(f"**{i+1}.** {decision.replace('_', ' ').title()}")
            
            st.caption(f"Total decisions selected: {len(selected_decisions)}")
        else:
            st.caption("No decisions selected")
    
    # Run simulation button (like original app.py)
    st.sidebar.markdown("---")
    
    def run_simulation_from_sidebar():
        """Run simulation using original app.py logic"""
        try:
            with st.spinner("🔄 Generating synthetic agents and running simulation..."):
                # Helper to run simulation with chosen orchestrator and income mode
                def _run(pop_mode: str, inc_mode: str):
                    # Initialize appropriate orchestrator
                    if pop_mode == "documentation":
                        from src.orchestrator_doc_mode import OrchestratorDocMode
                        orchestrator = OrchestratorDocMode()
                    elif pop_mode == "depvar":
                        from src.orchestrator_depvar import OrchestratorDepVar
                        orchestrator = OrchestratorDepVar()
                        # Set raw output mode for dependent variable resampling
                        orchestrator.set_raw_output(st.session_state.raw_draw_mode)
                    else:  # copula
                        orchestrator = Orchestrator()
                    
                    # Override income specification in config based on choice (not for depvar mode)
                    if hasattr(orchestrator, 'config') and 'donation_default' in orchestrator.config:
                        if pop_mode != "depvar":  # depvar mode doesn't use these settings
                            orchestrator.config['donation_default']['regression']['income_mode'] = inc_mode
                            # Set stochastic flag for copula mode if checkbox is enabled
                            if pop_mode == "copula":
                                orchestrator.config['donation_default']['stochastic']['in_copula'] = st.session_state.sigma_in_copula
                            # Apply selected sigma value
                            orchestrator.config['donation_default']['stochastic']['sigma_value'] = st.session_state.sigma_value_ui
                            # Apply chosen anchor weights
                            orchestrator.config['donation_default']['anchor_weights']['observed'] = st.session_state.anchor_observed_weight
                            orchestrator.config['donation_default']['anchor_weights']['predicted'] = 1 - st.session_state.anchor_observed_weight
                            # Set raw output flag if applicable
                            if pop_mode == "documentation" or (pop_mode == "copula" and st.session_state.sigma_in_copula):
                                orchestrator.config['donation_default']['stochastic']['raw_output'] = st.session_state.raw_draw_mode

                        # --- NEW: push updated income distribution parameters ---
                        if hasattr(orchestrator, 'simulation_config'):
                            sim_section = orchestrator.simulation_config.setdefault('simulation', {})
                            sim_section['income_distribution'] = st.session_state.sim_params.income_distribution
                            sim_section['income_min'] = st.session_state.sim_params.income_min
                            sim_section['income_max'] = st.session_state.sim_params.income_max
                            sim_section['income_avg'] = st.session_state.sim_params.income_avg
                            # Rebuild income transformer so changes take effect
                            if hasattr(orchestrator, 'income_transformer'):
                                from src.income_transformer import IncomeTransformer
                                orchestrator.income_transformer = IncomeTransformer(orchestrator.simulation_config)
                        # --- END NEW ---
                    
                    # Handle multiple decisions
                    decision_param = None if len(st.session_state.decision_params.selected_decisions) == len(all_decisions) else st.session_state.decision_params.selected_decisions
                    return orchestrator.run_simulation(
                        n_agents=st.session_state.n_agents,
                        seed=st.session_state.seed if st.session_state.simulation_mode == "Single Run" else st.session_state.base_seed,
                        single_decision=decision_param
                    )
                
                # Run based on population and income specification modes
                results = {}
                
                if st.session_state.population_mode == "Compare both":
                    # Compare population modes
                    for pop_name, pop_type in [("copula", "copula"), ("doc_mode", "documentation")]:
                        if st.session_state.income_spec_mode == "compare side-by-side":
                            results[f"{pop_name}_categorical"] = _run(pop_type, "categorical")
                            results[f"{pop_name}_continuous"] = _run(pop_type, "continuous")
                        elif st.session_state.income_spec_mode == "continuous only":
                            results[f"{pop_name}_continuous"] = _run(pop_type, "continuous")
                        else:  # categorical only
                            results[f"{pop_name}_categorical"] = _run(pop_type, "categorical")
                elif st.session_state.population_mode == "Dependent variable resampling":
                    # Dependent variable mode - only one result regardless of income spec
                    results["depvar"] = _run("depvar", "categorical")  # income mode is ignored
                else:
                    # Single population mode
                    pop_type = "documentation" if "Documentation" in st.session_state.population_mode else "copula"
                    if st.session_state.income_spec_mode == "compare side-by-side":
                        results["categorical"] = _run(pop_type, "categorical")
                        results["continuous"] = _run(pop_type, "continuous")
                    elif st.session_state.income_spec_mode == "continuous only":
                        results["continuous"] = _run(pop_type, "continuous")
                    else:  # categorical only
                        results["categorical"] = _run(pop_type, "categorical")
                
                # Save results if requested
                if st.session_state.save_results:
                    output_dir = Path("outputs")
                    output_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Create decision suffix for filename
                    if len(st.session_state.decision_params.selected_decisions) == len(all_decisions):
                        decision_suffix = "_all"
                    elif len(st.session_state.decision_params.selected_decisions) == 1:
                        decision_suffix = f"_{st.session_state.decision_params.selected_decisions[0]}"
                    else:
                        decision_suffix = f"_{len(st.session_state.decision_params.selected_decisions)}decisions"
                    
                    for mode, df in results.items():
                        filename = f"enhanced_simulation_{mode}_seed{st.session_state.seed if st.session_state.simulation_mode == 'Single Run' else st.session_state.base_seed}_agents{st.session_state.n_agents}{decision_suffix}_{timestamp}.parquet"
                        filepath = output_dir / filename
                        df.to_parquet(filepath, index=False)
                    
                    st.sidebar.caption(f"✅ Results saved with timestamp {timestamp}")
                
                st.session_state.simulation_results = results
                st.session_state.page = 'results'
                st.rerun()
                
        except Exception as e:
            st.caption(f"❌ Simulation failed: {str(e)}")
    
    if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
        run_simulation_from_sidebar()
    
    # Navigation (only show on page2, not on results page)
    if st.session_state.page == 'page2':
        st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.button("← Back to Common Parameters", on_click=go_to_page1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display results on results page
    if st.session_state.page == 'results' and st.session_state.simulation_results is not None:
        # Show parameter summary
        with st.expander("📊 Simulation Parameters Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Time & Market**")
                st.write(f"- Periods: {st.session_state.sim_params.periods}")
                st.write(f"- Duration: {st.session_state.sim_params.duration_hours} hours/period")
                st.write(f"- Vendors: {st.session_state.sim_params.num_vendors}")
                st.write(f"- Market Price: ${st.session_state.sim_params.market_price:.2f}")
            
            with col2:
                st.markdown("**Product & Pricing**")
                st.write(f"- Products/Vendor: {st.session_state.sim_params.products_per_vendor}")
                st.write(f"- Bidding %: {st.session_state.sim_params.bidding_percentage:.0%}")
                st.write(f"- Platform Markup: {st.session_state.sim_params.platform_markup:.0%}")
                st.write(f"- Price Range: ±{st.session_state.sim_params.price_range:.0%}")
            
            with col3:
                st.markdown("**Income & Agents**")
                st.write(f"- Distribution: {st.session_state.sim_params.income_distribution}")
                st.write(f"- Range: ${st.session_state.sim_params.income_min:.0f} - ${st.session_state.sim_params.income_max:.0f}")
                st.write(f"- Agents: {st.session_state.n_agents}")
                st.write(f"- Decisions: {len(st.session_state.decision_params.selected_decisions)}")
        
        results_dict = st.session_state.simulation_results
        
        # Show results based on mode (like original app.py)
        if st.session_state.population_mode == "Compare both":
            st.markdown("### 🔬 Population Mode Comparison")
            
            # Create layout based on income mode
            if st.session_state.income_spec_mode == "compare side-by-side":
                # 2x2 grid: copula vs doc_mode x categorical vs continuous
                st.markdown("#### Copula (Synthetic Agents)")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Categorical Income**")
                    if "copula_categorical" in results_dict:
                        _show_overview(results_dict["copula_categorical"], " (Copula, Cat)")
                    else:
                        st.caption("Categorical results not available")
                with col2:
                    st.markdown("**Continuous Income**")
                    if "copula_continuous" in results_dict:
                        _show_overview(results_dict["copula_continuous"], " (Copula, Cont)")
                    else:
                        st.caption("Continuous results not available")
                
                st.markdown("---")
                st.markdown("#### Documentation Mode (Original + Stochastic)")
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Categorical Income**")
                    if "doc_mode_categorical" in results_dict:
                        _show_overview(results_dict["doc_mode_categorical"], " (Doc, Cat)")
                    else:
                        st.caption("Categorical results not available")
                with col4:
                    st.markdown("**Continuous Income**")
                    if "doc_mode_continuous" in results_dict:
                        _show_overview(results_dict["doc_mode_continuous"], " (Doc, Cont)")
                    else:
                        st.caption("Continuous results not available")
                
                # Default for individual analysis - use first available result
                df = next((results_dict[k] for k in ["copula_categorical", "doc_mode_categorical", "copula_continuous", "doc_mode_continuous"] if k in results_dict), pd.DataFrame())
            else:
                # Single income mode, compare population modes
                income_type = "continuous" if st.session_state.income_spec_mode == "continuous only" else "categorical"
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🧬 Copula (Synthetic)")
                    copula_key = f"copula_{income_type}"
                    if copula_key in results_dict:
                        _show_overview(results_dict[copula_key], f" (Copula, {income_type.title()})")
                    else:
                        st.caption(f"Copula {income_type} results not available")
                
                with col2:
                    st.markdown("#### 📄 Documentation Mode")
                    doc_key = f"doc_mode_{income_type}"
                    if doc_key in results_dict:
                        _show_overview(results_dict[doc_key], f" (Doc, {income_type.title()})")
                    else:
                        st.caption(f"Documentation {income_type} results not available")
                
                # Use first available result for individual analysis
                df = next((results_dict[k] for k in [f"copula_{income_type}", f"doc_mode_{income_type}"] if k in results_dict), pd.DataFrame())
        
        elif st.session_state.population_mode == "Dependent variable resampling":
            # Special display for dependent variable mode
            raw_suffix = " (Raw Pre-truncation)" if st.session_state.raw_draw_mode else ""
            st.markdown(f"### 📊 Dependent Variable Resampling{raw_suffix}")
            if st.session_state.raw_draw_mode:
                st.caption("This mode resamples from the empirical distribution of RAW (pre-truncation) donation rates computed from the original 280 participants. These values represent the Normal(anchor, σ) draw before flooring at 0 and rescaling by personal maximum.")
            else:
                st.caption("This mode resamples from the empirical distribution of donation rates computed from the original 280 participants. No trait information is preserved.")
            
            df = results_dict["depvar"]
            
            # Show comparison of original vs resampled
            try:
                from src.orchestrator_depvar import OrchestratorDepVar
                temp_orch = OrchestratorDepVar()
                # Set the same raw output mode as was used for simulation
                temp_orch.set_raw_output(st.session_state.raw_draw_mode)
                emp_stats = temp_orch.get_empirical_stats()
                original_donations = temp_orch.get_empirical_distribution()
                
                # Determine the column name based on raw mode
                donation_col = 'donation_default_raw' if st.session_state.raw_draw_mode else 'donation_default'
                
                # Create two columns for side-by-side comparison
                col_orig, col_resamp = st.columns(2)
                
                with col_orig:
                    st.subheader("📊 Original 280 Participants")
                    
                    # Stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean", f"{emp_stats['mean']:.1%}")
                        st.metric("Min", f"{emp_stats['min']:.1%}")
                    with col2:
                        st.metric("Std Dev", f"{emp_stats['std']:.4f}")
                        st.metric("Max", f"{emp_stats['max']:.1%}")
                    
                    # Histogram
                    fig_orig = px.histogram(
                        pd.DataFrame({donation_col: original_donations}),
                        x=donation_col,
                        nbins=30,
                        title="Original Distribution (n=280)",
                        labels={donation_col: 'Donation Rate', 'count': 'Number of Participants'},
                        marginal="box"
                    )
                    fig_orig.update_layout(
                        xaxis_tickformat='.0%',
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_orig, use_container_width=True)
                
                with col_resamp:
                    st.subheader(f"📊 Resampled ({len(df):,} agents)")
                    
                    # Stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean", f"{df[donation_col].mean():.1%}")
                        st.metric("Min", f"{df[donation_col].min():.1%}")
                    with col2:
                        st.metric("Std Dev", f"{df[donation_col].std():.4f}")
                        st.metric("Max", f"{df[donation_col].max():.1%}")
                    
                    # Histogram
                    fig_resamp = px.histogram(
                        df,
                        x=donation_col,
                        nbins=30,
                        title=f"Resampled Distribution (n={len(df):,})",
                        labels={donation_col: 'Donation Rate', 'count': 'Number of Agents'},
                        marginal="box"
                    )
                    fig_resamp.update_layout(
                        xaxis_tickformat='.0%',
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_resamp, use_container_width=True)
                
                # Additional info
                st.caption(f"The resampled distribution is created by bootstrap sampling with replacement from the {len(original_donations)} original donation rates.")
                
                # Combined comparison plot
                st.subheader("📊 Distribution Comparison")
                
                # Create combined dataframe for comparison
                orig_df = pd.DataFrame({
                    donation_col: original_donations,
                    'source': 'Original (n=280)'
                })
                resamp_df = pd.DataFrame({
                    donation_col: df[donation_col].values,
                    'source': f'Resampled (n={len(df):,})'
                })
                combined_df = pd.concat([orig_df, resamp_df])
                
                # Create overlaid histogram
                fig_combined = px.histogram(
                    combined_df,
                    x=donation_col,
                    color='source',
                    nbins=30,
                    barmode='overlay',
                    opacity=0.7,
                    title="Original vs Resampled Distribution Overlay",
                    labels={donation_col: 'Donation Rate', 'count': 'Count'}
                )
                fig_combined.update_layout(
                    xaxis_tickformat='.0%',
                    height=400
                )
                st.plotly_chart(fig_combined, use_container_width=True)
                
                # Show unique values info
                st.markdown("### 📊 Distribution Details")
                col1, col2 = st.columns(2)
                with col1:
                    unique_orig = len(np.unique(original_donations))
                    st.metric("Unique values in original", unique_orig)
                    st.caption(f"Maximum possible unique values: {len(original_donations)}")
                with col2:
                    unique_resamp = len(np.unique(df[donation_col]))
                    st.metric("Unique values in resampled", unique_resamp)
                    st.caption(f"Limited by original {unique_orig} unique values")
                
            except Exception as e:
                st.caption(f"Error loading empirical distribution: {e}")
                _show_overview(df)
                
        elif st.session_state.income_spec_mode == "compare side-by-side":
            st.markdown("### 📊 Income Specification Comparison")
            
            col_cat, col_cont = st.columns(2, gap="large")
            
            with col_cat:
                st.markdown("#### 📋 Categorical Income")
                if "categorical" in results_dict:
                    _show_overview(results_dict["categorical"], " (Categorical)")
                else:
                    st.caption("Categorical results not available")
            
            with col_cont:
                st.markdown("#### 📈 Continuous Income") 
                if "continuous" in results_dict:
                    _show_overview(results_dict["continuous"], " (Continuous)")
                else:
                    st.caption("Continuous results not available")
            
            # Use first available for individual agent analysis
            df = next((results_dict[k] for k in ["categorical", "continuous"] if k in results_dict), pd.DataFrame())
        else:
            # Single mode display
            df = next(iter(results_dict.values()))
            mode_name = next(iter(results_dict.keys()))
            _show_overview(df, f" ({mode_name.title()})")
        
        # Individual agent details (like original app.py)
        if st.session_state.show_individual_agents and not df.empty:
            # Check if this is dependent variable mode
            is_depvar_mode = len(df.columns) == 1 and 'donation_default' in df.columns
            
            if not is_depvar_mode:
                st.subheader("🔍 Individual Agent Details")
                
                # Agent selection
                agent_id = st.selectbox(
                    "Select Agent to Examine",
                    options=range(len(df)),
                    format_func=lambda x: f"Agent {x+1}"
                )
                
                agent_data = df.iloc[agent_id]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("** Agent Traits**")
                    trait_data = {}
                    for col in ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program', 
                               'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']:
                        if col in agent_data:
                            trait_data[col] = agent_data[col]
                    
                    trait_df = pd.DataFrame(list(trait_data.items()), columns=['Trait', 'Value'])
                    # Convert all values to strings to avoid PyArrow serialization issues
                    trait_df['Value'] = trait_df['Value'].astype(str)
                    st.dataframe(trait_df, hide_index=True)
                
                with col2:
                    st.markdown("**🎯 Agent Decisions**")
                    decision_data = {}
                    for col in df.columns:
                        if col not in ['Assigned Allowance Level', 'Group_experiment', 'Honesty_Humility', 
                                      'Study Program', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']:
                            decision_data[col] = agent_data[col]
                    
                    decision_df = pd.DataFrame(list(decision_data.items()), columns=['Decision', 'Value'])
                    if 'donation_default' in decision_data:
                        decision_df.loc[decision_df['Decision'] == 'donation_default', 'Value'] = \
                            f"{decision_data['donation_default']:.1%}"
                    if 'donation_default_raw' in decision_data:
                        decision_df.loc[decision_df['Decision'] == 'donation_default_raw', 'Value'] = \
                            f"{decision_data['donation_default_raw']:.1%}"
                    # Convert all values to strings to avoid PyArrow serialization issues
                    decision_df['Value'] = decision_df['Value'].astype(str)
                    st.dataframe(decision_df, hide_index=True)
            else:
                st.caption("Individual agent details not available in dependent variable resampling mode (no trait information)")
        
        # Raw data download (like original app.py)
        if not df.empty:
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"enhanced_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if st.button("🔄 Clear Results"):
                    st.session_state.simulation_results = None
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
    Enhanced AI Agent Simulation Framework | Two-Page Interface
</div>
""", unsafe_allow_html=True)
