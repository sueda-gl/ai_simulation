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
from src.parameter_applicability import param_manager

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
if 'mc_results' not in st.session_state:
    st.session_state.mc_results = None

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

elif st.session_state.page == 'page2':
    # Page 2: Decision-Specific Parameters
    st.markdown('<h2 class="page-header">Page 2: Decision-Specific Parameters</h2>', unsafe_allow_html=True)
    
    # Decision selection (like in original app.py)
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
                default=st.session_state.decision_params.selected_decisions or [],
                help="Select one or more decisions to run"
            )
        
        # Show message if no decisions selected (but don't force a default)
        if not selected_decisions:
            st.caption("Please select at least one decision")
        
        st.session_state.decision_params.selected_decisions = selected_decisions
        
        # Parameter Applicability Analysis
        if selected_decisions:
            st.markdown('<h3 class="section-header">📋 Parameter Applicability Analysis</h3>', unsafe_allow_html=True)
            
            # Show overall summary
            total_applicable = set()
            total_not_applicable = set()
            
            for decision in selected_decisions:
                applicable = param_manager.get_applicable_parameters(decision)
                not_applicable = param_manager.get_not_applicable_parameters(decision)
                total_applicable.update(applicable)
                total_not_applicable.update(not_applicable)
            
            # Remove overlap (if a parameter is applicable for any decision, consider it applicable)
            total_not_applicable = total_not_applicable - total_applicable
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Parameters", len(param_manager.all_parameters))
            with col2:
                st.metric("✅ Applicable", len(total_applicable))
            with col3:
                st.metric("❌ Not Applicable", len(total_not_applicable))
            
            # Show parameter breakdown by category
            with st.expander("🔍 Parameter Breakdown by Category", expanded=False):
                # Aggregate parameters across all selected decisions
                aggregated_categories = {}
                
                for decision in selected_decisions:
                    categories = param_manager.get_parameters_by_category(decision)
                    for category, params in categories.items():
                        if category not in aggregated_categories:
                            aggregated_categories[category] = {
                                'applicable': set(),
                                'not_applicable': set()
                            }
                        
                        # Add applicable parameters
                        for param_info in params['applicable']:
                            aggregated_categories[category]['applicable'].add(param_info.name)
                        
                        # Add not applicable parameters (but remove if it's applicable in any decision)
                        for param_info in params['not_applicable']:
                            if param_info.name not in total_applicable:
                                aggregated_categories[category]['not_applicable'].add(param_info.name)
                
                # Display the aggregated results
                for category, params in aggregated_categories.items():
                    st.markdown(f"**{category}**")
                    
                    col_app, col_not_app = st.columns(2)
                    
                    with col_app:
                        if params['applicable']:
                            st.markdown("✅ **Applicable:**")
                            for param in sorted(params['applicable']):
                                st.markdown(f"  • {param.replace('_', ' ').title()}")
                        else:
                            st.markdown("✅ **Applicable:** None")
                    
                    with col_not_app:
                        if params['not_applicable']:
                            st.markdown("❌ **Not Applicable:**")
                            for param in sorted(params['not_applicable']):
                                st.markdown(f"  • {param.replace('_', ' ').title()}")
                        else:
                            st.markdown("❌ **Not Applicable:** None")
                    
                    st.markdown("---")
            
            # Show decision-specific analysis
            with st.expander("📊 Decision-Specific Parameter Analysis", expanded=False):
                for decision in selected_decisions:
                    summary = param_manager.get_decision_summary(decision)
                    
                    st.markdown(f"**{decision.replace('_', ' ').title()}**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Applicable", summary['applicable_count'])
                    with col2:
                        st.metric("Not Applicable", summary['not_applicable_count'])
                    with col3:
                        st.metric("Applicability %", f"{summary['applicability_ratio']:.0%}")
                    
                    if summary['reason']:
                        st.caption(f"💡 {summary['reason']}")
                    
                    if summary['applicable_parameters']:
                        st.markdown("✅ **Applicable Parameters:**")
                        applicable_formatted = [p.replace('_', ' ').title() for p in summary['applicable_parameters']]
                        st.markdown(f"  {', '.join(applicable_formatted)}")
                    
                    if summary['not_applicable_parameters']:
                        st.markdown("❌ **Not Applicable Parameters:**")
                        not_applicable_formatted = [p.replace('_', ' ').title() for p in summary['not_applicable_parameters']]
                        st.markdown(f"  {', '.join(not_applicable_formatted)}")
                    
                    st.markdown("---")
    
    # Sidebar configuration (like original app.py) - DYNAMIC BASED ON SELECTED DECISIONS
    st.sidebar.title("⚙️ Decision Parameters")
    
    # DYNAMIC PARAMETERS BASED ON SELECTED DECISIONS
    selected_decisions = st.session_state.decision_params.selected_decisions
    
    # Only show sidebar content if decisions are selected
    if not selected_decisions:
        st.sidebar.info("👈 Select decisions on the main page to see applicable parameters")
        # Store minimal settings in session state for no decisions selected
        st.session_state.n_agents = 1000
        st.session_state.simulation_mode = "Single Run"
        st.session_state.seed = 42
        st.session_state.show_individual_agents = False
        st.session_state.save_results = True
    else:
        # Show decision-specific parameters only for selected decisions
        
        # Only show donation-specific parameters if donation_default is selected
        if "donation_default" in selected_decisions:
            # Population mode selector (like original app.py) - ONLY for donation_default
            st.sidebar.subheader("Population Generation")
            population_mode = st.sidebar.radio(
                "Population Mode",
                ["Copula (synthetic)", "Research Specification", "Compare both"],
                index=0,
                help="Copula: Generate synthetic agents via fitted copula\nResearch: Use original participants with stochastic draws\nCompare both: Show Copula vs Research side-by-side"
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
                    help="When enabled, Copula mode will also use the stochastic component (Normal distribution draw) like Research mode"
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
                # Provide sigma slider for research & compare-both modes as well
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
            # If donation_default is not selected, set default values
            population_mode = "Copula (synthetic)"  # Default population mode
            income_spec_mode = "categorical only"
            sigma_in_copula = False
            sigma_value_ui = 9.0
            anchor_observed_weight = 0.75
            raw_draw_mode = False

        # Show applicable global parameters dynamically
        # Get all applicable parameters across selected decisions
        all_applicable = set()
        for decision in selected_decisions:
            applicable_params = param_manager.get_applicable_parameters(decision)
            all_applicable.update(applicable_params)
        
        # Only show income parameters if they are applicable
        if any(param in all_applicable for param in ['income_distribution', 'income_min', 'income_max', 'income_avg']):
            st.sidebar.subheader("💵 Income Distribution")
            st.sidebar.caption("✅ Applicable for selected decisions")
            
            if 'income_distribution' in all_applicable:
                income_dist_type = st.sidebar.selectbox(
                    "Distribution Type",
                    ["lognormal", "pareto", "weibull"],
                    index=["lognormal", "pareto", "weibull"].index(st.session_state.sim_params.income_distribution),
                    help="✅ Applicable for selected decisions"
                )
                st.session_state.sim_params.income_distribution = income_dist_type
            
            if 'income_min' in all_applicable:
                st.session_state.sim_params.income_min = st.sidebar.number_input(
                    "Minimum Income",
                    min_value=0.0,
                    value=st.session_state.sim_params.income_min,
                    help="✅ Applicable for selected decisions"
                )
            
            if 'income_avg' in all_applicable:
                st.session_state.sim_params.income_avg = st.sidebar.number_input(
                    "Average / Median Income",
                    min_value=st.session_state.sim_params.income_min,
                    value=st.session_state.sim_params.income_avg,
                    help="✅ Applicable for selected decisions"
                )
            
            if 'income_max' in all_applicable:
                st.session_state.sim_params.income_max = st.sidebar.number_input(
                    "Maximum Income",
                    min_value=st.session_state.sim_params.income_avg,
                    value=st.session_state.sim_params.income_max,
                    help="✅ Applicable for selected decisions"
                )

        # Only show market parameters if they are applicable
        market_params = ['num_vendors', 'market_price', 'vendor_price_min', 'vendor_price_max']
        if any(param in all_applicable for param in market_params):
            st.sidebar.subheader("🏪 Market Parameters")
            st.sidebar.caption("✅ Applicable for selected decisions")
            
            if 'num_vendors' in all_applicable:
                st.session_state.sim_params.num_vendors = st.sidebar.number_input(
                    "Number of Vendors",
                    min_value=1,
                    max_value=50,
                    value=st.session_state.sim_params.num_vendors,
                    help="✅ Applicable for selected decisions"
                )
            
            if 'market_price' in all_applicable:
                st.session_state.sim_params.market_price = st.sidebar.number_input(
                    "Average Market Price ($)",
                    min_value=0.01,
                    max_value=1000.0,
                    value=st.session_state.sim_params.market_price,
                    step=0.01,
                    help="✅ Applicable for selected decisions"
                )

        # Only show pricing parameters if they are applicable
        pricing_params = ['platform_markup', 'price_range', 'price_grid', 'bidding_percentage']
        if any(param in all_applicable for param in pricing_params):
            st.sidebar.subheader("💰 Pricing Parameters")
            st.sidebar.caption("✅ Applicable for selected decisions")
            
            if 'platform_markup' in all_applicable:
                st.session_state.sim_params.platform_markup = st.sidebar.slider(
                    "Platform Markup (m)",
                    min_value=0.0,
                    max_value=0.5,
                    value=st.session_state.sim_params.platform_markup,
                    step=0.01,
                    help="✅ Applicable for selected decisions"
                )
            
            if 'bidding_percentage' in all_applicable:
                st.session_state.sim_params.bidding_percentage = st.sidebar.slider(
                    "Bidding Percentage (bp)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.sim_params.bidding_percentage,
                    step=0.05,
                    help="✅ Applicable for selected decisions"
                )
        
        # Simulation parameters (always show if decisions are selected)
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
                value=10,
                step=10,
                help="Number of Monte-Carlo repetitions (Note: 100+ runs can take several minutes)"
            )
            base_seed = st.sidebar.number_input(
                "Base Seed",
                min_value=1,
                max_value=2147483647,
                value=42,
                help="Starting seed (subsequent runs use base_seed + i)"
            )
        
        # Advanced options (always show if decisions are selected)
        st.sidebar.subheader("🔧 Advanced Options")

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
        
        # Summary info
        if st.sidebar.button("📊 Show Parameter Summary"):
            total_params = len(param_manager.all_parameters)
            applicable_count = len(all_applicable)
            st.sidebar.success(f"✅ {applicable_count}/{total_params} parameters applicable ({applicable_count/total_params:.0%})")
    
    # Run simulation button - moved outside the decision selection conditional block
    st.sidebar.markdown("---")
    
    def run_monte_carlo_study():
        """Run Monte-Carlo study and return results."""
        try:
            # Create a container for real-time updates
            status_container = st.container()
            
            st.info(f"🔄 Starting Monte-Carlo study with {st.session_state.n_runs} runs of {st.session_state.n_agents} agents each...")
            
            # Show estimated time
            estimated_time_per_run = 2  # seconds, rough estimate
            total_estimated_time = st.session_state.n_runs * estimated_time_per_run
            st.caption(f"⏱️ Estimated time: ~{total_estimated_time} seconds ({total_estimated_time/60:.1f} minutes)")
            
            # Build command
            cmd = [
                sys.executable, 'scripts/run_mc_study.py',
                '--agents', str(st.session_state.n_agents),
                '--runs', str(st.session_state.n_runs),
                '--base-seed', str(st.session_state.base_seed),
                '--anchor-observed', str(st.session_state.anchor_observed_weight)
            ]
            
            # Handle multiple decisions for Monte Carlo
            if len(st.session_state.decision_params.selected_decisions) < len(all_decisions):
                # Pass each selected decision as a separate argument
                for decision in st.session_state.decision_params.selected_decisions:
                    cmd.extend(['--decision', decision])
            
            # Debug: print command
            with st.expander("🔧 Debug Information", expanded=False):
                st.code(' '.join(cmd))
            
            # Create progress tracking elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            output_container = st.container()
            
            # Change to project directory to ensure scripts can be found
            cwd = Path(__file__).resolve().parent
            
            # Run with real-time output capture using Popen instead of run
            import subprocess
            import time
            
            status_text.text("🚀 Launching Monte-Carlo simulations...")
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd),
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Collect output
            stdout_lines = []
            stderr_lines = []
            last_update_time = time.time()
            
            # Monitor the process
            while True:
                # Check if process is still running
                poll = process.poll()
                
                # Read any available output
                line = process.stdout.readline()
                if line:
                    stdout_lines.append(line.strip())
                    
                    # Parse progress from output
                    if "Run" in line and "/" in line:
                        try:
                            # Extract run number (e.g., "Run  10/100:")
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if "/" in part:
                                    current_run = int(parts[i-1])
                                    total_runs = int(part.split("/")[1].split(":")[0])
                                    progress = current_run / total_runs
                                    progress_bar.progress(progress)
                                    status_text.text(f"🔄 Progress: Run {current_run}/{total_runs}")
                                    break
                        except:
                            pass
                    
                    # Show last few lines of output
                    if time.time() - last_update_time > 0.5:  # Update every 0.5 seconds
                        with output_container.container():
                            st.text("📊 Recent output:")
                            st.code('\n'.join(stdout_lines[-5:]))
                        last_update_time = time.time()
                
                # If process finished, break
                if poll is not None:
                    break
                
                # Small delay to prevent CPU spinning
                time.sleep(0.1)
            
            # Get any remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                stdout_lines.extend(remaining_stdout.strip().split('\n'))
            if remaining_stderr:
                stderr_lines.extend(remaining_stderr.strip().split('\n'))
            
            # Join all output
            stdout = '\n'.join(stdout_lines)
            stderr = '\n'.join(stderr_lines)
            
            # Show final output
            if stdout:
                with st.expander("📋 Monte Carlo Output", expanded=False):
                    st.text(stdout)
            
            if stderr:
                with st.expander("⚠️ Monte Carlo Errors", expanded=True):
                    st.text(stderr)
            
            if process.returncode == 0:
                # Parse output to find result files
                output_lines = stdout.strip().split('\n')
                summary_file = None
                detailed_file = None
                
                for line in output_lines:
                    if 'Summary saved to:' in line:
                        summary_file = line.split('Summary saved to:')[1].strip()
                    elif 'Detailed results saved to:' in line:
                        detailed_file = line.split('Detailed results saved to:')[1].strip()
                
                progress_bar.progress(1.0)
                status_text.success("✅ Monte-Carlo study completed!")
                
                # Load results - handle relative paths
                if summary_file and not Path(summary_file).is_absolute():
                    summary_file = str(cwd / summary_file)
                if detailed_file and not Path(detailed_file).is_absolute():
                    detailed_file = str(cwd / detailed_file)
                
                # Load results
                mc_summary = pd.read_csv(summary_file) if summary_file and Path(summary_file).exists() else None
                mc_detailed = pd.read_csv(detailed_file) if detailed_file and Path(detailed_file).exists() else None
                
                # If files not found, show debug info
                if mc_summary is None and summary_file:
                    st.warning(f"Summary file not found at: {summary_file}")
                if mc_detailed is None and detailed_file:
                    st.warning(f"Detailed file not found at: {detailed_file}")
                
                return mc_summary, mc_detailed, stdout
            else:
                st.error(f"❌ Monte-Carlo study failed with return code: {process.returncode}")
                st.error(f"Error output: {stderr}")
                return None, None, None
                    
        except Exception as e:
            st.error(f"❌ Monte-Carlo study failed: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return None, None, None
    
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
                    pop_type = "documentation" if "Research" in st.session_state.population_mode else "copula"
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
    
    # Run simulation button - moved outside the function
    if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
        # Check if simulation is already running (simple lock mechanism)
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
            
        if st.session_state.simulation_running:
            st.warning("⚠️ A simulation is already running. Please wait for it to complete.")
        else:
            st.session_state.simulation_running = True
            try:
                if st.session_state.simulation_mode == "Single Run":
                    run_simulation_from_sidebar()
                    st.session_state.mc_results = None
                else:
                    mc_summary, mc_detailed, output_log = run_monte_carlo_study()
                    if mc_summary is not None:
                        st.session_state.mc_results = {
                            'summary': mc_summary,
                            'detailed': mc_detailed,
                            'log': output_log
                        }
                        st.session_state.simulation_results = None
                        st.session_state.page = 'results'
                        st.success("✅ Monte Carlo results saved to session state. Redirecting to results page...")
                        st.rerun()
                    else:
                        st.error("❌ Monte Carlo simulation returned no results")
            finally:
                st.session_state.simulation_running = False
    
    # Navigation
    st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.button("← Back to Common Parameters", on_click=go_to_page1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'results':
    # Results page
    st.markdown('<h2 class="page-header">Simulation Results</h2>', unsafe_allow_html=True)
    
    # Debug info
    with st.expander("🔧 Debug: Session State", expanded=False):
        st.write(f"simulation_results: {'Yes' if st.session_state.simulation_results is not None else 'No'}")
        st.write(f"mc_results: {'Yes' if st.session_state.mc_results is not None else 'No'}")
        if st.session_state.mc_results is not None:
            st.write(f"mc_results keys: {list(st.session_state.mc_results.keys())}")
            st.write(f"summary shape: {st.session_state.mc_results['summary'].shape if st.session_state.mc_results['summary'] is not None else 'None'}")
            st.write(f"detailed shape: {st.session_state.mc_results['detailed'].shape if st.session_state.mc_results['detailed'] is not None else 'None'}")
    
    # Display single run results
    if st.session_state.simulation_results is not None:
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
        
        # Show parameter applicability summary for the run decisions
        with st.expander("📋 Parameter Applicability Summary for This Run", expanded=False):
            selected_decisions = st.session_state.decision_params.selected_decisions
            
            if selected_decisions:
                # Calculate overall applicability
                total_applicable = set()
                total_not_applicable = set()
                
                for decision in selected_decisions:
                    applicable = param_manager.get_applicable_parameters(decision)
                    not_applicable = param_manager.get_not_applicable_parameters(decision)
                    total_applicable.update(applicable)
                    total_not_applicable.update(not_applicable)
                
                # Remove overlap
                total_not_applicable = total_not_applicable - total_applicable
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total Parameters", len(param_manager.all_parameters))
                with col2:
                    st.metric("✅ Applicable", len(total_applicable))
                with col3:
                    st.metric("❌ Not Applicable", len(total_not_applicable))
                with col4:
                    applicability_pct = len(total_applicable) / len(param_manager.all_parameters) * 100 if param_manager.all_parameters else 0
                    st.metric("📈 Efficiency", f"{applicability_pct:.0f}%")
                
                # Show which parameters were actually used vs unused
                col_used, col_unused = st.columns(2)
                
                with col_used:
                    st.markdown("### ✅ Parameters Used in This Simulation")
                    if total_applicable:
                        # Group by category
                        used_by_category = {}
                        for param in total_applicable:
                            category = param_manager._get_parameter_category(param)
                            if category not in used_by_category:
                                used_by_category[category] = []
                            used_by_category[category].append(param)
                        
                        for category, params in sorted(used_by_category.items()):
                            st.markdown(f"**{category}:**")
                            for param in sorted(params):
                                st.markdown(f"  • {param.replace('_', ' ').title()}")
                    else:
                        st.caption("No parameters were applicable for the selected decisions.")
                
                with col_unused:
                    st.markdown("### ❌ Parameters Not Used in This Simulation")
                    if total_not_applicable:
                        # Group by category
                        unused_by_category = {}
                        for param in total_not_applicable:
                            category = param_manager._get_parameter_category(param)
                            if category not in unused_by_category:
                                unused_by_category[category] = []
                            unused_by_category[category].append(param)
                        
                        for category, params in sorted(unused_by_category.items()):
                            st.markdown(f"**{category}:**")
                            for param in sorted(params):
                                st.markdown(f"  • {param.replace('_', ' ').title()}")
                    else:
                        st.caption("All parameters were used in this simulation.")
                
                # Show decision-specific breakdown
                st.markdown("### 📊 Parameter Usage by Decision")
                
                for decision in selected_decisions:
                    summary = param_manager.get_decision_summary(decision)
                    
                    with st.container():
                        col_title, col_metrics = st.columns([2, 3])
                        
                        with col_title:
                            st.markdown(f"**{decision.replace('_', ' ').title()}**")
                            if summary['reason']:
                                st.caption(f"💡 {summary['reason']}")
                        
                        with col_metrics:
                            sub_col1, sub_col2, sub_col3 = st.columns(3)
                            with sub_col1:
                                st.metric("Applicable", summary['applicable_count'], label_visibility="collapsed")
                            with sub_col2:
                                st.metric("Not Applicable", summary['not_applicable_count'], label_visibility="collapsed")
                            with sub_col3:
                                st.metric("Efficiency", f"{summary['applicability_ratio']:.0%}", label_visibility="collapsed")
                        
                        st.markdown("---")
            else:
                st.caption("No decisions were selected for this simulation.")
        
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
                st.markdown("#### Research Mode (Original + Stochastic)")
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Categorical Income**")
                    if "doc_mode_categorical" in results_dict:
                        _show_overview(results_dict["doc_mode_categorical"], " (Research, Cat)")
                    else:
                        st.caption("Categorical results not available")
                with col4:
                    st.markdown("**Continuous Income**")
                    if "doc_mode_continuous" in results_dict:
                        _show_overview(results_dict["doc_mode_continuous"], " (Research, Cont)")
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
                    st.markdown("#### 📄 Research Mode")
                    doc_key = f"doc_mode_{income_type}"
                    if doc_key in results_dict:
                        _show_overview(results_dict[doc_key], f" (Research, {income_type.title()})")
                    else:
                        st.caption(f"Research {income_type} results not available")
                
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
    
    # Display Monte Carlo results
    elif st.session_state.mc_results is not None:
        mc_data = st.session_state.mc_results
        
        if mc_data['summary'] is not None:
            st.subheader("📈 Monte-Carlo Analysis Results")
            
            summary_df = mc_data['summary']
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
        
            if 'donation_default' in summary_df['decision'].values:
                donation_row = summary_df[summary_df['decision'] == 'donation_default'].iloc[0]
            
                with col1:
                    st.metric("Mean Donation Rate", f"{donation_row['mean']:.1%}")
                
                with col2:
                    st.metric("Standard Deviation", f"{donation_row['std']:.4f}")
                
                with col3:
                    st.metric("95% CI Lower", f"{donation_row['p2.5']:.1%}")
                
                with col4:
                    st.metric("95% CI Upper", f"{donation_row['p97.5']:.1%}")
        
            # Monte-Carlo convergence plot
            if mc_data['detailed'] is not None:
                detailed_df = mc_data['detailed']
                
                if 'donation_default_mean' in detailed_df.columns:
                    st.subheader("📊 Monte-Carlo Convergence")
                    
                    # Calculate running average
                    detailed_df['running_mean'] = detailed_df['donation_default_mean'].expanding().mean()
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Individual Run Results", "Running Average Convergence"),
                        vertical_spacing=0.1
                    )
                    
                    # Individual runs
                    fig.add_trace(
                        go.Scatter(
                            x=detailed_df['run'] + 1,
                            y=detailed_df['donation_default_mean'],
                            mode='markers+lines',
                            name='Individual Runs',
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
                    fig.update_yaxes(tickformat='.1%')
                    fig.update_xaxes(title="Run Number")
                    
                    st.plotly_chart(fig, use_container_width=True)
        
            # Summary statistics table
            st.subheader("📋 Summary Statistics")
        
            # Format the summary table for display
            display_summary = summary_df.copy()
            for col in ['mean', 'p2.5', 'p97.5']:
                if col in display_summary.columns:
                    display_summary[col] = display_summary[col].apply(lambda x: f"{x:.1%}")
            
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
    
    # Show message if no results available
    else:
        st.info("🔍 No simulation results available yet.")
        st.write("Please configure your simulation parameters and click '🚀 Run Simulation' in the sidebar.")
        if st.button("← Back to Decision Parameters", type="primary"):
            st.session_state.page = 'page2'
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
    Enhanced AI Agent Simulation Framework | Two-Page Interface
</div>
""", unsafe_allow_html=True)
