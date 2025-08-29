# AI Agent Simulation Dashboard
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.orchestrator import Orchestrator
from src.trait_engine import TraitEngine

# Page configuration
st.set_page_config(
    page_title="AI Agent Simulation Dashboard",
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
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
}
.stAlert {
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">AI Agent Simulation Dashboard</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ Simulation Controls")

# Initialize session state - fix KeyError issues
_DEFAULTS = {
    "simulation_mode": "Single Run",
    "population_mode": "Copula (synthetic)",
    "income_spec_mode": "categorical only",
    "simulation_results": None,
    "mc_results": None,
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

# Sidebar inputs
st.sidebar.subheader("Simulation Parameters")

simulation_mode = st.sidebar.radio(
    "Simulation Mode",
    ["Single Run", "Monte-Carlo Study"],
    index=0 if st.session_state.simulation_mode == "Single Run" else 1,
    key="simulation_mode_radio",
    help="Single Run: One simulation with specified parameters\nMonte-Carlo: Multiple runs for uncertainty analysis"
)

n_agents = st.sidebar.number_input(
    "Number of Agents",
    min_value=10,
    max_value=50000,
    value=1000,
    step=100,
    help="Number of synthetic agents to generate"
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

# Decision selection
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

# Multi-select with "Select All" functionality
st.sidebar.subheader("Decision Selection")
select_all = st.sidebar.checkbox("Select All Decisions", value=True)

if select_all:
    selected_decisions = st.sidebar.multiselect(
        "Decisions to Run",
        all_decisions,
        default=all_decisions,
        help="Select one or more decisions to run",
        disabled=True
    )
else:
    selected_decisions = st.sidebar.multiselect(
        "Decisions to Run",
        all_decisions,
        default=["donation_default"],
        help="Select one or more decisions to run"
    )

# Ensure at least one decision is selected
if not selected_decisions:
    st.sidebar.error("Please select at least one decision")
    selected_decisions = ["donation_default"]

# Advanced options
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

# Population mode selector
st.sidebar.subheader("Population Generation")
population_mode = st.sidebar.radio(
    "Population Mode",
    ["Copula (synthetic)", "Documentation (original + stochastic)", "Compare both"],
    index=0,
    help="Copula: Generate synthetic agents via fitted copula\nDocumentation: Use original participants with stochastic draws\nCompare both: Show Copula vs Documentation side-by-side"
)

# Income specification selector with comparison mode (not relevant for dependent variable mode)
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

# Stochastic component option for copula mode (not relevant for dependent variable mode)
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

# Anchor weights slider (not for dependent variable mode which uses pre-computed values)
if population_mode != "Dependent variable resampling":
    st.sidebar.subheader("Anchor Mix")
    anchor_observed_weight = st.sidebar.slider(
        "Weight on OBSERVED prosocial score",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Anchor = w × Observed + (1-w) × Predicted. Default is 0.75 observed + 0.25 predicted."
    )
    st.sidebar.caption(f"Predicted weight: {1 - anchor_observed_weight:.2f}")
else:
    anchor_observed_weight = 0.75  # Default value used in pre-computation

# Raw output option
if population_mode != "Copula (synthetic)" or sigma_in_copula:
    st.sidebar.subheader("Output Options")
    raw_draw_mode = st.sidebar.checkbox(
        "Show pre-truncation (raw) donation rate",
        value=False,
        help="Display the raw Normal(anchor, σ) draw before any processing. This shows negative values and the full range of the stochastic draw before flooring at 0 and rescaling by personal maximum."
    )
else:
    raw_draw_mode = False  # Not applicable for deterministic copula mode

# Run simulation button
st.sidebar.markdown("---")

def run_single_simulation():
    """Run a single simulation and return results."""
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
                    orchestrator.set_raw_output(raw_draw_mode)
                else:  # copula
                    orchestrator = Orchestrator()
                
                # Override income specification in config based on choice (not for depvar mode)
                if hasattr(orchestrator, 'config') and 'donation_default' in orchestrator.config:
                    if pop_mode != "depvar":  # depvar mode doesn't use these settings
                        orchestrator.config['donation_default']['regression']['income_mode'] = inc_mode
                        # Set stochastic flag for copula mode if checkbox is enabled
                        if pop_mode == "copula":
                            orchestrator.config['donation_default']['stochastic']['in_copula'] = sigma_in_copula
                        # Apply selected sigma value
                        orchestrator.config['donation_default']['stochastic']['sigma_value'] = sigma_value_ui
                        # Apply chosen anchor weights
                        orchestrator.config['donation_default']['anchor_weights']['observed'] = anchor_observed_weight
                        orchestrator.config['donation_default']['anchor_weights']['predicted'] = 1 - anchor_observed_weight
                        # Set raw output flag if applicable
                        if pop_mode == "documentation" or (pop_mode == "copula" and sigma_in_copula):
                            orchestrator.config['donation_default']['stochastic']['raw_output'] = raw_draw_mode
                
                # Handle multiple decisions
                decision_param = None if len(selected_decisions) == len(all_decisions) else selected_decisions
                return orchestrator.run_simulation(
                    n_agents=n_agents,
                    seed=seed,
                    single_decision=decision_param
                )
            
            # Run based on population and income specification modes
            results = {}
            
            if population_mode == "Compare both":
                # Compare population modes
                for pop_name, pop_type in [("copula", "copula"), ("doc_mode", "documentation")]:
                    if income_spec_mode == "compare side-by-side":
                        results[f"{pop_name}_categorical"] = _run(pop_type, "categorical")
                        results[f"{pop_name}_continuous"] = _run(pop_type, "continuous")
                    elif income_spec_mode == "continuous only":
                        results[f"{pop_name}_continuous"] = _run(pop_type, "continuous")
                    else:  # categorical only
                        results[f"{pop_name}_categorical"] = _run(pop_type, "categorical")
            elif population_mode == "Dependent variable resampling":
                # Dependent variable mode - only one result regardless of income spec
                results["depvar"] = _run("depvar", "categorical")  # income mode is ignored
            else:
                # Single population mode
                pop_type = "documentation" if "Documentation" in population_mode else "copula"
                if income_spec_mode == "compare side-by-side":
                    results["categorical"] = _run(pop_type, "categorical")
                    results["continuous"] = _run(pop_type, "continuous")
                elif income_spec_mode == "continuous only":
                    results["continuous"] = _run(pop_type, "continuous")
                else:  # categorical only
                    results["categorical"] = _run(pop_type, "categorical")
            
            # Save results if requested
            if save_results:
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Create decision suffix for filename
                if len(selected_decisions) == len(all_decisions):
                    decision_suffix = "_all"
                elif len(selected_decisions) == 1:
                    decision_suffix = f"_{selected_decisions[0]}"
                else:
                    decision_suffix = f"_{len(selected_decisions)}decisions"
                
                for mode, df in results.items():
                    filename = f"webapp_simulation_{mode}_seed{seed}_agents{n_agents}{decision_suffix}_{timestamp}.parquet"
                    filepath = output_dir / filename
                    df.to_parquet(filepath, index=False)
                
                st.sidebar.success(f"✅ Results saved with timestamp {timestamp}")
            
            return results
            
    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        return None

def run_monte_carlo_study():
    """Run Monte-Carlo study and return results."""
    try:
        with st.spinner(f"🔄 Running {n_runs} Monte-Carlo simulations..."):
            # Build command
            cmd = [
                sys.executable, 'scripts/run_mc_study.py',
                '--agents', str(n_agents),
                '--runs', str(n_runs),
                '--base-seed', str(base_seed),
                '--anchor-observed', str(anchor_observed_weight)
            ]
            
            # Handle multiple decisions for Monte Carlo
            if len(selected_decisions) < len(all_decisions):
                # Pass each selected decision as a separate argument
                for decision in selected_decisions:
                    cmd.extend(['--decision', decision])
            
            # Run Monte-Carlo study
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse output to find result files
                output_lines = result.stdout.strip().split('\n')
                summary_file = None
                detailed_file = None
                
                for line in output_lines:
                    if 'Summary saved to:' in line:
                        summary_file = line.split('Summary saved to:')[1].strip()
                    elif 'Detailed results saved to:' in line:
                        detailed_file = line.split('Detailed results saved to:')[1].strip()
                
                progress_bar.progress(1.0)
                status_text.success("✅ Monte-Carlo study completed!")
                
                # Load results
                mc_summary = pd.read_csv(summary_file) if summary_file else None
                mc_detailed = pd.read_csv(detailed_file) if detailed_file else None
                
                return mc_summary, mc_detailed, result.stdout
            else:
                st.error(f"❌ Monte-Carlo study failed: {result.stderr}")
                return None, None, None
                
    except Exception as e:
        st.error(f"❌ Monte-Carlo study failed: {str(e)}")
        return None, None, None

if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
    if simulation_mode == "Single Run":
        st.session_state.simulation_results = run_single_simulation()
        st.session_state.mc_results = None
    else:
        mc_summary, mc_detailed, output_log = run_monte_carlo_study()
        st.session_state.mc_results = {
            'summary': mc_summary,
            'detailed': mc_detailed,
            'log': output_log
        }
        st.session_state.simulation_results = None

def _show_overview(df, title_suffix=""):
    """Helper function to show simulation overview for a DataFrame"""
    st.subheader(f"Simulation Overview{title_suffix}")
    
    # Check if this is dependent variable mode (only has donation_default column)
    is_depvar_mode = len(df.columns) == 1 and 'donation_default' in df.columns
    
    # Display anchor weights info (not for depvar mode)
    if not is_depvar_mode:
        st.caption(f"📊 Anchor mix: {anchor_observed_weight:.2f} observed | {1 - anchor_observed_weight:.2f} predicted")
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

# Main content area
if st.session_state.simulation_results is not None:
    results_dict = st.session_state.simulation_results
    
    # Show results based on mode
    if population_mode == "Compare both":
        st.markdown("### 🔬 Population Mode Comparison")
        
        # Create layout based on income mode
        if income_spec_mode == "compare side-by-side":
            # 2x2 grid: copula vs doc_mode x categorical vs continuous
            st.markdown("#### Copula (Synthetic Agents)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Categorical Income**")
                if "copula_categorical" in results_dict:
                    _show_overview(results_dict["copula_categorical"], " (Copula, Cat)")
                else:
                    st.info("Categorical results not available")
            with col2:
                st.markdown("**Continuous Income**")
                if "copula_continuous" in results_dict:
                    _show_overview(results_dict["copula_continuous"], " (Copula, Cont)")
                else:
                    st.info("Continuous results not available")
            
            st.markdown("---")
            st.markdown("#### Documentation Mode (Original + Stochastic)")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Categorical Income**")
                if "doc_mode_categorical" in results_dict:
                    _show_overview(results_dict["doc_mode_categorical"], " (Doc, Cat)")
                else:
                    st.info("Categorical results not available")
            with col4:
                st.markdown("**Continuous Income**")
                if "doc_mode_continuous" in results_dict:
                    _show_overview(results_dict["doc_mode_continuous"], " (Doc, Cont)")
                else:
                    st.info("Continuous results not available")
            
            # Default for individual analysis - use first available result
            df = next((results_dict[k] for k in ["copula_categorical", "doc_mode_categorical", "copula_continuous", "doc_mode_continuous"] if k in results_dict), pd.DataFrame())
        else:
            # Single income mode, compare population modes
            income_type = "continuous" if income_spec_mode == "continuous only" else "categorical"
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🧬 Copula (Synthetic)")
                copula_key = f"copula_{income_type}"
                if copula_key in results_dict:
                    _show_overview(results_dict[copula_key], f" (Copula, {income_type.title()})")
                else:
                    st.info(f"Copula {income_type} results not available")
            
            with col2:
                st.markdown("#### 📄 Documentation Mode")
                doc_key = f"doc_mode_{income_type}"
                if doc_key in results_dict:
                    _show_overview(results_dict[doc_key], f" (Doc, {income_type.title()})")
                else:
                    st.info(f"Documentation {income_type} results not available")
            
            # Use first available result for individual analysis
            df = next((results_dict[k] for k in [f"copula_{income_type}", f"doc_mode_{income_type}"] if k in results_dict), pd.DataFrame())
    
    elif population_mode == "Dependent variable resampling":
        # Special display for dependent variable mode
        raw_suffix = " (Raw Pre-truncation)" if raw_draw_mode else ""
        st.markdown(f"### 📊 Dependent Variable Resampling{raw_suffix}")
        if raw_draw_mode:
            st.info("This mode resamples from the empirical distribution of RAW (pre-truncation) donation rates computed from the original 280 participants. These values represent the Normal(anchor, σ) draw before flooring at 0 and rescaling by personal maximum.")
        else:
            st.info("This mode resamples from the empirical distribution of donation rates computed from the original 280 participants. No trait information is preserved.")
        
        df = results_dict["depvar"]
        
        # Show comparison of original vs resampled
        try:
            from src.orchestrator_depvar import OrchestratorDepVar
            temp_orch = OrchestratorDepVar()
            # Set the same raw output mode as was used for simulation
            temp_orch.set_raw_output(raw_draw_mode)
            emp_stats = temp_orch.get_empirical_stats()
            original_donations = temp_orch.get_empirical_distribution()
            
            # Determine the column name based on raw mode
            donation_col = 'donation_default_raw' if raw_draw_mode else 'donation_default'
            
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
            st.info(f"The resampled distribution is created by bootstrap sampling with replacement from the {len(original_donations)} original donation rates.")
            
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
            st.error(f"Error loading empirical distribution: {e}")
            _show_overview(df)
            
    elif income_spec_mode == "compare side-by-side":
        st.markdown("### 📊 Income Specification Comparison")
        
        col_cat, col_cont = st.columns(2, gap="large")
        
        with col_cat:
            st.markdown("#### 📋 Categorical Income")
            if "categorical" in results_dict:
                _show_overview(results_dict["categorical"], " (Categorical)")
            else:
                st.info("Categorical results not available")
        
        with col_cont:
            st.markdown("#### 📈 Continuous Income") 
            if "continuous" in results_dict:
                _show_overview(results_dict["continuous"], " (Continuous)")
            else:
                st.info("Continuous results not available")
        
        # Use first available for individual agent analysis
        df = next((results_dict[k] for k in ["categorical", "continuous"] if k in results_dict), pd.DataFrame())
    else:
        # Single mode display
        df = next(iter(results_dict.values()))
        mode_name = next(iter(results_dict.keys()))
        _show_overview(df, f" ({mode_name.title()})")
    

    
    # Individual agent details
    if show_individual_agents and not df.empty:
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
            st.info("Individual agent details not available in dependent variable resampling mode (no trait information)")
    
    # Raw data download
    if not df.empty:
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("🔄 Clear Results"):
                st.session_state.simulation_results = None
                st.rerun()

elif st.session_state.mc_results is not None:
    # Monte-Carlo results
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
        
        display_summary.columns = ['Decision', 'Mean', 'Std Dev', '2.5%ile', '97.5%ile', 'Runs', 'Agents/Run']
        st.dataframe(display_summary, hide_index=True)
        
        # Results summary
        if 'donation_default' in summary_df['decision'].values:
            donation_row = summary_df[summary_df['decision'] == 'donation_default'].iloc[0]
            
            st.subheader("📊 Results Summary")
            summary_text = f"""
            **Monte-Carlo Analysis Results:**
            
            Across {donation_row['runs']} simulation runs with {donation_row['agents_per_run']:,} agents each, 
            the average donation rate was {donation_row['mean']:.1%} with a 95% confidence interval of 
            [{donation_row['p2.5']:.1%}, {donation_row['p97.5']:.1%}].
            """
            st.markdown(summary_text)
    
    # Download Monte-Carlo results
    st.subheader("💾 Export Monte-Carlo Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if mc_data['summary'] is not None:
            csv_data = mc_data['summary'].to_csv(index=False)
            st.download_button(
                label="📥 Download Summary CSV",
                data=csv_data,
                file_name=f"mc_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔄 Clear Results"):
            st.session_state.mc_results = None
            st.rerun()

else:
    # No welcome screen - just show system status
    pass
    
    # Instructions
    st.subheader("Getting Started")
    
    st.markdown("""
    **1. Set Parameters**
    - Choose number of agents in the sidebar
    - Select simulation mode (Single Run or Monte-Carlo Study)
    - Pick which decisions to run
    
    **2. Run Simulation**
    - Click "Run Simulation" button
    - View results and interactive charts
    - Download data for further analysis
    
    **3. Analyze Results**
    - Examine donation rate distributions
    - Inspect individual agent profiles
    - Export results as CSV files
    """)
    
    # System status (minimal)
    try:
        # Check if copula model exists
        model_path = Path("config/trait_model.pkl")
        if model_path.exists():
            st.success("✅ System ready")
        else:
            st.warning("⚠️ Copula model not found. Run `python scripts/train_copula.py` first.")
            
    except Exception as e:
        st.error(f"❌ System error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
     AI Agent Simulation Framework | Built with Streamlit
</div>
""", unsafe_allow_html=True)