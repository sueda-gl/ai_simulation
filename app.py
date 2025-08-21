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

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'mc_results' not in st.session_state:
    st.session_state.mc_results = None

# Sidebar inputs
st.sidebar.subheader("Simulation Parameters")

simulation_mode = st.sidebar.radio(
    "Simulation Mode",
    ["Single Run", "Monte-Carlo Study"],
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
available_decisions = [
    "All Decisions",
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

selected_decision = st.sidebar.selectbox(
    "Decision to Run",
    available_decisions,
    help="Select specific decision or run all 13 decisions"
)

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

# Run simulation button
st.sidebar.markdown("---")

def run_single_simulation():
    """Run a single simulation and return results."""
    try:
        with st.spinner("🔄 Generating synthetic agents and running simulation..."):
            # Initialize orchestrator
            orchestrator = Orchestrator()
            
            # Run simulation
            decision_param = None if selected_decision == "All Decisions" else selected_decision
            results_df = orchestrator.run_simulation(
                n_agents=n_agents,
                seed=seed,
                single_decision=decision_param
            )
            
            # Save results if requested
            if save_results:
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                decision_suffix = f"_{selected_decision}" if selected_decision != "All Decisions" else "_all"
                filename = f"webapp_simulation_seed{seed}_agents{n_agents}{decision_suffix}_{timestamp}.parquet"
                filepath = output_dir / filename
                results_df.to_parquet(filepath, index=False)
                st.sidebar.success(f"✅ Results saved to {filename}")
            
            return results_df
            
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
                '--base-seed', str(base_seed)
            ]
            
            if selected_decision != "All Decisions":
                cmd.extend(['--decision', selected_decision])
            
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

# Main content area
if st.session_state.simulation_results is not None:
    # Single simulation results
    df = st.session_state.simulation_results
    
    # Overview metrics
    st.subheader("📊 Simulation Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", f"{len(df):,}")
    
    with col2:
        trait_cols = ['Assigned Allowance Level', 'Group_experiment', 'Honesty_Humility', 
                     'Study Program', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
        st.metric("Traits Available", len([c for c in trait_cols if c in df.columns]))
    
    with col3:
        decision_cols = [c for c in df.columns if c not in trait_cols]
        st.metric("Decisions Computed", len(decision_cols))
    
    with col4:
        if 'donation_default' in df.columns:
            st.metric("Avg Donation Rate", f"{df['donation_default'].mean():.1%}")
    
    # Donation rate analysis (if available)
    if 'donation_default' in df.columns:
        st.subheader("💰 Donation Rate Analysis")
        
        # Distribution plot
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(
                df, 
                x='donation_default',
                nbins=30,
                title="Distribution of Donation Rates",
                labels={'donation_default': 'Donation Rate', 'count': 'Number of Agents'},
                marginal="box"
            )
            fig.update_layout(
                xaxis_tickformat='.0%',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📈 Statistics**")
            donation_stats = df['donation_default'].describe()
            
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
    

    
    # Individual agent details
    if show_individual_agents:
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
            st.markdown("**🧬 Agent Traits**")
            trait_data = {}
            for col in ['Honesty_Humility', 'Assigned Allowance Level', 'Study Program', 
                       'Group_experiment', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']:
                if col in agent_data:
                    trait_data[col] = agent_data[col]
            
            trait_df = pd.DataFrame(list(trait_data.items()), columns=['Trait', 'Value'])
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
            st.dataframe(decision_df, hide_index=True)
    
    # Raw data download
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