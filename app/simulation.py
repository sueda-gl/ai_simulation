# app/simulation.py
"""
Simulation execution logic for the Enhanced AI Agent Simulation.
Handles both single runs and Monte Carlo studies.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator import Orchestrator
from src.orchestrator_doc_mode import OrchestratorDocMode
from src.orchestrator_depvar import OrchestratorDepVar
from src.orchestrator_baseline import OrchestratorBaseline
from src.trait_engine import TraitEngine
from app.models import ALL_DECISIONS


# =============================================================================
# HELPER FUNCTIONS - Shared logic for mode runners
# =============================================================================

def _assign_global_transaction_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign unique, chronologically ordered transaction IDs to all purchase requests across all agents.
    
    This ensures that Transaction IDs are consistent across different exports (Decision 6, Decision 9).
    
    Logic:
    1. Collect all requests from all agents.
    2. Sort globally by timestamp_hours.
    3. Assign sequential IDs (1, 2, 3...).
    4. Write IDs back to the agent's purchase_requests structure.
    """
    if 'purchase_requests' not in df.columns:
        return df
    
    # 1. Collect all requests with metadata to trace back
    # We need a flat list of (timestamp, agent_idx, req_idx, request_obj)
    all_requests = []
    
    for idx, row in df.iterrows():
        requests = row.get('purchase_requests', [])
        if isinstance(requests, list):
            for req_idx, req in enumerate(requests):
                if isinstance(req, dict):
                    # Store reference to the mutable dict object so we can update it in place
                    all_requests.append({
                        'timestamp': req.get('timestamp_hours', 0),
                        'request_obj': req
                    })
    
    # 2. Sort globally by timestamp
    # Use a stable sort to ensure determinism for identical timestamps
    all_requests.sort(key=lambda x: x['timestamp'])
    
    # 3. Assign sequential IDs
    for i, item in enumerate(all_requests):
        transaction_id = i + 1
        
        # Update the request object IN PLACE
        # This updates the dict inside the list inside the DataFrame
        item['request_obj']['transaction_id'] = transaction_id
        
    return df

def _load_original_participants(n_agents: int, seed: int, random_sample: bool = True) -> pd.DataFrame:
    """
    Load original 280 participants with configurable sampling.
    
    Args:
        n_agents: Number of agents to load
        seed: Random seed for reproducibility
        random_sample: 
            True (Research Spec) → Random sampling for n≤280
            False (Research Baseline) → Sequential selection for n≤280
    
    Behavior:
        n_agents > 280:  Both modes → Bootstrap (random WITH replacement)
        n_agents <= 280:
            random_sample=True  → Random WITHOUT replacement (Research Spec)
            random_sample=False → Sequential [0, 1, 2, ..., n-1] (Research Baseline)
    """
    temp_orchestrator = OrchestratorBaseline()
    n_original = len(temp_orchestrator.original_data)
    rng = np.random.default_rng(seed)
    
    if n_agents <= n_original:
        if random_sample:
            # Research Spec: Random sample WITHOUT replacement
            indices = rng.choice(n_original, size=n_agents, replace=False)
        else:
            # Research Baseline: Sequential/in-order selection
            indices = list(range(n_agents))
        df = temp_orchestrator.original_data.iloc[indices].copy()
        df.index = range(len(df))
        return df
    else:
        # n_agents > 280
        if random_sample:
            # Research Spec: Bootstrap sample WITH replacement
            indices = rng.choice(n_original, size=n_agents, replace=True)
        else:
            # Research Baseline: "Always in order" -> Deterministic wrapping
            # [0, 1, ..., 279, 0, 1, ..., 279, 0, 1, ...]
            full_repeats = n_agents // n_original
            remainder = n_agents % n_original
            indices = list(range(n_original)) * full_repeats + list(range(remainder))
            
        df = temp_orchestrator.original_data.iloc[indices].copy()
        df.index = range(len(df))
        return df


def _apply_simulation_params(orchestrator):
    """
    Apply Page 1 simulation parameters to orchestrator.
    
    This ensures user-configured values from Page 1 take precedence over config/simulation.yaml.
    """
    if not hasattr(orchestrator, 'simulation_config'):
        return
    
    if 'simulation' not in orchestrator.simulation_config:
        orchestrator.simulation_config['simulation'] = {}
    
    sim_params = st.session_state.sim_params
    sim_config = orchestrator.simulation_config['simulation']
    
    # Income distribution parameters - CRITICAL for disclose_documents eligibility
    sim_config['income_distribution'] = sim_params.income_distribution
    sim_config['discount_income_threshold'] = sim_params.discount_income_threshold
    
    # Lognormal parameters
    sim_config['lognormal_mu'] = sim_params.lognormal_mu
    sim_config['lognormal_sigma'] = sim_params.lognormal_sigma
    sim_config['lognormal_min'] = sim_params.lognormal_min
    sim_config['lognormal_max'] = sim_params.lognormal_max
    
    # Generalised Gamma parameters
    sim_config['gg_k'] = sim_params.gg_k
    sim_config['gg_c'] = sim_params.gg_c
    sim_config['gg_lambda'] = sim_params.gg_lambda
    sim_config['gg_min'] = sim_params.gg_min
    sim_config['gg_max'] = sim_params.gg_max
    
    # Dagum parameters
    sim_config['dagum_a'] = sim_params.dagum_a
    sim_config['dagum_p'] = sim_params.dagum_p
    sim_config['dagum_b'] = sim_params.dagum_b
    sim_config['dagum_min'] = sim_params.dagum_min
    sim_config['dagum_max'] = sim_params.dagum_max
    
    # Market parameters - used by bid_value and other decisions
    sim_config['market_price'] = sim_params.market_price
    sim_config['platform_markup'] = sim_params.platform_markup
    sim_config['price_range'] = sim_params.price_range
    sim_config['bidding_percentage'] = sim_params.bidding_percentage
    sim_config['num_vendors'] = sim_params.num_vendors
    
    # Vendor configuration parameters
    sim_config['vendor_config_mode'] = sim_params.vendor_config_mode
    sim_config['vendor_price_source'] = sim_params.vendor_price_source
    sim_config['vendor_price_min'] = sim_params.vendor_price_min
    sim_config['vendor_price_max'] = sim_params.vendor_price_max
    sim_config['vendor_products_min'] = sim_params.vendor_products_min
    sim_config['vendor_products_max'] = sim_params.vendor_products_max
    sim_config['vendor_products_avg'] = sim_params.vendor_products_avg
    
    # Vendor carryover parameters
    sim_config['vendor_carryover_probability'] = sim_params.vendor_carryover_probability
    sim_config['override_carryover'] = sim_params.override_carryover
    sim_config['global_carryover'] = sim_params.global_carryover
    
    # Vendor configuration data (if uploaded via CSV)
    if hasattr(sim_params, 'vendor_config_data') and sim_params.vendor_config_data is not None:
        sim_config['vendor_config_data'] = sim_params.vendor_config_data
    
    # Legacy vendor parameters (for backward compatibility)
    sim_config['products_per_vendor'] = sim_params.products_per_vendor
    sim_config['carryover'] = sim_params.carryover
    if hasattr(sim_params, 'vendor_prices') and sim_params.vendor_prices is not None:
        sim_config['vendor_prices'] = sim_params.vendor_prices
    
    # Time parameters
    sim_config['periods'] = sim_params.periods
    sim_config['duration_hours'] = sim_params.duration_hours
    
    # Income categories
    sim_config['num_discount_categories'] = sim_params.num_discount_categories
    sim_config['num_fixed_categories'] = sim_params.num_fixed_categories
    
    # Consumption parameters
    sim_config['max_purchases_per_term'] = sim_params.max_purchases_per_term


def _apply_decision_settings(orchestrator, decision_settings: dict):
    """
    Apply decision settings (random probabilities, defaults, etc.) to orchestrator.
    """
    if decision_settings:
        if hasattr(orchestrator, 'simulation_config'):
            orchestrator.simulation_config['random_decisions'] = decision_settings
            orchestrator.simulation_config['default_decisions'] = decision_settings
        else:
            orchestrator.simulation_config = {
                'random_decisions': decision_settings,
                'default_decisions': decision_settings
            }
    
    # Pass purchasing limits if enabled
    if st.session_state.sim_params.apply_purchasing_limits:
        if hasattr(orchestrator, 'simulation_config'):
            orchestrator.simulation_config['purchasing_limits'] = st.session_state.sim_params.purchasing_limits
        else:
            orchestrator.simulation_config = {
                'purchasing_limits': st.session_state.sim_params.purchasing_limits
            }
    
    # Pass information about custom vs default decisions
    if hasattr(st.session_state, 'custom_decisions') and hasattr(st.session_state, 'default_decisions'):
        if not hasattr(orchestrator, 'simulation_config'):
            orchestrator.simulation_config = {}
        orchestrator.simulation_config['custom_decisions'] = st.session_state.custom_decisions
        orchestrator.simulation_config['default_decisions_list'] = st.session_state.default_decisions


def _apply_donation_config(orchestrator, pop_mode: str, inc_mode: str):
    """
    Apply donation-specific configuration to orchestrator.
    
    Handles income mode, stochastic settings, anchor weights, and coefficients.
    
    Note: If a saved donation configuration exists, its income mode takes precedence
    over the passed inc_mode parameter. This ensures donation_default uses its
    own configured income mode independently of other decisions.
    """
    if not hasattr(orchestrator, 'config') or 'donation_default' not in orchestrator.config:
        return
    
    if pop_mode == "depvar":
        return  # depvar mode doesn't use these settings
    
    donation_config = orchestrator.config['donation_default']
    
    # Determine the actual income mode for donation_default
    # Priority: saved config > passed parameter
    actual_inc_mode = inc_mode
    if hasattr(st.session_state, 'selected_donation_config'):
        saved_config = st.session_state.selected_donation_config
        # Check for new key first, fall back to old key for backwards compatibility
        actual_inc_mode = saved_config.get('donation_income_mode', saved_config.get('income_spec_mode', inc_mode))
        # Normalize the mode string
        if 'continuous' in str(actual_inc_mode).lower():
            actual_inc_mode = 'continuous'
        else:
            actual_inc_mode = 'categorical'
    
    # Set income_mode in both legacy and new locations for compatibility
    donation_config['regression']['income_mode'] = actual_inc_mode
    if 'regression_coefficients' not in donation_config:
        donation_config['regression_coefficients'] = {}
    donation_config['regression_coefficients']['income_mode'] = actual_inc_mode
    
    # Set stochastic flag for copula mode if checkbox is enabled
    if pop_mode == "copula":
        donation_config['stochastic']['in_copula'] = st.session_state.sigma_in_copula
    
    # Apply sigma value based on mode and user preferences
    if pop_mode == "documentation" and not st.session_state.sigma_in_research:
        # Research mode with sigma disabled - set to 0
        donation_config['stochastic']['sigma_value'] = 0.0
    else:
        # Apply selected sigma value
        donation_config['stochastic']['sigma_value'] = st.session_state.sigma_value_ui
    
    # Apply chosen anchor weights
    donation_config['anchor_weights']['observed'] = st.session_state.anchor_observed_weight
    donation_config['anchor_weights']['predicted'] = 1 - st.session_state.anchor_observed_weight
    
    # Apply selected donation configuration if exists
    if hasattr(st.session_state, 'selected_donation_config'):
        apply_selected_donation_config(orchestrator, pop_mode, inc_mode)
    # Fallback: Apply custom regression coefficients if they exist
    elif hasattr(st.session_state, 'custom_coefficients') and 'donation_default' in st.session_state.custom_coefficients:
        custom_coeffs = st.session_state.custom_coefficients['donation_default']
        if 'regression_coefficients' not in donation_config:
            donation_config['regression_coefficients'] = {}
        donation_config['regression_coefficients'].update(custom_coeffs)
        donation_config['regression_coefficients']['income_mode'] = inc_mode
    # NEW FALLBACK: Use current session state coefficients if no custom coefficients are set
    else:
        from app.models import load_donation_coefficients_from_yaml
        if 'donation_coeff_intercept' not in st.session_state:
            load_donation_coefficients_from_yaml()
        
        from app.pages.decision_execution import get_current_coefficients
        current_coeffs = get_current_coefficients()
        current_coeffs['income_mode'] = inc_mode
        
        if 'regression_coefficients' not in donation_config:
            donation_config['regression_coefficients'] = {}
        donation_config['regression_coefficients'].update(current_coeffs)


def _apply_disclose_income_config(orchestrator, pop_mode: str, inc_mode: str = None):
    """
    Apply disclose_income-specific configuration to orchestrator.
    
    Handles income mode, stochastic settings, and anchor weights for disclose_income.
    
    Args:
        orchestrator: The orchestrator instance to configure
        pop_mode: Population mode (documentation, copula, baseline, depvar)
        inc_mode: Optional explicit income mode override (categorical/continuous).
                  If provided, this takes precedence over session state.
                  This is critical when running "Compare both" mode - the runner
                  passes the specific mode for each comparison run.
    """
    if not hasattr(orchestrator, 'config') or 'disclose_income' not in orchestrator.config:
        return
    
    if pop_mode == "depvar":
        return  # depvar mode doesn't use these settings
    
    di_config = orchestrator.config['disclose_income']
    
    # Determine income mode to use
    # Priority: explicit inc_mode parameter > session state
    if inc_mode is not None:
        # Explicit mode passed (e.g., from "Compare both" runner)
        # Normalize to match expected format
        if 'continuous' in str(inc_mode).lower():
            di_config['income_mode'] = 'continuous'
        else:
            di_config['income_mode'] = 'categorical'
        print(f"[DiscloseIncome] Using explicit inc_mode: {inc_mode} -> {di_config['income_mode']}")
    elif 'di_income_mode' in st.session_state:
        # Session state mode - check if it's "Compare both"
        session_mode = st.session_state.di_income_mode
        if 'compare' in str(session_mode).lower() or 'both' in str(session_mode).lower():
            # "Compare both" in session state but no explicit inc_mode passed
            # This shouldn't happen in normal flow, but default to categorical as safeguard
            di_config['income_mode'] = 'categorical'
            print(f"[DiscloseIncome] WARNING: 'Compare both' in session state but no explicit inc_mode - defaulting to categorical")
        else:
            # Single mode selected (Categorical only / Continuous only)
            di_config['income_mode'] = session_mode
            print(f"[DiscloseIncome] Using session state mode: {session_mode}")
    
    # Apply intercept from session state if available
    if 'di_intercept' in st.session_state:
        di_config['intercept'] = st.session_state.di_intercept
    
    # Apply anchor weights from session state if available
    if 'anchor_weights' not in di_config:
        di_config['anchor_weights'] = {}
    
    if 'di_wopb' in st.session_state:
        di_config['anchor_weights']['observed_prosocial'] = st.session_state.di_wopb
    
    if 'di_wpb' in st.session_state:
        di_config['anchor_weights']['prosocial_weight'] = st.session_state.di_wpb
    
    # Apply stochastic settings from session state if available
    if 'stochastic' not in di_config:
        di_config['stochastic'] = {}
    
    # Check if stochastic is enabled via UI
    sigma_enabled = st.session_state.get('di_sigma_enabled', False)
    
    if pop_mode == "baseline":
        # Baseline mode: no stochastic component
        di_config['stochastic']['sigma_value'] = 0.0
    elif sigma_enabled:
        # Stochastic enabled - use overall sigma
        di_config['stochastic']['sigma_value'] = 9.899547
        if 'di_sigma_strategy' in st.session_state:
            di_config['stochastic']['sigma_strategy'] = st.session_state.di_sigma_strategy
        if 'di_scale_factor' in st.session_state:
            di_config['stochastic']['scale_factor'] = st.session_state.di_scale_factor
        # Pass quintile-specific scale factors if in quintile mode
        if 'di_quintile_scale_factors' in st.session_state:
            di_config['stochastic']['quintile_scale_factors'] = st.session_state.di_quintile_scale_factors
    else:
        # Stochastic disabled
        di_config['stochastic']['sigma_value'] = 0.0


# =============================================================================
# MODE RUNNER FUNCTIONS - Each mode encapsulates its own agent sampling
# =============================================================================

def run_copula_mode(n_agents: int, seed: int, inc_mode: str, decision_settings: dict,
                    single_decision=None) -> pd.DataFrame:
    """
    Run simulation in Copula mode.
    
    - Uses synthetic agents sampled from copula
    - Uses regular Orchestrator
    """
    # 1. Sample copula agents
    trait_engine = TraitEngine()
    agents_df = trait_engine.sample(n_agents, seed)
    
    # 2. Create orchestrator
    orchestrator = Orchestrator()
    
    # 3. Apply configurations
    _apply_donation_config(orchestrator, "copula", inc_mode)
    _apply_disclose_income_config(orchestrator, "copula", inc_mode)  # Pass inc_mode for disclose_income
    _apply_simulation_params(orchestrator)
    _apply_decision_settings(orchestrator, decision_settings)
    
    # 4. Run simulation
    return orchestrator.run_simulation(n_agents, seed, single_decision, agents_df=agents_df)


def run_research_spec_mode(n_agents: int, seed: int, inc_mode: str, decision_settings: dict,
                           single_decision=None) -> pd.DataFrame:
    """
    Run simulation in Research Specification mode.
    
    - Uses original 280 participants
    - Uses OrchestratorDocMode (stochastic version)
    - Random sampling for n≤280 (without replacement)
    """
    # 1. Load original participants (random sampling)
    agents_df = _load_original_participants(n_agents, seed, random_sample=True)
    
    # 2. Create orchestrator
    orchestrator = OrchestratorDocMode()
    
    # 3. Apply configurations
    _apply_donation_config(orchestrator, "documentation", inc_mode)
    _apply_disclose_income_config(orchestrator, "documentation", inc_mode)  # Pass inc_mode for disclose_income
    _apply_simulation_params(orchestrator)
    _apply_decision_settings(orchestrator, decision_settings)
    
    # 4. Run simulation
    return orchestrator.run_simulation(n_agents, seed, single_decision, agents_df=agents_df)


def run_research_baseline_mode(n_agents: int, seed: int, inc_mode: str, decision_settings: dict,
                               single_decision=None) -> pd.DataFrame:
    """
    Run simulation in Research Baseline mode.
    
    - Uses original 280 participants
    - Uses OrchestratorBaseline (deterministic anchor)
    - Sequential selection for n≤280 (agents 0, 1, 2, ..., n-1)
    """
    # 1. Load original participants (sequential selection)
    agents_df = _load_original_participants(n_agents, seed, random_sample=False)
    
    # 2. Create orchestrator
    orchestrator = OrchestratorBaseline()
    
    # 3. Apply configurations
    _apply_donation_config(orchestrator, "baseline", inc_mode)
    _apply_disclose_income_config(orchestrator, "baseline", inc_mode)  # Pass inc_mode for disclose_income
    _apply_simulation_params(orchestrator)
    _apply_decision_settings(orchestrator, decision_settings)
    
    # 4. Run simulation
    return orchestrator.run_simulation(n_agents, seed, single_decision, agents_df=agents_df)


def run_depvar_mode(n_agents: int, seed: int, inc_mode: str, decision_settings: dict,
                    single_decision=None) -> pd.DataFrame:
    """
    Run simulation in Dependent Variable Resampling mode.
    
    - No agent sampling (resamples outcomes only)
    - Uses OrchestratorDepVar
    """
    # 1. No agent sampling for depvar mode
    
    # 2. Create orchestrator
    orchestrator = OrchestratorDepVar()
    
    # 3. Apply configurations (no donation config for depvar)
    _apply_simulation_params(orchestrator)
    _apply_decision_settings(orchestrator, decision_settings)
    
    # 4. Run simulation
    return orchestrator.run_simulation(n_agents, seed, single_decision)


# =============================================================================
# MODE DISPATCHER - Maps mode names to runner functions
# =============================================================================

MODE_RUNNERS = {
    "copula": run_copula_mode,
    "documentation": run_research_spec_mode,
    "baseline": run_research_baseline_mode,
    "depvar": run_depvar_mode,
}


def get_pop_type(population_mode: str) -> str:
    """Map UI population mode name to internal type."""
    return {
        "Copula (synthetic)": "copula",
        "Research Specification": "documentation",
        "Research Baseline": "baseline",
        "Dependent variable resampling": "depvar",
    }.get(population_mode, "copula")


def get_inc_mode(income_spec_mode: str) -> str:
    """Map UI income mode name to internal type."""
    if "continuous" in income_spec_mode.lower():
        return "continuous"
    return "categorical"


def run_monte_carlo_study() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Run Monte-Carlo study and return results."""
    try:
        # Check if we're in comparison mode
        is_pop_comparison = st.session_state.population_mode == "Compare all"
        is_income_comparison = st.session_state.get('income_spec_mode', 'categorical only') == "Compare both"
        
        # Map UI modes to script arguments
        population_mode_map = {
            'Copula (synthetic)': 'copula',
            'Research Specification': 'documentation',
            'Research Baseline': 'baseline',
            'Dependent variable resampling': 'depvar'
        }
        
        income_mode_map = {
            'categorical only': 'categorical',
            'continuous only': 'continuous'
        }
        
        # Determine which mode combinations to run
        if is_pop_comparison:
            # Run all 3 population modes
            pop_modes = [
                ('copula', 'Copula'),
                ('documentation', 'Research Spec'),
                ('baseline', 'Baseline')
            ]
        else:
            pop_key = st.session_state.population_mode
            pop_label = pop_key.replace(' (synthetic)', '').replace(' ', '_')
            pop_modes = [(population_mode_map.get(pop_key, 'copula'), pop_label)]
        
        if is_income_comparison:
            # Run both income modes (only if not doing population comparison)
            income_modes = [
                ('categorical', 'Categorical'),
                ('continuous', 'Continuous')
            ]
        else:
            income_key = st.session_state.get('income_spec_mode', 'categorical only')
            income_label = income_key.replace(' only', '').title()
            income_modes = [(income_mode_map.get(income_key, 'categorical'), income_label)]
        
        # For comparison mode, use the first mode and show a message
        pop_mode_arg, pop_label = pop_modes[0]
        income_mode_arg, income_label = income_modes[0]
        
        if len(pop_modes) > 1 or len(income_modes) > 1:
            st.warning(f"⚠️ **Comparison Mode Limitation**: Monte Carlo will run with **{pop_label} + {income_label}** mode only")
            st.info("""
            💡 **To compare multiple modes with Monte Carlo:**
            1. Run Monte Carlo with current mode
            2. Export/save results  
            3. Change to different mode (e.g., Research Specification)
            4. Run Monte Carlo again
            5. Compare the exported results
            
            This approach gives you better control and avoids extremely long run times.
            """)
        
        st.info(f"🔄 Starting Monte-Carlo study with {st.session_state.n_runs} runs of {st.session_state.n_agents} agents each...")
        st.caption(f"📊 Mode: {pop_label} + {income_label}")
        
        # Show estimated time
        estimated_time_per_run = 2
        total_estimated_time = st.session_state.n_runs * estimated_time_per_run
        st.caption(f"⏱️ Estimated time: ~{total_estimated_time} seconds ({total_estimated_time/60:.1f} minutes)")
        
        # Build command
        cmd = [
            sys.executable, 'scripts/run_mc_study.py',
            '--agents', str(st.session_state.n_agents),
            '--runs', str(st.session_state.n_runs),
            '--base-seed', str(st.session_state.base_seed),
            '--anchor-observed', str(st.session_state.anchor_observed_weight),
            '--population-mode', pop_mode_arg,
            '--income-mode', income_mode_arg
        ]
        
        # Handle multiple decisions for Monte Carlo
        if len(st.session_state.decision_params.selected_decisions) < len(ALL_DECISIONS):
            # Pass each selected decision as a separate argument
            for decision in st.session_state.decision_params.selected_decisions:
                cmd.extend(['--decision', decision])
        
        # Change to project directory to ensure scripts can be found
        cwd = Path(__file__).resolve().parents[1]
        
        # Debug: print command and environment
        with st.expander("🔧 Debug Information", expanded=True):
            st.code(' '.join(cmd))
            st.caption(f"Working directory: {cwd}")
            st.caption(f"Python executable: {sys.executable}")
            st.caption(f"Population mode: {st.session_state.population_mode} → {pop_mode_arg}")
            st.caption(f"Income mode: {st.session_state.get('income_spec_mode', 'categorical only')} → {income_mode_arg}")
            st.caption(f"Selected decisions: {st.session_state.decision_params.selected_decisions}")
            st.caption(f"Number of runs: {st.session_state.n_runs}")
            st.caption(f"Agents per run: {st.session_state.n_agents}")
        
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        output_container = st.container()
        
        # Run with real-time output capture using Popen instead of run
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
            # Read any available output (blocking until we get a line or EOF)
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
            
            # Check if process is done AND we've read all output
            # Empty line from readline() means EOF when process is done
            poll = process.poll()
            if poll is not None and not line:
                # Process finished and no more output
                break
            
            # If we got an empty line but process still running, continue
            if not line and poll is None:
                time.sleep(0.1)
                continue
        
        # Get any remaining stderr
        _, remaining_stderr = process.communicate()
        if remaining_stderr:
            stderr_lines.extend(remaining_stderr.strip().split('\n'))
        
        # Debug: Show total lines captured
        st.info(f"🔍 Total lines captured from stdout: {len(stdout_lines)}")
        
        # Join all output
        stdout = '\n'.join(stdout_lines)
        stderr = '\n'.join(stderr_lines)
        
        # Show final output
        if stdout:
            with st.expander("📋 Monte Carlo Output", expanded=True):
                st.text(stdout)
        else:
            st.warning("⚠️ No stdout output captured!")
        
        if stderr:
            with st.expander("⚠️ Monte Carlo Errors", expanded=True):
                st.text(stderr)
        
        # Debug: Show return code
        st.info(f"🔍 Process return code: {process.returncode}")
        
        if process.returncode == 0:
            # Parse output to find result files
            output_lines = stdout.strip().split('\n') if stdout else []
            summary_file = None
            detailed_file = None
            
            # Debug: Show what we're parsing
            st.info(f"🔍 Parsing {len(output_lines)} lines of output...")
            
            for i, line in enumerate(output_lines):
                if 'Summary saved to:' in line:
                    summary_file = line.split('Summary saved to:')[1].strip()
                    st.success(f"✅ Found summary file in line {i+1}: {summary_file}")
                elif 'Detailed results saved to:' in line:
                    detailed_file = line.split('Detailed results saved to:')[1].strip()
                    st.success(f"✅ Found detailed file in line {i+1}: {detailed_file}")
            
            progress_bar.progress(1.0)
            status_text.success("✅ Monte-Carlo study completed!")
            
            # Debug: Show what files were detected
            st.info(f"📁 Detected files - Summary: {summary_file}, Detailed: {detailed_file}")
            
            # Load results - handle relative paths
            if summary_file and not Path(summary_file).is_absolute():
                summary_file = str(cwd / summary_file)
            if detailed_file and not Path(detailed_file).is_absolute():
                detailed_file = str(cwd / detailed_file)
            
            # Debug: Show resolved paths
            st.info(f"📍 Resolved paths - Summary: {summary_file}, Detailed: {detailed_file}")
            
            # Check if files exist before loading
            summary_exists = Path(summary_file).exists() if summary_file else False
            detailed_exists = Path(detailed_file).exists() if detailed_file else False
            
            st.info(f"✅ File existence - Summary: {summary_exists}, Detailed: {detailed_exists}")
            
            # Load results
            mc_summary = pd.read_csv(summary_file) if summary_file and summary_exists else None
            mc_detailed = pd.read_csv(detailed_file) if detailed_file and detailed_exists else None
            
            # If files not found, show debug info
            if mc_summary is None:
                if not summary_file:
                    st.error("❌ Summary file path not detected in output!")
                elif not summary_exists:
                    st.error(f"❌ Summary file not found at: {summary_file}")
                    # List files in outputs directory
                    outputs_dir = cwd / "outputs"
                    if outputs_dir.exists():
                        recent_files = sorted(outputs_dir.glob("mc_summary*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
                        if recent_files:
                            st.warning(f"📂 Recent MC summary files found:")
                            for f in recent_files:
                                st.caption(f"  - {f.name}")
                    
            if mc_detailed is None:
                if not detailed_file:
                    st.warning("⚠️ Detailed file path not detected in output (this is optional)")
                elif not detailed_exists:
                    st.warning(f"⚠️ Detailed file not found at: {detailed_file}")
            
            # Show loaded data shape
            if mc_summary is not None:
                st.success(f"✅ Loaded summary: {mc_summary.shape}")
            if mc_detailed is not None:
                st.success(f"✅ Loaded detailed: {mc_detailed.shape}")
            
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
    """
    Run simulation using mode runner functions.
    
    Each population mode (Copula, Research Spec, Research Baseline, DepVar) 
    uses its own dedicated runner function that encapsulates:
    - Agent sampling (mode-appropriate)
    - Orchestrator creation
    - Configuration application
    
    This ensures consistent behavior across all code paths.
    """
    try:
        with st.spinner("🔄 Running simulation..."):
            # Get common parameters
            n_agents = st.session_state.n_agents
            if st.session_state.sim_params.simulation_mode == "Single Run":
                seed = st.session_state.get('seed_input', st.session_state.seed)
            else:
                seed = st.session_state.get('base_seed_input', st.session_state.base_seed)
            
            # Determine which decisions to run
            single_decision = None if len(st.session_state.decision_params.selected_decisions) == len(ALL_DECISIONS) else st.session_state.decision_params.selected_decisions
            
            # NOTE: selected_donation_config is used ONLY for donation_default decision.
            # Each decision has its own income mode setting that is respected independently.
            # We no longer override global income_spec_mode - each decision reads its own settings.
            if hasattr(st.session_state, 'selected_donation_config'):
                config = st.session_state.selected_donation_config
                
                # Show info about donation config (but don't override global settings)
                donation_income_mode = config.get('donation_income_mode', config.get('income_spec_mode', 'categorical only'))
                st.info(f"🎯 Donation Default will use saved configuration: {donation_income_mode}")
                
                # Use original seed/n_agents for reproducibility of donation_default results
                if 'original_seed' in config:
                    seed = config['original_seed']
                    st.caption(f"🔑 Using original seed: {seed}")
                if 'original_n_agents' in config:
                    n_agents = config['original_n_agents']
                    st.caption(f"👥 Using original agent count: {n_agents}")
            
            # Collect current decision settings
            decision_settings = collect_decision_settings()
            
            # Debug: Show decision settings being applied
            if decision_settings:
                setting_info = []
                for decision, settings in decision_settings.items():
                    decision_type = settings.get('type')
                    if decision_type == 'random_probability':
                        prob_y = settings['probability_y']
                        options = settings['options']
                        setting_info.append(f"{decision}: {prob_y:.0%} {options[0]} / {1-prob_y:.0%} {options[1]}")
                    elif decision_type == 'prioritized_selection':
                        priority_template = settings.get('priority_template', [])
                        if len(priority_template) == 1:
                            setting_info.append(f"{decision}: {priority_template[0]} only")
                        else:
                            setting_info.append(f"{decision}: {len(priority_template)} priorities")
                    elif decision_type == 'checkbox_selection':
                        selected = settings.get('selected_params', [])
                        setting_info.append(f"{decision}: {len(selected)} params selected ({', '.join(selected)})")
                    elif decision_type == 'radio_selection':
                        selected = settings.get('selected_option', 'unknown')
                        setting_info.append(f"{decision}: {selected}")
                    elif decision_type == 'numeric':
                        value = settings.get('value', 0)
                        try:
                            float_value = float(value)
                            if 0 <= float_value <= 1:
                                setting_info.append(f"{decision}: {float_value:.1%}")
                            else:
                                setting_info.append(f"{decision}: {float_value}")
                        except (ValueError, TypeError):
                            setting_info.append(f"{decision}: {value}")
                    elif decision_type == 'placeholder':
                        value = settings.get('value', 'default')
                        setting_info.append(f"{decision}: {value}")
                
                if setting_info:
                    st.success(f"🎲 Using configured defaults: {', '.join(setting_info)}")
            
            # =================================================================
            # RUN SIMULATION USING MODE RUNNERS
            # Each mode uses its natural agent source - no more agent mismatch!
            # =================================================================
            results = {}
            
            # Determine effective income mode for simulation loop control
            # This determines the result keys (categorical/continuous) for the DataFrame storage
            # Each decision reads its OWN income mode from session state during execution
            
            # Get income mode based on what's being run
            if single_decision == ['disclose_income']:
                # Running ONLY disclose_income - use its specific mode
                effective_income_mode = st.session_state.get('di_income_mode', 'Categorical only')
                # CRITICAL: Sync di_income_mode to income_spec_mode so results page shows correct view
                # The results page checks income_spec_mode to decide whether to show side-by-side comparison
                st.session_state.income_spec_mode = effective_income_mode
                st.caption(f"🎯 Using Disclose Income specific mode: {effective_income_mode}")
            elif single_decision == ['donation_default']:
                # Running ONLY donation_default - check for saved config first
                if hasattr(st.session_state, 'selected_donation_config'):
                    config = st.session_state.selected_donation_config
                    effective_income_mode = config.get('donation_income_mode', config.get('income_spec_mode', 'categorical only'))
                else:
                    effective_income_mode = st.session_state.get('income_spec_mode', 'categorical only')
                st.caption(f"🎯 Using Donation Default income mode: {effective_income_mode}")
            else:
                # Combined simulation or other decisions - use the global income_spec_mode setting
                # Individual decisions will read their own settings during execution
                effective_income_mode = st.session_state.get('income_spec_mode', 'categorical only')
            
            if st.session_state.population_mode == "Compare all":
                # Compare all three population modes - each uses its natural agent source
                st.info("🔄 Running Compare All mode - each population uses its natural agent source")
                
                for result_name, pop_type in [("copula", "copula"), ("research_spec", "documentation"), ("research_baseline", "baseline")]:
                    runner = MODE_RUNNERS[pop_type]
                    
                    if effective_income_mode == "Compare both":
                        results[f"{result_name}_categorical"] = runner(n_agents, seed, "categorical", decision_settings, single_decision)
                        results[f"{result_name}_continuous"] = runner(n_agents, seed, "continuous", decision_settings, single_decision)
                    elif "continuous" in effective_income_mode.lower():
                        results[f"{result_name}_continuous"] = runner(n_agents, seed, "continuous", decision_settings, single_decision)
                    else:  # categorical only
                        results[f"{result_name}_categorical"] = runner(n_agents, seed, "categorical", decision_settings, single_decision)
            
            elif st.session_state.population_mode == "Dependent variable resampling":
                # DepVar mode
                results["depvar"] = run_depvar_mode(n_agents, seed, "categorical", decision_settings, single_decision)
            
            else:
                # Single population mode - use the appropriate runner
                pop_type = get_pop_type(st.session_state.population_mode)
                runner = MODE_RUNNERS[pop_type]
                
                # Show which agent source is being used
                if pop_type == "copula":
                    st.info(f"🎲 Using synthetic agents from copula")
                elif pop_type in ["documentation", "baseline"]:
                    st.info(f"📊 Using original 280 participants")
                    
                if effective_income_mode == "Compare both":
                    results["categorical"] = runner(n_agents, seed, "categorical", decision_settings, single_decision)
                    results["continuous"] = runner(n_agents, seed, "continuous", decision_settings, single_decision)
                elif "continuous" in effective_income_mode.lower():
                    results["continuous"] = runner(n_agents, seed, "continuous", decision_settings, single_decision)
                else:  # categorical only
                    results["categorical"] = runner(n_agents, seed, "categorical", decision_settings, single_decision)
            
            # Assign global transaction IDs to ensure consistency across exports
            for key in results:
                results[key] = _assign_global_transaction_ids(results[key])
            
            # Store results
            st.session_state.simulation_results = results
            
            # Extract and store vendor data from first DataFrame
            for df in results.values():
                if hasattr(df, 'attrs') and 'vendors' in df.attrs:
                    st.session_state.vendors = df.attrs['vendors']
                    break
            
            # Add flag to indicate we're using selected configuration
            if hasattr(st.session_state, 'selected_donation_config'):
                st.session_state._using_selected_config = True
            
            st.session_state.page = 'results'
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        import traceback
        st.text(traceback.format_exc())


def collect_decision_settings():
    """Collect current default decision settings from session state (probabilities, selections, etc.)
    
    IMPORTANT: This function checks both session state AND _persistent_defaults (shadow state).
    The _persistent_defaults dictionary is used by default_config.py widgets to preserve values
    across page navigation. We must check it here to ensure configured values are used even when
    Page 2 hasn't rendered (e.g., when running simulation from Results page).
    """
    
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    
    decision_settings = {}
    
    # Get the persistent defaults dictionary (shadow state from default_config.py)
    # This is critical for preserving user-configured values when Page 2 hasn't rendered
    persistent_defaults = st.session_state.get('_persistent_defaults', {})
    
    # Check each decision for configured settings
    for decision_name, default_value in DEFAULT_DECISION_VALUES.items():
        if isinstance(default_value, dict):
            decision_type = default_value.get("type")
            
            # Handle random probability decisions (disclose_income, disclose_documents, purchase_vs_bid)
            if decision_type == "random_probability":
                # Priority order for probability values:
                # 1. Post-simulation adjustment from Results page ({decision_name}_probability_y)
                # 2. Pre-configured default from Overview tab - persistent storage (_persistent_defaults)
                # 3. Pre-configured default from Overview tab - session state ({decision_name}_default_probability_y)
                # 4. Hard-coded default from DEFAULT_DECISION_VALUES
                
                post_sim_key = f"{decision_name}_probability_y"
                pre_config_key = f"{decision_name}_default_probability_y"
                hardcoded_default = default_value.get("probability_y", 0.5)
                
                # Check in priority order
                if post_sim_key in st.session_state:
                    current_prob = st.session_state[post_sim_key]
                elif pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    # Session state keys might be reset to defaults by initialization scripts during reruns
                    current_prob = persistent_defaults[pre_config_key]
                elif pre_config_key in st.session_state:
                    current_prob = st.session_state[pre_config_key]
                else:
                    current_prob = hardcoded_default
                
                # Debug print to verify source
                # if pre_config_key in persistent_defaults:
                #    print(f"[DEBUG] {decision_name}: Using persistent value {current_prob}")
                
                decision_settings[decision_name] = {
                    "probability_y": current_prob,
                    "options": default_value.get("options", ["Y", "N"]),
                    "type": "random_probability"
                }
            
            # Handle checkbox selection decisions (vendor_choice_weights)
            elif decision_type == "checkbox_selection":
                # Priority order:
                # 1. Post-simulation adjustment from Results page (vendor_choice_weights_selection)
                # 2. Pre-configured default from Overview tab - persistent storage (_persistent_defaults)
                # 3. Pre-configured default from Overview tab - session state (vendor_choice_weights_default_params)
                # 4. Hard-coded default from DEFAULT_DECISION_VALUES
                
                post_sim_key = f"{decision_name}_selection"  # e.g., "vendor_choice_weights_selection"
                pre_config_key = f"{decision_name}_default_params"  # e.g., "vendor_choice_weights_default_params"
                hardcoded_default = default_value.get("default_selection", [])
                
                # Check in priority order
                if post_sim_key in st.session_state:
                    selected_params = st.session_state[post_sim_key]
                elif pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    selected_params = persistent_defaults[pre_config_key]
                elif pre_config_key in st.session_state:
                    selected_params = st.session_state[pre_config_key]
                else:
                    selected_params = hardcoded_default
                
                # Calculate equal weights for selected parameters
                if len(selected_params) > 0:
                    weight_per_param = 1.0 / len(selected_params)
                    weights = {}
                    
                    # Set weights for all parameters
                    for param_key in default_value.get("parameters", {}).keys():
                        if param_key in selected_params:
                            weights[param_key] = weight_per_param
                        else:
                            weights[param_key] = 0.0
                else:
                    # Fallback to equal weights if nothing selected
                    params = list(default_value.get("parameters", {}).keys())
                    weight_per_param = 1.0 / len(params) if params else 0.25
                    weights = {param: weight_per_param for param in params}
                
                decision_settings[decision_name] = {
                    "selected_params": selected_params,
                    "weights": weights,
                    "type": "checkbox_selection"
                }
            
            # Handle prioritized selection decisions (rejected_transaction_defaults with priority lists)
            elif decision_type == "prioritized_selection":
                # Priority order:
                # 1. Pre-configured priority template from Overview tab - persistent storage (_persistent_defaults)
                # 2. Pre-configured priority template from Overview tab - session state ({decision_name}_priority_template)
                # 3. Hard-coded default from DEFAULT_DECISION_VALUES
                
                pre_config_key = f"{decision_name}_priority_template"
                hardcoded_default = default_value.get("priority_template", ["forgo_transaction"])
                
                # Check in priority order
                if pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    priority_template = persistent_defaults[pre_config_key]
                elif pre_config_key in st.session_state:
                    priority_template = st.session_state[pre_config_key]
                else:
                    priority_template = hardcoded_default
                
                decision_settings[decision_name] = {
                    "priority_template": priority_template,
                    "type": "prioritized_selection"
                }
            
            # Handle radio selection decisions (rejected_transaction_option)
            elif decision_type == "radio_selection":
                # Priority order:
                # 1. Post-simulation adjustment from Results page (specific to each decision)
                # 2. Pre-configured default from Overview tab - persistent storage (_persistent_defaults)
                # 3. Pre-configured default from Overview tab - session state ({decision_name}_default_selection)
                # 4. Hard-coded default from DEFAULT_DECISION_VALUES
                
                # Map decision names to their post-simulation keys
                post_sim_keys = {
                    "rejected_transaction_defaults": "rejected_transaction_defaults_option",
                    "rejected_transaction_option": "rejected_transaction_option_selection"
                }
                
                post_sim_key = post_sim_keys.get(decision_name)
                pre_config_key = f"{decision_name}_default_selection"
                hardcoded_default = default_value.get("default_option", "")
                
                # Check in priority order
                if post_sim_key and post_sim_key in st.session_state:
                    selected_option = st.session_state[post_sim_key]
                elif pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    selected_option = persistent_defaults[pre_config_key]
                elif pre_config_key in st.session_state:
                    selected_option = st.session_state[pre_config_key]
                else:
                    selected_option = hardcoded_default
                
                decision_settings[decision_name] = {
                    "selected_option": selected_option,
                    "type": "radio_selection"
                }
        
        else:
            # Handle simple values (numeric or string placeholders)
            pre_config_key = f"{decision_name}_default_value"
            
            # Check in priority order: persistent_defaults → session state → hardcoded default
            if pre_config_key in persistent_defaults:
                configured_value = persistent_defaults[pre_config_key]
            elif pre_config_key in st.session_state:
                configured_value = st.session_state[pre_config_key]
            else:
                configured_value = default_value
            
            # Determine if it's numeric or a placeholder string
            if isinstance(configured_value, (int, float)):
                # print(f"[DEBUG] collect_settings: {decision_name} (numeric) = {configured_value}")
                decision_settings[decision_name] = {
                    "value": configured_value,
                    "type": "numeric"
                }
            else:
                # It's a placeholder string like "RANDOM_WITHIN_LIMIT", "NA", etc.
                # print(f"[DEBUG] collect_settings: {decision_name} (placeholder) = {configured_value}")
                decision_settings[decision_name] = {
                    "value": configured_value,
                    "type": "placeholder"
                }
    
    return decision_settings


def run_simulation():
    """Run simulation with current parameters"""
    st.session_state.page = 'results'
    # Simulation will be triggered on results page


def apply_selected_donation_config(orchestrator, pop_mode, inc_mode):
    """Apply the selected donation configuration to the orchestrator"""
    
    config = st.session_state.selected_donation_config
    
    # Override coefficients
    if 'regression_coefficients' not in orchestrator.config['donation_default']:
        orchestrator.config['donation_default']['regression_coefficients'] = {}
    
    # Apply all coefficients from selected configuration
    orchestrator.config['donation_default']['regression_coefficients'].update(config['coefficients'])
    
    # Use the saved donation config's income mode (not the passed parameter)
    # This ensures donation_default uses its own configured income mode
    saved_inc_mode = config.get('donation_income_mode', config.get('income_spec_mode', inc_mode))
    # Normalize
    if 'continuous' in str(saved_inc_mode).lower():
        saved_inc_mode = 'continuous'
    else:
        saved_inc_mode = 'categorical'
    orchestrator.config['donation_default']['regression_coefficients']['income_mode'] = saved_inc_mode
    
    # Apply stochastic parameters
    stoch_params = config['stochastic_params']
    
    # Update stochastic settings
    orchestrator.config['donation_default']['stochastic'].update({
        'sigma_value': stoch_params['stochastic']['sigma_value'],
        'sigma_coefficient': stoch_params['stochastic']['sigma_coefficient'],
        'sigma_in_copula': stoch_params['stochastic']['sigma_in_copula'],
        'sigma_in_research': stoch_params['stochastic']['sigma_in_research']
    })
    
    # Update anchor weights
    orchestrator.config['donation_default']['anchor_weights'].update(stoch_params['anchor_weights'])
    
    # Override session state variables to match selected configuration
    # This ensures the UI reflects the selected configuration
    st.session_state.sigma_value_ui = stoch_params['stochastic']['sigma_value']
    st.session_state.sigma_coefficient = stoch_params['stochastic']['sigma_coefficient']
    st.session_state.sigma_in_copula = stoch_params['stochastic']['sigma_in_copula']
    st.session_state.sigma_in_research = stoch_params['stochastic']['sigma_in_research']
    st.session_state.anchor_observed_weight = stoch_params['anchor_weights']['observed']
    
    # Override coefficient session state variables
    coeffs = config['coefficients']
    st.session_state.donation_coeff_intercept = coeffs['intercept']
    st.session_state.donation_coeff_hh = coeffs['beta_hh']
    st.session_state.donation_coeff_linear = coeffs['beta_income_linear']
    
    # Group coefficients
    for group, coeff in coeffs['beta_group'].items():
        if group == 'MidSub':
            st.session_state.donation_coeff_midsub = coeff
        elif group == 'NoSub':
            st.session_state.donation_coeff_nosub = coeff
        elif group == 'FullSub':
            st.session_state.donation_coeff_fullsub = coeff
    
    # Income quintile coefficients
    for quintile, coeff in coeffs['beta_income_q'].items():
        if quintile == 'Q1':
            st.session_state.donation_coeff_q1 = coeff
        elif quintile == 'Q2':
            st.session_state.donation_coeff_q2 = coeff
        elif quintile == 'Q3':
            st.session_state.donation_coeff_q3 = coeff
        elif quintile == 'Q4_Q5':
            st.session_state.donation_coeff_q45 = coeff
    
    # Study programme coefficients
    for study, coeff in coeffs['beta_study'].items():
        if study == 'Incoming':
            st.session_state.donation_coeff_incoming = coeff
        elif study == 'Law5yr':
            st.session_state.donation_coeff_law = coeff
        elif study == 'UG3yr':
            st.session_state.donation_coeff_ug = coeff
        elif study == 'Grad2yr':
            st.session_state.donation_coeff_grad = coeff
