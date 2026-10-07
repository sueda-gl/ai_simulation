# app/models.py
"""
Data models and session state management for the Enhanced AI Agent Simulation.
"""
import copy

import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from app.seam.config_repo import read_yaml_file
import numpy as np
from app.seam.sentinels import donation_sigma_overall
from app.reports import preview


# R12: one sigma constant for the donation model, read from
# `donation_default.stochastic.sigma_overall` in config/decisions.yaml (never a
# literal in app/). The UI multiplies it by the sigma coefficient slider.
DONATION_SIGMA_OVERALL = donation_sigma_overall()


@dataclass
class SimulationParameters:
    """Common simulation parameters (Page 1)"""
    # Simulation mode
    simulation_execution_mode: str = "snapshot"  # "snapshot" or "live"
    simulation_mode: str = "Single Run"  # "Single Run" or "Monte-Carlo Study"
    
    # Time parameters
    periods: int = 1  # ✅ As specified
    duration_hours: float = 1.0  # ✅ As specified
    
    # Market parameters
    num_vendors: int = 1  # ✅ Changed from 5 to 1
    market_price: float = 100.0  # ✅ Changed from 10.0 to 100.0
    vendor_price_min: float = 50.0  # ✅ Changed from 8.0 to 50.0
    vendor_price_max: float = 150.0  # ✅ Changed from 12.0 to 150.0
    
    # Product offering
    products_per_vendor: int = 100  # ✅ As specified (legacy - for backward compatibility)
    carryover: bool = False  # ✅ As specified (legacy global carryover)
    
    # Vendor configuration
    vendor_config_mode: str = "random"  # ✅ Generate randomly as specified
    
    # Random vendor generation parameters
    vendor_price_min: float = 50.0  # ✅ As specified
    vendor_price_max: float = 150.0  # ✅ As specified
    vendor_products_min: int = 50  # ✅ As specified
    vendor_products_max: int = 150  # ✅ As specified
    vendor_products_avg: int = 100  # ✅ As specified
    vendor_carryover_probability: float = 0.0  # ✅ Changed from 0.5 to 0.0 (unchecked = no carryover)
    override_carryover: bool = False  # ✅ As specified
    global_carryover: bool = False  # ✅ Changed to False (unchecked)
    
    # Uploaded vendor configuration
    vendor_config_data: Optional[List[Dict]] = None  # List of vendor configs from CSV
    vendor_prices: Optional[List[float]] = None  # Legacy - for backward compatibility
    vendor_price_source: str = "random"  # ✅ Generate randomly as specified
    
    # Pricing parameters
    bidding_percentage: float = 0.5  # bp (proportion available for bidding)
    platform_markup: float = 0.1  # ✅ 10% as specified
    price_range: float = 0.25  # ✅ As specified
    price_grid: int = 11  # ✅ As specified
    
    # Income distribution parameters
    income_min: float = 0.0  # ✅ Changed from 1000.0 to 0
    income_max: float = 100000.0  # ✅ Changed from 10000.0 to 100000
    income_avg: float = 25000.0  # ✅ Changed from 5000.0 to 25000
    income_avg_type: str = "average"  # ✅ As specified
    discount_income_threshold: float = 12500.0  # Set to middle of new range
    income_distribution: str = "lognormal"  # ✅ As specified
    
    # Distribution-specific parameters
    # Lognormal parameters
    lognormal_mu: float = 10.0  # Location parameter (mean of log)
    lognormal_sigma: float = 0.5  # Shape parameter (standard deviation of log)
    lognormal_min: float = 0.0  # Minimum value (linear shift)
    lognormal_max: Optional[float] = None  # Maximum value (rejection sampling)
    
    # Generalised Gamma parameters  
    gg_k: float = 1.5  # Shape parameter 1 (k)
    gg_c: float = 2.0  # Shape parameter 2 (c)
    gg_lambda: float = 20000.0  # Scale parameter (λ)
    gg_min: float = 0.0  # Minimum value (linear shift)
    gg_max: Optional[float] = None  # Maximum value (rejection sampling)
    
    # Dagum parameters
    dagum_a: float = 2.0  # Shape parameter (tail thickness)
    dagum_p: float = 1.5  # Shape parameter (body shape)
    dagum_b: float = 25000.0  # Scale parameter (median-like)
    dagum_min: float = 0.0  # Minimum value (linear shift)
    dagum_max: Optional[float] = None  # Maximum value (rejection sampling)
    
    # Income categories
    num_discount_categories: int = 10  # ✅ Changed from 3 to 10
    num_fixed_categories: int = 10  # ✅ Changed from 5 to 10
    
    # Consumption limits
    apply_purchasing_limits: bool = False  # ✅ Changed from True to False (unchecked)
    purchasing_limits: Dict[str, float] = field(default_factory=dict)
    purchasing_limits_source: str = "manual"  # "manual" or "upload"
    max_purchases_per_term: int = 50  # Fallback maximum when purchasing limits disabled
    
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
        """Calculate number of products available for auction per vendor (legacy method)"""
        return int(self.products_per_vendor * self.bidding_percentage)
    
    def sample_income_distribution(self, n_samples: int = 1000, seed: int = 42) -> np.ndarray:
        """Sample from the configured income distribution"""
        return preview.sample_income_distribution(
            n_samples=n_samples,
            seed=seed,
            **preview.distribution_kwargs(self),
        )

    def get_discount_qualification_rate(self, n_samples: int = 1000) -> float:
        """Calculate the percentage of agents that would qualify for discounts

        No seed is passed on purpose: the rate has always been computed off the
        preview's default seed, never off the seed the histogram was drawn with.
        """
        return preview.discount_qualification_rate(
            discount_income_threshold=self.discount_income_threshold,
            n_samples=n_samples,
            **preview.distribution_kwargs(self),
        )


#: Session-state values seeded once per session by initialize_session_state (the
#: ones not held in SimulationParameters). Page 1's "Reset to Default Values"
#: puts its Page-1 entries (n_agents, seed, n_runs, base_seed,
#: show_individual_agents, population_mode) back to these same values (Q-39).
SESSION_DEFAULTS = {
    'population_mode': 'Copula (synthetic)',
    'income_spec_mode': 'categorical only',
    'sigma_in_copula': False,
    'sigma_in_research': True,  # Enable sigma in Research mode by default
    'sigma_value_ui': DONATION_SIGMA_OVERALL,  # Static empirical SD value
    'sigma_coefficient': 1.0,  # Coefficient to multiply the static SD (0-2)
    'anchor_observed_weight': 0.75,
    'n_agents': 1000,
    'seed': 42,
    'n_runs': 10,
    'base_seed': 42,
    'show_individual_agents': False,
    'save_results': True,  # NOTE: Feature disabled in UI but kept for backward compatibility
    'simulation_running': False,
    'individual_results': {}  # New: store individual decision results
}


@dataclass
class DecisionParameters:
    """Decision-specific parameters (Page 2)"""
    selected_decisions: List[str] = field(default_factory=list)
    decision_configs: Dict[str, Dict] = field(default_factory=dict)


def initialize_session_state():
    """Initialize all session state variables."""
    if 'page' not in st.session_state:
        st.session_state.page = 'page1'
    if 'sim_params' not in st.session_state:
        st.session_state.sim_params = SimulationParameters()
    
    # Initialize donation coefficient variables from YAML config
    if 'donation_coeff_intercept' not in st.session_state:
        load_donation_coefficients_from_yaml()
    else:
        # Migrate old session state objects by adding missing attributes
        sim_params = st.session_state.sim_params
        
        # Add lognormal parameters if missing
        if not hasattr(sim_params, 'lognormal_mu'):
            sim_params.lognormal_mu = 10.0
        if not hasattr(sim_params, 'lognormal_min'):
            sim_params.lognormal_min = 0.0
        if not hasattr(sim_params, 'lognormal_max'):
            sim_params.lognormal_max = None
            
        # Add Generalised Gamma parameters if missing
        if not hasattr(sim_params, 'gg_k'):
            sim_params.gg_k = 1.5
        if not hasattr(sim_params, 'gg_c'):
            sim_params.gg_c = 2.0
        if not hasattr(sim_params, 'gg_lambda'):
            sim_params.gg_lambda = 20000.0
        if not hasattr(sim_params, 'gg_min'):
            sim_params.gg_min = 0.0
        if not hasattr(sim_params, 'gg_max'):
            sim_params.gg_max = None
            
        # Add Dagum parameters if missing
        if not hasattr(sim_params, 'dagum_a'):
            sim_params.dagum_a = 2.0
        if not hasattr(sim_params, 'dagum_p'):
            sim_params.dagum_p = 1.5
        if not hasattr(sim_params, 'dagum_b'):
            sim_params.dagum_b = 25000.0
        if not hasattr(sim_params, 'dagum_min'):
            sim_params.dagum_min = 0.0
        if not hasattr(sim_params, 'dagum_max'):
            sim_params.dagum_max = None
            
        # Migrate old income distribution types to new ones
        if hasattr(sim_params, 'income_distribution'):
            if sim_params.income_distribution == 'pareto':
                sim_params.income_distribution = 'dagum'  # Migrate Pareto to Dagum
            elif sim_params.income_distribution == 'weibull':
                sim_params.income_distribution = 'generalised_gamma'  # Migrate Weibull to GG

        # R9: the vendor price / product "value migration" block that used to sit
        # here is gone. It rewrote vendor_price_min/max, market_price and
        # vendor_products_min/max on every rerun whenever they happened to equal one
        # of the old defaults, so a single vendor could not be run at the Page-1
        # average price / average products the user typed. A single vendor now runs
        # at exactly those values.

    if 'decision_params' not in st.session_state:
        st.session_state.decision_params = DecisionParameters()
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'mc_results' not in st.session_state:
        st.session_state.mc_results = None
    
    # Add missing defaults used across the UI and simulation
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(default_value)
    
    # Initialize all default decision parameters (CRITICAL: prevents state loss)
    initialize_default_decision_parameters()


def load_donation_coefficients_from_yaml():
    """Load donation_default coefficients from YAML config into session state variables
    
    IMPORTANT: YAML is the SINGLE source of truth. No fallback values are used.
    If coefficients are missing from YAML, an error will be raised.
    """
    config_path = Path(__file__).parent.parent / "config" / "decisions.yaml"
    config = read_yaml_file(config_path)
    
    # Get donation config - MUST exist
    donation_config = config['donation_default']
    regression_coeffs = donation_config['regression_coefficients']
    
    # Load both categorical and continuous coefficient sets for Compare both mode
    if 'categorical' in regression_coeffs and 'continuous' in regression_coeffs:
        # Load categorical coefficients
        cat_coeffs = regression_coeffs['categorical']
        load_coefficient_set(cat_coeffs, 'cat')
        
        # Load continuous coefficients  
        cont_coeffs = regression_coeffs['continuous']
        load_coefficient_set(cont_coeffs, 'cont')
        
        # Determine which to use for main session state variables based on mode
        income_mode = st.session_state.get('income_spec_mode', 'categorical only')
        if 'continuous' in income_mode.lower() and 'compare' not in income_mode.lower():
            coeffs = cont_coeffs
        else:
            coeffs = cat_coeffs  # default for categorical only and compare both
    else:
        # Fall back to legacy format
        coeffs = regression_coeffs if regression_coeffs else donation_config.get('regression', {})
        load_coefficient_set(coeffs, 'cat')  # Store as categorical
        load_coefficient_set(coeffs, 'cont')  # Store as continuous (same values)
    
    # Load main session state variables (used by individual decision execution)
    # NO FALLBACK VALUES - coefficients MUST exist in YAML
    st.session_state.donation_coeff_intercept = coeffs['intercept']
    st.session_state.donation_coeff_hh = coeffs['beta_hh']
    st.session_state.donation_coeff_linear = coeffs.get('beta_income_linear', 0.0)  # Only linear can default to 0
    
    # Debug: Print what we're loading
    print(f"[DEBUG] Loading coefficients for mode: {st.session_state.get('income_spec_mode', 'unknown')}")
    print(f"[DEBUG] Selected coeffs intercept: {coeffs['intercept']}")
    print(f"[DEBUG] Selected coeffs linear: {coeffs.get('beta_income_linear', 'NOT FOUND')}")
    
    # Group coefficients
    beta_group = coeffs['beta_group']
    st.session_state.donation_coeff_midsub = beta_group['MidSub']
    st.session_state.donation_coeff_nosub = beta_group['NoSub']
    st.session_state.donation_coeff_fullsub = beta_group['FullSub']
    
    # Income quintile coefficients (for categorical mode)
    beta_income_q = coeffs.get('beta_income_q', {})
    st.session_state.donation_coeff_q1 = beta_income_q.get('Q1', 0.0)
    st.session_state.donation_coeff_q2 = beta_income_q.get('Q2', 0.0)
    st.session_state.donation_coeff_q3 = beta_income_q.get('Q3', 0.0)
    st.session_state.donation_coeff_q4 = beta_income_q.get('Q4', 0.0)
    st.session_state.donation_coeff_q5 = beta_income_q.get('Q5', beta_income_q.get('Q4_Q5', 0.0))  # Support both Q5 and legacy Q4_Q5
    
    # Study programme coefficients
    beta_study = coeffs['beta_study']
    st.session_state.donation_coeff_incoming = beta_study['Incoming']
    st.session_state.donation_coeff_law = beta_study['Law5yr']
    st.session_state.donation_coeff_ug = beta_study['UG3yr']
    st.session_state.donation_coeff_grad = beta_study['Grad2yr']
    
    # Load adjustment parameter
    adjustment_params = donation_config.get('adjustment', {})
    st.session_state.donation_adjustment_shift = adjustment_params.get('shift_value', 0.0)


def load_coefficient_set(coeffs, mode_suffix):
    """Load a coefficient set into session state with mode-specific suffix (cat/cont)
    
    IMPORTANT: YAML is the SINGLE source of truth. No fallback values are used.
    """
    # Load coefficients with suffix - NO FALLBACK VALUES
    st.session_state[f'donation_coeff_intercept_{mode_suffix}'] = coeffs['intercept']
    st.session_state[f'donation_coeff_hh_{mode_suffix}'] = coeffs['beta_hh']
    st.session_state[f'donation_coeff_linear_{mode_suffix}'] = coeffs.get('beta_income_linear', 0.0)  # Only linear can default to 0
    
    # Group coefficients
    beta_group = coeffs['beta_group']
    st.session_state[f'donation_coeff_midsub_{mode_suffix}'] = beta_group['MidSub']
    st.session_state[f'donation_coeff_nosub_{mode_suffix}'] = beta_group['NoSub']
    st.session_state[f'donation_coeff_fullsub_{mode_suffix}'] = beta_group['FullSub']
    
    # Income quintile coefficients (for categorical mode)
    beta_income_q = coeffs.get('beta_income_q', {})
    st.session_state[f'donation_coeff_q1_{mode_suffix}'] = beta_income_q.get('Q1', 0.0)
    st.session_state[f'donation_coeff_q2_{mode_suffix}'] = beta_income_q.get('Q2', 0.0)
    st.session_state[f'donation_coeff_q3_{mode_suffix}'] = beta_income_q.get('Q3', 0.0)
    st.session_state[f'donation_coeff_q4_{mode_suffix}'] = beta_income_q.get('Q4', 0.0)
    st.session_state[f'donation_coeff_q5_{mode_suffix}'] = beta_income_q.get('Q5', beta_income_q.get('Q4_Q5', 0.0))  # Support both Q5 and legacy Q4_Q5
    
    # Study programme coefficients
    beta_study = coeffs['beta_study']
    st.session_state[f'donation_coeff_incoming_{mode_suffix}'] = beta_study['Incoming']
    st.session_state[f'donation_coeff_law_{mode_suffix}'] = beta_study['Law5yr']
    st.session_state[f'donation_coeff_ug_{mode_suffix}'] = beta_study['UG3yr']
    st.session_state[f'donation_coeff_grad_{mode_suffix}'] = beta_study['Grad2yr']


# Helper functions for parameter analysis
def get_decision_global_parameters(selected_decisions: List[str]) -> set:
    """Get all global parameters used by selected decisions from decisions.yaml"""
    try:
        decisions_path = Path(__file__).resolve().parents[1] / "config" / "decisions.yaml"
        decisions_config = read_yaml_file(decisions_path)
        
        all_global_params = set()
        for decision in selected_decisions:
            decision_config = decisions_config.get(decision, {})
            global_params = decision_config.get('uses_global_parameters', [])
            all_global_params.update(global_params)
        
        return all_global_params
    except Exception as e:
        print(f"Error reading decision parameters: {e}")
        return set()


def get_all_global_parameters() -> set:
    """Get all possible global parameters from simulation.yaml"""
    try:
        simulation_path = Path(__file__).resolve().parents[1] / "config" / "simulation.yaml"
        simulation_config = read_yaml_file(simulation_path)
        
        return set(simulation_config.get('simulation', {}).keys())
    except Exception as e:
        print(f"Error reading simulation parameters: {e}")
        return set()


# All available decisions list
# NOTE: This order matches the chronological execution order in orchestrators
# Decisions execute in this sequence (1-13)
ALL_DECISIONS = [
    "disclose_income",               # 1
    "disclose_documents",            # 2
    "donation_default",              # 3
    "rejected_transaction_defaults", # 4
    "vendor_choice_weights",         # 5
    "purchasing_quantity",           # 6
    "purchasing_frequency",          # 7
    "vendor_selection",              # 8
    "purchase_vs_bid",               # 9 (deprecated - kept for backward compatibility)
    "bid_value",                     # 10 (deprecated - kept for backward compatibility)
    "rejected_transaction_option",   # 11
    "rejected_bid_value",            # 12
    "final_donation_rate"            # 13
]


def initialize_default_decision_parameters():
    """
    Initialize all default decision parameters in session state at app startup.
    
    This ensures parameter values persist across reruns even when widgets are not rendered.
    CRITICAL: This prevents state loss when decisions are selected/unselected.
    """
    # Import here to avoid circular dependency
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    
    for decision_name, default_value in DEFAULT_DECISION_VALUES.items():
        if isinstance(default_value, dict):
            decision_type = default_value.get("type")
            
            # Handle random probability decisions (disclose_income, disclose_documents, purchase_vs_bid)
            if decision_type == "random_probability":
                key = f"{decision_name}_default_probability_y"
                if key not in st.session_state:
                    st.session_state[key] = default_value.get("probability_y", 0.5)
            
            # Handle checkbox selection decisions (vendor_choice_weights)
            elif decision_type == "checkbox_selection":
                key = f"{decision_name}_default_params"
                if key not in st.session_state:
                    st.session_state[key] = default_value.get("default_selection", []).copy()
                
                # Initialize individual checkbox keys
                parameters = default_value.get("parameters", {})
                default_selection = default_value.get("default_selection", [])
                for param_key in parameters.keys():
                    checkbox_key = f"{decision_name}_default_param_{param_key}"
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = param_key in default_selection
            
            # Handle radio selection decisions (rejected_transaction_defaults, rejected_transaction_option)
            elif decision_type == "radio_selection":
                key = f"{decision_name}_default_selection"
                if key not in st.session_state:
                    st.session_state[key] = default_value.get("default_option", "")
            
            # Handle prioritized selection decisions (rejected_transaction_defaults)
            elif decision_type == "prioritized_selection":
                key = f"{decision_name}_priority_template"
                if key not in st.session_state:
                    st.session_state[key] = default_value.get("priority_template", []).copy()
        
        else:
            # Handle numeric or string placeholder values
            key = f"{decision_name}_default_value"
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    # Mark as initialized (kept for debugging, but logic now allows re-checking)
    st.session_state._default_params_initialized = True



