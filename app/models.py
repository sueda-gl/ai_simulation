# app/models.py
"""
Data models and session state management for the Enhanced AI Agent Simulation.
"""
import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import numpy as np
from scipy import stats
from scipy.stats import gengamma


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
    apply_consumption_limits: bool = False  # ✅ Changed from True to False (unchecked)
    consumption_limits: Dict[str, float] = field(default_factory=dict)
    consumption_limits_source: str = "manual"  # "manual" or "upload"
    max_purchases_per_term: int = 50  # Fallback maximum when consumption limits disabled
    
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
    
    def validate_vendor_products_avg(self) -> bool:
        """Validate that average products per vendor is within min/max range"""
        return self.vendor_products_min <= self.vendor_products_avg <= self.vendor_products_max
    
    def get_total_products_range(self) -> Tuple[int, int]:
        """Get the total products range across all vendors"""
        min_total = self.vendor_products_min * self.num_vendors
        max_total = self.vendor_products_max * self.num_vendors
        return min_total, max_total
    
    def get_expected_total_products(self) -> int:
        """Get expected total products across all vendors"""
        return self.vendor_products_avg * self.num_vendors
    
    def sample_income_distribution(self, n_samples: int = 1000, seed: int = 42) -> np.ndarray:
        """Sample from the configured income distribution"""
        rng = np.random.default_rng(seed)
        
        if self.income_distribution == "lognormal":
            # Use user-specified mu and sigma parameters
            mu = self.lognormal_mu
            sigma = self.lognormal_sigma
            
            # Sample from lognormal distribution
            Y = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_samples, random_state=rng)
            
            # Apply linear shift (X = a + Y)
            samples = self.lognormal_min + Y
            
            # Apply rejection sampling if maximum is set
            if self.lognormal_max is not None:
                # Keep resampling values that exceed the maximum
                max_iterations = 1000  # Prevent infinite loops
                for _ in range(max_iterations):
                    mask = samples > self.lognormal_max
                    if not np.any(mask):
                        break
                    # Resample values that are too high
                    n_resample = np.sum(mask)
                    Y_new = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_resample, random_state=rng)
                    samples[mask] = self.lognormal_min + Y_new
                
                # Final clip to ensure no values exceed max
                samples = np.clip(samples, self.lognormal_min, self.lognormal_max)
            
        elif self.income_distribution == "generalised_gamma":
            # Use user-specified k, c, and lambda parameters
            k = self.gg_k
            c = self.gg_c
            lambda_param = self.gg_lambda
            
            # Sample from Generalised Gamma distribution
            # scipy.stats.gengamma uses (a, c, scale) parameterization
            # where a=c (shape2), c=k (shape1), scale=lambda
            Y = stats.gengamma.rvs(a=c, c=k, scale=lambda_param, size=n_samples, random_state=rng)
            
            # Apply linear shift (X = a + Y)
            samples = self.gg_min + Y
            
            # Apply rejection sampling if maximum is set
            if self.gg_max is not None:
                # Keep resampling values that exceed the maximum
                max_iterations = 1000  # Prevent infinite loops
                for _ in range(max_iterations):
                    mask = samples > self.gg_max
                    if not np.any(mask):
                        break
                    # Resample values that are too high
                    n_resample = np.sum(mask)
                    Y_new = stats.gengamma.rvs(a=c, c=k, scale=lambda_param, size=n_resample, random_state=rng)
                    samples[mask] = self.gg_min + Y_new
                
                # Final clip to ensure no values exceed max
                samples = np.clip(samples, self.gg_min, self.gg_max)
            
        elif self.income_distribution == "dagum":
            # Use user-specified a (tail), p (body), and b (scale) parameters
            a = self.dagum_a
            p = self.dagum_p
            b = self.dagum_b
            
            # Sample from Dagum distribution using inverse CDF method
            # Dagum CDF: F(x) = (1 + (x/b)^(-a))^(-p)
            # Inverse CDF: x = b * ((U^(-1/p) - 1)^(-1/a))
            U = rng.random(n_samples)
            samples = b * np.power(np.power(U, -1/p) - 1, -1/a)
            
            # Apply linear shift
            samples = self.dagum_min + samples
            
            # Apply rejection sampling if maximum is set
            if self.dagum_max is not None:
                # Keep resampling values that exceed the maximum
                max_iterations = 1000  # Prevent infinite loops
                for _ in range(max_iterations):
                    mask = samples > self.dagum_max
                    if not np.any(mask):
                        break
                    # Resample values that are too high
                    n_resample = np.sum(mask)
                    U_new = rng.random(n_resample)
                    new_values = b * np.power(np.power(U_new, -1/p) - 1, -1/a)
                    samples[mask] = self.dagum_min + new_values
                
                # Final clip to ensure no values exceed max
                samples = np.clip(samples, self.dagum_min, self.dagum_max)
        
        else:
            # Fallback to uniform distribution
            samples = rng.uniform(self.income_min, self.income_max, n_samples)
        

        return samples
    
    def get_discount_qualification_rate(self, n_samples: int = 1000) -> float:
        """Calculate the percentage of agents that would qualify for discounts"""
        samples = self.sample_income_distribution(n_samples)
        qualified = np.sum(samples <= self.discount_income_threshold)
        return qualified / len(samples)


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
            
        # Migrate old vendor price/product values to new defaults
        # This ensures users get the updated default values
        # Check for various old values that need migration
        if hasattr(sim_params, 'vendor_price_min'):
            if sim_params.vendor_price_min in [8.0, 100.0]:  # Old values
                sim_params.vendor_price_min = 50.0
        if hasattr(sim_params, 'vendor_price_max'):
            if sim_params.vendor_price_max in [12.0, 100.0]:  # Old values  
                sim_params.vendor_price_max = 150.0
        if hasattr(sim_params, 'market_price'):
            if sim_params.market_price == 10.0:  # Old value
                sim_params.market_price = 100.0
        if hasattr(sim_params, 'vendor_products_min'):
            if sim_params.vendor_products_min in [1, 100]:  # Old values
                sim_params.vendor_products_min = 50
        if hasattr(sim_params, 'vendor_products_max'):
            if sim_params.vendor_products_max in [1, 100]:  # Old values
                sim_params.vendor_products_max = 150
            
    if 'decision_params' not in st.session_state:
        st.session_state.decision_params = DecisionParameters()
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'mc_results' not in st.session_state:
        st.session_state.mc_results = None
    
    # Add missing defaults used in configure_sidebar and simulation
    defaults = {
        'population_mode': 'Copula (synthetic)',
        'income_spec_mode': 'categorical only',
        'sigma_in_copula': False,
        'sigma_in_research': True,  # Enable sigma in Research mode by default
        'sigma_value_ui': 9.8995,  # Static empirical SD value
        'sigma_coefficient': 1.0,  # Coefficient to multiply the static SD (0-2)
        'anchor_observed_weight': 0.75,
        'raw_draw_mode': False,
        'n_agents': 1000,
        'seed': 42,
        'n_runs': 10,
        'base_seed': 42,
        'show_individual_agents': False,
        'save_results': True,
        'simulation_running': False,
        'individual_results': {}  # New: store individual decision results
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def load_donation_coefficients_from_yaml():
    """Load donation_default coefficients from YAML config into session state variables
    
    IMPORTANT: YAML is the SINGLE source of truth. No fallback values are used.
    If coefficients are missing from YAML, an error will be raised.
    """
    config_path = Path(__file__).parent.parent / "config" / "decisions.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
        with open(decisions_path, 'r') as f:
            decisions_config = yaml.safe_load(f)
        
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
        with open(simulation_path, 'r') as f:
            simulation_config = yaml.safe_load(f)
        
        return set(simulation_config.get('simulation', {}).keys())
    except Exception as e:
        print(f"Error reading simulation parameters: {e}")
        return set()


# All available decisions list
ALL_DECISIONS = [
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


