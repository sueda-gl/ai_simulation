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
    
    # Pareto parameters  
    pareto_alpha: float = 2.5  # Shape parameter (tail index)
    pareto_x_m: float = 1000.0  # Minimum value (scale parameter)
    pareto_max: Optional[float] = None  # Maximum value (rejection sampling)
    
    # Weibull parameters
    weibull_k: float = 2.0  # Shape parameter (k)
    weibull_lambda: float = 10000.0  # Scale parameter (λ)
    weibull_min: float = 0.0  # Minimum value (linear shift)
    weibull_max: Optional[float] = None  # Maximum value (rejection sampling)
    
    # Income categories
    num_discount_categories: int = 10  # ✅ Changed from 3 to 10
    num_fixed_categories: int = 10  # ✅ Changed from 5 to 10
    
    # Consumption limits
    apply_consumption_limits: bool = False  # ✅ Changed from True to False (unchecked)
    consumption_limits: Dict[str, float] = field(default_factory=dict)
    consumption_limits_source: str = "manual"  # "manual" or "upload"
    
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
            
        elif self.income_distribution == "pareto":
            # Use user-specified x_m (minimum) and alpha parameters
            x_m = self.pareto_x_m
            alpha = self.pareto_alpha
            
            # Pareto distribution with scale parameter x_m
            # Note: scipy's pareto is defined as P(X > x) = (x_m/x)^alpha for x >= x_m
            samples = stats.pareto.rvs(b=alpha, scale=x_m, size=n_samples, random_state=rng)
            
            # Apply rejection sampling if maximum is set
            if self.pareto_max is not None:
                # Keep resampling values that exceed the maximum
                max_iterations = 1000  # Prevent infinite loops
                for _ in range(max_iterations):
                    mask = samples > self.pareto_max
                    if not np.any(mask):
                        break
                    # Resample values that are too high
                    n_resample = np.sum(mask)
                    samples[mask] = stats.pareto.rvs(b=alpha, scale=x_m, size=n_resample, random_state=rng)
                
                # Final clip to ensure no values exceed max
                samples = np.clip(samples, x_m, self.pareto_max)
            
        elif self.income_distribution == "weibull":
            # Use user-specified k (shape) and lambda (scale) parameters
            k = self.weibull_k
            lambda_param = self.weibull_lambda
            
            # Sample from Weibull distribution
            Y = stats.weibull_min.rvs(c=k, scale=lambda_param, size=n_samples, random_state=rng)
            
            # Apply linear shift (X = a + Y)
            samples = self.weibull_min + Y
            
            # Apply rejection sampling if maximum is set
            if self.weibull_max is not None:
                # Keep resampling values that exceed the maximum
                max_iterations = 1000  # Prevent infinite loops
                for _ in range(max_iterations):
                    mask = samples > self.weibull_max
                    if not np.any(mask):
                        break
                    # Resample values that are too high
                    n_resample = np.sum(mask)
                    Y_new = stats.weibull_min.rvs(c=k, scale=lambda_param, size=n_resample, random_state=rng)
                    samples[mask] = self.weibull_min + Y_new
                
                # Final clip to ensure no values exceed max
                samples = np.clip(samples, self.weibull_min, self.weibull_max)
        
        else:
            # Fallback to uniform distribution
            samples = rng.uniform(self.income_min, self.income_max, n_samples)
        
        # Clip to specified bounds
        samples = np.clip(samples, self.income_min, self.income_max)
        
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
            
        # Add pareto parameters if missing
        if not hasattr(sim_params, 'pareto_x_m'):
            sim_params.pareto_x_m = 1000.0
        if not hasattr(sim_params, 'pareto_max'):
            sim_params.pareto_max = None
            
        # Add weibull parameters if missing
        if not hasattr(sim_params, 'weibull_k'):
            # Check if old weibull_shape exists and migrate it
            if hasattr(sim_params, 'weibull_shape'):
                sim_params.weibull_k = sim_params.weibull_shape
            else:
                sim_params.weibull_k = 2.0
        if not hasattr(sim_params, 'weibull_lambda'):
            sim_params.weibull_lambda = 10000.0
        if not hasattr(sim_params, 'weibull_min'):
            sim_params.weibull_min = 0.0
        if not hasattr(sim_params, 'weibull_max'):
            sim_params.weibull_max = None
            
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


