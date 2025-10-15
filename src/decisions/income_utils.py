# src/decisions/income_utils.py
"""
Centralized income generation utilities.

This module is the SINGLE SOURCE OF TRUTH for income generation across all decisions.
All decisions that need to generate or retrieve agent income should use these functions.
"""

import numpy as np
from scipy import stats
from typing import Optional, Dict


def get_agent_income(agent_state: dict, simulation_config: dict, rng: np.random.Generator) -> float:
    """
    Get or generate agent income - the MAIN entry point for all decisions.
    
    This is the single source of truth for income retrieval/generation.
    
    Logic:
    1. If income already exists in agent_state, return it
    2. Otherwise, generate using Page 1 distribution parameters
    3. Store in agent_state for reuse by other decisions
    
    Args:
        agent_state: Agent's state dict (may already contain 'income')
        simulation_config: Full simulation configuration from orchestrator
        rng: Random number generator for reproducibility
        
    Returns:
        float: Agent's annual income in dollars
        
    Example:
        income = get_agent_income(agent_state, simulation_config, rng)
        # Returns existing income or generates new one
    """
    
    # Check if income already exists
    if 'income' in agent_state and agent_state['income'] is not None:
        return float(agent_state['income'])
    
    # Generate income using Page 1 distribution parameters
    income = generate_income_from_distribution(simulation_config, rng)
    
    # Store in agent_state for other decisions to use
    agent_state['income'] = income
    
    return income


def generate_income_from_distribution(simulation_config: dict, rng: np.random.Generator) -> float:
    """
    Generate a single income value using Page 1 distribution parameters.
    
    Supports three distribution types:
    - lognormal: X = min + Y where Y ~ Lognormal(mu, sigma)
    - generalised_gamma: X = min + Y where Y ~ GenGamma(k, c, lambda)
    - dagum: X = min + Y where Y ~ Dagum(a, p, b)
    
    Args:
        simulation_config: Contains Page 1 parameters in ['simulation'] sub-dict
        rng: Random number generator for reproducibility
        
    Returns:
        float: Generated income value in dollars
        
    Example:
        income = generate_income_from_distribution(simulation_config, rng)
        # Returns: 45678.92
    """
    
    if not simulation_config or not isinstance(simulation_config, dict):
        # Fallback if no config provided
        return 50000.0
    
    sim_params = simulation_config.get('simulation', {})
    dist_type = sim_params.get('income_distribution', 'lognormal')
    
    if dist_type == 'lognormal':
        return _generate_lognormal_income(sim_params, rng)
    
    elif dist_type == 'generalised_gamma':
        return _generate_generalised_gamma_income(sim_params, rng)
    
    elif dist_type == 'dagum':
        return _generate_dagum_income(sim_params, rng)
    
    else:
        # Fallback: uniform distribution (legacy)
        min_val = sim_params.get('income_min', 0.0)
        max_val = sim_params.get('income_max', 100000.0)
        return float(rng.uniform(min_val, max_val))


def _generate_lognormal_income(sim_params: dict, rng: np.random.Generator) -> float:
    """
    Generate income from lognormal distribution.
    
    Formula: X = a + Y where Y ~ Lognormal(mu, sigma)
    
    Args:
        sim_params: Page 1 simulation parameters
        rng: Random number generator
        
    Returns:
        float: Generated income
    """
    mu = sim_params.get('lognormal_mu', 10.0)
    sigma = sim_params.get('lognormal_sigma', 0.5)
    min_val = sim_params.get('lognormal_min', 0.0)
    max_val = sim_params.get('lognormal_max', None)
    
    # Sample: Y ~ Lognormal(mu, sigma)
    Y = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=1, random_state=rng)[0]
    income = min_val + Y
    
    # Apply maximum if specified
    if max_val is not None:
        income = min(income, max_val)
    
    return float(income)


def _generate_generalised_gamma_income(sim_params: dict, rng: np.random.Generator) -> float:
    """
    Generate income from Generalised Gamma distribution.
    
    Formula: X = a + Y where Y ~ GenGamma(k, c, lambda)
    
    Args:
        sim_params: Page 1 simulation parameters
        rng: Random number generator
        
    Returns:
        float: Generated income
    """
    k = sim_params.get('gg_k', 1.5)
    c = sim_params.get('gg_c', 2.0)
    lambda_param = sim_params.get('gg_lambda', 20000.0)
    min_val = sim_params.get('gg_min', 0.0)
    max_val = sim_params.get('gg_max', None)
    
    # Sample from Generalised Gamma
    # scipy uses (a=c, c=k, scale=lambda) parameterization
    Y = stats.gengamma.rvs(a=c, c=k, scale=lambda_param, size=1, random_state=rng)[0]
    income = min_val + Y
    
    # Apply maximum if specified
    if max_val is not None:
        income = min(income, max_val)
    
    return float(income)


def _generate_dagum_income(sim_params: dict, rng: np.random.Generator) -> float:
    """
    Generate income from Dagum (Type I) distribution.
    
    Formula: X = min + Y where Y ~ Dagum(a, p, b)
    Using inverse CDF: Y = b * ((U^(-1/p) - 1)^(-1/a))
    
    Args:
        sim_params: Page 1 simulation parameters
        rng: Random number generator
        
    Returns:
        float: Generated income
    """
    a = sim_params.get('dagum_a', 2.0)
    p = sim_params.get('dagum_p', 1.5)
    b = sim_params.get('dagum_b', 25000.0)
    min_val = sim_params.get('dagum_min', 0.0)
    max_val = sim_params.get('dagum_max', None)
    
    # Sample using inverse CDF
    U = rng.random()
    Y = b * np.power(np.power(U, -1/p) - 1, -1/a)
    income = min_val + Y
    
    # Apply maximum if specified
    if max_val is not None:
        income = min(income, max_val)
    
    return float(income)


def get_simulation_param(simulation_config: Optional[Dict], key: str, default=None):
    """
    Safely extract parameter from simulation_config['simulation'].
    
    Standard helper used by all decisions to access Page 1 parameters.
    
    Args:
        simulation_config: Full simulation configuration dict
        key: Parameter name to retrieve
        default: Fallback value if not found
        
    Returns:
        Parameter value or default
        
    Example:
        threshold = get_simulation_param(simulation_config, 'discount_income_threshold', 12500.0)
    """
    if not simulation_config or not isinstance(simulation_config, dict):
        return default
    
    sim = simulation_config.get('simulation', {})
    return sim.get(key, default)


def determine_customer_type(agent_state: dict, simulation_config: dict) -> str:
    """
    Determine customer type based on disclosure decisions and income.
    
    This should be called AFTER Decision 1 (disclose_income) and Decision 2 (disclose_documents).
    
    Customer Type Logic:
    1. DISCOUNT: income < threshold AND disclose_documents = "Y"
       - Qualifies for discount pricing
       - Only available to low-income agents who submitted documents
    
    2. FIXED: disclose_income = "Y" (regardless of income level)
       - Fixed prices, no bidding option
       - Available to anyone who disclosed their income upfront
    
    3. REGULAR: Everyone else (disclose_income = "N")
       - Purchase Now or Bid options
       - Default customer type for non-disclosers
    
    Args:
        agent_state: Agent's state dict containing:
            - income: Agent's annual income
            - disclose_income: "Y" or "N" (from Decision 1)
            - disclose_documents: "Y", "N", or "NA" (from Decision 2)
        simulation_config: Contains discount_income_threshold from Page 1
        
    Returns:
        str: "discount", "fixed", or "regular"
        
    Example:
        customer_type = determine_customer_type(agent_state, simulation_config)
        # Returns: "discount" if low income + documents submitted
    """
    
    # Get required values
    income = agent_state.get('income', 0)
    disclose_income = agent_state.get('disclose_income', 'N')
    disclose_documents = agent_state.get('disclose_documents', 'NA')
    
    # Get threshold from Page 1
    threshold = get_simulation_param(simulation_config, 'discount_income_threshold', 12500.0)
    
    # Priority 1: Check for DISCOUNT customer
    # Must have low income AND submitted documents
    if income <= threshold and disclose_documents == "Y":
        return "discount"
    
    # Priority 2: Check for FIXED customer
    # Disclosed income upfront (regardless of income level)
    if disclose_income == "Y":
        return "fixed"
    
    # Priority 3: REGULAR customer (default)
    # Everyone else - can use Purchase Now or Bid
    return "regular"


def get_customer_type(agent_state: dict, simulation_config: dict) -> str:
    """
    Get or determine customer type - with caching.
    
    This is the main entry point for getting customer type.
    If already determined, returns cached value.
    Otherwise, determines and stores it.
    
    Args:
        agent_state: Agent's state dict
        simulation_config: Full simulation configuration
        
    Returns:
        str: "discount", "fixed", or "regular"
        
    Example:
        customer_type = get_customer_type(agent_state, simulation_config)
    """
    
    # Check if already determined
    if 'customer_type' in agent_state and agent_state['customer_type'] is not None:
        return agent_state['customer_type']
    
    # Determine customer type
    customer_type = determine_customer_type(agent_state, simulation_config)
    
    # Store for reuse by other decisions
    agent_state['customer_type'] = customer_type
    
    return customer_type


def analyze_customer_types(results_df) -> Dict[str, any]:
    """
    Analyze customer type distribution from simulation results.
    
    Provides counts and percentages for each customer type.
    
    Args:
        results_df: DataFrame with simulation results (must have 'customer_type' column)
        
    Returns:
        dict: Statistics about customer type distribution
        
    Example:
        stats = analyze_customer_types(results_df)
        # Returns: {
        #   'discount': {'count': 45, 'percentage': 15.0},
        #   'fixed': {'count': 135, 'percentage': 45.0},
        #   'regular': {'count': 120, 'percentage': 40.0},
        #   'total': 300
        # }
    """
    
    if 'customer_type' not in results_df.columns:
        return {
            'error': 'customer_type column not found in results',
            'discount': {'count': 0, 'percentage': 0.0},
            'fixed': {'count': 0, 'percentage': 0.0},
            'regular': {'count': 0, 'percentage': 0.0},
            'total': 0
        }
    
    total = len(results_df)
    
    if total == 0:
        return {
            'discount': {'count': 0, 'percentage': 0.0},
            'fixed': {'count': 0, 'percentage': 0.0},
            'regular': {'count': 0, 'percentage': 0.0},
            'total': 0
        }
    
    # Count each type
    discount_count = len(results_df[results_df['customer_type'] == 'discount'])
    fixed_count = len(results_df[results_df['customer_type'] == 'fixed'])
    regular_count = len(results_df[results_df['customer_type'] == 'regular'])
    
    return {
        'discount': {
            'count': discount_count,
            'percentage': (discount_count / total) * 100
        },
        'fixed': {
            'count': fixed_count,
            'percentage': (fixed_count / total) * 100
        },
        'regular': {
            'count': regular_count,
            'percentage': (regular_count / total) * 100
        },
        'total': total
    }

