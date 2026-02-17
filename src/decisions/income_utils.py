# src/decisions/income_utils.py
"""
Centralized income generation utilities - CATEGORY-FIRST ARCHITECTURE.

This module is the SINGLE SOURCE OF TRUTH for income generation across all decisions.

ARCHITECTURE OVERVIEW (Category-First):
======================================
1. Source of Truth: 'Assigned Allowance Level' (1-5) from TraitEngine/Copula
2. Derived Values (generated once per agent, cached in agent_state):
   
   a) 'actual_allowance' (float, 12-200 scale)
      - Deterministic mapping: Level 1→12, 2→32, 3→72, 4→128, 5→200
      - Used ONLY by donation_default regression (trained on this scale)
   
   b) 'income' (float, large €/$ scale)
      - Stochastic draw from percentile bucket defined by Assigned Allowance Level
      - Uses PPF (inverse CDF) to map Level → income quintile → random $ amount
      - Used by all other decisions (discount thresholds, histograms, etc.)

WHY THIS MATTERS:
================
- Preserves trait correlations from original copula model
- Prevents logical inconsistencies (high level + low income or vice versa)
- Ensures regression uses correct scale (12-200) while other decisions use realistic dollars
- Single generation per agent guarantees consistency across all decisions
"""

import numpy as np
from scipy import stats
from typing import Optional, Dict, Tuple


# ============================================================================
# ALLOWANCE CREDIT MAPPING (12-200 scale for regression)
# ============================================================================
ALLOWANCE_CREDIT_MAPPING = {
    1: 12,   # Lowest income category
    2: 32,
    3: 72,
    4: 128,
    5: 200   # Highest income category
}


# ============================================================================
# PERCENTILE BOUNDARY CALCULATION (PPF-based)
# ============================================================================

def _get_distribution_object(sim_params: dict):
    """
    Create scipy distribution object from Page 1 parameters.
    
    Returns:
        A scipy.stats distribution object with a .ppf() method
    """
    dist_type = sim_params.get('income_distribution', 'lognormal')
    
    if dist_type == 'lognormal':
        mu = sim_params.get('lognormal_mu', 10.0)
        sigma = sim_params.get('lognormal_sigma', 0.5)
        min_val = sim_params.get('lognormal_min', 0.0)
        # Return shifted lognormal
        return stats.lognorm(s=sigma, scale=np.exp(mu), loc=min_val)
    
    elif dist_type == 'generalised_gamma':
        k = sim_params.get('gg_k', 1.5)
        c = sim_params.get('gg_c', 2.0)
        lambda_param = sim_params.get('gg_lambda', 20000.0)
        min_val = sim_params.get('gg_min', 0.0)
        return stats.gengamma(a=c, c=k, scale=lambda_param, loc=min_val)
    
    elif dist_type == 'dagum':
        # Dagum requires custom handling - return None and handle separately
        return None
    
    else:
        # Fallback: uniform distribution
        min_val = sim_params.get('income_min', 0.0)
        max_val = sim_params.get('income_max', 100000.0)
        return stats.uniform(loc=min_val, scale=max_val - min_val)


def get_percentile_boundaries(sim_params: dict) -> list:
    """
    Calculate dollar values for 20th, 40th, 60th, 80th percentiles.
    
    These boundaries define the income "buckets" for each Assigned Allowance Level:
    - Level 1: [0, p20]
    - Level 2: [p20, p40]
    - Level 3: [p40, p60]
    - Level 4: [p60, p80]
    - Level 5: [p80, ∞)
    
    Args:
        sim_params: Page 1 simulation parameters dict
        
    Returns:
        list: [p20, p40, p60, p80] dollar boundary values
    """
    dist = _get_distribution_object(sim_params)
    
    # Handle Dagum separately (requires inverse CDF formula)
    if dist is None and sim_params.get('income_distribution') == 'dagum':
        a = sim_params.get('dagum_a', 2.0)
        p = sim_params.get('dagum_p', 1.5)
        b = sim_params.get('dagum_b', 25000.0)
        min_val = sim_params.get('dagum_min', 0.0)
        
        boundaries = []
        for percentile in [0.20, 0.40, 0.60, 0.80]:
            Y = b * np.power(np.power(percentile, -1/p) - 1, -1/a)
            boundaries.append(min_val + Y)
        return boundaries
    
    # For other distributions, use PPF
    percentiles = [0.20, 0.40, 0.60, 0.80]
    boundaries = [dist.ppf(p) for p in percentiles]
    
    # Apply max clipping if specified
    dist_type = sim_params.get('income_distribution', 'lognormal')
    max_val = None
    
    if dist_type == 'lognormal':
        max_val = sim_params.get('lognormal_max', None)
    elif dist_type == 'generalised_gamma':
        max_val = sim_params.get('gg_max', None)
    elif dist_type == 'dagum':
        max_val = sim_params.get('dagum_max', None)
    
    if max_val is not None:
        boundaries = [min(b, max_val) for b in boundaries]
    
    return boundaries


def _get_percentile_range_for_level(level: int) -> Tuple[float, float]:
    """
    Map Assigned Allowance Level (1-5) to percentile range.
    
    Args:
        level: Assigned Allowance Level (1-5)
        
    Returns:
        tuple: (lower_percentile, upper_percentile)
        
    Example:
        Level 3 → (0.40, 0.60)  # Middle 20% of distribution
    """
    percentile_ranges = {
        1: (0.00, 0.20),  # Bottom 20%
        2: (0.20, 0.40),
        3: (0.40, 0.60),  # Middle 20%
        4: (0.60, 0.80),
        5: (0.80, 1.00)   # Top 20% (no upper bound for many distributions)
    }
    return percentile_ranges.get(level, (0.80, 1.00))


def _generate_income_within_percentile_range(
    sim_params: dict,
    percentile_low: float,
    percentile_high: float,
    rng: np.random.Generator
) -> float:
    """
    Generate a random income within a specific percentile range using PPF.
    
    This is the core Category-First logic: we sample uniformly in percentile space,
    then convert to dollar space via the inverse CDF.
    
    Args:
        sim_params: Page 1 simulation parameters
        percentile_low: Lower percentile bound (e.g., 0.40)
        percentile_high: Upper percentile bound (e.g., 0.60)
        rng: Random number generator
        
    Returns:
        float: Dollar income within the specified percentile range
    """
    # Step 1: Draw uniform random percentile within range
    random_percentile = rng.uniform(percentile_low, percentile_high)
    
    # Step 2: Convert percentile to dollar value via PPF (inverse CDF)
    dist_type = sim_params.get('income_distribution', 'lognormal')
    
    if dist_type == 'dagum':
        # Dagum: use inverse CDF formula
        a = sim_params.get('dagum_a', 2.0)
        p = sim_params.get('dagum_p', 1.5)
        b = sim_params.get('dagum_b', 25000.0)
        min_val = sim_params.get('dagum_min', 0.0)
        max_val = sim_params.get('dagum_max', None)
        
        Y = b * np.power(np.power(random_percentile, -1/p) - 1, -1/a)
        income = min_val + Y
        
        if max_val is not None:
            income = min(income, max_val)
        
        return float(income)
    
    else:
        # All other distributions: use scipy PPF
        dist = _get_distribution_object(sim_params)
        income = dist.ppf(random_percentile)
        
        # Apply max clipping if specified
        max_val = None
        if dist_type == 'lognormal':
            max_val = sim_params.get('lognormal_max', None)
        elif dist_type == 'generalised_gamma':
            max_val = sim_params.get('gg_max', None)
        
        if max_val is not None:
            income = min(income, max_val)
        
        return float(income)


# ============================================================================
# MAIN INCOME GENERATION FUNCTIONS (Category-First)
# ============================================================================

def get_agent_income(agent_state: dict, simulation_config: dict, rng: np.random.Generator) -> float:
    """
    Get or generate agent income - CATEGORY-FIRST ARCHITECTURE.
    
    This is the MAIN entry point for all decisions needing large-scale dollar income.
    
    LOGIC:
    ======
    1. If 'income' already exists in agent_state, return it (cached)
    2. Otherwise, generate BOTH 'income' and 'actual_allowance' from the agent's
       'Assigned Allowance Level' and cache them in agent_state
    3. 'income' is drawn stochastically from the percentile bucket
    4. 'actual_allowance' is mapped deterministically (12-200 scale)
    
    This ensures:
    - One-time generation per agent
    - Logical consistency between categorical level and continuous income
    - Correct scale for regression (actual_allowance) vs. realistic decisions (income)
    
    Args:
        agent_state: Agent's state dict containing 'Assigned Allowance Level'
        simulation_config: Full simulation configuration from orchestrator
        rng: Random number generator for reproducibility
        
    Returns:
        float: Agent's annual income in large-scale dollars
        
    Example:
        income = get_agent_income(agent_state, simulation_config, rng)
        # First call: generates both 'income' ($47,500) and 'actual_allowance' (72)
        # Subsequent calls: returns cached $47,500
    """
    
    # Check if income already exists (already processed)
    if 'income' in agent_state and agent_state['income'] is not None:
        return float(agent_state['income'])
    
    # Get the agent's Assigned Allowance Level (the source of truth)
    level = agent_state.get('Assigned Allowance Level')
    
    if level is None:
        # Fallback for edge cases (shouldn't happen in normal flow)
        raise ValueError("Agent missing 'Assigned Allowance Level' - cannot generate income")
    
    level = int(level)
    
    # STEP 1: Generate actual_allowance (12-200 scale for regression)
    actual_allowance = ALLOWANCE_CREDIT_MAPPING.get(level, 200)
    agent_state['actual_allowance'] = float(actual_allowance)
    
    # STEP 2: Generate large-scale dollar income from percentile bucket
    sim_params = simulation_config.get('simulation', {})
    percentile_low, percentile_high = _get_percentile_range_for_level(level)
    
    income = _generate_income_within_percentile_range(
        sim_params,
        percentile_low,
        percentile_high,
        rng
    )
    
    # Store in agent_state for reuse
    agent_state['income'] = income
    
    return income


def get_actual_allowance(agent_state: dict, simulation_config: dict, rng: np.random.Generator) -> float:
    """
    Get actual allowance credit value (12-200 scale).
    
    This is used ONLY by donation_default regression.
    If not yet generated, triggers full income generation.
    
    Args:
        agent_state: Agent's state dict
        simulation_config: Full simulation configuration
        rng: Random number generator
        
    Returns:
        float: Allowance credit value on 12-200 scale
        
    Example:
        allowance = get_actual_allowance(agent_state, simulation_config, rng)
        # Returns: 72.0 (for Level 3 agent)
    """
    # If actual_allowance already exists, return it
    if 'actual_allowance' in agent_state and agent_state['actual_allowance'] is not None:
        return float(agent_state['actual_allowance'])
    
    # Otherwise, trigger income generation (which generates both)
    get_agent_income(agent_state, simulation_config, rng)
    
    return float(agent_state['actual_allowance'])


# ============================================================================
# LEGACY FUNCTIONS (No longer used in Category-First architecture)
# ============================================================================
# These functions are kept for backward compatibility but are NOT used
# in the new Category-First workflow. Income is now generated via PPF
# within percentile buckets defined by Assigned Allowance Level.

def generate_income_from_distribution(simulation_config: dict, rng: np.random.Generator) -> float:
    """
    [LEGACY - NOT USED IN CATEGORY-FIRST]
    
    Generate a single income value using Page 1 distribution parameters.
    
    NOTE: This function is no longer used in the Category-First architecture.
    Income is now generated via get_agent_income() which uses PPF-based
    percentile bucketing.
    
    Kept for backward compatibility only.
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
    1. DISCOUNT: disclose_income = "Y" AND income < threshold AND disclose_documents = "Y"
       - Qualifies for discount pricing
       - Only available to agents who disclosed income AND have low income AND submitted documents
    
    2. FIXED: disclose_income = "Y" (and not discount)
       - Fixed prices, no bidding option
       - Available to anyone who disclosed their income upfront
       - Includes agents who disclosed income but not documents, or have income above threshold
    
    3. REGULAR: disclose_income = "N"
       - Purchase Now or Bid options
       - Default customer type for non-disclosers
       - These agents are NEVER asked to disclose documents
    
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
        # Returns: "discount" if disclosed income + low income + documents submitted
    """
    
    # Get required values
    income = agent_state.get('income', 0)
    disclose_income = agent_state.get('disclose_income', 'N')
    disclose_documents = agent_state.get('disclose_documents', 'NA')
    
    # Get threshold from Page 1
    threshold = get_simulation_param(simulation_config, 'discount_income_threshold', 12500.0)
    
    # Priority 1: Check for DISCOUNT customer
    # Must have disclosed income AND low income AND submitted documents
    if disclose_income == "Y" and income <= threshold and disclose_documents == "Y":
        return "discount"
    
    # Priority 2: Check for FIXED customer
    # Disclosed income upfront (regardless of income level)
    # This includes agents who disclosed income but not documents, or income above threshold
    if disclose_income == "Y":
        return "fixed"
    
    # Priority 3: REGULAR customer (default)
    # Did not disclose income - can use Purchase Now or Bid
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

