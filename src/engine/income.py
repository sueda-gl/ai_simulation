# src/engine/income.py
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
from typing import Tuple


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

def _has_income(agent_state: dict) -> bool:
    """True when agent_state already carries a usable income (NaN counts as absent)."""
    income = agent_state.get('income', None)
    if income is None:
        return False
    try:
        return not np.isnan(float(income))
    except (TypeError, ValueError):
        return False


def _ensure_actual_allowance(agent_state: dict) -> None:
    """
    Derive 'actual_allowance' (12-200 credit scale) from 'Assigned Allowance Level'
    if it has not been set yet.

    'actual_allowance' is a deterministic mapping, independent of the income draw, so
    it must exist whenever income does - including when income was pre-attached to the
    agent (Research Baseline / Research Specification carry the fixed reference income).
    Silently does nothing when the level is missing; the caller that actually needs a
    level (the generation path) raises instead.
    """
    if agent_state.get('actual_allowance', None) is not None:
        return
    level = agent_state.get('Assigned Allowance Level')
    if level is None:
        return
    try:
        level = int(level)
    except (TypeError, ValueError):
        return
    agent_state['actual_allowance'] = float(ALLOWANCE_CREDIT_MAPPING.get(level, 200))


def get_agent_income(agent_state: dict, simulation_config: dict, rng: np.random.Generator) -> float:
    """
    Get or generate agent income - CATEGORY-FIRST ARCHITECTURE.
    
    This is the MAIN entry point for all decisions needing large-scale dollar income.
    
    LOGIC:
    ======
    1. If 'income' already exists in agent_state, return it (cached) - but still make
       sure 'actual_allowance' is derived from 'Assigned Allowance Level', since the
       cached income may have been supplied by the caller rather than generated here
       (the parity tests inject the reference Stata income this way; the app never
       does - every population mode generates it here from the Page 1 distribution)
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

    # Check if income already exists (already processed, or pre-attached by the caller)
    if _has_income(agent_state):
        # The derived 12-200 allowance credit is NOT part of the income draw - it is a
        # deterministic function of 'Assigned Allowance Level' - so it must be set here
        # too, otherwise a pre-attached income would leave donation_default's
        # get_actual_allowance() without its regressor.
        _ensure_actual_allowance(agent_state)
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
    
    # Otherwise, trigger income generation (which generates both). When income was
    # pre-attached, get_agent_income() only derives the allowance credit, which needs
    # 'Assigned Allowance Level' - so surface a clear error if that is missing too.
    get_agent_income(agent_state, simulation_config, rng)

    if agent_state.get('actual_allowance', None) is None:
        raise ValueError("Agent missing 'Assigned Allowance Level' - cannot derive "
                         "'actual_allowance' (income was supplied without a level)")

    return float(agent_state['actual_allowance'])
