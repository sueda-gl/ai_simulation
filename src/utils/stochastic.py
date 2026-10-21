# src/utils/stochastic.py
"""
Centralized stochastic component utilities for decision functions.

This module provides reusable functions for applying stochastic (random) variation
to decision models. It supports:
- Overall sigma (single value for all agents)
- Quintile-specific sigma (different values per income level)
- Configurable scale factors
- Automatic unit conversion (original → z-score scale)

Usage:
    from src.utils.stochastic import get_stochastic_sigma, should_use_stochastic

    if should_use_stochastic(stochastic_params, pop_context):
        sigma = get_stochastic_sigma(level, stochastic_params)
        draw = rng.normal(anchor, sigma)
"""

from typing import Dict, Any


# Default sigma values from empirical data (TWT+Sospeso observed prosocial behavior)
DEFAULT_SIGMA_OVERALL = 9.899547

# Quintile-specific sigma values (SD of TWT+Sospeso within each income quintile)
DEFAULT_SIGMA_QUINTILE = {
    '1': 5.705052,   # Level 1 (€12)
    '2': 3.069326,   # Level 2 (€32)
    '3': 3.532226,   # Level 3 (€72)
    '4': 12.219622,  # Level 4 (€128)
    '5': 16.854622,  # Level 5 (€200)
}


def should_use_stochastic(
    stochastic_params: Dict[str, Any],
    pop_context: str
) -> bool:
    """
    The mode's stochastic tick box, as ONE rule for every decision that draws noise.

    Logic:
    - 'copula'        -> stochastic_params['in_copula'] (the copula tick)
    - 'documentation' -> stochastic_params['sigma_value'] > 0 (the research tick)
    - 'baseline'      -> OFF, always (Research Baseline never adds noise)
    - anything else   -> OFF

    Args:
        stochastic_params: Dict containing sigma_value, in_copula, etc.
        pop_context: Population context ('documentation', 'baseline', 'copula')

    Returns:
        True if the stochastic component should be applied
    """
    stochastic_params = stochastic_params or {}
    sigma_value = stochastic_params.get('sigma_value', 0)

    return bool(
        (stochastic_params.get('in_copula', False) and pop_context == 'copula') or
        (pop_context == 'documentation' and sigma_value > 0)
    )


def get_stochastic_sigma(
    level: int,
    stochastic_params: Dict[str, Any],
    convert_to_z_scale: bool = False,
    z_scale_divisor: float = DEFAULT_SIGMA_OVERALL
) -> float:
    """
    Get sigma value based on strategy and income level.

    Supports two strategies:
    - 'overall': Single sigma for all agents
    - 'quintile': Level-specific sigma values

    Args:
        level: Income level (1-5)
        stochastic_params: Dict with:
            - sigma_strategy: 'overall' or 'quintile'
            - sigma_overall: Base sigma for overall mode
            - sigma_quintile: Dict mapping level to sigma for quintile mode
            - scale_factor: Multiplier for overall mode
            - quintile_scale_factors: Dict mapping level to multiplier for quintile mode
        convert_to_z_scale: If True, divide sigma by z_scale_divisor to convert
                           from original units to z-score units
        z_scale_divisor: Value to divide by when converting to z-score scale
                        (typically the overall SD of the original variable)

    Returns:
        Effective sigma value (raw * scale_factor, optionally converted to z-scale)
    """
    strategy = stochastic_params.get('sigma_strategy', 'overall')

    if strategy == 'quintile':
        # Quintile-specific sigma
        sigma_quintile = stochastic_params.get('sigma_quintile', DEFAULT_SIGMA_QUINTILE)
        sigma_raw = float(sigma_quintile.get(str(level), stochastic_params.get('sigma_overall', DEFAULT_SIGMA_OVERALL)))

        # Quintile-specific scale factor (falls back to overall scale_factor)
        quintile_scale_factors = stochastic_params.get('quintile_scale_factors', {})
        scale_factor = float(quintile_scale_factors.get(str(level), stochastic_params.get('scale_factor', 1.0)))
    else:
        # Overall sigma (default)
        sigma_raw = float(stochastic_params.get('sigma_overall', DEFAULT_SIGMA_OVERALL))
        scale_factor = float(stochastic_params.get('scale_factor', 1.0))

    sigma_scaled = sigma_raw * scale_factor

    # Convert to z-score scale if requested
    if convert_to_z_scale and z_scale_divisor > 0:
        sigma_scaled = sigma_scaled / z_scale_divisor

    return sigma_scaled
