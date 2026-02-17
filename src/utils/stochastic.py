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
    from src.utils.stochastic import get_stochastic_sigma, apply_stochastic_draw, should_use_stochastic

    if should_use_stochastic(stochastic_params, pop_context):
        sigma = get_stochastic_sigma(level, stochastic_params)
        draw = apply_stochastic_draw(anchor, sigma, rng)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


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
    Determine if stochastic component should be applied.

    Logic:
    - Documentation mode + sigma_value > 0 → ON
    - Copula mode + in_copula=True → ON
    - Baseline mode → OFF (always deterministic)
    - Otherwise → OFF

    Args:
        stochastic_params: Dict containing sigma_value, in_copula, etc.
        pop_context: Population context ('documentation', 'baseline', 'copula')

    Returns:
        True if stochastic component should be applied
    """
    sigma_value = stochastic_params.get('sigma_value', 0)

    return (
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


def apply_stochastic_draw(
    anchor_value: float,
    sigma: float,
    rng: np.random.Generator,
    floor_at_zero: bool = False,
    clip_range: Optional[Tuple[float, float]] = None
) -> float:
    """
    Apply stochastic Normal(anchor, sigma) draw.

    Args:
        anchor_value: Mean of the normal distribution
        sigma: Standard deviation
        rng: Random number generator
        floor_at_zero: If True, clip negative values to 0
        clip_range: Optional (min, max) tuple to clip the result

    Returns:
        Drawn value
    """
    if sigma <= 0:
        # No stochastic component - return anchor directly
        draw = anchor_value
    else:
        draw = rng.normal(anchor_value, sigma)

    if floor_at_zero:
        draw = max(draw, 0.0)

    if clip_range is not None:
        draw = np.clip(draw, clip_range[0], clip_range[1])

    return float(draw)


def get_stochastic_config_defaults() -> Dict[str, Any]:
    """
    Get default stochastic configuration structure.

    Returns a dict that can be used as a template for decision configs.
    """
    return {
        'sigma_strategy': 'overall',  # 'overall' or 'quintile'
        'sigma_value': 0,             # 0 = disabled, >0 = enabled
        'sigma_overall': DEFAULT_SIGMA_OVERALL,
        'sigma_quintile': DEFAULT_SIGMA_QUINTILE.copy(),
        'scale_factor': 1.0,
        'quintile_scale_factors': {
            '1': 1.0,
            '2': 1.0,
            '3': 1.0,
            '4': 1.0,
            '5': 1.0,
        },
        'in_copula': False,
    }


# Convenience function combining the above
def compute_stochastic_value(
    anchor_value: float,
    level: int,
    stochastic_params: Dict[str, Any],
    pop_context: str,
    rng: np.random.Generator,
    convert_to_z_scale: bool = False,
    z_scale_divisor: float = DEFAULT_SIGMA_OVERALL,
    floor_at_zero: bool = False,
    clip_range: Optional[Tuple[float, float]] = None
) -> Tuple[float, bool]:
    """
    All-in-one function to compute stochastic value.

    Combines should_use_stochastic, get_stochastic_sigma, and apply_stochastic_draw.

    Args:
        anchor_value: The deterministic anchor/mean value
        level: Income level (1-5)
        stochastic_params: Stochastic configuration dict
        pop_context: Population context ('documentation', 'baseline', 'copula')
        rng: Random number generator
        convert_to_z_scale: Convert sigma to z-score scale
        z_scale_divisor: Divisor for z-scale conversion
        floor_at_zero: Floor negative values at 0
        clip_range: Optional (min, max) clip range

    Returns:
        Tuple of (stochastic_value, was_stochastic_applied)
    """
    if not should_use_stochastic(stochastic_params, pop_context):
        # Return anchor unchanged
        result = anchor_value
        if floor_at_zero:
            result = max(result, 0.0)
        if clip_range is not None:
            result = np.clip(result, clip_range[0], clip_range[1])
        return float(result), False

    sigma = get_stochastic_sigma(
        level=level,
        stochastic_params=stochastic_params,
        convert_to_z_scale=convert_to_z_scale,
        z_scale_divisor=z_scale_divisor
    )

    draw = apply_stochastic_draw(
        anchor_value=anchor_value,
        sigma=sigma,
        rng=rng,
        floor_at_zero=floor_at_zero,
        clip_range=clip_range
    )

    return draw, True
