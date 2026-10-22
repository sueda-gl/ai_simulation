# src/utils/stochastic.py
"""
The one stochastic gate shared by the decision functions that draw noise.

The sigma values themselves live in config/decisions.yaml (each decision's
``stochastic`` block) and are read strictly by the decision modules; this module
carries no sigma constant.

Usage:
    from src.utils.stochastic import should_use_stochastic

    if should_use_stochastic(stochastic_params, pop_context):
        draw = rng.normal(anchor, sigma)
"""

from typing import Dict, Any


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
