# src/engine/config_access.py
"""
Read access to the run configuration: the Page-1 parameters live under the
``simulation`` key of ``simulation_config``.
"""

from typing import Optional, Dict


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
