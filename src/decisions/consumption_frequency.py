# src/decisions/consumption_frequency.py
"""
Decision 7: consumption_frequency

Calculates consumption frequency (items per hour) from consumption quantity and term duration.

This is a COMPUTED decision - not random. It derives its value from:
1. consumption_quantity (from Decision 6)
2. Term duration (periods × duration_hours from Page 1)

Formula: consumption_frequency = consumption_quantity / term_duration

When fully simulated (future):
- More sophisticated models for temporal patterns
- Consideration of peak/off-peak hours
- Seasonal variations in consumption patterns
"""
from typing import Optional, Dict, Any


def _get_simulation_param(simulation_config: Optional[Dict], key: str, default=None):
    """
    Helper to safely extract parameters from simulation_config.
    
    Args:
        simulation_config: Config dict with 'simulation' sub-dict containing Page 1 parameters
        key: Parameter name to retrieve
        default: Fallback value if not found
        
    Returns:
        Parameter value from simulation_config['simulation'][key] or default
    """
    if not simulation_config or not isinstance(simulation_config, dict):
        return default
    
    sim = simulation_config.get('simulation', {})
    return sim.get(key, default)


def consumption_frequency(agent_state: dict, params: dict, rng, 
                         simulation_config: dict = None, **kwargs) -> dict:
    """
    Decision 7: Calculate consumption frequency from quantity and term duration.
    
    This is a deterministic, computed decision (not stochastic).
    
    Formula:
        frequency = consumption_quantity / term_duration
        
    Where:
        term_duration = periods × duration_hours
    
    Args:
        agent_state: Agent's state dict (must include 'consumption_quantity' from Decision 6)
        params: Decision-specific parameters from decisions.yaml (not used for defaults)
        rng: Random number generator (not used - this is deterministic)
        simulation_config: Page 1 parameters in ['simulation'] sub-dict
        
    Returns:
        Dictionary with:
        - consumption_frequency: Items per hour (float)
    
    Example:
        Agent has consumption_quantity = 10 items
        Term = 2 periods × 1 hour = 2 hours
        Frequency = 10 / 2 = 5.0 items/hour
    """
    
    # STEP 1: Get consumption quantity from Decision 6
    # This should have been set by the consumption_quantity decision
    consumption_quantity = agent_state.get('consumption_quantity', 0)
    
    # Handle edge case: if somehow quantity is missing, default to 0
    if consumption_quantity is None:
        consumption_quantity = 0
    
    # Ensure it's numeric
    try:
        consumption_quantity = float(consumption_quantity)
    except (ValueError, TypeError):
        consumption_quantity = 0.0
    
    # STEP 2: Get term duration from Page 1 parameters
    periods = _get_simulation_param(simulation_config, 'periods', 1)
    duration_hours = _get_simulation_param(simulation_config, 'duration_hours', 1.0)
    
    # Calculate term duration
    term_duration = float(periods * duration_hours)
    
    # STEP 3: Calculate frequency
    # Frequency = quantity / term_duration (items per hour)
    if term_duration > 0:
        frequency = consumption_quantity / term_duration
    else:
        # Safety: if term_duration is somehow 0, frequency is 0
        frequency = 0.0
    
    # Return result
    return {
        "consumption_frequency": float(frequency)
    }

