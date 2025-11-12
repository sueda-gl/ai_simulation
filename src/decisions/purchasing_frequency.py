# src/decisions/purchasing_frequency.py
"""
Decision 7: purchasing_frequency

Calculates purchasing frequency (items per hour) from purchasing quantity and term duration.

This is a COMPUTED decision - not random. It derives its value from:
1. purchasing_quantity (from Decision 6)
2. Term duration (periods × duration_hours from Page 1)

Formula: purchasing_frequency = purchasing_quantity / term_duration

When fully simulated (future):
- More sophisticated models for temporal patterns
- Consideration of peak/off-peak hours
- Seasonal variations in purchasing patterns
"""
from typing import Optional, Dict, Any

from src.decisions.income_utils import get_simulation_param


def purchasing_frequency(agent_state: dict, params: dict, rng, 
                         simulation_config: dict = None, **kwargs) -> dict:
    """
    Decision 7: Calculate purchasing frequency from quantity and term duration.
    
    This is a deterministic, computed decision (not stochastic).
    
    Formula:
        frequency = purchasing_quantity / term_duration
        
    Where:
        term_duration = periods × duration_hours
    
    Args:
        agent_state: Agent's state dict (must include 'purchasing_quantity' from Decision 6)
        params: Decision-specific parameters from decisions.yaml (not used for defaults)
        rng: Random number generator (not used - this is deterministic)
        simulation_config: Page 1 parameters in ['simulation'] sub-dict
        
    Returns:
        Dictionary with:
        - purchasing_frequency: Items per hour (float)
    
    Example:
        Agent has purchasing_quantity = 10 items
        Term = 2 periods × 1 hour = 2 hours
        Frequency = 10 / 2 = 5.0 items/hour
    """
    
    # STEP 1: Get purchasing quantity from Decision 6
    # This should have been set by the purchasing_quantity decision
    purchasing_quantity = agent_state.get('purchasing_quantity', 0)
    
    # Handle edge case: if somehow quantity is missing, default to 0
    if purchasing_quantity is None:
        purchasing_quantity = 0
    
    # Ensure it's numeric
    try:
        purchasing_quantity = float(purchasing_quantity)
    except (ValueError, TypeError):
        purchasing_quantity = 0.0
    
    # STEP 2: Get term duration from Page 1 parameters
    periods = get_simulation_param(simulation_config, 'periods', 1)
    duration_hours = get_simulation_param(simulation_config, 'duration_hours', 1.0)
    
    # Calculate term duration
    term_duration = float(periods * duration_hours)
    
    # STEP 3: Calculate frequency
    # Frequency = quantity / term_duration (items per hour)
    if term_duration > 0:
        frequency = purchasing_quantity / term_duration
    else:
        # Safety: if term_duration is somehow 0, frequency is 0
        frequency = 0.0
    
    # Return result
    return {
        "purchasing_frequency": float(frequency)
    }


