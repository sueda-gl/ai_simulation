# src/decisions/purchasing_frequency.py
"""
Decision 7: purchasing_frequency

Calculates purchasing frequency (items per period) from purchasing quantity and number of periods.

This is a COMPUTED decision - not random. It derives its value from:
1. purchasing_quantity (from Decision 6)
2. Number of periods (from Page 1)

Formula: purchasing_frequency = purchasing_quantity / periods

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
    Decision 7: Calculate purchasing frequency from quantity and number of periods.
    
    This is a deterministic, computed decision (not stochastic).
    
    Formula:
        frequency = purchasing_quantity / periods
        
    Args:
        agent_state: Agent's state dict (must include 'purchasing_quantity' from Decision 6)
        params: Decision-specific parameters from decisions.yaml (not used for defaults)
        rng: Random number generator (not used - this is deterministic)
        simulation_config: Page 1 parameters in ['simulation'] sub-dict
        
    Returns:
        Dictionary with:
        - purchasing_frequency: Items per period (float)
    
    Example:
        Agent has purchasing_quantity = 10 items
        Periods = 5
        Frequency = 10 / 5 = 2.0 items/period
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
    # Frequency = quantity / periods (items per period)
    # The user requested "number of purchase requests divided by period"
    # This represents the average number of requests per period, regardless of period duration.
    if periods > 0:
        frequency = purchasing_quantity / periods
    else:
        # Safety: if periods is somehow 0, frequency is 0
        frequency = 0.0
    
    # Return result
    return {
        "purchasing_frequency": float(frequency)
    }


