# src/decisions/consumption_quantity.py
import numpy as np

def consumption_quantity(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """
    Decision 6: Decide total consumption quantity for the term
    
    Simple default: Random integer within consumption limit
    
    Args:
        agent_state: Current agent state
        params: Decision parameters
        rng: Random number generator
        simulation_config: Global simulation configuration
        
    Returns:
        dict: {"consumption_quantity": int}
    """
    
    # Get consumption limit for this agent's income category
    # The consumption limit is per term (not per period)
    consumption_limit = None
    
    if simulation_config and 'consumption_limits' in simulation_config:
        limits = simulation_config['consumption_limits']
        
        # Try to get income_category from agent_state
        # If not available, default to category 1 (lowest income)
        income_category = agent_state.get('income_category')
        
        if income_category is None:
            # Fallback: Use category 1 as default (lowest income/discount customers)
            income_category = 1
        
        category_key = f"cat_{income_category}"
        
        # Get limit for this category
        if category_key in limits:
            consumption_limit = limits[category_key]
    
    # If no limit specified or consumption limits disabled, use a default upper bound
    if consumption_limit is None:
        consumption_limit = 50  # default maximum consumption per term
    
    # Generate random consumption quantity (integer)
    # Random between 0 and consumption_limit (inclusive)
    if consumption_limit > 0:
        quantity = rng.integers(0, consumption_limit + 1)
    else:
        quantity = 0
    
    return {"consumption_quantity": int(quantity)}