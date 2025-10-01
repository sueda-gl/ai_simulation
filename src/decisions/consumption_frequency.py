# src/decisions/consumption_frequency.py
import numpy as np

def consumption_frequency(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """
    Decision 7: Calculate consumption frequency (units per hour)
    
    Simple default: Consumption Quantity / Term Duration
    
    Formula: frequency = consumption_quantity / (periods × duration_hours)
    
    Args:
        agent_state: Current agent state (must include consumption_quantity)
        params: Decision parameters
        rng: Random number generator (not used in simple version)
        simulation_config: Global simulation configuration
        
    Returns:
        dict: {"consumption_frequency": float} - units per hour
    """
    
    # Get consumption quantity from previous decision
    consumption_quantity = agent_state.get('consumption_quantity', 0)
    
    # Get term duration from simulation config
    if simulation_config and 'simulation' in simulation_config:
        periods = simulation_config['simulation'].get('periods', 1)
        duration_hours = simulation_config['simulation'].get('duration_hours', 1.0)
        term_duration = periods * duration_hours
    else:
        # Fallback defaults
        term_duration = 1.0
    
    # Calculate frequency: consumption per hour over the term
    if term_duration > 0 and consumption_quantity > 0:
        frequency = consumption_quantity / term_duration
    else:
        frequency = 0.0
    
    return {"consumption_frequency": float(frequency)}