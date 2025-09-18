# src/decisions/consumption_frequency.py

def consumption_frequency(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 7: Decide consumption frequency per day"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("consumption_frequency")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"consumption_frequency": default_value}