# src/decisions/rejected_bid_value.py

def rejected_bid_value(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 12: Select bid value after rejected transaction"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("rejected_bid_value")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"rejected_bid_value": default_value}