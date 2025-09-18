# src/decisions/bid_value.py

def bid_value(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 10: For Regular customers - select bid value if Bid chosen"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("bid_value")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"bid_value": default_value}