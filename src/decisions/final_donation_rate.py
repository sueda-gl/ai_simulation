# src/decisions/final_donation_rate.py

def final_donation_rate(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 13: Select donation rate after transaction accepted"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("final_donation_rate")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"final_donation_rate": default_value}