# src/decisions/disclose_income.py

def disclose_income(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 1: Disclose income for Fixed status at time of registration/review"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("disclose_income")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"disclose_income": default_value}