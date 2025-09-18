# src/decisions/consumption_quantity.py

def consumption_quantity(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 6: Decide how much of product type X to consume per calendar period"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("consumption_quantity")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"consumption_quantity": default_value}