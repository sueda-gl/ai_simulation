# src/decisions/vendor_choice_weights.py

def vendor_choice_weights(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 5: Select default weights for vendor/product choice"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("vendor_choice_weights")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"vendor_choice_weights": default_value}