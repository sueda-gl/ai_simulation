# src/decisions/vendor_selection.py

def vendor_selection(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 8: Select vendor/product from sorted list"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("vendor_selection")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"vendor_selection": default_value}