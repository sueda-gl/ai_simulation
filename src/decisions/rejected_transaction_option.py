# src/decisions/rejected_transaction_option.py

def rejected_transaction_option(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 11: Select option after rejected transaction"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("rejected_transaction_option")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"rejected_transaction_option": default_value}