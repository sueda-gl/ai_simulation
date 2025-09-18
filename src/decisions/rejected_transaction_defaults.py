# src/decisions/rejected_transaction_defaults.py

def rejected_transaction_defaults(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 4: Select prioritized defaults for handling rejected transactions"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("rejected_transaction_defaults")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"rejected_transaction_defaults": default_value}