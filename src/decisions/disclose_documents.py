# src/decisions/disclose_documents.py

def disclose_documents(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 2: Disclose documents for Discount status"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("disclose_documents")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"disclose_documents": default_value}