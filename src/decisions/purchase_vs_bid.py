# src/decisions/purchase_vs_bid.py

def purchase_vs_bid(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 9: For Regular customers - decide between Purchase Now and Bid"""
    # Try to import our default values function
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("purchase_vs_bid")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"purchase_vs_bid": default_value}