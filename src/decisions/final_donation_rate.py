# src/decisions/final_donation_rate.py

def final_donation_rate(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 13: Select donation rate after transaction accepted
    
    This decision uses the donation_default value when available (matching UI behavior).
    The UI shows donation_default distribution when:
    - A donation configuration is selected OR
    - Only one donation configuration was generated
    - AND donation_default column exists
    
    Since we're in execution, we apply the same logic: if donation_default was computed
    (exists in agent_state), use it. Otherwise, use the configured default.
    """
    
    # Check if donation_default was computed for this agent
    if 'donation_default' in agent_state:
        # Use the computed donation_default value (matches UI override behavior)
        return {"final_donation_rate": agent_state['donation_default']}
    
    # No donation_default available - use configured default value
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("final_donation_rate")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"final_donation_rate": default_value}