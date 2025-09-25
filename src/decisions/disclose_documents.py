# src/decisions/disclose_documents.py

def disclose_documents(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 2: Disclose documents for Discount status"""
    
    # Check if probability settings are available from simulation config
    if simulation_config and 'random_decisions' in simulation_config:
        prob_config = simulation_config['random_decisions'].get('disclose_documents')
        if prob_config and prob_config.get("type") == "random_probability":
            # Use weighted random choice with proper RNG
            probability_y = prob_config.get("probability_y", 0.5)
            options = prob_config.get("options", ["Y", "N"])
            
            # Use provided RNG for reproducibility
            if rng.random() < probability_y:
                choice = options[0]  # Y
            else:
                choice = options[1]  # N
            return {"disclose_documents": choice}
    
    # Fallback to legacy behavior
    try:
        from app.pages.decision_execution import get_actual_default_value
        default_value = get_actual_default_value("disclose_documents")
    except ImportError:
        # Fallback to old behavior if import fails
        default_value = params.get("default_value", "NA")
    
    return {"disclose_documents": default_value}