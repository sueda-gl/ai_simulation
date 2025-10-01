# src/decisions/vendor_choice_weights.py

def vendor_choice_weights(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 5: Select default weights for vendor/product choice"""
    
    # Check if this decision is using configured defaults (when unselected)
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'vendor_choice_weights' in simulation_config.get('default_decisions_list', []):
            # This decision is unselected - use configured default values from Overview tab
            if 'default_decisions' in simulation_config:
                vendor_config = simulation_config['default_decisions'].get('vendor_choice_weights')
                if vendor_config and vendor_config.get("type") == "checkbox_selection":
                    # Use the pre-calculated weights from the configuration
                    weights = vendor_config.get("weights", {})
                    return {"vendor_choice_weights": weights}
    
    # This decision is selected OR no configuration available - use YAML params or fallback
    # For now, since there's no custom tab for vendor_choice_weights, always use equal weights
    # In the future, this could read from params if a custom tab is added
    default_weights = {
        "price": 0.25,
        "quality": 0.25,
        "proximity": 0.25,
        "sustainability": 0.25
    }
    
    return {"vendor_choice_weights": default_weights}