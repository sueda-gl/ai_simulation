# src/decisions/rejected_transaction_defaults.py

def rejected_transaction_defaults(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 4: Select prioritized defaults for handling rejected transactions"""
    
    # Check if configuration is available from simulation_config
    if simulation_config and 'default_decisions' in simulation_config:
        config = simulation_config['default_decisions'].get('rejected_transaction_defaults')
        if config and config.get("type") == "radio_selection":
            # Use the selected option from the configuration
            selected_option = config.get("selected_option", "forgo_transaction")
            return {"rejected_transaction_defaults": selected_option}
    
    # Fallback to default
    return {"rejected_transaction_defaults": "forgo_transaction"}