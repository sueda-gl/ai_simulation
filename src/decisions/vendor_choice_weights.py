# src/decisions/vendor_choice_weights.py

def vendor_choice_weights(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 5: Select default weights for vendor/product choice"""
    return {"vendor_choice_weights": params.get("default_value", "NA")}