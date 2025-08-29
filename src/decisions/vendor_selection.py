# src/decisions/vendor_selection.py

def vendor_selection(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 8: Select vendor/product from sorted list"""
    return {"vendor_selection": params.get("default_value", "NA")}