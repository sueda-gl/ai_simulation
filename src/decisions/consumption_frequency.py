# src/decisions/consumption_frequency.py

def consumption_frequency(agent_state: dict, params: dict, rng) -> dict:
    """Decision 7: Decide consumption frequency per day"""
    return {"consumption_frequency": params.get("default_value", "NA")}