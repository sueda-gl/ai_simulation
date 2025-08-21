# src/decisions/consumption_quantity.py

def consumption_quantity(agent_state: dict, params: dict, rng) -> dict:
    """Decision 6: Decide how much of product type X to consume per calendar period"""
    return {"consumption_quantity": params.get("default_value", "NA")}