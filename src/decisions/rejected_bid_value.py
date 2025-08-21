# src/decisions/rejected_bid_value.py

def rejected_bid_value(agent_state: dict, params: dict, rng) -> dict:
    """Decision 12: Select bid value after rejected transaction"""
    return {"rejected_bid_value": params.get("default_value", "NA")}