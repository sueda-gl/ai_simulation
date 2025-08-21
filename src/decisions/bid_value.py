# src/decisions/bid_value.py

def bid_value(agent_state: dict, params: dict, rng) -> dict:
    """Decision 10: For Regular customers - select bid value if Bid chosen"""
    return {"bid_value": params.get("default_value", "NA")}