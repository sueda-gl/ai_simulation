# src/decisions/purchase_vs_bid.py

def purchase_vs_bid(agent_state: dict, params: dict, rng) -> dict:
    """Decision 9: For Regular customers - decide between Purchase Now and Bid"""
    return {"purchase_vs_bid": params.get("default_value", "NA")}