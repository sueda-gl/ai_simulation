# src/decisions/final_donation_rate.py

def final_donation_rate(agent_state: dict, params: dict, rng) -> dict:
    """Decision 13: Select donation rate after transaction accepted"""
    return {"final_donation_rate": params.get("default_value", "NA")}