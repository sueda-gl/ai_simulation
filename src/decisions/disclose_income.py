# src/decisions/disclose_income.py

def disclose_income(agent_state: dict, params: dict, rng) -> dict:
    """Decision 1: Disclose income for Fixed status at time of registration/review"""
    return {"disclose_income": params.get("default_value", "NA")}