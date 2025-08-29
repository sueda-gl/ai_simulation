# src/decisions/rejected_transaction_option.py

def rejected_transaction_option(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 11: Select option after rejected transaction"""
    return {"rejected_transaction_option": params.get("default_value", "NA")}