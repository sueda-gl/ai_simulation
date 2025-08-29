# src/decisions/rejected_transaction_defaults.py

def rejected_transaction_defaults(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 4: Select prioritized defaults for handling rejected transactions"""
    return {"rejected_transaction_defaults": params.get("default_value", "NA")}