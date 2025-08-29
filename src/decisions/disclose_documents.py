# src/decisions/disclose_documents.py

def disclose_documents(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 2: Disclose documents for Discount status"""
    return {"disclose_documents": params.get("default_value", "NA")}