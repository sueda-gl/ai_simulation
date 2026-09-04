# src/decisions/rejected_transaction_option.py

FALLBACK_OPTION = "forgo_transaction"


def rejected_transaction_option(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 11: Select option after rejected transaction.

    Returns simulation_config['default_decisions']['rejected_transaction_option']['selected_option']
    (the value collected from the UI's Overview tab), falling back to 'forgo_transaction'
    when simulation_config or any of the keys is missing. Reads nothing from agent_state; rng unused.
    """
    entry = ((simulation_config or {}).get('default_decisions') or {}).get('rejected_transaction_option')
    selected = entry.get('selected_option') if isinstance(entry, dict) else None
    if selected is None:
        selected = FALLBACK_OPTION
    return {"rejected_transaction_option": selected}
