# src/decisions/rejected_bid_value.py

FALLBACK_VALUE = "NA"


def rejected_bid_value(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 12: Select bid value after rejected transaction.

    Returns simulation_config['default_decisions']['rejected_bid_value']['value']
    (the value collected from the UI's Overview tab), falling back to 'NA'
    when simulation_config or any of the keys is missing. Reads nothing from agent_state; rng unused.
    """
    entry = ((simulation_config or {}).get('default_decisions') or {}).get('rejected_bid_value')
    value = entry.get('value') if isinstance(entry, dict) else None
    if value is None:
        value = FALLBACK_VALUE
    return {"rejected_bid_value": value}
