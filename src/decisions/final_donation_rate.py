# src/decisions/final_donation_rate.py

FALLBACK_RATE = 0.10


def final_donation_rate(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 13: Select donation rate after transaction accepted.

    Uses the agent's computed donation_default when Decision 3 ran (present in
    agent_state). Otherwise returns
    simulation_config['default_decisions']['final_donation_rate']['value']
    (the value collected from the UI's Overview tab), falling back to 0.10 when
    simulation_config or any of the keys is missing. rng unused.
    """
    if 'donation_default' in agent_state:
        return {"final_donation_rate": agent_state['donation_default']}

    entry = ((simulation_config or {}).get('default_decisions') or {}).get('final_donation_rate')
    value = entry.get('value') if isinstance(entry, dict) else None
    if value is None:
        value = FALLBACK_RATE
    return {"final_donation_rate": value}
