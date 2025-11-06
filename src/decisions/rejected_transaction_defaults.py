# src/decisions/rejected_transaction_defaults.py
import numpy as np

def rejected_transaction_defaults(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """
    Decision 4: Select prioritized defaults for handling rejected transactions
    
    Each agent gets a prioritized list of default options (can be 1-5 options).
    If Option 5 (forgo_transaction) is included, it must be last in the priority list.
    
    Returns:
        dict: {"rejected_transaction_defaults": list of option strings in priority order}
              Example: ["current_vendor_pn", "higher_price_category", "forgo_transaction"]
    """
    
    # All available options
    all_options = [
        "higher_price_category",   # Option 1
        "lower_pn_vendor",         # Option 2
        "current_vendor_pn",       # Option 3
        "place_bid",               # Option 4
        "forgo_transaction"        # Option 5
    ]
    
    # Check if configuration is available from simulation_config
    if simulation_config and 'default_decisions' in simulation_config:
        config = simulation_config['default_decisions'].get('rejected_transaction_defaults')
        if config and config.get("type") == "prioritized_selection":
            # Get configured priority template
            priority_template = config.get("priority_template", ["forgo_transaction"])
            
            # Each agent can have different priority list (for now, we use the template)
            # In future versions, this could vary by agent characteristics
            return {"rejected_transaction_defaults": priority_template}
    
    # Fallback to default behavior: generate random prioritized list for each agent
    # Randomly choose how many options (1-5)
    num_options = rng.integers(1, 6)
    
    # Randomly select options without replacement
    selected_options = list(rng.choice(all_options, size=num_options, replace=False))
    
    # If "forgo_transaction" is in the list but not last, move it to the end
    if "forgo_transaction" in selected_options:
        selected_options.remove("forgo_transaction")
        selected_options.append("forgo_transaction")
    
    return {"rejected_transaction_defaults": selected_options}