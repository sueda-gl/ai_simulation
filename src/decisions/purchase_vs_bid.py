# src/decisions/purchase_vs_bid.py
"""
Decision 9: Purchase Now vs Bid

This decision ONLY applies to REGULAR customers.
- DISCOUNT customers: get discount pricing (not applicable)
- FIXED customers: use fixed pricing only (not applicable)
- REGULAR customers: choose between Purchase Now or Bid

NOTE: This decision can be called either:
1. Agent-level: Once per agent (legacy/backward compatibility)
2. Request-level: Multiple times per agent, once per purchase request
"""

from src.decisions.income_utils import get_customer_type


def purchase_vs_bid_single(customer_type: str, params: dict, rng, simulation_config: dict = None) -> str:
    """
    Make a single purchase vs bid decision for a REGULAR customer.
    
    This is the core decision logic that can be called multiple times per agent.
    
    Args:
        customer_type: "discount", "fixed", or "regular"
        params: Decision parameters
        rng: Random number generator
        simulation_config: Global simulation configuration
        
    Returns:
        str: "Purchase Now", "bid", "NA_discount", or "NA_fixed"
    """
    # Normalize customer type
    customer_type = str(customer_type).lower().strip()
    
    # Only REGULAR customers make this choice
    if customer_type == 'discount':
        return "NA_discount"  # Uses discount pricing
    
    if customer_type == 'fixed':
        return "NA_fixed"  # Uses fixed pricing only
    
    # For REGULAR customers, apply probability-based decision
    # Check if probability settings are available from simulation config
    if simulation_config and 'random_decisions' in simulation_config:
        prob_config = simulation_config['random_decisions'].get('purchase_vs_bid')
        if prob_config and prob_config.get("type") == "random_probability":
            # Use weighted random choice with proper RNG
            probability_y = prob_config.get("probability_y", 0.5)
            options = prob_config.get("options", ["Purchase Now", "bid"])
            
            # Use provided RNG for reproducibility
            if rng.random() < probability_y:
                return options[0]  # Purchase Now
            else:
                return options[1]  # bid
    
    # Fallback to simple 50/50 random choice
    return rng.choice(["Purchase Now", "bid"])


def purchase_vs_bid(agent_state: dict, params: dict, rng, simulation_config: dict = None, 
                   request_context: dict = None) -> dict:
    """
    Decision 9: For Regular customers - decide between Purchase Now and Bid
    
    Can be called in two modes:
    1. Agent-level (request_context=None): Returns agent-level decision for backward compatibility
    2. Request-level (request_context provided): Returns decision for a specific purchase request
    
    Args:
        agent_state: Agent's state dict
        params: Decision parameters
        rng: Random number generator
        simulation_config: Global simulation configuration
        request_context: Optional dict with request-specific info (for per-request calls)
        
    Returns:
        dict: {"purchase_vs_bid": str} - the decision result
    """
    
    # STEP 1: Get customer type
    customer_type = agent_state.get('customer_type')
    
    # If customer type not yet determined, determine it now
    if customer_type is None:
        customer_type = get_customer_type(agent_state, simulation_config)
    
    # STEP 2: Make the decision using core logic
    decision = purchase_vs_bid_single(customer_type, params, rng, simulation_config)
    
    # STEP 3: Return result
    # If request_context provided, we're being called per-request
    # Otherwise, this is an agent-level call (legacy)
    return {"purchase_vs_bid": decision}