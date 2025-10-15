# src/decisions/bid_value.py
"""
Decision 10: Bid Value

This decision ONLY applies to REGULAR customers who chose to BID.
- DISCOUNT customers: NA (use discount pricing)
- FIXED customers: NA (use fixed pricing only)
- REGULAR customers who chose Purchase Now: NA (no bid needed)
- REGULAR customers who chose Bid: Generate bid value

NOTE: This decision can be called either:
1. Agent-level: Once per agent (legacy/backward compatibility)
2. Request-level: Multiple times per agent, once per bid request (generates unique bid each time)
"""

import numpy as np
from src.decisions.income_utils import get_simulation_param


def generate_single_bid_value(rng, simulation_config: dict = None, params: dict = None) -> float:
    """
    Generate a single random bid value within the bidding range.
    
    This is the core bid generation logic that can be called multiple times per agent.
    Each call generates a NEW random bid value.
    
    Formula:
        Baseline Price (Pc) = (1 + platform_markup) × vendor_price
        Min Bid (Pmb) = (1 - price_range) × Pc
        Max Bid (Ppn) = (1 + price_range) × Pc
        Bid Value = Uniform random in [Pmb, Ppn]
    
    Args:
        rng: Random number generator
        simulation_config: Global simulation configuration with pricing parameters
        params: Decision-specific parameters (unused currently)
        
    Returns:
        float: Random bid amount (rounded to 2 decimal places)
    """
    # Get pricing parameters from Page 1 using centralized helper
    vendor_price = get_simulation_param(simulation_config, 'market_price', 100.0)
    platform_markup = get_simulation_param(simulation_config, 'platform_markup', 0.1)
    price_range = get_simulation_param(simulation_config, 'price_range', 0.25)
    
    # Calculate bidding range using the formula
    baseline_price = (1 + platform_markup) * vendor_price  # Pc = (1+m) × vendor_price
    min_bid_price = (1 - price_range) * baseline_price     # Pmb = (1-r) × Pc
    max_bid_price = (1 + price_range) * baseline_price     # Ppn = (1+r) × Pc
    
    # Generate random bid value within the range using provided RNG
    # Use uniform distribution over [min_bid_price, max_bid_price]
    bid_amount = rng.uniform(min_bid_price, max_bid_price)
    
    # Round to 2 decimal places (standard for currency)
    bid_amount = round(bid_amount, 2)
    
    return bid_amount


def bid_value(agent_state: dict, params: dict, rng, simulation_config: dict = None,
             request_context: dict = None) -> dict:
    """
    Decision 10: For Regular customers - select bid value if Bid chosen
    
    Can be called in two modes:
    1. Agent-level (request_context=None): Returns single bid for agent (backward compatibility)
    2. Request-level (request_context provided): Returns new bid for specific purchase request
    
    Args:
        agent_state: Current agent state including previous decisions
        params: Decision-specific parameters from decisions.yaml
        rng: Random number generator for this agent
        simulation_config: Global simulation configuration with pricing parameters
        request_context: Optional dict with request-specific info (for per-request calls)
        
    Returns:
        dict: {"bid_value": float or np.nan}
    """
    
    # STEP 1: Check customer type - only applies to REGULAR customers
    customer_type = agent_state.get('customer_type', 'regular')
    
    if customer_type != 'regular':
        # Discount or Fixed customers don't bid
        return {"bid_value": np.nan}
    
    # STEP 2: Check if this request is a bid
    # If request_context provided, use its purchase_vs_bid decision
    # Otherwise, use agent-level decision (backward compatibility)
    if request_context is not None:
        purchase_vs_bid_choice = request_context.get('purchase_vs_bid', 'Purchase Now')
    else:
        purchase_vs_bid_choice = agent_state.get('purchase_vs_bid', 'Purchase Now')
    
    # Handle the new NA values for non-regular customers
    if purchase_vs_bid_choice in ['NA_discount', 'NA_fixed']:
        return {"bid_value": np.nan}
    
    if purchase_vs_bid_choice != 'bid':
        # Not a bid request, so no bid value needed
        return {"bid_value": np.nan}
    
    # STEP 3: Generate bid value using core logic
    # Each call generates a NEW random bid value (important for per-request calls)
    bid_amount = generate_single_bid_value(rng, simulation_config, params)
    
    return {"bid_value": bid_amount}