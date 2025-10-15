# src/decisions/enrich_purchase_requests.py
"""
Decision 6b: Enrich Purchase Requests

This decision runs AFTER consumption_quantity (Decision 6) and enriches each purchase request
with transaction-level decisions:
- platformPrice: "DISCOUNT", "FIXED", "PN", or "BID" based on per-request purchase_vs_bid decision
- bid_value: Actual bid amount for BID requests, or "N/A" otherwise

This implements the professor's requirement that purchase decisions are made PER REQUEST,
not per agent. So Agent 1 with 7 purchase requests could choose:
- PN for 4 requests
- BID for 3 requests
Each with potentially different bid values.
"""

import numpy as np
from src.decisions.purchase_vs_bid import purchase_vs_bid_single
from src.decisions.bid_value import generate_single_bid_value


def enrich_purchase_requests(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """
    Decision 6b: Enrich purchase requests with per-request transaction decisions.
    
    Loops through each purchase request and:
    1. Makes a purchase_vs_bid decision for REGULAR customers
    2. Generates bid_value if the decision was "bid"
    3. Sets platformPrice based on customer type and decision
    4. Updates the purchase_requests list in agent_state
    
    Args:
        agent_state: Agent's state dict containing:
            - purchase_requests: List of purchase request dicts (from Decision 6)
            - customer_type: "discount", "fixed", or "regular"
        params: Decision parameters (not used currently)
        rng: Random number generator for this agent
        simulation_config: Global simulation configuration
        
    Returns:
        dict: {"purchase_requests": enriched list, "enriched_requests_count": int}
    """
    
    # Get purchase requests from agent_state
    purchase_requests = agent_state.get('purchase_requests', [])
    
    if not isinstance(purchase_requests, list) or len(purchase_requests) == 0:
        # No purchase requests to enrich
        return {
            "purchase_requests": [],
            "enriched_requests_count": 0
        }
    
    # Get customer type
    customer_type = agent_state.get('customer_type', 'regular')
    
    # Enrich each purchase request
    enriched_count = 0
    
    for request in purchase_requests:
        if not isinstance(request, dict):
            continue
        
        # STEP 1: Determine purchase decision for this request
        # For DISCOUNT and FIXED customers, decision is predetermined
        # For REGULAR customers, make a new random decision for each request
        
        if customer_type == 'discount':
            purchase_decision = "NA_discount"
            platform_price = "DISCOUNT"
            bid_value = "N/A"
        
        elif customer_type == 'fixed':
            purchase_decision = "NA_fixed"
            platform_price = "FIXED"
            bid_value = "N/A"
        
        elif customer_type == 'regular':
            # Make a new purchase_vs_bid decision for THIS specific request
            # This allows different requests to have different decisions
            purchase_decision = purchase_vs_bid_single(
                customer_type=customer_type,
                params=params,
                rng=rng,
                simulation_config=simulation_config
            )
            
            # STEP 2: Set platform price based on decision
            if purchase_decision == "Purchase Now":
                platform_price = "PN"
                bid_value = "N/A"
            
            elif purchase_decision == "bid":
                platform_price = "BID"
                
                # STEP 3: Generate bid value for this specific bid request
                # Each bid gets its own unique random value
                bid_value = generate_single_bid_value(
                    rng=rng,
                    simulation_config=simulation_config,
                    params=params
                )
            
            else:
                # Fallback (should not happen)
                platform_price = "PN"
                bid_value = "N/A"
        
        else:
            # Unknown customer type (should not happen)
            purchase_decision = "Purchase Now"
            platform_price = "PN"
            bid_value = "N/A"
        
        # STEP 4: Update the request dict with enriched fields
        request['platformPrice'] = platform_price
        request['bid_value'] = bid_value
        request['purchase_vs_bid'] = purchase_decision  # Store for reference
        
        enriched_count += 1
    
    # Return enriched purchase requests
    return {
        "purchase_requests": purchase_requests,
        "enriched_requests_count": enriched_count
    }

