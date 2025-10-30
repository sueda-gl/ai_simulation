# src/decisions/vendor_selection.py
"""
Decision 8: Vendor Selection

Selects vendor for each purchase request based on weighted composite scores.

Default behavior (when not selected for custom configuration):
1. Get vendor_choice_weights from agent_state (set by Decision 5)
2. Get vendor pool from simulation_config (quality, sustainability, price)
3. Generate proximity scores for this customer-vendor dyad (fixed for this agent)
4. For each purchase request:
   - Calculate composite score for each vendor
   - Select vendor with highest score (deterministic)
   - Update vendorID in the purchase request

Composite Score Formula:
   score = w_price × norm_price + w_quality × norm_quality + 
           w_proximity × norm_proximity + w_sustainability × norm_sustainability

Where each attribute is standardized to [0, 1]:
- Price: Inverted (lower price = higher score)
- Quality: (value - 1) / 4 for range [1,5]
- Sustainability: (value - 1) / 4 for range [1,5]  
- Proximity: value / 100 for range [0,100]
"""

import numpy as np
from src.vendor_attribute_generator import generate_proximity_scores, select_best_vendor


def vendor_selection(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """
    Decision 8: Select vendor for each purchase request based on weighted scores.
    
    This is a REQUEST-LEVEL decision - updates vendorID for each purchase request.
    
    Args:
        agent_state: Agent's state dict containing:
            - vendor_choice_weights: Weights dict from Decision 5
            - purchase_requests: List of purchase request dicts from Decision 6
        params: Decision-specific parameters (not used for defaults)
        rng: Random number generator for this agent (used for proximity generation)
        simulation_config: Global configuration containing vendors list
        
    Returns:
        dict: {
            "vendor_selection": int (vendor_id of first/most common selection),
            "purchase_requests": updated list with vendorID for each request
        }
    """
    
    # Check if this decision is using defaults (when NOT selected for custom config)
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'vendor_selection' in simulation_config.get('default_decisions_list', []):
            # ========== DEFAULT MODE ==========
            return _vendor_selection_default(agent_state, rng, simulation_config)
    
    # ========== CUSTOM MODE (future implementation) ==========
    # When vendor_selection is selected for custom configuration,
    # this would read custom parameters from params and implement
    # a sophisticated vendor selection algorithm
    
    # For now, fall back to default behavior
    return _vendor_selection_default(agent_state, rng, simulation_config)


def _vendor_selection_default(agent_state: dict, rng, simulation_config: dict) -> dict:
    """
    Default vendor selection implementation with supply constraints.
    
    Enforces vendor capacity limits in snapshot mode:
    - Tracks remaining capacity for each vendor across all agents
    - Agents processed in order get their preferred vendor if capacity available
    - If preferred vendor sold out, agent gets next-best available vendor
    """
    import numpy as np
    from src.vendor_attribute_generator import calculate_vendor_composite_score
    
    # STEP 1: Get purchase requests from agent_state
    purchase_requests = agent_state.get('purchase_requests', [])
    
    if not isinstance(purchase_requests, list) or len(purchase_requests) == 0:
        # No purchase requests - return NaN instead of "NA" for numeric compatibility
        return {"vendor_selection": np.nan}
    
    # Count how many units this agent wants to purchase
    agent_demand = len(purchase_requests)
    
    # STEP 2: Get vendors from simulation_config
    vendors = simulation_config.get('vendors', [])
    
    if not vendors or len(vendors) == 0:
        # No vendors configured - keep default vendorID=1
        return {"vendor_selection": 1}
    
    # STEP 3: Initialize vendor capacity tracking (first agent only)
    if 'vendor_remaining_capacity' not in simulation_config:
        # Initialize capacity for each vendor
        simulation_config['vendor_remaining_capacity'] = {}
        for vendor in vendors:
            vendor_id = vendor['vendor_id']
            capacity = vendor.get('quantity_offered', 100)  # Default to 100 if not specified
            simulation_config['vendor_remaining_capacity'][vendor_id] = capacity
    
    remaining_capacity = simulation_config['vendor_remaining_capacity']
    
    # STEP 4: Get vendor choice weights from agent_state (set by Decision 5)
    weights = agent_state.get('vendor_choice_weights', {
        'price': 0.25,
        'quality': 0.25,
        'proximity': 0.25,
        'sustainability': 0.25
    })
    
    # STEP 5: Generate proximity scores for this agent (customer-vendor dyad)
    if 'vendor_proximity_scores' not in agent_state:
        agent_id = agent_state.get('agent_id', agent_state.get('index', 0) + 1)
        proximity_scores = generate_proximity_scores(agent_id, len(vendors), rng)
        agent_state['vendor_proximity_scores'] = proximity_scores
    else:
        proximity_scores = agent_state['vendor_proximity_scores']
    
    # STEP 6: Calculate composite score for ALL vendors
    # Create list of (vendor_id, score) tuples
    vendor_scores = []
    for vendor in vendors:
        vendor_id = vendor['vendor_id']
        proximity = proximity_scores.get(str(vendor_id), 50.0)
        score = calculate_vendor_composite_score(vendor, weights, proximity, vendors)
        vendor_scores.append((vendor_id, score))
    
    # Sort by score descending (best vendor first)
    vendor_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Store agent's preferred vendor (before capacity constraints)
    preferred_vendor_id = vendor_scores[0][0] if vendor_scores else None
    
    # STEP 7: Select best vendor that has capacity
    selected_vendor_id = None
    vendor_rank = None
    
    for rank, (vendor_id, score) in enumerate(vendor_scores, 1):
        # Check if this vendor has enough capacity for this agent's demand
        if remaining_capacity.get(vendor_id, 0) >= agent_demand:
            selected_vendor_id = vendor_id
            vendor_rank = rank
            # Reserve capacity for this agent
            remaining_capacity[vendor_id] -= agent_demand
            break
    
    # STEP 8: Fallback if no vendor has sufficient capacity
    if selected_vendor_id is None:
        # Find vendor with most remaining capacity
        max_capacity = 0
        fallback_vendor_id = None
        for vendor_id, capacity in remaining_capacity.items():
            if capacity > max_capacity:
                max_capacity = capacity
                fallback_vendor_id = vendor_id
        
        # Take what we can get IF there's any capacity left
        if fallback_vendor_id is not None and max_capacity > 0:
            selected_vendor_id = fallback_vendor_id
            remaining_capacity[fallback_vendor_id] -= min(agent_demand, max_capacity)
            vendor_rank = next((i+1 for i, (vid, _) in enumerate(vendor_scores) if vid == fallback_vendor_id), len(vendor_scores))
        else:
            # ALL VENDORS SOLD OUT - agent cannot be allocated
            # Return NaN to indicate no vendor available
            return {
                "vendor_selection": np.nan,
                "purchase_requests": purchase_requests,
                "vendor_rank": np.nan,
                "preferred_vendor": preferred_vendor_id,
                "got_preferred": False,
                "allocation_failed": True  # Flag to indicate allocation failure
            }
    
    # STEP 9: Update vendorID for ALL purchase requests (only if vendor was found)
    if selected_vendor_id is not None:
        for request in purchase_requests:
            if isinstance(request, dict):
                request['vendorID'] = selected_vendor_id
    
    # Update agent_state with modified purchase_requests
    agent_state['purchase_requests'] = purchase_requests
    
    # STEP 10: Return results including tracking info
    return {
        "vendor_selection": selected_vendor_id,
        "purchase_requests": purchase_requests,
        "vendor_rank": vendor_rank,  # Which preference rank (1st choice, 2nd choice, etc.)
        "preferred_vendor": preferred_vendor_id,  # What they wanted before capacity constraints
        "got_preferred": (selected_vendor_id == preferred_vendor_id),  # Did they get their first choice?
        "allocation_failed": False  # Successful allocation
    }
