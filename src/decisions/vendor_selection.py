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
    Default vendor selection implementation.
    
    Deterministic selection based on highest weighted composite score.
    """
    import numpy as np
    
    # STEP 1: Get purchase requests from agent_state
    purchase_requests = agent_state.get('purchase_requests', [])
    
    if not isinstance(purchase_requests, list) or len(purchase_requests) == 0:
        # No purchase requests - return NaN instead of "NA" for numeric compatibility
        return {"vendor_selection": np.nan}
    
    # STEP 2: Get vendors from simulation_config
    vendors = simulation_config.get('vendors', [])
    
    if not vendors or len(vendors) == 0:
        # No vendors configured - keep default vendorID=1
        return {"vendor_selection": 1}
    
    # STEP 3: Get vendor choice weights from agent_state (set by Decision 5)
    weights = agent_state.get('vendor_choice_weights', {
        'price': 0.25,
        'quality': 0.25,
        'proximity': 0.25,
        'sustainability': 0.25
    })
    
    # STEP 4: Generate proximity scores for this agent (customer-vendor dyad)
    # These scores are FIXED for this agent across all their purchases
    # (Same agent always has same proximity to same vendor)
    
    if 'vendor_proximity_scores' not in agent_state:
        # Generate proximity for each vendor
        agent_id = agent_state.get('agent_id', agent_state.get('index', 0) + 1)
        proximity_scores = generate_proximity_scores(agent_id, len(vendors), rng)
        agent_state['vendor_proximity_scores'] = proximity_scores
    else:
        proximity_scores = agent_state['vendor_proximity_scores']
    
    # STEP 5: Select best vendor for this agent
    # Since weights, proximity, and vendor attributes are all fixed,
    # the same vendor will be selected for ALL requests from this agent (deterministic)
    best_vendor_id = select_best_vendor(vendors, weights, proximity_scores)
    
    # STEP 6: Update vendorID for ALL purchase requests
    for request in purchase_requests:
        if isinstance(request, dict):
            request['vendorID'] = best_vendor_id
    
    # Update agent_state with modified purchase_requests
    agent_state['purchase_requests'] = purchase_requests
    
    # STEP 7: Return results
    # Return the selected vendor ID as the decision outcome
    return {
        "vendor_selection": best_vendor_id,
        "purchase_requests": purchase_requests  # Updated list
    }
