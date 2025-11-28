# src/decisions/vendor_selection.py
"""
Decision 8: Vendor Selection

NOTE: This decision currently only showcases agent preferences (which vendor they want).
The vendorID in purchase requests is calculated in Decision 6 (purchasing_quantity) based on 
weighted composite scores.

This decision will be used by the algorithm to determine actual vendor assignments 
(considering capacity constraints, availability, etc.), but currently it only returns 
the agent's preferred vendor for visualization and analysis purposes.

Composite Score Formula (calculated in Decision 6):
   score = w_price × norm_price + w_quality × norm_quality + 
           w_proximity × norm_proximity + w_sustainability × norm_sustainability

Where each attribute is standardized to [0, 1] using FIXED reference ranges:
- Price: Inverted using configured bounds (vendor_price_min, vendor_price_max)
  Formula: 1 - (clamped_price - price_min) / (price_max - price_min)
  NOTE: Uses configured bounds (not actual min/max) for equal discriminatory power
- Quality: (value - 1) / 4 for range [1,5]
- Sustainability: (value - 1) / 4 for range [1,5]  
- Proximity: value / 100 for range [0,100]
"""

import numpy as np
from src.vendor_attribute_generator import generate_proximity_scores, select_best_vendor


def vendor_selection(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """
    Decision 8: Vendor Selection - Currently showcases agent preferences only.
    
    NOTE: This decision returns the agent's preferred vendor (calculated in Decision 6).
    In the future, this will be used by the algorithm to assign actual vendors based on:
    - Capacity constraints
    - Availability
    - Dynamic pricing
    - Other market conditions
    
    Currently, it simply reads the vendorID from purchase requests (set in Decision 6)
    and returns summary information about the agent's vendor preference.
    
    Args:
        agent_state: Agent's state dict containing:
            - preferred_vendor: Vendor with highest score (set in Decision 6)
            - purchase_requests: List of purchase requests with vendorID
        params: Decision-specific parameters (not used for defaults)
        rng: Random number generator for this agent
        simulation_config: Global configuration
        
    Returns:
        dict: {
            "vendor_selection": int (preferred vendor ID),
            "preferred_vendor": int (same as vendor_selection for now)
        }
    """
    
    # Simply return the preferred vendor that was already calculated in Decision 6
    return _showcase_vendor_preference(agent_state, simulation_config)


def _showcase_vendor_preference(agent_state: dict, simulation_config: dict) -> dict:
    """
    Showcase the agent's vendor preference (calculated in Decision 6).
    
    This function simply reads and returns the preferred vendor information
    that was already determined when creating purchase requests.
    
    NOTE: This is for visualization/analysis purposes only.
    The actual vendor assignment algorithm will be implemented here in the future.
    """
    import numpy as np
    
    # Get the preferred vendor that was already calculated in Decision 6
    preferred_vendor = agent_state.get('preferred_vendor', None)
    
    # Get purchase requests to verify consistency
    purchase_requests = agent_state.get('purchase_requests', [])
    
    if not isinstance(purchase_requests, list) or len(purchase_requests) == 0:
        # No purchase requests
            return {
                "vendor_selection": np.nan,
            "preferred_vendor": np.nan,
            "note": "Agent preference calculated in Decision 6 (purchasing_quantity)"
        }
    
    # If preferred_vendor wasn't stored, extract from first purchase request
    if preferred_vendor is None:
        first_request = purchase_requests[0] if purchase_requests else {}
        preferred_vendor = first_request.get('vendorID', 1)
    
    # Return the preference information
    # NOTE: In the future, this will be replaced by actual vendor assignment logic
    return {
        "vendor_selection": preferred_vendor,
        "preferred_vendor": preferred_vendor,
        "total_requests": len(purchase_requests),
        "note": "Currently shows agent preference only. Actual vendor assignment will be implemented by the algorithm."
    }
