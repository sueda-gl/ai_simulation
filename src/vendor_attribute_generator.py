# src/vendor_attribute_generator.py
"""
Vendor Attribute Generator Module

Generates vendor attributes for vendor selection decisions:
- Quality: Random value between 1 and 5 (per vendor, fixed across simulation)
- Sustainability: Random value between 1 and 5 (per vendor, fixed across simulation)  
- Proximity: Random value between 0 and 100 (per customer-vendor dyad, fixed for that dyad)
- Price: Already generated from Page 1 parameters

These attributes are used by vendor_selection decision to calculate composite scores.
"""

import numpy as np
from typing import List, Dict, Optional


def generate_vendor_attributes(num_vendors: int, vendor_prices: List[float], 
                               rng: np.random.Generator) -> List[Dict]:
    """
    Generate vendor attributes for vendor selection.
    
    According to professor's specification:
    - Price: Already determined from Page 1 parameters (passed in)
    - Quality: Random integer in [1, 5] per vendor
    - Sustainability: Random integer in [1, 5] per vendor
    - Proximity: Generated per customer-vendor dyad (not here)
    
    Args:
        num_vendors: Number of vendors in the simulation
        vendor_prices: List of vendor prices (from Page 1 configuration)
        rng: Random number generator for reproducibility
        
    Returns:
        List of vendor dictionaries with attributes
    """
    vendors = []
    
    for i in range(num_vendors):
        vendor_id = i + 1  # Vendors numbered 1, 2, 3, ...
        
        # Get price (from Page 1 parameters)
        price = vendor_prices[i] if i < len(vendor_prices) else 100.0
        
        # Generate quality: random integer in [1, 5]
        quality = int(rng.integers(1, 6))  # 6 is exclusive, so generates 1-5
        
        # Generate sustainability: random integer in [1, 5]  
        sustainability = int(rng.integers(1, 6))
        
        vendor = {
            'vendor_id': vendor_id,
            'price': float(price),
            'quality': quality,
            'sustainability': sustainability
            # Note: proximity is NOT included here - it's customer-vendor specific
        }
        
        vendors.append(vendor)
    
    return vendors


def generate_proximity_scores(agent_id: int, num_vendors: int, 
                              rng: np.random.Generator) -> Dict[str, float]:
    """
    Generate proximity scores for a specific customer-vendor dyad.
    
    According to professor's specification:
    - Proximity: Random value in [0, 100]
    - Fixed for each customer-vendor dyad (same agent always has same proximity to same vendor)
    - Higher value means closer proximity
    
    Args:
        agent_id: Customer/agent ID
        num_vendors: Number of vendors
        rng: Random number generator for this agent
        
    Returns:
        Dictionary mapping vendor_id (as STRING) to proximity score
        Note: Keys must be strings for Parquet serialization
    """
    proximity_scores = {}
    
    for vendor_id in range(1, num_vendors + 1):
        # Generate random proximity between 0 and 100
        proximity = float(rng.uniform(0, 100))
        # Use STRING key for Parquet compatibility
        proximity_scores[str(vendor_id)] = proximity
    
    return proximity_scores


def calculate_vendor_composite_score(vendor: Dict, weights: Dict, 
                                     proximity: float,
                                     all_vendors: List[Dict]) -> float:
    """
    Calculate weighted composite score for a vendor.
    
    Steps:
    1. Standardize each attribute to [0, 1] range:
       - Price: Normalized and INVERTED (lower price = higher score)
       - Quality: (quality - 1) / (5 - 1) = [0, 1]
       - Sustainability: (sustainability - 1) / (5 - 1) = [0, 1]
       - Proximity: proximity / 100 = [0, 1]
    
    2. Calculate weighted sum:
       score = w_price × norm_price + w_quality × norm_quality + 
               w_proximity × norm_proximity + w_sustainability × norm_sustainability
    
    Args:
        vendor: Vendor dict with price, quality, sustainability
        weights: Weight dict from vendor_choice_weights decision
        proximity: Proximity score for this customer-vendor dyad
        all_vendors: List of all vendors (needed for price normalization)
        
    Returns:
        Composite score (float)
    """
    
    # Extract vendor attributes
    vendor_price = vendor['price']
    vendor_quality = vendor['quality']
    vendor_sustainability = vendor['sustainability']
    
    # STEP 1: Standardize each attribute to [0, 1]
    
    # 1a. Price normalization (INVERTED - lower price is better)
    all_prices = [v['price'] for v in all_vendors]
    min_price = min(all_prices)
    max_price = max(all_prices)
    
    if max_price > min_price:
        # Normalize to [0, 1] then invert
        norm_price = 1.0 - (vendor_price - min_price) / (max_price - min_price)
    else:
        # All prices are the same
        norm_price = 1.0
    
    # 1b. Quality normalization (1-5 scale)
    norm_quality = (vendor_quality - 1) / 4.0  # Maps [1, 5] → [0, 1]
    
    # 1c. Sustainability normalization (1-5 scale)
    norm_sustainability = (vendor_sustainability - 1) / 4.0  # Maps [1, 5] → [0, 1]
    
    # 1d. Proximity normalization (0-100 scale)
    norm_proximity = proximity / 100.0  # Maps [0, 100] → [0, 1]
    
    # STEP 2: Calculate weighted composite score
    composite_score = (
        weights.get('price', 0.0) * norm_price +
        weights.get('quality', 0.0) * norm_quality +
        weights.get('proximity', 0.0) * norm_proximity +
        weights.get('sustainability', 0.0) * norm_sustainability
    )
    
    return composite_score


def select_best_vendor(vendors: List[Dict], weights: Dict, 
                      proximity_scores: Dict[str, float]) -> int:
    """
    Select vendor with highest weighted composite score (deterministic).
    
    Args:
        vendors: List of vendor dicts with attributes
        weights: Weight dict from vendor_choice_weights decision
        proximity_scores: Dict mapping vendor_id (as STRING) to proximity score for this customer
        
    Returns:
        vendor_id of the best vendor (int)
    """
    
    if not vendors:
        return 1  # Default to vendor 1 if no vendors
    
    best_vendor_id = None
    best_score = -float('inf')
    
    for vendor in vendors:
        vendor_id = vendor['vendor_id']
        # Convert vendor_id to string for proximity_scores lookup
        proximity = proximity_scores.get(str(vendor_id), 50.0)  # Default to middle if missing
        
        # Calculate composite score for this vendor
        score = calculate_vendor_composite_score(vendor, weights, proximity, vendors)
        
        # Track best vendor
        if score > best_score:
            best_score = score
            best_vendor_id = vendor_id
    
    # Return best vendor ID (default to 1 if somehow none found)
    return best_vendor_id if best_vendor_id is not None else 1

