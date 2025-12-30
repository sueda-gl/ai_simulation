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
                               rng: np.random.Generator,
                               price_min: float = None,
                               price_max: float = None,
                               quantity_min: int = None,
                               quantity_max: int = None,
                               num_periods: int = 1) -> List[Dict]:
    """
    Generate vendor attributes for vendor selection.
    
    According to professor's specification:
    - Price: Randomized within [price_min, price_max] if provided, otherwise use vendor_prices
    - Quality: Random integer in [1, 5] per vendor
    - Sustainability: Random integer in [1, 5] per vendor
    - Quantity Offered: Random integer in [quantity_min, quantity_max] per vendor per period
      (Each period gets a NEW random quantity within the range)
    - Proximity: Generated per customer-vendor dyad (not here)
    
    Args:
        num_vendors: Number of vendors in the simulation
        vendor_prices: List of vendor prices (from Page 1 configuration, used if price range not provided)
        rng: Random number generator for reproducibility
        price_min: Minimum price for randomization (optional)
        price_max: Maximum price for randomization (optional)
        quantity_min: Minimum quantity offered per period (optional)
        quantity_max: Maximum quantity offered per period (optional)
        num_periods: Number of periods in the simulation (default: 1)
        
    Returns:
        List of vendor dictionaries with attributes
    """
    vendors = []
    
    for i in range(num_vendors):
        vendor_id = i + 1  # Vendors numbered 1, 2, 3, ...
        
        # Generate price - randomize if range provided, otherwise use list
        if price_min is not None and price_max is not None:
            # Randomize price within range
            price = float(rng.uniform(price_min, price_max))
        else:
            # Use provided price list
            price = vendor_prices[i] if i < len(vendor_prices) else 100.0
        
        # Generate quality: random integer in [1, 5]
        quality = int(rng.integers(1, 6))  # 6 is exclusive, so generates 1-5
        
        # Generate sustainability: random integer in [1, 5]  
        sustainability = int(rng.integers(1, 6))
        
        # Generate quantity offered PER PERIOD
        # Each period gets a NEW random quantity within [quantity_min, quantity_max]
        quantity_offered_per_period = {}
        
        if quantity_min is not None and quantity_max is not None:
            for period in range(1, num_periods + 1):
                # Generate new random quantity for this period
                quantity_offered_per_period[period] = int(rng.integers(quantity_min, quantity_max + 1))
        else:
            # Default: 100 products per period
            for period in range(1, num_periods + 1):
                quantity_offered_per_period[period] = 100
        
        # For backward compatibility, keep quantity_offered as average across all periods
        # This is used for display and any code that hasn't been updated yet
        quantity_offered = int(np.mean(list(quantity_offered_per_period.values())))
        
        vendor = {
            'vendor_id': vendor_id,
            'price': float(price),
            'quality': quality,
            'sustainability': sustainability,
            'quantity_offered': quantity_offered,  # Average (for backward compatibility)
            'quantity_offered_per_period': quantity_offered_per_period  # NEW: Period-specific quantities
            # Note: proximity is NOT included here - it's customer-vendor specific
        }
        
        vendors.append(vendor)
    
    return vendors


def generate_proximity_scores(agent_id: int, num_vendors: int, 
                              rng: np.random.Generator,
                              base_seed: int = 42) -> Dict[str, float]:
    """
    Generate proximity scores for a specific customer-vendor dyad.
    
    Each agent-vendor proximity is generated randomly and independently:
    - Uniformly distributed in [0, 100] range
    - Same agent always gets same proximity to same vendor (fixed dyad)
    - Different agents get different proximities to same vendor
    - No predefined vendor "location types" - purely random
    
    Args:
        agent_id: Customer/agent ID
        num_vendors: Number of vendors
        rng: Random number generator for this agent
        base_seed: Base seed (kept for API compatibility, not used)
        
    Returns:
        Dictionary mapping vendor_id (as STRING) to proximity score [0, 100]
        Note: Keys must be strings for Parquet serialization
    """
    proximity_scores = {}
    
    for vendor_id in range(1, num_vendors + 1):
        # Generate purely random proximity for each agent-vendor pair
        # Uniform distribution in [0, 100]
        proximity = float(rng.uniform(0.0, 100.0))
        
        # Use STRING key for Parquet compatibility
        proximity_scores[str(vendor_id)] = proximity
    
    return proximity_scores


def calculate_vendor_score_with_breakdown(vendor: Dict, weights: Dict, 
                                          proximity: float,
                                          all_vendors: List[Dict] = None,
                                          price_min_config: float = None,
                                          price_max_config: float = None) -> Dict:
    """
    Calculate vendor integrated score with full breakdown of normalized and weighted components.
    
    THIS IS THE SINGLE SOURCE OF TRUTH for all vendor score calculations.
    All other code should import and use this function instead of re-implementing the logic.
    
    Normalization formulas:
    - Price: 1.0 - (clamped_price - min_price) / (max_price - min_price)  [inverted, lower=better]
    - Quality: (value - 1) / 4  [maps 1-5 to 0-1]
    - Sustainability: (value - 1) / 4  [maps 1-5 to 0-1]
    - Proximity: value / 100  [maps 0-100 to 0-1]
    
    Args:
        vendor: Vendor dict with 'price', 'quality', 'sustainability' keys
        weights: Weight dict with 'price', 'quality', 'proximity', 'sustainability' keys
        proximity: Proximity score for this customer-vendor dyad [0-100]
        all_vendors: List of all vendors (for fallback price bounds if config not provided)
        price_min_config: Configured minimum price bound (from vendor_price_min)
        price_max_config: Configured maximum price bound (from vendor_price_max)
        
    Returns:
        dict: {
            'integrated_score': float,      # Final weighted score
            'norm_price': float,            # Normalized price [0,1]
            'norm_quality': float,          # Normalized quality [0,1]
            'norm_sustainability': float,   # Normalized sustainability [0,1]
            'norm_proximity': float,        # Normalized proximity [0,1]
            'weighted_price': float,        # weight_price * norm_price
            'weighted_quality': float,      # weight_quality * norm_quality
            'weighted_sustainability': float,  # weight_sustainability * norm_sustainability
            'weighted_proximity': float,    # weight_proximity * norm_proximity
            'weight_price': float,          # The price weight used
            'weight_quality': float,        # The quality weight used
            'weight_sustainability': float, # The sustainability weight used
            'weight_proximity': float       # The proximity weight used
        }
    """
    # Extract vendor attributes with safe defaults
    vendor_price = vendor.get('price', 0)
    vendor_quality = vendor.get('quality', 3)
    vendor_sustainability = vendor.get('sustainability', 3)
    
    # Determine price bounds for normalization
    if price_min_config is not None and price_max_config is not None:
        min_price = price_min_config
        max_price = price_max_config
    elif all_vendors and len(all_vendors) > 0:
        all_prices = [v.get('price', 0) for v in all_vendors]
        min_price = min(all_prices)
        max_price = max(all_prices)
    else:
        min_price = 0
        max_price = 1
    
    # === NORMALIZE EACH ATTRIBUTE TO [0, 1] ===
    
    # Price normalization (INVERTED - lower price is better)
    if max_price > min_price:
        clamped_price = max(min_price, min(vendor_price, max_price))
        norm_price = 1.0 - (clamped_price - min_price) / (max_price - min_price)
    else:
        norm_price = 1.0
    
    # Quality normalization: [1, 5] → [0, 1]
    norm_quality = (vendor_quality - 1) / 4.0 if vendor_quality >= 1 else 0.0
    
    # Sustainability normalization: [1, 5] → [0, 1]
    norm_sustainability = (vendor_sustainability - 1) / 4.0 if vendor_sustainability >= 1 else 0.0
    
    # Proximity normalization: [0, 100] → [0, 1]
    norm_proximity = proximity / 100.0 if proximity >= 0 else 0.0
    
    # === GET WEIGHTS ===
    weight_price = weights.get('price', 0.0)
    weight_quality = weights.get('quality', 0.0)
    weight_proximity = weights.get('proximity', 0.0)
    weight_sustainability = weights.get('sustainability', 0.0)
    
    # === CALCULATE WEIGHTED COMPONENTS ===
    weighted_price = weight_price * norm_price
    weighted_quality = weight_quality * norm_quality
    weighted_proximity = weight_proximity * norm_proximity
    weighted_sustainability = weight_sustainability * norm_sustainability
    
    # === CALCULATE INTEGRATED SCORE ===
    integrated_score = weighted_price + weighted_quality + weighted_proximity + weighted_sustainability
    
    return {
        'integrated_score': integrated_score,
        'norm_price': norm_price,
        'norm_quality': norm_quality,
        'norm_sustainability': norm_sustainability,
        'norm_proximity': norm_proximity,
        'weighted_price': weighted_price,
        'weighted_quality': weighted_quality,
        'weighted_sustainability': weighted_sustainability,
        'weighted_proximity': weighted_proximity,
        'weight_price': weight_price,
        'weight_quality': weight_quality,
        'weight_sustainability': weight_sustainability,
        'weight_proximity': weight_proximity
    }


def calculate_vendor_composite_score(vendor: Dict, weights: Dict, 
                                     proximity: float,
                                     all_vendors: List[Dict],
                                     price_min_config: float = None,
                                     price_max_config: float = None) -> float:
    """
    Calculate weighted composite score for a vendor.
    
    This is a convenience wrapper around calculate_vendor_score_with_breakdown()
    that returns only the final score. Use this when you don't need the breakdown.
    
    Args:
        vendor: Vendor dict with price, quality, sustainability
        weights: Weight dict from vendor_choice_weights decision
        proximity: Proximity score for this customer-vendor dyad
        all_vendors: List of all vendors (for fallback if config not provided)
        price_min_config: Configured minimum price bound (from vendor_price_min)
        price_max_config: Configured maximum price bound (from vendor_price_max)
        
    Returns:
        Composite score (float)
    """
    result = calculate_vendor_score_with_breakdown(
        vendor=vendor,
        weights=weights,
        proximity=proximity,
        all_vendors=all_vendors,
        price_min_config=price_min_config,
        price_max_config=price_max_config
    )
    return result['integrated_score']


def select_best_vendor(vendors: List[Dict], weights: Dict, 
                      proximity_scores: Dict[str, float],
                      price_min_config: float = None,
                      price_max_config: float = None) -> int:
    """
    Select vendor with highest weighted composite score (deterministic).
    
    Args:
        vendors: List of vendor dicts with attributes
        weights: Weight dict from vendor_choice_weights decision
        proximity_scores: Dict mapping vendor_id (as STRING) to proximity score for this customer
        price_min_config: Configured minimum price bound (from vendor_price_min)
        price_max_config: Configured maximum price bound (from vendor_price_max)
        
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
        score = calculate_vendor_composite_score(
            vendor, weights, proximity, vendors,
            price_min_config=price_min_config,
            price_max_config=price_max_config
        )
        
        # Track best vendor
        if score > best_score:
            best_score = score
            best_vendor_id = vendor_id
    
    # Return best vendor ID (default to 1 if somehow none found)
    return best_vendor_id if best_vendor_id is not None else 1

