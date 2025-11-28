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
                              rng: np.random.Generator) -> Dict[str, float]:
    """
    Generate proximity scores for a specific customer-vendor dyad.
    
    Vendors have different "location characteristics" (urban/suburban/rural):
    - Urban vendors: Closer to most customers (higher average proximity ~75)
    - Suburban vendors: Medium distance to customers (average proximity ~50)
    - Rural vendors: Farther from most customers (lower average proximity ~25)
    
    Within each vendor's location distribution, there's customer-specific variation:
    - Same agent always gets same proximity to same vendor (fixed dyad)
    - Different agents get different proximities to same vendor
    - But vendors maintain meaningfully different average proximity values
    
    Args:
        agent_id: Customer/agent ID
        num_vendors: Number of vendors
        rng: Random number generator for this agent
        
    Returns:
        Dictionary mapping vendor_id (as STRING) to proximity score [0, 100]
        Note: Keys must be strings for Parquet serialization
    """
    proximity_scores = {}
    
    # Assign each vendor a distinct "location characteristic"
    # This ensures vendors have different average proximity values
    # Distribute vendors across the proximity spectrum
    
    if num_vendors == 1:
        # Single vendor: medium proximity
        vendor_means = [50.0]
    elif num_vendors == 2:
        # Two vendors: one urban (close), one rural (far)
        vendor_means = [70.0, 30.0]
    elif num_vendors == 3:
        # Three vendors: urban, suburban, rural
        vendor_means = [75.0, 50.0, 25.0]
    elif num_vendors == 4:
        # Four vendors: distribute across spectrum
        vendor_means = [80.0, 60.0, 40.0, 20.0]
    elif num_vendors == 5:
        # Five vendors: distribute evenly
        vendor_means = [85.0, 65.0, 50.0, 35.0, 15.0]
    else:
        # Many vendors: distribute evenly across 15-85 range
        vendor_means = [15 + (70 * i / (num_vendors - 1)) for i in range(num_vendors)]
    
    # SHUFFLE the means deterministically so Vendor ID != Location Quality
    # Use a fixed seed so all agents agree on which vendor is "urban" vs "rural"
    # (e.g. Vendor 3 is always the urban one in this run, regardless of agent)
    # We use a fixed seed (42) combined with num_vendors to ensure reproducibility
    # while breaking the linear correlation between Vendor ID and proximity score.
    import random
    shuffle_rng = random.Random(42 + num_vendors)
    shuffle_rng.shuffle(vendor_means)
    
    for vendor_id in range(1, num_vendors + 1):
        vendor_idx = vendor_id - 1
        mean_proximity = vendor_means[vendor_idx]
        
        # Generate proximity using normal distribution centered at vendor's location
        # Standard deviation of 20 creates customer-specific variation
        proximity = float(rng.normal(mean_proximity, 20))
        
        # Clip to [0, 100] range
        proximity = np.clip(proximity, 0.0, 100.0)
        
        # Use STRING key for Parquet compatibility
        proximity_scores[str(vendor_id)] = proximity
    
    return proximity_scores


def calculate_vendor_composite_score(vendor: Dict, weights: Dict, 
                                     proximity: float,
                                     all_vendors: List[Dict],
                                     price_min_config: float = None,
                                     price_max_config: float = None) -> float:
    """
    Calculate weighted composite score for a vendor.
    
    Steps:
    1. Standardize each attribute to [0, 1] range:
       - Price: Normalized using FIXED reference bounds (lower price = higher score)
         Uses configured price_min and price_max as the reference scale, NOT the
         actual min/max of vendor prices. This ensures price normalization is
         consistent with how other attributes are normalized (using theoretical ranges).
       - Quality: (quality - 1) / (5 - 1) = [0, 1]
       - Sustainability: (sustainability - 1) / (5 - 1) = [0, 1]
       - Proximity: proximity / 100 = [0, 1]
    
    2. Calculate weighted sum:
       score = w_price × norm_price + w_quality × norm_quality + 
               w_proximity × norm_proximity + w_sustainability × norm_sustainability
    
    NOTE: Price normalization now uses fixed reference bounds (price_min_config, price_max_config)
    to ensure equal discriminatory power across all attributes. Previously, min-max normalization
    on actual prices gave price artificially more weight than the configured percentage.
    
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
    
    # Extract vendor attributes
    vendor_price = vendor['price']
    vendor_quality = vendor['quality']
    vendor_sustainability = vendor['sustainability']
    
    # STEP 1: Standardize each attribute to [0, 1]
    
    # 1a. Price normalization (INVERTED - lower price is better)
    # Use FIXED reference bounds from configuration for consistent normalization
    # This ensures price has the same discriminatory power as other attributes
    if price_min_config is not None and price_max_config is not None:
        # Use configured bounds as fixed reference (consistent with other attributes)
        min_price = price_min_config
        max_price = price_max_config
    else:
        # Fallback: use actual vendor price range (legacy behavior)
        all_prices = [v['price'] for v in all_vendors]
        min_price = min(all_prices)
        max_price = max(all_prices)
    
    if max_price > min_price:
        # Normalize to [0, 1] then invert
        # Clamp vendor_price to bounds to avoid scores outside [0, 1]
        clamped_price = max(min_price, min(vendor_price, max_price))
        norm_price = 1.0 - (clamped_price - min_price) / (max_price - min_price)
    else:
        # All prices are the same (or invalid range)
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

