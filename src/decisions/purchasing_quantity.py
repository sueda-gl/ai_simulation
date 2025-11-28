# src/decisions/purchasing_quantity.py
"""
Decision 6: purchasing_quantity

Determines the total quantity of items an agent wishes to purchase during the term
and generates purchase requests with timestamps.

Default behavior (when not simulated):
1. Uses agent's income to assign to an income category (equal interval division)
2. Determines customer type (discount, fixed, or regular)
3. Gets purchasing limit based on customer type:
   - Discount customers: Use Category 1 limit (lowest)
   - Regular customers: Use Category N limit (highest)
   - Fixed customers: Use their actual income category limit
4. Generates random total quantity in [0, limit]
5. Creates purchase requests (1 item each) with random timestamps across term

⚠️ IMPORTANT: Consumption limits apply to COMPLETED TRANSACTIONS, not REQUESTS
- In default mode: We generate requests up to the limit (assuming 100% completion)
- In reality: Agents could make MORE requests than the limit, anticipating rejections
- Example: Limit=50, agent makes 100 requests, 50% complete = 50 transactions (OK!)
- When Decision 10 is fully simulated, this behavior will be updated

When fully simulated (future):
- Decouple request generation from limit enforcement
- Simulate transaction completion/rejection outcomes
- Enforce limit on completed transactions (not requests)
- More sophisticated models for quantity per purchase
- Consideration of agent traits, budget constraints, etc.
"""
import numpy as np
from typing import Dict, Any, Optional, List

from src.decisions.income_utils import get_agent_income, get_simulation_param


def _calculate_preferred_vendor(agent_state: dict, simulation_config: dict, rng) -> int:
    """
    Calculate which vendor the agent prefers based on weighted composite scores.
    
    This uses the same logic as vendor_selection but happens at request creation time.
    
    Returns:
        int: vendor_id of the preferred vendor (highest score)
    """
    import numpy as np
    
    # Get vendors from simulation_config
    vendors = simulation_config.get('vendors', [])
    
    if not vendors or len(vendors) == 0:
        # No vendors configured - default to vendor 1
        return 1
    
    # Get vendor choice weights from agent_state (set by Decision 5)
    weights = agent_state.get('vendor_choice_weights', {
        'price': 0.25,
        'quality': 0.25,
        'proximity': 0.25,
        'sustainability': 0.25
    })
    
    # Generate proximity scores for this agent (customer-vendor dyad)
    if 'vendor_proximity_scores' not in agent_state:
        agent_id = agent_state.get('agent_id', agent_state.get('index', 0) + 1)
        from src.vendor_attribute_generator import generate_proximity_scores
        proximity_scores = generate_proximity_scores(agent_id, len(vendors), rng)
        agent_state['vendor_proximity_scores'] = proximity_scores
    else:
        proximity_scores = agent_state['vendor_proximity_scores']
    
    # Calculate composite score for ALL vendors
    # Get configured price bounds for consistent normalization
    price_min_config = simulation_config.get('vendor_price_min', 50.0)
    price_max_config = simulation_config.get('vendor_price_max', 150.0)
    
    from src.vendor_attribute_generator import calculate_vendor_composite_score
    vendor_scores = []
    for vendor in vendors:
        vendor_id = vendor['vendor_id']
        proximity = proximity_scores.get(str(vendor_id), 50.0)
        score = calculate_vendor_composite_score(
            vendor, weights, proximity, vendors,
            price_min_config=price_min_config,
            price_max_config=price_max_config
        )
        vendor_scores.append((vendor_id, score))
    
    # Sort by score descending (best vendor first)
    vendor_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return the vendor with the highest score
    preferred_vendor_id = vendor_scores[0][0] if vendor_scores else 1
    
    # Store preferred vendor in agent state for reference
    agent_state['preferred_vendor'] = preferred_vendor_id
    
    return preferred_vendor_id


def _enrich_purchase_requests(requests: List[Dict], customer_type: str, rng: np.random.Generator, 
                               simulation_config: Optional[Dict], agent_state: dict) -> List[Dict]:
    """
    Enrich purchase requests with per-request decisions:
    - platformPrice and bid_value (existing)
    - final_donation_rate (NEW)
    
    This function adds these fields to each purchase request based on:
    - customer_type: "discount", "fixed", or "regular"
    - agent_state: Contains agent-level decisions like donation_default and final_donation_rate
    - For regular customers: Makes purchase_vs_bid decision and generates bid_value if needed
    
    Args:
        requests: List of purchase request dictionaries
        customer_type: Customer type ("discount", "fixed", or "regular")
        rng: Random number generator
        simulation_config: Simulation configuration
        agent_state: Agent's current state with all decisions up to this point
        
    Returns:
        Enriched list of purchase requests
    """
    from src.decisions.purchase_vs_bid import purchase_vs_bid_single
    from src.decisions.bid_value import generate_single_bid_value
    
    # ========================================================================
    # NEW: Get agent's baseline donation rate
    # ========================================================================
    # Priority order:
    # 1. Use final_donation_rate if it exists (Decision 13 ran)
    # 2. Fall back to donation_default if it exists (Decision 3 ran)
    # 3. Fall back to 0.10 (10%) if neither exists
    
    agent_baseline_rate = agent_state.get('final_donation_rate', 
                                          agent_state.get('donation_default', 0.10))
    
    # Convert to float if it's not already (handles string "0.10" or numeric)
    try:
        agent_baseline_rate = float(agent_baseline_rate)
    except (ValueError, TypeError):
        agent_baseline_rate = 0.10  # Default if conversion fails
    
    # Ensure it's in valid range [0, 1]
    agent_baseline_rate = np.clip(agent_baseline_rate, 0.0, 1.0)
    # ========================================================================
    
    enriched_requests = []
    
    for request in requests:
        # Create a copy to avoid modifying original
        enriched_request = request.copy()
        
        # ====================================================================
        # EXISTING: Set platformPrice and bid_value based on customer type
        # ====================================================================
        if customer_type.lower() == 'discount':
            enriched_request['platformPrice'] = 'DISCOUNT'
            enriched_request['bid_value'] = 'N/A'
        
        elif customer_type.lower() == 'fixed':
            enriched_request['platformPrice'] = 'FIXED'
            enriched_request['bid_value'] = 'N/A'
        
        elif customer_type.lower() == 'regular':
            # Make purchase_vs_bid decision for this specific request
            decision = purchase_vs_bid_single(customer_type, {}, rng, simulation_config)
            
            if decision == 'bid':
                enriched_request['platformPrice'] = 'BID'
                # Generate unique bid value for this request
                bid_amount = generate_single_bid_value(rng, simulation_config, {})
                enriched_request['bid_value'] = bid_amount
            else:
                # Purchase Now
                enriched_request['platformPrice'] = 'PN'
                enriched_request['bid_value'] = 'N/A'
        else:
            # Unknown customer type - default to PN
            enriched_request['platformPrice'] = 'PN'
            enriched_request['bid_value'] = 'N/A'
        
        # ====================================================================
        # NEW: Add final_donation_rate to this specific request
        # ====================================================================
        # For now, use the agent's baseline rate for each request
        # This creates the infrastructure for per-request rates
        # Later, you can add variation here if needed:
        #   - Random variation: agent_baseline_rate * rng.normal(1.0, 0.1)
        #   - Price-based: higher price → higher donation
        #   - Time-based: later requests → different rates
        
        enriched_request['final_donation_rate'] = agent_baseline_rate
        # ====================================================================
        
        enriched_requests.append(enriched_request)
    
    return enriched_requests


def _assign_income_category(income: float, simulation_config: Optional[Dict]) -> int:
    """
    Assign agent to an income category (1 to N) based on their income.
    
    PROFESSOR'S SPECIFICATION:
    - The income range is split into N equal intervals
    - ALL customers (discount, fixed, regular) are assigned to categories based ONLY on income
    - NO distinction by customer type during category assignment
    - Customer type only affects which purchasing limit applies (not which category)
    
    IMPLEMENTATION:
    - Uses the ACTUAL income range generated by the income generation system
    - Min = PPF(0.00) - the lowest percentile used (Level 1 agents)
    - Max = PPF(1.00) or configured max - the highest percentile used (Level 5 agents)
    - This ensures categories match the real income distribution, not theoretical bounds
    
    Example with Lognormal(μ=10, σ=0.5) and N=14:
        Min income: ~$4,697 (PPF(0.00))
        Max income: ~$176,000 (PPF(1.00))
        Category 1:  [$4,697  - $16,943)   - Populated by Level 1 agents
        Category 2:  [$16,943 - $29,189)
        ...
        Category 14: [$151,508 - $176,000]  - Populated by Level 5 agents
    
    Args:
        income: Agent's annual income (dollar amount)
        simulation_config: Contains num_fixed_categories and distribution parameters
        
    Returns:
        Income category number (1 to N, where N = num_fixed_categories)
    """
    # Get number of income categories from Page 1
    num_categories = get_simulation_param(simulation_config, 'num_fixed_categories', 10)
    num_categories = max(1, int(num_categories))
    
    # If only 1 category exists, everyone goes there
    if num_categories == 1:
        return 1
    
    # STEP 1: Determine the ACTUAL income range that can be generated
    # This matches what _generate_income_within_percentile_range() produces
    # Level 1 uses percentile 0.00-0.20, Level 5 uses 0.80-1.00
    # So the actual range is PPF(0.00) to PPF(1.00)
    
    dist_type = get_simulation_param(simulation_config, 'income_distribution', 'lognormal')
    sim_params = simulation_config.get('simulation', {}) if simulation_config else {}
    
    # Calculate min_income using PPF(0.00) - the actual minimum that can be generated
    if dist_type == 'dagum':
        # Dagum: use inverse CDF formula for percentile 0.00
        a = get_simulation_param(simulation_config, 'dagum_a', 2.0)
        p = get_simulation_param(simulation_config, 'dagum_p', 1.5)
        b = get_simulation_param(simulation_config, 'dagum_b', 25000.0)
        min_val = get_simulation_param(simulation_config, 'dagum_min', 0.0)
        
        # PPF(0.00) for Dagum approaches min_val
        # Use a very small percentile instead (PPF of near-zero)
        small_percentile = 0.0001
        Y_min = b * np.power(np.power(small_percentile, -1/p) - 1, -1/a)
        min_income = min_val + Y_min
    else:
        # For lognormal, generalised_gamma, etc: use scipy
        from scipy import stats
        
        if dist_type == 'lognormal':
            mu = get_simulation_param(simulation_config, 'lognormal_mu', 10.0)
            sigma = get_simulation_param(simulation_config, 'lognormal_sigma', 0.5)
            loc = get_simulation_param(simulation_config, 'lognormal_min', 0.0)
            dist = stats.lognorm(s=sigma, scale=np.exp(mu), loc=loc)
        elif dist_type == 'generalised_gamma':
            k = get_simulation_param(simulation_config, 'gg_k', 1.5)
            c = get_simulation_param(simulation_config, 'gg_c', 2.0)
            lambda_param = get_simulation_param(simulation_config, 'gg_lambda', 20000.0)
            loc = get_simulation_param(simulation_config, 'gg_min', 0.0)
            dist = stats.gengamma(a=c, c=k, scale=lambda_param, loc=loc)
        else:
            # Fallback: uniform distribution
            min_val = get_simulation_param(simulation_config, 'income_min', 0.0)
            max_val = get_simulation_param(simulation_config, 'income_max', 100000.0)
            dist = stats.uniform(loc=min_val, scale=max_val - min_val)
        
        # Use a very small percentile (not exactly 0.00 to avoid numerical issues)
        min_income = dist.ppf(0.0001)
    
    # Calculate max_income using PPF(1.00) or configured max
    if dist_type == 'dagum':
        max_income = get_simulation_param(simulation_config, 'dagum_max', None)
        if max_income is None:
            # Use a high percentile for Dagum (approaching 1.00)
            high_percentile = 0.9999
            Y_max = b * np.power(np.power(high_percentile, -1/p) - 1, -1/a)
            max_income = min_val + Y_max
    else:
        # Check if there's a configured max
        if dist_type == 'lognormal':
            max_income = get_simulation_param(simulation_config, 'lognormal_max', None)
        elif dist_type == 'generalised_gamma':
            max_income = get_simulation_param(simulation_config, 'gg_max', None)
        else:
            max_income = get_simulation_param(simulation_config, 'income_max', None)
        
        # If no max configured, use PPF(0.9999) to get the actual high end
        if max_income is None:
            max_income = dist.ppf(0.9999)
    
    # Safety check: ensure max > min
    if max_income <= min_income:
        max_income = min_income + 100000.0  # Default range
    
    # STEP 2: Calculate interval width
    income_range = max_income - min_income
    interval_width = income_range / num_categories
    
    # STEP 3: Determine which interval the agent's income falls into
    # Category i contains incomes in range [min + (i-1)*width, min + i*width)
    # Special case: Category N includes the upper bound
    
    if income <= min_income:
        # Edge case: income at or below minimum → Category 1
        return 1
    elif income >= max_income:
        # Edge case: income at or above maximum → Category N (highest)
        return num_categories
    else:
        # Normal case: find which interval contains this income
        position = (income - min_income) / income_range  # Fraction in [0, 1]
        category_index = int(np.floor(position * num_categories))  # 0 to N-1
        category = category_index + 1  # Convert to 1-based (1 to N)
        
        # Safety clamp to valid range
        category = max(1, min(category, num_categories))
        
        return int(category)


def purchasing_quantity(agent_state: dict, params: dict, rng: np.random.Generator, 
                        simulation_config: dict = None, **kwargs) -> dict:
    """
    Decision 6: Determine purchasing quantity and generate purchase requests.
    
    Default behavior (this implementation):
    1. Get/generate agent income
    2. Assign to income category (1 to NFIC) based on equal intervals
    3. Determine customer type (discount/fixed/regular)
    4. Get purchasing limit based on customer type:
       - Discount: Use Category 1 limit
       - Regular: Use Category N limit  
       - Fixed: Use actual income category limit
    5. Generate random total quantity in [0, limit]
    6. Create purchase requests (1 item each) spread randomly across term
    
    Args:
        agent_state: Agent's state dict (includes 'income' if set by disclose_documents)
        params: Decision-specific parameters from decisions.yaml
        rng: Random number generator for reproducibility
        simulation_config: Page 1 parameters in ['simulation'] sub-dict
        
    Returns:
        Dictionary with:
        - purchasing_quantity: Total items to purchase (int)
        - purchase_requests: List of dicts with request_id, quantity=1, timestamp_hours
        - income_category: Assigned category (int, 1 to NFIC)
        - income: Agent income (stored for consistency)
    """
    
    # STEP 1: Get or generate agent income using centralized utility
    # Income should already exist from disclose_documents (Decision 2)
    # If not, get_agent_income will generate it using Page 1 parameters
    income = get_agent_income(agent_state, simulation_config, rng)
    
    # STEP 2: Assign agent to income category (1 to NFIC)
    income_category = _assign_income_category(income, simulation_config)
    
    # STEP 3: Determine customer type (discount, fixed, or regular)
    from src.decisions.income_utils import get_customer_type
    customer_type = get_customer_type(agent_state, simulation_config)
    
    # STEP 4: Get purchasing limit based on CUSTOMER TYPE
    # According to professor's specification:
    # - Discount customers: Use Category 1 limit (lowest)
    # - Regular customers: Use Category N limit (highest)
    # - Fixed customers: Use their actual income category limit
    
    # Check if purchasing limits are enabled and configured
    purchasing_limits = {}
    if isinstance(simulation_config, dict):
        purchasing_limits = simulation_config.get('purchasing_limits', {}) or {}
    
    # Get fallback maximum for when limits are disabled
    fallback_max = get_simulation_param(simulation_config, 'max_purchases_per_term', 50)
    fallback_max = max(0, int(fallback_max))
    
    # Determine which category's limit to use based on customer type
    num_categories = get_simulation_param(simulation_config, 'num_fixed_categories', 10)
    
    if customer_type == "discount":
        # Discount customers always use Category 1 limit (lowest)
        limit_category = 1
    elif customer_type == "regular":
        # Regular customers always use Category N limit (highest)
        limit_category = num_categories
    else:  # customer_type == "fixed"
        # Fixed customers use their actual income category limit
        limit_category = income_category
    
    # Look up limit for the determined category
    limit_key = f"cat_{limit_category}"
    if isinstance(purchasing_limits, dict) and limit_key in purchasing_limits:
        try:
            purchasing_limit = int(purchasing_limits[limit_key])
        except (ValueError, TypeError):
            purchasing_limit = fallback_max
    else:
        # No limit configured for this category - use fallback
        purchasing_limit = fallback_max
    
    purchasing_limit = max(0, purchasing_limit)  # Ensure non-negative
    
    # STEP 5: Generate total quantity for the term
    # Random integer in [0, purchasing_limit] inclusive
    # 
    # ⚠️ IMPORTANT NOTE: This is DEFAULT BEHAVIOR when transaction outcomes are not simulated
    # In reality, the consumption LIMIT applies to COMPLETED TRANSACTIONS, not PURCHASE REQUESTS
    # An agent could make MORE requests than the limit (e.g., 100 requests with 50% completion rate)
    # For now, we use the limit as a proxy for requests, assuming 100% completion
    # When Decision 10 (rejected_transaction_option) is fully simulated, this will be decoupled
    if purchasing_limit > 0:
        total_quantity = int(rng.integers(0, purchasing_limit + 1))
    else:
        total_quantity = 0
    
    # STEP 6: Generate purchase requests with timestamps and vendor preference
    # Each request = 1 item for defaults
    # Timestamps spread randomly across term duration
    # 
    # NOTE: These are REQUESTS only, not actual transactions
    # The system tracks what agents want to purchase, not transaction outcomes
    
    purchase_requests = []
    
    if total_quantity > 0:
        # Calculate term duration from Page 1 parameters
        periods = get_simulation_param(simulation_config, 'periods', 1)
        duration_hours = get_simulation_param(simulation_config, 'duration_hours', 1.0)
        term_duration = float(periods * duration_hours)
        
        # Generate random timestamps uniformly distributed in [0, term_duration]
        # Sort them so they're in chronological order
        timestamps = sorted(rng.uniform(0, term_duration, size=total_quantity))
        
        # Get agent ID (use agent_id if available, otherwise index + 1)
        agent_id = agent_state.get('agent_id', agent_state.get('index', 0) + 1)
        
        # Get customer type from agent_state (should be set by disclose_documents)
        from src.decisions.income_utils import get_customer_type
        customer_type = get_customer_type(agent_state, simulation_config)
        
        # STEP 6a: Calculate preferred vendor based on scores
        # This determines which vendor the agent wants to buy from
        preferred_vendor_id = _calculate_preferred_vendor(agent_state, simulation_config, rng)
        
        # Create purchase request objects with vendor preference
        for i, timestamp in enumerate(timestamps):
            purchase_requests.append({
                "request_id": i + 1,
                "quantity": 1,  # 1 item per request for defaults
                "timestamp_hours": float(timestamp),
                
                # Basic request metadata
                "customer_id": int(agent_id),
                "customer_type": customer_type,  # "discount", "fixed", or "regular"
                "vendorID": preferred_vendor_id  # Vendor agent wants to buy from (based on scores)
            })
    
    # STEP 7: Enrich purchase requests with per-request decisions
    # (platformPrice, bid_value, final_donation_rate)
    # This happens AFTER basic requests are created but BEFORE returning
    if purchase_requests:
        purchase_requests = _enrich_purchase_requests(
            requests=purchase_requests,
            customer_type=customer_type,
            rng=rng,
            simulation_config=simulation_config,
            agent_state=agent_state  # NEW: Pass agent_state for donation rates
        )
    
    # Return results
    return {
        "purchasing_quantity": int(total_quantity),
        "purchase_requests": purchase_requests,
        "income_category": int(income_category),
        "income": float(income)  # Store/update income for consistency
    }


