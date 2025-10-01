# src/decisions/bid_value.py
import numpy as np

def bid_value(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """
    Decision 10: For Regular customers - select bid value if Bid chosen
    
    Generates a random bid amount within the calculated bidding range using the formula:
        Baseline Price (Pc) = (1 + platform_markup) × vendor_price
        Min Bid (Pmb) = (1 - price_range) × Pc
        Max Bid (Ppn) = (1 + price_range) × Pc
        Bid Value = Uniform random in [Pmb, Ppn]
    
    Args:
        agent_state: Current agent state including previous decisions
        params: Decision-specific parameters from decisions.yaml
        rng: Random number generator for this agent
        simulation_config: Global simulation configuration with pricing parameters
        
    Returns:
        dict: {"bid_value": float or np.nan}
    """
    # Check if agent chose to bid (if not, return NaN)
    purchase_vs_bid_choice = agent_state.get('purchase_vs_bid', 'purchase')
    if purchase_vs_bid_choice != 'bid':
        # Agent chose to purchase immediately, so no bid value needed
        return {"bid_value": np.nan}
    
    # Get pricing parameters from simulation_config
    if simulation_config and 'simulation' in simulation_config:
        sim_params = simulation_config['simulation']
        vendor_price = sim_params.get('market_price', 100.0)
        platform_markup = sim_params.get('platform_markup', 0.1)
        price_range = sim_params.get('price_range', 0.25)
    else:
        # Fallback to default values if parameters not provided
        vendor_price = 100.0
        platform_markup = 0.1
        price_range = 0.25
    
    # Calculate bidding range using the formula
    baseline_price = (1 + platform_markup) * vendor_price  # Pc = (1+m) × vendor_price
    min_bid_price = (1 - price_range) * baseline_price     # Pmb = (1-r) × Pc
    max_bid_price = (1 + price_range) * baseline_price     # Ppn = (1+r) × Pc
    
    # Generate random bid value within the range using provided RNG
    # Use uniform distribution over [min_bid_price, max_bid_price]
    bid_amount = rng.uniform(min_bid_price, max_bid_price)
    
    # Round to 2 decimal places (standard for currency)
    bid_amount = round(bid_amount, 2)
    
    return {"bid_value": bid_amount}