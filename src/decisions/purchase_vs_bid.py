# src/decisions/purchase_vs_bid.py

def purchase_vs_bid(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 9: For Regular customers - decide between Purchase Now and Bid"""
    
    # Check if probability settings are available from simulation config
    if simulation_config and 'random_decisions' in simulation_config:
        prob_config = simulation_config['random_decisions'].get('purchase_vs_bid')
        if prob_config and prob_config.get("type") == "random_probability":
            # Use weighted random choice with proper RNG
            probability_y = prob_config.get("probability_y", 0.5)
            options = prob_config.get("options", ["purchase", "bid"])
            
            # Use provided RNG for reproducibility
            if rng.random() < probability_y:
                choice = options[0]  # Purchase Now
            else:
                choice = options[1]  # bid
            return {"purchase_vs_bid": choice}
    
    # Fallback to simple 50/50 random choice
    choice = rng.choice(["Purchase Now", "bid"])
    return {"purchase_vs_bid": choice}