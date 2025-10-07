# src/decisions/disclose_documents.py
import numpy as np
from scipy import stats

def disclose_documents(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 2: Disclose documents for Discount status
    
    This decision only applies to agents who qualify for discount (income below threshold).
    For agents with income >= threshold, returns "NA" (not applicable).
    """
    
    # STEP 1: Generate or retrieve agent's income
    # Income must be consistent across all decisions, so we generate it once and store it
    if 'income' not in agent_state:
        # Generate income using distribution parameters from simulation config
        if simulation_config and 'simulation' in simulation_config:
            sim_params = simulation_config['simulation']
            income_dist = sim_params.get('income_distribution', 'lognormal')
            
            # Generate income based on distribution type
            if income_dist == 'lognormal':
                mu = sim_params.get('lognormal_mu', 10.0)
                sigma = sim_params.get('lognormal_sigma', 0.5)
                min_val = sim_params.get('lognormal_min', 0.0)
                max_val = sim_params.get('lognormal_max', None)
                
                # Sample from lognormal: X = min + Y where Y ~ Lognormal(mu, sigma)
                Y = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=1, random_state=rng)[0]
                income = min_val + Y
                
                # Apply max if specified
                if max_val is not None:
                    income = min(income, max_val)
                    
            elif income_dist == 'generalised_gamma':
                k = sim_params.get('gg_k', 1.5)
                c = sim_params.get('gg_c', 2.0)
                lambda_param = sim_params.get('gg_lambda', 20000.0)
                min_val = sim_params.get('gg_min', 0.0)
                max_val = sim_params.get('gg_max', None)
                
                # Sample from Generalised Gamma
                Y = stats.gengamma.rvs(a=c, c=k, scale=lambda_param, size=1, random_state=rng)[0]
                income = min_val + Y
                
                # Apply max if specified
                if max_val is not None:
                    income = min(income, max_val)
                    
            elif income_dist == 'dagum':
                a = sim_params.get('dagum_a', 2.0)
                p = sim_params.get('dagum_p', 1.5)
                b = sim_params.get('dagum_b', 25000.0)
                min_val = sim_params.get('dagum_min', 0.0)
                max_val = sim_params.get('dagum_max', None)
                
                # Sample from Dagum using inverse CDF
                U = rng.random()
                income = b * np.power(np.power(U, -1/p) - 1, -1/a)
                income = min_val + income
                
                # Apply max if specified
                if max_val is not None:
                    income = min(income, max_val)
            else:
                # Fallback: uniform distribution
                min_val = sim_params.get('income_min', 0.0)
                max_val = sim_params.get('income_max', 100000.0)
                income = rng.uniform(min_val, max_val)
            
            # Store income in agent_state for consistency across decisions
            agent_state['income'] = income
        else:
            # No simulation config available - use a default middle value
            agent_state['income'] = 50000.0
    
    # STEP 2: Check eligibility based on income threshold
    agent_income = agent_state.get('income', 50000.0)
    
    # Get discount threshold from simulation config
    threshold = 12500.0  # Default fallback
    if simulation_config and 'simulation' in simulation_config:
        threshold = simulation_config['simulation'].get('discount_income_threshold', 12500.0)
    
    # If agent's income is >= threshold, they don't qualify for discount
    # So disclose_documents decision does not apply
    if agent_income >= threshold:
        return {"disclose_documents": "NA"}
    
    # STEP 3: Agent qualifies for discount - apply probability-based decision
    # Check if probability settings are available from simulation config
    if simulation_config and 'random_decisions' in simulation_config:
        prob_config = simulation_config['random_decisions'].get('disclose_documents')
        if prob_config and prob_config.get("type") == "random_probability":
            # Use weighted random choice with proper RNG
            probability_y = prob_config.get("probability_y", 0.5)
            options = prob_config.get("options", ["Y", "N"])
            
            # Use provided RNG for reproducibility
            if rng.random() < probability_y:
                choice = options[0]  # Y
            else:
                choice = options[1]  # N
            return {"disclose_documents": choice}
    
    # Fallback to simple 50/50 random choice for qualified agents
    choice = rng.choice(["Y", "N"])
    return {"disclose_documents": choice}