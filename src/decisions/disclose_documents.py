# src/decisions/disclose_documents.py
"""
Decision 2: Disclose documents for Discount status

This decision only applies to agents who qualify for discount (income below threshold).
For agents with income >= threshold, returns "NA" (not applicable).
"""

from src.decisions.income_utils import get_agent_income, get_simulation_param, get_customer_type


def disclose_documents(agent_state: dict, params: dict, rng, simulation_config: dict = None) -> dict:
    """Decision 2: Disclose documents for Discount status
    
    This decision only applies to agents who qualify for discount (income below threshold).
    For agents with income >= threshold, returns "NA" (not applicable).
    """
    
    # STEP 1: Get or generate agent's income using centralized utility
    # This is the single source of truth for income generation
    income = get_agent_income(agent_state, simulation_config, rng)
    
    # STEP 2: Check eligibility based on income threshold
    agent_income = income  # Use the income we just retrieved/generated
    
    # Get discount threshold from simulation config using helper
    threshold = get_simulation_param(simulation_config, 'discount_income_threshold', 12500.0)
    
    # If agent's income is >= threshold, they don't qualify for discount
    # So disclose_documents decision does not apply
    if agent_income >= threshold:
        # Update agent_state BEFORE determining customer type
        agent_state['disclose_documents'] = "NA"
        
        # Determine customer type (will be fixed or regular based on disclose_income)
        customer_type = get_customer_type(agent_state, simulation_config)
        return {
            "disclose_documents": "NA",
            "customer_type": customer_type
        }
    
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
            
            # Update agent_state BEFORE determining customer type
            agent_state['disclose_documents'] = choice
            
            # Now determine customer type (needs disclose_documents to be set)
            customer_type = get_customer_type(agent_state, simulation_config)
            return {
                "disclose_documents": choice,
                "customer_type": customer_type
            }
    
    # Fallback to simple 50/50 random choice for qualified agents
    choice = rng.choice(["Y", "N"])
    
    # Update agent_state BEFORE determining customer type
    agent_state['disclose_documents'] = choice
    
    # Now determine customer type (needs disclose_documents to be set)
    customer_type = get_customer_type(agent_state, simulation_config)
    return {
        "disclose_documents": choice,
        "customer_type": customer_type
    }