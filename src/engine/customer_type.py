# src/engine/customer_type.py
"""
Customer type (discount / fixed / regular) from the two disclosure decisions and the
agent's income, cached on the agent state once determined.
"""

from src.engine.config_access import get_simulation_param


def determine_customer_type(agent_state: dict, simulation_config: dict) -> str:
    """
    Determine customer type based on disclosure decisions and income.
    
    This should be called AFTER Decision 1 (disclose_income) and Decision 2 (disclose_documents).
    
    Customer Type Logic:
    1. DISCOUNT: disclose_income = "Y" AND income < threshold AND disclose_documents = "Y"
       - Qualifies for discount pricing
       - Only available to agents who disclosed income AND have low income AND submitted documents
    
    2. FIXED: disclose_income = "Y" (and not discount)
       - Fixed prices, no bidding option
       - Available to anyone who disclosed their income upfront
       - Includes agents who disclosed income but not documents, or have income above threshold
    
    3. REGULAR: disclose_income = "N"
       - Purchase Now or Bid options
       - Default customer type for non-disclosers
       - These agents are NEVER asked to disclose documents
    
    Args:
        agent_state: Agent's state dict containing:
            - income: Agent's annual income
            - disclose_income: "Y" or "N" (from Decision 1)
            - disclose_documents: "Y", "N", or "NA" (from Decision 2)
        simulation_config: Contains discount_income_threshold from Page 1
        
    Returns:
        str: "discount", "fixed", or "regular"
        
    Example:
        customer_type = determine_customer_type(agent_state, simulation_config)
        # Returns: "discount" if disclosed income + low income + documents submitted
    """
    
    # Get required values
    income = agent_state.get('income', 0)
    disclose_income = agent_state.get('disclose_income', 'N')
    disclose_documents = agent_state.get('disclose_documents', 'NA')
    
    # Get threshold from Page 1
    threshold = get_simulation_param(simulation_config, 'discount_income_threshold', 12500.0)
    
    # Priority 1: Check for DISCOUNT customer
    # Must have disclosed income AND low income AND submitted documents
    if disclose_income == "Y" and income <= threshold and disclose_documents == "Y":
        return "discount"
    
    # Priority 2: Check for FIXED customer
    # Disclosed income upfront (regardless of income level)
    # This includes agents who disclosed income but not documents, or income above threshold
    if disclose_income == "Y":
        return "fixed"
    
    # Priority 3: REGULAR customer (default)
    # Did not disclose income - can use Purchase Now or Bid
    return "regular"


def get_customer_type(agent_state: dict, simulation_config: dict) -> str:
    """
    Get or determine customer type - with caching.
    
    This is the main entry point for getting customer type.
    If already determined, returns cached value.
    Otherwise, determines and stores it.
    
    Args:
        agent_state: Agent's state dict
        simulation_config: Full simulation configuration
        
    Returns:
        str: "discount", "fixed", or "regular"
        
    Example:
        customer_type = get_customer_type(agent_state, simulation_config)
    """
    
    # Check if already determined
    if 'customer_type' in agent_state and agent_state['customer_type'] is not None:
        return agent_state['customer_type']
    
    # Determine customer type
    customer_type = determine_customer_type(agent_state, simulation_config)
    
    # Store for reuse by other decisions
    agent_state['customer_type'] = customer_type
    
    return customer_type
