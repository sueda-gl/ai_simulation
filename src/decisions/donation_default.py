# src/decisions/donation_default.py
import numpy as np
from scipy.stats import norm

def donation_default(agent_state: dict, params: dict, rng: np.random.Generator) -> dict:
    """
    Decision 3: Set up default donation rate
    
    Implements 6-step process:
    1. Compute anchor (0.75 * observed + 0.25 * predicted)
    2. Select appropriate sigma
    3. Draw from Normal(anchor, sigma)
    4. Floor negative values at 0
    5. Compute personal 99th percentile maximum
    6. Clip and rescale to [0,1]
    """
    
    # Extract required traits
    hh_score = agent_state['Honesty_Humility']
    income_level = agent_state['Assigned Allowance Level']
    study_program = agent_state['Study Program']
    group = agent_state['Group_experiment']
    observed_prosocial = agent_state['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    
    # Step 1: Compute predicted prosocial behavior using regression
    regression = params['regression']
    
    # Start with intercept
    predicted = regression['intercept']
    
    # Add group effect (reference: FullSub)
    if group in regression['beta_group']:
        predicted += regression['beta_group'][group]
    
    # Add income quintile effect (map allowance level to quintile)
    # Assuming Assigned Allowance Level 1-5 maps to Q1-Q5
    income_quintiles = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4_Q5', 5: 'Q4_Q5'}
    income_q = income_quintiles.get(int(income_level), 'Q4_Q5')
    if income_q in regression['beta_income_q']:
        predicted += regression['beta_income_q'][income_q]
    
    # Add study program effect (reference: Graduate 2-yr)
    # Map study program names to categories
    study_mapping = {
        'Incoming': 'Incoming',
        'Law': 'Law5yr',
        'UG': 'UG3yr'
    }
    study_category = 'Grad2yr'  # default to reference
    for key, category in study_mapping.items():
        if key in study_program:
            study_category = category
            break
    
    if study_category in regression['beta_study']:
        predicted += regression['beta_study'][study_category]
    
    # Add honesty-humility effect (assuming it's already z-scored)
    predicted += regression['beta_hh'] * hh_score
    
    # Step 1 continued: Min-max scale both values to 0-100 and compute anchor
    # For now, assume observed_prosocial and predicted are already on similar scales
    # TODO: Add proper min-max scaling when we have the original data ranges
    
    # Compute anchor with specified weights
    weights = params['anchor_weights']
    anchor = weights['observed'] * observed_prosocial + weights['predicted'] * predicted
    
    # Step 2: Select sigma based on strategy
    sd_params = params['stochastic']
    if sd_params['sigma_strategy'] == 'overall_sd_twt_sospeso':
        sigma = sd_params['sigma_value']  # empirical SD from 280-person data
    else:
        sigma = 5.0  # fallback
    
    # Step 3: Draw from Normal(anchor, sigma)
    draw = rng.normal(anchor, sigma)
    
    # Step 4: Floor negative values at 0
    if draw < 0:
        draw = 0.0
    
    # Step 5: Compute personal 99th percentile maximum
    percentile_max = params['truncation']['percentile_max']
    personal_max = anchor + norm.ppf(percentile_max) * sigma
    
    # Step 6: Clip to personal maximum and rescale to [0,1]
    if personal_max > 0:
        donation_rate = min(draw, personal_max) / personal_max
    else:
        # Handle case where personal_max <= 0
        # Use the raw draw relative to sigma for scaling
        if sigma > 0:
            donation_rate = max(0, draw / (3 * sigma))  # Scale by 3-sigma range
        else:
            donation_rate = 0.0
    
    # Final clipping to ensure [0,1] range
    donation_rate = np.clip(donation_rate, 0.0, 1.0)
    
    return {"donation_default": donation_rate}