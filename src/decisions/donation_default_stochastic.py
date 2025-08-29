# src/decisions/donation_default_stochastic.py
import numpy as np
from scipy.stats import norm

def donation_default_stochastic(agent_state: dict, params: dict, rng: np.random.Generator, simulation_config: dict = None, **kwargs) -> dict:
    """
    Decision 3: Set up default donation rate (Documentation mode with stochastic component)
    
    Implements the full 6-step process from documentation:
    1. Compute predicted prosocial from regression
    2. Scale both observed and predicted to 0-100
    3. Compute anchor (0.75 * observed + 0.25 * predicted)
    4. Add stochastic component: Draw from Normal(anchor, sigma)
    5. Floor negative values at 0
    6. Compute personal 99th percentile maximum and rescale to [0,1]
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
    
    # Add group effect (map HighSub to FullSub for coefficient lookup)
    group_mapped = 'FullSub' if group == 'HighSub' else group
    if group_mapped in regression['beta_group']:
        predicted += regression['beta_group'][group_mapped]
    
    # Add income effect
    income_mode = regression.get('income_mode', 'categorical')
    if income_mode == 'continuous':
        # Linear income effect
        beta_lin = regression.get('beta_income_linear', 0.0)
        predicted += beta_lin * income_level
    else:
        # Categorical income effect
        income_quintiles = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4_Q5', 5: 'Q4_Q5'}
        income_q = income_quintiles.get(int(income_level), 'Q4_Q5')
        if income_q in regression['beta_income_q']:
            predicted += regression['beta_income_q'][income_q]
    
    # Add study program effect
    study_category = 'Grad2yr'  # default to reference
    
    # More comprehensive mapping based on typical program names
    if any(prog in study_program.upper() for prog in ['INCOMING', 'EXCHANGE']):
        study_category = 'Incoming'
    elif any(prog in study_program.upper() for prog in ['LAW', 'CLMG']):
        study_category = 'Law5yr'
    elif any(prog in study_program.upper() for prog in ['BESS', 'BIEM', 'BIG', 'BAI', 'BEMACS']):
        study_category = 'UG3yr'
    # Graduate programs (CLEF, CLEAM, BIEF, etc.) remain as reference
    
    if study_category in regression['beta_study']:
        predicted += regression['beta_study'][study_category]
    
    # Add honesty-humility effect
    # Z-score the HH score based on empirical mean/std from analysis
    hh_mean = 3.3922  # from data analysis
    hh_std = 0.5587   # from data analysis
    hh_zscore = (hh_score - hh_mean) / hh_std
    predicted += regression['beta_hh'] * hh_zscore
    
    # Step 2: Standardize both values to 0-100 scale
    # Use empirical min/max from the original 280 participants
    obs_min = 0.0      # from data analysis
    obs_max = 112.0    # from data analysis
    
    # For predicted, use the empirical range from regression analysis
    pred_min = -4.0778  # empirical min from regression analysis
    pred_max = 7.2030   # empirical max from regression analysis
    
    # Scale to 0-100
    s100_observed = 100 * (observed_prosocial - obs_min) / (obs_max - obs_min)
    s100_predicted = 100 * (predicted - pred_min) / (pred_max - pred_min)
    
    # Ensure scaled values are in [0, 100]
    s100_observed = np.clip(s100_observed, 0, 100)
    s100_predicted = np.clip(s100_predicted, 0, 100)
    
    # Step 3: Compute anchor with specified weights (still in 0-100 scale)
    weights = params['anchor_weights']
    s100_anchor = weights['observed'] * s100_observed + weights['predicted'] * s100_predicted
    
    # Step 4: Add stochastic component
    # Use the overall SD from observed behavior, scaled to 0-100 range
    sd_params = params['stochastic']
    if sd_params['sigma_strategy'] == 'overall_sd_twt_sospeso':
        # The sigma value should be the SD of the 0-100 scaled observed behavior
        # Original SD = 9.8995 on 0-112 scale
        # Scaled SD = 9.8995 * 100 / 112 ≈ 8.84
        sigma_0_100 = sd_params['sigma_value'] * 100 / (obs_max - obs_min)
    else:
        sigma_0_100 = 8.84  # fallback based on calculation above
    
    # Draw from Normal(anchor, sigma) (0-100 scale)
    draw_raw = rng.normal(s100_anchor, sigma_0_100)

    # Store raw draw (can be negative)
    raw_flag = params.get('stochastic', {}).get('raw_output', False)
    out = {}
    if raw_flag:
        out["donation_default_raw"] = draw_raw / 100.0  # keep original scale as proportion

    # Always keep non-negative version for later scaling
    draw_pos = max(draw_raw, 0.0)
    out["donation_default_raw_pos"] = draw_pos  # still 0-100 scale, no scaling yet

    # Final donation rate: simply scale the (non-negative) draw from 0-100 to 0-1.
    # This replicates the Stata implementation and avoids the personal 99th-percentile rescale.
    donation_rate = np.clip(draw_pos / 100.0, 0.0, 1.0)
    out["donation_default"] = donation_rate

    return out
