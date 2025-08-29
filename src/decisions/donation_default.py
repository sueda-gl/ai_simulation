# src/decisions/donation_default.py
import numpy as np
from scipy.stats import norm

def donation_default(agent_state: dict, params: dict, rng: np.random.Generator, simulation_config: dict = None, **kwargs) -> dict:
    """
    Decision 3: Set up default donation rate
    
    Implements 6-step process from documentation:
    1. Compute predicted prosocial from regression
    2. Scale both observed and predicted to 0-100
    3. Compute anchor (0.75 * observed + 0.25 * predicted)
    4. Draw from Normal(anchor, sigma) where sigma is from observed behavior
    5. Floor negative values at 0
    6. Compute personal 99th percentile maximum and rescale to [0,1]
    
    Now enhanced with income distribution support through the IncomeTransformer.
    """
    
    # Extract required traits
    hh_score = agent_state['Honesty_Humility']
    study_program = agent_state['Study Program']
    group = agent_state['Group_experiment']
    observed_prosocial = agent_state['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    
    # Get income based on transformation (fallback to original if not available)
    if 'income_quintile' in agent_state and 'income_continuous' in agent_state:
        # Use transformed income
        income_quintile = agent_state['income_quintile']
        income_continuous = agent_state['income_continuous']
        income_level_original = agent_state.get('income_level_original', agent_state.get('Assigned Allowance Level', 3))
    else:
        # Fallback to original system
        income_level_original = agent_state.get('Assigned Allowance Level', 3)
        income_quintile = None
        income_continuous = None
    
    # Step 1: Compute predicted prosocial behavior using regression
    regression = params['regression']
    
    # Start with intercept
    predicted = regression['intercept']
    
    # Add group effect (map HighSub to FullSub for coefficient lookup)
    group_mapped = 'FullSub' if group == 'HighSub' else group
    if group_mapped in regression['beta_group']:
        predicted += regression['beta_group'][group_mapped]
    
    # ---------------- Income effect ----------------
    income_mode = regression.get('income_mode', 'categorical')
    if income_mode == 'continuous':
        beta_lin = regression.get('beta_income_linear', 0.0)
        # Use transformed continuous income if available
        if income_continuous is not None:
            predicted += beta_lin * income_continuous
        else:
            # Fallback to original level
            predicted += beta_lin * income_level_original
    else:
        # Categorical (default)
        if income_quintile is not None:
            # Use transformed quintile directly
            if income_quintile in regression['beta_income_q']:
                predicted += regression['beta_income_q'][income_quintile]
        else:
            # Fallback to original mapping
            income_quintiles = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4_Q5', 5: 'Q4_Q5'}
            income_q = income_quintiles.get(int(income_level_original), 'Q4_Q5')
            if income_q in regression['beta_income_q']:
                predicted += regression['beta_income_q'][income_q]
    
    # Add study program effect (reference: Graduate 2-yr)
    # Map study programs to categories based on documentation patterns
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
    
    # For predicted, based on empirical analysis, the regression produces values
    # on a different scale (roughly -4 to 7, suggesting it was fit on transformed data)
    # We need to use the actual range of predictions from the original regression
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
    
    # Step 4: Determine if we should use stochastic component
    # Check population context and stochastic flag
    pop_context = kwargs.get('pop_context', 'copula')
    use_stochastic = (
        (params['stochastic'].get('in_copula', False) and pop_context == 'copula') or
        pop_context == 'documentation'
    )
    
    if use_stochastic:
        out = {}
        raw_flag = params.get('stochastic', {}).get('raw_output', False)
        # Apply stochastic component with Normal(anchor, σ) draw
        # Use the same sigma logic as documentation mode
        sigma_0_100 = params['stochastic']['sigma_value']  # 9.8995 on 0-112 scale
        # Convert to 0-100 scale
        sigma_0_100_scaled = sigma_0_100 * (100.0 / 112.0)
        
        # Step 4a: Draw from Normal(anchor, σ)
        draw_0_100 = rng.normal(s100_anchor, sigma_0_100_scaled)
        
        if raw_flag:
            out = {"donation_default_raw": draw_0_100 / 100.0}
        draw_0_100 = max(draw_0_100, 0.0)
        out["donation_default_raw_pos"] = draw_0_100
        # Step 6: Simple scaling to proportion (0-1) matching documentation-mode logic
        donation_rate = draw_0_100 / 100.0
        out["donation_default"] = np.clip(donation_rate, 0.0, 1.0)
        return out
    else:
        # Use anchor directly (no additional stochastic component)
        # The copula sampling already provides natural variability
        draw_0_100 = s100_anchor
        
        # Step 6: Rescale to [0,1] range using simple scaling
        donation_rate = draw_0_100 / 100.0
        return {"donation_default": np.clip(donation_rate,0.0,1.0)}