# src/decisions/donation_default.py
import numpy as np
from scipy.stats import norm
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import income utilities for Category-First architecture
from src.decisions.income_utils import get_actual_allowance

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
    
    Now enhanced with global parameter context for more realistic behavior.
    """
    
    # Check if this decision is using a simple default value (when unselected)
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'donation_default' in simulation_config.get('default_decisions_list', []):
            # This decision is unselected - use configured default value
            default_config = simulation_config.get('default_decisions', {}).get('donation_default')
            if default_config:
                if isinstance(default_config, dict) and default_config.get('type') == 'numeric':
                    # Wrapped numeric value
                    return {"donation_default": float(default_config.get('value', 0.1))}
                elif isinstance(default_config, (int, float)):
                    # Direct numeric value (backward compatibility)
                    return {"donation_default": float(default_config)}
    
    # Extract required traits
    hh_score = agent_state['Honesty_Humility']
    income_level = agent_state['Assigned Allowance Level']
    study_program = agent_state['Study Program']
    group = agent_state['Group_experiment']
    observed_prosocial = agent_state['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']
    
    # Step 1: Compute predicted prosocial behavior using configurable regression
    # Try to use new structured coefficients, fall back to legacy regression block
    regression_coeffs = params.get('regression_coefficients', {})
    
    # Determine income mode (can be overridden at runtime)
    # Check both new and legacy locations for income_mode parameter
    income_mode = regression_coeffs.get('income_mode', params.get('regression', {}).get('income_mode', 'categorical'))
    
    # Normalize UI format to internal format
    if 'continuous' in str(income_mode).lower():
        normalized_mode = 'continuous'
    else:
        normalized_mode = 'categorical'  # default
    
    # Select appropriate coefficient set based on income mode
    if 'categorical' in regression_coeffs and 'continuous' in regression_coeffs:
        # New structured format - select based on normalized income mode
        if normalized_mode == 'continuous':
            coeffs = regression_coeffs['continuous']
        else:
            coeffs = regression_coeffs['categorical']  # default to categorical
    else:
        # Fall back to legacy format or old regression block
        coeffs = regression_coeffs if regression_coeffs else params.get('regression', {})
    
    # Start with intercept
    predicted = coeffs.get('intercept', 1.22985660120368)
    
    # Add group effect (map HighSub to FullSub for coefficient lookup)
    group_mapped = 'FullSub' if group == 'HighSub' else group
    beta_group = coeffs.get('beta_group', {})
    if group_mapped in beta_group:
        predicted += beta_group[group_mapped]
    
    # ---------------- Income effect ----------------
    if normalized_mode == 'continuous':
        # CATEGORY-FIRST: Get actual_allowance (12-200 scale) from income_utils
        # This ensures consistent mapping across all decisions
        actual_allowance = get_actual_allowance(agent_state, simulation_config, rng)
        
        beta_lin = coeffs.get('beta_income_linear', 0.0256)
        predicted += beta_lin * actual_allowance
    else:
        # Categorical (default)
        # Map income levels to quintiles (level 1/16 is reference, levels 2-5 are 32,72,128,200)
        income_quintiles = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5'}
        income_q = income_quintiles.get(int(income_level), 'Q5')
        beta_income_q = coeffs.get('beta_income_q', {})
        if income_q in beta_income_q:
            predicted += beta_income_q[income_q]
    
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
    
    beta_study = coeffs.get('beta_study', {})
    if study_category in beta_study:
        predicted += beta_study[study_category]
    
    # Add honesty-humility effect
    # Z-score the HH score based on empirical mean/std from analysis
    hh_mean = 3.3922  # from data analysis
    hh_std = 0.5587   # from data analysis
    hh_zscore = (hh_score - hh_mean) / hh_std
    beta_hh = coeffs.get('beta_hh', 0.634001208840808)
    predicted += beta_hh * hh_zscore
    
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
    
    # Step 3b: Apply adjustment parameter to shift the distribution
    adjustment_params = params.get('adjustment', {})
    shift_value = adjustment_params.get('shift_value', 0.0)
    s100_anchor_adjusted = s100_anchor + shift_value
    
    # Note: donation_default currently doesn't use any global parameters
    # The anchor is computed purely from research-based regression and agent traits
    
    # Step 4: Determine if we should use stochastic component
    # Check population context and stochastic flag
    pop_context = kwargs.get('pop_context', 'copula')
    use_stochastic = (
        (params['stochastic'].get('in_copula', False) and pop_context == 'copula') or
        pop_context == 'documentation'
    )
    
    if use_stochastic:
        # Apply stochastic component with Normal(anchor, σ) draw
        # Use the same sigma logic as documentation mode
        sigma_0_100 = params['stochastic']['sigma_value']  # 9.8995 on 0-112 scale
        # Convert to 0-100 scale
        sigma_0_100_scaled = sigma_0_100 * (100.0 / 112.0)
        
        # Step 4a: Draw from Normal(adjusted_anchor, σ)
        draw_0_100 = rng.normal(s100_anchor_adjusted, sigma_0_100_scaled)
        
        # Floor negative values at 0
        draw_0_100 = max(draw_0_100, 0.0)
        
        # Step 6: Simple scaling to proportion (0-1) matching documentation-mode logic
        donation_rate = draw_0_100 / 100.0
        return {"donation_default": np.clip(donation_rate, 0.0, 1.0)}
    else:
        # Use adjusted anchor directly (no additional stochastic component)
        # The copula sampling already provides natural variability
        draw_0_100 = s100_anchor_adjusted
        
        # Step 6: Rescale to [0,1] range using simple scaling
        donation_rate = draw_0_100 / 100.0
        return {"donation_default": np.clip(donation_rate,0.0,1.0)}