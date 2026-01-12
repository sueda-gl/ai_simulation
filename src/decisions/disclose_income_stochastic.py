# src/decisions/disclose_income_stochastic.py
"""
Decision 1: Disclose Income - Research Specification Mode

Implements the two-stage mediation model for income disclosure intention with
proper population-level standardization as specified in the documentation.

TWO-PASS APPROACH:
- Pass 1: compute_disclose_income_raw_values() computes raw values for all agents
- Pass 2: disclose_income_stochastic() uses population stats for proper z-scoring

Equation 1: Prosocial Behavior (PB_i) - Mediating Variable
    weighted_prosocial = 0.023776*A + 0.016537*O + 0.0295482*HH + 0.0677157*R
    z_weighted_prosocial = std(weighted_prosocial)  # Population standardization

Equation 2: Disclosure Intention (DI_i) - Dependent Variable
    For CONTINUOUS mode:
        direct_effect = 0.00674934*E + 0.0173732*N + 0.0163905*HH - 0.008988*I
    
    For CATEGORICAL mode (level-specific intercepts, NO income coefficient):
        direct_effect = intercept[level] + 0.00674934*E + 0.0173732*N + 0.0163905*HH
    
    z_direct_effect = std(direct_effect)  # Population standardization
    
    anchored_PB = WOPB * z_obs_PB + (1-WOPB) * z_weighted_prosocial
    z_anchored_PB = std(anchored_PB)  # Population standardization
    
    DI_i = β0 + (1-WPB) * z_direct_effect + WPB * z_anchored_PB * income_high

Output: "Y" if final draw > 0, "N" otherwise
"""

import numpy as np
from typing import Dict, Any, Optional, List


# Level-specific intercepts for categorical mode (from regression in documentation)
# These replace the income coefficient in categorical mode
CATEGORICAL_INTERCEPTS = {
    1: 0.0089094,                      # Base intercept for level 1
    2: 0.0089094 - 0.0033691,          # = 0.0055403
    3: 0.0089094 - 0.0065954,          # = 0.0023140
    4: 0.0089094 - 0.0121239,          # = -0.0032145
    5: 0.0089094 - 0.0234673,          # = -0.0145579
}


def _z_score_trait(value: float, trait_name: str, z_params: Dict) -> float:
    """Z-score a raw trait value using population parameters from config."""
    trait_params = z_params.get(trait_name, {})
    mean = trait_params.get('mean', 0)
    sd = trait_params.get('sd', 1)
    if sd == 0:
        return 0.0
    return (value - mean) / sd


def _compute_religiosity_composite(agent_state: Dict, params: Dict, z_params: Dict) -> float:
    """
    Compute religiosity composite as per documentation:
    1. Scale ReligiousService to 0-1 range
    2. Average with ReligiousAffiliation (equal weights)
    3. Z-score using population parameters
    """
    religious_affiliation = agent_state.get('ReligiousAffiliation', 0)  # binary 0/1
    religious_service = agent_state.get('ReligiousService', 0)          # ordinal
    
    # Scale ReligiousService to 0-1 range
    religiosity_params = params.get('religiosity', {})
    rs_min = religiosity_params.get('service_min', 0)
    rs_max = religiosity_params.get('service_max', 4)
    
    if rs_max > rs_min:
        rs_01 = (religious_service - rs_min) / (rs_max - rs_min)
    else:
        rs_01 = 0.0
    
    # Equal-weight composite
    religious_composite_raw = (religious_affiliation + rs_01) / 2
    
    # Z-score using population parameters
    return _z_score_trait(religious_composite_raw, 'religious_composite', z_params)


def compute_disclose_income_raw_values(
    agent_state: Dict[str, Any],
    params: Dict[str, Any],
    simulation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Pass 1: Compute raw values for population statistics.
    
    This function computes the intermediate values BEFORE population-level
    standardization. These values are collected across all agents to compute
    population mean/SD, which are then used in Pass 2.
    
    Args:
        agent_state: Agent's trait values and current state
        params: Configuration from decisions.yaml
        simulation_config: Global simulation config
        
    Returns:
        dict with:
            - weighted_prosocial: Raw Equation 1 output (before population std)
            - anchored_pb: Anchored prosocial behavior (before population std)
            - direct_effect: Direct effect component (before population std)
            - income_high: Binary indicator (1 if above median)
    """
    z_params = params.get('z_scoring', {})
    
    # ========================================================================
    # Z-score personality traits using individual trait parameters
    # ========================================================================
    
    z_agreeable = _z_score_trait(agent_state.get('Agreeable', 0), 'Agreeable', z_params)
    z_openness = _z_score_trait(agent_state.get('OpennessBig5', 0), 'OpennessBig5', z_params)
    z_honesty_humility = _z_score_trait(agent_state.get('Honesty_Humility', 0), 'Honesty_Humility', z_params)
    z_extraversion = _z_score_trait(agent_state.get('ExtraversionBig5', 0), 'ExtraversionBig5', z_params)
    z_neuroticism = _z_score_trait(agent_state.get('NeuroticismBig5', 0), 'NeuroticismBig5', z_params)
    z_religious = _compute_religiosity_composite(agent_state, params, z_params)
    
    # ========================================================================
    # EQUATION 1: Compute weighted_prosocial (raw, before population std)
    # ========================================================================
    
    eq1_coeffs = params.get('equation1_coefficients', {})
    
    weighted_prosocial = (
        eq1_coeffs.get('agreeable', 0.023776) * z_agreeable +
        eq1_coeffs.get('openness', 0.016537) * z_openness +
        eq1_coeffs.get('honesty_humility', 0.0295482) * z_honesty_humility +
        eq1_coeffs.get('religious', 0.0677157) * z_religious
    )
    
    # ========================================================================
    # ANCHOR: Compute anchored_pb (raw, before population std)
    # ========================================================================
    
    anchor_weights = params.get('anchor_weights', {})
    WOPB = anchor_weights.get('observed_prosocial', 0.25)
    
    # Get observed prosocial from experiment and z-score it
    obs_PB = agent_state.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', 0)
    twt_params = z_params.get('TWT_Sospeso', {})
    twt_mean = twt_params.get('mean', 9.139286)
    twt_sd = twt_params.get('sd', 9.899547)
    
    if twt_sd > 0:
        z_obs_PB = (obs_PB - twt_mean) / twt_sd
    else:
        z_obs_PB = 0.0
    
    # NOTE: In Pass 1, we use weighted_prosocial directly (not yet population-standardized)
    # The documentation shows: anchored_pb = WOPB * z_obs_PB + (1-WOPB) * z_weighted_prosocial
    # But z_weighted_prosocial requires population stats, so we compute a preliminary anchored_pb
    # This will be properly computed in Pass 2 after we have population stats for weighted_prosocial
    
    # For Pass 1, we compute anchored_pb with the raw weighted_prosocial
    # This gives us the distribution to compute population stats
    anchored_pb = WOPB * z_obs_PB + (1 - WOPB) * weighted_prosocial
    
    # ========================================================================
    # INCOME MODE AND INCOME_HIGH INDICATOR
    # ========================================================================
    
    income_mode = params.get('income_mode', 'categorical')
    
    # Normalize income mode string
    if 'continuous' in str(income_mode).lower():
        normalized_mode = 'continuous'
    else:
        normalized_mode = 'categorical'
    
    if normalized_mode == 'categorical':
        level = agent_state.get('Assigned Allowance Level', 3)
        income_high = 1 if int(level) > 3 else 0
    else:
        agent_income = agent_state.get('income', 0)
        income_median = 0
        if simulation_config:
            income_median = simulation_config.get('income_median', 0)
        income_high = 1 if agent_income > income_median else 0
    
    # ========================================================================
    # DIRECT EFFECT: Compute based on income mode
    # ========================================================================
    
    eq2_coeffs = params.get('equation2_coefficients', {})
    
    if normalized_mode == 'categorical':
        # CATEGORICAL MODE: Use level-specific intercepts, NO income coefficient
        level = int(agent_state.get('Assigned Allowance Level', 3))
        cat_intercepts = params.get('categorical_intercepts', CATEGORICAL_INTERCEPTS)
        
        # Get intercept for this level (use default mapping if not in config)
        if isinstance(cat_intercepts, dict):
            # Config might have string keys like 'level_1'
            intercept = cat_intercepts.get(f'level_{level}', 
                        cat_intercepts.get(str(level),
                        cat_intercepts.get(level, 
                        CATEGORICAL_INTERCEPTS.get(level, 0.0089094))))
        else:
            intercept = CATEGORICAL_INTERCEPTS.get(level, 0.0089094)
        
        direct_effect = (
            intercept +
            eq2_coeffs.get('extraversion', 0.00674934) * z_extraversion +
            eq2_coeffs.get('neuroticism', 0.0173732) * z_neuroticism +
            eq2_coeffs.get('honesty_humility', 0.0163905) * z_honesty_humility
            # NO income coefficient in categorical mode!
        )
    else:
        # CONTINUOUS MODE: Include income coefficient
        level = agent_state.get('Assigned Allowance Level', 3)
        z_income = (level - 3) / 1.41  # Z-score based on 1-5 scale
        
        direct_effect = (
            eq2_coeffs.get('extraversion', 0.00674934) * z_extraversion +
            eq2_coeffs.get('neuroticism', 0.0173732) * z_neuroticism +
            eq2_coeffs.get('honesty_humility', 0.0163905) * z_honesty_humility +
            eq2_coeffs.get('income', -0.008988) * z_income
        )
    
    return {
        'weighted_prosocial': weighted_prosocial,
        'anchored_pb': anchored_pb,
        'direct_effect': direct_effect,
        'income_high': income_high,
        # Also return z-scored traits for Pass 2 (avoid recomputation)
        'z_agreeable': z_agreeable,
        'z_openness': z_openness,
        'z_honesty_humility': z_honesty_humility,
        'z_extraversion': z_extraversion,
        'z_neuroticism': z_neuroticism,
        'z_religious': z_religious,
        'z_obs_PB': z_obs_PB,
    }


def disclose_income_stochastic(
    agent_state: Dict[str, Any],
    params: Dict[str, Any],
    rng: np.random.Generator,
    simulation_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Decision 1: Disclose income using two-stage mediation model with proper
    population-level standardization.
    
    REQUIRES: simulation_config must contain 'disclose_income_population_stats'
    with mean/sd for: weighted_prosocial, anchored_pb, direct_effect
    
    These stats are computed by the orchestrator in Pass 1 using
    compute_disclose_income_raw_values() across all agents.
    
    Args:
        agent_state: Agent's trait values and current state
        params: Configuration from decisions.yaml (coefficients, weights, etc.)
        rng: Random number generator for stochastic component
        simulation_config: Global simulation config with population stats
        
    Returns:
        dict: {"disclose_income": "Y" or "N"}
    """
    
    # Check if this decision is using a simple default value (when unselected)
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'disclose_income' in simulation_config.get('default_decisions_list', []):
            default_config = simulation_config.get('default_decisions', {}).get('disclose_income')
            if default_config:
                if isinstance(default_config, dict):
                    if default_config.get('type') == 'random_probability':
                        prob_y = default_config.get('probability_y', 0.5)
                        choice = "Y" if rng.random() < prob_y else "N"
                        return {"disclose_income": choice}
                    else:
                        return {"disclose_income": str(default_config.get('value', 'NA'))}
                else:
                    return {"disclose_income": str(default_config)}
    
    # ========================================================================
    # GET RAW VALUES (recompute or use cached from Pass 1)
    # ========================================================================
    
    raw_values = compute_disclose_income_raw_values(agent_state, params, simulation_config)
    
    weighted_prosocial = raw_values['weighted_prosocial']
    anchored_pb_raw = raw_values['anchored_pb']
    direct_effect_raw = raw_values['direct_effect']
    income_high = raw_values['income_high']
    z_obs_PB = raw_values['z_obs_PB']
    
    # ========================================================================
    # GET POPULATION STATISTICS FROM SIMULATION CONFIG
    # ========================================================================
    
    pop_stats = simulation_config.get('disclose_income_population_stats', {}) if simulation_config else {}
    
    # ========================================================================
    # POPULATION-LEVEL STANDARDIZATION (as per documentation)
    # ========================================================================
    
    anchor_weights = params.get('anchor_weights', {})
    WOPB = anchor_weights.get('observed_prosocial', 0.25)
    
    # Step 1: z_weighted_prosocial = std(weighted_prosocial)
    wp_stats = pop_stats.get('weighted_prosocial', {})
    wp_mean = wp_stats.get('mean', 0)
    wp_sd = wp_stats.get('sd', 1)
    
    if wp_sd > 0:
        z_weighted_prosocial = (weighted_prosocial - wp_mean) / wp_sd
    else:
        z_weighted_prosocial = weighted_prosocial
    
    # Step 2: Recompute anchored_pb with standardized weighted_prosocial
    # anchored_pb = WOPB * z_obs_PB + (1-WOPB) * z_weighted_prosocial
    anchored_pb = WOPB * z_obs_PB + (1 - WOPB) * z_weighted_prosocial
    
    # Step 3: z_anchored_pb = std(anchored_pb)
    ap_stats = pop_stats.get('anchored_pb', {})
    ap_mean = ap_stats.get('mean', 0)
    ap_sd = ap_stats.get('sd', 1)
    
    if ap_sd > 0:
        z_anchored_pb = (anchored_pb - ap_mean) / ap_sd
    else:
        z_anchored_pb = anchored_pb
    
    # Step 4: z_direct_effect = std(direct_effect)
    de_stats = pop_stats.get('direct_effect', {})
    de_mean = de_stats.get('mean', 0)
    de_sd = de_stats.get('sd', 1)
    
    if de_sd > 0:
        z_direct_effect = (direct_effect_raw - de_mean) / de_sd
    else:
        z_direct_effect = direct_effect_raw
    
    # ========================================================================
    # FINAL EQUATION: DI_i
    # ========================================================================
    
    WPB = anchor_weights.get('prosocial_weight', 0.50)
    beta_0 = params.get('intercept', 0.1)
    
    # DI = β0 + (1-WPB) * z_direct_effect + WPB * z_anchored_pb * income_high
    prosocial_effect = z_anchored_pb * income_high
    DI_i = beta_0 + (1 - WPB) * z_direct_effect + WPB * prosocial_effect
    
    # ========================================================================
    # STOCHASTIC COMPONENT
    # ========================================================================
    
    stochastic_params = params.get('stochastic', {})
    sigma_value = stochastic_params.get('sigma_value', 0)
    
    if sigma_value > 0:
        sigma_strategy = stochastic_params.get('sigma_strategy', 'overall')
        
        if sigma_strategy == 'quintile':
            level = int(agent_state.get('Assigned Allowance Level', 3))
            sigma_quintile = stochastic_params.get('sigma_quintile', {})
            sigma_raw = sigma_quintile.get(str(level), stochastic_params.get('sigma_overall', 9.899547))
        else:
            sigma_raw = stochastic_params.get('sigma_overall', 9.899547)
        
        scale_factor = stochastic_params.get('scale_factor', 0.1)
        sigma_scaled = sigma_raw * scale_factor
        
        draw = rng.normal(DI_i, sigma_scaled)
    else:
        draw = DI_i
    
    # ========================================================================
    # FINAL DECISION
    # ========================================================================
    
    disclose_income = "Y" if draw > 0 else "N"
    
    return {"disclose_income": disclose_income}
