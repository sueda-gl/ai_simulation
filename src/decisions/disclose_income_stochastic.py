# src/decisions/disclose_income_stochastic.py
"""
Decision 1: Disclose Income - Research Specification Mode

Implements the two-stage mediation model for income disclosure intention with
proper population-level standardization as specified in the documentation.

THREE-PASS APPROACH (corrected implementation):
- Pass 1: compute_pass1_values() computes weighted_prosocial and direct_effect
- Pass 2: compute_pass2_anchored_pb() uses weighted_prosocial stats to compute anchored_pb
- Pass 3: disclose_income_stochastic() uses all population stats for proper z-scoring

Equation 1: Prosocial Behavior (PB_i) - Mediating Variable
    weighted_prosocial = 0.023776*A + 0.016537*O + 0.0295482*HH + 0.0677157*R
    z_weighted_prosocial = std(weighted_prosocial)  # Population standardization

Equation 2: Disclosure Intention (DI_i) - Dependent Variable
    For CONTINUOUS mode:
        direct_effect = 0.00674934*E + 0.0173732*N + 0.0295482*HH - 0.008988*I
    
    For CATEGORICAL mode (level-specific intercepts, NO income coefficient):
        direct_effect = intercept[level] + 0.00674934*E + 0.0173732*N + 0.0295482*HH
    
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


def compute_pass1_values(
    agent_state: Dict[str, Any],
    params: Dict[str, Any],
    simulation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Pass 1: Compute weighted_prosocial and direct_effect for population statistics.
    
    This function computes the intermediate values that do NOT depend on any
    population-level statistics (only on pre-defined trait z-scoring parameters).
    These values are collected across all agents to compute population mean/SD.
    
    NOTE: This does NOT compute anchored_pb - that requires z_weighted_prosocial
    which needs population stats from Pass 1.
    
    Args:
        agent_state: Agent's trait values and current state
        params: Configuration from decisions.yaml
        simulation_config: Global simulation config
        
    Returns:
        dict with:
            - weighted_prosocial: Equation 1 output (before population std)
            - direct_effect: Direct effect component (before population std)
            - income_high: Binary indicator (1 if above median)
            - z_obs_PB: Z-scored observed prosocial behavior
            - z_* traits: All z-scored traits for later use
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
    # Z-SCORE OBSERVED PROSOCIAL BEHAVIOR
    # ========================================================================
    
    obs_PB = agent_state.get('TWT+Sospeso [=AW2+AX2]{Periods 1+2}', 0)
    twt_params = z_params.get('TWT_Sospeso', {})
    twt_mean = twt_params.get('mean', 9.139286)
    twt_sd = twt_params.get('sd', 9.899547)
    
    if twt_sd > 0:
        z_obs_PB = (obs_PB - twt_mean) / twt_sd
    else:
        z_obs_PB = 0.0
    
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
            eq2_coeffs.get('honesty_humility', 0.0295482) * z_honesty_humility
            # NO income coefficient in categorical mode!
        )
    else:
        # CONTINUOUS MODE: Use actual income z-scored against population
        # Per documentation: I_i = Income (standardized z-score)
        # egen z_net_income = std(income)
        agent_income = agent_state.get('income', 0)
        income_stats = simulation_config.get('income_stats', {}) if simulation_config else {}
        income_mean = income_stats.get('mean', 0)
        income_sd = income_stats.get('sd', 1)
        
        if income_sd > 0:
            z_income = (agent_income - income_mean) / income_sd
        else:
            z_income = 0.0
        
        direct_effect = (
            eq2_coeffs.get('extraversion', 0.00674934) * z_extraversion +
            eq2_coeffs.get('neuroticism', 0.0173732) * z_neuroticism +
            eq2_coeffs.get('honesty_humility', 0.0295482) * z_honesty_humility +
            eq2_coeffs.get('income', -0.008988) * z_income
        )
    
    return {
        'weighted_prosocial': weighted_prosocial,
        'direct_effect': direct_effect,
        'income_high': income_high,
        'z_obs_PB': z_obs_PB,
        # Also return z-scored traits (avoid recomputation in Pass 3)
        'z_agreeable': z_agreeable,
        'z_openness': z_openness,
        'z_honesty_humility': z_honesty_humility,
        'z_extraversion': z_extraversion,
        'z_neuroticism': z_neuroticism,
        'z_religious': z_religious,
    }


def compute_pass2_anchored_pb(
    pass1_values: Dict[str, float],
    weighted_prosocial_stats: Dict[str, float],
    params: Dict[str, Any]
) -> float:
    """
    Pass 2: Compute anchored_pb using population-standardized weighted_prosocial.
    
    This function takes the Pass 1 values and the population statistics for
    weighted_prosocial, then computes anchored_pb using the correct formula:
    
        z_weighted_prosocial = (weighted_prosocial - μ_wp) / σ_wp
        anchored_pb = WOPB * z_obs_PB + (1-WOPB) * z_weighted_prosocial
    
    Args:
        pass1_values: Dict from compute_pass1_values() with weighted_prosocial, z_obs_PB
        weighted_prosocial_stats: {'mean': float, 'sd': float} from Pass 1 population
        params: Configuration from decisions.yaml
        
    Returns:
        anchored_pb: The correctly computed anchored prosocial behavior value
    """
    weighted_prosocial = pass1_values['weighted_prosocial']
    z_obs_PB = pass1_values['z_obs_PB']
    
    # Get anchor weight
    anchor_weights = params.get('anchor_weights', {})
    WOPB = anchor_weights.get('observed_prosocial', 0.25)
    
    # Z-score weighted_prosocial using population statistics
    wp_mean = weighted_prosocial_stats.get('mean', 0)
    wp_sd = weighted_prosocial_stats.get('sd', 1)
    
    if wp_sd > 0:
        z_weighted_prosocial = (weighted_prosocial - wp_mean) / wp_sd
    else:
        z_weighted_prosocial = weighted_prosocial
    
    # Compute anchored_pb with the correctly z-scored weighted_prosocial
    anchored_pb = WOPB * z_obs_PB + (1 - WOPB) * z_weighted_prosocial
    
    return anchored_pb


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
    
    Supports multiple population modes:
    - 'documentation' (Research Specification): Uses stochastic Normal(DI, σ) draw
    - 'baseline' (Research Baseline): Uses DI value directly (no stochastic component)
    - 'copula': Reserved for future implementation
    
    REQUIRES: simulation_config must contain:
    - 'disclose_income_population_stats' with mean/sd for: weighted_prosocial, anchored_pb, direct_effect
    - 'disclose_income_cache' with cached values keyed by agent index (optional but recommended)
    
    The three-pass approach ensures that:
    - anchored_pb stats are computed from the correct formula using z_weighted_prosocial
    - All z-scored values have proper mean≈0, sd≈1 across the population
    
    Args:
        agent_state: Agent's trait values and current state
        params: Configuration from decisions.yaml (coefficients, weights, etc.)
        rng: Random number generator for stochastic component
        simulation_config: Global simulation config with population stats and cache
        **kwargs: Additional arguments including:
            - pop_context: Population mode ('documentation', 'baseline', 'copula')
        
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
    # POPULATION CONTEXT: Determine if stochastic component should be used
    # ========================================================================
    
    pop_context = kwargs.get('pop_context', 'documentation')
    
    # ========================================================================
    # GET CACHED VALUES OR RECOMPUTE
    # ========================================================================
    
    cache = simulation_config.get('disclose_income_cache', {}) if simulation_config else {}
    cache_index = agent_state.get('_cache_index')
    cached_values = cache.get(cache_index) if cache_index is not None else None
    
    if cached_values:
        # Use cached values from Pass 1 and Pass 2
        weighted_prosocial = cached_values['weighted_prosocial']
        direct_effect = cached_values['direct_effect']
        income_high = cached_values['income_high']
        z_obs_PB = cached_values['z_obs_PB']
        anchored_pb = cached_values['anchored_pb']  # From Pass 2 (correctly computed)
    else:
        # Fallback: recompute values (this path should not be used with proper orchestration)
        pass1_values = compute_pass1_values(agent_state, params, simulation_config)
        weighted_prosocial = pass1_values['weighted_prosocial']
        direct_effect = pass1_values['direct_effect']
        income_high = pass1_values['income_high']
        z_obs_PB = pass1_values['z_obs_PB']
        
        # Get population stats
        pop_stats = simulation_config.get('disclose_income_population_stats', {}) if simulation_config else {}
        wp_stats = pop_stats.get('weighted_prosocial', {'mean': 0, 'sd': 1})
        
        # Compute anchored_pb using Pass 2 function
        anchored_pb = compute_pass2_anchored_pb(pass1_values, wp_stats, params)
    
    # ========================================================================
    # GET POPULATION STATISTICS FROM SIMULATION CONFIG
    # ========================================================================
    
    pop_stats = simulation_config.get('disclose_income_population_stats', {}) if simulation_config else {}
    
    # ========================================================================
    # POPULATION-LEVEL STANDARDIZATION (Pass 3)
    # ========================================================================
    
    # z_anchored_pb = std(anchored_pb) - using correctly computed stats from Pass 2
    ap_stats = pop_stats.get('anchored_pb', {})
    ap_mean = ap_stats.get('mean', 0)
    ap_sd = ap_stats.get('sd', 1)
    
    if ap_sd > 0:
        z_anchored_pb = (anchored_pb - ap_mean) / ap_sd
    else:
        z_anchored_pb = anchored_pb
    
    # z_direct_effect = std(direct_effect)
    de_stats = pop_stats.get('direct_effect', {})
    de_mean = de_stats.get('mean', 0)
    de_sd = de_stats.get('sd', 1)
    
    if de_sd > 0:
        z_direct_effect = (direct_effect - de_mean) / de_sd
    else:
        z_direct_effect = direct_effect
    
    # ========================================================================
    # FINAL EQUATION: DI_i
    # ========================================================================
    
    anchor_weights = params.get('anchor_weights', {})
    WPB = anchor_weights.get('prosocial_weight', 0.50)
    beta_0 = params.get('intercept', 0.1)
    
    # DI = β0 + (1-WPB) * z_direct_effect + WPB * z_anchored_pb * income_high
    prosocial_effect = z_anchored_pb * income_high
    DI_i = beta_0 + (1 - WPB) * z_direct_effect + WPB * prosocial_effect
    
    # ========================================================================
    # STOCHASTIC COMPONENT (conditional on population context)
    # ========================================================================
    
    stochastic_params = params.get('stochastic', {})
    sigma_value = stochastic_params.get('sigma_value', 0)
    
    # Determine whether to use stochastic component based on population context
    use_stochastic = (
        pop_context == 'documentation' and
        sigma_value > 0
    )
    
    if use_stochastic:
        # Documentation mode with stochastic enabled: Apply Normal(DI, σ) draw
        sigma_strategy = stochastic_params.get('sigma_strategy', 'overall')
        
        if sigma_strategy == 'quintile':
            # Quintile mode: Use level-specific base sigma and scale factor
            level = int(agent_state.get('Assigned Allowance Level', 3))
            sigma_quintile = stochastic_params.get('sigma_quintile', {})
            sigma_raw = sigma_quintile.get(str(level), stochastic_params.get('sigma_overall', 9.899547))
            
            # Get quintile-specific scale factor (falls back to overall scale_factor)
            quintile_scale_factors = stochastic_params.get('quintile_scale_factors', {})
            scale_factor = quintile_scale_factors.get(str(level), stochastic_params.get('scale_factor', 0.1))
        else:
            # Overall mode: Use single sigma and scale factor for all agents
            sigma_raw = stochastic_params.get('sigma_overall', 9.899547)
            scale_factor = stochastic_params.get('scale_factor', 0.1)
        
        sigma_scaled = sigma_raw * float(scale_factor)
        
        draw = rng.normal(DI_i, sigma_scaled)
    else:
        # Baseline mode OR documentation with sigma disabled: Use DI_i directly
        draw = DI_i
    
    # ========================================================================
    # FINAL DECISION
    # ========================================================================
    
    disclose_income = "Y" if draw > 0 else "N"
    
    # Return both the final decision AND the raw DI value for analysis
    # The raw value is the draw value BEFORE classification (what we compare to 0)
    return {
        "disclose_income": disclose_income,
        "disclose_income_raw": float(draw),  # Raw value before Y/N classification
        "disclose_income_di": float(DI_i),   # The DI_i value before stochastic draw
    }


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS (deprecated, will be removed)
# =============================================================================

def compute_disclose_income_raw_values(
    agent_state: Dict[str, Any],
    params: Dict[str, Any],
    simulation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    DEPRECATED: Use compute_pass1_values() instead.
    
    This function is kept for backward compatibility during transition.
    It calls compute_pass1_values() and adds a dummy anchored_pb for old code
    that expects it (though this anchored_pb is NOT correctly computed).
    """
    pass1_values = compute_pass1_values(agent_state, params, simulation_config)
    
    # Add dummy anchored_pb for backward compatibility (using raw weighted_prosocial)
    # WARNING: This is the OLD incorrect approach - only for backward compat
    anchor_weights = params.get('anchor_weights', {})
    WOPB = anchor_weights.get('observed_prosocial', 0.25)
    dummy_anchored_pb = WOPB * pass1_values['z_obs_PB'] + (1 - WOPB) * pass1_values['weighted_prosocial']
    
    return {
        **pass1_values,
        'anchored_pb': dummy_anchored_pb,  # Deprecated - do not rely on this
    }
