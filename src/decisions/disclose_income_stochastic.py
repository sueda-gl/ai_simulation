# src/decisions/disclose_income_stochastic.py
"""
Decision 1: Disclose Income - Research Specification Mode

Implements the two-stage mediation model for income disclosure intention.

STANDARDIZATION APPROACH (per documentation):
- Individual traits are z-scored using the ORIGINAL 280 participants' mean/SD (from config)
- Composite variables are z-scored using FIXED statistics from original 280 (from config)
- Statistics are NOT recomputed for each bootstrap sample
- Natural variation in bootstrap samples is preserved as legitimate

This approach follows Stata's `egen z_var = std(var)` which standardizes ONCE on the
original data. When bootstrapping, we use the SAME statistics (not recomputed).

Equation 1: Prosocial Behavior (PB_i) - Mediating Variable
    weighted_prosocial = 0.023776*z_A + 0.016537*z_O + 0.0295482*z_HH + 0.0677157*z_R
    z_weighted_prosocial = (weighted_prosocial - mean_280) / sd_280  # Standardize using fixed stats
    anchored_PB = WOPB * z_obs_PB + (1-WOPB) * z_weighted_prosocial
    z_anchored_PB = (anchored_PB - mean_280) / sd_280  # Using fixed stats

Equation 2: Disclose Income (DI_i) - Dependent Variable
    For CONTINUOUS mode:
        direct_effect = 0.00680238*z_E + 0.0173732*z_N + 0.0163905*z_HH - 0.008988*z_I
    
    For CATEGORICAL mode (level-specific intercepts, NO income coefficient):
        direct_effect = intercept[level] + 0.00680238*z_E + 0.0173732*z_N + 0.0163905*z_HH
    
    z_direct_effect = (direct_effect - mean_280) / sd_280  # Using fixed stats (categorical-specific)
    
    DI_i = β0 + (1-WPB) * z_direct_effect + WPB * z_anchored_PB * income_high

Output: "Y" if final draw > 0, "N" otherwise
"""

import numpy as np
from typing import Dict, Any, Optional, List


# Level-specific intercepts for categorical mode (from regression in documentation)
# These replace the income coefficient in categorical mode
CATEGORICAL_INTERCEPTS = {
    1: 0.0089007,                      # Base intercept for level 1
    2: 0.0089007 - 0.0033655,          # = 0.0055352
    3: 0.0089007 - 0.0065898,          # = 0.0023109
    4: 0.0089007 - 0.0121223,          # = -0.0032216
    5: 0.0089007 - 0.0234331,          # = -0.0145324
}


def _z_score_trait(value: float, trait_name: str, z_params: Dict) -> float:
    """Z-score a raw trait value using population parameters from config."""
    trait_params = z_params.get(trait_name, {})
    mean = trait_params.get('mean', 0)
    sd = trait_params.get('sd', 1)
    if sd == 0:
        return 0.0
    return (value - mean) / sd


def _compute_religiosity_composite_raw(agent_state: Dict, params: Dict) -> float:
    """
    Compute raw religiosity composite (non-standardized):
    1. Scale ReligiousService to 0-1 range
    2. Average with ReligiousAffiliation (equal weights)
    
    Returns:
        Raw composite value (NOT z-scored)
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
    
    # Equal-weight composite (raw, non-standardized)
    return (religious_affiliation + rs_01) / 2


def _compute_religiosity_composite(agent_state: Dict, params: Dict, z_params: Dict) -> float:
    """
    Compute religiosity composite as per documentation:
    1. Scale ReligiousService to 0-1 range
    2. Average with ReligiousAffiliation (equal weights)
    3. Z-score using population parameters
    
    Returns:
        Z-scored composite value
    """
    religious_composite_raw = _compute_religiosity_composite_raw(agent_state, params)
    
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
    # Z-SCORE WEIGHTED_PROSOCIAL using fixed stats from original 280
    # Per documentation: egen z_weighted_prosocial = std(weighted_prosocial)
    # ========================================================================

    composite_z = params.get('composite_z_scoring', {})
    wp_z_params = composite_z.get('weighted_prosocial', {})
    wp_mean = wp_z_params.get('mean', 0)
    wp_sd = wp_z_params.get('sd', 0.08608372)

    if wp_sd > 0:
        z_weighted_prosocial = (weighted_prosocial - wp_mean) / wp_sd
    else:
        z_weighted_prosocial = weighted_prosocial

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
                        CATEGORICAL_INTERCEPTS.get(level, 0.0089007))))
        else:
            intercept = CATEGORICAL_INTERCEPTS.get(level, 0.0089007)
        
        direct_effect = (
            intercept +
            eq2_coeffs.get('extraversion', 0.00680238) * z_extraversion +
            eq2_coeffs.get('neuroticism', 0.0173732) * z_neuroticism +
            eq2_coeffs.get('honesty_humility', 0.0163905) * z_honesty_humility
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
            eq2_coeffs.get('extraversion', 0.00680238) * z_extraversion +
            eq2_coeffs.get('neuroticism', 0.0173732) * z_neuroticism +
            eq2_coeffs.get('honesty_humility', 0.0163905) * z_honesty_humility +
            eq2_coeffs.get('income', -0.008988) * z_income
        )
    
    return {
        'weighted_prosocial': weighted_prosocial,
        'z_weighted_prosocial': z_weighted_prosocial,  # Z-scored version for anchored_pb
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


def compute_anchored_pb(
    pass1_values: Dict[str, float],
    params: Dict[str, Any]
) -> float:
    """
    Compute anchored_pb by combining observed and z-scored weighted prosocial behavior.

    Per documentation (Stata):
        gen weighted_prosocial = (coeffs × z_traits)
        egen z_weighted_prosocial = std(weighted_prosocial)  ← standardize using original 280 stats
        gen anchored_prosocial_behavior = 0.25 * z_obs_PB + 0.75 * z_weighted_prosocial

    Formula:
        anchored_pb = WOPB * z_obs_PB + (1-WOPB) * z_weighted_prosocial

    Args:
        pass1_values: Dict from compute_pass1_values() with z_weighted_prosocial, z_obs_PB
        params: Configuration from decisions.yaml

    Returns:
        anchored_pb: The anchored prosocial behavior value
    """
    z_weighted_prosocial = pass1_values['z_weighted_prosocial']
    z_obs_PB = pass1_values['z_obs_PB']

    # Get anchor weight
    anchor_weights = params.get('anchor_weights', {})
    WOPB = anchor_weights.get('observed_prosocial', 0.25)

    # Compute anchored_pb using z-scored weighted_prosocial
    # Per documentation: anchored_pb = WOPB * z_obs_PB + (1-WOPB) * z_weighted_prosocial
    anchored_pb = WOPB * z_obs_PB + (1 - WOPB) * z_weighted_prosocial

    return anchored_pb


# Keep old function name for backward compatibility during transition
def compute_pass2_anchored_pb(
    pass1_values: Dict[str, float],
    weighted_prosocial_stats: Dict[str, float],
    params: Dict[str, Any]
) -> float:
    """
    DEPRECATED: Use compute_anchored_pb() instead.
    
    This wrapper ignores weighted_prosocial_stats and calls compute_anchored_pb().
    Kept for backward compatibility with orchestrators during transition.
    """
    return compute_anchored_pb(pass1_values, params)


def disclose_income_stochastic(
    agent_state: Dict[str, Any],
    params: Dict[str, Any],
    rng: np.random.Generator,
    simulation_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Decision 1: Disclose income using two-stage mediation model.
    
    STANDARDIZATION APPROACH:
    - Individual traits are z-scored using original 280 participants' stats (from config)
    - Composite variables are NOT re-standardized for each bootstrap sample
    - Natural variation in bootstrap samples is preserved as legitimate
    
    STOCHASTIC COMPONENT (per documentation):
    - Stochastic draw is applied to anchored_pb: Normal(anchored_pb, σ)
    - σ = sd(TWT+Sospeso) × scale_factor (natural variability of observed prosocial behavior)
    - The stochastic anchored_pb is then z-scored and used in the DI equation
    
    Supports multiple population modes:
    - 'documentation' (Research Specification): Uses stochastic Normal(anchored_pb, σ) draw
    - 'baseline' (Research Baseline): Uses anchored_pb directly (no stochastic component)
    
    Args:
        agent_state: Agent's trait values and current state
        params: Configuration from decisions.yaml (coefficients, weights, etc.)
        rng: Random number generator for stochastic component
        simulation_config: Global simulation config (used for income stats in continuous mode)
        **kwargs: Additional arguments including:
            - pop_context: Population mode ('documentation', 'baseline', 'copula')
        
    Returns:
        dict: {"disclose_income": "Y" or "N", "disclose_income_raw": float, "disclose_income_di": float}
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
    # COMPUTE VALUES (single-pass, no re-standardization)
    # ========================================================================
    
    # Check for cached values (for backward compatibility with orchestrators)
    cache = simulation_config.get('disclose_income_cache', {}) if simulation_config else {}
    cache_index = agent_state.get('_cache_index')
    cached_values = cache.get(cache_index) if cache_index is not None else None
    
    if cached_values:
        # Use cached values if available
        weighted_prosocial = cached_values['weighted_prosocial']
        direct_effect = cached_values['direct_effect']
        income_high = cached_values['income_high']
        z_obs_PB = cached_values['z_obs_PB']
        # Compute anchored_pb (may or may not be in cache depending on orchestrator version)
        anchored_pb = cached_values.get('anchored_pb')
        if anchored_pb is None:
            anchored_pb = compute_anchored_pb(cached_values, params)
    else:
        # Compute all values fresh
        pass1_values = compute_pass1_values(agent_state, params, simulation_config)
        weighted_prosocial = pass1_values['weighted_prosocial']
        direct_effect = pass1_values['direct_effect']
        income_high = pass1_values['income_high']
        z_obs_PB = pass1_values['z_obs_PB']
        
        # Compute anchored_pb WITHOUT re-standardization
        anchored_pb = compute_anchored_pb(pass1_values, params)
    
    # Store deterministic anchored_pb for export
    anchored_pb_deterministic = anchored_pb
    
    # ========================================================================
    # STOCHASTIC COMPONENT (applied to anchored_pb per documentation)
    # Per documentation: draw_k ~ Normal(μ = Anchor_k, σ = σ_overall)
    # ========================================================================
    
    stochastic_params = params.get('stochastic', {})
    sigma_value = stochastic_params.get('sigma_value', 0)
    
    # Determine whether to use stochastic component based on population context
    # Three-way logic (same pattern as donation_default):
    # 1. Copula mode with in_copula=True → stochastic ON
    # 2. Documentation mode with sigma > 0 → stochastic ON
    # 3. Otherwise (baseline, or copula with in_copula=False) → stochastic OFF
    use_stochastic = (
        (stochastic_params.get('in_copula', False) and pop_context == 'copula') or
        (pop_context == 'documentation' and sigma_value > 0)
    )
    
    if use_stochastic:
        # Apply stochastic to anchored_pb (NOT to DI_i)
        sigma_strategy = stochastic_params.get('sigma_strategy', 'overall')
        
        # Determine income mode: quintile sigma only applies to categorical mode.
        # Continuous mode uses a single income coefficient, so level-specific
        # sigmas are not meaningful — always fall back to overall sigma.
        income_mode = params.get('income_mode', 'categorical')
        is_continuous = 'continuous' in str(income_mode).lower()
        
        if sigma_strategy == 'quintile' and not is_continuous:
            # Quintile mode (categorical only): Use level-specific base sigma and scale factor
            level = int(agent_state.get('Assigned Allowance Level', 3))
            sigma_quintile = stochastic_params.get('sigma_quintile', {})
            sigma_raw = sigma_quintile.get(str(level), stochastic_params.get('sigma_overall', 9.899547))
            
            # Get quintile-specific scale factor (falls back to overall scale_factor)
            quintile_scale_factors = stochastic_params.get('quintile_scale_factors', {})
            scale_factor = quintile_scale_factors.get(str(level), stochastic_params.get('scale_factor', 0.1))
        else:
            # Overall mode OR continuous income mode: Use single sigma and scale factor for all agents
            sigma_raw = stochastic_params.get('sigma_overall', 9.899547)
            scale_factor = stochastic_params.get('scale_factor', 0.1)
        
        sigma_scaled = sigma_raw * float(scale_factor)
        stochastic_anchored_pb = rng.normal(anchored_pb, sigma_scaled)
    else:
        # Baseline mode OR documentation with sigma disabled: Use anchored_pb directly
        stochastic_anchored_pb = anchored_pb
    
    # ========================================================================
    # Z-SCORE COMPOSITES USING ORIGINAL 280's STATISTICS (not recomputed per bootstrap)
    # ========================================================================
    
    composite_z = params.get('composite_z_scoring', {})
    
    # z_direct_effect: use mode-appropriate stats
    # - Categorical: fixed YAML stats from original 280 participants
    # - Continuous: runtime-computed stats (because income is stochastic)
    income_mode = params.get('income_mode', 'categorical')
    is_continuous = 'continuous' in str(income_mode).lower()
    
    if is_continuous and simulation_config and 'di_cont_de_stats' in simulation_config:
        de_stats = simulation_config['di_cont_de_stats']
    else:
        de_stats = composite_z.get('weighted_disclosure_categorical', {'mean': 0, 'sd': 0.025040462})
    de_mean = de_stats.get('mean', 0)
    de_sd = de_stats.get('sd', 0.025040462)
    if de_sd > 0:
        z_direct_effect = (direct_effect - de_mean) / de_sd
    else:
        z_direct_effect = direct_effect
    
    # z_anchored_pb using fixed stats from original 280 (uses stochastic value)
    ap_stats = composite_z.get('anchored_pb', {'mean': 0, 'sd': 0.7984211971})
    ap_mean = ap_stats.get('mean', 0)
    ap_sd = ap_stats.get('sd', 0.7984211971)
    if ap_sd > 0:
        z_anchored_pb = (stochastic_anchored_pb - ap_mean) / ap_sd
    else:
        z_anchored_pb = stochastic_anchored_pb
    
    # ========================================================================
    # FINAL EQUATION: DI_i
    # ========================================================================
    
    anchor_weights = params.get('anchor_weights', {})
    WPB = anchor_weights.get('prosocial_weight', 0.50)
    beta_0 = params.get('intercept', 0.0)
    
    # DI = β0 + (1-WPB) * z_direct_effect + WPB * z_anchored_pb * income_high
    prosocial_effect = z_anchored_pb * income_high
    DI_i = beta_0 + (1 - WPB) * z_direct_effect + WPB * prosocial_effect
    
    # Stochastic already applied to anchored_pb above, so DI_i is used directly
    draw = DI_i
    
    # ========================================================================
    # FINAL DECISION
    # ========================================================================
    
    disclose_income = "Y" if draw > 0 else "N"
    
    # ========================================================================
    # COMPUTE RAW VALUES FOR EXCEL EXPORT
    # ========================================================================
    
    # Get WOPB for export (observed prosocial weight)
    WOPB = anchor_weights.get('observed_prosocial', 0.25)
    
    # Compute raw religious composite (non-standardized)
    religious_composite_raw = _compute_religiosity_composite_raw(agent_state, params)
    
    # Return both the final decision AND all intermediate values for Excel export
    # The raw value is the draw value BEFORE classification (what we compare to 0)
    return {
        "disclose_income": disclose_income,
        "disclose_income_raw": float(draw),  # Raw value before Y/N classification (same as DI_i now)
        "disclose_income_di": float(DI_i),   # The DI_i value (stochastic already in anchored_pb)
        # NEW: Values for Excel export
        "disclose_income_religious_composite": float(religious_composite_raw),  # Raw religious composite (non-z-scored)
        "disclose_income_income_high": int(income_high),  # I-High indicator
        "disclose_income_wopb": float(WOPB),  # Observed prosocial weight
        "disclose_income_wpb": float(WPB),  # Prosocial behavior weight
        "disclose_income_intercept": float(beta_0),  # β₀ intercept
        "disclose_income_weighted_prosocial": float(weighted_prosocial),  # Trait-based predicted PB (before anchoring)
        "disclose_income_anchored_pb": float(stochastic_anchored_pb),  # PB_i with stochastic (if enabled)
        "disclose_income_anchored_pb_deterministic": float(anchored_pb_deterministic),  # PB_i without stochastic
    }


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================

def compute_disclose_income_raw_values(
    agent_state: Dict[str, Any],
    params: Dict[str, Any],
    simulation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Compute all raw values for disclose_income decision.
    
    Returns pass1_values plus anchored_pb computed without re-standardization.
    """
    pass1_values = compute_pass1_values(agent_state, params, simulation_config)
    
    # Compute anchored_pb (no re-standardization)
    anchored_pb = compute_anchored_pb(pass1_values, params)
    
    return {
        **pass1_values,
        'anchored_pb': anchored_pb,
    }


def compute_continuous_de_stats(agents_df, all_incomes: List[float], di_params: Dict, simulation_config: Dict) -> Dict[str, float]:
    """
    Compute mean and SD of the continuous direct_effect across the population.
    
    Called from orchestrators after income_stats are computed, so that
    continuous mode can z-score direct_effect using population-specific stats
    (matching the Stata approach: egen z_weighted_cont = std(weighted_disclosure_cont)).
    
    Args:
        agents_df: DataFrame of all agents in the population
        all_incomes: List of generated income values (one per agent, same order as agents_df)
        di_params: disclose_income config from decisions.yaml
        simulation_config: Must already contain 'income_stats' (mean, sd)
        
    Returns:
        dict with 'mean' and 'sd' of continuous direct_effect across population
    """
    z_params = di_params.get('z_scoring', {})
    eq2_coeffs = di_params.get('equation2_coefficients', {})
    income_stats = simulation_config.get('income_stats', {})
    income_mean = income_stats.get('mean', 0)
    income_sd = income_stats.get('sd', 1)
    
    all_de = []
    for i, (_, row) in enumerate(agents_df.iterrows()):
        z_e = _z_score_trait(row.get('ExtraversionBig5', 0), 'ExtraversionBig5', z_params)
        z_n = _z_score_trait(row.get('NeuroticismBig5', 0), 'NeuroticismBig5', z_params)
        z_hh = _z_score_trait(row.get('Honesty_Humility', 0), 'Honesty_Humility', z_params)
        
        agent_income = all_incomes[i] if i < len(all_incomes) else 0
        if income_sd > 0:
            z_income = (agent_income - income_mean) / income_sd
        else:
            z_income = 0.0
        
        de = (
            eq2_coeffs.get('extraversion', 0.00680238) * z_e +
            eq2_coeffs.get('neuroticism', 0.0173732) * z_n +
            eq2_coeffs.get('honesty_humility', 0.0163905) * z_hh +
            eq2_coeffs.get('income', -0.008988) * z_income
        )
        all_de.append(de)
    
    all_de = np.array(all_de)
    return {
        'mean': float(np.mean(all_de)),
        'sd': float(np.std(all_de, ddof=1))  # ddof=1 to match Stata's egen std()
    }
