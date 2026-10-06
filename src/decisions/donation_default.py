# src/decisions/donation_default.py
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import income utilities for Category-First architecture
from src.decisions.income_utils import get_actual_allowance, get_agent_income
from src.utils.stochastic import get_stochastic_sigma, should_use_stochastic

# ---------------------------------------------------------------------------
# Methodology-document constants (ruling R-D3). config/decisions.yaml is what the engine
# reads; these mirror it and are used only when a params dict lacks a key (direct calls).
# Doc: "Donation_Rate_Decision_Methodology-2-1 250925.docx", sections 3, 5 and 6.
# ---------------------------------------------------------------------------
DOC_COEFFICIENTS = {
    'categorical': {'intercept': 1.519818, 'beta_hh': 0.6042141},
    'continuous': {'intercept': -0.139596, 'beta_hh': 0.7840063, 'beta_income_linear': 0.0255512},
}
# r(min) / r(max) of `predict predprosocial` over the 280 participants (doc section 5)
DOC_PREDICTED_RANGE = {
    'categorical': (-0.5084039942, 8.5422410303),
    'continuous': (-1.2783853173, 8.7549317689),
}
# r(min) / r(max) of twtsospesoaw2ax2periods12 over the 280 participants
DOC_OBSERVED_RANGE = (0.0, 112.0)
# Stata `encode studyprogramcategory` levels -> beta_study keys (base: G 2-year Program)
STUDY_CATEGORY_KEYS = {
    'G 2-year Program': 'Grad2yr',
    'UG 3-year Program': 'UG3yr',
    'Law 5-year Program': 'Law5yr',
    'Incoming': 'Incoming',
}
# Study Program -> Study Program Category in the 280-participant data (a function: every
# programme has exactly one category). Copula agents carry only `Study Program`.
PROGRAM_TO_CATEGORY = {
    'ACME': 'G 2-year Program', 'AFC': 'G 2-year Program', 'BAI': 'UG 3-year Program',
    'BEMACS': 'UG 3-year Program', 'BESS-CLES': 'UG 3-year Program', 'BIEF': 'UG 3-year Program',
    'BIEM': 'UG 3-year Program', 'BIG': 'UG 3-year Program', 'CLEACC': 'UG 3-year Program',
    'CLEAM': 'UG 3-year Program', 'CLEF': 'UG 3-year Program', 'CLELI': 'G 2-year Program',
    'CLMG': 'Law 5-year Program', 'CYBER': 'G 2-year Program', 'DES-ESS': 'G 2-year Program',
    'DSBA': 'G 2-year Program', 'EMIT': 'G 2-year Program', 'FIN': 'G 2-year Program',
    'GIO': 'G 2-year Program', 'IM': 'G 2-year Program', 'M': 'G 2-year Program',
    'MM': 'G 2-year Program', 'PPA': 'G 2-year Program', 'RI': 'Incoming',
}

# simulation_config key the engine fills in Pass 1 with max_j max(draw_j, 0) (0-100 scale)
POPULATION_MAX_KEY = 'donation_population_max'


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def study_category_key(agent_state: dict, params: dict) -> str:
    """
    The agent's beta_study key (Grad2yr / UG3yr / Law5yr / Incoming) from the doc's
    regressor i.studyprogramcategorycat (ruling R-D3).

    Research agents carry `Study Program Category` and use it directly. Copula agents carry
    only `Study Program`; it is mapped through study_program.program_to_category. An
    unknown programme or category raises (R4: errors raise, no silent reference category).
    """
    sp_cfg = params.get('study_program') or {}
    category_keys = sp_cfg.get('category_keys') or STUDY_CATEGORY_KEYS
    category = agent_state.get('Study Program Category')
    if _is_missing(category):
        program = agent_state.get('Study Program')
        program_map = sp_cfg.get('program_to_category') or PROGRAM_TO_CATEGORY
        if program not in program_map:
            raise ValueError(f"donation_default: unknown Study Program {program!r} "
                             f"(no Study Program Category to map it to)")
        category = program_map[program]
    if category not in category_keys:
        raise ValueError(f"donation_default: unknown Study Program Category {category!r}")
    return category_keys[category]


def _income_mode(params: dict) -> str:
    """'continuous' or 'categorical' (UI spellings normalised)."""
    regression_coeffs = params.get('regression_coefficients', {})
    income_mode = regression_coeffs.get('income_mode', params.get('regression', {}).get('income_mode', 'categorical'))
    return 'continuous' if 'continuous' in str(income_mode).lower() else 'categorical'


def donation_draw(agent_state: dict, params: dict, rng: np.random.Generator,
                  simulation_config: dict = None, pop_context: str = 'copula') -> float:
    """
    Steps 1-5 for one agent: max(draw_k, 0) on the 0-100 scale - the numerator of the
    doc's section 6 step 4. Deterministic when no noise is drawn; otherwise consumes
    exactly one rng.normal. The engine calls it once per agent in Pass 1 (to get the
    population maximum) and again, with an identical fresh RNG, inside donation_default.
    """
    # Extract required traits
    hh_score = agent_state['Honesty_Humility']
    income_level = agent_state['Assigned Allowance Level']
    group = agent_state['Group_experiment']
    observed_prosocial = agent_state['TWT+Sospeso [=AW2+AX2]{Periods 1+2}']

    # Step 1: Compute predicted prosocial behavior using configurable regression
    # Try to use new structured coefficients, fall back to legacy regression block
    regression_coeffs = params.get('regression_coefficients', {})
    normalized_mode = _income_mode(params)

    # Select appropriate coefficient set based on income mode
    if 'categorical' in regression_coeffs and 'continuous' in regression_coeffs:
        coeffs = regression_coeffs[normalized_mode]
    else:
        # Flat set (what the app's patch carries) or the old regression block
        coeffs = regression_coeffs if regression_coeffs else params.get('regression', {})
    doc = DOC_COEFFICIENTS[normalized_mode]

    # Start with intercept
    predicted = coeffs.get('intercept', doc['intercept'])

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

        beta_lin = coeffs.get('beta_income_linear', DOC_COEFFICIENTS['continuous']['beta_income_linear'])
        predicted += beta_lin * actual_allowance
    else:
        # Categorical (default): i.totalallowance, level 1 (12 credits) is the reference
        income_quintiles = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5'}
        income_q = income_quintiles.get(int(income_level), 'Q5')
        beta_income_q = coeffs.get('beta_income_q', {})
        if income_q in beta_income_q:
            predicted += beta_income_q[income_q]

    # Study programme effect: doc i.studyprogramcategorycat (reference G 2-year Program)
    beta_study = coeffs.get('beta_study', {})
    study_category = study_category_key(agent_state, params)
    if study_category in beta_study:
        predicted += beta_study[study_category]

    # Honesty-humility effect: the doc's coefficient is on the RAW score (ruling R-D3)
    beta_hh = coeffs.get('beta_hh', doc['beta_hh'])
    predicted += beta_hh * hh_score

    # Step 2: Standardize both values to 0-100 with the 280-participant min/max (doc section 5)
    scaling = params.get('scaling') or {}
    obs_min, obs_max = scaling.get('observed_range', DOC_OBSERVED_RANGE)
    pred_min, pred_max = (scaling.get('predicted_range') or {}).get(
        normalized_mode, DOC_PREDICTED_RANGE[normalized_mode])

    s100_observed = 100 * (observed_prosocial - obs_min) / (obs_max - obs_min)
    s100_predicted = 100 * (predicted - pred_min) / (pred_max - pred_min)

    # Keep within [0, 100]: never binds for the 280 participants (the range is their own
    # min/max); guards synthetic trait combinations outside the sample's range.
    s100_observed = np.clip(s100_observed, 0, 100)
    s100_predicted = np.clip(s100_predicted, 0, 100)

    # Step 3: Compute anchor with specified weights (still in 0-100 scale)
    weights = params['anchor_weights']
    s100_anchor = weights['observed'] * s100_observed + weights['predicted'] * s100_predicted

    # Step 3b: Apply adjustment parameter to shift the distribution
    # (app default -4.0 in adjustment.shift_value; the doc has no shift, i.e. 0)
    adjustment_params = params.get('adjustment', {})
    shift_value = adjustment_params.get('shift_value', 0.0)
    s100_anchor_adjusted = s100_anchor + shift_value

    # Step 4: Determine if we should use stochastic component.
    # ONE gate for every mode (src.utils.stochastic.should_use_stochastic):
    #   copula        -> stochastic.in_copula
    #   documentation -> stochastic.sigma_value > 0
    #   baseline      -> never
    # and additionally the resolved sigma must be > 0 (a zero sigma adds no noise).
    stochastic_params = params.get('stochastic', {})
    use_stochastic = should_use_stochastic(stochastic_params, pop_context)

    sigma_0_100_scaled = 0.0
    if use_stochastic:
        # Resolve sigma: quintile (categorical only) or sigma_value with sigma_overall fallback
        sigma_strategy = stochastic_params.get('sigma_strategy', 'overall')

        # Quintile sigma only applies to categorical mode.
        # Continuous mode uses a single income coefficient, so level-specific
        # sigmas are not meaningful — always fall back to overall sigma.
        if sigma_strategy == 'quintile' and normalized_mode != 'continuous':
            # Quintile mode (categorical only): use centralized utility for level-specific sigma
            level = int(income_level)
            sigma_raw = get_stochastic_sigma(
                level=level,
                stochastic_params=stochastic_params,
                convert_to_z_scale=False,
            )
        else:
            # Overall mode OR continuous income mode
            # sigma_value should be set by simulation.py from UI's sigma_value_ui
            # If sigma_value is 0 or not set, fall back to sigma_overall or default
            sigma_raw = stochastic_params.get('sigma_value', 0)
            if sigma_raw == 0:
                # Fallback: use sigma_overall * scale_factor (like the centralized utility)
                sigma_overall = stochastic_params.get('sigma_overall', 9.899547)
                scale_factor = stochastic_params.get('scale_factor', 1.0)
                sigma_raw = sigma_overall * scale_factor

        # DEBUG: Print sigma values to diagnose distribution differences
        print(f"[DonationDefault DEBUG] strategy={sigma_strategy}, sigma_raw={sigma_raw}, "
              f"scale_factor={stochastic_params.get('scale_factor')}, "
              f"sigma_value={stochastic_params.get('sigma_value')}, "
              f"sigma_overall={stochastic_params.get('sigma_overall')}")

        # Convert sigma from the 0-112 credit scale to the 0-100 anchor scale
        sigma_0_100_scaled = sigma_raw * (100.0 / 112.0)

    if use_stochastic and sigma_0_100_scaled > 0:
        # Step 4a: Draw from Normal(adjusted_anchor, sigma)
        draw_0_100 = rng.normal(s100_anchor_adjusted, sigma_0_100_scaled)
    else:
        # No noise: the draw is the adjusted anchor
        draw_0_100 = s100_anchor_adjusted

    # Step 5: Floor negative values at 0 (doc section 6 step 4: max(draw_k, 0))
    return float(max(draw_0_100, 0.0))


def configured_default_value(simulation_config: dict = None):
    """
    The configured default rate when Decision 3 is unselected (in default_decisions_list
    with a numeric default), else None - in which case the model runs.
    """
    if simulation_config and 'default_decisions_list' in simulation_config:
        if 'donation_default' in simulation_config.get('default_decisions_list', []):
            # This decision is unselected - use configured default value
            default_config = simulation_config.get('default_decisions', {}).get('donation_default')
            if default_config:
                if isinstance(default_config, dict) and default_config.get('type') == 'numeric':
                    # Wrapped numeric value
                    return float(default_config.get('value', 0.1))
                elif isinstance(default_config, (int, float)):
                    # Direct numeric value (backward compatibility)
                    return float(default_config)
    return None


def donation_default(agent_state: dict, params: dict, rng: np.random.Generator, simulation_config: dict = None, **kwargs) -> dict:
    """
    Decision 3: Set up default donation rate.

    The ONE donation module for every population mode (copula, documentation,
    baseline). Methodology document sections 3, 5 and 6 (ruling R-D3). Steps:
    1. Predicted prosocial from the regression (RAW honesty-humility score; study
       programme from Study Program Category)
    2. Scale observed and predicted to 0-100 with the 280-participant min/max
       (config scaling.observed_range / scaling.predicted_range[income mode])
    3. Anchor = weights['observed'] * observed + weights['predicted'] * predicted,
       plus adjustment.shift_value
    4. Noise: draw from Normal(anchor, sigma * 100/112) iff the mode's stochastic
       tick is on (src.utils.stochastic.should_use_stochastic: copula -> in_copula,
       documentation -> sigma_value > 0, baseline -> never) AND the resolved sigma > 0
    5. Floor negative values at 0
    6. Divide by the population maximum of the floored draws (doc section 6 step 4:
       score_k = max(draw_k, 0) / max_j max(draw_j, 0)). The engine supplies
       simulation_config['donation_population_max'] from its Pass-1 hook; a direct call
       without it (no population) divides by 100 and clips to [0, 1].

    Output: {"donation_default": rate} in every mode.
    """

    # Check if this decision is using a simple default value (when unselected)
    default_value = configured_default_value(simulation_config)
    if default_value is not None:
        return {"donation_default": default_value}

    pop_context = kwargs.get('pop_context', 'copula')
    draw_0_100 = donation_draw(agent_state, params, rng, simulation_config, pop_context)

    # Step 6: population rescale (doc section 6 step 4)
    population_max = (simulation_config or {}).get(POPULATION_MAX_KEY)
    if population_max is None:
        donation_rate = float(np.clip(draw_0_100 / 100.0, 0.0, 1.0))
    elif population_max > 0:
        donation_rate = draw_0_100 / population_max
    else:
        donation_rate = 0.0  # every floored draw is 0
    return {"donation_default": donation_rate}


def compute_donation_population_max(agents_df, params: dict, simulation_config: dict,
                                    pop_context: str, agent_base_seeds, decision_offset: int) -> float:
    """
    max_j max(draw_j, 0) over the whole population (0-100 scale) - the denominator of the
    doc's section 6 step 4. Replays every agent's Pass-2 donation draw exactly: the same
    agent state (row + income from default_rng(base + 999999), as Pass 2 builds it) and a
    fresh default_rng(base + decision_offset), the stream Pass 2 hands donation_default.
    """
    draws = []
    for (_, row), base in zip(agents_df.iterrows(), agent_base_seeds):
        state = row.to_dict()
        get_agent_income(state, simulation_config, np.random.default_rng(int(base) + 999999))
        rng = np.random.default_rng(int(base) + decision_offset)
        draws.append(donation_draw(state, params, rng, simulation_config, pop_context))
    return float(max(draws)) if draws else 0.0
