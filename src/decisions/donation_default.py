# src/decisions/donation_default.py
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import income utilities for Category-First architecture
from src.engine.income import get_actual_allowance, get_agent_income
from src.utils.stochastic import should_use_stochastic
from src.decisions._shared import normalise_income_mode, require, file_block

DECISION = 'donation_default'

# ---------------------------------------------------------------------------
# Every constant of the methodology document ("Donation_Rate_Decision_Methodology-2-1
# 250925.docx", sections 3, 5 and 6) lives in config/decisions.yaml and is read
# strictly from ``params`` through ``require``: a missing key raises naming the
# decision and the key; there is no module fallback.
#
# The names below are the FILE's values, read once at import through the one loader
# (src/config.py), for the readers that import them by name
# (tests/test_donation_default_doc.py). The decision never reads them.
# ---------------------------------------------------------------------------
_FILE = file_block(DECISION)
# The doc's final regression tables (section 3), per income mode.
DOC_COEFFICIENTS = {
    mode: dict(require(_FILE, DECISION, 'regression_coefficients', mode))
    for mode in ('categorical', 'continuous')
}
# r(min) / r(max) of `predict predprosocial` over the 280 participants (doc section 5)
DOC_PREDICTED_RANGE = {
    mode: tuple(bounds)
    for mode, bounds in require(_FILE, DECISION, 'scaling', 'predicted_range').items()
}
# Stata `encode studyprogramcategory` levels -> beta_study keys (base: G 2-year Program)
STUDY_CATEGORY_KEYS = dict(require(_FILE, DECISION, 'study_program', 'category_keys'))
# Study Program -> Study Program Category in the 280-participant data (a function: every
# programme has exactly one category). Copula agents carry only `Study Program`.
PROGRAM_TO_CATEGORY = dict(require(_FILE, DECISION, 'study_program', 'program_to_category'))

# simulation_config key the engine fills in Pass 1 with max_j max(draw_j, 0) (0-100 scale)
POPULATION_MAX_KEY = 'donation_population_max'


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def study_category_key(agent_state: dict, params: dict) -> str:
    """
    The agent's beta_study key (Grad2yr / UG3yr / Law5yr / Incoming) from the doc's
    regressor i.studyprogramcategorycat.

    Research agents carry `Study Program Category` and use it directly. Copula agents carry
    only `Study Program`; it is mapped through study_program.program_to_category. An
    unknown programme or category raises (errors raise in every mode; no silent reference category).
    """
    category_keys = require(params, DECISION, 'study_program', 'category_keys')
    category = agent_state.get('Study Program Category')
    if _is_missing(category):
        program = agent_state.get('Study Program')
        program_map = require(params, DECISION, 'study_program', 'program_to_category')
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
    return normalise_income_mode(income_mode)


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
        coeffs_path = f'regression_coefficients.{normalized_mode}'
    elif regression_coeffs:
        coeffs, coeffs_path = regression_coeffs, 'regression_coefficients'   # the app's flat set
    else:
        coeffs, coeffs_path = params.get('regression', {}), 'regression'      # the old block

    def coef(name):
        return require(coeffs, DECISION, name, prefix=coeffs_path)

    # Start with intercept
    predicted = coef('intercept')

    # Add group effect (map HighSub to FullSub for coefficient lookup)
    group_mapped = 'FullSub' if group == 'HighSub' else group
    beta_group = coef('beta_group')
    if group_mapped in beta_group:
        predicted += beta_group[group_mapped]

    # ---------------- Income effect ----------------
    if normalized_mode == 'continuous':
        # CATEGORY-FIRST: Get actual_allowance (12-200 scale) from src.engine.income
        # This ensures consistent mapping across all decisions
        actual_allowance = get_actual_allowance(agent_state, simulation_config, rng)

        beta_lin = coef('beta_income_linear')
        predicted += beta_lin * actual_allowance
    else:
        # Categorical (default): i.totalallowance, level 1 (12 credits) is the reference
        income_quintiles = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5'}
        income_q = income_quintiles.get(int(income_level), 'Q5')
        beta_income_q = coef('beta_income_q')
        if income_q in beta_income_q:
            predicted += beta_income_q[income_q]

    # Study programme effect: doc i.studyprogramcategorycat (reference G 2-year Program)
    beta_study = coef('beta_study')
    study_category = study_category_key(agent_state, params)
    if study_category in beta_study:
        predicted += beta_study[study_category]

    # Honesty-humility effect: the doc's coefficient is on the RAW score, not a z-score
    beta_hh = coef('beta_hh')
    predicted += beta_hh * hh_score

    # Step 2: Standardize both values to 0-100 with the 280-participant min/max (doc section 5)
    obs_min, obs_max = require(params, DECISION, 'scaling', 'observed_range')
    pred_min, pred_max = require(params, DECISION, 'scaling', 'predicted_range', normalized_mode)

    s100_observed = 100 * (observed_prosocial - obs_min) / (obs_max - obs_min)
    s100_predicted = 100 * (predicted - pred_min) / (pred_max - pred_min)

    # Keep within [0, 100]: never binds for the 280 participants (the range is their own
    # min/max); guards synthetic trait combinations outside the sample's range.
    s100_observed = np.clip(s100_observed, 0, 100)
    s100_predicted = np.clip(s100_predicted, 0, 100)

    # Step 3: Compute anchor with specified weights (still in 0-100 scale)
    w_observed = require(params, DECISION, 'anchor_weights', 'observed')
    w_predicted = require(params, DECISION, 'anchor_weights', 'predicted')
    s100_anchor = w_observed * s100_observed + w_predicted * s100_predicted

    # Step 3b: Apply adjustment parameter to shift the distribution
    # (config/decisions.yaml adjustment.shift_value, app default -4.0; the doc has no
    # shift, i.e. 0, which the doc reference test sets explicitly)
    shift_value = require(params, DECISION, 'adjustment', 'shift_value')
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
            # Quintile mode (categorical only): the level's base sigma times its scale factor
            level = int(income_level)
            sigma_quintile = require(params, DECISION, 'stochastic', 'sigma_quintile')
            quintile_scale_factors = require(params, DECISION, 'stochastic', 'quintile_scale_factors')
            sigma_raw = (float(require(sigma_quintile, DECISION, str(level), prefix='stochastic.sigma_quintile'))
                         * float(require(quintile_scale_factors, DECISION, str(level),
                                         prefix='stochastic.quintile_scale_factors')))
        else:
            # Overall mode OR continuous income mode
            # sigma_value is set by the seam from the tab's sigma coefficient;
            # a sigma_value of 0 means sigma_overall * scale_factor
            sigma_raw = require(params, DECISION, 'stochastic', 'sigma_value')
            if sigma_raw == 0:
                sigma_overall = require(params, DECISION, 'stochastic', 'sigma_overall')
                scale_factor = require(params, DECISION, 'stochastic', 'scale_factor')
                sigma_raw = sigma_overall * scale_factor

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
    baseline). Methodology document sections 3, 5 and 6. Steps:
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
