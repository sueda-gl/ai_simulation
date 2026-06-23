# src/decisions/disclose_documents_stochastic.py
"""
Decision 2: Disclose Documents - Privacy-Calculus model.

Source: "Decision 2 - Disclosure of Documents" design document (Bansal et al. 2016 +
Dinev & Hart 2006). Validated bit-for-bit against the professor's Stata file
`Stata_File_Decision 2_260326 - Final.dta` (disclosedoc_cont 280/280,
disclosedoc_categorical 280/280; all intermediate columns agree to ~1e-7).

This is STRUCTURALLY DIFFERENT from Disclose Income (DI):
- traits used: Extraversion, Neuroticism, Agreeableness ONLY
  (NO Honesty-Humility, NO Openness, NO Religiosity, NO prosocial anchoring)
- income enters as INVERSE income ("Personal Incentive"); single intercept beta0 = -0.5
- the stochastic sigma is sourced from donation CONSUMPTION (receiving), not giving

Reduced-form score (document lines 404/428; coefficients verified vs Stata):
    weighted_dd = bE*z_E + bN*z_N + bA*z_A  [+ bPI*z_picont  for continuous income]
Then standardize the composite and threshold at 0:
    dd_deterministic = beta0 + z_weighted_dd
    disclose if dd_deterministic > 0   (after an optional Normal(dd, sigma) draw)

CATEGORICAL income: drop the income term; instead add a per-allowance-level intercept
(from the professor's regression on inverse allowance).

CONTINUOUS income: picont = max_income - income. Because z(picont) = -z(income) exactly
(the max cancels in standardization), continuous mode reuses the orchestrator's runtime
`income_stats` and a runtime composite-stats hook (compute_continuous_dd_stats), exactly
mirroring how disclose_income handles continuous income.

ELIGIBILITY GATE (platform logic; SEPARATE from the model): returns "NA" unless the agent
disclosed income (disclose_income == "Y") AND has income below the discount threshold.
NOTE: the document's validation tables are the UNGATED model over all 280 participants;
use compute_dd_score() for that. The gate is applied at simulation time only.
"""

import numpy as np
from typing import Dict, Any, Optional, List

from src.decisions.income_utils import (
    get_agent_income, get_simulation_param, get_customer_type, ALLOWANCE_CREDIT_MAPPING,
)

# ---------------------------------------------------------------------------
# Constants verified against the professor's Stata Decision 2 file
# ---------------------------------------------------------------------------
B_EXTRA = 0.015584630336545     # z_E coefficient (document line 404/428)
B_NEURO = -0.022306825775166    # z_N coefficient (published reduced-form)
B_AGREE = -0.016604320441445    # z_A coefficient (published reduced-form)
B_PI = 0.14735467793568         # z_picont (Personal Incentive) coefficient
BETA0 = -0.5                    # final intercept (document line 415/460: betadd0)

# Per-allowance-LEVEL categorical intercepts (added to the trait composite before
# standardization). Verified vs professor: weighted_dd_categorical - trait_terms by level.
#   level 1 (TA 12) +0.1464773 ; level 2 (TA 32) +0.0902694 ; level 3 (TA 72) +0.0384204
#   level 4 (TA 128) -0.0522756 ; level 5 (TA 200) -0.2393718
CATEGORICAL_INTERCEPTS = {
    1: 0.1464773,
    2: 0.0902694,
    3: 0.0384204,
    4: -0.0522756,
    5: -0.2393718,
}

# Composite SDs (egen std over original 280; used to standardize the composite).
# Categorical: matches the professor's .dta exactly and is used directly.
# Continuous: this is only a FALLBACK — the orchestrators always recompute the population SD at
# runtime via compute_continuous_dd_stats and inject it as simulation_config['dd_cont_stats'],
# so this value is overridden before any agent is scored. Set to the professor's .dta value
# (std(weighted_dd_cont) ddof=1 = 0.1509454399) so the fallback also matches the gold file.
SD_WEIGHTED_DD_CONT = 0.1509454399
SD_WEIGHTED_DD_CATEGORICAL = 0.13755775490611996


def _z_score_trait(value: float, trait_name: str, z_params: Dict) -> float:
    """Z-score a raw trait using fixed original-280 mean/SD from config."""
    p = z_params.get(trait_name, {})
    mean = p.get('mean', 0)
    sd = p.get('sd', 1)
    if sd == 0:
        return 0.0
    return (value - mean) / sd


def _get_categorical_intercept(params: Dict, level: int) -> float:
    """Resolve the per-level categorical intercept from config (supports level_N keys)."""
    cfg = params.get('categorical_intercepts')
    if isinstance(cfg, dict):
        # accept 'level_1', '1', or 1 keys; fall back to module default
        return float(
            cfg.get(f'level_{level}',
            cfg.get(str(level),
            cfg.get(level,
            CATEGORICAL_INTERCEPTS.get(level, 0.0))))
        )
    return CATEGORICAL_INTERCEPTS.get(level, 0.0)


def compute_dd_score(
    agent_state: Dict[str, Any],
    params: Dict[str, Any],
    simulation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute the deterministic DD score for an agent (UNGATED, no stochastic draw).

    This is the pure privacy-calculus model used for validation against the professor's
    `dd_deterministic_*` / `disclosedoc_*` columns. The income mode is read from
    params['income_mode'] ('Categorical only' by default).

    Returns a dict of the deterministic score plus all intermediates for export/validation.
    """
    z_params = params.get('z_scoring', {})
    coeffs = params.get('equation_coefficients', {})
    bE = coeffs.get('extraversion', B_EXTRA)
    bN = coeffs.get('neuroticism', B_NEURO)
    bA = coeffs.get('agreeable', B_AGREE)
    bPI = coeffs.get('personal_incentive', B_PI)
    beta0 = params.get('intercept', BETA0)

    z_E = _z_score_trait(agent_state.get('ExtraversionBig5', 0), 'ExtraversionBig5', z_params)
    z_N = _z_score_trait(agent_state.get('NeuroticismBig5', 0), 'NeuroticismBig5', z_params)
    z_A = _z_score_trait(agent_state.get('Agreeable', 0), 'Agreeable', z_params)
    trait_terms = bE * z_E + bN * z_N + bA * z_A

    level = int(agent_state.get('Assigned Allowance Level', 3))
    total_allowance = float(ALLOWANCE_CREDIT_MAPPING.get(level, 200))

    income_mode = params.get('income_mode', 'categorical')
    is_continuous = 'continuous' in str(income_mode).lower()
    composite_z = params.get('composite_z_scoring', {})

    z_picont = 0.0
    if is_continuous:
        # picont = max_income - income  =>  z(picont) = -z(income); max cancels in std.
        income = agent_state.get('income', None)
        income_stats = (simulation_config or {}).get('income_stats', {})
        income_mean = income_stats.get('mean', 0)
        income_sd = income_stats.get('sd', 1)
        if income is None:
            # Fallback only (orchestrator pre-generates income in Pass 1/2).
            income = get_agent_income(agent_state, simulation_config, np.random.default_rng(0))
        z_income = (income - income_mean) / income_sd if income_sd > 0 else 0.0
        z_picont = -z_income
        weighted_dd = trait_terms + bPI * z_picont
        # composite stats: runtime (population) if available, else fixed config / module default
        if simulation_config and 'dd_cont_stats' in simulation_config:
            cstats = simulation_config['dd_cont_stats']
        else:
            cstats = composite_z.get('weighted_dd_cont', {'mean': 0.0, 'sd': SD_WEIGHTED_DD_CONT})
    else:
        intercept = _get_categorical_intercept(params, level)
        weighted_dd = trait_terms + intercept
        cstats = composite_z.get('weighted_dd_categorical', {'mean': 0.0, 'sd': SD_WEIGHTED_DD_CATEGORICAL})

    c_mean = cstats.get('mean', 0.0)
    c_sd = cstats.get('sd', 1.0)
    z_weighted_dd = (weighted_dd - c_mean) / c_sd if c_sd > 0 else weighted_dd
    dd_deterministic = beta0 + z_weighted_dd

    return {
        'dd_deterministic': float(dd_deterministic),
        'weighted_dd': float(weighted_dd),
        'z_weighted_dd': float(z_weighted_dd),
        'trait_terms': float(trait_terms),
        'z_extraversion': float(z_E),
        'z_neuroticism': float(z_N),
        'z_agreeable': float(z_A),
        'z_picont': float(z_picont),
        'intercept': float(beta0),
        'is_continuous': bool(is_continuous),
        'level': level,
        'total_allowance': total_allowance,
    }


def _resolve_default_choice(default_config: Any, rng: np.random.Generator) -> str:
    """Resolve the simple default (random Y/N) for an eligible agent when DD is unconfigured.

    Mirrors the previous placeholder behaviour (and DI's default short-circuit): a
    configurable probability_y, else a 50/50 draw.
    """
    if isinstance(default_config, dict) and default_config.get('type') == 'random_probability':
        prob_y = default_config.get('probability_y', 0.5)
        options = default_config.get('options', ['Y', 'N'])
        return options[0] if rng.random() < prob_y else options[1]
    if isinstance(default_config, str) and default_config in ('Y', 'N'):
        return default_config
    return 'Y' if rng.random() < 0.5 else 'N'


def _apply_stochastic(dd_det: float, params: Dict, pop_context: str,
                      agent_state: Dict, rng: np.random.Generator):
    """Apply the optional Normal(dd_det, sigma) draw. Returns (final_value, sigma_used).

    Three-way rule mirrors disclose_income exactly:
      - copula mode: ON only if in_copula flag set
      - documentation mode: ON if sigma_value > 0
      - baseline mode: always OFF
    """
    stoch = params.get('stochastic', {})
    sigma_value = stoch.get('sigma_value', 0)
    use_stochastic = (
        (stoch.get('in_copula', False) and pop_context == 'copula') or
        (pop_context == 'documentation' and sigma_value > 0)
    )
    if not use_stochastic:
        return dd_det, 0.0

    strategy = stoch.get('sigma_strategy', 'overall')
    if strategy == 'quintile':
        level = int(agent_state.get('Assigned Allowance Level', 3))
        sigma_raw = stoch.get('sigma_quintile', {}).get(str(level), stoch.get('sigma_overall', 0.1606568355))
        scale = stoch.get('quintile_scale_factors', {}).get(str(level), stoch.get('scale_factor', 1.0))
    else:
        sigma_raw = stoch.get('sigma_overall', 0.1606568355)
        scale = stoch.get('scale_factor', 1.0)
    sigma_scaled = float(sigma_raw) * float(scale)
    return float(rng.normal(dd_det, sigma_scaled)), sigma_scaled


def disclose_documents_stochastic(
    agent_state: Dict[str, Any],
    params: Dict[str, Any],
    rng: np.random.Generator,
    simulation_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Decision 2: Disclose Documents (gated, simulation entry point).

    Flow:
      1. ELIGIBILITY GATE -> "NA" unless disclose_income=="Y" AND income < threshold.
      2. DEFAULT short-circuit -> if 'disclose_documents' is unconfigured (in
         default_decisions_list), eligible agents get a simple random Y/N (preserves the
         prior placeholder behaviour).
      3. FULL MODEL -> privacy-calculus score + optional stochastic draw, threshold at 0.
    """
    sim = simulation_config or {}
    pop_context = kwargs.get('pop_context', 'documentation')

    # ---- STEP 1: ELIGIBILITY GATE (only when disclose_income was actually computed) -----
    # In a COMBINED run, disclose_income (Decision 1) runs first and writes agent_state
    # ['disclose_income'], so the platform gate applies: NA unless the agent disclosed income
    # AND has income below the discount threshold.
    # In a STANDALONE disclose_documents run (e.g. the compare-all config-selection workflow),
    # disclose_income has NOT run, so there is no income-disclosure status to gate on. We then
    # compute the UNGATED model for all agents - matching the document's validation tables, which
    # apply the privacy-calculus model to every participant (high-income agents simply score ~0%).
    if 'disclose_income' in agent_state:
        if agent_state.get('disclose_income', 'N') != 'Y':
            agent_state['disclose_documents'] = 'NA'
            return {'disclose_documents': 'NA', 'customer_type': get_customer_type(agent_state, simulation_config)}

        income = get_agent_income(agent_state, simulation_config, rng)
        threshold = get_simulation_param(simulation_config, 'discount_income_threshold', 12500.0)
        if income >= threshold:
            agent_state['disclose_documents'] = 'NA'
            return {'disclose_documents': 'NA', 'customer_type': get_customer_type(agent_state, simulation_config)}

    # ---- STEP 2: DEFAULT short-circuit (DD unconfigured) -------------------
    if 'disclose_documents' in sim.get('default_decisions_list', []):
        default_config = sim.get('default_decisions', {}).get('disclose_documents')
        choice = _resolve_default_choice(default_config, rng)
        agent_state['disclose_documents'] = choice
        return {'disclose_documents': choice, 'customer_type': get_customer_type(agent_state, simulation_config)}

    # ---- STEP 3: FULL PRIVACY-CALCULUS MODEL ------------------------------
    score = compute_dd_score(agent_state, params, simulation_config)
    dd_det = score['dd_deterministic']
    final_value, sigma_used = _apply_stochastic(dd_det, params, pop_context, agent_state, rng)

    choice = 'Y' if final_value > 0 else 'N'
    agent_state['disclose_documents'] = choice
    customer_type = get_customer_type(agent_state, simulation_config)

    return {
        'disclose_documents': choice,
        'customer_type': customer_type,
        # raw values for Excel export / inspection
        'disclose_documents_raw': float(final_value),
        'disclose_documents_score': float(dd_det),
        'disclose_documents_intercept': float(score['intercept']),
        'disclose_documents_trait_terms': float(score['trait_terms']),
        'disclose_documents_z_extraversion': float(score['z_extraversion']),
        'disclose_documents_z_neuroticism': float(score['z_neuroticism']),
        'disclose_documents_z_agreeable': float(score['z_agreeable']),
        'disclose_documents_z_picont': float(score['z_picont']),
        'disclose_documents_weighted_dd': float(score['weighted_dd']),
        'disclose_documents_z_weighted_dd': float(score['z_weighted_dd']),
        'disclose_documents_sigma_used': float(sigma_used),
        'disclose_documents_income_mode': 'continuous' if score['is_continuous'] else 'categorical',
    }


def compute_continuous_dd_stats(
    agents_df,
    all_incomes: List[float],
    dd_params: Dict,
    simulation_config: Dict,
) -> Dict[str, float]:
    """
    Compute mean/SD of the continuous composite weighted_dd_cont across the population.

    Mirrors disclose_income_stochastic.compute_continuous_de_stats. Called from the
    orchestrators (after income_stats are computed) so continuous mode standardizes the
    composite using population-specific stats (matching Stata's `egen z = std(...)`).
    """
    z_params = dd_params.get('z_scoring', {})
    coeffs = dd_params.get('equation_coefficients', {})
    bE = coeffs.get('extraversion', B_EXTRA)
    bN = coeffs.get('neuroticism', B_NEURO)
    bA = coeffs.get('agreeable', B_AGREE)
    bPI = coeffs.get('personal_incentive', B_PI)

    income_stats = simulation_config.get('income_stats', {})
    income_mean = income_stats.get('mean', 0)
    income_sd = income_stats.get('sd', 1)

    vals = []
    for i, (_, row) in enumerate(agents_df.iterrows()):
        z_E = _z_score_trait(row.get('ExtraversionBig5', 0), 'ExtraversionBig5', z_params)
        z_N = _z_score_trait(row.get('NeuroticismBig5', 0), 'NeuroticismBig5', z_params)
        z_A = _z_score_trait(row.get('Agreeable', 0), 'Agreeable', z_params)
        income = all_incomes[i] if i < len(all_incomes) else 0
        z_income = (income - income_mean) / income_sd if income_sd > 0 else 0.0
        z_picont = -z_income
        vals.append(bE * z_E + bN * z_N + bA * z_A + bPI * z_picont)

    vals = np.array(vals)
    return {'mean': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1))}
