# src/decisions/rejected_transaction_defaults.py
"""
Decision 4: Rejected Transaction Defaults - five trait-based sub-decisions plus the
Section-6 rank aggregation that integrates them into ONE default list per agent.

Sources: the "Decision 4 - Rejected Transaction Defaults" design document, rev
040926-2 (its Stata code is the operative specification), and the reference Stata
file `Stata_File_Decision4_050926.dta` (280 participants; frozen extracts
data/stata_d4_verification.csv, data/stata_d4_flexibility_verification.csv). With
every intercept at 0 the model reproduces that file 280/280 for every mechanism
(tests/test_rejected_transaction_defaults.py). Coefficients: config/decisions.yaml,
read strictly from ``params`` (COEFFICIENT_KEYS below lists the keys each equation uses).

Sub-decisions (doc section in brackets; priority sequences in PRIORITY_SEQUENCES):
  1. LIST LENGTH (Tendency to Plan) [1]: weighted_ttp over z-scored E/A/N/C plus
     reducation (= Education - 1), min-max rescaled to 0-6 and floored -> 0-5
     options (rtd_weighted_ttp, rtd_choice_length).
  2-5. LOYALTY [2], WTP [3], RISK-TAKING [4] and FLEXIBILITY [5] rankings (WTP and
     RT use z_income, SAMPLE SD like Stata's `egen std()`; the flexibility score is
     anchored 0.25 * z_stdactions + 0.75 * z_Flexibility, where stdactions - the
     participant's SD in actions per cycle over the eight experiment cycles - comes
     from the reference .dta via data/stata_stdactions.csv and is a copula trait; an
     agent without it gets a neutral anchor, rtd_flex_stdactions_missing). Each:
     standardize (population `egen std`) -> min-max rescale -> floor into five
     EQUAL-WIDTH segments 1..5 -> segment s receives the LAST s options of the
     priority sequence, seq[5 - s:] (segment 5 = the full list, segment 1 = the last
     option only; what the .dta's choice1..5 columns store) -> rtd_<loyalty|wtp|rt|flex>_ranking.

CATEGORICAL INCOME (params['income_mode']): the income term of WTP / RT is replaced
by the doc's per-allowance-level effects (params['categorical_income_effects']), the
personality part applied at slope 1 as the doc's cond() code does; z_agreeable is
used for every level (the .dta's level-1 WTP branch still uses the raw score, so WTP
categorical matches 279/280). The continuous sigma constants are reused.

INTERCEPTS (params['intercepts']), FIXED-CUTOFF semantics: population statistics and
segment cutoffs come from the intercept-free scores; each agent's operative score
then adds the intercept's raw-scale equivalent, so a nonzero beta visibly moves
agents across the fixed boundaries. TTP beta0 is applied on the RAW weighted_ttp
composite, the scale the document gives it (TTP_i = beta0 + ..., never
standardized); beta1-beta3 on the standardized score (raw shift beta * sd0); beta4
on the standardized calculated flexibility score before anchoring (shift
0.75 * beta4). beta = 0 is bit-identical to the intercept-free pipeline.

STOCHASTIC LAYER: Normal(anchor, sigma) per mechanism, anchored on the continuous
operative score (raw composite for ttp/wtp, population z for the others; the
doc-literal binned anchor via stochastic.anchor = 'binned'), re-binned on the
population min/max of the DRAWN values. sigma = (score range / 18) * mean(stdactions
within allowance group), the document's formula (config/decisions.yaml
stochastic.mechanisms.<m>, which notes the two slips in the document's printed sigma
tables; SIGMA_FACTORS / MEAN_STDACTIONS below document the derivation). Population
statistics, the draws' min/max included, are computed after Pass 1 by
compute_rtd_population_stats (src/engine/core.py), which replays each agent's
decision RNG stream bit-for-bit.

RANK AGGREGATION (doc Section 6; params['aggregation']): the four rankings' COMPLETE
orderings (the priority sequence rotated to the segment's first choice) are
integrated by Kemeny-Young (120 permutations, equal weights); ties go to Schulze,
then Copeland, then Spearman footrule, then a random order from the agent's decision
RNG after the five normals. Only the integrated list is then truncated to
rtd_choice_length and cut after Option 5 (forgo). Output: `rejected_transaction_defaults`
(option codes), rtd_default_list and the rtd_consensus_* diagnostics (renamed
rtd_integrated_* in every export). The observed-flexibility z-score is emitted as
`rtd_z_stdactions` (the .dta's `z_stdactionsP`).
"""

import numpy as np
from typing import Any, Dict, List, Optional

from src.decisions.rtd_rank_aggregation import integrate_default_list
from src.decisions._shared import z_score_trait, normalise_income_mode, require, file_block
from src.utils.stochastic import should_use_stochastic

DECISION = "rejected_transaction_defaults"

# ---------------------------------------------------------------------------
# Option codes (canonical across the app - see DEFAULT_DECISION_VALUES) keyed by
# the document's option numbers 1-5.
# ---------------------------------------------------------------------------
OPTION_CODES = {
    1: "higher_price_category",   # Option 1: higher price category, same vendor
    2: "lower_pn_vendor",         # Option 2: other vendor at lower PN price
    3: "current_vendor_pn",       # Option 3: current vendor at PN price
    4: "place_bid",               # Option 4: place a bid
    5: "forgo_transaction",       # Option 5: forgo the transaction
}

# ---------------------------------------------------------------------------
# Every constant of the model lives in config/decisions.yaml and is read strictly
# from ``params`` through ``require`` (a missing key raises naming the decision and
# the key; there is no module fallback):
#   coefficients.<mechanism>.*        the five equations (doc-derived, .dta-verified
#                                     to <=1e-7 on all 280 rows); the keys each
#                                     equation uses are COEFFICIENT_KEYS below
#   flexibility_anchor.*              observed_weight / calculated_weight of
#                                     anchored_flexibility (doc: 25% observed)
#   categorical_income_effects.*      intercept + level_2..5 dummies for WTP / RT
#                                     (level_1, EUR 12, is the base level)
#   z_scoring.<trait>.{mean,sd}       frozen original-280 statistics (egen std)
#   intercepts.<mechanism>            beta0..beta4
#   stochastic.mechanisms.<m>.*       the sigma tables
#   aggregation.*                     the Section-6 rank aggregation settings
# ---------------------------------------------------------------------------
COEFFICIENT_KEYS = {
    "ttp": ("extraversion", "agreeable", "neuroticism", "conscientiousness",
            "education"),             # education applies to reducation = Education - 1 (0/1)
    "loyalty": ("extraversion", "openness", "agreeable"),
    "wtp": ("extraversion", "agreeable", "income"),
    "risk_taking": ("extraversion", "openness", "agreeable", "conscientiousness",
                    "neuroticism", "income"),
    "flexibility": ("extraversion", "openness", "neuroticism", "agreeable",
                    "conscientiousness"),
}

# ---------------------------------------------------------------------------
# Stochastic sigma constants. sigma = (score range / 18) * mean(stdactions),
# where stdactions is the per-participant SD in actions per cycle in the original
# experiment (overall mean 1.1863376; per-allowance-group means below, doc-verified).
# ---------------------------------------------------------------------------
MEAN_STDACTIONS = {          # by Assigned Allowance Level (1..5 = 12/32/72/128/200 EUR)
    1: 1.0228235,
    2: 1.1366981,
    3: 1.3106473,
    4: 1.2256943,
    5: 1.2544901,
    "overall": 1.1863376,
}
# Per-mechanism quintile multiplier applied to mean(stdactions within group); every
# value below is the range/18 figure printed by doc rev 040926-2 and each range is
# confirmed by Stata_File_Decision4_050926.dta:
#   ttp     6/18            = 0.3333333    (doc TTP sigma table, unchanged)
#   loyalty 7.0783045/18    = 0.393239139  (rev 040926-2 loyalty sigma table, on the
#                            STANDARDIZED weighted_loyalty = std(bs) scale; the range
#                            grew from 6.432015 because the corrected Agreeableness
#                            weight 0.0123046 reshaped bs_weighted_loyalty)
#   wtp     6.868056/18     = 0.3815586667 (rev 040926-2 WTP sigma table; the previous
#                            0.3815592397 came from the superseded range 6.868064)
#   rt      5.979747/18     = 0.332208167  (rev 040926-2 RT sigma table - it now
#                            multiplies by mean(stdactions) like the other mechanisms
#                            and its quintile table uses 0.332208167, replacing the old
#                            0.28 ~ 5/18 multiplier and the factor-only overall sigma)
#   flex    6.5944228/18    = 0.3663568222 (rev 040926-2 Section 5 sigma table; the
#                            range moved from 6.6140623 with the adopted O/N/A
#                            coefficients, and the printed numerator is now correct)
SIGMA_FACTORS = {
    "ttp": 6.0 / 18.0,
    "loyalty": 0.393239139,
    "wtp": 0.3815586667,
    "risk_taking": 0.332208167,
    "flexibility": 0.3663568222,
}
# The sigma values (sigma_overall = factor * 1.1863376, sigma_quintile = factor *
# mean(stdactions within each allowance level)) live only in config/decisions.yaml,
# stochastic.mechanisms.<m>; SIGMA_OVERALL below is the file's column, read at import.
# They follow the document's formula; the two arithmetic slips in the document's
# printed sigma tables are noted next to the values in the YAML.

MECHANISMS = ("ttp", "loyalty", "wtp", "risk_taking", "flexibility")

# Ranking mechanisms (aggregation inputs, in the document's order) -> short key used
# in the rtd_* columns (rtd_<key>_ranking).
RANKING_KEYS = {"loyalty": "loyalty", "wtp": "wtp", "risk_taking": "rt", "flexibility": "flex"}

# Rescale targets: (low, span) so rescaled = low + (span - 0.0001) * u.
# ttp: 0..6 -> floor gives lengths 0..5; rankings: 1..5(.9999) -> floor gives 1..5.
_RESCALE = {
    "ttp": (0.0, 6.0),
    "loyalty": (1.0, 5.0),
    "wtp": (1.0, 5.0),
    "risk_taking": (1.0, 5.0),
    "flexibility": (1.0, 5.0),
}

# ---------------------------------------------------------------------------
# The FILE's values, read once at import through the one loader (src/config.py),
# for the readers that import them by name: app/pages/results/visualizations/
# transaction_viz.py (PRIORITY_SEQUENCES) and the Decision-4 tests. The decision
# reads every constant from ``params`` (the run's patched block), never from here -
# except the priority sequences, which the segment -> option-list mapping below
# takes from the file (as before this module never read params['priority_sequences']).
# ---------------------------------------------------------------------------
_FILE = file_block(DECISION)
# Priority sequences (highest-construct option first), verified against the .dta:
# segment s (1..5) receives the LAST s options, sequence[5-s:] (the document's prose
# and its rev 040926-2 Stata code; see _ranking_for_segment).
PRIORITY_SEQUENCES = {m: list(seq) for m, seq in require(_FILE, DECISION, "priority_sequences").items()}
FLEX_COEFFS = dict(require(_FILE, DECISION, "coefficients", "flexibility"))
FLEX_ANCHOR_WEIGHTS = {"observed": require(_FILE, DECISION, "flexibility_anchor", "observed_weight"),
                       "calculated": require(_FILE, DECISION, "flexibility_anchor", "calculated_weight")}
# frozen original-280 mean/SD (the name predates the file being the only source; it is
# NOT a fallback - the decision z-scores with params['z_scoring'])
DEFAULT_Z_SCORING = {t: dict(stats) for t, stats in require(_FILE, DECISION, "z_scoring").items()}
SIGMA_OVERALL = {m: require(_FILE, DECISION, "stochastic", "mechanisms", m, "sigma_overall")
                 for m in MECHANISMS}
# the file's Section-6 aggregation settings: what a partial params['aggregation'] block
# is merged over (see _aggregation_settings)
_FILE_AGGREGATION = {k: require(_FILE, DECISION, "aggregation", k)
                     for k in ("enabled", "mechanisms", "last_resort")}


def _z_score_trait(value: float, trait_name: str, z_params: Dict) -> float:
    """Z-score a raw trait using fixed original-280 mean/SD from config."""
    return z_score_trait(value, trait_name, z_params, DECISION)


def _coeffs(params: Dict, mechanism: str) -> Dict:
    """The mechanism's equation coefficients, params['coefficients'][mechanism], as floats."""
    cfg = require(params, DECISION, "coefficients", mechanism)
    prefix = f"coefficients.{mechanism}"
    return {k: float(require(cfg, DECISION, k, prefix=prefix)) for k in COEFFICIENT_KEYS[mechanism]}


def _income_mode(params: Dict) -> str:
    """Resolve the decision's income mode: 'categorical' or 'continuous' (default)."""
    return normalise_income_mode(params.get("income_mode", "continuous"), default="continuous")


def _categorical_income_effect(params: Dict, mechanism: str, level: int) -> float:
    """intercept + budget-level dummy for the categorical-income score (level 1 = base)."""
    cfg = require(params, DECISION, "categorical_income_effects", mechanism)
    prefix = f"categorical_income_effects.{mechanism}"
    value = float(require(cfg, DECISION, "intercept", prefix=prefix))
    if level in (2, 3, 4, 5):
        value += float(require(cfg, DECISION, f"level_{level}", prefix=prefix))
    return value


def compute_rtd_scores(agent_state: Dict[str, Any], params: Dict[str, Any],
                       simulation_config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Compute the five raw mechanism scores for one agent - always BETA0-FREE.

    Intercepts are deliberately NOT applied here: this function also feeds
    compute_rtd_population_stats, whose statistics must be intercept-free so the
    segment cutoffs stay fixed (see the module docstring's FIXED-CUTOFF semantics).
    The per-agent decision function adds each intercept's raw-scale equivalent
    downstream.

    Trait z-scores use the frozen original-280 stats (Stata `egen std` semantics,
    same convention as the other modelled decisions). z_income is population-level:
    (income - income_stats.mean) / income_stats.sd from the engine's Pass-1
    runtime stats, mirroring the disclose-decisions' continuous-income handling.

    The cognitive-flexibility operative score (anchored_flexibility) needs the
    population z-stats of the calculated score and of stdactions; it is returned as
    'flexibility' once rtd_population_stats['flexibility'] exists (None before the
    Pass-1 hook has run), alongside the population-free 'flexibility_calc' and the
    agent's 'stdactions'.
    """
    z_params = require(params, DECISION, "z_scoring")
    z_E = _z_score_trait(agent_state["ExtraversionBig5"], "ExtraversionBig5", z_params)
    z_A = _z_score_trait(agent_state["Agreeable"], "Agreeable", z_params)
    z_N = _z_score_trait(agent_state["NeuroticismBig5"], "NeuroticismBig5", z_params)
    z_C = _z_score_trait(agent_state["ConscientiousnessBig5"], "ConscientiousnessBig5", z_params)
    z_O = _z_score_trait(agent_state["OpennessBig5"], "OpennessBig5", z_params)

    # reducation = Education - 1 (survey coding 1=undergraduate, 2=graduate -> 0/1)
    reducation = float(agent_state["Education"]) - 1.0

    # z_income: prefer the Decision-4 population stats computed with SAMPLE SD (ddof=1,
    # Stata `egen std` semantics) by compute_rtd_population_stats; fall back to the
    # engine's shared income_stats (also sample SD) when the hook has not run.
    income = agent_state.get("income", None)
    sim = simulation_config or {}
    income_stats = (sim.get("rtd_population_stats", {}) or {}).get("income") \
        or sim.get("income_stats", {})
    i_mean = income_stats.get("mean", 0.0)
    i_sd = income_stats.get("sd", 1.0)
    if income is None or i_sd <= 0:
        z_I = 0.0
    else:
        z_I = (float(income) - i_mean) / i_sd

    c_ttp = _coeffs(params, "ttp")
    c_loy = _coeffs(params, "loyalty")
    c_wtp = _coeffs(params, "wtp")
    c_rt = _coeffs(params, "risk_taking")
    c_flex = _coeffs(params, "flexibility")

    # Flexibility (Section 5): the CALCULATED score from the Big 5; its
    # population standardisation and the 25/75 anchoring with z_stdactions happen in
    # flex_anchored_score() once the population stats exist. stdactions (observed SD
    # in actions per cycle) is a trait column; if it is absent the observed anchor is
    # neutral (z = 0) and the output flags it - the pipeline then reduces to the
    # calculated score alone.
    flexibility_ivw = (c_flex["extraversion"] * z_E + c_flex["openness"] * z_O
                       + c_flex["neuroticism"] * z_N + c_flex["agreeable"] * z_A
                       + c_flex["conscientiousness"] * z_C)
    stdactions = agent_state.get("stdactions", None)
    stdactions_missing = stdactions is None or (isinstance(stdactions, float) and np.isnan(stdactions))
    z_std = 0.0 if stdactions_missing else _z_score_trait(float(stdactions), "stdactions", z_params)

    weighted_ttp = (c_ttp["extraversion"] * z_E + c_ttp["agreeable"] * z_A
                    + c_ttp["neuroticism"] * z_N + c_ttp["conscientiousness"] * z_C
                    + c_ttp["education"] * reducation)
    weighted_loyalty = (c_loy["extraversion"] * z_E + c_loy["openness"] * z_O
                        + c_loy["agreeable"] * z_A)

    # Income-using elements (WTP, Risk-Taking): continuous income term or fitted
    # per-budget-level categorical effects. The personality part is identical in
    # both modes and enters the categorical score at coefficient 1.0 (doc-literal
    # cond() construction; see module docstring).
    income_mode = _income_mode(params)
    wtp_personality = c_wtp["extraversion"] * z_E + c_wtp["agreeable"] * z_A
    rt_personality = (c_rt["extraversion"] * z_E + c_rt["openness"] * z_O
                      + c_rt["agreeable"] * z_A + c_rt["conscientiousness"] * z_C
                      + c_rt["neuroticism"] * z_N)
    if income_mode == "categorical":
        # assignedallowancelevel 1..5 <-> totalallowance 12/32/72/128/200 (verified
        # 1:1 in the .dta).
        level = int(agent_state["Assigned Allowance Level"])
        wtp_calculated = wtp_personality + _categorical_income_effect(params, "wtp", level)
        rt_calculated = rt_personality + _categorical_income_effect(params, "risk_taking", level)
    else:
        wtp_calculated = wtp_personality + c_wtp["income"] * z_I
        rt_calculated = rt_personality + c_rt["income"] * z_I

    return {
        "ttp": float(weighted_ttp),
        "loyalty": float(weighted_loyalty),
        "wtp": float(wtp_calculated),
        "risk_taking": float(rt_calculated),
        "flexibility_ivw": float(flexibility_ivw),      # Flexibility_calculated_ivw
        "z_stdactions": float(z_std),
        "stdactions_missing": bool(stdactions_missing),
        "z_extraversion": float(z_E), "z_agreeable": float(z_A),
        "z_neuroticism": float(z_N), "z_conscientiousness": float(z_C),
        "z_openness": float(z_O), "z_income": float(z_I),
        "reducation": float(reducation),
    }


def _flex_weights(params: Dict):
    """(observed_weight, calculated_weight) of the flexibility anchor
    (params['flexibility_anchor']; doc defaults 0.25 / 0.75)."""
    w_obs = float(require(params, DECISION, "flexibility_anchor", "observed_weight"))
    w_calc = float(require(params, DECISION, "flexibility_anchor", "calculated_weight"))
    return w_obs, w_calc


def flex_anchored_score(scores: Dict[str, float], pop: Dict, params: Dict):
    """
    anchored_flexibility = w_obs * z_stdactions + w_calc * z_Flexibility_calculated_ivw,
    with z_Flexibility the POPULATION z-score (egen std) of the calculated score taken
    from pop['flexibility_ivw'] (mean/sd, BETA4-FREE). Returns (anchored, z_flexibility).
    Beta4 is added downstream by the per-agent function (FIXED-CUTOFF semantics).
    """
    st = pop.get("flexibility_ivw", {}) or {}
    sd = st.get("sd", 1.0) or 1.0
    z_flex = (scores["flexibility_ivw"] - st.get("mean", 0.0)) / sd
    w_obs, w_calc = _flex_weights(params)
    return float(w_obs * scores["z_stdactions"] + w_calc * z_flex), float(z_flex)


def _rescale(value: float, low: float, span: float, vmin: float, vmax: float) -> float:
    """Stata-style min-max rescale: low + (span - 0.0001) * (v - min)/(max - min)."""
    rng = vmax - vmin
    u = 0.5 if rng <= 0 else (value - vmin) / rng   # degenerate-population guard
    return low + (span - 0.0001) * u


def _segment(value: float, mechanism: str, vmin: float, vmax: float) -> int:
    low, span = _RESCALE[mechanism]
    rescaled = _rescale(value, low, span, vmin, vmax)
    seg = int(np.floor(rescaled))
    # By construction in [low, low+span-0.0001]; clip defensively for float edge cases.
    return int(np.clip(seg, int(low), int(low) + int(span) - 1))


def _ranking_for_segment(mechanism: str, segment: int) -> List[int]:
    """Segment s (1..5) -> the LAST s options of the mechanism's priority sequence.

    DOCUMENT-PROSE direction (what the document's own prose states - e.g. Section 5: "the highest-Flexibility 20% segment,
    Flexibility_calculated15 == 5, takes Option 2 as its first choice"):

        segment 5 (HIGHEST construct scores) -> the FULL sequence, top option first
                                                (flexibility: 2 > 4 > 3 > 1 > 5)
        segment 1 (LOWEST construct scores)  -> only the LAST option, list length 1
                                                (flexibility: Option 5)

    i.e. seq[5 - s:]. Document rev 040926-2 and Stata_File_Decision4_050926.dta have
    ADOPTED this direction: every `replace choice<p>_*` statement (deterministic and
    stochastic, all four elements) was rewritten and the file's stored choice1..5_*
    columns now equal these lists exactly, NaN pattern included, so the verification
    tests compare DIRECTLY against the stored columns (280/280 per element). The
    superseded 290826 file implemented the opposite direction, segment s -> seq[s-1:].
    """
    seq = PRIORITY_SEQUENCES[mechanism]
    return list(seq[len(seq) - segment:])


def _complete_ranking_for_segment(mechanism: str, segment: int) -> List[int]:
    """Segment s (1..5) -> the mechanism's COMPLETE ranking of ALL FIVE options.

    This is what the Section-6 rank aggregation consumes. Document rev 040926-2,
    Section 6, first paragraph: "The preceding sections produce, for each participant,
    four COMPLETE rankings of the five rejected transaction default options, one per
    explanatory variable ..., each depending on the participant's weighted factor score
    and the resulting first choice", and the two output rules (length truncation and the
    Option-5 cut) are applied only "Once the integrated ranking is formed".

    The complete ranking is the priority sequence ROTATED to start at the segment's
    first choice:

        seq[5 - s:] + seq[:5 - s]

    i.e. the element's own list (_ranking_for_segment, the tail that the .dta's
    choice1..5 columns store) followed by the options that list omits, kept in priority
    order. The two natural readings of "complete ranking ... depending on the score and
    the resulting first choice" - rotate the sequence to the first choice, or append the
    unlisted options in priority order - coincide exactly.

    Example (loyalty, sequence 3 > 1 > 4 > 5 > 2):
        segment 5 -> [3, 1, 4, 5, 2]   (full sequence; identical to the element list)
        segment 3 -> [4, 5, 2, 3, 1]   (element list [4, 5, 2], then 3 and 1)
        segment 1 -> [2, 3, 1, 4, 5]   (element list [2], then the rest)

    Before 2026-09-17 the aggregation was fed the TRUNCATED element lists with the
    unlisted options tied at the bottom. That left most option pairs with a zero
    pairwise margin, so Kemeny, Schulze, Copeland and the footrule could not separate
    them and 61.8% of the 280 participants reached the random last resort (the
    review's remark that "close to 50% of the rankings are resolved randomly"). Feeding the
    document's complete rankings drops that to 33.2%, close to the document's own
    100,000-case figure of 26.3%.

    NOTE: the element's OWN output list (rtd_<element>_ranking) stays the truncated
    tail - it is verified 280/280 against the file's choice1..5 columns. Only the
    aggregation input is the complete ranking.
    """
    seq = PRIORITY_SEQUENCES[mechanism]
    cut = len(seq) - int(segment)
    return list(seq[cut:]) + list(seq[:cut])


def _intercept(params: Dict, mechanism: str) -> float:
    """The element's configured intercept, params['intercepts'][mechanism].

    An absent block, an absent element or a null is 0: the reference tests pin a run
    with no 'intercepts' key (and one with a partial block) as bit-identical to
    explicit zeros, the Stata file's own setting (test_zero_intercepts_bit_identical,
    test_categorical_intercepts_apply_same_way). This is the one key of this decision
    that is not read strictly.
    """
    return float((params.get("intercepts") or {}).get(mechanism, 0.0) or 0.0)


def _intercept_raw_shift(mechanism: str, intercept: float, pop_m: Dict,
                         params: Optional[Dict] = None) -> float:
    """
    Raw-scale equivalent of the element's intercept (FIXED-CUTOFF semantics).

    ttp: beta0 applies on the RAW weighted_ttp composite -> shift = beta0. This is
    the document's scale: Section 1 writes TTP_i = beta0 + (-0.0152556564*E_i + ...)
    over z-scored traits, defines beta0 without the "after standardization" qualifier
    that beta1-beta4 carry, and the Stata code rescales the raw weighted_ttp to 0-6
    without ever standardizing it. (An earlier version had moved beta0 to the
    standardized scale, beta0*sd0; the intercept is applied on the scale the document
    specifies. Note the raw composite spans only ~0.21 on the 280
    participants, so beta0 = 0.05 moves weighted_ttp06 by ~+1.43; beta0 = 0, the
    Stata value, reproduces the .dta exactly.)

    loyalty/wtp/risk_taking: beta1/beta2/beta3 apply on the STANDARDIZED scale (doc:
    `replace weighted_loyalty = std(bs) + beta1`, `replace z_WTP = z_WTP + beta2`,
    `replace z_RT = z_RT + beta3`), z = (raw0 - mean0)/sd0 + beta with mean0/sd0 the
    BETA0-FREE population stats. The raw-scale equivalent is therefore beta * sd0:
      (raw0 + beta*sd0 - mean0)/sd0 = z0 + beta,
    and min-max rescaling against the beta0-free min0/max0 gives the identical bin
    whether performed on the shifted raw score or the shifted z-score (min-max is
    invariant to the affine raw->z map), so the raw representation is used
    throughout.

    flexibility: beta4 applies on the standardized CALCULATED score BEFORE anchoring
    (doc: `replace z_Flexibility_calculated_ivw = z_Flexibility_calculated_ivw + beta4`),
    so the anchored (operative) score shifts by calculated_weight * beta4.
    """
    if intercept == 0.0:
        return 0.0
    if mechanism in ("loyalty", "wtp", "risk_taking"):
        sd0 = pop_m.get("sd", 1.0) or 1.0
        return intercept * sd0
    if mechanism == "flexibility":
        return intercept * _flex_weights(params or {})[1]
    return intercept   # ttp: raw composite scale (doc)


def _resolve_sigma(stoch_m: Dict, agent_state: Dict, mechanism: str) -> float:
    """Resolve sigma * scale for one mechanism (overall or per-allowance-group)."""
    strategy = stoch_m.get("sigma_strategy", "overall")
    prefix = f"stochastic.mechanisms.{mechanism}"
    if strategy == "quintile":
        level = int(agent_state["Assigned Allowance Level"])
        sigma_raw = float(require(stoch_m, DECISION, "sigma_quintile", str(level), prefix=prefix))
        scale = float(require(stoch_m, DECISION, "quintile_scale_factors", str(level), prefix=prefix))
    else:
        sigma_raw = float(require(stoch_m, DECISION, "sigma_overall", prefix=prefix))
        scale = float(require(stoch_m, DECISION, "scale_factor", prefix=prefix))
    return sigma_raw * scale


def _stochastic_enabled(params: Dict, pop_context: str) -> bool:
    """Decision-wide stochastic gate; the same three-way rule as disclose_documents."""
    return should_use_stochastic(params.get("stochastic", {}), pop_context)


def _mechanism_stoch(params: Dict) -> Dict[str, Dict]:
    """Per-mechanism stochastic blocks (params['stochastic']['mechanisms'][name])."""
    stoch = params.get("stochastic", {})
    mechs = stoch.get("mechanisms", {})
    return {m: (mechs.get(m) or {}) for m in MECHANISMS}


def _draw_noise(rng: np.random.Generator) -> Dict[str, float]:
    """One standard normal per mechanism (five), in fixed MECHANISMS order.

    Always drawn together (even for mechanisms whose sigma is 0) so each mechanism's
    outcome is invariant to the other mechanisms' settings. compute_rtd_population_stats
    replicates this stream exactly - keep the two in lockstep. (Sequential generation
    means the first four values are identical to the pre-flexibility four-normal draw.)
    """
    n = rng.standard_normal(len(MECHANISMS))
    return dict(zip(MECHANISMS, n))


def _anchor_value(mechanism: str, stoch_m: Dict, raw: float, det_segment: int,
                  pop: Dict) -> float:
    """
    Anchor of the stochastic draw.

    'continuous' (default): the mechanism's continuous operative score - raw composite
    for ttp/wtp, the population-z score for loyalty (weighted_loyalty = std(bs)),
    risk_taking and flexibility (z_anchored_flexibility), whose sigmas are derived
    from the z-score ranges. This is the .dta-verified behaviour for WTP and the
    doc's stated behaviour for TTP.
    'binned': the deterministic 1-5 segment (doc-literal reading for loyalty/RT/flex).
    Note ttp's continuous anchor is the 0-6 rescaled weighted_ttp06 (doc line 191).

    Intercepts (FIXED-CUTOFF semantics): `raw` is whatever the caller operates on -
    the population hook passes the BETA0-FREE composite (so s_min/s_max stay
    intercept-free), the per-agent function passes the shifted operative score, so
    its anchor moves by the intercept in the anchor's own units (for risk_taking
    (raw0 + beta*sd0 - mean0)/sd0 = z0 + beta exactly).
    """
    anchor_mode = stoch_m.get("anchor", "continuous")
    if anchor_mode == "binned" and mechanism != "ttp":
        return float(det_segment)
    if mechanism == "ttp":
        return _rescale(raw, 0.0, 6.0, pop["ttp"]["min"], pop["ttp"]["max"])
    if mechanism in ("loyalty", "risk_taking", "flexibility"):
        sd = pop[mechanism].get("sd", 1.0) or 1.0
        return (raw - pop[mechanism].get("mean", 0.0)) / sd
    return raw   # wtp: raw composite (dta-verified)


def compute_rtd_population_stats(agents_df, all_incomes: List[float], params: Dict,
                                 simulation_config: Dict, pop_context: str = "documentation",
                                 agent_base_seeds: Optional[List[int]] = None,
                                 decision_offset: Optional[int] = None) -> Dict[str, Dict]:
    """
    Population-level statistics for Decision 4, called from the engine after
    Pass 1 (incomes known). Returns, per mechanism:
      {'min','max'}          of the deterministic composite score,
      {'mean','sd'}          of the raw composite (z reporting; wtp/rt intercept scale),
      {'s_min','s_max'}      of the STOCHASTIC draws across the population (only when
                             stochastic is active and agent_base_seeds are provided).

    ALL statistics are computed on the BETA0-FREE scores (compute_rtd_scores never
    applies intercepts, and the replicated draws below are anchored intercept-free):
    FIXED-CUTOFF semantics - the segment cutoffs and s_min/s_max are anchored on the
    intercept-free population, and the per-agent function adds each intercept's
    raw-scale equivalent AFTERWARD so a nonzero beta0 visibly moves agents across
    the fixed bin boundaries.

    The stochastic draws are replicated bit-for-bit from each agent's decision RNG
    stream: default_rng(agent_base_seed + decision_offset) . standard_normal(5),
    matching _draw_noise() in the per-agent function.
    """
    # Decision-4-specific income standardization stats: SAMPLE SD (ddof=1), matching
    # Stata `egen z_net_income = std(income)`. (The engine's shared income_stats uses
    # the same sample SD; Decision 4 keeps its own hook regardless.)
    inc = np.asarray([x for x in all_incomes if x is not None], dtype=float)
    if len(inc) > 1 and float(inc.std(ddof=1)) > 0:
        rtd_income_stats = {"mean": float(inc.mean()), "sd": float(inc.std(ddof=1))}
    else:
        rtd_income_stats = dict(simulation_config.get("income_stats", {})) or {"mean": 0.0, "sd": 1.0}
    sim = {"rtd_population_stats": {"income": rtd_income_stats}}

    raws = {m: [] for m in MECHANISMS}
    rows, all_scores = [], []
    for i, (_, row) in enumerate(agents_df.iterrows()):
        state = dict(row)
        state["income"] = all_incomes[i] if i < len(all_incomes) else None
        scores = compute_rtd_scores(state, params, sim)
        rows.append(state)
        all_scores.append(scores)
        for m in MECHANISMS:
            if m != "flexibility":
                raws[m].append(scores[m])

    # Flexibility is two-stage: `egen z_Flexibility = std(Flexibility_calculated_ivw)`
    # over the population FIRST, then the 25/75 anchoring with z_stdactions - so the
    # anchored (operative) scores need the calculated score's population mean/sd.
    ivw = np.asarray([s["flexibility_ivw"] for s in all_scores], dtype=float)
    pop_ivw = {"mean": float(ivw.mean()) if len(ivw) else 0.0,
               "sd": float(ivw.std(ddof=1)) if len(ivw) > 1 else 1.0}
    for s in all_scores:
        raws["flexibility"].append(flex_anchored_score(s, {"flexibility_ivw": pop_ivw}, params)[0])
    n_missing = int(sum(s["stdactions_missing"] for s in all_scores))
    if n_missing:
        print(f"[Decision 4] WARNING: stdactions missing for {n_missing}/{len(all_scores)} agents - "
              "the Flexibility observed anchor is neutral (z = 0) for them.")

    # reducation = Education - 1 must be the doc's binary 0/1 (undergraduate/graduate).
    # Verified 2026-09-16: the copula emits Education as an integer in {1, 2} on every
    # population tested, so reducation is exactly {0.0, 1.0}. Warn (never silently
    # coerce) if a retrained copula ever changes that - the TTP education term 0.0297
    # is the largest single coefficient in weighted_ttp, so a broken coding would
    # distort the whole option-list-length allocation.
    bad_reduc = int(sum(1 for s in all_scores if s["reducation"] not in (0.0, 1.0)))
    if bad_reduc:
        print(f"[Decision 4] WARNING: reducation (Education - 1) is not 0/1 for "
              f"{bad_reduc}/{len(all_scores)} agents - the doc codes Education as a "
              "binary undergraduate(1)/graduate(2) variable.")

    pop: Dict[str, Dict] = {"income": rtd_income_stats, "flexibility_ivw": pop_ivw}
    for m in MECHANISMS:
        arr = np.asarray(raws[m], dtype=float)
        pop[m] = {"min": float(arr.min()), "max": float(arr.max()),
                  "mean": float(arr.mean()),
                  "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 1.0}
    pop["flexibility"]["stdactions_missing"] = n_missing

    # ---- stochastic aggregates (min/max of the drawn values across the population) ----
    if _stochastic_enabled(params, pop_context) and agent_base_seeds is not None \
            and decision_offset is not None:
        mech_stoch = _mechanism_stoch(params)
        draws = {m: [] for m in MECHANISMS}
        for i, state in enumerate(rows):
            rng_i = np.random.default_rng(int(agent_base_seeds[i]) + int(decision_offset))
            noise = _draw_noise(rng_i)
            for m in MECHANISMS:
                stoch_m = mech_stoch[m]
                sigma = _resolve_sigma(stoch_m, state, m)
                det_seg = _segment(raws[m][i], m, pop[m]["min"], pop[m]["max"]) \
                    if m != "ttp" else 0
                # BETA0-FREE anchor: intercepts are deliberately ignored here so the
                # s_min/s_max cutoffs stay fixed (per-agent draws then shift by the
                # intercept and genuinely re-bin).
                anchor = _anchor_value(m, stoch_m, raws[m][i], det_seg, pop)
                draws[m].append(anchor + sigma * noise[m])
        for m in MECHANISMS:
            arr = np.asarray(draws[m], dtype=float)
            pop[m]["s_min"] = float(arr.min())
            pop[m]["s_max"] = float(arr.max())

    return pop


def _aggregation_settings(params: Dict) -> Dict:
    """params['aggregation'] (enabled / mechanisms / last_resort) merged over the FILE's
    aggregation block (_FILE_AGGREGATION, read at import - not a module copy).

    A partial block is an interface tests/test_rtd_rank_aggregation.py pins
    ({'enabled': False}, {'enabled': True, 'mechanisms': [...]}), so an absent or null
    key takes the file's value instead of raising; with ``intercepts`` it is the only
    block of this decision not read strictly. ``mechanisms`` null = every ranking
    mechanism the model computes.
    """
    cfg = params.get("aggregation") or {}
    out = dict(_FILE_AGGREGATION)
    if isinstance(cfg, dict):
        for k in out:
            if k in cfg and cfg[k] is not None:
                out[k] = cfg[k]
    out["enabled"] = bool(out["enabled"])
    out["last_resort"] = "lowest_option" if str(out["last_resort"]).lower() == "lowest_option" else "random"
    return out


def _aggregate_mechanism_rankings(out: Dict[str, Any], params: Dict, rng) -> None:
    """
    Section-6 rank aggregation: integrate the ranking mechanisms' lists into one
    default list and write the main output column + rtd_consensus_* diagnostics.

    Runs AFTER all mechanisms so the random last resort draws from the decision RNG
    only once the mechanisms' five standard normals have been consumed.
    """
    agg = _aggregation_settings(params)
    if not agg["enabled"]:
        return
    mechs = agg["mechanisms"] or [m for m in RANKING_KEYS if f"rtd_{RANKING_KEYS[m]}_ranking" in out]
    # The aggregation consumes each element's COMPLETE ranking of the five options
    # (doc Section 6; _complete_ranking_for_segment), NOT the truncated element list -
    # the element's own list stays the .dta-verified tail in rtd_<element>_ranking.
    inputs = {}
    for m in mechs:
        if m not in RANKING_KEYS:
            continue
        key = RANKING_KEYS[m]
        segment = out.get(f"rtd_{key}_segment")
        if segment is None or not isinstance(out.get(f"rtd_{key}_ranking"), list):
            continue
        inputs[m] = _complete_ranking_for_segment(m, int(segment))
    if not inputs:
        return
    res = integrate_default_list(list(inputs.values()), int(out.get("rtd_choice_length", 0)),
                                 rng=rng, last_resort=agg["last_resort"])
    out["rejected_transaction_defaults"] = [OPTION_CODES[o] for o in res["default_list"]]
    out["rtd_default_list"] = list(res["default_list"])
    out["rtd_default_list_length"] = int(res["default_list_length"])
    out["rtd_consensus_ranking"] = list(res["consensus"])
    out["rtd_consensus_ranking_codes"] = [OPTION_CODES[o] for o in res["consensus"]]
    out["rtd_consensus_settled_by"] = res["settled_by"]
    out["rtd_consensus_kemeny_status"] = res["kemeny_status"]
    out["rtd_consensus_phase1"] = res["phase1"]
    out["rtd_consensus_kemeny_distance"] = int(res["kemeny_distance"])
    out["rtd_consensus_n_kemeny_optimal"] = int(res["n_kemeny_optimal"])
    out["rtd_consensus_is_kemeny_optimal"] = bool(res["is_kemeny_optimal"])
    out["rtd_consensus_truncated_by"] = res["truncated_by"]
    out["rtd_consensus_inputs"] = list(inputs.keys())
    out["rtd_consensus_input_rankings"] = {m: list(r) for m, r in inputs.items()}
    out["rtd_consensus_last_resort"] = agg["last_resort"]


def _resolve_default_template(simulation_config: Dict, rng: np.random.Generator) -> List[str]:
    """Legacy default path: the configured priority template (or a random fallback)."""
    config = (simulation_config or {}).get("default_decisions", {}).get("rejected_transaction_defaults")
    if config and config.get("type") == "prioritized_selection":
        return list(config.get("priority_template", ["forgo_transaction"]))
    all_options = [OPTION_CODES[k] for k in (1, 2, 3, 4, 5)]
    num_options = rng.integers(1, 6)
    selected = list(rng.choice(all_options, size=num_options, replace=False))
    if "forgo_transaction" in selected:
        selected.remove("forgo_transaction")
        selected.append("forgo_transaction")
    return selected


def _mechanism_outputs(m: str, scores: Dict[str, Any], pop: Dict[str, Any], params: Dict[str, Any],
                       agent_state: Dict[str, Any], mech_stoch: Dict[str, Dict],
                       use_stochastic: bool, noise: Dict[str, float], out: Dict[str, Any]) -> None:
    """One mechanism's columns, written into ``out`` in order (the entry function's loop body).
    Uses no random numbers: ``noise`` holds the five normals drawn before the loop."""
    # FIXED-CUTOFF intercept semantics: pop[m] holds BETA0-FREE statistics (the
    # hook never applies intercepts), so the segment cutoffs are frozen; the
    # operative score is the beta0-free composite plus the intercept's raw-scale
    # equivalent (beta0 itself for ttp - the doc's raw composite scale; beta*sd0
    # for loyalty/wtp/risk_taking, i.e. beta is measured in SDs of the element's
    # own score; 0.75*beta4 for flexibility). The
    # min-max rescaled value then shifts by shift*(span-0.0001)/(max0-min0) and
    # the agent can cross bin boundaries (clipped at the end bins).
    raw0 = scores[m]                                     # beta0-free composite
    shift = _intercept_raw_shift(m, _intercept(params, m), pop[m], params)
    raw = raw0 + shift                                   # operative score
    vmin, vmax = pop[m]["min"], pop[m]["max"]            # beta0-free cutoffs
    stoch_m = mech_stoch[m]
    sigma = _resolve_sigma(stoch_m, agent_state, m) if use_stochastic else 0.0

    if m == "ttp":
        # ttp06 rescales the operative score against the beta0-free min/max, so
        # a nonzero beta0 shifts it by beta0*(6-0.0001)/(max0-min0) (it may
        # leave [0, 6); the clip below caps the length at the 0/5 boundary bins).
        ttp06 = _rescale(raw, 0.0, 6.0, vmin, vmax)
        det_len = int(np.clip(int(np.floor(ttp06)), 0, 5))
        out["rtd_weighted_ttp"] = raw
        out["rtd_weighted_ttp06"] = float(ttp06)
        out["rtd_choice_length_deterministic"] = det_len
        final_len = det_len
        drawn = use_stochastic and sigma > 0 and "s_min" in pop[m]
        if drawn:
            draw = ttp06 + sigma * noise[m]
            draw06 = _rescale(draw, 0.0, 6.0, pop[m]["s_min"], pop[m]["s_max"])
            final_len = int(np.clip(int(np.floor(draw06)), 0, 5))
            out["rtd_ttp_draw"] = float(draw)
        out["rtd_choice_length"] = final_len
        out["rtd_sigma_used_ttp"] = float(sigma if drawn else 0.0)
        return

    det_seg = _segment(raw, m, vmin, vmax)
    # z-score of the operative score over the BETA0-FREE population (reporting;
    # and the continuous anchor of loyalty/risk_taking/flexibility). With
    # raw = raw0 + beta*sd0 this is exactly z0 + beta for loyalty/wtp/risk_taking
    # - the doc's standardized-scale intercept; for flexibility it is
    # z0 + w_calc*beta/sd0. Matches the .dta's weighted_loyalty / z_WTP / z_RT /
    # z_anchored_flexibility columns at beta = 0.
    sd = pop[m].get("sd", 1.0) or 1.0
    z_val = (raw - pop[m].get("mean", 0.0)) / sd

    final_seg = det_seg
    draw_val = None
    drawn = use_stochastic and sigma > 0 and "s_min" in pop[m]
    if drawn:
        # Anchor on the operative score: shifts by the intercept in anchor units
        # (raw units for loyalty/wtp, z units for risk_taking, quantized segment
        # for 'binned'), while s_min/s_max stay beta0-free -> the re-binning
        # shifts consistently with the deterministic segment.
        anchor = _anchor_value(m, stoch_m, raw, det_seg, pop)
        draw_val = anchor + sigma * noise[m]
        low, span = _RESCALE[m]
        rescaled = _rescale(draw_val, low, span, pop[m]["s_min"], pop[m]["s_max"])
        final_seg = int(np.clip(int(np.floor(rescaled)), 1, 5))

    ranking = _ranking_for_segment(m, final_seg)
    key = RANKING_KEYS[m]
    # rtd_*_score is the OPERATIVE score: the composite + the intercept's
    # raw-scale equivalent (beta*sd0 for the standardized-scale intercepts, which
    # appear as exactly +beta on rtd_*_z; w_calc*beta for flexibility). At
    # beta = 0 it is the raw composite (for flexibility: anchored_flexibility).
    out[f"rtd_{key}_score"] = raw
    out[f"rtd_{key}_z"] = float(z_val)
    out[f"rtd_{key}_segment_deterministic"] = det_seg
    out[f"rtd_{key}_segment"] = final_seg
    out[f"rtd_{key}_ranking"] = ranking
    out[f"rtd_{key}_ranking_codes"] = [OPTION_CODES[o] for o in ranking]
    if draw_val is not None:
        out[f"rtd_{key}_draw"] = float(draw_val)
    out[f"rtd_sigma_used_{key}"] = float(sigma if drawn else 0.0)


def rejected_transaction_defaults(agent_state: Dict[str, Any], params: Dict[str, Any],
                                  rng, simulation_config: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> Dict[str, Any]:
    """
    Decision 4 simulation entry point.

    DEFAULT path (decision unselected, i.e. listed in default_decisions_list):
    unchanged legacy behaviour - every agent receives the configured priority template.

    MODEL path (decision selected): computes the five sub-decision
    mechanisms (rtd_* columns) and then integrates the ranking mechanisms' lists into
    ONE default list per the document's Section 6 (Kemeny-Young + tie-break
    hierarchy, truncated to the TTP choice length and at Option 5). That list is the
    main 'rejected_transaction_defaults' column (option codes; empty when the choice
    length is 0 or the aggregation is disabled in config).

    Per-element intercepts (params['intercepts']) follow the FIXED-CUTOFF semantics
    (module docstring): cutoffs from the beta0-free population, intercept added to
    this agent's operative score afterward, so beta0 observably shifts the
    allocation. beta = 0 is bit-identical to the intercept-free pipeline.
    """
    sim = simulation_config or {}
    pop_context = kwargs.get("pop_context", "documentation")

    # The trait model runs whenever the decision is selected; there is no separate
    # 'model_enabled' switch (the former flag was always True and gated nothing).
    if "rejected_transaction_defaults" in sim.get("default_decisions_list", []):
        template = _resolve_default_template(sim, rng)
        return {"rejected_transaction_defaults": template}

    pop = sim["rtd_population_stats"]   # computed by the engine after Pass 1

    scores = compute_rtd_scores(agent_state, params, sim)
    # Flexibility operative score: anchored_flexibility (beta4-free here;
    # the intercept's raw-scale equivalent is added in the loop below like the others).
    scores["flexibility"], z_flex_ivw = flex_anchored_score(scores, pop, params)
    use_stochastic = _stochastic_enabled(params, pop_context)
    mech_stoch = _mechanism_stoch(params)
    noise = _draw_noise(rng)   # always consumed: keeps draws aligned with the pop hook

    out: Dict[str, Any] = {"rejected_transaction_defaults": []}

    # ---- diagnostics shared by all mechanisms ----
    out["rtd_income_mode"] = _income_mode(params)
    out["rtd_z_extraversion"] = scores["z_extraversion"]
    out["rtd_z_agreeable"] = scores["z_agreeable"]
    out["rtd_z_neuroticism"] = scores["z_neuroticism"]
    out["rtd_z_conscientiousness"] = scores["z_conscientiousness"]
    out["rtd_z_openness"] = scores["z_openness"]
    out["rtd_z_income"] = scores["z_income"]
    out["rtd_reducation"] = scores["reducation"]
    # Flexibility intermediates (Stata: Flexibility_calculated_ivw, z_Flexibility_calculated_ivw,
    # z_stdactions); the anchored score and its z are rtd_flex_score / rtd_flex_z below.
    out["rtd_flex_ivw"] = scores["flexibility_ivw"]
    out["rtd_flex_z_ivw"] = float(z_flex_ivw)
    out["rtd_z_stdactions"] = scores["z_stdactions"]
    out["rtd_flex_stdactions_missing"] = scores["stdactions_missing"]

    for m in MECHANISMS:
        _mechanism_outputs(m, scores, pop, params, agent_state, mech_stoch, use_stochastic, noise, out)

    # ---- Section 6: integrate the ranking mechanisms into one default list ----
    _aggregate_mechanism_rankings(out, params, rng)

    return out
