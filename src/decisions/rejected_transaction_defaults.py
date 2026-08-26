# src/decisions/rejected_transaction_defaults.py
"""
Decision 4: Rejected Transaction Defaults - four trait-based sub-decision mechanisms.

Source: "Decision 4 - Rejected Transaction Defaults" design document (100726) +
professor's Stata file `Stata_File_Decision4_050626.dta` (280 participants, the
ground-truth arbiter wherever the document is ambiguous).

The decision is FOUR separate sub-decisions, each producing its own output per agent
(they are NOT combined here - rank aggregation across mechanisms is a later, separate
step that the document explicitly leaves unspecified):

  1. LIST LENGTH (Tendency to Plan, TTP)  -> how many default options to pre-select, 0-5
       weighted_ttp = -0.0152556564*z_E + 0.0177638642*z_A + 0.01959*z_N
                      + 0.00901465*z_C + 0.0297*reducation        (doc line 164)
       weighted_ttp06 = (6-0.0001) * minmax(weighted_ttp);  length = floor(weighted_ttp06)
  2. LOYALTY ranking   -> priority sequence Option 3 > 1 > 4 > 5 > 2
       weighted_loyalty = 0.09*z_E + 0.0273*z_O + 0.0045*z_A      (doc line 598)
  3. WILLINGNESS-TO-PAY ranking -> priority sequence Option 3 > 2 > 1 > 4 > 5
       WTP = 0.0788796127824*z_E - 0.012328716*z_A + 0.69814232*z_income
       (Hunter & Schmidt pooling, doc line 1333; the doc's H&S weight sum uses 4764
        instead of 4765 - the .dta provably embeds the 4764-based coefficients, and the
        error is a uniform scale on all three coefficients so every downstream output
        is identical; we therefore keep the .dta-verified values.)
  4. RISK-TAKING ranking -> priority sequence Option 4 > 2 > 1 > 3 > 5
       RT = 0.025942386297*z_E + 0.023699214948*z_O - 0.038734315188*z_A
            - 0.037739440732*z_C - 0.025388697852*z_N + 0.006874197106*z_income
       (Hunter & Schmidt pooling, doc line 2436; H&S retained per doc line 2444.)

CATEGORICAL INCOME (params['income_mode'] = 'categorical'; default 'continuous'):
Only WTP and Risk-Taking use income; TTP and Loyalty are identical in both modes.
The categorical variant (doc rev 130826, WTP lines 1297-1327 / RT lines 2470-2500)
replaces the continuous income term with fitted per-budget-level effects from an OLS
of the full score on the income-free personality part plus i.totalallowance dummies
(both regressions re-verified against the .dta, coefficients reproduce to <=1.2e-7):
  score_cat = intercept + level_dummy(assignedallowancelevel) + personality_part
where personality_part is the element's equation minus the income term applied at
coefficient 1.0 - DOC-LITERAL: the doc's operative cond() code applies it at slope 1;
the fitted slope on the personality part (0.5439389 WTP / 0.9991365 RT) is NOT
applied (flagged to the professor separately).
  WTP: intercept -0.6919842; dummies +0.2672136/+0.5058749/+0.9413666/+1.822843
       (levels 2-5 = EUR 32/72/128/200; level 1 = EUR 12 gets the intercept only).
       NOTE the doc's cond() level-1 branch writes raw 'agreeable' instead of
       'z_agreeable' - a confirmed typo contradicting the doc's own regression
       construction (WTP_noincome uses z_agreeable, line 1298); z_agreeable is used
       for ALL levels.
  RT:  intercept -0.0068307; dummies +0.0026128/+0.0050555/+0.0092738/+0.0179812.
Downstream (standardize -> min-max -> floor into 5 segments -> priority-sequence
tail mapping -> optional stochastic layer) is IDENTICAL with the categorical score
substituted (doc lines 1386/1435/2522: "replacing z_WTP_calculated with
z_WTP_categorical"); min-max rescaling is invariant to standardization and additive
intercepts, so binning the raw categorical score gives identical segments to binning
its standardized version. The professor has NOT yet specified categorical-specific
sigma values ("I have not checked that yet") - the continuous sigma config is reused
unchanged.

For each ranking mechanism the operative score is min-max rescaled into five
EQUAL-WIDTH segments 1..5 (floor(1 + (5-0.0001)*u), doc lines 615/1350/2449; note the
document's "20% of observations" narrative describes quantiles but the Stata code and
.dta implement equal-width bins, with e.g. 60% of participants in the lowest WTP bin)
and segment s receives the TAIL of the mechanism's priority sequence starting at
position 6-s (MIRRORED mapping, per the professor's 2026-08 ruling):
  segment 5 (highest scores) -> all five options starting with the top option,
  segment 1 (lowest scores)  -> only the last option.
This follows the document's prose ("the 20% highest ... will opt for Option 3 first").
The document's own Stata replace statements - and hence the choice1..5_* columns in
Stata_File_Decision4_050626.dta - implement the inverse; the professor confirmed that
was a bug in the Stata code, so the stored choice columns are known-flipped (choice
parity is asserted against their mirror; segments still verify against the .dta
exactly). A corrected .dta is expected for direct parity.

DTA-verified deviations from the document's literal Stata listings (all confirmed
against Stata_File_Decision4_050626.dta):
  - Loyalty: the documented `egen weighted_loyalty = std(bs)` + beta0 step was NOT
    executed; the stored weighted_loyalty IS the raw composite (min/max
    -0.26831895/0.21461959 as the doc itself quotes at line 646). Affine-invariant
    for all downstream outputs, so we use the raw composite.
  - WTP: z_WTP_calculated = std(WTP_calculated) exactly, NO beta1 added (the doc's
    `replace` line is a typo). The .dta embeds no intercepts; nonzero intercepts act
    through the FIXED-CUTOFF semantics described below (beta = 0 reproduces the
    .dta pipeline bit-for-bit).
  - WTP stochastic (the ONLY mechanism with stochastic ground truth in the .dta):
    the draw is anchored on the RAW continuous WTP_calculated (NOT the binned 1-5
    score the doc states) with ONE common sigma = 0.45265807275 (per-quintile fit is
    statistically rejected on the .dta). We therefore default every mechanism's
    stochastic anchor to its CONTINUOUS operative score; the doc-literal binned
    anchor for loyalty/RT remains available via stochastic.anchor = 'binned'.
  - Sigma constants follow sigma = (range/18) * mean(stdactions-within-group) with
    documented values; the loyalty allowance-32 table cell 0.04975281 is a dropped-
    digit typo for 0.0304975281 (we use the corrected value), and the RT sigma
    0.332208167 omits the *mean(stdactions) factor its own formula states (a doc
    inconsistency; we follow the doc's stated number since no .dta arbiter exists).

INTERCEPTS (params['intercepts'][mechanism]) - FIXED-CUTOFF semantics: the TTP and
Loyalty beta0 apply on the raw composite scale; the WTP beta1 / RT beta2 apply on the
standardized scale (doc lines 1338/2431). All population statistics (min/max/mean/sd
and the stochastic s_min/s_max) are computed on the BETA0-FREE scores, which freezes
the segment cutoffs; each agent's operative score is then the beta0-free composite
plus the intercept's raw-scale equivalent (beta for ttp/loyalty; beta*sd0 for
wtp/risk_taking, so that (raw0 + beta*sd0 - mean0)/sd0 = z0 + beta). The min-max
rescaled value therefore shifts by beta_raw * (span - 0.0001) / (max0 - min0) and
agents genuinely cross the fixed bin boundaries (clipped at the end bins). The
stochastic anchor shifts by the same amount in its own units while s_min/s_max stay
beta0-free, so the drawn values and their re-binning shift consistently. beta = 0
reproduces the intercept-free outputs bit-for-bit. (Previously the intercepts were
added BEFORE the population min/max was taken, which made them provably inert:
min-max rescaling cancels a uniform shift.)

Population-level statistics: `egen min/max/std` are POPULATION operations, so the
orchestrators call compute_rtd_population_stats() after Pass 1 (mirroring the
disclose-decisions' continuous-stats hooks). For the stochastic variants the bins
depend on the min/max of the DRAWN values across the whole population, so the hook
replicates each agent's decision RNG stream (default_rng(agent_base_seed +
decision_index*1000), first four standard normals) bit-for-bit; the per-agent
function then reproduces its own draw from the rng it is handed. A parity test
asserts the replication.

Verified against the .dta (tests/test_rejected_transaction_defaults.py): all four
deterministic mechanisms reproduce 280/280 exactly (float32 tolerance), including the
per-segment NaN pattern of the choice lists, and the WTP stochastic mapping keyed on
sWTP_calculated15.
"""

import numpy as np
from typing import Any, Dict, List, Optional

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

# Priority sequences (highest-construct option first), verified against the .dta:
# segment s (1..5) receives sequence[s-1:].
PRIORITY_SEQUENCES = {
    "loyalty": [3, 1, 4, 5, 2],       # doc line 609
    "wtp": [3, 2, 1, 4, 5],           # doc line 752/1343
    "risk_taking": [4, 2, 1, 3, 5],   # doc line 1856/2455
}

# ---------------------------------------------------------------------------
# Equation coefficients (doc-derived, .dta-verified to <=1e-7 on all 280 rows).
# ---------------------------------------------------------------------------
TTP_COEFFS = {
    "extraversion": -0.0152556564,
    "agreeable": 0.0177638642,
    "neuroticism": 0.01959,
    "conscientiousness": 0.00901465,
    "education": 0.0297,            # applies to reducation = Education - 1 (0/1)
}
LOYALTY_COEFFS = {
    "extraversion": 0.09,
    "openness": 0.0273,
    "agreeable": 0.0045,
}
WTP_COEFFS = {                       # Hunter & Schmidt pooling (doc line 1312/1333)
    "extraversion": 0.0788796127824,
    "agreeable": -0.012328716,
    "income": 0.69814232,
}
RT_COEFFS = {                        # Hunter & Schmidt pooling (doc line 2436)
    "extraversion": 0.025942386297,
    "openness": 0.023699214948,
    "agreeable": -0.038734315188,
    "conscientiousness": -0.037739440732,
    "neuroticism": -0.025388697852,
    "income": 0.006874197106,
}

# Categorical-income effects (doc rev 130826 regressions, lines 1303-1320 WTP /
# 2477-2492 RT; re-verified against the .dta). level_1 (EUR 12) is the base level:
# intercept only. Config override: params['categorical_income_effects'].
CATEGORICAL_INCOME_EFFECTS = {
    "wtp": {
        "intercept": -0.6919842,
        "level_2": 0.2672136,    # EUR 32
        "level_3": 0.5058749,    # EUR 72
        "level_4": 0.9413666,    # EUR 128
        "level_5": 1.822843,     # EUR 200
    },
    "risk_taking": {
        "intercept": -0.0068307,
        "level_2": 0.0026128,
        "level_3": 0.0050555,
        "level_4": 0.0092738,
        "level_5": 0.0179812,
    },
}

# Fallback z-scoring stats (frozen original-280 mean/SD, matching config/decisions.yaml
# and Stata `egen std` on the professor's file).
DEFAULT_Z_SCORING = {
    "ExtraversionBig5": {"mean": 3.557857, "sd": 0.6989565},
    "Agreeable": {"mean": 3.546071, "sd": 0.3732712},
    "NeuroticismBig5": {"mean": 2.702143, "sd": 0.6839657},
    "ConscientiousnessBig5": {"mean": 3.657143, "sd": 0.5596521},
    "OpennessBig5": {"mean": 4.060714, "sd": 0.5068274},
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
# Per-mechanism quintile multiplier applied to mean(stdactions within group):
#   ttp     6/18            = 0.3333333  (doc line 178-189)
#   loyalty 0.48293854/18   = 0.0268299279 (doc line 647-658; allowance-32 cell corrected)
#   wtp     6.868064/18     = 0.3815592397 (doc line 1382-1393)
#   rt      0.28            (doc line 2481-2489; NOTE: the doc's overall RT sigma
#                            0.332208167 = 5.979747/18 omits the *mean(stdactions)
#                            factor used by the other three mechanisms - followed
#                            as stated, flagged in the design notes)
SIGMA_FACTORS = {
    "ttp": 6.0 / 18.0,
    "loyalty": 0.0268299279,
    "wtp": 0.3815592397,
    "risk_taking": 0.28,
}
SIGMA_OVERALL = {
    "ttp": 0.395446,          # (6/18)*1.186338          (doc line 178)
    "loyalty": 0.0318293523,  # (0.48293854/18)*1.186338 (doc line 647)
    "wtp": 0.45265807275,     # (6.868064/18)*1.186338   (doc line 1382; .dta-consistent)
    "risk_taking": 0.332208167,  # 5.979747/18           (doc line 2479, see note above)
}

MECHANISMS = ("ttp", "loyalty", "wtp", "risk_taking")

# Rescale targets: (low, span) so rescaled = low + (span - 0.0001) * u.
# ttp: 0..6 -> floor gives lengths 0..5; rankings: 1..5(.9999) -> floor gives 1..5.
_RESCALE = {
    "ttp": (0.0, 6.0),
    "loyalty": (1.0, 5.0),
    "wtp": (1.0, 5.0),
    "risk_taking": (1.0, 5.0),
}


def _z_score_trait(value: float, trait_name: str, z_params: Dict) -> float:
    """Z-score a raw trait using fixed original-280 mean/SD from config."""
    p = z_params.get(trait_name, DEFAULT_Z_SCORING.get(trait_name, {}))
    mean = p.get("mean", 0)
    sd = p.get("sd", 1)
    if sd == 0:
        return 0.0
    return (value - mean) / sd


def _coeffs(params: Dict, mechanism: str, defaults: Dict) -> Dict:
    cfg = (params.get("coefficients") or {}).get(mechanism)
    if isinstance(cfg, dict):
        return {k: float(cfg.get(k, v)) for k, v in defaults.items()}
    return defaults


def _income_mode(params: Dict) -> str:
    """Resolve the decision's income mode: 'categorical' or 'continuous' (default)."""
    mode = str(params.get("income_mode", "continuous")).lower()
    return "categorical" if "categorical" in mode else "continuous"


def _categorical_income_effect(params: Dict, mechanism: str, level: int) -> float:
    """intercept + budget-level dummy for the categorical-income score (level 1 = base)."""
    cfg = (params.get("categorical_income_effects") or {}).get(mechanism)
    defaults = CATEGORICAL_INCOME_EFFECTS[mechanism]
    if isinstance(cfg, dict):
        eff = {k: float(cfg.get(k, v)) for k, v in defaults.items()}
    else:
        eff = defaults
    value = eff["intercept"]
    if level in (2, 3, 4, 5):
        value += eff[f"level_{level}"]
    return value


def compute_rtd_scores(agent_state: Dict[str, Any], params: Dict[str, Any],
                       simulation_config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Compute the four raw mechanism scores for one agent - always BETA0-FREE.

    Intercepts are deliberately NOT applied here: this function also feeds
    compute_rtd_population_stats, whose statistics must be intercept-free so the
    segment cutoffs stay fixed (see the module docstring's FIXED-CUTOFF semantics).
    The per-agent decision function adds each intercept's raw-scale equivalent
    downstream.

    Trait z-scores use the frozen original-280 stats (Stata `egen std` semantics,
    same convention as the other modelled decisions). z_income is population-level:
    (income - income_stats.mean) / income_stats.sd from the orchestrator's Pass-1
    runtime stats, mirroring the disclose-decisions' continuous-income handling.
    """
    z_params = params.get("z_scoring", {})
    z_E = _z_score_trait(agent_state.get("ExtraversionBig5", 0), "ExtraversionBig5", z_params)
    z_A = _z_score_trait(agent_state.get("Agreeable", 0), "Agreeable", z_params)
    z_N = _z_score_trait(agent_state.get("NeuroticismBig5", 0), "NeuroticismBig5", z_params)
    z_C = _z_score_trait(agent_state.get("ConscientiousnessBig5", 0), "ConscientiousnessBig5", z_params)
    z_O = _z_score_trait(agent_state.get("OpennessBig5", 0), "OpennessBig5", z_params)

    # reducation = Education - 1 (survey coding 1=undergraduate, 2=graduate -> 0/1)
    reducation = float(agent_state.get("Education", 1)) - 1.0

    # z_income: prefer the Decision-4 population stats computed with SAMPLE SD (ddof=1,
    # Stata `egen std` semantics) by compute_rtd_population_stats; fall back to the
    # orchestrators' shared income_stats (population SD) when the hook has not run.
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

    c_ttp = _coeffs(params, "ttp", TTP_COEFFS)
    c_loy = _coeffs(params, "loyalty", LOYALTY_COEFFS)
    c_wtp = _coeffs(params, "wtp", WTP_COEFFS)
    c_rt = _coeffs(params, "risk_taking", RT_COEFFS)

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
        # 1:1 in the .dta); default 3 mirrors _resolve_sigma / disclose_documents.
        level = int(agent_state.get("Assigned Allowance Level", 3))
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
        "z_extraversion": float(z_E), "z_agreeable": float(z_A),
        "z_neuroticism": float(z_N), "z_conscientiousness": float(z_C),
        "z_openness": float(z_O), "z_income": float(z_I),
        "reducation": float(reducation),
    }


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
    """Segment s (1..5) -> the tail of the mechanism's priority sequence from position 6-s.

    MIRRORED mapping (professor's ruling, 2026-08 feedback): the HIGHEST segment gets
    the full priority list starting with the top option; the lowest segment gets only
    the last option. This matches the design document's prose ("the 20% highest ...
    opt for Option 3 as their first choice"). NOTE: the document's own Stata replace
    statements — and therefore the choice1-5_* columns stored in
    Stata_File_Decision4_050626.dta — implement the INVERSE (segment 1 gets the top
    option); the professor confirmed that was a bug in the Stata code, so the stored
    choice columns are known-flipped and parity for the choice lists is asserted
    against their mirror until a corrected .dta is provided. Segments themselves are
    unaffected and still verify against the .dta exactly.
    """
    seq = PRIORITY_SEQUENCES[mechanism]
    return list(seq[5 - segment:])


def _intercept(params: Dict, mechanism: str) -> float:
    """The element's configured intercept (params['intercepts'][mechanism]; 0 absent)."""
    return float((params.get("intercepts") or {}).get(mechanism, 0.0) or 0.0)


def _intercept_raw_shift(mechanism: str, intercept: float, pop_m: Dict) -> float:
    """
    Raw-scale equivalent of the element's intercept (FIXED-CUTOFF semantics).

    ttp/loyalty: beta0 applies directly on the raw composite scale -> shift = beta.
    wtp/risk_taking: beta1/beta2 apply on the STANDARDIZED scale (doc lines
    1338/2431), z = (raw0 - mean0)/sd0 + beta with mean0/sd0 the BETA0-FREE
    population stats. The raw-scale equivalent is therefore beta * sd0:
      (raw0 + beta*sd0 - mean0)/sd0 = z0 + beta,
    and min-max rescaling against the beta0-free min0/max0 gives the identical bin
    whether performed on the shifted raw score or the shifted z-score (min-max is
    invariant to the affine raw->z map), so the raw representation is used
    throughout.
    """
    if intercept == 0.0:
        return 0.0
    if mechanism in ("wtp", "risk_taking"):
        sd0 = pop_m.get("sd", 1.0) or 1.0
        return intercept * sd0
    return intercept


def _resolve_sigma(stoch_m: Dict, agent_state: Dict, mechanism: str) -> float:
    """Resolve sigma * scale for one mechanism (overall or per-allowance-group)."""
    strategy = stoch_m.get("sigma_strategy", "overall")
    if strategy == "quintile":
        level = int(agent_state.get("Assigned Allowance Level", 3))
        default_q = SIGMA_FACTORS[mechanism] * MEAN_STDACTIONS.get(level, MEAN_STDACTIONS["overall"])
        sigma_raw = float(stoch_m.get("sigma_quintile", {}).get(str(level), default_q))
        scale = float(stoch_m.get("quintile_scale_factors", {}).get(str(level), stoch_m.get("scale_factor", 1.0)))
    else:
        sigma_raw = float(stoch_m.get("sigma_overall", SIGMA_OVERALL[mechanism]))
        scale = float(stoch_m.get("scale_factor", 1.0))
    return sigma_raw * scale


def _stochastic_enabled(params: Dict, pop_context: str) -> bool:
    """Decision-wide stochastic gate; the same three-way rule as disclose_documents."""
    stoch = params.get("stochastic", {})
    return (
        (stoch.get("in_copula", False) and pop_context == "copula")
        or (pop_context == "documentation" and stoch.get("sigma_value", 0) > 0)
    )


def _mechanism_stoch(params: Dict) -> Dict[str, Dict]:
    """Per-mechanism stochastic blocks (params['stochastic']['mechanisms'][name])."""
    stoch = params.get("stochastic", {})
    mechs = stoch.get("mechanisms", {})
    return {m: (mechs.get(m) or {}) for m in MECHANISMS}


def _draw_noise(rng: np.random.Generator) -> Dict[str, float]:
    """Four standard normals in fixed mechanism order.

    Always drawn together (even for mechanisms whose sigma is 0) so each mechanism's
    outcome is invariant to the other mechanisms' settings. compute_rtd_population_stats
    replicates this stream exactly - keep the two in lockstep.
    """
    n = rng.standard_normal(4)
    return dict(zip(MECHANISMS, n))


def _anchor_value(mechanism: str, stoch_m: Dict, raw: float, det_segment: int,
                  pop: Dict) -> float:
    """
    Anchor of the stochastic draw.

    'continuous' (default): the mechanism's continuous operative score - raw composite
    for ttp/loyalty/wtp, the population-z score for risk_taking (whose sigma is
    derived from the z-score range). This is the .dta-verified behaviour for WTP and
    the doc's stated behaviour for TTP.
    'binned': the deterministic 1-5 segment (doc-literal reading for loyalty/RT).
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
    if mechanism == "risk_taking":
        sd = pop["risk_taking"].get("sd", 1.0) or 1.0
        return (raw - pop["risk_taking"].get("mean", 0.0)) / sd
    return raw   # loyalty, wtp: raw composite (dta-verified for wtp)


def compute_rtd_population_stats(agents_df, all_incomes: List[float], params: Dict,
                                 simulation_config: Dict, pop_context: str = "documentation",
                                 agent_base_seeds: Optional[List[int]] = None,
                                 decision_offset: Optional[int] = None) -> Dict[str, Dict]:
    """
    Population-level statistics for Decision 4, called from the orchestrators after
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
    stream: default_rng(agent_base_seed + decision_offset) . standard_normal(4),
    matching _draw_noise() in the per-agent function.
    """
    # Decision-4-specific income standardization stats: SAMPLE SD (ddof=1), matching
    # Stata `egen z_net_income = std(income)`. (The orchestrators' shared income_stats
    # uses population SD; that convention is left untouched for the other decisions.)
    inc = np.asarray([x for x in all_incomes if x is not None], dtype=float)
    if len(inc) > 1 and float(inc.std(ddof=1)) > 0:
        rtd_income_stats = {"mean": float(inc.mean()), "sd": float(inc.std(ddof=1))}
    else:
        rtd_income_stats = dict(simulation_config.get("income_stats", {})) or {"mean": 0.0, "sd": 1.0}
    sim = {"rtd_population_stats": {"income": rtd_income_stats}}

    raws = {m: [] for m in MECHANISMS}
    rows = []
    for i, (_, row) in enumerate(agents_df.iterrows()):
        state = dict(row)
        state["income"] = all_incomes[i] if i < len(all_incomes) else None
        scores = compute_rtd_scores(state, params, sim)
        rows.append(state)
        for m in MECHANISMS:
            raws[m].append(scores[m])

    pop: Dict[str, Dict] = {"income": rtd_income_stats}
    for m in MECHANISMS:
        arr = np.asarray(raws[m], dtype=float)
        pop[m] = {"min": float(arr.min()), "max": float(arr.max()),
                  "mean": float(arr.mean()),
                  "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 1.0}

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


def rejected_transaction_defaults(agent_state: Dict[str, Any], params: Dict[str, Any],
                                  rng, simulation_config: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> Dict[str, Any]:
    """
    Decision 4 simulation entry point.

    DEFAULT path (decision unselected, or model config absent): unchanged legacy
    behaviour - every agent receives the configured priority template.

    MODEL path (decision selected with the trait model): computes the four sub-decision
    mechanisms. The main 'rejected_transaction_defaults' column is an EMPTY list -
    the document does not specify how the four mechanisms combine into one default
    list (rank aggregation is explicitly out of scope), so no aggregated list is
    fabricated. All mechanism outputs are emitted as rtd_* columns.

    Per-element intercepts (params['intercepts']) follow the FIXED-CUTOFF semantics
    (module docstring): cutoffs from the beta0-free population, intercept added to
    this agent's operative score afterward, so beta0 observably shifts the
    allocation. beta = 0 is bit-identical to the intercept-free pipeline.
    """
    sim = simulation_config or {}
    pop_context = kwargs.get("pop_context", "documentation")

    is_default = "rejected_transaction_defaults" in sim.get("default_decisions_list", [])
    has_model = bool(params) and bool(params.get("model_enabled", False))
    if is_default or not has_model:
        template = _resolve_default_template(sim, rng)
        return {"rejected_transaction_defaults": template}

    pop = sim.get("rtd_population_stats")
    if not pop:
        # Orchestrator did not provide population stats (should not happen once wired);
        # fall back to the legacy template rather than emitting wrong bins.
        template = _resolve_default_template(sim, rng)
        return {"rejected_transaction_defaults": template,
                "rtd_model_error": "missing rtd_population_stats"}

    scores = compute_rtd_scores(agent_state, params, sim)
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

    for m in MECHANISMS:
        # FIXED-CUTOFF intercept semantics: pop[m] holds BETA0-FREE statistics (the
        # hook never applies intercepts), so the segment cutoffs are frozen; the
        # operative score is the beta0-free composite plus the intercept's raw-scale
        # equivalent (beta for ttp/loyalty; beta*sd0 for wtp/risk_taking). The
        # min-max rescaled value then shifts by shift*(span-0.0001)/(max0-min0) and
        # the agent can cross bin boundaries (clipped at the end bins).
        raw0 = scores[m]                                     # beta0-free composite
        shift = _intercept_raw_shift(m, _intercept(params, m), pop[m])
        raw = raw0 + shift                                   # operative score
        vmin, vmax = pop[m]["min"], pop[m]["max"]            # beta0-free cutoffs
        stoch_m = mech_stoch[m]
        sigma = _resolve_sigma(stoch_m, agent_state, m) if use_stochastic else 0.0

        if m == "ttp":
            # ttp06 rescales the operative score against the beta0-free min/max, so
            # a nonzero beta0 shifts it by beta*(6-0.0001)/(max0-min0) (it may leave
            # [0, 6); the clip below caps the length at the 0/5 boundary bins).
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
            continue

        det_seg = _segment(raw, m, vmin, vmax)
        # z-score of the operative score over the BETA0-FREE population (reporting;
        # and the risk_taking continuous anchor). With raw = raw0 + beta*sd0 this is
        # exactly z0 + beta for wtp/risk_taking - the doc's standardized-scale
        # intercept (lines 1338/2431); for loyalty (beta on the raw scale) it is
        # z0 + beta/sd0. Matches the .dta's z_WTP/z_RT columns at beta = 0.
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
        key = {"loyalty": "loyalty", "wtp": "wtp", "risk_taking": "rt"}[m]
        # rtd_*_score is the OPERATIVE score: for ttp/loyalty the composite + beta0
        # (beta0 lives on the raw scale); for wtp/risk_taking the composite +
        # beta*sd0 (the raw-scale equivalent of the standardized-scale intercept,
        # which appears as exactly +beta on rtd_*_z). At beta = 0 it is the raw
        # composite, bit-identical to the pre-intercept behaviour.
        out[f"rtd_{key}_score"] = raw
        out[f"rtd_{key}_z"] = float(z_val)
        out[f"rtd_{key}_segment_deterministic"] = det_seg
        out[f"rtd_{key}_segment"] = final_seg
        out[f"rtd_{key}_ranking"] = ranking
        out[f"rtd_{key}_ranking_codes"] = [OPTION_CODES[o] for o in ranking]
        if draw_val is not None:
            out[f"rtd_{key}_draw"] = float(draw_val)
        out[f"rtd_sigma_used_{key}"] = float(sigma if drawn else 0.0)

    return out
