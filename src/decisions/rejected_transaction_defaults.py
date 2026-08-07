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

For each ranking mechanism the operative score is min-max rescaled into five
EQUAL-WIDTH segments 1..5 (floor(1 + (5-0.0001)*u), doc lines 615/1350/2449; note the
document's "20% of observations" narrative is wrong - the Stata code and .dta implement
equal-width bins, with e.g. 60% of participants in the lowest WTP bin) and segment s
receives the TAIL of the mechanism's priority sequence starting at position s:
  segment 1 -> all five options, segment 5 -> only the last option.
This mapping direction is verified 100% against the .dta (choice1..5_* columns) even
though it INVERTS the document's prose ("the 20% highest ... will opt for Option 3
first"); per project rule the Stata file arbitrates. The inversion is flagged in the
design notes for the professor to resolve; do not "fix" it here without a new .dta.

DTA-verified deviations from the document's literal Stata listings (all confirmed
against Stata_File_Decision4_050626.dta):
  - Loyalty: the documented `egen weighted_loyalty = std(bs)` + beta0 step was NOT
    executed; the stored weighted_loyalty IS the raw composite (min/max
    -0.26831895/0.21461959 as the doc itself quotes at line 646). Affine-invariant
    for all downstream outputs, so we use the raw composite.
  - WTP: z_WTP_calculated = std(WTP_calculated) exactly, NO beta1 added (the doc's
    `replace` line is a typo). beta intercepts (beta0/beta1/beta2) are provably inert
    here anyway: every output depends on the score only through min-max rescaling,
    which is invariant to adding a constant - so no intercept parameter is exposed.
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


def compute_rtd_scores(agent_state: Dict[str, Any], params: Dict[str, Any],
                       simulation_config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Compute the four raw mechanism scores for one agent.

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
    wtp_calculated = (c_wtp["extraversion"] * z_E + c_wtp["agreeable"] * z_A
                      + c_wtp["income"] * z_I)
    rt_calculated = (c_rt["extraversion"] * z_E + c_rt["openness"] * z_O
                     + c_rt["agreeable"] * z_A + c_rt["conscientiousness"] * z_C
                     + c_rt["neuroticism"] * z_N + c_rt["income"] * z_I)

    # Intercepts (doc: TTP beta0 on the composite, loyalty beta0 on the composite).
    # WTP beta1 / RT beta2 apply on the STANDARDIZED scale (doc lines 1338/2431) and are
    # added downstream where z is computed - adding them to the raw composite here would
    # be cancelled by the population standardization. All four intercepts shift the score
    # scale only: every allocation uses population min-max rescaling, which is invariant
    # to a constant shift.
    icpt = params.get("intercepts") or {}
    weighted_ttp += float(icpt.get("ttp", 0.0) or 0.0)
    weighted_loyalty += float(icpt.get("loyalty", 0.0) or 0.0)

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
    """Segment s (1..5) -> the tail of the mechanism's priority sequence from position s."""
    seq = PRIORITY_SEQUENCES[mechanism]
    return list(seq[segment - 1:])


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
                  pop: Dict, intercept: float = 0.0) -> float:
    """
    Anchor of the stochastic draw.

    'continuous' (default): the mechanism's continuous operative score - raw composite
    for ttp/loyalty/wtp, the population-z score (+ beta2 intercept) for risk_taking
    (whose sigma is derived from the z-score range). This is the .dta-verified
    behaviour for WTP and the doc's stated behaviour for TTP.
    'binned': the deterministic 1-5 segment (doc-literal reading for loyalty/RT).
    Note ttp's continuous anchor is the 0-6 rescaled weighted_ttp06 (doc line 191).
    """
    anchor_mode = stoch_m.get("anchor", "continuous")
    if anchor_mode == "binned" and mechanism != "ttp":
        return float(det_segment)
    if mechanism == "ttp":
        return _rescale(raw, 0.0, 6.0, pop["ttp"]["min"], pop["ttp"]["max"])
    if mechanism == "risk_taking":
        sd = pop["risk_taking"].get("sd", 1.0) or 1.0
        return (raw - pop["risk_taking"].get("mean", 0.0)) / sd + float(intercept)
    return raw   # loyalty, wtp: raw composite (dta-verified for wtp)


def compute_rtd_population_stats(agents_df, all_incomes: List[float], params: Dict,
                                 simulation_config: Dict, pop_context: str = "documentation",
                                 agent_base_seeds: Optional[List[int]] = None,
                                 decision_offset: Optional[int] = None) -> Dict[str, Dict]:
    """
    Population-level statistics for Decision 4, called from the orchestrators after
    Pass 1 (incomes known). Returns, per mechanism:
      {'min','max'}          of the deterministic operative score,
      {'mean','sd'}          of the raw composite (wtp/risk_taking, for z reporting),
      {'s_min','s_max'}      of the STOCHASTIC draws across the population (only when
                             stochastic is active and agent_base_seeds are provided).

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
                icpt_m = float((params.get("intercepts") or {}).get(m, 0.0) or 0.0)
                anchor = _anchor_value(m, stoch_m, raws[m][i], det_seg, pop, intercept=icpt_m)
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
    out["rtd_z_extraversion"] = scores["z_extraversion"]
    out["rtd_z_agreeable"] = scores["z_agreeable"]
    out["rtd_z_neuroticism"] = scores["z_neuroticism"]
    out["rtd_z_conscientiousness"] = scores["z_conscientiousness"]
    out["rtd_z_openness"] = scores["z_openness"]
    out["rtd_z_income"] = scores["z_income"]
    out["rtd_reducation"] = scores["reducation"]

    for m in MECHANISMS:
        raw = scores[m]
        vmin, vmax = pop[m]["min"], pop[m]["max"]
        stoch_m = mech_stoch[m]
        sigma = _resolve_sigma(stoch_m, agent_state, m) if use_stochastic else 0.0

        if m == "ttp":
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
        # z-score of the composite over the current population (reporting; and the
        # risk_taking continuous anchor). Matches the .dta's z_WTP/z_RT columns.
        # WTP beta1 / RT beta2 intercepts apply on this standardized scale (doc
        # lines 1338/2431); allocations are min-max-invariant to them.
        sd = pop[m].get("sd", 1.0) or 1.0
        icpt_m = float((params.get("intercepts") or {}).get(m, 0.0) or 0.0)
        z_val = (raw - pop[m].get("mean", 0.0)) / sd + (icpt_m if m in ("wtp", "risk_taking") else 0.0)

        final_seg = det_seg
        draw_val = None
        drawn = use_stochastic and sigma > 0 and "s_min" in pop[m]
        if drawn:
            anchor = _anchor_value(m, stoch_m, raw, det_seg, pop, intercept=icpt_m)
            draw_val = anchor + sigma * noise[m]
            low, span = _RESCALE[m]
            rescaled = _rescale(draw_val, low, span, pop[m]["s_min"], pop[m]["s_max"])
            final_seg = int(np.clip(int(np.floor(rescaled)), 1, 5))

        ranking = _ranking_for_segment(m, final_seg)
        key = {"loyalty": "loyalty", "wtp": "wtp", "risk_taking": "rt"}[m]
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
