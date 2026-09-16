# src/decisions/rejected_transaction_defaults.py
"""
Decision 4: Rejected Transaction Defaults - trait-based sub-decision mechanisms plus
the Section-6 rank aggregation that integrates them into ONE default list per agent.

Source: "Decision 4 - Rejected Transaction Defaults" design document (latest revision
040926-2; its Stata code is the operative specification except where the professor's
written instructions supersede it, rank aggregation per its Section 6)
+ professor's Stata file `Stata_File_Decision4_050926.dta` (280 participants, 233
variables, the ground-truth arbiter wherever the document is ambiguous; frozen
extract data/stata_d4_verification.csv).

Re-verified coefficient-by-coefficient against rev 040926-2 / the 050926 file on
2026-09-16 (docs/d4_alignment_050926_2026-09-16.md): every score, z-score and
population statistic reproduces the .dta to <= 8.1e-6 (float32 storage), all five
segment/bin columns are 280/280, the 17-bin histograms (Stata's default bin rule,
k = round(min(sqrt(N), 10*log10(N))) = 17 at N = 280) of weighted_ttp /
weighted_loyalty / z_WTP_calculated / z_RT_calculated_hs /
z_Flexibility_calculated_ivw / z_anchored_flexibility / z_stdactionsP are
bin-for-bin identical to the file, and the choice1..5 lists now match the stored
columns DIRECTLY (the 050926 file adopted the flipped mapping direction; see the
mapping note below). The one professor-instructed change still not in the .dta is
the TTP beta0 scale (standardized since 2026-09-16), described below.

The decision is FIVE separate sub-decisions, each producing its own output per agent;
the four ranking mechanisms' lists are then reconciled into one integrated default
list by Kemeny-Young aggregation with the document's tie-breaking hierarchy (see
rtd_rank_aggregation.py and the RANK AGGREGATION section below):

  1. LIST LENGTH (Tendency to Plan, TTP)  -> how many default options to pre-select, 0-5
       weighted_ttp = -0.0152556564*z_E + 0.0177638642*z_A + 0.01959*z_N
                      + 0.00901465*z_C + 0.0297*reducation        (doc section 1)
       weighted_ttp06 = (6-0.0001) * minmax(weighted_ttp);  length = floor(weighted_ttp06)
  2. LOYALTY ranking   -> priority sequence Option 3 > 1 > 4 > 5 > 2
       bs_weighted_loyalty = -0.009828468*z_E + 0.01096706*z_O + 0.0123046*z_A
       weighted_loyalty = std(bs_weighted_loyalty) [+ beta1]        (doc section 2,
       rev 040926-2; the June equation 0.09/0.0273/0.0045 is superseded. The
       Agreeableness weight typo that produced 0.03092465 - w_A printed as 0.67139
       instead of 0.267139 - was corrected by the professor in rev 040926-2:
       0.267139 * 0.04606063 = 0.0123046, and the 050926 file embeds it)
  3. WILLINGNESS-TO-PAY ranking -> priority sequence Option 3 > 2 > 1 > 4 > 5
       WTP = 0.078863062*z_E - 0.012326128*z_A + 0.698*z_income    (doc section 3 code,
       Hunter & Schmidt pooling; the .dta still embeds -0.0123242131 for z_A - a
       uniform 2e-4 relative scale, so every segment is identical. Doc rev 040926-2
       revised the NARRATIVE WTP equation to 0.21784909*E - 0.03470828*A +
       0.0363565*I, but its operative Stata line and the 050926 file are both
       unchanged, so the code follows the Stata line - see the report)
  4. RISK-TAKING ranking -> priority sequence Option 4 > 2 > 1 > 3 > 5
       RT = 0.025942386297*z_E + 0.023699214948*z_O - 0.038734315188*z_A
            - 0.037739440732*z_C - 0.025388697852*z_N + 0.006874197106*z_income
       (Hunter & Schmidt pooling, doc section 4; .dta-exact)
  5. FLEXIBILITY ranking -> priority sequence Option 2 > 4 > 3 > 1 > 5
       (doc rev 040926-2 Section 5; pipeline verified 280/280 against
        Stata_File_Decision4_050926.dta, frozen extract
        data/stata_d4_flexibility_verification.csv)
       Flexibility_calculated_ivw = 0.0206*z_E + 0.0294118*z_O - 0.04921357*z_N
                                    + 0.04339814*z_A + 0.04811179*z_C     (IVW retained;
            the arithmetically corrected O/N/A values the app has carried since the
            professor's 2026-09 review were ADOPTED in rev 040926-2 and are what the
            050926 file now embeds, so the former DTA_FLEX_COEFFS distinction is gone
            and the file reproduces 280/280 with these coefficients)
       z_Flexibility = std(Flexibility_calculated_ivw) [+ beta4]           (egen std, population)
       anchored_flexibility = 0.25*z_stdactionsP + 0.75*z_Flexibility     (doc: observed
            flexibility - the participant's SD in the number of actions per cycle over the
            eight experiment cycle-weeks - gets a 25% weight; the doc's Stata line prints
            "0.2$5*$z_stdactions", the .dta embeds 0.25. The 050926 file renamed the
            variable z_stdactions -> z_stdactionsP; the values are unchanged,
            = egen std(stdactions))
       z_anchored_flexibility = std(anchored_flexibility);  Flexibility_combined15 =
            floor(1 + (5-0.0001)*minmax(z_anchored_flexibility))  -> segments 1..5
       stdactions comes from the
       professor's Stata file (data/stata_stdactions.csv, merged by Participant ID; it is a
       copula trait for synthetic populations). z_stdactions uses the frozen original-280
       stats like the other traits; z_Flexibility and z_anchored use POPULATION stats
       (egen std semantics) computed by compute_rtd_population_stats. The stochastic sigma
       (doc rev 040926-2: sigma_overall = 0.3663568222*mean(stdactions) = 0.4346228732,
       where 0.3663568222 = 6.5944228/18, the file's z_anchored range / 18) lives on the
       z_anchored scale, so the
       'continuous' anchor is z_anchored_flexibility (as for risk_taking); the doc-literal
       'binned' anchor (Flexibility_combined15) is available via stochastic.anchor.
       The .dta has no flexibility stochastic columns (no arbiter).

RANK AGGREGATION (doc rev 040926-2, Section 6; params['aggregation']):
The ranking mechanisms' priority lists are integrated into one consensus ranking of
the five options by Kemeny-Young aggregation (minimum total Kendall-tau distance,
exhaustive search over the 120 permutations, equal weights) with the document's
tie-breaking hierarchy - Schulze ordering when Kemeny returns several equally good
orderings, then Copeland, then Spearman footrule, then a RANDOM order of the
still-tied options (drawn from the agent's decision RNG AFTER the mechanisms' five
standard normals, so every per-mechanism output is unchanged by the aggregation).
Two output rules then apply: the list is truncated to the TTP choice length
(rtd_choice_length) and everything after Option 5 is dropped. The result is the
decision's main output column `rejected_transaction_defaults` (option codes) plus
rtd_default_list / rtd_consensus_* diagnostics.
  aggregation.enabled      (default True)
  aggregation.mechanisms   ranking mechanisms to reconcile (default: every ranking
                           mechanism the model computes - loyalty, wtp, risk_taking,
                           flexibility - the document's four inputs)
  aggregation.last_resort  'random' (doc) | 'lowest_option' (the V1 report's rule)
The mechanisms' lists are TAILS of their priority sequences (partial rankings); the
options absent from a list are treated as tied at the bottom of that list. The two
output rules (length truncation and the Option-5 cut) apply to the INTEGRATED
consensus list only - a mechanism's own list may legitimately still contain options
after Option 5 (e.g. Loyalty's segment-5 list 3 > 1 > 4 > 5 > 2 ends with Option 2).

CATEGORICAL INCOME (params['income_mode'] = 'categorical'; default 'continuous'):
Only WTP and Risk-Taking use income; TTP and Loyalty are identical in both modes.
The categorical variant (doc rev 040926-2 Section 3 / Section 4 cond() code)
replaces the continuous income term with fitted per-budget-level effects from an OLS
of the full score on the income-free personality part plus i.totalallowance dummies
(both regressions re-verified against the .dta, coefficients reproduce to <=1.2e-7):
  score_cat = intercept + level_dummy(assignedallowancelevel) + personality_part
where personality_part is the element's equation minus the income term applied at
coefficient 1.0 - DOC-LITERAL: the doc's operative cond() code applies it at slope 1;
the fitted slope on the personality part (0.5439353 WTP / 0.9991365 RT) is NOT
applied (flagged to the professor separately).
  WTP: intercept -0.691843; dummies +0.2671588/+0.5057716/+0.9411747/+1.822471
       (rev 040926-2 cond() code; levels 2-5 = EUR 32/72/128/200; level 1 = EUR 12
       gets the intercept only).
       NOTE the .dta's weighted_WTP_categorical is STILL built with raw 'agreeable' in
       the level-1 branch (rev 040926-2 fixed the last remaining "0.0123242128" typo in
       that code line, so the doc now says z_agreeable in all five branches, consistent
       with its own WTP_noincome construction, but the 050926 file was not regenerated);
       z_agreeable is used for ALL levels here, which moves one of the 280 participants
       across a segment boundary vs the file (279/280).
  RT:  intercept -0.0068307; dummies +0.0026128/+0.0050555/+0.0092738/+0.0179812.
Downstream (standardize -> min-max -> floor into 5 segments -> priority-sequence
tail mapping -> optional stochastic layer) is IDENTICAL with the categorical score
substituted (doc lines 1386/1435/2522: "replacing z_WTP_calculated with
z_WTP_categorical"); min-max rescaling is invariant to standardization and additive
intercepts, so binning the raw categorical score gives identical segments to binning
its standardized version. The professor has NOT yet specified categorical-specific
sigma values ("I have not checked that yet") - the continuous sigma config is reused
unchanged.

SEGMENT -> OPTION LIST MAPPING (flipped 2026-09-16 on the professor's written
instruction; ADOPTED by doc rev 040926-2 and by Stata_File_Decision4_050926.dta, so
the app and the arbiter now agree; see _ranking_for_segment):
For each ranking mechanism the operative score is min-max rescaled into five
EQUAL-WIDTH segments 1..5 (floor(1 + (5-0.0001)*u); note the document's "20% of
observations" narrative describes quantiles but the Stata code and .dta implement
equal-width bins, with e.g. 60% of participants in the lowest WTP bin) and segment s
receives the LAST s options of the mechanism's priority sequence, seq[5 - s:]:
  segment 5 (HIGHEST scores) -> all five options starting with the top option,
  segment 1 (LOWEST scores)  -> only the last option (list length 1).
Source: the professor's 2026-09-16 instruction - "under 'Cognitive Flexibility score
segment' replace the order so that 5 (Highest 20% ...) corresponds to 2>4>3>1>5, and
(Lowest 20% ...) corresponds to 5 with option list length 1. Correct all other
decision elements based on the updated Decision 4 document" - which matches the
document's prose ("the highest-Flexibility 20% segment, Flexibility_calculated15 ==
5, takes Option 2 as its first choice"; identically for loyalty/WTP/risk-taking).
Rev 040926-2 rewrote every `replace choice<p>_*` statement (deterministic AND
stochastic, all four elements) in this direction and the 050926 file's stored
choice1..5_* columns follow it, so the verification tests now assert the model's
lists EQUAL the stored columns directly, NaN pattern included (280/280 per element).
The superseded 290826 file embedded the opposite direction (segment 1 = full list).

DTA-verified notes (Stata_File_Decision4_050926.dta):
  - Loyalty: weighted_loyalty = std(bs_weighted_loyalty) is executed in the .dta
    (range -3.1419842..3.9363203 = 7.0783045); the segments are affine-invariant so
    the raw composite is binned here and the standardized score is reported/plotted.
    The stored min_/max_weighted_loyalty columns are correct in this file (they were
    swapped in the 290826 file).
  - WTP: z_WTP_calculated = std(WTP_calculated) exactly, NO beta added. The .dta
    embeds no intercepts anywhere; nonzero intercepts act through the FIXED-CUTOFF
    semantics described below (beta = 0 reproduces the .dta pipeline bit-for-bit).
  - WTP categorical: the .dta's level-1 branch still multiplies RAW `agreeable`
    instead of z_agreeable (rev 040926-2 fixed the doc's code line but the file was
    not regenerated); z_agreeable is used for ALL levels here, which moves one
    participant across a segment boundary vs the file (279/280). RT categorical is
    280/280.
  - Stochastic layer: the 050926 file carries NO stochastic columns. The June file's
    WTP draws (anchored on the RAW continuous WTP_calculated with ONE common sigma ~
    0.4526575455) remain the only stochastic ground truth and are kept in the frozen
    extract; every mechanism's stochastic anchor therefore defaults to its
    CONTINUOUS operative score (raw composite for ttp/wtp, population z for
    loyalty/risk_taking/flexibility whose sigmas are derived from the z ranges);
    the doc-literal binned anchor remains available via stochastic.anchor = 'binned'.
    NOTE the June file's stored choice*_WTP_calculated_stoc columns predate the
    direction flip, so they carry the OLD mapping; the DRAW -> sWTP_calculated15
    pipeline (which is what the test validates) is direction-independent.
  - Sigma constants follow sigma = (range/18) * mean(stdactions-within-group) with the
    documented values (rev 040926-2 tables, standardized scale). Rev 040926-2 fixed
    the two long-standing inconsistencies: the RT sigma now INCLUDES the
    *mean(stdactions) factor (0.332208167 * 1.1863376 = 0.39522204, quintile
    multiplier 0.332208167 rather than the old 0.28), and the loyalty / flexibility
    tables were recomputed from the 050926 file's own ranges. Every range the doc
    quotes was re-checked against the .dta on 2026-09-16 and reproduces:
    weighted_loyalty 7.0783043 (doc 7.0783045), z_RT 5.979747,
    z_anchored_flexibility 6.594423 (doc 6.5944228), z_WTP 6.868057 (doc 6.868056).
  - Flexibility needs each agent's `stdactions` (SD in the number of actions
    per cycle over the eight experiment cycle-weeks). The experiment workbook has no
    per-cycle counts, so it is taken from the professor's .dta (data/stata_stdactions.csv,
    merged into the participant table by Participant ID in src/validate_traits.py) and
    is a COPULA TRAIT (config/trait_requirements.yaml) so synthetic populations carry
    it with its correlations to the Big 5; an agent without it gets a neutral observed
    anchor (z = 0, flagged in rtd_flex_stdactions_missing).

INTERCEPTS (params['intercepts'][mechanism]) - FIXED-CUTOFF semantics, all five
intercepts on the STANDARDIZED scale of their own element (doc: `replace z_x = z_x +
beta`; TTP's beta0 switched from the raw composite scale to the standardized scale on
2026-09-16 - see _intercept_raw_shift for the reasoning and the flag). The
Flexibility beta4 is added to the standardized CALCULATED score before anchoring
(doc: `replace z_Flexibility_calculated_ivw = ... + beta4`), i.e. it shifts the
anchored score by 0.75*beta4. All population statistics (min/max/mean/sd
and the stochastic s_min/s_max) are computed on the BETA0-FREE scores, which freezes
the segment cutoffs; each agent's operative score is then the beta0-free composite
plus the intercept's raw-scale equivalent (beta*sd0 for
ttp/loyalty/wtp/risk_taking, so that (raw0 + beta*sd0 - mean0)/sd0 = z0 + beta;
0.75*beta4 for flexibility). The min-max
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
decision_index*1000), first five standard normals, flexibility last so the first
four are unchanged from the four-mechanism version) bit-for-bit; the per-agent
function then reproduces its own draw from the rng it is handed. A parity test
asserts the replication.

Verified against the .dta (tests/test_rejected_transaction_defaults.py): all five
deterministic mechanisms reproduce Stata_File_Decision4_050926.dta 280/280 exactly
(float32 tolerance), including the choice1..5 columns and their per-segment NaN
pattern, and the June WTP stochastic mapping keyed on sWTP_calculated15.

NAMING: the model's own output key stays `rtd_z_stdactions` (the app's column name);
it is the .dta's `z_stdactionsP` (renamed from `z_stdactions` in the 050926 file,
same values).
"""

import numpy as np
from typing import Any, Dict, List, Optional

from src.decisions.rtd_rank_aggregation import integrate_default_list

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
# segment s (1..5) receives the LAST s options, sequence[5-s:] (professor's 2026-09-16
# instruction / the document's prose; see _ranking_for_segment).
PRIORITY_SEQUENCES = {
    "loyalty": [3, 1, 4, 5, 2],       # doc line 609
    "wtp": [3, 2, 1, 4, 5],           # doc line 752/1343
    "risk_taking": [4, 2, 1, 3, 5],   # doc line 1856/2455
    "flexibility": [2, 4, 3, 1, 5],   # doc Section 5 ("Option 2 -> 4 -> 3 -> 1 -> 5")
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
LOYALTY_COEFFS = {                   # doc rev 040926-2 section 2 (bs_weighted_loyalty)
    "extraversion": -0.009828468,    # = 0.476815 * (-0.02061275)
    "openness": 0.01096706,          # = 0.256046 * 0.042832382
    "agreeable": 0.0123046,          # = 0.267139 * 0.04606063 (rev 040926-2 fixed the
                                     # w_A typo that printed 0.67139 -> 0.03092465)
}
WTP_COEFFS = {                       # Hunter & Schmidt pooling (doc section 3 code)
    "extraversion": 0.078863062,
    "agreeable": -0.012326128,
    "income": 0.698,
}
RT_COEFFS = {                        # Hunter & Schmidt pooling (doc line 2436)
    "extraversion": 0.025942386297,
    "openness": 0.023699214948,
    "agreeable": -0.038734315188,
    "conscientiousness": -0.037739440732,
    "neuroticism": -0.025388697852,
    "income": 0.006874197106,
}
FLEX_COEFFS = {                      # IVW pooling (doc rev 040926-2 Section 5 final equation)
    "extraversion": 0.0206,          # O/N/A: the arithmetically corrected values adopted in
    "openness": 0.0294118,           # the professor's 2026-09 review; rev 040926-2 wrote them
    "neuroticism": -0.04921357,      # into the document and Stata_File_Decision4_050926.dta
    "agreeable": 0.04339814,         # embeds them (Flexibility_calculated_ivw reproduces to
    "conscientiousness": 0.04811179, # <=1e-7, segments 280/280). The former DTA_FLEX_COEFFS
}                                    # (0.0293241 / -0.053781925 / 0.04921357) are obsolete.
# anchored_flexibility = observed_weight*z_stdactions + calculated_weight*z_Flexibility
# (doc: "we start by giving the observed flexibility a weight of 25%"; .dta-verified).
FLEX_ANCHOR_WEIGHTS = {"observed": 0.25, "calculated": 0.75}

# Categorical-income effects (doc rev 040926-2 regression output, Section 3 WTP /
# Section 4 RT; re-verified against the .dta). level_1 (EUR 12) is the base level:
# intercept only. Config override: params['categorical_income_effects'].
CATEGORICAL_INCOME_EFFECTS = {
    "wtp": {                     # doc rev 040926-2 section 3 cond() code
        "intercept": -0.691843,
        "level_2": 0.2671588,    # EUR 32
        "level_3": 0.5057716,    # EUR 72
        "level_4": 0.9411747,    # EUR 128
        "level_5": 1.822471,     # EUR 200
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
    # observed flexibility: SD in the number of actions per cycle (original 280; the doc's
    # mean 1.1863376; z_stdactionsP = egen std(stdactions), .dta-verified)
    "stdactions": {"mean": 1.1863375902, "sd": 0.6752842665},
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
SIGMA_OVERALL = {
    "ttp": 0.395446,          # (6/18)*1.186338              (doc TTP sigma table)
    "loyalty": 0.4665145336,  # 0.393239139*1.186338         (rev 040926-2, z scale)
    "wtp": 0.4526575455,      # 0.3815586667*1.186338        (rev 040926-2)
    "risk_taking": 0.39522204,    # rev 040926-2, doc-literal - see slip (a) below
    "flexibility": 0.4346228732,  # 0.3663568222*1.1863376   (rev 040926-2 Section 5)
}
# TWO ARITHMETIC SLIPS inside rev 040926-2's own sigma tables. Both are followed
# DOC-LITERALLY (the document's printed tables are the operative spec) and pinned by
# tests/test_rejected_transaction_defaults.py::
# test_sigma_tables_internal_consistency_and_the_two_doc_slips; both are flagged to the
# professor in docs/d4_alignment_050926_2026-09-16.md:
#   (a) RISK-TAKING overall. The doc writes "0.332208167 x 1.1863376 = 0.39522204";
#       that product is 0.39411104, so the printed result is 1.1e-3 (0.28%) too high.
#       The five RT quintile entries are consistent with the factor.
#   (b) LOYALTY quintiles. The doc states the factor 0.393239139 and its Total row
#       (0.4665145336) agrees, but all five per-allowance entries are the PREVIOUS
#       revision's column (factor 0.357334167) multiplied once more by 1.0228235,
#       implying a factor of 0.365489783. The overall value is consistent.
# Everything else (TTP / WTP / Flexibility, overall AND quintiles) satisfies
# sigma = factor * mean(stdactions within group) to <= 1e-6.

MECHANISMS = ("ttp", "loyalty", "wtp", "risk_taking", "flexibility")

# Ranking mechanisms (aggregation inputs, in the document's order) -> short key used
# in the rtd_* columns (rtd_<key>_ranking).
RANKING_KEYS = {"loyalty": "loyalty", "wtp": "wtp", "risk_taking": "rt", "flexibility": "flex"}
AGGREGATION_DEFAULTS = {"enabled": True, "mechanisms": None, "last_resort": "random"}

# Rescale targets: (low, span) so rescaled = low + (span - 0.0001) * u.
# ttp: 0..6 -> floor gives lengths 0..5; rankings: 1..5(.9999) -> floor gives 1..5.
_RESCALE = {
    "ttp": (0.0, 6.0),
    "loyalty": (1.0, 5.0),
    "wtp": (1.0, 5.0),
    "risk_taking": (1.0, 5.0),
    "flexibility": (1.0, 5.0),
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
    Compute the five raw mechanism scores for one agent - always BETA0-FREE.

    Intercepts are deliberately NOT applied here: this function also feeds
    compute_rtd_population_stats, whose statistics must be intercept-free so the
    segment cutoffs stay fixed (see the module docstring's FIXED-CUTOFF semantics).
    The per-agent decision function adds each intercept's raw-scale equivalent
    downstream.

    Trait z-scores use the frozen original-280 stats (Stata `egen std` semantics,
    same convention as the other modelled decisions). z_income is population-level:
    (income - income_stats.mean) / income_stats.sd from the orchestrator's Pass-1
    runtime stats, mirroring the disclose-decisions' continuous-income handling.

    The cognitive-flexibility operative score (anchored_flexibility) needs the
    population z-stats of the calculated score and of stdactions; it is returned as
    'flexibility' once rtd_population_stats['flexibility'] exists (None before the
    Pass-1 hook has run), alongside the population-free 'flexibility_calc' and the
    agent's 'stdactions'.
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
    c_flex = _coeffs(params, "flexibility", FLEX_COEFFS)

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
    cfg = params.get("flexibility_anchor") or {}
    w_obs = float(cfg.get("observed_weight", FLEX_ANCHOR_WEIGHTS["observed"]))
    w_calc = float(cfg.get("calculated_weight", FLEX_ANCHOR_WEIGHTS["calculated"]))
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

    DOCUMENT-PROSE direction (professor's written instruction of 2026-09-16, which the
    document's own prose states - e.g. Section 5: "the highest-Flexibility 20% segment,
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


def _intercept(params: Dict, mechanism: str) -> float:
    """The element's configured intercept (params['intercepts'][mechanism]; 0 absent)."""
    return float((params.get("intercepts") or {}).get(mechanism, 0.0) or 0.0)


def _intercept_raw_shift(mechanism: str, intercept: float, pop_m: Dict,
                         params: Optional[Dict] = None) -> float:
    """
    Raw-scale equivalent of the element's intercept (FIXED-CUTOFF semantics).

    ttp/loyalty/wtp/risk_taking: beta0/beta1/beta2/beta3 apply on the STANDARDIZED scale (doc:
    `replace weighted_loyalty = std(bs) + beta1`, `replace z_WTP = z_WTP + beta2`,
    `replace z_RT = z_RT + beta3`), z = (raw0 - mean0)/sd0 + beta with mean0/sd0 the
    BETA0-FREE population stats. The raw-scale equivalent is therefore beta * sd0:
      (raw0 + beta*sd0 - mean0)/sd0 = z0 + beta,
    and min-max rescaling against the beta0-free min0/max0 gives the identical bin
    whether performed on the shifted raw score or the shifted z-score (min-max is
    invariant to the affine raw->z map), so the raw representation is used
    throughout.

    TTP's beta0 uses the SAME standardized-scale semantics (decision of 2026-09-16,
    flagged to the professor). The document's TTP section writes `TTP_i = beta0 + ...`
    over z-scored traits without the "after standardization" qualifier that beta1-beta4
    carry, and the .dta embeds no intercept at all - so the document does not settle
    the scale. Applying the research default beta0 = 0.05 on the RAW weighted_ttp
    composite (the app's behaviour before 2026-09-16) is untenable empirically: that
    composite spans only ~0.21 on the 280 participants, so 0.05 is ~24% of the whole
    range = +1.43 list positions for EVERY agent, which wipes out all zero-length
    lists (exactly the professor's "I get no observations with 0 Option List Length")
    and moves the allocation far away from the Stata output. On the standardized
    scale beta0 = 0.05 is 0.05 SD ~ 0.0018 raw ~ +0.05 list positions - a small
    nudge that preserves the Stata allocation and keeps zero-length lists. All five
    elements therefore share ONE intercept semantics: beta is measured in standard
    deviations of the element's own beta0-free score.

    flexibility: beta4 applies on the standardized CALCULATED score BEFORE anchoring
    (doc: `replace z_Flexibility_calculated_ivw = z_Flexibility_calculated_ivw + beta4`),
    so the anchored (operative) score shifts by calculated_weight * beta4.
    """
    if intercept == 0.0:
        return 0.0
    if mechanism in ("ttp", "loyalty", "wtp", "risk_taking"):
        sd0 = pop_m.get("sd", 1.0) or 1.0
        return intercept * sd0
    if mechanism == "flexibility":
        return intercept * _flex_weights(params or {})[1]
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
    """params['aggregation'] merged over AGGREGATION_DEFAULTS (config/decisions.yaml)."""
    cfg = params.get("aggregation") or {}
    out = dict(AGGREGATION_DEFAULTS)
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
    inputs = {m: out.get(f"rtd_{RANKING_KEYS[m]}_ranking") for m in mechs
              if m in RANKING_KEYS and isinstance(out.get(f"rtd_{RANKING_KEYS[m]}_ranking"), list)}
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


def rejected_transaction_defaults(agent_state: Dict[str, Any], params: Dict[str, Any],
                                  rng, simulation_config: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> Dict[str, Any]:
    """
    Decision 4 simulation entry point.

    DEFAULT path (decision unselected, or model config absent): unchanged legacy
    behaviour - every agent receives the configured priority template.

    MODEL path (decision selected with the trait model): computes the five sub-decision
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
        # FIXED-CUTOFF intercept semantics: pop[m] holds BETA0-FREE statistics (the
        # hook never applies intercepts), so the segment cutoffs are frozen; the
        # operative score is the beta0-free composite plus the intercept's raw-scale
        # equivalent (beta*sd0 for ttp/loyalty/wtp/risk_taking, i.e. beta is measured
        # in SDs of the element's own score; 0.75*beta4 for flexibility). The
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
            # a nonzero beta0 shifts it by beta*sd0*(6-0.0001)/(max0-min0) (it may
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
            continue

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

    # ---- Section 6: integrate the ranking mechanisms into one default list ----
    _aggregate_mechanism_rankings(out, params, rng)

    return out
