"""
Validation tests for Decision 4: Rejected Transaction Defaults.

Ground truth: data/stata_d4_verification.csv - a frozen extract of the professor's
`Stata_File_Decision4_050926.dta` (280 participants) holding the raw inputs, the
Stata z-scores, and every derived Decision-4 column (TTP, Loyalty incl. the
standardized weighted_loyalty, WTP, RT, categorical scores, Flexibility) plus the
June file's WTP stochastic columns (the 050926 file carries none).

All deterministic mechanisms must reproduce 280/280 EXACTLY (bin/choice level;
float32 tolerance on continuous scores). Doc rev 040926-2 and the 050926 file ADOPTED
the segment -> option-list direction the professor instructed on 2026-09-16
(segment s -> seq[5-s:], segment 5 = the full list, segment 1 = one option), so the
stored choice1..5_* columns now EQUAL the model's lists directly - NaN pattern
included - for all four ranking elements. (The superseded 290826 file embedded the
mirror.) The WTP stochastic PIPELINE (rescale +
floor + choice mapping keyed on sWTP_calculated15) is validated on the professor's
stored draws; the draws themselves are RNG-dependent and not reproducible. NOTE the
June file predates the direction flip, so its stored choice*_WTP_calculated_stoc
columns still carry the OLD direction; the draw -> sWTP_calculated15 step, which is
what the pipeline test validates, is direction-independent.

Also tested: the population-stats hook's bit-for-bit RNG replication of the
per-agent decision function's draws, sigma-slider semantics (scale 0 == deterministic),
seeded reproducibility, the per-element intercepts' FIXED-CUTOFF semantics (beta=0
bit-identical; nonzero beta shifts scores, draws AND allocations across the
beta0-free-anchored cutoffs), and an AppTest end-to-end run through the app's
session-state plumbing.
"""
import os
import numpy as np
import pandas as pd
import yaml
import pytest

from src.decisions.rejected_transaction_defaults import (
    MEAN_STDACTIONS, MECHANISMS, OPTION_CODES, PRIORITY_SEQUENCES,
    SIGMA_FACTORS, SIGMA_OVERALL,
    compute_rtd_scores, compute_rtd_population_stats, rejected_transaction_defaults,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFICATION_CSV = os.path.join(REPO, "data", "stata_d4_verification.csv")
DECISIONS_YAML = os.path.join(REPO, "config", "decisions.yaml")

# float32 storage tolerance for continuous scores
ATOL = 5e-6

# .dta ground-truth distributions (Stata_File_Decision4_050926.dta)
EXPECTED_LENGTH_DIST = {0: 20, 1: 92, 2: 95, 3: 57, 4: 14, 5: 2}
# rev 040926-2 corrected the Loyalty Agreeableness weight (0.03092465 -> 0.0123046),
# which moved the loyalty bins from the 290826 file's 14/71/137/56/2
EXPECTED_LOYALTY_DIST = {1: 12, 2: 87, 3: 148, 4: 31, 5: 2}
EXPECTED_WTP_DIST = {1: 168, 2: 82, 3: 26, 4: 3, 5: 1}
EXPECTED_RT_DIST = {1: 20, 2: 100, 3: 110, 4: 46, 5: 4}
# rev 040926-2 adopted the app's corrected O/N/A IVW coefficients (was 17/83/141/35/4)
EXPECTED_FLEX_DIST = {1: 17, 2: 85, 3: 137, 4: 37, 5: 4}

# Stochastic sigmas printed by doc rev 040926-2 (see test_sigma_constants_match_the_document)
DOC_SIGMA_FACTORS = {
    "ttp": 6.0 / 18.0, "loyalty": 0.393239139, "wtp": 0.3815586667,
    "risk_taking": 0.332208167, "flexibility": 0.3663568222,
}
DOC_SIGMA_OVERALL = {
    "ttp": 0.395446, "loyalty": 0.4665145336, "wtp": 0.4526575455,
    "risk_taking": 0.39522204, "flexibility": 0.4346228732,
}
DOC_SIGMA_QUINTILES = {
    "ttp": {'1': 0.3409411667, '2': 0.3788993667, '3': 0.4368824333,
            '4': 0.4085647667, '5': 0.4181633667},
    "loyalty": {'1': 0.3738315395, '2': 0.4154515424, '3': 0.4790281978,
                '4': 0.4479787442, '5': 0.4585033149},
    "wtp": {'1': 0.3902671709, '2': 0.4337170115, '3': 0.5000888363,
            '4': 0.4676742829, '5': 0.4786615699},
    "risk_taking": {'1': 0.33979032, '2': 0.37762039, '3': 0.43540774,
                    '4': 0.40718566, '5': 0.41675186},
    "flexibility": {'1': 0.3747183671, '2': 0.4164371037, '3': 0.4801645799,
                    '4': 0.4490414678, '5': 0.4595910065},
}
# doc rev 040926-2 score ranges, all re-derived from the 050926 file below
DOC_RANGES = {"loyalty": 7.0783045, "wtp": 6.868056,
              "risk_taking": 5.979747, "flexibility": 6.5944228}
WTP_SIGMA = DOC_SIGMA_OVERALL["wtp"]   # the sigma the stochastic tests pin
# Research default beta0 = 0.05 on the STANDARDIZED weighted_ttp scale (2026-09-16
# semantics): a 0.05-SD nudge, so the allocation stays close to the Stata output and
# zero-length lists survive. (On the OLD raw-composite scale beta0 = 0.05 gave
# {0:0, 1:6, 2:59, 3:95, 4:86, 5:34}, mean 3.30 - the professor's "no observations
# with 0 Option List Length".)
EXPECTED_LENGTH_DIST_BETA0_005 = {0: 17, 1: 88, 2: 97, 3: 59, 4: 17, 5: 2}


@pytest.fixture(scope="module")
def gold():
    return pd.read_csv(VERIFICATION_CSV)


@pytest.fixture(scope="module")
def params():
    with open(DECISIONS_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg["rejected_transaction_defaults"]


@pytest.fixture(scope="module")
def sim_config(gold):
    # egen std semantics: sample SD (ddof=1) over the professor's income column
    return {
        "income_stats": {
            "mean": float(gold["income"].mean()),
            "sd": float(gold["income"].std(ddof=1)),
        }
    }


def _agent_state(row) -> dict:
    return {
        "ExtraversionBig5": row["extraversionbig5"],
        "Agreeable": row["agreeable"],
        "NeuroticismBig5": row["neuroticismbig5"],
        "ConscientiousnessBig5": row["conscientiousnessbig5"],
        "OpennessBig5": row["opennessbig5"],
        "Education": row["education"],
        "Assigned Allowance Level": row["assignedallowancelevel"],
        "income": row["income"],
        "stdactions": row["stdactions"],   # Flexibility observed anchor
    }


@pytest.fixture(scope="module")
def all_scores(gold, params, sim_config):
    return [compute_rtd_scores(_agent_state(r), params, sim_config)
            for _, r in gold.iterrows()]


# ---------------------------------------------------------------------------
# Input parity: our z-scores == Stata z-scores
# ---------------------------------------------------------------------------
def test_z_scores_match_stata(gold, all_scores):
    pairs = [
        ("z_extraversion", "z_extraversionbig5"),
        ("z_agreeable", "z_agreeable"),
        ("z_neuroticism", "z_neuroticismbig5"),
        ("z_conscientiousness", "z_conscientiousnessbig5"),
        ("z_openness", "z_opennessbig5"),
        ("z_income", "z_net_income"),
    ]
    for ours, stata in pairs:
        computed = np.array([s[ours] for s in all_scores])
        # frozen YAML stats are 7-digit roundings of the exact 280-sample stats,
        # so allow a slightly looser tolerance than raw float32 storage
        np.testing.assert_allclose(computed, gold[stata], atol=5e-5,
                                   err_msg=f"{ours} != {stata}")


def test_reducation(gold, all_scores):
    computed = np.array([s["reducation"] for s in all_scores])
    np.testing.assert_array_equal(computed, gold["reducation"])


# ---------------------------------------------------------------------------
# Mechanism 1: TTP list length
# ---------------------------------------------------------------------------
def test_ttp_scores(gold, all_scores):
    computed = np.array([s["ttp"] for s in all_scores])
    np.testing.assert_allclose(computed, gold["weighted_ttp"], atol=ATOL)


def test_ttp_choice_length_exact(gold, all_scores):
    scores = np.array([s["ttp"] for s in all_scores])
    vmin, vmax = scores.min(), scores.max()
    np.testing.assert_allclose([vmin, vmax],
                               [gold["min_weighted_ttp"].iloc[0], gold["max_weighted_ttp"].iloc[0]],
                               atol=ATOL)
    ttp06 = (6 - 0.0001) * (scores - vmin) / (vmax - vmin)
    np.testing.assert_allclose(ttp06, gold["weighted_ttp06"], atol=5e-4)
    lengths = np.floor(ttp06).astype(int)
    assert (lengths == gold["choice_length_deterministic"].astype(int)).all(), \
        "choice_length_deterministic mismatch"
    dist = pd.Series(lengths).value_counts().to_dict()
    assert dist == EXPECTED_LENGTH_DIST


# ---------------------------------------------------------------------------
# Ranking mechanisms: shared checks
# ---------------------------------------------------------------------------
def _stored_lists_by_segment(gold, choice_suffix, seg_col):
    """{segment: the option list the .dta stores for it} from choice1..5<suffix>.

    Since the 050926 file / doc rev 040926-2 the stored columns follow the model's own
    direction (segment s -> seq[5-s:]), so a model list for segment s is checked
    against this table's entry for segment s DIRECTLY. Every segment present in the
    file must have one consistent stored list, and the positions beyond the list
    length must be NaN, contiguously."""
    out = {}
    for _, row in gold.iterrows():
        seg = int(row[seg_col])
        vals = [row[f"choice{pos}{choice_suffix}"] for pos in range(1, 6)]
        lst = [int(v) for v in vals if not pd.isna(v)]
        assert out.setdefault(seg, lst) == lst, f"inconsistent stored list for segment {seg}"
        assert [pd.isna(v) for v in vals] == [False] * len(lst) + [True] * (5 - len(lst)), \
            f"non-contiguous NaN pattern for segment {seg}"
    assert set(out) == {1, 2, 3, 4, 5}, f"missing segments in the .dta: {sorted(out)}"
    return out


def _check_ranking_mechanism(gold, all_scores, mech, score_key, gold_score_col,
                             gold_seg_col, gold_choice_prefix, gold_choice_suffix,
                             expected_dist, atol=ATOL):
    scores = np.array([s[score_key] for s in all_scores])
    np.testing.assert_allclose(scores, gold[gold_score_col], atol=atol)

    vmin, vmax = scores.min(), scores.max()
    segs = np.floor(1 + (5 - 0.0001) * (scores - vmin) / (vmax - vmin)).astype(int)
    assert (segs == gold[gold_seg_col].astype(int)).all(), f"{mech} segments mismatch"
    assert pd.Series(segs).value_counts().to_dict() == expected_dist

    # The stored .dta choice columns follow the MODEL's direction (segment s -> the
    # LAST s options, seq[5-s:]) since doc rev 040926-2 / the 050926 file, so they are
    # asserted directly here and again through the model path in
    # test_model_path_deterministic_matches_stata.
    seq = PRIORITY_SEQUENCES[mech]
    for i, seg in enumerate(segs):
        expected_tail = seq[5 - seg:]   # 040926-2 direction, as stored in the .dta
        for pos in range(1, 6):
            col = f"{gold_choice_prefix}{pos}{gold_choice_suffix}"
            stata_val = gold[col].iloc[i]
            if pos <= len(expected_tail):
                assert not pd.isna(stata_val), f"{mech} row {i} choice{pos}: expected value, got NaN"
                assert int(stata_val) == expected_tail[pos - 1], \
                    f"{mech} row {i} choice{pos}: {stata_val} != {expected_tail[pos - 1]}"
            else:
                assert pd.isna(stata_val), f"{mech} row {i} choice{pos}: expected NaN"


def test_loyalty_ranking(gold, all_scores):
    # the raw composite is bs_weighted_loyalty; weighted_loyalty = std(bs) (+ beta1)
    _check_ranking_mechanism(gold, all_scores, "loyalty", "loyalty",
                             "bs_weighted_loyalty", "weighted_loyalty15",
                             "choice", "_loyalty_deterministic", EXPECTED_LOYALTY_DIST)
    scores = np.array([s["loyalty"] for s in all_scores])
    z = (scores - scores.mean()) / scores.std(ddof=1)
    np.testing.assert_allclose(z, gold["weighted_loyalty"], atol=5e-5)


def test_wtp_ranking(gold, all_scores):
    # 1e-5: the .dta's WTP_calculated carries float32 rounding of the income z-score
    _check_ranking_mechanism(gold, all_scores, "wtp", "wtp",
                             "WTP_calculated", "WTP_calculated15",
                             "choice", "_WTP_deterministic", EXPECTED_WTP_DIST, atol=1e-5)
    # z_WTP_calculated = std(WTP_calculated), no beta added (.dta-verified)
    scores = np.array([s["wtp"] for s in all_scores])
    z = (scores - scores.mean()) / scores.std(ddof=1)
    np.testing.assert_allclose(z, gold["z_WTP_calculated"], atol=5e-5)


def test_rt_ranking(gold, all_scores):
    _check_ranking_mechanism(gold, all_scores, "risk_taking", "risk_taking",
                             "RT_calculated_hs", "RT_calculated15",
                             "choice", "_RT_deterministic", EXPECTED_RT_DIST)
    scores = np.array([s["risk_taking"] for s in all_scores])
    z = (scores - scores.mean()) / scores.std(ddof=1)
    np.testing.assert_allclose(z, gold["z_RT_calculated_hs"], atol=5e-5)


# ---------------------------------------------------------------------------
# Full intermediate-column parity against Stata_File_Decision4_050926.dta
# ---------------------------------------------------------------------------
def test_every_intermediate_column_matches_the_dta(gold, model_params, sim_config):
    """Every derived column the .dta stores is reproduced by the model path at beta = 0,
    to float32 storage tolerance (the frozen z-scoring constants in the YAML are 7-digit
    roundings of the exact 280-sample statistics, hence 5e-5 on the standardized scores).
    Covers the whole 040926-2 pipeline end to end."""
    results = _run_model(gold, model_params, sim_config, stochastic=False,
                         intercepts={m: 0.0 for m in MECHANISMS})
    pairs = [
        # (model key, .dta column, tolerance)
        ("rtd_weighted_ttp", "weighted_ttp", ATOL),
        ("rtd_weighted_ttp06", "weighted_ttp06", 5e-4),
        ("rtd_choice_length", "choice_length_deterministic", 0),
        ("rtd_loyalty_score", "bs_weighted_loyalty", ATOL),
        ("rtd_loyalty_z", "weighted_loyalty", 5e-5),
        ("rtd_wtp_score", "WTP_calculated", 1e-5),
        ("rtd_wtp_z", "z_WTP_calculated", 5e-5),
        ("rtd_rt_score", "RT_calculated_hs", ATOL),
        ("rtd_rt_z", "z_RT_calculated_hs", 5e-5),
        ("rtd_flex_ivw", "Flexibility_calculated_ivw", ATOL),
        ("rtd_flex_z_ivw", "z_Flexibility_calculated_ivw", 5e-5),
        ("rtd_z_stdactions", "z_stdactionsP", 5e-5),   # renamed in the 050926 file
        ("rtd_flex_score", "anchored_flexibility", 5e-5),
        ("rtd_flex_z", "z_anchored_flexibility", 5e-5),
    ]
    for key, col, tol in pairs:
        got = np.array([r[key] for r in results], dtype=float)
        np.testing.assert_allclose(got, gold[col].to_numpy(dtype=float), atol=tol,
                                   err_msg=f"{key} != .dta {col}")
    # the five bin/segment columns, element by element
    for key, col in (("rtd_choice_length", "choice_length_deterministic"),
                     ("rtd_loyalty_segment", "weighted_loyalty15"),
                     ("rtd_wtp_segment", "WTP_calculated15"),
                     ("rtd_rt_segment", "RT_calculated15"),
                     ("rtd_flex_segment", "Flexibility_combined15")):
        got = np.array([r[key] for r in results], dtype=int)
        assert int((got == gold[col].to_numpy(dtype=int)).sum()) == len(gold) == 280, col


def test_categorical_columns_match_the_dta(gold, cat_model_params, sim_config):
    """weighted_WTP_categorical / weighted_RT_categorical. RT is 280/280 to float32
    tolerance. WTP reproduces to float32 tolerance for allowance levels 2-5; the file's
    level-1 branch still multiplies RAW `agreeable` where the app (and the doc's code
    line, fixed in rev 040926-2) uses z_agreeable, so those 12 rows differ by exactly
    the coefficient times (agreeable - z_agreeable) and one participant crosses a
    segment boundary (279/280)."""
    results = _run_model(gold, cat_model_params, sim_config, stochastic=False,
                         intercepts={m: 0.0 for m in MECHANISMS})
    lvl = gold["assignedallowancelevel"].to_numpy(dtype=int)

    rt = np.array([r["rtd_rt_score"] for r in results])
    np.testing.assert_allclose(rt, gold["weighted_RT_categorical"].to_numpy(), atol=5e-5)
    rt_segs = np.array([r["rtd_rt_segment"] for r in results])
    assert int((rt_segs == _cat_segments(gold["weighted_RT_categorical"].to_numpy())).sum()) == 280

    wtp = np.array([r["rtd_wtp_score"] for r in results])
    stored = gold["weighted_WTP_categorical"].to_numpy()
    hi = lvl != 1
    np.testing.assert_allclose(wtp[hi], stored[hi], atol=5e-5)
    # the level-1 gap is exactly the raw-vs-standardized Agreeableness substitution
    expected_gap = -0.012326128 * (gold["agreeable"].to_numpy() - gold["z_agreeable"].to_numpy())
    np.testing.assert_allclose((stored - wtp)[~hi], expected_gap[~hi], atol=5e-5)
    wtp_segs = np.array([r["rtd_wtp_segment"] for r in results])
    assert int((wtp_segs == _cat_segments(stored)).sum()) == 279


# ---------------------------------------------------------------------------
# Stochastic sigma constants (doc rev 040926-2 tables)
# ---------------------------------------------------------------------------
def test_sigma_constants_match_the_document(gold, params):
    """SIGMA_FACTORS / SIGMA_OVERALL and the YAML must carry the rev 040926-2 values
    VERBATIM, and each factor must be the .dta's own score range / 18."""
    assert SIGMA_FACTORS == DOC_SIGMA_FACTORS
    assert SIGMA_OVERALL == DOC_SIGMA_OVERALL
    for mech in MECHANISMS:
        cfg = params["stochastic"]["mechanisms"][mech]
        assert cfg["sigma_overall"] == DOC_SIGMA_OVERALL[mech], mech
        assert cfg["sigma_quintile"] == DOC_SIGMA_QUINTILES[mech], mech
    # every doc range is the .dta's own range (TTP's factor is the doc's 6/18 scaling)
    for mech, col in (("loyalty", "weighted_loyalty"), ("wtp", "z_WTP_calculated"),
                      ("risk_taking", "z_RT_calculated_hs"),
                      ("flexibility", "z_anchored_flexibility")):
        x = gold[col].to_numpy()
        rng = float(x.max() - x.min())
        assert abs(rng - DOC_RANGES[mech]) < 1e-5, (mech, rng)
        assert abs(SIGMA_FACTORS[mech] - rng / 18.0) < 1e-6, mech
    # mean(stdactions) constants are the file's own
    assert abs(MEAN_STDACTIONS["overall"] - gold["stdactions"].mean()) < 1e-6
    for q in range(1, 6):
        sub = gold.loc[gold["assignedallowancelevel"] == q, "stdactions"]
        assert abs(MEAN_STDACTIONS[q] - sub.mean()) < 1e-6, q


def test_sigma_tables_internal_consistency_and_the_two_doc_slips(params):
    """sigma = factor * mean(stdactions within group). TTP, WTP and Flexibility satisfy
    that identity for BOTH the overall value and all five quintiles; Risk-Taking and
    Loyalty each carry exactly one arithmetic slip in rev 040926-2, which is followed
    doc-literally (the document's printed tables are the operative spec) and pinned
    here so the deviation is recorded rather than silently inherited:

      * RISK-TAKING, overall: the doc writes "0.332208167 x 1.1863376 = 0.39522204",
        but that product is 0.39411104 - the printed result is 1.1e-3 (0.28%) too high.
        The five quintile entries ARE consistent with the factor 0.332208167.
      * LOYALTY, quintiles: the doc states the factor 0.393239139 and its Total row
        (0.4665145336) agrees, but all five per-allowance entries are the PREVIOUS
        revision's column (factor 0.357334167) multiplied once more by 1.0228235, i.e.
        they imply a factor of 0.365489783 instead of 0.393239139.
    """
    exact = ("ttp", "wtp", "flexibility")
    for mech in exact:
        assert abs(SIGMA_FACTORS[mech] * MEAN_STDACTIONS["overall"]
                   - DOC_SIGMA_OVERALL[mech]) < 1e-6, mech
        for q, val in DOC_SIGMA_QUINTILES[mech].items():
            assert abs(SIGMA_FACTORS[mech] * MEAN_STDACTIONS[int(q)] - val) < 1e-6, (mech, q)

    # Risk-Taking: quintiles consistent, overall off by the doc's own multiplication slip
    for q, val in DOC_SIGMA_QUINTILES["risk_taking"].items():
        assert abs(SIGMA_FACTORS["risk_taking"] * MEAN_STDACTIONS[int(q)] - val) < 1e-6, q
    recomputed_rt = SIGMA_FACTORS["risk_taking"] * MEAN_STDACTIONS["overall"]
    assert abs(recomputed_rt - 0.39411104) < 1e-8
    assert abs(DOC_SIGMA_OVERALL["risk_taking"] - recomputed_rt - 0.001111) < 1e-8

    # Loyalty: overall consistent, quintile column implies the superseded factor
    assert abs(SIGMA_FACTORS["loyalty"] * MEAN_STDACTIONS["overall"]
               - DOC_SIGMA_OVERALL["loyalty"]) < 1e-6
    implied = [DOC_SIGMA_QUINTILES["loyalty"][str(q)] / MEAN_STDACTIONS[q]
               for q in range(1, 6)]
    assert all(abs(f - 0.365489783) < 1e-8 for f in implied), implied
    assert abs(0.365489783 - 0.357334167 * MEAN_STDACTIONS[1]) < 1e-8


# ---------------------------------------------------------------------------
# WTP stochastic pipeline on the professor's stored draws
# ---------------------------------------------------------------------------
def test_wtp_stochastic_pipeline_on_stata_draws(gold):
    """Our rescale+floor+mapping reproduces sWTP_calculated15 and the stochastic
    choices from the professor's stored sWTP_calculated draws."""
    draws = gold["sWTP_calculated"].to_numpy()
    smin, smax = draws.min(), draws.max()
    np.testing.assert_allclose(
        [smin, smax],
        [gold["min_sWTP_calculated"].iloc[0], gold["max_sWTP_calculated"].iloc[0]],
        atol=ATOL)
    segs = np.floor(1 + (5 - 0.0001) * (draws - smin) / (smax - smin)).astype(int)
    assert (segs == gold["sWTP_calculated15"].astype(int)).all()

    # The stochastic columns come from the JUNE file, which predates the 2026-09-16
    # direction flip, so they still carry the OLD direction (segment s -> seq[s-1:]).
    # The model's own stochastic lists follow seq[5-s:], exactly like the deterministic
    # ones; rev 040926-2 rewrote the doc's `replace choice<p>_WTP_calculated_stoc`
    # statements to match, but no regenerated stochastic .dta exists yet.
    seq = PRIORITY_SEQUENCES["wtp"]
    for i, seg in enumerate(segs):
        tail = seq[seg - 1:]   # OLD (June) Stata direction, as stored in that file
        for pos in range(1, 6):
            stata_val = gold[f"choice{pos}_WTP_calculated_stoc"].iloc[i]
            if pos <= len(tail):
                assert int(stata_val) == tail[pos - 1]
            else:
                assert pd.isna(stata_val)


def test_wtp_stochastic_anchor_is_raw_score(gold):
    """The professor's draws are anchored on the RAW WTP_calculated (not the binned
    1-5 score): residual mean ~0 and sd consistent with sigma_overall 0.45266."""
    resid = gold["sWTP_calculated"] - gold["WTP_calculated"]
    n = len(resid)
    sigma = WTP_SIGMA
    assert abs(resid.mean()) < 3 * sigma / np.sqrt(n)
    assert 0.9 * sigma < resid.std(ddof=1) < 1.1 * sigma
    resid_binned = gold["sWTP_calculated"] - gold["WTP_calculated15"]
    assert abs(resid_binned.mean()) > 1.0   # binned anchor decisively rejected


# ---------------------------------------------------------------------------
# Full simulation entry point (model path, deterministic)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_params(params):
    p = dict(params)
    p["model_enabled"] = True
    return p


def _run_model(gold, model_params, sim_config, stochastic=False, in_copula=False,
               pop_context="documentation", scale=1.0, seed=1234, intercepts=None):
    import copy
    p = copy.deepcopy(model_params)
    p.setdefault("stochastic", {})
    p["stochastic"]["sigma_value"] = 1.0 if stochastic else 0
    p["stochastic"]["in_copula"] = in_copula
    for m in MECHANISMS:
        p["stochastic"].setdefault("mechanisms", {}).setdefault(m, {})["scale_factor"] = scale
    if intercepts is not None:
        p["intercepts"] = dict(intercepts)

    agents_df = pd.DataFrame([_agent_state(r) for _, r in gold.iterrows()])
    incomes = gold["income"].tolist()
    rng_seeds = np.random.default_rng(seed).integers(0, 1_000_000_000, len(gold))
    offset = 4000   # decision_index * 1000 stand-in

    sim = dict(sim_config)
    sim["rtd_population_stats"] = compute_rtd_population_stats(
        agents_df, incomes, p, sim, pop_context=pop_context,
        agent_base_seeds=list(rng_seeds), decision_offset=offset)

    results = []
    for i, (_, r) in enumerate(gold.iterrows()):
        rng = np.random.default_rng(int(rng_seeds[i]) + offset)
        results.append(rejected_transaction_defaults(
            _agent_state(r), p, rng, sim, pop_context=pop_context))
    return results


def test_model_path_deterministic_matches_stata(gold, model_params, sim_config):
    # Explicit zero intercepts: the .dta embeds NO intercepts, while the config's
    # research default is now ttp beta0 = 0.05 (professor, 2026-08). Parity of the
    # mechanism arithmetic is therefore asserted at beta = 0 (bit-identical to a
    # no-intercepts run per test_zero_intercepts_bit_identical).
    results = _run_model(gold, model_params, sim_config, stochastic=False,
                         intercepts={m: 0.0 for m in MECHANISMS})
    lengths = [r["rtd_choice_length"] for r in results]
    assert (np.array(lengths) == gold["choice_length_deterministic"].astype(int)).all()
    for mech, key, seg_col, gold_choice_suffix, expected_dist in (
            ("loyalty", "loyalty", "weighted_loyalty15", "_loyalty_deterministic",
             EXPECTED_LOYALTY_DIST),
            ("wtp", "wtp", "WTP_calculated15", "_WTP_deterministic", EXPECTED_WTP_DIST),
            ("risk_taking", "rt", "RT_calculated15", "_RT_deterministic", EXPECTED_RT_DIST),
            ("flexibility", "flex", "Flexibility_combined15", "_flex_deterministic",
             EXPECTED_FLEX_DIST)):
        segs = [r[f"rtd_{key}_segment"] for r in results]
        assert (np.array(segs) == gold[seg_col].astype(int)).all(), mech
        assert pd.Series(segs).value_counts().to_dict() == expected_dist, mech
        seq = PRIORITY_SEQUENCES[mech]
        # The .dta's stored per-segment lists (040926-2 direction). The model's list for
        # segment s must EQUAL the stored list for segment s - no mirroring any more.
        stored_by_segment = _stored_lists_by_segment(gold, gold_choice_suffix, seg_col)
        for i, r in enumerate(results):
            seg = r[f"rtd_{key}_segment"]
            # 2026-09-16 direction: segment s -> the LAST s options, seq[5-s:]
            assert r[f"rtd_{key}_ranking"] == seq[5 - seg:]
            assert len(r[f"rtd_{key}_ranking"]) == seg
            assert r[f"rtd_{key}_ranking"] == stored_by_segment[seg], \
                f"{mech} segment {seg}: does not equal the .dta's stored list"
            # per-row parity including the NaN pattern beyond the list length
            for pos in range(1, 6):
                stata_val = gold[f"choice{pos}{gold_choice_suffix}"].iloc[i]
                if pos <= seg:
                    assert int(stata_val) == r[f"rtd_{key}_ranking"][pos - 1]
                else:
                    assert pd.isna(stata_val)
            assert r[f"rtd_{key}_ranking_codes"] == \
                [OPTION_CODES[o] for o in r[f"rtd_{key}_ranking"]]
    # the main column is the Section-6 integrated default list (option codes of
    # rtd_default_list, a prefix of the consensus ranking cut at the TTP length /
    # Option 5) - see tests/test_rtd_rank_aggregation.py for the aggregation itself
    for r in results:
        assert r["rejected_transaction_defaults"] == [OPTION_CODES[o] for o in r["rtd_default_list"]]
        assert r["rtd_default_list"] == r["rtd_consensus_ranking"][:len(r["rtd_default_list"])]
        assert len(r["rtd_default_list"]) <= r["rtd_choice_length"]
    assert all(r["rtd_sigma_used_wtp"] == 0.0 for r in results)


def test_first_choice_distribution_matches_stata_columns(gold, model_params, sim_config):
    """2026-09-16 direction, now also the .dta's: segment s takes seq[5-s] first, so
    the WTP first choices EQUAL the .dta's choice1_WTP_deterministic column - segment 1
    (168 people, the lowest WTP scores) takes Option 5 and only that option, segment 5
    (1 person) takes Option 3 first and carries the full sequence."""
    results = _run_model(gold, model_params, sim_config, stochastic=False)
    seq = PRIORITY_SEQUENCES["wtp"]
    first = [r["rtd_wtp_ranking"][0] for r in results]
    counts = pd.Series(first).value_counts().to_dict()
    # segment counts {1:168, 2:82, 3:26, 4:3, 5:1} -> first choice seq[5-s]
    assert counts == {5: 168, 4: 82, 1: 26, 2: 3, 3: 1}
    # exactly the stored column, element by element
    assert first == [int(v) for v in gold["choice1_WTP_deterministic"]]
    stored = _stored_lists_by_segment(gold, "_WTP_deterministic", "WTP_calculated15")
    assert first == [stored[r["rtd_wtp_segment"]][0] for r in results]
    # segment 5 (not 1) carries the full sequence; segment 1 carries one option
    assert all(r["rtd_wtp_ranking"] == seq for r in results if r["rtd_wtp_segment"] == 5)
    assert all(r["rtd_wtp_ranking"] == [seq[-1]] for r in results if r["rtd_wtp_segment"] == 1)


def test_segment_to_option_mapping_direction(gold, model_params, sim_config):
    """The 2026-09-16 mapping rule for ALL four ranking elements: segment s receives
    the LAST s options of the priority sequence. Segment 5 (highest construct scores)
    gets the full sequence starting with the top option; segment 1 (lowest) gets only
    the final option, list length 1 - the professor's instruction, verbatim for
    Flexibility ("5 (Highest 20% ...) corresponds to 2>4>3>1>5, and (Lowest 20% ...)
    corresponds to 5 with option list length 1")."""
    from src.decisions.rejected_transaction_defaults import _ranking_for_segment
    for mech, seq in PRIORITY_SEQUENCES.items():
        for seg in range(1, 6):
            assert _ranking_for_segment(mech, seg) == seq[5 - seg:]
            assert len(_ranking_for_segment(mech, seg)) == seg
        assert _ranking_for_segment(mech, 5) == seq          # full list, top option first
        assert _ranking_for_segment(mech, 1) == [seq[-1]]    # one option
    assert _ranking_for_segment("flexibility", 5) == [2, 4, 3, 1, 5]
    assert _ranking_for_segment("flexibility", 1) == [5]

    # end to end on the 280: every mechanism's emitted list follows the rule and its
    # length equals its segment
    results = _run_model(gold, model_params, sim_config, stochastic=False)
    for r in results:
        for mech, key in (("loyalty", "loyalty"), ("wtp", "wtp"),
                          ("risk_taking", "rt"), ("flexibility", "flex")):
            seg = r[f"rtd_{key}_segment"]
            assert r[f"rtd_{key}_ranking"] == PRIORITY_SEQUENCES[mech][5 - seg:]
            assert len(r[f"rtd_{key}_ranking"]) == seg


def test_stochastic_rng_replication_and_reproducibility(gold, model_params, sim_config):
    """The population hook's replicated draws must equal the per-agent function's own
    draws (same seeds), and a re-run with the same seed must be identical."""
    res1 = _run_model(gold, model_params, sim_config, stochastic=True, seed=99)
    res2 = _run_model(gold, model_params, sim_config, stochastic=True, seed=99)
    for a, b in zip(res1, res2):
        assert a["rtd_wtp_segment"] == b["rtd_wtp_segment"]
        assert a["rtd_choice_length"] == b["rtd_choice_length"]
        np.testing.assert_allclose(a["rtd_wtp_draw"], b["rtd_wtp_draw"])

    # draws recomputed here exactly as the hook does -> must match rtd_*_draw
    p_sigma = WTP_SIGMA
    rng_seeds = np.random.default_rng(99).integers(0, 1_000_000_000, len(gold))
    for i, r in enumerate(res1):
        rng = np.random.default_rng(int(rng_seeds[i]) + 4000)
        noise = rng.standard_normal(4)
        expected_wtp_draw = r["rtd_wtp_score"] + p_sigma * noise[2]  # wtp is 3rd mechanism
        np.testing.assert_allclose(r["rtd_wtp_draw"], expected_wtp_draw, rtol=1e-12)

    # stochastic on but scale 0 -> degenerates to deterministic
    res0 = _run_model(gold, model_params, sim_config, stochastic=True, scale=0.0)
    det = _run_model(gold, model_params, sim_config, stochastic=False)
    for a, b in zip(res0, det):
        assert a["rtd_choice_length"] == b["rtd_choice_length"]
        assert a["rtd_loyalty_segment"] == b["rtd_loyalty_segment"]
        assert a["rtd_wtp_segment"] == b["rtd_wtp_segment"]
        assert a["rtd_rt_segment"] == b["rtd_rt_segment"]


def test_stochastic_moves_some_segments(gold, model_params, sim_config):
    det = _run_model(gold, model_params, sim_config, stochastic=False)
    sto = _run_model(gold, model_params, sim_config, stochastic=True, seed=7)
    moved = sum(a["rtd_wtp_segment"] != b["rtd_wtp_segment"] for a, b in zip(sto, det))
    assert moved > 0, "stochastic WTP should move at least some agents across segments"
    assert all(s["rtd_sigma_used_wtp"] > 0 for s in sto)


def test_baseline_mode_never_stochastic(gold, model_params, sim_config):
    sto = _run_model(gold, model_params, sim_config, stochastic=True,
                     pop_context="baseline")
    det = _run_model(gold, model_params, sim_config, stochastic=False)
    for a, b in zip(sto, det):
        assert a["rtd_wtp_segment"] == b["rtd_wtp_segment"]
        assert a["rtd_sigma_used_wtp"] == 0.0


# ---------------------------------------------------------------------------
# Per-element intercepts: FIXED-CUTOFF semantics (professor's request that beta0
# observably shift the allocation). All population statistics (min/max/mean/sd,
# s_min/s_max) are anchored on the BETA0-FREE scores, freezing the segment
# cutoffs; the per-agent operative score adds the intercept's raw-scale
# equivalent afterward (beta for ttp; beta*sd0 for loyalty/wtp/risk_taking,
# whose intercepts live on the standardized scale; 0.75*beta4 for flexibility),
# so agents genuinely cross
# the fixed bin boundaries. beta = 0 must be bit-identical to no intercepts.
# ---------------------------------------------------------------------------
INTERCEPTS_MIXED = {'ttp': 0.7, 'loyalty': -0.3, 'wtp': 1.5, 'risk_taking': -2.0}


def test_research_default_intercepts(params):
    """Config research defaults: all five intercepts 0 (TTP beta0 was 0.05 from the
    professor's 2026-08 request until the 2026-09-17 review set it to 0)."""
    assert params["intercepts"] == {
        "ttp": 0.0, "loyalty": 0.0, "wtp": 0.0, "risk_taking": 0.0, "flexibility": 0.0}


def test_ttp_intercept_is_on_the_standardized_scale(gold, model_params, sim_config):
    """2026-09-16 decision (flagged to the professor): beta0 applies on the
    STANDARDIZED weighted_ttp scale, like beta1/beta2/beta3 - the raw operative score
    shifts by beta0 * sd0, NOT by beta0 itself."""
    zeros = {m: 0.0 for m in MECHANISMS}
    base = _run_model(gold, model_params, sim_config, stochastic=False, intercepts=zeros)
    sd0 = float(np.std([b["rtd_weighted_ttp"] for b in base], ddof=1))
    for beta in (0.05, -0.5, 2.0):
        shifted = _run_model(gold, model_params, sim_config, stochastic=False,
                             intercepts={**zeros, "ttp": beta})
        for b, s in zip(base, shifted):
            np.testing.assert_allclose(s["rtd_weighted_ttp"] - b["rtd_weighted_ttp"],
                                       beta * sd0, rtol=1e-9, atol=1e-15)
    # sd0 is small (~0.035 on the 280), so the raw shift for beta0 = 0.05 is ~0.0018 -
    # under 1% of the composite's ~0.21 range, not the ~24% the raw-scale reading gave
    assert 0.03 < sd0 < 0.04
    ttp = np.array([b["rtd_weighted_ttp"] for b in base])
    assert 0.05 * sd0 / (ttp.max() - ttp.min()) < 0.01


def test_research_default_ttp_intercept_keeps_zero_length_lists(gold, model_params, sim_config):
    """Professor's check ("I get no observations with 0 Option List Length" /
    "check that the resulting distribution is similar to the Stata output"): with the
    research default beta0 = 0.05 the 280 participants keep zero-length option lists
    and the whole allocation stays within a few agents of the Stata output."""
    zeros = {m: 0.0 for m in MECHANISMS}
    res = _run_model(gold, model_params, sim_config, stochastic=False,
                     intercepts={**zeros, "ttp": 0.05})
    lengths = np.array([r["rtd_choice_length"] for r in res])
    dist = {k: int((lengths == k).sum()) for k in range(6)}
    assert dist == EXPECTED_LENGTH_DIST_BETA0_005
    assert dist[0] > 0, "beta0 = 0.05 must not wipe out the zero-length option lists"
    # close to the Stata allocation: at most 5 agents move in any bin, and the mean
    # list length moves by less than 0.1 options
    stata = {k: EXPECTED_LENGTH_DIST.get(k, 0) for k in range(6)}
    assert all(abs(dist[k] - stata[k]) <= 5 for k in range(6)), (dist, stata)
    stata_mean = sum(k * v for k, v in stata.items()) / sum(stata.values())
    assert abs(float(lengths.mean()) - stata_mean) < 0.1
    # and beta0 = 0 still reproduces the Stata allocation exactly
    base = _run_model(gold, model_params, sim_config, stochastic=False, intercepts=zeros)
    base_dist = pd.Series([r["rtd_choice_length"] for r in base]).value_counts().to_dict()
    assert base_dist == EXPECTED_LENGTH_DIST


def test_zero_intercepts_bit_identical(gold, model_params, sim_config):
    """beta = 0 (explicit zeros) reproduces a no-intercepts-key run bit-for-bit:
    every output - scores, z, segments, rankings, lengths, stochastic draws."""
    import copy
    no_key = copy.deepcopy(model_params)
    no_key.pop("intercepts", None)
    zeros = {m: 0.0 for m in MECHANISMS}
    for stoch in (False, True):
        base = _run_model(gold, no_key, sim_config, stochastic=stoch, seed=99)
        zero = _run_model(gold, no_key, sim_config, stochastic=stoch, seed=99,
                          intercepts=zeros)
        for b, z in zip(base, zero):
            assert set(b.keys()) == set(z.keys())
            for k in b:
                assert b[k] == z[k], f"{k} not identical at beta=0 (stochastic={stoch})"


def test_intercepts_shift_scores_on_natural_scale(gold, model_params, sim_config):
    """Every intercept shifts its element's STANDARDIZED score by exactly the entered
    value (2026-09-16: ttp's beta0 joined loyalty/wtp/risk_taking on the standardized
    scale), so the raw operative score shifts by beta * sd0."""
    # Intercept-free baseline (the YAML research default is now ttp beta0 = 0.05,
    # which would otherwise contaminate the exact-shift assertions below).
    zeros = {m: 0.0 for m in MECHANISMS}
    base = _run_model(gold, model_params, sim_config, stochastic=False,
                      intercepts=zeros)
    shifted = _run_model(gold, model_params, sim_config, stochastic=False,
                         intercepts=INTERCEPTS_MIXED)
    ttp_sd0 = float(np.std([b["rtd_weighted_ttp"] for b in base], ddof=1))
    loy_sd0 = float(np.std([b["rtd_loyalty_score"] for b in base], ddof=1))
    wtp_sd0 = float(np.std([b["rtd_wtp_score"] for b in base], ddof=1))
    rt_sd0 = float(np.std([b["rtd_rt_score"] for b in base], ddof=1))
    for b, s in zip(base, shifted):
        # ttp: raw composite shifts by beta0 * sd0, i.e. the standardized score by beta0
        np.testing.assert_allclose(s["rtd_weighted_ttp"] - b["rtd_weighted_ttp"],
                                   0.7 * ttp_sd0, rtol=1e-9)
        np.testing.assert_allclose(s["rtd_loyalty_z"] - b["rtd_loyalty_z"], -0.3, rtol=1e-7)
        np.testing.assert_allclose(s["rtd_loyalty_score"] - b["rtd_loyalty_score"],
                                   -0.3 * loy_sd0, rtol=1e-6)
        np.testing.assert_allclose(s["rtd_wtp_z"] - b["rtd_wtp_z"], 1.5, rtol=1e-7)
        np.testing.assert_allclose(s["rtd_rt_z"] - b["rtd_rt_z"], -2.0, rtol=1e-7)
        # raw-scale equivalents of the standardized-scale intercepts
        np.testing.assert_allclose(s["rtd_wtp_score"] - b["rtd_wtp_score"],
                                   1.5 * wtp_sd0, rtol=1e-6)
        np.testing.assert_allclose(s["rtd_rt_score"] - b["rtd_rt_score"],
                                   -2.0 * rt_sd0, rtol=1e-6)


def test_negative_ttp_intercept_shortens_option_lists(gold, model_params, sim_config):
    """The professor's use case: a negative TTP beta0 visibly shortens the Options
    lists - every agent's length is <= its baseline, some strictly lower, and the
    mean strictly decreases. beta0 is now in SDs of weighted_ttp (2026-09-16), so
    -0.5 SD is used here (on the old raw scale -0.05 was 24% of the whole range)."""
    base = _run_model(gold, model_params, sim_config, stochastic=False)
    shifted = _run_model(gold, model_params, sim_config, stochastic=False,
                         intercepts={'ttp': -0.5})
    b_len = np.array([r["rtd_choice_length"] for r in base])
    s_len = np.array([r["rtd_choice_length"] for r in shifted])
    assert (s_len <= b_len).all()
    assert (s_len < b_len).any()
    assert s_len.mean() < b_len.mean()
    # only TTP was touched: the other three mechanisms are unchanged
    for b, s in zip(base, shifted):
        assert s["rtd_loyalty_segment"] == b["rtd_loyalty_segment"]
        assert s["rtd_wtp_segment"] == b["rtd_wtp_segment"]
        assert s["rtd_rt_segment"] == b["rtd_rt_segment"]


def test_positive_loyalty_intercept_shifts_segments_up(gold, model_params, sim_config):
    """A positive loyalty beta1 (standardized scale) shifts the segments weakly
    upward (some strictly), and the rankings follow the shifted segments' tails."""
    base = _run_model(gold, model_params, sim_config, stochastic=False)
    shifted = _run_model(gold, model_params, sim_config, stochastic=False,
                         intercepts={'loyalty': 0.3})
    b_seg = np.array([r["rtd_loyalty_segment"] for r in base])
    s_seg = np.array([r["rtd_loyalty_segment"] for r in shifted])
    assert (s_seg >= b_seg).all()
    assert (s_seg > b_seg).any()
    for r in shifted:
        seg = r["rtd_loyalty_segment"]
        assert r["rtd_loyalty_ranking"] == PRIORITY_SEQUENCES["loyalty"][5 - seg:]


def test_intercepts_shift_stochastic_draws_and_rebinning(gold, model_params, sim_config):
    """Stochastic mode, same seed: each mechanism's draw shifts by exactly the
    intercept in the anchor's units (s_min/s_max stay beta0-free) and the re-binned
    outcomes shift weakly in the intercept's direction; the hook's bit-for-bit RNG
    replication still reproduces the shifted draws."""
    seed = 321
    # Intercept-free baseline (YAML research default ttp beta0 = 0.05 would
    # otherwise offset the exact draw-shift assertions).
    base = _run_model(gold, model_params, sim_config, stochastic=True, seed=seed,
                      intercepts={m: 0.0 for m in MECHANISMS})
    shifted = _run_model(gold, model_params, sim_config, stochastic=True, seed=seed,
                         intercepts=INTERCEPTS_MIXED)

    ttp_scores = np.array([b["rtd_weighted_ttp"] for b in base])
    ttp_range0 = float(ttp_scores.max() - ttp_scores.min())
    ttp_sd0 = float(ttp_scores.std(ddof=1))
    wtp_sd0 = float(np.std([b["rtd_wtp_score"] for b in base], ddof=1))

    for b, s in zip(base, shifted):
        # draws shift by the intercept in the anchor's own units:
        # ttp06 units (beta0 * sd0 in raw units), raw loyalty/wtp units, rt z units
        np.testing.assert_allclose(s["rtd_ttp_draw"] - b["rtd_ttp_draw"],
                                   0.7 * ttp_sd0 * (6 - 0.0001) / ttp_range0, rtol=1e-6)
        np.testing.assert_allclose(s["rtd_loyalty_draw"] - b["rtd_loyalty_draw"],
                                   -0.3, rtol=1e-6)
        np.testing.assert_allclose(s["rtd_wtp_draw"] - b["rtd_wtp_draw"],
                                   1.5 * wtp_sd0, rtol=1e-6)
        np.testing.assert_allclose(s["rtd_rt_draw"] - b["rtd_rt_draw"],
                                   -2.0, rtol=1e-6)
        # re-binning shifts weakly in the intercept's direction
        assert s["rtd_choice_length"] >= b["rtd_choice_length"]
        assert s["rtd_loyalty_segment"] <= b["rtd_loyalty_segment"]
        assert s["rtd_wtp_segment"] >= b["rtd_wtp_segment"]
        assert s["rtd_rt_segment"] <= b["rtd_rt_segment"]
    # and some agents genuinely move
    assert any(s["rtd_wtp_segment"] != b["rtd_wtp_segment"] for b, s in zip(base, shifted))
    assert any(s["rtd_rt_segment"] != b["rtd_rt_segment"] for b, s in zip(base, shifted))

    # RNG replication with intercepts ON: the pinned per-agent noise stream (which
    # the population hook mirrors) reproduces the shifted draws - wtp anchored on
    # the operative score, rt on its z (z0 + beta2)
    rng_seeds = np.random.default_rng(seed).integers(0, 1_000_000_000, len(gold))
    for i, s in enumerate(shifted):
        noise = np.random.default_rng(int(rng_seeds[i]) + 4000).standard_normal(4)
        np.testing.assert_allclose(
            s["rtd_wtp_draw"], s["rtd_wtp_score"] + s["rtd_sigma_used_wtp"] * noise[2],
            rtol=1e-12)
        np.testing.assert_allclose(
            s["rtd_rt_draw"], s["rtd_rt_z"] + s["rtd_sigma_used_rt"] * noise[3],
            rtol=1e-12)


def test_extreme_intercepts_saturate_boundary_bins(gold, model_params, sim_config):
    """Extreme intercepts saturate every agent at the boundary bins without errors.
    +/-5 (the UI slider bound) fully saturates ttp (5 SD of weighted_ttp is ~5.0 list
    positions) and risk_taking; the standardized-scale shifts of loyalty (5 z-units vs
    a 6.43 range) and WTP (5*sd0 ~0.73 of its long-tailed range) do not, so full
    saturation is asserted at +/-8 for both (the model imposes no bound) and +/-5 is
    asserted to shift weakly and stay valid, incl. under stochastic. Rankings follow
    the 2026-09-16 direction (segment 5 -> the FULL sequence, segment 1 -> the last
    option only)."""
    base = _run_model(gold, model_params, sim_config, stochastic=False)

    hi = _run_model(gold, model_params, sim_config, stochastic=False,
                    intercepts={'ttp': 5.0, 'loyalty': 8.0, 'wtp': 8.0, 'risk_taking': 5.0})
    assert all(r["rtd_choice_length"] == 5 for r in hi)
    assert all(r["rtd_loyalty_segment"] == 5 for r in hi)
    assert all(r["rtd_wtp_segment"] == 5 for r in hi)
    assert all(r["rtd_rt_segment"] == 5 for r in hi)
    assert all(r["rtd_wtp_ranking"] == PRIORITY_SEQUENCES["wtp"] for r in hi)

    lo = _run_model(gold, model_params, sim_config, stochastic=False,
                    intercepts={'ttp': -5.0, 'loyalty': -8.0, 'wtp': -8.0, 'risk_taking': -5.0})
    assert all(r["rtd_choice_length"] == 0 for r in lo)
    assert all(r["rtd_loyalty_segment"] == 1 for r in lo)
    assert all(r["rtd_wtp_segment"] == 1 for r in lo)
    assert all(r["rtd_rt_segment"] == 1 for r in lo)
    assert all(r["rtd_loyalty_ranking"] == PRIORITY_SEQUENCES["loyalty"][4:] for r in lo)

    # WTP at the +/-5 UI bound: monotone shift (deterministic) and valid bins with
    # no errors under stochastic draws
    for beta, cmp in ((5.0, np.greater_equal), (-5.0, np.less_equal)):
        det = _run_model(gold, model_params, sim_config, stochastic=False,
                         intercepts={'wtp': beta})
        segs = np.array([r["rtd_wtp_segment"] for r in det])
        assert cmp(segs, np.array([b["rtd_wtp_segment"] for b in base])).all()
        sto = _run_model(gold, model_params, sim_config, stochastic=True, seed=11,
                         intercepts={'wtp': beta})
        assert set(r["rtd_wtp_segment"] for r in sto) <= {1, 2, 3, 4, 5}


def test_categorical_intercepts_apply_same_way(gold, cat_model_params, sim_config):
    """Categorical income mode: intercepts follow the identical fixed-cutoff
    semantics on the categorical scores - beta=0 bit-identical, z shifts by exactly
    beta, segments shift weakly in beta's direction (income-free elements and any
    element with beta=0 unchanged)."""
    # No-intercepts-key baseline: the YAML research default (ttp beta0 = 0.05)
    # must be excluded so that explicit zeros == no intercepts at all.
    import copy
    cat_no_key = copy.deepcopy(cat_model_params)
    cat_no_key.pop("intercepts", None)
    base = _run_model(gold, cat_no_key, sim_config, stochastic=False)
    zero = _run_model(gold, cat_no_key, sim_config, stochastic=False,
                      intercepts={m: 0.0 for m in MECHANISMS})
    for b, z in zip(base, zero):
        assert b == z

    shifted = _run_model(gold, cat_no_key, sim_config, stochastic=False,
                         intercepts={'wtp': 1.5, 'risk_taking': -2.0})
    for b, s in zip(base, shifted):
        np.testing.assert_allclose(s["rtd_wtp_z"] - b["rtd_wtp_z"], 1.5, rtol=1e-7)
        np.testing.assert_allclose(s["rtd_rt_z"] - b["rtd_rt_z"], -2.0, rtol=1e-7)
        assert s["rtd_wtp_segment"] >= b["rtd_wtp_segment"]
        assert s["rtd_rt_segment"] <= b["rtd_rt_segment"]
        assert s["rtd_choice_length"] == b["rtd_choice_length"]
        assert s["rtd_loyalty_segment"] == b["rtd_loyalty_segment"]
    assert any(s["rtd_wtp_segment"] != b["rtd_wtp_segment"] for b, s in zip(base, shifted))
    assert any(s["rtd_rt_segment"] != b["rtd_rt_segment"] for b, s in zip(base, shifted))


def test_default_template_path_unchanged(gold, model_params, sim_config):
    """Unselected decision keeps the legacy template behaviour."""
    sim = {"default_decisions_list": ["rejected_transaction_defaults"],
           "default_decisions": {"rejected_transaction_defaults": {
               "type": "prioritized_selection",
               "priority_template": ["current_vendor_pn", "forgo_transaction"]}}}
    rng = np.random.default_rng(0)
    out = rejected_transaction_defaults(_agent_state(gold.iloc[0]), model_params, rng, sim)
    assert out["rejected_transaction_defaults"] == ["current_vendor_pn", "forgo_transaction"]
    assert "rtd_wtp_segment" not in out


# ---------------------------------------------------------------------------
# CATEGORICAL INCOME (doc rev 040926-2 Section 3 WTP / Section 4 RT cond() code)
#
# Reference values are re-derived here INDEPENDENTLY from the doc's cond()
# construction (regression coefficients hardcoded from the doc's printed OLS
# output, which reproduces from the .dta to <=1.2e-7). z_agreeable is used for
# ALL levels: rev 040926-2 FIXED the doc's cond() line (all five branches now
# read z_agreeable, consistent with its own WTP_noincome construction), but the
# 050926 file was not regenerated and its level-1 branch still multiplies raw
# 'agreeable' - which moves one of the 280 participants across a bin (279/280).
# The personality part enters at coefficient 1.0 (doc-literal; the fitted slope
# 0.5439389/0.9991365 is NOT applied).
# ---------------------------------------------------------------------------
DOC_WTP_CAT_INTERCEPT = -0.691843          # doc rev 040926-2 cond() code
DOC_WTP_CAT_DUMMIES = {1: 0.0, 2: 0.2671588, 3: 0.5057716, 4: 0.9411747, 5: 1.822471}
DOC_RT_CAT_INTERCEPT = -0.0068307
DOC_RT_CAT_DUMMIES = {1: 0.0, 2: 0.0026128, 3: 0.0050555, 4: 0.0092738, 5: 0.0179812}

# Segment distributions of the 280 verification agents under the categorical
# scores (derived once from the frozen CSV via the doc construction). WTP
# segment 4 is GENUINELY EMPTY: the level-5 dummy (+1.822843) opens a gap in
# the score range that the equal-width bins map to an unpopulated bin.
EXPECTED_WTP_CAT_DIST = {1: 67, 2: 102, 3: 57, 5: 54}
EXPECTED_RT_CAT_DIST = {1: 21, 2: 103, 3: 109, 4: 43, 5: 4}


@pytest.fixture(scope="module")
def cat_params(params):
    p = dict(params)
    p["income_mode"] = "categorical"
    return p


@pytest.fixture(scope="module")
def cat_model_params(cat_params):
    p = dict(cat_params)
    p["model_enabled"] = True
    return p


@pytest.fixture(scope="module")
def cat_scores(gold, cat_params, sim_config):
    return [compute_rtd_scores(_agent_state(r), cat_params, sim_config)
            for _, r in gold.iterrows()]


def _doc_categorical_reference(gold):
    """weighted_WTP_categorical / weighted_RT_categorical per the doc's cond()
    construction, computed from the Stata z-columns of the frozen extract."""
    lvl = gold["assignedallowancelevel"].astype(int)
    ref_wtp = (DOC_WTP_CAT_INTERCEPT + lvl.map(DOC_WTP_CAT_DUMMIES)
               + 0.078863062 * gold["z_extraversionbig5"]
               - 0.012326128 * gold["z_agreeable"])
    ref_rt = (DOC_RT_CAT_INTERCEPT + lvl.map(DOC_RT_CAT_DUMMIES)
              + 0.025942386297 * gold["z_extraversionbig5"]
              + 0.023699214948 * gold["z_opennessbig5"]
              - 0.038734315188 * gold["z_agreeable"]
              - 0.037739440732 * gold["z_conscientiousnessbig5"]
              - 0.025388697852 * gold["z_neuroticismbig5"])
    return ref_wtp.to_numpy(), ref_rt.to_numpy()


def _cat_segments(scores):
    vmin, vmax = scores.min(), scores.max()
    return np.floor(1 + (5 - 0.0001) * (scores - vmin) / (vmax - vmin)).astype(int)


def test_categorical_wtp_matches_doc_construction(gold, cat_scores):
    ref_wtp, _ = _doc_categorical_reference(gold)
    computed = np.array([s["wtp"] for s in cat_scores])
    # frozen YAML z-stat roundings, same tolerance as test_z_scores_match_stata
    np.testing.assert_allclose(computed, ref_wtp, atol=5e-5)


def test_categorical_rt_matches_doc_construction(gold, cat_scores):
    _, ref_rt = _doc_categorical_reference(gold)
    computed = np.array([s["risk_taking"] for s in cat_scores])
    np.testing.assert_allclose(computed, ref_rt, atol=5e-5)


def test_categorical_segments_and_distributions(gold, cat_scores):
    ref_wtp, ref_rt = _doc_categorical_reference(gold)
    for key, ref, expected in (("wtp", ref_wtp, EXPECTED_WTP_CAT_DIST),
                               ("risk_taking", ref_rt, EXPECTED_RT_CAT_DIST)):
        computed = np.array([s[key] for s in cat_scores])
        segs = _cat_segments(computed)
        assert (segs == _cat_segments(ref)).all(), f"{key} categorical segments mismatch"
        assert pd.Series(segs).value_counts().to_dict() == expected
    # WTP segment 4 must be empty (level-5 dummy gap), all others populated
    wtp_segs = _cat_segments(np.array([s["wtp"] for s in cat_scores]))
    assert 4 not in set(wtp_segs)


def test_categorical_affine_invariance(cat_scores):
    """Binning the raw categorical score == binning its standardized version
    (+ any beta intercept): min-max rescaling is invariant to affine transforms."""
    for key, beta in (("wtp", 0.37), ("risk_taking", -1.2)):
        raw = np.array([s[key] for s in cat_scores])
        z = (raw - raw.mean()) / raw.std(ddof=1) + beta
        np.testing.assert_array_equal(_cat_segments(z), _cat_segments(raw),
                                      err_msg=f"{key} affine invariance violated")


def test_categorical_leaves_income_free_elements_unchanged(all_scores, cat_scores):
    """TTP and Loyalty use no income: bit-identical between the two modes."""
    for key in ("ttp", "loyalty"):
        cont = np.array([s[key] for s in all_scores])
        cat = np.array([s[key] for s in cat_scores])
        np.testing.assert_array_equal(cont, cat, err_msg=f"{key} changed by income_mode")


def test_continuous_mode_unchanged_by_explicit_key(gold, params, sim_config, all_scores):
    """income_mode='continuous' (explicit) == income_mode absent == gold columns."""
    p = dict(params)
    p["income_mode"] = "continuous"
    explicit = [compute_rtd_scores(_agent_state(r), p, sim_config)
                for _, r in gold.iterrows()]
    for key in ("ttp", "loyalty", "wtp", "risk_taking"):
        np.testing.assert_array_equal(np.array([s[key] for s in explicit]),
                                      np.array([s[key] for s in all_scores]),
                                      err_msg=f"{key} explicit-continuous mismatch")
    np.testing.assert_allclose(np.array([s["wtp"] for s in explicit]),
                               gold["WTP_calculated"], atol=1e-5)   # float32 income z
    np.testing.assert_allclose(np.array([s["risk_taking"] for s in explicit]),
                               gold["RT_calculated_hs"], atol=ATOL)


def test_categorical_model_path_deterministic(gold, cat_model_params, sim_config, cat_scores):
    """Full entry point in categorical mode: population hook and per-agent function
    operate on the SAME categorical scores; segments follow the expected dists;
    TTP/Loyalty outputs equal the continuous-mode run exactly."""
    results = _run_model(gold, cat_model_params, sim_config, stochastic=False)
    cont_results = _run_model(gold, {**cat_model_params, "income_mode": "continuous"},
                              sim_config, stochastic=False)

    raw_wtp = np.array([s["wtp"] for s in cat_scores])
    raw_rt = np.array([s["risk_taking"] for s in cat_scores])
    # hook stats == stats of the categorical scores (verified via segment identity)
    segs_wtp = [r["rtd_wtp_segment"] for r in results]
    segs_rt = [r["rtd_rt_segment"] for r in results]
    np.testing.assert_array_equal(segs_wtp, _cat_segments(raw_wtp))
    np.testing.assert_array_equal(segs_rt, _cat_segments(raw_rt))
    assert pd.Series(segs_wtp).value_counts().to_dict() == EXPECTED_WTP_CAT_DIST
    assert pd.Series(segs_rt).value_counts().to_dict() == EXPECTED_RT_CAT_DIST

    for r in results:
        assert r["rtd_income_mode"] == "categorical"
    for r in cont_results:
        assert r["rtd_income_mode"] == "continuous"

    # reported z == standardization of the categorical score over the population
    z_wtp = np.array([r["rtd_wtp_z"] for r in results])
    np.testing.assert_allclose(
        z_wtp, (raw_wtp - raw_wtp.mean()) / raw_wtp.std(ddof=1), atol=1e-9)

    # income-free elements identical between the two full-model runs
    for a, b in zip(results, cont_results):
        assert a["rtd_choice_length"] == b["rtd_choice_length"]
        assert a["rtd_loyalty_segment"] == b["rtd_loyalty_segment"]
        assert a["rtd_loyalty_ranking"] == b["rtd_loyalty_ranking"]
    # income-using elements must differ for some agents
    assert any(a["rtd_wtp_segment"] != b["rtd_wtp_segment"]
               for a, b in zip(results, cont_results))
    assert any(a["rtd_rt_segment"] != b["rtd_rt_segment"]
               for a, b in zip(results, cont_results))


def test_categorical_stochastic_rng_replication(gold, cat_model_params, sim_config):
    """Mirror of test_stochastic_rng_replication_and_reproducibility in categorical
    mode: the population hook's replicated draws (which set s_min/s_max) must be
    bit-identical to the per-agent function's own draws, anchored on the
    CATEGORICAL score; sigma config is reused unchanged (no categorical-specific
    sigma has been specified)."""
    res1 = _run_model(gold, cat_model_params, sim_config, stochastic=True, seed=99)
    res2 = _run_model(gold, cat_model_params, sim_config, stochastic=True, seed=99)
    for a, b in zip(res1, res2):
        assert a["rtd_wtp_segment"] == b["rtd_wtp_segment"]
        np.testing.assert_allclose(a["rtd_wtp_draw"], b["rtd_wtp_draw"])

    p_sigma = WTP_SIGMA   # continuous-spec sigma, reused unchanged
    rng_seeds = np.random.default_rng(99).integers(0, 1_000_000_000, len(gold))
    for i, r in enumerate(res1):
        rng = np.random.default_rng(int(rng_seeds[i]) + 4000)
        noise = rng.standard_normal(4)
        # anchor = raw CATEGORICAL wtp score (continuous-anchor semantics)
        expected_wtp_draw = r["rtd_wtp_score"] + p_sigma * noise[2]
        np.testing.assert_allclose(r["rtd_wtp_draw"], expected_wtp_draw, rtol=1e-12)
        assert r["rtd_sigma_used_wtp"] == p_sigma

    # stochastic on but scale 0 -> degenerates to the deterministic categorical run
    res0 = _run_model(gold, cat_model_params, sim_config, stochastic=True, scale=0.0)
    det = _run_model(gold, cat_model_params, sim_config, stochastic=False)
    for a, b in zip(res0, det):
        assert a["rtd_choice_length"] == b["rtd_choice_length"]
        assert a["rtd_wtp_segment"] == b["rtd_wtp_segment"]
        assert a["rtd_rt_segment"] == b["rtd_rt_segment"]


# ---------------------------------------------------------------------------
# End-to-end through the app plumbing (streamlit AppTest): the Decision-4 tab
# widget (key rtd_tab_intercept_ttp) writes st.session_state['rtd_intercept_ttp'],
# which app.simulation._apply_rejected_transaction_config copies into
# params['intercepts']['ttp'] for the orchestrator run - so setting the session
# key exercises the exact path a user takes through the tab.
# ---------------------------------------------------------------------------
def _apptest_rtd_script():
    import streamlit as st
    from src.orchestrator_baseline import OrchestratorBaseline
    from app.simulation import _apply_rejected_transaction_config

    orch = OrchestratorBaseline()
    _apply_rejected_transaction_config(orch, "baseline", "continuous")
    agents = orch.original_data.iloc[:80].copy()
    agents.index = range(len(agents))
    df = orch.run_simulation(len(agents), 123, ['rejected_transaction_defaults'],
                             agents_df=agents)
    st.session_state['result_lengths'] = [int(x) for x in df['rtd_choice_length']]


def test_apptest_negative_ttp_intercept_end_to_end():
    """AppTest end-to-end (professor's use case): rtd_intercept_ttp = -0.5 set via
    the session key the tab widget writes must change the simulated
    rtd_choice_length distribution and shift it lower. (-0.5 SD rather than the
    original -0.05: since the research default became 0 on 2026-09-17, a 0.05-SD
    nudge on the standardized scale moves no agent across a segment boundary.)"""
    from streamlit.testing.v1 import AppTest

    at0 = AppTest.from_function(_apptest_rtd_script)
    at0.run(timeout=300)
    assert not at0.exception
    base = at0.session_state['result_lengths']

    at1 = AppTest.from_function(_apptest_rtd_script)
    at1.session_state['rtd_intercept_ttp'] = -0.5
    at1.run(timeout=300)
    assert not at1.exception
    shifted = at1.session_state['result_lengths']

    assert shifted != base
    assert all(s <= b for s, b in zip(shifted, base))
    assert float(np.mean(shifted)) < float(np.mean(base))
