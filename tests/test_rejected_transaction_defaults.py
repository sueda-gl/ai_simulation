"""
Validation tests for Decision 4: Rejected Transaction Defaults.

Ground truth: data/stata_d4_verification.csv - a frozen extract of the professor's
`Stata_File_Decision4_050626.dta` (280 participants) holding the raw inputs, the
Stata z-scores, and every derived Decision-4 column (TTP, Loyalty, WTP incl.
stochastic, RT).

All four deterministic mechanisms must reproduce 280/280 EXACTLY (bin/choice level;
float32 tolerance on continuous scores). The WTP stochastic PIPELINE (rescale +
floor + choice mapping keyed on sWTP_calculated15) is validated on the professor's
stored draws; the draws themselves are RNG-dependent and not reproducible.

Also tested: the population-stats hook's bit-for-bit RNG replication of the
per-agent decision function's draws, sigma-slider semantics (scale 0 == deterministic),
and seeded reproducibility.
"""
import os
import numpy as np
import pandas as pd
import yaml
import pytest

from src.decisions.rejected_transaction_defaults import (
    MECHANISMS, OPTION_CODES, PRIORITY_SEQUENCES,
    compute_rtd_scores, compute_rtd_population_stats, rejected_transaction_defaults,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFICATION_CSV = os.path.join(REPO, "data", "stata_d4_verification.csv")
DECISIONS_YAML = os.path.join(REPO, "config", "decisions.yaml")

# float32 storage tolerance for continuous scores
ATOL = 5e-6

# .dta ground-truth distributions (verification agents, 2026-08-05)
EXPECTED_LENGTH_DIST = {0: 20, 1: 92, 2: 95, 3: 57, 4: 14, 5: 2}
EXPECTED_LOYALTY_DIST = {1: 17, 2: 46, 3: 86, 4: 101, 5: 30}
EXPECTED_WTP_DIST = {1: 168, 2: 82, 3: 26, 4: 3, 5: 1}
EXPECTED_RT_DIST = {1: 20, 2: 100, 3: 110, 4: 46, 5: 4}


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
def _check_ranking_mechanism(gold, all_scores, mech, score_key, gold_score_col,
                             gold_seg_col, gold_choice_prefix, gold_choice_suffix,
                             expected_dist):
    scores = np.array([s[score_key] for s in all_scores])
    np.testing.assert_allclose(scores, gold[gold_score_col], atol=ATOL)

    vmin, vmax = scores.min(), scores.max()
    segs = np.floor(1 + (5 - 0.0001) * (scores - vmin) / (vmax - vmin)).astype(int)
    assert (segs == gold[gold_seg_col].astype(int)).all(), f"{mech} segments mismatch"
    assert pd.Series(segs).value_counts().to_dict() == expected_dist

    seq = PRIORITY_SEQUENCES[mech]
    for i, seg in enumerate(segs):
        expected_tail = seq[seg - 1:]
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
    _check_ranking_mechanism(gold, all_scores, "loyalty", "loyalty",
                             "weighted_loyalty", "weighted_loyalty15",
                             "choice", "_loyalty_deterministic", EXPECTED_LOYALTY_DIST)


def test_wtp_ranking(gold, all_scores):
    _check_ranking_mechanism(gold, all_scores, "wtp", "wtp",
                             "WTP_calculated", "WTP_calculated15",
                             "choice", "_WTP_deterministic", EXPECTED_WTP_DIST)
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

    seq = PRIORITY_SEQUENCES["wtp"]
    for i, seg in enumerate(segs):
        tail = seq[seg - 1:]
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
    sigma = 0.45265807275
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
               pop_context="documentation", scale=1.0, seed=1234):
    import copy
    p = copy.deepcopy(model_params)
    p.setdefault("stochastic", {})
    p["stochastic"]["sigma_value"] = 1.0 if stochastic else 0
    p["stochastic"]["in_copula"] = in_copula
    for m in MECHANISMS:
        p["stochastic"].setdefault("mechanisms", {}).setdefault(m, {})["scale_factor"] = scale

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
    results = _run_model(gold, model_params, sim_config, stochastic=False)
    lengths = [r["rtd_choice_length"] for r in results]
    assert (np.array(lengths) == gold["choice_length_deterministic"].astype(int)).all()
    for mech, key, seg_col in (("loyalty", "loyalty", "weighted_loyalty15"),
                               ("wtp", "wtp", "WTP_calculated15"),
                               ("risk_taking", "rt", "RT_calculated15")):
        segs = [r[f"rtd_{key}_segment"] for r in results]
        assert (np.array(segs) == gold[seg_col].astype(int)).all(), mech
        for i, r in enumerate(results):
            seg = r[f"rtd_{key}_segment"]
            assert r[f"rtd_{key}_ranking"] == PRIORITY_SEQUENCES[mech][seg - 1:]
            assert r[f"rtd_{key}_ranking_codes"] == \
                [OPTION_CODES[o] for o in r[f"rtd_{key}_ranking"]]
    # no aggregated list is fabricated
    assert all(r["rejected_transaction_defaults"] == [] for r in results)
    assert all(r["rtd_sigma_used_wtp"] == 0.0 for r in results)


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
    p_sigma = 0.45265807275
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


def test_intercepts_shift_scores_but_not_allocations(gold, model_params, sim_config):
    """Per-element intercepts (beta0/beta0/beta1/beta2) shift each element's score
    distribution but leave every allocation invariant (min-max rescaling)."""
    import copy
    base = _run_model(gold, model_params, sim_config, stochastic=False)

    p = copy.deepcopy(model_params)
    p['intercepts'] = {'ttp': 0.7, 'loyalty': -0.3, 'wtp': 1.5, 'risk_taking': -2.0}
    agents_df = pd.DataFrame([_agent_state(r) for _, r in gold.iterrows()])
    incomes = gold["income"].tolist()
    sim = dict(sim_config)
    sim["rtd_population_stats"] = compute_rtd_population_stats(
        agents_df, incomes, p, sim, pop_context="documentation")
    shifted = []
    for i, (_, r) in enumerate(gold.iterrows()):
        rng = np.random.default_rng(i)
        shifted.append(rejected_transaction_defaults(
            _agent_state(r), p, rng, sim, pop_context="documentation"))

    for b, s in zip(base, shifted):
        # scores shift by exactly the intercept
        np.testing.assert_allclose(s["rtd_weighted_ttp"] - b["rtd_weighted_ttp"], 0.7, rtol=1e-9)
        np.testing.assert_allclose(s["rtd_loyalty_score"] - b["rtd_loyalty_score"], -0.3, rtol=1e-9)
        np.testing.assert_allclose(s["rtd_wtp_z"] - b["rtd_wtp_z"], 1.5, rtol=1e-7)
        np.testing.assert_allclose(s["rtd_rt_z"] - b["rtd_rt_z"], -2.0, rtol=1e-7)
        # allocations are bit-identical
        assert s["rtd_choice_length"] == b["rtd_choice_length"]
        assert s["rtd_loyalty_segment"] == b["rtd_loyalty_segment"]
        assert s["rtd_wtp_segment"] == b["rtd_wtp_segment"]
        assert s["rtd_rt_segment"] == b["rtd_rt_segment"]
        assert s["rtd_loyalty_ranking"] == b["rtd_loyalty_ranking"]


def test_intercepts_invariant_under_stochastic(gold, model_params, sim_config):
    """With stochastic draws ON, intercepts still cannot move any allocation: the
    anchor and the population s_min/s_max shift by the same constant (hook and
    per-agent path must apply the intercept identically)."""
    import copy
    base = _run_model(gold, model_params, sim_config, stochastic=True, seed=321)

    p = copy.deepcopy(model_params)
    p['intercepts'] = {'ttp': 0.7, 'loyalty': -0.3, 'wtp': 1.5, 'risk_taking': -2.0}
    p.setdefault("stochastic", {})["sigma_value"] = 1.0
    p["stochastic"]["in_copula"] = False
    for m in MECHANISMS:
        p["stochastic"].setdefault("mechanisms", {}).setdefault(m, {})["scale_factor"] = 1.0

    agents_df = pd.DataFrame([_agent_state(r) for _, r in gold.iterrows()])
    incomes = gold["income"].tolist()
    rng_seeds = np.random.default_rng(321).integers(0, 1_000_000_000, len(gold))
    sim = dict(sim_config)
    sim["rtd_population_stats"] = compute_rtd_population_stats(
        agents_df, incomes, p, sim, pop_context="documentation",
        agent_base_seeds=list(rng_seeds), decision_offset=4000)
    shifted = []
    for i, (_, r) in enumerate(gold.iterrows()):
        rng = np.random.default_rng(int(rng_seeds[i]) + 4000)
        shifted.append(rejected_transaction_defaults(
            _agent_state(r), p, rng, sim, pop_context="documentation"))

    for b, s in zip(base, shifted):
        assert s["rtd_choice_length"] == b["rtd_choice_length"]
        assert s["rtd_loyalty_segment"] == b["rtd_loyalty_segment"]
        assert s["rtd_wtp_segment"] == b["rtd_wtp_segment"]
        assert s["rtd_rt_segment"] == b["rtd_rt_segment"]
        # RT draw itself shifts by exactly beta2 (anchor includes the intercept)
        np.testing.assert_allclose(s["rtd_rt_draw"] - b["rtd_rt_draw"], -2.0, rtol=1e-7)


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
