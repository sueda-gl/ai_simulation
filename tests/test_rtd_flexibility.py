"""
Validation tests for Decision 4, Section 5: the Cognitive Flexibility mechanism.

Ground truth: data/stata_d4_flexibility_verification.csv - a frozen extract of the
professor's `Stata_File_Decision4_290826.dta` (280 participants) holding the raw
inputs, stdactions and every Flexibility column (Flexibility_calculated_ivw, its z,
z_stdactions, anchored_flexibility, z_anchored_flexibility, min/max,
Flexibility_combined15, choice1-5_flex_deterministic).

The deterministic pipeline must reproduce the .dta 280/280 at the bin level and to
float32 storage tolerance on every continuous intermediate. The choice lists are
asserted directly against the stored columns (STATA direction: segment s -> tail
seq[s-1:], the project rule since 2026-09-04 that the Stata file arbitrates). Also tested: stdactions z-scoring
constants, beta4 fixed-cutoff semantics, the neutral fallback when stdactions is
absent, the stochastic layer's RNG replication, the copula trait, and the
aggregation now receiving the document's four inputs.
"""
import copy
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from src.decisions.rejected_transaction_defaults import (
    DEFAULT_Z_SCORING, FLEX_ANCHOR_WEIGHTS, FLEX_COEFFS, MECHANISMS, PRIORITY_SEQUENCES,
    SIGMA_OVERALL, SIGMA_FACTORS, MEAN_STDACTIONS,
    compute_rtd_population_stats, compute_rtd_scores, flex_anchored_score,
    rejected_transaction_defaults,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLEX_CSV = os.path.join(REPO, "data", "stata_d4_flexibility_verification.csv")
DECISIONS_YAML = os.path.join(REPO, "config", "decisions.yaml")

ATOL = 5e-6                      # float32 storage tolerance
EXPECTED_FLEX_DIST = {1: 17, 2: 83, 3: 141, 4: 35, 5: 4}
SEQ = [2, 4, 3, 1, 5]


@pytest.fixture(scope="module")
def flexgold():
    return pd.read_csv(FLEX_CSV)


@pytest.fixture(scope="module")
def params():
    with open(DECISIONS_YAML) as f:
        cfg = yaml.safe_load(f)
    p = dict(cfg["rejected_transaction_defaults"])
    p["model_enabled"] = True
    # the .dta embeds no intercepts
    p["intercepts"] = {m: 0.0 for m in MECHANISMS}
    return p


@pytest.fixture(scope="module")
def sim_config(flexgold):
    return {"income_stats": {"mean": float(flexgold["income"].mean()),
                             "sd": float(flexgold["income"].std(ddof=1))}}


def _agent_state(row, with_stdactions=True):
    s = {
        "ExtraversionBig5": row["extraversionbig5"], "Agreeable": row["agreeable"],
        "NeuroticismBig5": row["neuroticismbig5"], "ConscientiousnessBig5": row["conscientiousnessbig5"],
        "OpennessBig5": row["opennessbig5"], "Education": row["education"],
        "Assigned Allowance Level": row["assignedallowancelevel"], "income": row["income"],
    }
    if with_stdactions:
        s["stdactions"] = row["stdactions"]
    return s


def _run(flexgold, params, sim_config, stochastic=False, seed=1234, intercepts=None,
         with_stdactions=True, scale=1.0, anchor=None):
    p = copy.deepcopy(params)
    p.setdefault("stochastic", {})
    p["stochastic"]["sigma_value"] = 1.0 if stochastic else 0
    p["stochastic"]["in_copula"] = False
    for m in MECHANISMS:
        mcfg = p["stochastic"].setdefault("mechanisms", {}).setdefault(m, {})
        mcfg["scale_factor"] = scale
        if anchor is not None:
            mcfg["anchor"] = anchor
    if intercepts is not None:
        p["intercepts"] = dict(intercepts)
    agents_df = pd.DataFrame([_agent_state(r, with_stdactions) for _, r in flexgold.iterrows()])
    incomes = flexgold["income"].tolist()
    seeds = np.random.default_rng(seed).integers(0, 1_000_000_000, len(flexgold))
    offset = 4000
    sim = dict(sim_config)
    sim["rtd_population_stats"] = compute_rtd_population_stats(
        agents_df, incomes, p, sim, pop_context="documentation",
        agent_base_seeds=list(seeds), decision_offset=offset)
    out = []
    for i, (_, r) in enumerate(flexgold.iterrows()):
        rng = np.random.default_rng(int(seeds[i]) + offset)
        out.append(rejected_transaction_defaults(_agent_state(r, with_stdactions), p, rng, sim,
                                                 pop_context="documentation"))
    return out, sim


@pytest.fixture(scope="module")
def results(flexgold, params, sim_config):
    return _run(flexgold, params, sim_config)[0]


# ---------------------------------------------------------------------------
# Constants (doc Section 5 / .dta)
# ---------------------------------------------------------------------------
def test_constants_follow_the_document(params):
    assert FLEX_COEFFS == {"extraversion": 0.0206, "openness": 0.0293241,
                           "neuroticism": -0.053781925, "agreeable": 0.04921357,
                           "conscientiousness": 0.04811179}
    assert params["coefficients"]["flexibility"] == FLEX_COEFFS
    assert PRIORITY_SEQUENCES["flexibility"] == SEQ == params["priority_sequences"]["flexibility"]
    assert FLEX_ANCHOR_WEIGHTS == {"observed": 0.25, "calculated": 0.75}
    assert params["flexibility_anchor"] == {"observed_weight": 0.25, "calculated_weight": 0.75}
    assert SIGMA_OVERALL["flexibility"] == 0.4359172665
    # the doc multiplies rounded factors (0.367448 * 1.1863376), so ~1e-7 slack
    assert abs(SIGMA_FACTORS["flexibility"] * MEAN_STDACTIONS["overall"] - 0.4359172665) < 1e-6
    doc_quintiles = {'1': 0.3758343529, '2': 0.4176773361, '3': 0.4815946054,
                     '4': 0.4503788034, '5': 0.4609597598}
    assert params["stochastic"]["mechanisms"]["flexibility"]["sigma_quintile"] == doc_quintiles
    for lvl, val in doc_quintiles.items():
        assert abs(SIGMA_FACTORS["flexibility"] * MEAN_STDACTIONS[int(lvl)] - val) < 1e-6
    assert params["intercepts"]["flexibility"] == 0.0


def test_stdactions_z_scoring_uses_the_original_280_stats(flexgold, params):
    z = params["z_scoring"]["stdactions"]
    assert abs(z["mean"] - flexgold["stdactions"].mean()) < 1e-7
    assert abs(z["sd"] - flexgold["stdactions"].std(ddof=1)) < 1e-7
    assert DEFAULT_Z_SCORING["stdactions"] == z


# ---------------------------------------------------------------------------
# Deterministic pipeline vs the .dta
# ---------------------------------------------------------------------------
def test_flexibility_intermediates_match_stata(flexgold, results):
    for key, col in (("rtd_flex_ivw", "Flexibility_calculated_ivw"),
                     ("rtd_flex_z_ivw", "z_Flexibility_calculated_ivw"),
                     ("rtd_z_stdactions", "z_stdactions"),
                     ("rtd_flex_score", "anchored_flexibility"),
                     ("rtd_flex_z", "z_anchored_flexibility")):
        got = np.array([r[key] for r in results])
        assert np.allclose(got, flexgold[col].values, atol=ATOL), key


def test_flexibility_population_min_max_match_stata(flexgold, params, sim_config):
    _, sim = _run(flexgold, params, sim_config)
    pop = sim["rtd_population_stats"]["flexibility"]
    # binning is on z_anchored: min/max of the anchored score standardised over the 280
    sd = pop["sd"]
    z_min = (pop["min"] - pop["mean"]) / sd
    z_max = (pop["max"] - pop["mean"]) / sd
    assert abs(z_min - flexgold["min_Flexibility_combined"].dropna().iloc[0]) < ATOL
    assert abs(z_max - flexgold["max_Flexibility_combined"].dropna().iloc[0]) < ATOL
    assert sim["rtd_population_stats"]["flexibility_ivw"]["sd"] > 0


def test_flexibility_segments_exact(flexgold, results):
    segs = np.array([r["rtd_flex_segment"] for r in results])
    assert (segs == flexgold["Flexibility_combined15"].astype(int).values).all()
    assert (np.array([r["rtd_flex_segment_deterministic"] for r in results]) == segs).all()
    dist = pd.Series(segs).value_counts().to_dict()
    assert dist == EXPECTED_FLEX_DIST


def test_flexibility_choice_lists_match_stata_columns(flexgold, results):
    """STATA direction: segment s -> SEQ[s-1:] (segment 1 takes Option 2 first, segment
    5 only Option 5), identical to the stored choice1-5_flex_deterministic columns."""
    choice_cols = [f"choice{i}_flex_deterministic" for i in range(1, 6)]
    for r, (_, g) in zip(results, flexgold.iterrows()):
        s = int(g["Flexibility_combined15"])
        stored = [int(v) for v in g[choice_cols].values if not pd.isna(v)]
        assert r["rtd_flex_ranking"] == SEQ[s - 1:] == stored
        assert len(stored) == 6 - s
    firsts = pd.Series([r["rtd_flex_ranking"][0] for r in results]).value_counts().to_dict()
    # segment counts 17/83/141/35/4 -> first options 2/4/3/1/5
    assert firsts == {2: 17, 4: 83, 3: 141, 1: 35, 5: 4}
    assert [r["rtd_flex_ranking"][0] for r in results] == \
        flexgold["choice1_flex_deterministic"].astype(int).tolist()


def test_other_mechanisms_unaffected_by_flexibility(flexgold, results):
    """TTP and Risk-Taking still reproduce the .dta bins (their inputs are unchanged);
    the flexibility computation only appends outputs."""
    # the flexibility extract carries no TTP/RT columns; check against the June extract
    gold = pd.read_csv(os.path.join(REPO, "data", "stata_d4_verification.csv"))
    assert (gold["participantid"].values == flexgold["participantid"].values).all()
    assert (np.array([r["rtd_choice_length"] for r in results]) == gold["choice_length_deterministic"].values).all()
    assert (np.array([r["rtd_rt_segment"] for r in results]) == gold["RT_calculated15"].values).all()


# ---------------------------------------------------------------------------
# beta4, fallback, stochastic layer, aggregation, copula
# ---------------------------------------------------------------------------
def test_beta4_zero_is_bit_identical_and_positive_beta4_shifts_segments_up(flexgold, params, sim_config):
    base, _ = _run(flexgold, params, sim_config)
    explicit, _ = _run(flexgold, params, sim_config, intercepts={**{m: 0.0 for m in MECHANISMS}})
    assert [r["rtd_flex_segment"] for r in base] == [r["rtd_flex_segment"] for r in explicit]
    up, _ = _run(flexgold, params, sim_config, intercepts={**{m: 0.0 for m in MECHANISMS}, "flexibility": 1.0})
    seg_base = np.array([r["rtd_flex_segment"] for r in base])
    seg_up = np.array([r["rtd_flex_segment"] for r in up])
    assert (seg_up >= seg_base).all() and (seg_up > seg_base).any()
    # beta4 lives on the standardized CALCULATED score before anchoring:
    # anchored shifts by 0.75 * beta4, z_anchored by 0.75 * beta4 / sd(anchored)
    shift = np.array([u["rtd_flex_score"] - b["rtd_flex_score"] for u, b in zip(up, base)])
    assert np.allclose(shift, 0.75, atol=1e-9)
    # the calculated-score intermediates are beta-free
    assert np.allclose([u["rtd_flex_ivw"] for u in up], [b["rtd_flex_ivw"] for b in base])


def test_missing_stdactions_falls_back_to_the_calculated_score(flexgold, params, sim_config):
    res, sim = _run(flexgold, params, sim_config, with_stdactions=False)
    assert all(r["rtd_flex_stdactions_missing"] for r in res)
    assert all(r["rtd_z_stdactions"] == 0.0 for r in res)
    assert sim["rtd_population_stats"]["flexibility"]["stdactions_missing"] == len(flexgold)
    # anchored = 0.75 * z_flex -> binning equals binning z_Flexibility_calculated_ivw alone
    z = flexgold["z_Flexibility_calculated_ivw"].values
    expected = np.floor(1 + (5 - 0.0001) * (z - z.min()) / (z.max() - z.min())).astype(int)
    assert (np.array([r["rtd_flex_segment"] for r in res]) == expected).all()


def test_flex_anchored_score_helper(params):
    scores = {"flexibility_ivw": 0.3, "z_stdactions": -1.0}
    anchored, z_flex = flex_anchored_score(scores, {"flexibility_ivw": {"mean": 0.1, "sd": 0.2}}, params)
    assert abs(z_flex - 1.0) < 1e-12
    assert abs(anchored - (0.25 * -1.0 + 0.75 * 1.0)) < 1e-12
    custom = dict(params, flexibility_anchor={"observed_weight": 0.5, "calculated_weight": 0.5})
    anchored2, _ = flex_anchored_score(scores, {"flexibility_ivw": {"mean": 0.1, "sd": 0.2}}, custom)
    assert abs(anchored2 - 0.0) < 1e-12


def test_flexibility_stochastic_layer(flexgold, params, sim_config):
    det, _ = _run(flexgold, params, sim_config)
    a, sim_a = _run(flexgold, params, sim_config, stochastic=True, seed=99)
    b, _ = _run(flexgold, params, sim_config, stochastic=True, seed=99)
    pop = sim_a["rtd_population_stats"]["flexibility"]
    assert "s_min" in pop and "s_max" in pop
    assert all(r["rtd_sigma_used_flex"] == SIGMA_OVERALL["flexibility"] for r in a)
    assert [r["rtd_flex_segment"] for r in a] == [r["rtd_flex_segment"] for r in b]
    # draws are anchored on z_anchored (continuous) and land inside the population range
    draws = np.array([r["rtd_flex_draw"] for r in a])
    assert pop["s_min"] - 1e-9 <= draws.min() and draws.max() <= pop["s_max"] + 1e-9
    assert [r["rtd_flex_segment"] for r in a] != [r["rtd_flex_segment"] for r in det]
    # sigma scale 0 reproduces the deterministic segments; deterministic segments unchanged
    zero, _ = _run(flexgold, params, sim_config, stochastic=True, seed=99, scale=0.0)
    assert [r["rtd_flex_segment"] for r in zero] == [r["rtd_flex_segment"] for r in det]
    assert [r["rtd_flex_segment_deterministic"] for r in a] == [r["rtd_flex_segment"] for r in det]
    # doc-literal binned anchor also runs
    binned, _ = _run(flexgold, params, sim_config, stochastic=True, seed=99, anchor="binned")
    assert all(1 <= r["rtd_flex_segment"] <= 5 for r in binned)


def test_aggregation_receives_the_documents_four_inputs(results):
    for r in results:
        assert r["rtd_consensus_inputs"] == ["loyalty", "wtp", "risk_taking", "flexibility"]
        assert r["rtd_flex_ranking_codes"][0] in (
            "lower_pn_vendor", "place_bid", "current_vendor_pn", "higher_price_category", "forgo_transaction")


def test_copula_model_carries_stdactions():
    from src.trait_engine import TraitEngine
    te = TraitEngine()
    assert "stdactions" in te.traits
    sample = te.sample(500, 7)
    assert sample["stdactions"].notna().all() and (sample["stdactions"] >= 0).all()
