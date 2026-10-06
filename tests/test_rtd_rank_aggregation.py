"""
Tests for the Decision 4 Section-6 rank aggregation
(src/decisions/rtd_rank_aggregation.py) and its integration into
rejected_transaction_defaults().

Spec: "Decision 4 - Rejected Transaction Defaults" rev 040926-2, Section 6 -
Kemeny-Young consensus (minimum total Kendall-tau distance over the 120 orderings,
equal weights) with the tie-break hierarchy Schulze -> Copeland -> footrule -> random,
then truncation to the TTP choice length and at Option 5. The professor's "Ranking
Cascade Results (V1)" report gives the reference stage shares over 100,000 random
complete-ranking cases (Kemeny alone 7.7%, Schulze 0.0%, Copeland 43.3%, Footrule
22.7%, last resort 26.3%; final ranking Kemeny-optimal in 99.93%). Section 6 is
UNCHANGED in rev 040926-2; the inputs it reconciles moved, because rev 040926-2
corrected the Loyalty Agreeableness weight and adopted the Flexibility O/N/A
coefficients and the segment -> option-list direction (all four mechanism lists now
equal Stata_File_Decision4_050926.dta's stored choice1..5_* columns directly).
"""
import copy
import os
from itertools import permutations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.decisions.rtd_rank_aggregation import (
    ALL_PERMS, KEMENY_STATUSES, OPTIONS, STAGES,
    aggregate_rankings, apply_output_rules, collapse_to_groups, copeland_scores,
    footrule_positions, integrate_default_list, kemeny_optimal_set, kemeny_phase,
    kendall_tau_distance, pairwise_matrix, refine_by_copeland, refine_by_footrule,
    schulze_weak_order, simulate_stage_shares,
)
from src.decisions.rejected_transaction_defaults import (
    MECHANISMS, OPTION_CODES, PRIORITY_SEQUENCES, RANKING_KEYS,
    compute_rtd_population_stats, rejected_transaction_defaults,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFICATION_CSV = os.path.join(REPO, "data", "stata_d4_verification.csv")
DECISIONS_YAML = os.path.join(REPO, "config", "decisions.yaml")

# The literature review's worked example (four complete rankings).
LIT_EXAMPLE = [[1, 2, 3, 4, 5], [2, 1, 3, 5, 4], [1, 3, 2, 4, 5], [2, 3, 1, 5, 4]]


# ---------------------------------------------------------------------------
# Distances and the pairwise matrix (complete and partial inputs)
# ---------------------------------------------------------------------------
def test_pairwise_matrix_complete_ranking():
    d = pairwise_matrix([[1, 2, 3, 4, 5]])
    for x in OPTIONS:
        for y in OPTIONS:
            expected = 1 if x < y else 0
            assert d[x - 1, y - 1] == expected


def test_pairwise_matrix_partial_ranking_unlisted_tied_at_bottom():
    d = pairwise_matrix([[4, 5, 2]])
    # listed order
    assert d[3, 4] == 1 and d[4, 3] == 0          # 4 above 5
    assert d[3, 1] == 1 and d[4, 1] == 1          # 4, 5 above 2
    # every listed option beats every unlisted one (Options 1 and 3)
    for x in (4, 5, 2):
        for y in (1, 3):
            assert d[x - 1, y - 1] == 1 and d[y - 1, x - 1] == 0
    # the two unlisted options are not compared
    assert d[0, 2] == 0 and d[2, 0] == 0


def test_kendall_tau_distance():
    assert kendall_tau_distance([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 0
    assert kendall_tau_distance([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == 10
    assert kendall_tau_distance([1, 2, 3, 4, 5], [2, 1, 3, 5, 4]) == 2   # lit review example
    # partial input [5]: the consensus puts 1..4 above 5 -> four reversed pairs
    assert kendall_tau_distance([1, 2, 3, 4, 5], [5]) == 4
    assert kendall_tau_distance([5, 1, 2, 3, 4], [5]) == 0


def test_footrule_positions_partial_uses_mid_rank():
    pos = footrule_positions([4, 5, 2])
    assert pos == {4: 1.0, 5: 2.0, 2: 3.0, 1: 4.5, 3: 4.5}
    assert footrule_positions([5]) == {5: 1.0, 1: 3.5, 2: 3.5, 3: 3.5, 4: 3.5}


# ---------------------------------------------------------------------------
# Kemeny-Young
# ---------------------------------------------------------------------------
def test_kemeny_unique_for_unanimous_inputs():
    status, groups, optimal, dist = kemeny_phase([[3, 1, 4, 5, 2]] * 4)
    assert status == "unique" and optimal == [(3, 1, 4, 5, 2)] and dist == 0
    assert groups == [[3], [1], [4], [5], [2]]


def test_kemeny_optimal_set_matches_brute_force():
    rng = np.random.default_rng(11)
    for _ in range(200):
        n_inputs = int(rng.integers(1, 5))
        inputs = []
        for _ in range(n_inputs):
            perm = list(rng.permutation(OPTIONS))
            cut = int(rng.integers(1, 6))               # partial tails too
            inputs.append(perm[:cut])
        d = pairwise_matrix(inputs)
        optimal, min_dist = kemeny_optimal_set(d)
        brute = {p: sum(kendall_tau_distance(p, r) for r in inputs) for p in ALL_PERMS}
        best = min(brute.values())
        assert min_dist == best
        assert set(optimal) == {p for p, c in brute.items() if c == best}


def test_collapse_to_groups_transitive_closure():
    # optimal set that is exactly the linear extensions of {1=2} > 3 > {4=5}
    S = [(1, 2, 3, 4, 5), (2, 1, 3, 4, 5), (1, 2, 3, 5, 4), (2, 1, 3, 5, 4)]
    assert collapse_to_groups(S) == [[1, 2], [3], [4, 5]]
    # ties close transitively: 1~2 and 2~3 (with 1>3 everywhere) -> one group
    S = [(1, 2, 3, 4, 5), (2, 1, 3, 4, 5), (1, 3, 2, 4, 5)]
    assert collapse_to_groups(S) == [[1, 2, 3], [4], [5]]


def test_lit_review_example_kemeny_with_ties_falls_to_random():
    res = aggregate_rankings(LIT_EXAMPLE, rng=np.random.default_rng(0))
    assert res["kemeny_status"] == "unique_with_ties"
    assert res["stages"]["kemeny"] == "{1 = 2} > 3 > {4 = 5}"
    assert res["phase1"] == "kemeny"
    assert res["kemeny_distance"] == 6 and res["n_kemeny_optimal"] == 4
    # Copeland and footrule cannot separate the 2-2 splits; the last resort does
    assert res["stages"]["copeland"] == "{1 = 2} > 3 > {4 = 5}"
    assert res["stages"]["footrule"] == "{1 = 2} > 3 > {4 = 5}"
    assert res["settled_by"] == "random"
    c = res["consensus"]
    assert {c[0], c[1]} == {1, 2} and c[2] == 3 and {c[3], c[4]} == {4, 5}
    assert res["is_kemeny_optimal"]


def test_lowest_option_last_resort_is_deterministic():
    res = aggregate_rankings(LIT_EXAMPLE, rng=np.random.default_rng(0), last_resort="lowest_option")
    assert res["consensus"] == [1, 2, 3, 4, 5] and res["settled_by"] == "random"
    # no rng -> the deterministic rule regardless of the mode
    assert aggregate_rankings(LIT_EXAMPLE, rng=None)["consensus"] == [1, 2, 3, 4, 5]


def test_random_last_resort_is_seeded_and_unbiased():
    a = aggregate_rankings(LIT_EXAMPLE, rng=np.random.default_rng(99))["consensus"]
    b = aggregate_rankings(LIT_EXAMPLE, rng=np.random.default_rng(99))["consensus"]
    assert a == b
    firsts = {aggregate_rankings(LIT_EXAMPLE, rng=np.random.default_rng(s))["consensus"][0]
              for s in range(40)}
    assert firsts == {1, 2}   # both tied options get to be first across seeds


def test_footrule_stage_separates_a_copeland_tie():
    # (1,2) split 2-2; both beat 3 (3-1) and 4, 5 (4-0) -> Copeland tie (3 vs 3);
    # Option 1 is ranked first twice, Option 2 once -> footrule prefers 1 first.
    inputs = [[1, 2, 3, 4, 5], [2, 1, 3, 4, 5], [1, 2, 3, 4, 5], [3, 2, 1, 4, 5]]
    res = aggregate_rankings(inputs, rng=np.random.default_rng(0))
    assert res["kemeny_status"] == "unique_with_ties"
    assert res["stages"]["kemeny"] == "{1 = 2} > 3 > 4 > 5"
    assert res["stages"]["copeland"] == "{1 = 2} > 3 > 4 > 5"
    assert res["settled_by"] == "footrule"
    assert res["consensus"] == [1, 2, 3, 4, 5] and res["is_kemeny_optimal"]


def test_schulze_then_copeland_when_kemeny_has_several_orderings():
    # 1 > 3 by 3-1, while (1,2) and (2,3) are 2-2: the Kemeny optima are the three
    # orderings of {1,2,3} with 1 before 3 - not the extensions of one weak order.
    inputs = [[1, 2, 3, 4, 5], [2, 1, 3, 4, 5], [1, 3, 2, 4, 5], [3, 2, 1, 4, 5]]
    d = pairwise_matrix(inputs)
    status, _, optimal, _ = kemeny_phase(inputs)
    assert status == "multiple" and len(optimal) == 3
    assert schulze_weak_order(d) == [[1, 2], [3], [4], [5]]
    assert copeland_scores(d) == {1: 3, 2: 2, 3: 1, 4: -2, 5: -4}   # 3 vs 2 is tied
    res = aggregate_rankings(inputs, rng=np.random.default_rng(0))
    assert res["phase1"] == "schulze"
    assert res["stages"]["schulze"] == "{1 = 2} > 3 > 4 > 5"
    assert res["settled_by"] == "copeland"
    assert res["consensus"] == [1, 2, 3, 4, 5] and res["is_kemeny_optimal"]


def test_random_stage_on_symmetric_three_cycle():
    # perfect 3-cycle among 1,2,3 (each pair 2-1): Schulze, Copeland and footrule are
    # all symmetric -> the last resort orders them; 4 > 5 is unanimous.
    inputs = [[1, 2, 3, 4, 5], [2, 3, 1, 4, 5], [3, 1, 2, 4, 5]]
    res = aggregate_rankings(inputs, rng=np.random.default_rng(3))
    assert res["kemeny_status"] == "multiple" and res["phase1"] == "schulze"
    assert res["settled_by"] == "random"
    assert set(res["consensus"][:3]) == {1, 2, 3} and res["consensus"][3:] == [4, 5]


def test_refine_functions_only_split_within_groups():
    d = np.zeros((5, 5), dtype=int)
    d[0, 1] = d[1, 0] = 2                      # 1 vs 2 tied
    for y in (3, 4, 5):                        # 1 beats 3, 4, 5
        d[0, y - 1], d[y - 1, 0] = 3, 1
    d[1, 2], d[2, 1] = 3, 1                    # 2 beats 3 ...
    d[3, 1], d[1, 3] = 3, 1                    # ... loses to 4 and 5
    d[4, 1], d[1, 4] = 3, 1
    assert refine_by_copeland([[1, 2], [3], [4, 5]], d)[:2] == [[1], [2]]
    # footrule refinement keeps the block positions: group {1,2} at positions 1-2
    groups = refine_by_footrule([[1, 2], [3], [4], [5]], [[1, 2, 3, 4, 5], [2, 1, 3, 4, 5], [1, 2, 3, 4, 5]])
    assert groups == [[1], [2], [3], [4], [5]]


def test_every_output_is_a_permutation_and_stage_is_valid():
    rng = np.random.default_rng(5)
    seqs = [PRIORITY_SEQUENCES["loyalty"], PRIORITY_SEQUENCES["wtp"], PRIORITY_SEQUENCES["risk_taking"]]
    for _ in range(300):
        inputs = [seq[5 - int(rng.integers(1, 6)):] for seq in seqs]
        res = aggregate_rankings(inputs, rng=rng)
        assert sorted(res["consensus"]) == list(OPTIONS)
        assert res["settled_by"] in STAGES
        if res["settled_by"] != "random":
            # only the random last resort may leave the Kemeny-optimal set ... except
            # when Schulze's starting order itself does (rare; reported, not forbidden)
            assert res["phase1"] in ("kemeny", "schulze")
        if res["kemeny_status"] in ("unique", "unique_with_ties"):
            assert res["is_kemeny_optimal"]     # Phase 2 only refines the optimal weak order


# ---------------------------------------------------------------------------
# Output rules
# ---------------------------------------------------------------------------
def test_apply_output_rules():
    assert apply_output_rules([3, 1, 4, 5, 2], 5) == ([3, 1, 4, 5], "option5")
    assert apply_output_rules([3, 1, 4, 5, 2], 2) == ([3, 1], "length")
    assert apply_output_rules([5, 3, 1, 4, 2], 3) == ([5], "option5")
    assert apply_output_rules([3, 1, 4, 2, 5], 5) == ([3, 1, 4, 2, 5], "none")
    assert apply_output_rules([3, 5, 1, 4, 2], 2) == ([3, 5], "both")
    assert apply_output_rules([3, 1, 4, 5, 2], 0) == ([], "length")
    assert apply_output_rules([3, 1, 4, 5, 2], 9) == ([3, 1, 4, 5], "option5")   # clipped


def test_integrate_default_list_combines_both_steps():
    res = integrate_default_list([[4, 5, 2], [3, 2, 1, 4, 5], [1, 3, 5]], 3, rng=np.random.default_rng(1))
    assert res["default_list"] == res["consensus"][:res["default_list_length"]]
    assert res["default_list_length"] <= 3
    assert 5 not in res["default_list"][:-1]


# ---------------------------------------------------------------------------
# Hand-checked cases (migration port, 2026-10-06). Each expected value below was
# worked out by hand from the doc's Section 6 text, independently of the module.
# ---------------------------------------------------------------------------
def test_hand_case_three_against_one_is_a_unique_kemeny_ranking():
    """3 x (1>2>3>4>5) + 1 x the reverse: every pair is 3-1 for the natural order, so
    Kemeny is unique; the reverse ranking disagrees on all 10 pairs -> distance 10.
    TTP length 3 truncates (Option 5 is last, so only the length rule binds)."""
    res = integrate_default_list([[1, 2, 3, 4, 5]] * 3 + [[5, 4, 3, 2, 1]], 3, rng=None)
    assert res["kemeny_status"] == "unique" and res["settled_by"] == "kemeny"
    assert res["consensus"] == [1, 2, 3, 4, 5] and res["kemeny_distance"] == 10
    assert res["default_list"] == [1, 2, 3] and res["truncated_by"] == "length"


def test_hand_case_option5_cut_binds_before_the_length():
    """Unanimous 3>5>1>2>4 with TTP length 4: everything after Option 5 is dropped,
    so the default list is [3, 5] (doc rule 2), not four options."""
    res = integrate_default_list([[3, 5, 1, 2, 4]] * 4, 4, rng=None)
    assert res["consensus"] == [3, 5, 1, 2, 4] and res["kemeny_distance"] == 0
    assert res["default_list"] == [3, 5] and res["truncated_by"] == "option5"
    # length 0 -> empty list
    assert integrate_default_list([[3, 5, 1, 2, 4]] * 4, 0, rng=None)["default_list"] == []


def test_hand_case_two_two_split_survives_copeland_and_footrule():
    """2 x (1>2>3>4>5) + 2 x (2>1>3>4>5): the (1,2) margin is 0, every other pair is
    unanimous -> the Kemeny optimum is {12345, 21345} = the linear extensions of
    {1=2}>3>4>5 ('unique_with_ties', distance 2). Copeland: 1 and 2 both beat 3,4,5
    and tie each other (score 3 each). Footrule over positions 1-2: either order
    costs 2+2 = 4. So only the last resort separates them (lowest option with rng=None;
    both orders across seeds with the doc's random rule)."""
    rs = [[1, 2, 3, 4, 5]] * 2 + [[2, 1, 3, 4, 5]] * 2
    res = aggregate_rankings(rs, rng=None)
    assert res["kemeny_status"] == "unique_with_ties" and res["kemeny_distance"] == 2
    assert res["stages"]["kemeny"] == "{1 = 2} > 3 > 4 > 5"
    assert copeland_scores(pairwise_matrix(rs))[1] == copeland_scores(pairwise_matrix(rs))[2] == 3
    assert res["stages"]["footrule"] == "{1 = 2} > 3 > 4 > 5"
    assert res["settled_by"] == "random" and res["consensus"] == [1, 2, 3, 4, 5]
    firsts = {aggregate_rankings(rs, rng=np.random.default_rng(s))["consensus"][0] for s in range(30)}
    assert firsts == {1, 2}


def test_hand_case_partial_input_absent_options_tied_at_bottom():
    """A single partial ranking 4>5>2: listed options beat unlisted ones, 1 and 3 are
    tied at the bottom -> Kemeny {45213, 45231} ('unique_with_ties', distance 0);
    Copeland (-3 each) and footrule (both 4.5, cost 1 either way) cannot separate 1
    and 3. The integrated default list is cut after Option 5: [4, 5]."""
    res = integrate_default_list([[4, 5, 2]], 5, rng=None)
    assert res["kemeny_status"] == "unique_with_ties" and res["kemeny_distance"] == 0
    assert res["stages"]["kemeny"] == "4 > 5 > 2 > {1 = 3}"
    assert res["settled_by"] == "random" and res["consensus"] == [4, 5, 2, 1, 3]
    assert res["default_list"] == [4, 5] and res["truncated_by"] == "option5"


# ---------------------------------------------------------------------------
# Reference statistics from the document / V1 report (random complete rankings)
# ---------------------------------------------------------------------------
def test_stage_shares_reproduce_the_documents_experiment():
    stats = simulate_stage_shares(10_000, n_rankings=4, seed=1)
    s = stats["settled_by"]
    k = stats["kemeny_status"]
    assert abs(s["kemeny"] - 0.077) < 0.012          # doc 7.7%
    assert s["schulze"] < 0.005                      # doc 0.0%
    assert abs(s["copeland"] - 0.433) < 0.03         # doc 43.3%
    assert abs(s["footrule"] - 0.227) < 0.03         # doc 22.7%
    assert abs(s["random"] - 0.263) < 0.02           # doc 26.3%
    assert abs(k["unique_with_ties"] - 0.211) < 0.02  # doc 21.1%
    assert abs(k["multiple"] - 0.711) < 0.02          # doc 71.1%
    assert stats["kemeny_optimal_rate"] > 0.995      # doc 99.93%


# ---------------------------------------------------------------------------
# Integration into the Decision 4 function
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gold():
    return pd.read_csv(VERIFICATION_CSV)


@pytest.fixture(scope="module")
def params():
    with open(DECISIONS_YAML) as f:
        cfg = yaml.safe_load(f)
    # no 'model_enabled' switch (ruling R23): the model runs whenever selected
    return dict(cfg["rejected_transaction_defaults"])


@pytest.fixture(scope="module")
def sim_config(gold):
    return {"income_stats": {"mean": float(gold["income"].mean()),
                             "sd": float(gold["income"].std(ddof=1))}}


def _agent_state(row):
    return {
        "ExtraversionBig5": row["extraversionbig5"], "Agreeable": row["agreeable"],
        "NeuroticismBig5": row["neuroticismbig5"], "ConscientiousnessBig5": row["conscientiousnessbig5"],
        "OpennessBig5": row["opennessbig5"], "Education": row["education"],
        "Assigned Allowance Level": row["assignedallowancelevel"], "income": row["income"],
        "stdactions": row["stdactions"],
    }


def _run(gold, params, sim_config, aggregation=None, stochastic=False, seed=1234):
    p = copy.deepcopy(params)
    p.setdefault("stochastic", {})
    p["stochastic"]["sigma_value"] = 1.0 if stochastic else 0
    p["stochastic"]["in_copula"] = False
    if aggregation is not None:
        p["aggregation"] = aggregation
    agents_df = pd.DataFrame([_agent_state(r) for _, r in gold.iterrows()])
    incomes = gold["income"].tolist()
    seeds = np.random.default_rng(seed).integers(0, 1_000_000_000, len(gold))
    offset = 4000
    sim = dict(sim_config)
    sim["rtd_population_stats"] = compute_rtd_population_stats(
        agents_df, incomes, p, sim, pop_context="documentation",
        agent_base_seeds=list(seeds), decision_offset=offset)
    out = []
    for i, (_, r) in enumerate(gold.iterrows()):
        rng = np.random.default_rng(int(seeds[i]) + offset)
        out.append(rejected_transaction_defaults(_agent_state(r), p, rng, sim,
                                                 pop_context="documentation"))
    return out


def test_config_enables_aggregation_by_default(params):
    assert params["aggregation"]["enabled"] is True
    assert params["aggregation"]["last_resort"] == "random"
    assert params["aggregation"]["mechanisms"] is None


def test_integrated_default_list_per_agent(gold, params, sim_config):
    results = _run(gold, params, sim_config)
    for r in results:
        consensus = r["rtd_consensus_ranking"]
        default_list = r["rtd_default_list"]
        assert sorted(consensus) == list(OPTIONS)
        assert r["rtd_consensus_inputs"] == ["loyalty", "wtp", "risk_taking", "flexibility"]
        # the main output column IS the integrated list, as option codes
        assert r["rejected_transaction_defaults"] == [OPTION_CODES[o] for o in default_list]
        assert r["rtd_consensus_ranking_codes"] == [OPTION_CODES[o] for o in consensus]
        # output rule 1: at most the TTP choice length; rule 2: nothing after Option 5
        assert default_list == consensus[:len(default_list)]
        assert len(default_list) == r["rtd_default_list_length"] <= r["rtd_choice_length"]
        assert 5 not in default_list[:-1]
        if r["rtd_choice_length"] == 0:
            assert default_list == [] and r["rtd_consensus_truncated_by"] == "length"
        assert r["rtd_consensus_settled_by"] in STAGES
        assert r["rtd_consensus_kemeny_status"] in ("unique", "unique_with_ties", "multiple")
        assert r["rtd_consensus_last_resort"] == "random"
        # each mechanism list is respected wherever all four agree on a pair
        d = pairwise_matrix([r[f"rtd_{RANKING_KEYS[m]}_ranking"] for m in RANKING_KEYS])
        pos = {o: i for i, o in enumerate(consensus)}
        for x in OPTIONS:
            for y in OPTIONS:
                if d[x - 1, y - 1] == 4:
                    assert pos[x] < pos[y]
    # the population is not degenerate: several distinct lists and stages occur
    assert len({tuple(r["rtd_default_list"]) for r in results}) >= 5
    assert len({r["rtd_consensus_settled_by"] for r in results}) >= 3


def test_aggregation_leaves_mechanism_outputs_bit_identical(gold, params, sim_config):
    """The random last resort draws AFTER the mechanisms' four normals, so every
    per-mechanism rtd_* output is identical with and without the aggregation,
    deterministic and stochastic."""
    for stochastic in (False, True):
        on = _run(gold, params, sim_config, stochastic=stochastic)
        off = _run(gold, params, sim_config, aggregation={"enabled": False}, stochastic=stochastic)
        for a, b in zip(on, off):
            assert b["rejected_transaction_defaults"] == []
            assert "rtd_default_list" not in b
            for k, v in b.items():
                if k == "rejected_transaction_defaults":
                    continue
                assert a[k] == v, k


def test_aggregation_is_reproducible_and_seed_sensitive(gold, params, sim_config):
    a = _run(gold, params, sim_config, seed=7)
    b = _run(gold, params, sim_config, seed=7)
    assert [r["rtd_consensus_ranking"] for r in a] == [r["rtd_consensus_ranking"] for r in b]
    # a different seed changes only the randomly settled agents' consensus
    c = _run(gold, params, sim_config, seed=8)
    for ra, rc in zip(a, c):
        if ra["rtd_consensus_settled_by"] != "random":
            assert ra["rtd_consensus_ranking"] == rc["rtd_consensus_ranking"]


def test_lowest_option_mode_is_deterministic_across_seeds(gold, params, sim_config):
    a = _run(gold, params, sim_config, aggregation={"enabled": True, "last_resort": "lowest_option"}, seed=1)
    b = _run(gold, params, sim_config, aggregation={"enabled": True, "last_resort": "lowest_option"}, seed=2)
    assert [r["rtd_consensus_ranking"] for r in a] == [r["rtd_consensus_ranking"] for r in b]
    assert all(r["rtd_consensus_last_resort"] == "lowest_option" for r in a)


def test_mechanism_subset_can_be_configured(gold, params, sim_config):
    results = _run(gold, params, sim_config, aggregation={"enabled": True, "mechanisms": ["loyalty", "wtp"]})
    for r in results:
        assert r["rtd_consensus_inputs"] == ["loyalty", "wtp"]
        assert r["rtd_consensus_ranking"][0] in r["rtd_loyalty_ranking"] + r["rtd_wtp_ranking"]


def test_default_template_path_ignores_aggregation(gold, params, sim_config):
    p = copy.deepcopy(params)
    sim = dict(sim_config)
    sim["default_decisions_list"] = ["rejected_transaction_defaults"]
    sim["default_decisions"] = {"rejected_transaction_defaults": {
        "type": "prioritized_selection", "priority_template": ["current_vendor_pn", "forgo_transaction"]}}
    out = rejected_transaction_defaults(_agent_state(gold.iloc[0]), p, np.random.default_rng(0), sim)
    assert out == {"rejected_transaction_defaults": ["current_vendor_pn", "forgo_transaction"]}


# ---------------------------------------------------------------------------
# Section-6 inputs are COMPLETE rankings (professor 2026-09-17)
# ---------------------------------------------------------------------------
def test_complete_ranking_for_segment_rotates_the_priority_sequence():
    """Doc Section 6: the aggregation reconciles "four complete rankings of the five
    rejected transaction default options ... each depending on the participant's
    weighted factor score and the resulting first choice" - the priority sequence
    rotated to start at that segment's first choice."""
    from src.decisions.rejected_transaction_defaults import (
        _complete_ranking_for_segment, _ranking_for_segment, PRIORITY_SEQUENCES)

    assert _complete_ranking_for_segment('loyalty', 5) == [3, 1, 4, 5, 2]
    assert _complete_ranking_for_segment('loyalty', 3) == [4, 5, 2, 3, 1]
    assert _complete_ranking_for_segment('loyalty', 1) == [2, 3, 1, 4, 5]

    for mech, seq in PRIORITY_SEQUENCES.items():
        if mech == 'ttp':
            continue
        for seg in range(1, 6):
            full = _complete_ranking_for_segment(mech, seg)
            tail = _ranking_for_segment(mech, seg)
            assert sorted(full) == [1, 2, 3, 4, 5], (mech, seg)   # every option, once
            assert full[:len(tail)] == tail, (mech, seg)          # starts with the element list
            assert full[len(tail):] == list(seq[:len(seq) - seg])  # then the omitted ones, in order


def test_aggregation_is_fed_complete_rankings_not_the_element_lists(gold, params, sim_config):
    """The element's own list stays the .dta-verified tail in rtd_<element>_ranking,
    while the aggregation input carries all five options."""
    for r in _run(gold, params, sim_config):
        fed = r['rtd_consensus_input_rankings']
        assert set(fed) == {'loyalty', 'wtp', 'risk_taking', 'flexibility'}
        for mech, key in (('loyalty', 'loyalty'), ('wtp', 'wtp'),
                          ('risk_taking', 'rt'), ('flexibility', 'flex')):
            element_list = r[f'rtd_{key}_ranking']
            assert sorted(fed[mech]) == [1, 2, 3, 4, 5], mech             # complete ranking
            assert len(element_list) == int(r[f'rtd_{key}_segment'])      # element list = the tail
            assert fed[mech][:len(element_list)] == element_list


def test_complete_rankings_cut_the_random_last_resort(gold, params, sim_config):
    """The professor's "close to 50% resolved randomly" came from feeding the aggregation
    the TRUNCATED element lists: most option pairs then have a zero pairwise margin and no
    method can separate them. With the document's complete rankings the random share falls
    to roughly a third, near the document's own 100,000-case figure of 26.3%."""
    results = _run(gold, params, sim_config)
    n = len(results)
    random_share = sum(r['rtd_consensus_settled_by'] == 'random' for r in results) / n
    assert 0.25 <= random_share <= 0.45, random_share

    # the same population scored with the OLD truncated inputs, for contrast
    old = []
    for r in results:
        rankings = [r[f'rtd_{k}_ranking'] for k in ('loyalty', 'wtp', 'rt', 'flex')]
        old.append(aggregate_rankings(rankings, rng=np.random.default_rng(0))['settled_by'])
    old_share = sum(s == 'random' for s in old) / n
    assert old_share > random_share + 0.15, (old_share, random_share)
