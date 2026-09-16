# src/decisions/rtd_rank_aggregation.py
"""
Decision 4, Section 6 - integrating the mechanism rankings into ONE default list.

Source: "Decision 4 - Rejected Transaction Defaults" design document, rev 040926-2,
Section 6 "Integrating the model effects to predict rejected transaction defaults",
plus the professor's earlier "Ranking Cascade Results (V1)" report (the 100,000-case
evaluation the document quotes) and "Reconciling Four Rankings into One" review.

The mechanisms of Sections 2-5 each yield a priority list of the five rejected-
transaction options for a participant (loyalty, willingness to pay, risk-taking,
flexibility). These lists do not necessarily concur, so a single
consensus ranking of the five options is formed and then two output rules are
applied TO THE INTEGRATED CONSENSUS LIST ONLY:
  (1) the list is truncated to the configured list length (choice_length from the
      Tendency-to-Plan equation of Section 1), and
  (2) every option listed after Option 5 (forgo the transaction) is dropped - once
      the customer forgoes the transaction there can be no subsequent default.
NEITHER rule applies to a MECHANISM's own priority list: those are tails of a fixed
priority sequence and may legitimately continue past Option 5 (e.g. the Loyalty
sequence is 3 > 1 > 4 > 5 > 2, so a segment-5 participant's Loyalty list ends with
Option 2 after Option 5) and are not cut to the TTP length. The rules are applied
once, by apply_output_rules(), to the consensus ranking that aggregate_rankings()
returns.

THE AGGREGATION: Kemeny-Young with the document's tie-breaking hierarchy
-------------------------------------------------------------------------
Kendall-tau distance counts the option pairs two rankings order differently (10
unordered pairs for five options). The consensus is the ranking minimising the total
Kendall-tau distance to the input rankings (Kemeny 1959; Kemeny & Snell 1962), found
by exhaustive search over the 5! = 120 permutations. All inputs receive EQUAL weight
(doc: "there is no theoretical model to assign a different weight").

Kemeny-Young needs a fully specified tie-break rule because with four rankings of
five options ties are pervasive (doc: unique full ranking in only 7.7% of 100,000
random cases; several equally good orderings in 71.1%; a single optimal list that
still contained tied options in 21.1%). The hierarchy:

  Phase 1 - the initial ranking
    * Kemeny returns a UNIQUE, fully ordered ranking      -> done ('kemeny').
    * Kemeny returns a unique ordering WITH TIES           -> continue with it.
      (the set of Kemeny-optimal permutations is exactly the set of linear
      extensions of one weak order; the tied positions are the pairs the optimal
      permutations disagree on)
    * Kemeny returns SEVERAL equally good orderings        -> run Schulze (2011),
      the Condorcet-consistent strongest-path method, and continue with ITS
      ordering rather than selecting among the optimal permutations by an
      arbitrary rule (doc: Schulze's ordering is "in almost every case" itself a
      Kemeny-optimal list, so this selects from the optimal set rather than
      departing from it).
  Phase 2 - breaking the leftover ties, within each tied group in turn
    * Copeland (1951): pairwise wins minus pairwise losses           ('copeland')
    * Spearman footrule: among the orderings of the tied group, the one with the
      smallest total absolute positional displacement to the inputs  ('footrule')
    * Last resort: a RANDOM order of the still-tied options            ('random').
      Randomisation is preferred over a deterministic rule (e.g. lower-numbered
      option first, the V1 rule) because a deterministic rule introduces a
      systematic bias toward particular options. The V1 rule remains available
      as last_resort='lowest_option' for reproducibility comparisons.

The final ranking is one of the Kemeny-optimal orderings in all but a tiny share of
cases (doc: 99.93%); the exceptions arise only through the last resort.

INPUT RANKINGS MAY BE PARTIAL. The mechanisms' priority lists are TAILS of a fixed
priority sequence (segment s -> the last s options of the sequence; segment 5 = the
full list, segment 1 = one option - the direction flipped on the professor's
2026-09-16 instruction and adopted by doc rev 040926-2 /
Stata_File_Decision4_050926.dta, see rejected_transaction_defaults._ranking_for_segment), so a
participant's loyalty list may be e.g. [4, 5, 2] with Options 3 and 1 absent. The
document's aggregation text assumes complete rankings; here the options absent from
a list are treated as TIED AT THE BOTTOM of that list - the mechanism prefers each
listed option to each unlisted one and expresses no preference among the unlisted
ones. For the Kemeny objective this is equivalent to counting only the pairs the
input actually orders (an all-tied pair contributes the same to every candidate),
and it gives Copeland/Schulze the natural 'listed beats unlisted' preference.
Footrule positions of unlisted options are the average of the remaining positions
(standard mid-rank convention for ties).

Randomness: the last-resort draws come from the numpy Generator passed in
(the agent's Decision-4 RNG, consumed AFTER the mechanisms' four standard normals,
so all per-mechanism outputs are unaffected by the aggregation).
"""
from itertools import permutations
from math import factorial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

OPTIONS: Tuple[int, ...] = (1, 2, 3, 4, 5)
N_OPTIONS = len(OPTIONS)

# Unordered option pairs (x < y), fixed order used by the sign matrix below.
PAIRS: List[Tuple[int, int]] = [(x, y) for i, x in enumerate(OPTIONS) for y in OPTIONS[i + 1:]]

# All 120 candidate consensus rankings, and for each the +1/-1 sign per pair
# (+1 when x is ranked above y). A candidate's agreement with the pairwise
# margins is then a single matrix product (see kemeny_optimal_set).
ALL_PERMS: List[Tuple[int, ...]] = [tuple(p) for p in permutations(OPTIONS)]
_PERM_SIGNS = np.array(
    [[1 if p.index(x) < p.index(y) else -1 for (x, y) in PAIRS] for p in ALL_PERMS],
    dtype=np.int64,
)

STAGES = ("kemeny", "schulze", "copeland", "footrule", "random")
KEMENY_STATUSES = ("unique", "unique_with_ties", "multiple")
LAST_RESORT_MODES = ("random", "lowest_option")

Groups = List[List[int]]   # a weak order: ordered list of tied groups (each sorted)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def normalize_ranking(ranking: Iterable) -> List[int]:
    """Validate one (possibly partial) input ranking: distinct option numbers 1-5."""
    out: List[int] = []
    for o in ranking:
        if o is None or (isinstance(o, float) and np.isnan(o)):
            continue
        v = int(o)
        if v not in OPTIONS:
            raise ValueError(f"invalid option {o!r} in ranking {list(ranking)!r}")
        if v in out:
            raise ValueError(f"duplicate option {v} in ranking {list(ranking)!r}")
        out.append(v)
    return out


def pairwise_matrix(rankings: Sequence[Sequence[int]]) -> np.ndarray:
    """
    d[x-1, y-1] = number of input rankings placing x strictly above y.

    Unlisted options are tied at the bottom: a listed option is above every
    unlisted one; two unlisted options are not compared.
    """
    d = np.zeros((N_OPTIONS, N_OPTIONS), dtype=np.int64)
    for r in rankings:
        r = normalize_ranking(r)
        listed = set(r)
        for i, x in enumerate(r):
            for y in r[i + 1:]:
                d[x - 1, y - 1] += 1
            for y in OPTIONS:
                if y not in listed:
                    d[x - 1, y - 1] += 1
    return d


def footrule_positions(ranking: Sequence[int]) -> Dict[int, float]:
    """1-based positions; unlisted options share the mean of the remaining positions."""
    r = normalize_ranking(ranking)
    pos = {o: float(i + 1) for i, o in enumerate(r)}
    k = len(r)
    if k < N_OPTIONS:
        mid = (k + 1 + N_OPTIONS) / 2.0
        for o in OPTIONS:
            if o not in pos:
                pos[o] = mid
    return pos


def kendall_tau_distance(perm: Sequence[int], ranking: Sequence[int]) -> int:
    """Pairs ordered oppositely by a full permutation and one (possibly partial)
    ranking; pairs the ranking leaves unordered (both unlisted) contribute 0."""
    d = pairwise_matrix([ranking])
    p = {o: i for i, o in enumerate(perm)}
    dist = 0
    for x, y in PAIRS:
        if p[x] < p[y]:
            dist += int(d[y - 1, x - 1])
        else:
            dist += int(d[x - 1, y - 1])
    return dist


# ---------------------------------------------------------------------------
# Kemeny-Young
# ---------------------------------------------------------------------------
def kemeny_optimal_set(d: np.ndarray) -> Tuple[List[Tuple[int, ...]], int]:
    """
    All permutations minimising the total Kendall-tau distance to the inputs, and
    that minimal distance.

    For a candidate with sign s_k on pair k=(x,y), the disagreements on that pair are
    d[y,x] if s_k=+1 else d[x,y]  =  (d[x,y]+d[y,x])/2 - s_k*(d[x,y]-d[y,x])/2,
    so minimising total disagreement == maximising sum_k s_k * margin_k.
    """
    margins = np.array([d[x - 1, y - 1] - d[y - 1, x - 1] for (x, y) in PAIRS], dtype=np.int64)
    totals = int(sum(d[x - 1, y - 1] + d[y - 1, x - 1] for (x, y) in PAIRS))
    agreement = _PERM_SIGNS @ margins
    best = int(agreement.max())
    optimal = [ALL_PERMS[i] for i in np.flatnonzero(agreement == best)]
    min_distance = (totals - best) // 2
    return optimal, int(min_distance)


def collapse_to_groups(perms: Sequence[Sequence[int]], options: Sequence[int] = OPTIONS) -> Groups:
    """
    Collapse a set of linear orders over `options` into ordered tied groups.

    Two options are tied when the orders disagree on their relative position; ties
    are closed transitively (connected components), which makes the ordering
    between different groups unanimous across the set.
    """
    opts = list(options)
    idx = {o: i for i, o in enumerate(opts)}
    n = len(opts)
    pos = np.array([[list(p).index(o) for o in opts] for p in perms])   # (m, n)
    above_all = (pos[:, :, None] < pos[:, None, :]).all(axis=0)         # x above y in ALL
    tied = ~above_all & ~above_all.T
    np.fill_diagonal(tied, False)
    # connected components of the tie relation
    comp = [-1] * n
    c = 0
    for i in range(n):
        if comp[i] != -1:
            continue
        stack = [i]
        comp[i] = c
        while stack:
            u = stack.pop()
            for v in range(n):
                if tied[u, v] and comp[v] == -1:
                    comp[v] = c
                    stack.append(v)
        c += 1
    groups = [sorted(opts[i] for i in range(n) if comp[i] == k) for k in range(c)]
    # order groups by the (unanimous) position of their members in the first order
    first = list(perms[0])
    groups.sort(key=lambda g: min(first.index(o) for o in g))
    return groups


def kemeny_phase(rankings: Sequence[Sequence[int]]):
    """
    Phase-1 Kemeny step. Returns (status, groups, optimal_set, min_distance) where
    status is 'unique' (groups all singletons), 'unique_with_ties' (the optimal set
    is exactly the linear extensions of the collapsed weak order) or 'multiple'.
    """
    d = pairwise_matrix(rankings)
    optimal, min_distance = kemeny_optimal_set(d)
    groups = collapse_to_groups(optimal)
    if len(optimal) == 1:
        status = "unique"
    else:
        n_ext = 1
        for g in groups:
            n_ext *= factorial(len(g))
        status = "unique_with_ties" if n_ext == len(optimal) else "multiple"
    return status, groups, optimal, min_distance


# ---------------------------------------------------------------------------
# Schulze
# ---------------------------------------------------------------------------
def schulze_strongest_paths(d: np.ndarray) -> np.ndarray:
    """Strongest-path strengths p[x,y] (winning-votes variant; Floyd-Warshall)."""
    n = d.shape[0]
    p = np.where(d > d.T, d, 0).astype(np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                p[j, k] = max(p[j, k], min(p[j, i], p[i, k]))
    return p


def schulze_weak_order(d: np.ndarray) -> Groups:
    """
    Schulze ordering as tied groups: repeatedly take the options not beaten (via
    strongest paths) by any remaining option as the next group. Mutually unbeaten
    options are tied.
    """
    p = schulze_strongest_paths(d)
    beats = p > p.T
    remaining = list(OPTIONS)
    groups: Groups = []
    while remaining:
        unbeaten = [x for x in remaining
                    if not any(beats[y - 1, x - 1] for y in remaining if y != x)]
        if not unbeaten:            # cannot happen (Schulze relation is acyclic); guard anyway
            unbeaten = list(remaining)
        groups.append(sorted(unbeaten))
        remaining = [x for x in remaining if x not in unbeaten]
    return groups


# ---------------------------------------------------------------------------
# Phase-2 tie-breakers (each refines the tied groups; never reorders across groups)
# ---------------------------------------------------------------------------
def copeland_scores(d: np.ndarray) -> Dict[int, int]:
    """Pairwise wins minus pairwise losses over all options."""
    scores = {}
    for x in OPTIONS:
        s = 0
        for y in OPTIONS:
            if x == y:
                continue
            if d[x - 1, y - 1] > d[y - 1, x - 1]:
                s += 1
            elif d[x - 1, y - 1] < d[y - 1, x - 1]:
                s -= 1
        scores[x] = s
    return scores


def refine_by_copeland(groups: Groups, d: np.ndarray) -> Groups:
    scores = copeland_scores(d)
    out: Groups = []
    for g in groups:
        if len(g) == 1:
            out.append(list(g))
            continue
        by_score: Dict[int, List[int]] = {}
        for o in g:
            by_score.setdefault(scores[o], []).append(o)
        for s in sorted(by_score, reverse=True):
            out.append(sorted(by_score[s]))
    return out


def refine_by_footrule(groups: Groups, rankings: Sequence[Sequence[int]]) -> Groups:
    """
    Within each tied group occupying consensus positions start..start+|g|-1, keep
    the arrangements minimising the total Spearman footrule displacement to the
    inputs; the minimisers are collapsed back into (sub)groups.
    """
    positions = [footrule_positions(r) for r in rankings]
    out: Groups = []
    start = 1
    for g in groups:
        k = len(g)
        if k == 1:
            out.append(list(g))
            start += 1
            continue
        best_cost = None
        best: List[Tuple[int, ...]] = []
        for arrangement in permutations(g):
            cost = 0.0
            for offset, o in enumerate(arrangement):
                slot = start + offset
                for pos in positions:
                    cost += abs(slot - pos[o])
            if best_cost is None or cost < best_cost - 1e-9:
                best_cost, best = cost, [arrangement]
            elif abs(cost - best_cost) <= 1e-9:
                best.append(arrangement)
        out.extend(collapse_to_groups(best, options=g))
        start += k
    return out


def refine_by_last_resort(groups: Groups, rng: Optional[np.random.Generator],
                          last_resort: str = "random") -> Groups:
    """Random order of the still-tied options (doc), or lowest option number first
    (the V1 report's rule) when last_resort == 'lowest_option'."""
    if last_resort not in LAST_RESORT_MODES:
        raise ValueError(f"last_resort must be one of {LAST_RESORT_MODES}, got {last_resort!r}")
    out: Groups = []
    for g in groups:
        if len(g) == 1:
            out.append(list(g))
        elif last_resort == "lowest_option" or rng is None:
            out.extend([[o] for o in sorted(g)])
        else:
            order = rng.permutation(len(g))
            out.extend([[g[i]] for i in order])
    return out


def _is_strict(groups: Groups) -> bool:
    return all(len(g) == 1 for g in groups)


def _flatten(groups: Groups) -> List[int]:
    return [o for g in groups for o in g]


def groups_to_str(groups: Groups) -> str:
    """'3 > {1 = 4} > 5 > 2' style rendering of a weak order."""
    return " > ".join(str(g[0]) if len(g) == 1 else "{" + " = ".join(map(str, g)) + "}"
                      for g in groups)


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------
def aggregate_rankings(rankings: Sequence[Sequence[int]],
                       rng: Optional[np.random.Generator] = None,
                       last_resort: str = "random") -> Dict:
    """
    Consensus ranking of the five options for one participant.

    rankings : the mechanism priority lists (complete or partial, equal weight).
    rng      : numpy Generator for the random last resort (None -> lowest option
               first, i.e. the deterministic V1 rule, regardless of last_resort).
    Returns a dict with:
      consensus          full ranking of the five options (list of option numbers)
      settled_by         the stage that produced the FULL strict ranking - one of
                         STAGES ('kemeny', 'schulze', 'copeland', 'footrule',
                         'random'); the last stage that still had work to do
      kemeny_status      what the Kemeny step (Phase 1) returned:
                           'unique'            exactly ONE optimal permutation - a
                                               fully ordered ranking, procedure ends
                           'unique_with_ties'  several optimal permutations that are
                                               exactly the linear extensions of ONE
                                               weak order (i.e. a single optimal list
                                               that still contains tied options)
                           'multiple'          several optimal orderings that are NOT
                                               the extensions of one weak order ->
                                               Schulze supplies the Phase-1 ordering
                         "% of initial ties" in the reports = the share of agents with
                         kemeny_status != 'unique' (the Kemeny step did not yield a
                         unique fully ordered ranking)
      phase1             'kemeny' (the Kemeny weak order was carried into Phase 2) |
                         'schulze' (kemeny_status was 'multiple', so Schulze's
                         ordering was carried in instead)
      kemeny_distance    the minimal total Kendall-tau distance to the inputs
      n_kemeny_optimal   how many of the 120 permutations attain that minimum (1 when
                         kemeny_status == 'unique')
      is_kemeny_optimal  whether the FINAL consensus is one of those optimal
                         permutations; only the random last resort can make this False
      stages             {stage: weak order after that stage} for the stages run
      n_inputs           number of input rankings
    """
    rankings = [normalize_ranking(r) for r in rankings]
    if not rankings:
        raise ValueError("aggregate_rankings needs at least one input ranking")
    d = pairwise_matrix(rankings)
    status, groups, optimal, min_distance = kemeny_phase(rankings)
    optimal_set = set(optimal)
    stages: Dict[str, str] = {"kemeny": groups_to_str(groups)}
    settled_by = None
    phase1 = "kemeny"

    if status == "unique":
        settled_by = "kemeny"
    else:
        if status == "multiple":
            phase1 = "schulze"
            groups = schulze_weak_order(d)
            stages["schulze"] = groups_to_str(groups)
            if _is_strict(groups):
                settled_by = "schulze"
        if settled_by is None:
            groups = refine_by_copeland(groups, d)
            stages["copeland"] = groups_to_str(groups)
            if _is_strict(groups):
                settled_by = "copeland"
        if settled_by is None:
            groups = refine_by_footrule(groups, rankings)
            stages["footrule"] = groups_to_str(groups)
            if _is_strict(groups):
                settled_by = "footrule"
        if settled_by is None:
            groups = refine_by_last_resort(groups, rng, last_resort)
            stages["random"] = groups_to_str(groups)
            settled_by = "random"

    consensus = _flatten(groups)
    assert sorted(consensus) == list(OPTIONS), consensus
    return {
        "consensus": consensus,
        "settled_by": settled_by,
        "kemeny_status": status,
        "phase1": phase1,
        "kemeny_distance": int(min_distance),
        "n_kemeny_optimal": len(optimal),
        "is_kemeny_optimal": tuple(consensus) in optimal_set,
        "stages": stages,
        "n_inputs": len(rankings),
    }


def apply_output_rules(consensus: Sequence[int], choice_length: int) -> Tuple[List[int], str]:
    """
    The two output rules on the INTEGRATED consensus ranking (never on a mechanism's
    own priority list): truncate to choice_length, and drop everything after Option 5.

    Returns (default_list, truncated_by) with truncated_by describing which rule
    actually bound:
      'none'    neither rule cut anything - the full 5-option consensus is returned
                (only possible when choice_length == 5 and Option 5 is last),
      'length'  the TTP choice length cut the list (and did so at or before the
                Option-5 position, so the Option-5 rule was not the binding one),
      'option5' the Option-5 cut bound (Option 5 appears before position 5 and at or
                before the length cut),
      'both'    both rules cut at the SAME position.
    """
    consensus = list(consensus)
    cut_len = int(np.clip(int(choice_length), 0, N_OPTIONS))
    pos5 = consensus.index(5) + 1 if 5 in consensus else N_OPTIONS
    keep = min(cut_len, pos5)
    by_length = cut_len < N_OPTIONS and cut_len <= pos5
    by_option5 = pos5 < N_OPTIONS and pos5 <= cut_len
    if by_length and by_option5:
        truncated_by = "both"
    elif by_length:
        truncated_by = "length"
    elif by_option5:
        truncated_by = "option5"
    else:
        truncated_by = "none"
    return consensus[:keep], truncated_by


def integrate_default_list(rankings: Sequence[Sequence[int]], choice_length: int,
                           rng: Optional[np.random.Generator] = None,
                           last_resort: str = "random") -> Dict:
    """aggregate_rankings + apply_output_rules in one call (adds default_list,
    default_list_length, truncated_by, choice_length)."""
    res = aggregate_rankings(rankings, rng=rng, last_resort=last_resort)
    default_list, truncated_by = apply_output_rules(res["consensus"], choice_length)
    res.update({
        "default_list": default_list,
        "default_list_length": len(default_list),
        "truncated_by": truncated_by,
        "choice_length": int(choice_length),
    })
    return res


# ---------------------------------------------------------------------------
# Evaluation helper (reproduces the document's 100,000-case experiment design)
# ---------------------------------------------------------------------------
def simulate_stage_shares(n_cases: int = 100_000, n_rankings: int = 4, seed: int = 0,
                          last_resort: str = "random",
                          partial_sequences: Optional[Sequence[Sequence[int]]] = None) -> Dict:
    """
    Run the cascade on random inputs and tabulate the shares settled at each stage,
    the Kemeny status shares and the Kemeny-optimality rate.

    partial_sequences=None : n_rankings uniformly random COMPLETE permutations per
                             case (the document's evaluation design).
    partial_sequences given: one input per sequence, each a uniformly random TAIL
                             (length 1..5) of that priority sequence - the shape of
                             the actual mechanism outputs.
    """
    rng = np.random.default_rng(seed)
    settled = {s: 0 for s in STAGES}
    status = {s: 0 for s in KEMENY_STATUSES}
    optimal = 0
    for _ in range(n_cases):
        if partial_sequences is None:
            inputs = [list(rng.permutation(OPTIONS)) for _ in range(n_rankings)]
        else:
            inputs = [list(seq[N_OPTIONS - int(rng.integers(1, N_OPTIONS + 1)):])
                      for seq in partial_sequences]
        res = aggregate_rankings(inputs, rng=rng, last_resort=last_resort)
        settled[res["settled_by"]] += 1
        status[res["kemeny_status"]] += 1
        optimal += int(res["is_kemeny_optimal"])
    return {
        "n_cases": n_cases,
        "settled_by": {k: v / n_cases for k, v in settled.items()},
        "kemeny_status": {k: v / n_cases for k, v in status.items()},
        "kemeny_optimal_rate": optimal / n_cases,
    }
