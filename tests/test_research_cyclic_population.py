"""
Owner ruling R-CYC (2026-10-07), as clarified by the owner the same day:

RESEARCH BASELINE takes the participants in their real (file) order, cycling - agent
k (0-based) is participant k mod 280:

* N = 280  -> exactly the 280, in file order;
* N = 1000 -> three full cycles + the first 160;
* N < 280  -> the first N.

RESEARCH SPECIFICATION samples RANDOMLY, seeded by the run seed (reproducible):

* N < 280  -> a random subset of N distinct participants (without replacement);
* N = 280  -> all 280 (kept in file order, so sigma-off Specification == Baseline);
* N > 280  -> a bootstrap sample with replacement.

Both entry points (the app's sample_agents and the engine's CLI fallback) use the same
rule and the same default_rng(seed) stream. Each agent keeps its own RNG streams (base
seed from rng_pass1 by agent index).
"""
import numpy as np
import pandas as pd
import pytest

from src.data.participants import SURVEY_PATH, merged, original_data
from src.engine.core import Engine
from src.engine.profile import PROFILES
from src.engine.sampling import (cyclic_indices, load_original_participants,
                                 random_indices, research_population_text,
                                 sample_participants_internal)
from app.seam.execute import sample_agents

N_ORIG = 280


def _participant_ids():
    """Participant ID of each row of original_data(), in its order."""
    original = original_data()
    return merged().loc[original.index, "Participant ID"].tolist()


def _same_rows(agents: pd.DataFrame, positions) -> None:
    expected = original_data().iloc[list(positions)].reset_index(drop=True)
    pd.testing.assert_frame_equal(agents.reset_index(drop=True), expected)


# ---------------------------------------------------------------------------
# the selection rule
# ---------------------------------------------------------------------------
def test_n280_is_exactly_the_file_in_order():
    assert len(original_data()) == N_ORIG
    for pop in ("documentation", "baseline"):
        for seed in (0, 42, 12345):
            agents = sample_agents(pop, N_ORIG, seed)
            pd.testing.assert_frame_equal(agents, original_data().reset_index(drop=True))
    # original_data() keeps the survey workbook's row order (Participant IDs as in the file)
    survey_ids = pd.read_excel(SURVEY_PATH, sheet_name=0)["Participant ID"].tolist()
    ids = _participant_ids()
    assert ids == [pid for pid in survey_ids if pid in set(ids)]
    assert len(set(ids)) == N_ORIG


@pytest.mark.parametrize("n", [1000, 560, 281, 279, 60, 1])
def test_baseline_agent_k_is_participant_k_mod_280(n):
    ids = _participant_ids()
    for seed in (0, 42):
        agents = sample_agents("baseline", n, seed)
        assert list(agents.index) == list(range(n))
        _same_rows(agents, [k % N_ORIG for k in range(n)])
    agent_ids = [ids[k] for k in cyclic_indices(n, N_ORIG)]
    assert agent_ids == [ids[k % N_ORIG] for k in range(n)]
    if n == 1000:
        # three full cycles, then the first 160
        assert agent_ids[:840] == ids * 3 and agent_ids[840:] == ids[:160]
    if n < N_ORIG:
        assert agent_ids == ids[:n]


def _spec_positions(n, seed):
    """Row positions of the Research Specification agents (via the Participant IDs)."""
    ids = _participant_ids()
    pos = {pid: i for i, pid in enumerate(ids)}
    agents = sample_agents("documentation", n, seed)
    rows = original_data().reset_index(drop=True)
    # match each agent row to its participant row (rows are unique per participant)
    key = rows.astype(str).agg("|".join, axis=1)
    where = {k: i for i, k in enumerate(key)}
    got = [where[k] for k in agents.astype(str).agg("|".join, axis=1)]
    assert len(where) == N_ORIG
    return got


@pytest.mark.parametrize("n", [279, 200, 60, 1])
def test_spec_below_280_is_a_random_subset_without_replacement(n):
    a = _spec_positions(n, 42)
    assert len(a) == n and len(set(a)) == n                     # distinct participants
    assert a == _spec_positions(n, 42)                          # reproducible
    assert a == random_indices(n, N_ORIG, 42).tolist()
    if n > 1:
        assert a != _spec_positions(n, 43)                      # differs across seeds
        assert a != list(range(n))                              # not the first N
    _same_rows(sample_agents("documentation", n, 42), a)


@pytest.mark.parametrize("n", [281, 560, 1000])
def test_spec_above_280_is_a_bootstrap_with_replacement(n):
    a = _spec_positions(n, 42)
    assert len(a) == n
    assert len(set(a)) < n                                      # duplicates (with replacement)
    assert a == _spec_positions(n, 42)
    assert a == random_indices(n, N_ORIG, 42).tolist()
    assert a != _spec_positions(n, 7)
    assert a != [k % N_ORIG for k in range(n)]                  # not cyclic
    _same_rows(sample_agents("documentation", n, 42), a)


def test_spec_at_280_uses_all_280():
    for seed in (0, 1, 42):
        a = _spec_positions(N_ORIG, seed)
        assert sorted(a) == list(range(N_ORIG))


def test_both_entry_points_select_the_same_agents():
    """The engine's CLI fallback uses the app's rule and stream and leaves the setup
    RNG untouched; the seed is ignored by Research Baseline."""
    for random_sample, pop in ((True, "documentation"), (False, "baseline")):
        for n in (60, 280, 1000):
            internal = sample_participants_internal(original_data(), n, 42, random_sample)
            pd.testing.assert_frame_equal(internal, sample_agents(pop, n, 42))
            pd.testing.assert_frame_equal(load_original_participants(n, 42, random_sample),
                                          internal)
    reference = sample_agents("baseline", 1000, 0)
    for seed in (1, 42, 12345):
        pd.testing.assert_frame_equal(sample_agents("baseline", 1000, seed), reference)
    # the engine resolves agents through the same function
    for pop in ("documentation", "baseline"):
        engine = Engine(PROFILES[pop])
        rng = np.random.default_rng(7)
        state_before = rng.bit_generator.state
        resolved = engine._resolve_agents(None, 300, 42, rng)
        assert rng.bit_generator.state == state_before
        pd.testing.assert_frame_equal(resolved, sample_agents(pop, 300, 42))


def test_professor_income_follows_the_participant():
    incomes = original_data()["income"].to_numpy()
    agents = load_original_participants(1000)
    np.testing.assert_array_equal(agents["income"].to_numpy(),
                                  incomes[np.arange(1000) % N_ORIG])
    spec = load_original_participants(1000, 42, random_sample=True)
    np.testing.assert_array_equal(spec["income"].to_numpy(),
                                  incomes[random_indices(1000, N_ORIG, 42)])


def test_run_time_info_line():
    assert research_population_text(280) == "📊 Using original 280 participants"
    assert research_population_text(280, random_sample=True) == "📊 Using original 280 participants"
    assert "cycle" in research_population_text(1000)
    assert "k mod 280" in research_population_text(1000)
    assert research_population_text(60).startswith("📊 Using the first 60 of the original 280")
    assert "bootstrap" in research_population_text(1000, random_sample=True)
    assert "random subset" in research_population_text(60, random_sample=True)
    assert "order" not in research_population_text(60, random_sample=True)
    assert "order" not in research_population_text(1000, random_sample=True)


def test_page1_captions_do_not_claim_spec_is_in_order():
    from app.pages import page1_common_params as p1
    assert "randomly selects a unique subset" in p1.RESEARCH_SPEC_SAMPLING_NOTE
    assert "bootstrap" in p1.RESEARCH_SPEC_SAMPLING_NOTE
    assert "order" not in p1.RESEARCH_SPEC_SAMPLING_NOTE
    import inspect
    src = inspect.getsource(p1)
    spec_info = src[src.index('"📄 **Research Specification**'):][:400]
    assert "RESEARCH_SPEC_SAMPLING_NOTE" in spec_info and "RESEARCH_CYCLE_NOTE" not in spec_info


# ---------------------------------------------------------------------------
# engine consequences
# ---------------------------------------------------------------------------
def _baseline_d4(n):
    engine = Engine(PROFILES["baseline"])          # agents from the engine's internal path
    engine.config["rejected_transaction_defaults"]["income_mode"] = "continuous"
    return engine.run_simulation(n, 42, ["rejected_transaction_defaults"])


def test_d4_segment_shares_at_560_equal_280_exactly():
    a, b = _baseline_d4(280), _baseline_d4(560)
    for mech in ("loyalty", "wtp", "rt", "flex"):
        col = f"rtd_{mech}_segment"
        # every copy of a participant lands in the same segment ...
        np.testing.assert_array_equal(np.tile(a[col].to_numpy(), 2), b[col].to_numpy())
        # ... so the segment shares are identical
        pd.testing.assert_series_equal(a[col].value_counts(normalize=True).sort_index(),
                                       b[col].value_counts(normalize=True).sort_index())
    np.testing.assert_array_equal(np.tile(a["rtd_choice_length"].to_numpy(), 2),
                                  b["rtd_choice_length"].to_numpy())


def _full_run(pop, n, sigma_off=True):
    engine = Engine(PROFILES[pop])
    if sigma_off:
        for cfg in engine.config.values():
            if isinstance(cfg, dict) and isinstance(cfg.get("stochastic"), dict):
                cfg["stochastic"]["sigma_value"] = 0.0
                cfg["stochastic"]["in_copula"] = False
    return engine.run_simulation(n, 42, None, agents_df=sample_agents(pop, n, 42))


@pytest.mark.parametrize("n", [280])
def test_spec_sigma_off_equals_baseline_row_for_row(n):
    """At N = 280 both modes hold the 280 in file order, so with every sigma off they
    agree row for row (at other N, Specification samples randomly)."""
    spec, base = _full_run("documentation", n), _full_run("baseline", n)
    assert list(spec.columns) == list(base.columns)
    assert len(spec) == len(base) == n
    for col in spec.columns:
        assert spec[col].astype(str).tolist() == base[col].astype(str).tolist(), col


def test_spec_noise_differs_between_copies_of_a_participant():
    """Each repeated agent keeps its own RNG streams: with Research Specification noise
    on, two bootstrap copies of one participant do not share one draw."""
    engine = Engine(PROFILES["documentation"])
    cfg = engine.config["donation_default"]
    cfg.setdefault("stochastic", {})["sigma_value"] = float(cfg["stochastic"].get("sigma_overall", 9.899547))
    df = engine.run_simulation(560, 42, ["donation_default"],
                               agents_df=sample_agents("documentation", 560, 42))
    positions = random_indices(560, N_ORIG, 42)
    seen, pairs = {}, []
    for k, p in enumerate(positions):
        if p in seen:
            pairs.append((seen[p], k))
        seen[p] = k
    assert len(pairs) > 50
    rates = df["donation_default"].to_numpy()
    assert sum(not np.isclose(rates[i], rates[j]) for i, j in pairs) > 0.5 * len(pairs)   # (rates clipped at the bounds tie)
