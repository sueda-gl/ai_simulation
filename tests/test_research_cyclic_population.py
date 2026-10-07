"""
Owner ruling R-CYC (2026-10-07): the research population modes (Research Baseline AND
Research Specification) never bootstrap, resample or permute. Agent k (0-based) is
participant k mod 280, in the participants' real (file) order:

* N = 280  -> exactly the 280, in file order;
* N = 1000 -> three full cycles + the first 160;
* N < 280  -> the first N.

The selection draws no random numbers, so it does not depend on the seed and is the
same in both research modes (only Research Specification noise varies). Each repeated
agent keeps its own RNG streams (base seed from rng_pass1 by agent index).
"""
import numpy as np
import pandas as pd
import pytest

from src.data.participants import SURVEY_PATH, merged, original_data
from src.engine.core import Engine
from src.engine.profile import PROFILES
from src.engine.sampling import (cyclic_indices, load_original_participants,
                                 research_population_text, sample_participants_internal)
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
        agents = sample_agents(pop, N_ORIG, 42)
        pd.testing.assert_frame_equal(agents, original_data().reset_index(drop=True))
    # original_data() keeps the survey workbook's row order (Participant IDs as in the file)
    survey_ids = pd.read_excel(SURVEY_PATH, sheet_name=0)["Participant ID"].tolist()
    ids = _participant_ids()
    assert ids == [pid for pid in survey_ids if pid in set(ids)]
    assert len(set(ids)) == N_ORIG


@pytest.mark.parametrize("n", [1000, 560, 281, 279, 60, 1])
def test_agent_k_is_participant_k_mod_280(n):
    ids = _participant_ids()
    for pop in ("documentation", "baseline"):
        agents = sample_agents(pop, n, 42)
        assert list(agents.index) == list(range(n))
        _same_rows(agents, [k % N_ORIG for k in range(n)])
    agent_ids = [ids[k] for k in cyclic_indices(n, N_ORIG)]
    assert agent_ids == [ids[k % N_ORIG] for k in range(n)]
    if n == 1000:
        # three full cycles, then the first 160
        assert agent_ids[:840] == ids * 3 and agent_ids[840:] == ids[:160]
    if n < N_ORIG:
        assert agent_ids == ids[:n]


def test_selection_ignores_seed_and_mode_and_entry_point():
    reference = load_original_participants(1000)
    for seed in (0, 1, 42, 12345):
        for pop in ("documentation", "baseline"):
            pd.testing.assert_frame_equal(sample_agents(pop, 1000, seed), reference)
    # the engine's CLI fallback uses the same rule and draws nothing from the setup RNG
    rng = np.random.default_rng(7)
    state_before = rng.bit_generator.state
    internal = sample_participants_internal(original_data(), 1000, rng)
    assert rng.bit_generator.state == state_before
    pd.testing.assert_frame_equal(internal, reference)


def test_professor_income_follows_the_participant():
    agents = load_original_participants(1000)
    incomes = original_data()["income"].to_numpy()
    np.testing.assert_array_equal(agents["income"].to_numpy(),
                                  incomes[np.arange(1000) % N_ORIG])


def test_run_time_info_line():
    assert research_population_text(280) == "📊 Using original 280 participants"
    assert "cycle" in research_population_text(1000)
    assert "k mod 280" in research_population_text(1000)
    assert research_population_text(60).startswith("📊 Using the first 60 of the original 280")


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


@pytest.mark.parametrize("n", [280, 1000])
def test_spec_sigma_off_equals_baseline_row_for_row(n):
    spec, base = _full_run("documentation", n), _full_run("baseline", n)
    assert list(spec.columns) == list(base.columns)
    assert len(spec) == len(base) == n
    for col in spec.columns:
        assert spec[col].astype(str).tolist() == base[col].astype(str).tolist(), col


def test_spec_noise_differs_between_copies_of_a_participant():
    """Each repeated agent keeps its own RNG streams: with Research Specification noise
    on, the two copies of a participant at N=560 do not share one draw."""
    engine = Engine(PROFILES["documentation"])
    cfg = engine.config["donation_default"]
    cfg.setdefault("stochastic", {})["sigma_value"] = float(cfg["stochastic"].get("sigma_overall", 9.899547))
    df = engine.run_simulation(560, 42, ["donation_default"],
                               agents_df=sample_agents("documentation", 560, 42))
    first, second = df["donation_default"].to_numpy()[:280], df["donation_default"].to_numpy()[280:]
    assert not np.allclose(first, second)
