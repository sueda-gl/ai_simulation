"""
The Research Baseline / Research Specification populations must carry the professor's
income, not a freshly drawn one.

Both research modes run the 280 ORIGINAL participants. The professor's Stata files
carry ONE fixed per-participant `income` column (identical in
Stata_File_Decision4_050926.dta, Stata_File_Decision 2_260626 - CORRECTED.dta and
Stata_File_Decision 1.dta), and every income-using result in his documents is computed
with it. src/validate_traits.py therefore merges data/stata_incomes.csv into `merged`
by Participant ID, the two research orchestrators carry that column into
`original_data`, and income_utils.get_agent_income() finds it already cached.

What this pins down:
  (a) merged['income'] equals the professor's income participant-by-participant;
  (b) a Research Baseline Decision-4 run (all intercepts 0, deterministic) reproduces
      the .dta's WTP_calculated15 168/82/26/3/1 and RT_calculated15 20/100/110/46/4
      distributions AND the stored choice1..5 columns 280/280, NaN pattern included -
      the income-free elements (TTP, Loyalty, Flexibility) are unaffected, and
      z_net_income (sample SD) reproduces the .dta column to float32 tolerance;
  (c) COPULA populations are untouched: copula-sampled agents carry no income column,
      so income is still drawn from the Page-1 distribution, bit-identical to the
      behaviour before the change (values pinned below for seed 42).
"""
import os

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCOMES_CSV = os.path.join(REPO, "data", "stata_incomes.csv")
STDACTIONS_CSV = os.path.join(REPO, "data", "stata_stdactions.csv")
D4_CSV = os.path.join(REPO, "data", "stata_d4_verification.csv")
FLEX_CSV = os.path.join(REPO, "data", "stata_d4_flexibility_verification.csv")
PROF_DTA = "/Users/suedagul/Downloads/Stata_File_Decision4_050926.dta"

# .dta ground-truth distributions (Stata_File_Decision4_050926.dta). Only the two
# income-using elements move when the professor's income replaces a generated draw;
# the other three are listed so a regression in either direction is caught here too.
EXPECTED_WTP_DIST = {1: 168, 2: 82, 3: 26, 4: 3, 5: 1}
EXPECTED_RT_DIST = {1: 20, 2: 100, 3: 110, 4: 46, 5: 4}
EXPECTED_LOYALTY_DIST = {1: 12, 2: 87, 3: 148, 4: 31, 5: 2}
EXPECTED_FLEX_DIST = {1: 17, 2: 85, 3: 137, 4: 37, 5: 4}
EXPECTED_LENGTH_DIST = {0: 20, 1: 92, 2: 95, 3: 57, 4: 14, 5: 2}

# float32 storage tolerance of the .dta's z columns
ATOL = 5e-6

# Copula incomes for seed 42 with the shipped config/simulation.yaml distribution,
# captured from the code BEFORE income was attached to the research populations.
COPULA_SEED = 42
COPULA_N = 60
COPULA_INCOMES_FIRST10 = [
    10109.069650403353, 13655.23255836024, 27660.638012987067, 23106.803354348584,
    53849.243615519415, 12072.23441895969, 32899.82182864339, 11833.041830769616,
    12885.27250540582, 7783.318648051882,
]
COPULA_ALLOWANCE_FIRST10 = [12.0, 12.0, 128.0, 72.0, 200.0, 12.0, 128.0, 12.0, 12.0, 12.0]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def merged_df():
    from src.validate_traits import merged
    return merged


@pytest.fixture(scope="module")
def frozen_incomes():
    # round_trip: pandas' default CSV float parser is not correctly rounded and loses
    # the last ULP on two of the 280 values (same reader as src/validate_traits.py).
    return pd.read_csv(INCOMES_CSV, float_precision="round_trip")


@pytest.fixture(scope="module")
def gold():
    """The frozen Decision-4 extract of the professor's 050926 file (280 rows)."""
    main = pd.read_csv(D4_CSV)
    flex = pd.read_csv(FLEX_CSV)
    assert (main["participantid"].values == flex["participantid"].values).all()
    return main.join(flex[[c for c in flex.columns if c not in main.columns]])


@pytest.fixture(scope="module")
def baseline_d4():
    """One Research Baseline Decision-4 run on all 280 participants, deterministic,
    every intercept 0 (the .dta embeds no intercepts)."""
    from src.orchestrator_baseline import OrchestratorBaseline

    orch = OrchestratorBaseline()
    cfg = orch.config["rejected_transaction_defaults"]
    cfg["model_enabled"] = True
    cfg["income_mode"] = "continuous"
    cfg["intercepts"] = dict(cfg.get("intercepts", {}),
                             **{m: 0.0 for m in ("ttp", "loyalty", "wtp",
                                                 "risk_taking", "flexibility")})
    stoch = cfg.setdefault("stochastic", {})
    stoch["in_copula"] = False
    stoch["sigma_value"] = 0.0
    n = len(orch.original_data)
    assert n == 280
    return orch.run_simulation(n, 42, ["rejected_transaction_defaults"])


# ---------------------------------------------------------------------------
# (a) the frozen income file and the merged table
# ---------------------------------------------------------------------------
def test_frozen_income_file_layout():
    """data/stata_incomes.csv carries Participant ID, keeps original_index (read by
    tests/test_disclose_documents.py) and stays sorted by original_index."""
    inc = pd.read_csv(INCOMES_CSV, float_precision="round_trip")
    assert list(inc.columns) == ["Participant ID", "original_index", "income"]
    assert len(inc) == 280
    assert inc["original_index"].tolist() == list(range(280))
    assert inc["Participant ID"].is_unique
    assert inc["income"].notna().all()

    # the Participant ID <-> original_index join agrees with data/stata_stdactions.csv
    std = pd.read_csv(STDACTIONS_CSV).sort_values("original_index").reset_index(drop=True)
    assert (inc["Participant ID"].to_numpy() == std["Participant ID"].to_numpy()).all()
    assert (inc["original_index"].to_numpy() == std["original_index"].to_numpy()).all()


def test_merged_carries_the_professor_income(merged_df, frozen_incomes):
    """merged['income'] is the professor's income, joined by Participant ID, for all
    280 participants - no NaN, no positional accident."""
    assert "income" in merged_df.columns, \
        "src/validate_traits.py must merge data/stata_incomes.csv into `merged`"
    assert merged_df["income"].notna().all()

    by_pid = frozen_incomes.set_index("Participant ID")["income"]
    expected = by_pid.loc[merged_df["Participant ID"].to_numpy()].to_numpy()
    assert np.array_equal(merged_df["income"].to_numpy(), expected)

    # `merged` row order IS the .dta row order, i.e. original_index
    assert (merged_df["Participant ID"].to_numpy()
            == frozen_incomes["Participant ID"].to_numpy()).all()


@pytest.mark.skipif(not os.path.exists(PROF_DTA),
                    reason="professor's Decision 4 .dta not present")
def test_merged_income_equals_the_dta_income_per_participant(merged_df):
    """Exact, participant-by-participant, against the .dta itself."""
    dta = pd.read_stata(PROF_DTA, convert_categoricals=False)
    dta_income = pd.Series(dta["income"].to_numpy(dtype=float),
                           index=dta["participantid"].to_numpy().astype(int))
    got = merged_df.set_index(merged_df["Participant ID"].astype(int))["income"]
    assert set(got.index) == set(dta_income.index)
    assert np.array_equal(got.to_numpy(), dta_income.loc[got.index].to_numpy())


# ---------------------------------------------------------------------------
# (b) the Research Baseline Decision-4 run now reproduces the .dta
# ---------------------------------------------------------------------------
def test_baseline_run_uses_the_professor_income(baseline_d4, frozen_incomes):
    """The simulated population's income column IS the professor's column, and the
    12-200 allowance credit is still derived (get_agent_income's cached path)."""
    assert np.array_equal(baseline_d4["income"].to_numpy(),
                          frozen_incomes["income"].to_numpy())
    assert "actual_allowance" in baseline_d4.columns
    assert baseline_d4["actual_allowance"].notna().all()
    credit = {1: 12.0, 2: 32.0, 3: 72.0, 4: 128.0, 5: 200.0}
    expected = [credit[int(lvl)] for lvl in baseline_d4["Assigned Allowance Level"]]
    assert baseline_d4["actual_allowance"].tolist() == expected


def test_baseline_z_net_income_matches_stata(baseline_d4, gold):
    """z_net_income is `egen std(income)` (SAMPLE SD) over the professor's incomes."""
    z_app = baseline_d4["rtd_z_income"].to_numpy(dtype=float)
    assert np.allclose(z_app, gold["z_net_income"].to_numpy(dtype=float), atol=ATOL)
    # and it really is the sample-SD standardization of the stored incomes
    inc = gold["income"].to_numpy(dtype=float)
    assert np.allclose(z_app, (inc - inc.mean()) / inc.std(ddof=1), atol=ATOL)


def test_baseline_wtp_and_rt_distributions_match_the_dta(baseline_d4):
    """The whole point of the change: WTP 168/82/26/3/1 and RT 20/100/110/46/4."""
    wtp = baseline_d4["rtd_wtp_segment"].value_counts().to_dict()
    rt = baseline_d4["rtd_rt_segment"].value_counts().to_dict()
    assert {int(k): int(v) for k, v in wtp.items()} == EXPECTED_WTP_DIST
    assert {int(k): int(v) for k, v in rt.items()} == EXPECTED_RT_DIST


def test_baseline_income_free_elements_are_unchanged(baseline_d4):
    """TTP, Loyalty and Flexibility use no income - they must not move."""
    lengths = baseline_d4["rtd_choice_length"].to_numpy()
    assert {k: int((lengths == k).sum()) for k in range(6)} == EXPECTED_LENGTH_DIST
    loyalty = baseline_d4["rtd_loyalty_segment"].value_counts().to_dict()
    flex = baseline_d4["rtd_flex_segment"].value_counts().to_dict()
    assert {int(k): int(v) for k, v in loyalty.items()} == EXPECTED_LOYALTY_DIST
    assert {int(k): int(v) for k, v in flex.items()} == EXPECTED_FLEX_DIST


def test_baseline_segments_and_choice_columns_match_the_dta_280_of_280(baseline_d4, gold):
    """Every stored segment column AND every stored choice1..5 column, participant by
    participant, NaN pattern included."""
    for key, seg_col, suffix in (
            ("wtp", "WTP_calculated15", "_WTP_deterministic"),
            ("rt", "RT_calculated15", "_RT_deterministic"),
            ("loyalty", "weighted_loyalty15", "_loyalty_deterministic"),
            ("flex", "Flexibility_combined15", "_flex_deterministic")):
        segs = baseline_d4[f"rtd_{key}_segment"].to_numpy(dtype=int)
        stata_segs = gold[seg_col].to_numpy(dtype=float).astype(int)
        assert int((segs == stata_segs).sum()) == 280, f"{key} segments"

        rankings = list(baseline_d4[f"rtd_{key}_ranking"])
        for pos in range(1, 6):
            stored = gold[f"choice{pos}{suffix}"].to_numpy(dtype=float)
            matched = 0
            for i, ranking in enumerate(rankings):
                if pos <= len(ranking):
                    matched += int(stored[i] == ranking[pos - 1])
                else:
                    matched += int(np.isnan(stored[i]))
            assert matched == 280, f"{key} choice{pos}"

    lengths = baseline_d4["rtd_choice_length"].to_numpy(dtype=int)
    stata_len = gold["choice_length_deterministic"].to_numpy(dtype=float).astype(int)
    assert int((lengths == stata_len).sum()) == 280, "TTP option-list length"


# ---------------------------------------------------------------------------
# (c) Copula populations keep the generated income
# ---------------------------------------------------------------------------
def test_copula_sampled_agents_carry_no_income_column():
    """Structural guarantee: the copula emits traits only, so get_agent_income() still
    takes the generation path for copula agents."""
    from src.trait_engine import TraitEngine

    agents = TraitEngine().sample(COPULA_N, COPULA_SEED)
    assert "income" not in agents.columns
    assert "actual_allowance" not in agents.columns


def test_copula_incomes_unchanged_for_seed_42():
    """Regression pin: the copula population's incomes (and the derived 12-200 credit)
    for seed 42 are bit-identical to the behaviour before the research populations
    started carrying the professor's income."""
    from src.orchestrator import Orchestrator

    df = Orchestrator().run_simulation(COPULA_N, COPULA_SEED, ["donation_default"])
    assert df["income"].tolist()[:10] == COPULA_INCOMES_FIRST10
    assert df["actual_allowance"].tolist()[:10] == COPULA_ALLOWANCE_FIRST10
    # not the professor's incomes
    frozen = pd.read_csv(INCOMES_CSV, float_precision="round_trip")["income"].to_numpy()
    assert not np.array_equal(df["income"].to_numpy(), frozen[:COPULA_N])
