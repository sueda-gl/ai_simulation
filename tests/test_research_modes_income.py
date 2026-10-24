"""
Where each population's continuous income comes from, and the Decision 4 reference parity.

Every population mode - Copula, Research Specification, Research Baseline - draws each
agent's continuous income from the Page 1 distribution, within the band of the agent's
allowance level (src.engine.income.get_agent_income, seeded per agent), as the deployed
app always did. Changing the Page 1 income settings therefore moves every continuous
result in every mode; the categorical specification uses the allowance level only.

The reference Stata files (Decision 1 / 2 / 4) embed ONE fixed income per participant,
frozen in data/stata_incomes.csv (a February 2026 export made to compare the app with
Stata). The parity tests inject that column into the agents they run, so the Decision 4
Research Baseline run reproduces the .dta 280/280; the app itself never loads it. From
6 to 24 October 2026 src/data/participants.py merged the file into the research
populations, which silently made Page 1's income settings inert for the Research modes;
that is reverted and pinned here.

What this pins down:
  (a) the frozen file's layout, and that the research population carries NO income
      column (merged() and original_data());
  (b) a Research Baseline Decision-4 run with the reference income INJECTED (all
      intercepts 0, deterministic) reproduces the frozen extract of the .dta
      (data/stata_d4_verification.csv and data/stata_d4_flexibility_verification.csv):
      WTP_calculated15 168/82/26/3/1 and RT_calculated15 20/100/110/46/4, every stored
      segment and choice column 280/280, z_net_income to float32 tolerance;
  (c) without injection the research modes generate the income from Page 1: it is
      reproducible for a seed, differs from the reference file, and moves with the
      Page 1 lognormal sigma;
  (d) COPULA populations are untouched (values pinned for seed 42);
  (e) the same injected run in categorical income reproduces the document's recomputed
      WTP / RT segments and choice lists 280/280.
"""
import os

import numpy as np
import pandas as pd
import pytest

from conftest import REPO, require_reference_file

INCOMES_CSV = os.path.join(REPO, "data", "stata_incomes.csv")
STDACTIONS_CSV = os.path.join(REPO, "data", "stata_stdactions.csv")
D4_CSV = os.path.join(REPO, "data", "stata_d4_verification.csv")
FLEX_CSV = os.path.join(REPO, "data", "stata_d4_flexibility_verification.csv")
PROF_DTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stata", "Stata_File_Decision4_050926.dta")

# .dta ground-truth distributions (Stata_File_Decision4_050926.dta). Only the two
# income-using elements move when the reference income replaces a generated draw;
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
    from src.data.participants import merged
    return merged()


@pytest.fixture(scope="module")
def frozen_incomes():
    # round_trip: pandas' default CSV float parser is not correctly rounded and loses
    # the last ULP on two of the 280 values (same reader as src/data/participants.py).
    return pd.read_csv(INCOMES_CSV, float_precision="round_trip")


@pytest.fixture(scope="module")
def gold():
    """The frozen Decision-4 extract of the reference 050926 file (280 rows)."""
    main = pd.read_csv(D4_CSV)
    flex = pd.read_csv(FLEX_CSV)
    assert (main["participantid"].values == flex["participantid"].values).all()
    return main.join(flex[[c for c in flex.columns if c not in main.columns]])


@pytest.fixture(scope="module")
def baseline_d4():
    """One Research Baseline Decision-4 run on all 280 participants, deterministic,
    every intercept 0 (the .dta embeds no intercepts)."""
    return _baseline_run("continuous")


@pytest.fixture(scope="module")
def baseline_d4_categorical():
    """The same run in categorical-income mode."""
    return _baseline_run("categorical")


def _reference_agents(orch):
    """original_data() with the reference income INJECTED by Participant ID (the app never
    does this; the parity tests do, so the run reproduces the .dta)."""
    from src.data import participants as P
    agents = orch.original_data.copy()
    pids = P.merged().loc[agents.index, "Participant ID"].astype(int).to_numpy()
    by_pid = pd.read_csv(INCOMES_CSV, float_precision="round_trip").set_index("Participant ID")["income"]
    agents["income"] = by_pid.loc[pids].to_numpy()
    agents.index = range(len(agents))
    return agents


def _baseline_engine(income_mode):
    from src.engine.core import Engine
    from src.engine.profile import PROFILES

    orch = Engine(PROFILES["baseline"])
    cfg = orch.config["rejected_transaction_defaults"]
    cfg["income_mode"] = income_mode
    cfg["intercepts"] = dict(cfg.get("intercepts", {}),
                             **{m: 0.0 for m in ("ttp", "loyalty", "wtp",
                                                 "risk_taking", "flexibility")})
    stoch = cfg.setdefault("stochastic", {})
    stoch["in_copula"] = False
    stoch["sigma_value"] = 0.0
    return orch


def _baseline_run(income_mode):
    orch = _baseline_engine(income_mode)
    agents = _reference_agents(orch)
    assert len(agents) == 280
    return orch.run_simulation(len(agents), 42, ["rejected_transaction_defaults"], agents_df=agents)


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


def test_research_population_carries_no_income_column(merged_df):
    """The reference income is never merged into the population: the engine generates
    every participant's income from the Page 1 distribution (src/data/participants.py)."""
    from src.data.participants import original_data
    assert "income" not in merged_df.columns
    assert "income" not in original_data().columns
    # `merged` row order IS the .dta row order, i.e. original_index
    frozen = pd.read_csv(INCOMES_CSV, float_precision="round_trip")
    assert (merged_df["Participant ID"].to_numpy() == frozen["Participant ID"].to_numpy()).all()


def test_frozen_income_equals_the_dta_income_per_participant():
    """The frozen file is the .dta's income column, participant by participant."""
    require_reference_file(PROF_DTA, "PROF_DTA in tests/test_research_modes_income.py")
    dta = pd.read_stata(PROF_DTA, convert_categoricals=False)
    dta_income = pd.Series(dta["income"].to_numpy(dtype=float),
                           index=dta["participantid"].to_numpy().astype(int))
    frozen = pd.read_csv(INCOMES_CSV, float_precision="round_trip").set_index("Participant ID")["income"]
    assert set(frozen.index) == set(dta_income.index)
    assert np.array_equal(frozen.to_numpy(), dta_income.loc[frozen.index].to_numpy())


# ---------------------------------------------------------------------------
# (b) the Research Baseline Decision-4 run with the reference income injected reproduces the .dta
# ---------------------------------------------------------------------------
def test_injected_reference_income_is_kept(baseline_d4, frozen_incomes):
    """An income pre-attached to the agents is kept (get_agent_income's cached path) and
    the 12-200 allowance credit is still derived."""
    assert np.array_equal(baseline_d4["income"].to_numpy(),
                          frozen_incomes["income"].to_numpy())
    assert "actual_allowance" in baseline_d4.columns
    assert baseline_d4["actual_allowance"].notna().all()
    credit = {1: 12.0, 2: 32.0, 3: 72.0, 4: 128.0, 5: 200.0}
    expected = [credit[int(lvl)] for lvl in baseline_d4["Assigned Allowance Level"]]
    assert baseline_d4["actual_allowance"].tolist() == expected


def test_baseline_z_net_income_matches_stata(baseline_d4, gold):
    """z_net_income is `egen std(income)` (SAMPLE SD) over the reference incomes."""
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
# (c) without injection the research modes generate the income from Page 1
# ---------------------------------------------------------------------------
def _research_incomes(profile_name, seed=42, **page1):
    from src.engine.core import Engine
    from src.engine.profile import PROFILES

    orch = Engine(PROFILES[profile_name])
    orch.simulation_config.setdefault("simulation", {}).update(page1)
    df = orch.run_simulation(280, seed, ["donation_default"])
    return df["income"].to_numpy(dtype=float), df["Assigned Allowance Level"].to_numpy(dtype=int)


@pytest.mark.parametrize("profile_name", ["baseline", "documentation"])
def test_research_modes_generate_income_from_page1(profile_name, frozen_incomes):
    """The app's research runs do not carry the reference income: the engine draws it
    from the Page 1 distribution, reproducibly for a seed, inside the allowance band."""
    inc, levels = _research_incomes(profile_name)
    again, _ = _research_incomes(profile_name)
    assert np.array_equal(inc, again)                       # seeded
    assert not np.array_equal(inc, frozen_incomes["income"].to_numpy())
    assert np.isfinite(inc).all() and (inc > 0).all()
    # category-first: every level-1 income below every level-5 income
    assert inc[levels == 1].max() < inc[levels == 5].min()


@pytest.mark.parametrize("profile_name", ["baseline", "documentation"])
def test_page1_income_settings_move_the_research_modes(profile_name):
    """Changing the Page 1 lognormal sigma changes the research population's incomes
    (from 6 to 24 October 2026 it did not: the loader had merged a fixed income)."""
    narrow, _ = _research_incomes(profile_name, lognormal_sigma=0.5)
    wide, _ = _research_incomes(profile_name, lognormal_sigma=1.0)
    assert not np.array_equal(narrow, wide)
    assert wide.std(ddof=1) > 1.5 * narrow.std(ddof=1)


# ---------------------------------------------------------------------------
# (d) Copula populations keep the generated income
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
    for seed 42, unchanged since before October 2026."""
    from src.engine.core import Engine

    df = Engine("copula").run_simulation(COPULA_N, COPULA_SEED, ["donation_default"])
    assert df["income"].tolist()[:10] == COPULA_INCOMES_FIRST10
    assert df["actual_allowance"].tolist()[:10] == COPULA_ALLOWANCE_FIRST10
    # not the reference incomes
    frozen = pd.read_csv(INCOMES_CSV, float_precision="round_trip")["income"].to_numpy()
    assert not np.array_equal(df["income"].to_numpy(), frozen[:COPULA_N])


# ---------------------------------------------------------------------------
# (e) categorical income through the engine (Research Baseline, all 280, reference income injected)
# ---------------------------------------------------------------------------
def _seg15(v):
    v = np.asarray(v, dtype=float)
    return np.floor(1 + (5 - 0.0001) * (v - v.min()) / (v.max() - v.min())).astype(int)


def _doc_categorical_segments(gold):
    """Doc rev 040926-2 cond() construction, recomputed from the RAW inputs with
    egen std (N-1) z-scores - z_agreeable in ALL five branches."""
    def z(c):
        x = gold[c].to_numpy(dtype=float)
        return (x - x.mean()) / x.std(ddof=1)
    zE, zA, zN = z("extraversionbig5"), z("agreeable"), z("neuroticismbig5")
    zC, zO = z("conscientiousnessbig5"), z("opennessbig5")
    lvl = gold["assignedallowancelevel"].to_numpy(dtype=int)
    w_eff = np.array([{1: 0.0, 2: 0.2671588, 3: 0.5057716, 4: 0.9411747, 5: 1.822471}[l]
                      for l in lvl])
    r_eff = np.array([{1: 0.0, 2: 0.0026128, 3: 0.0050555, 4: 0.0092738, 5: 0.0179812}[l]
                      for l in lvl])
    wtp = w_eff - 0.691843 + 0.078863062 * zE - 0.012326128 * zA
    rt = (r_eff - 0.0068307 + 0.025942386297 * zE + 0.023699214948 * zO
          - 0.038734315188 * zA - 0.037739440732 * zC - 0.025388697852 * zN)
    return _seg15(wtp), _seg15(rt)


def test_baseline_categorical_matches_the_doc_280_of_280(baseline_d4_categorical, gold):
    """Engine run, categorical income: WTP and RT segments AND choice lists equal the
    doc-correct recomputation 280/280; TTP / Loyalty / Flexibility are unchanged."""
    wtp_doc, rt_doc = _doc_categorical_segments(gold)
    df = baseline_d4_categorical
    assert (df["rtd_wtp_segment"].to_numpy(dtype=int) == wtp_doc).all()
    assert (df["rtd_rt_segment"].to_numpy(dtype=int) == rt_doc).all()
    for key, seq, segs in (("wtp", [3, 2, 1, 4, 5], wtp_doc), ("rt", [4, 2, 1, 3, 5], rt_doc)):
        assert [list(r) for r in df[f"rtd_{key}_ranking"]] == [seq[5 - s:] for s in segs]
    assert (df["rtd_choice_length"].to_numpy(dtype=int)
            == gold["choice_length_deterministic"].to_numpy(dtype=int)).all()
    assert (df["rtd_loyalty_segment"].to_numpy(dtype=int)
            == gold["weighted_loyalty15"].to_numpy(dtype=int)).all()
    assert (df["rtd_flex_segment"].to_numpy(dtype=int)
            == gold["Flexibility_combined15"].to_numpy(dtype=int)).all()


def test_baseline_categorical_vs_the_file(baseline_d4_categorical, gold):
    """Against the FILE's stored categorical scores: RT 280/280; WTP 279/280 because the
    .dta's weighted_WTP_categorical uses RAW `agreeable` in the level-1 branch (57 rows,
    a file error - the doc uses z_agreeable). Through the population min/max that error
    moves participant 716 (level 3) across a segment boundary."""
    df = baseline_d4_categorical
    rt_file = _seg15(gold["z_RT_categorical"])
    assert (df["rtd_rt_segment"].to_numpy(dtype=int) == rt_file).all()
    wtp_file = _seg15(gold["z_WTP_categorical"])
    diff = np.flatnonzero(df["rtd_wtp_segment"].to_numpy(dtype=int) != wtp_file)
    assert gold["participantid"].to_numpy(dtype=int)[diff].tolist() == [716]
    np.testing.assert_allclose(df["rtd_rt_score"].to_numpy(dtype=float),
                               gold["weighted_RT_categorical"].to_numpy(dtype=float), atol=ATOL)
