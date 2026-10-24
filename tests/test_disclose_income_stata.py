"""
Validation tests for Decision 1: Disclose Income against the CORRECTED reference Stata run.

Reference data: "Decision 1 - Disclosure of Income 110226_Final.docx" and its data file
"Stata_File_Decision 1_Updated.dta" (280 participants). The corrected columns are
disclose_categorical (167 Y), disclose_cont (169 Y) and fs_deterministic_*; the old error
model kept in the *_err columns is NOT the reference.

The model uses exactly the constants of the Decision 1 document of 11 Feb 2026 (revision
110226_Final, with the corrected Extraversion weight): Extraversion 0.00680238, level
intercepts 0.0089007 ... -0.0145324, weighted_disclosure_categorical SD 0.025040462. That
document applies the Extraversion weight typo fix 641.15 -> 645.15; the reference .dta does
not, so the two runs differ slightly. These tests DOCUMENT that known, accepted difference:
with beta0 = 0.1, continuous matches 280/280, categorical matches 279/280 with the single
mismatch participant 848 (a near-threshold case), and the fs scores agree within
FS_TOL_ANDREI. The anchored_pb SD is NOT a document value: since 2026-10-22 it is the sample
SD (ddof=1, Stata `egen std()`) of anchored_pb over the 280 of the reference .dta,
0.7971466806 (the 0.7984211971 used before had no source), so it equals the .dta's own
column to ~1e-8. The app default beta0 stays 0.75.

The corrected Stata columns are frozen in data/stata_d1_verification.csv, so the parity
tests run without the reference .dta; test_frozen_csv_equals_professor_dta checks that the
frozen CSV still equals the .dta at PROF_DTA, and fails when that file is missing unless
COOPECON_ALLOW_DTA_SKIP=1.
"""
import os
import copy
import numpy as np
import pandas as pd
import pytest

from src.decisions.disclose_income_stochastic import (
    disclose_income_stochastic, compute_continuous_de_stats,
)

from conftest import REPO, load_decisions_yaml, require_reference_file

SURVEY = os.path.join(REPO, "data", "Student Survey Results - Period 1.xlsx")
EXPERIMENT = os.path.join(REPO, "data", "Student Experiment Results - Period 1-2.xlsx")
FROZEN = os.path.join(REPO, "data", "stata_d1_verification.csv")
PROF_DTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stata", "Stata_File_Decision 1_Updated.dta")

STATA_BETA0 = 0.1        # `gen beta0 = 0.1` in the corrected reference Stata code
APP_DEFAULT_BETA0 = 0.75  # the app's research default (kept)
# Measured max |fs_model - fs_stata| with the 11 Feb 2026 document's constants and the
# data-derived anchored_pb SD: 0.003341 (categorical) and 0.003301 (continuous), both at
# participant 838 (0.003414 / 0.003373 with the sourceless 0.7984211971). The bound sits just above.
FS_TOL_ANDREI = 3.5e-3
KNOWN_CATEGORICAL_MISMATCHES = [848]  # model Y (fs +0.00046) vs Stata N (fs -0.00022)
SEPTEMBER_SD_CATEGORICAL = 0.025040462   # the 11 Feb 2026 document's value (reflects the Extraversion fix)
DATA_SD_ANCHORED_PB = 0.7971466806       # sample SD of anchored_pb over the 280 (config/decisions.yaml comment)

FROZEN_COLS = [
    "participantid", "assignedallowancelevel", "income", "income_high", "income_high_cat",
    "weighted_disclosure_categorical", "weighted_disclosure_cont", "anchored_prosocial_behavior",
    "fs_deterministic_categorical", "fs_deterministic_cont",
    "disclose_categorical", "disclose_cont",
]


def _load_stata():
    """The corrected Stata columns, frozen in data/stata_d1_verification.csv (in git)."""
    return pd.read_csv(FROZEN, float_precision="round_trip")


@pytest.fixture(scope="module")
def stata():
    s = _load_stata()
    assert len(s) == 280
    return s


@pytest.fixture(scope="module")
def merged(stata):
    survey = pd.read_excel(SURVEY)
    exp = pd.read_excel(EXPERIMENT)
    m = survey.merge(exp, on="Participant ID", how="inner")
    m = m[m["Participant ID"].notna()].reset_index(drop=True)
    m["Participant ID"] = m["Participant ID"].astype(int)
    s = stata.copy()
    s["participantid"] = s["participantid"].astype(int)
    # match by participant id (never by row order)
    m = m.merge(s, left_on="Participant ID", right_on="participantid", how="inner", validate="one_to_one")
    assert len(m) == 280, f"expected 280 matched participants, got {len(m)}"
    assert (m["Assigned Allowance Level"].astype(int) == m["assignedallowancelevel"].astype(int)).all()
    return m


@pytest.fixture(scope="module")
def di_params():
    return load_decisions_yaml()["disclose_income"]


def _run(merged, di_params, income_mode):
    """Run Decision 1 deterministically (no noise) with beta0 = 0.1 on the 280 participants."""
    params = copy.deepcopy(di_params)
    params["intercept"] = STATA_BETA0
    params["income_mode"] = income_mode
    params.setdefault("stochastic", {})["sigma_value"] = 0.0

    # the reference file's own income realization; SD with ddof=1 like Stata `egen std()`
    inc = merged["income"].astype(float).values
    sim_config = {
        "income_median": float(np.median(inc)),
        "income_stats": {"mean": float(np.mean(inc)), "sd": float(np.std(inc, ddof=1))},
    }
    if "continuous" in income_mode.lower():
        sim_config["di_cont_de_stats"] = compute_continuous_de_stats(merged, list(inc), params, sim_config)

    rng = np.random.default_rng(0)
    decision, fs, high = [], [], []
    for i, (_, row) in enumerate(merged.iterrows()):
        agent = row.to_dict()
        agent["income"] = float(inc[i])
        out = disclose_income_stochastic(agent, params, rng, sim_config, pop_context="baseline")
        decision.append(1 if out["disclose_income"] == "Y" else 0)
        fs.append(out["disclose_income_di"])
        high.append(out["disclose_income_income_high"])
    return np.array(decision), np.array(fs), np.array(high)


def test_categorical_matches_stata_except_known_848(merged, di_params):
    """Known, accepted difference: 279/280, the one mismatch is participant 848."""
    decision, fs, high = _run(merged, di_params, "Categorical only")
    ref = merged["disclose_categorical"].astype(int).values
    mismatched = merged["Participant ID"].values[decision != ref].tolist()
    assert mismatched == KNOWN_CATEGORICAL_MISMATCHES, f"categorical mismatches (participant ids): {mismatched}"
    assert int((decision == ref).sum()) == 279
    assert int(decision.sum()) == 168  # Stata 167 + participant 848
    assert (high == merged["income_high_cat"].astype(int).values).all()
    np.testing.assert_allclose(fs, merged["fs_deterministic_categorical"].astype(float).values,
                               rtol=0, atol=FS_TOL_ANDREI)


def test_continuous_matches_stata_280_of_280(merged, di_params):
    decision, fs, high = _run(merged, di_params, "continuous")
    ref = merged["disclose_cont"].astype(int).values
    mismatched = merged["Participant ID"].values[decision != ref].tolist()
    assert mismatched == [], f"continuous mismatches (participant ids): {mismatched}"
    assert int(decision.sum()) == 169
    assert (high == merged["income_high"].astype(int).values).all()
    np.testing.assert_allclose(fs, merged["fs_deterministic_cont"].astype(float).values,
                               rtol=0, atol=FS_TOL_ANDREI)


def test_composite_sds(stata, di_params):
    """weighted_disclosure_categorical's SD is the 11 Feb 2026 document's value, not Stata's
    `egen std()` of the reference .dta (0.0250386349) -- the documented, accepted difference
    (the Extraversion fix). anchored_pb's SD is the data-derived value: the sample SD of the
    .dta's own anchored_prosocial_behavior column (float32: 0.7971466830), to ~1e-8."""
    cz = di_params["composite_z_scoring"]
    assert cz["weighted_disclosure_categorical"]["sd"] == SEPTEMBER_SD_CATEGORICAL
    assert cz["anchored_pb"]["sd"] == DATA_SD_ANCHORED_PB
    # the difference from the .dta is small (< 0.2%) and is the accepted one
    assert cz["weighted_disclosure_categorical"]["sd"] == pytest.approx(
        stata["weighted_disclosure_categorical"].astype(float).std(ddof=1), rel=2e-3)
    # computed from the data: equal to the .dta column's sample SD to float32 precision
    assert cz["anchored_pb"]["sd"] == pytest.approx(
        stata["anchored_prosocial_behavior"].astype(float).std(ddof=1), abs=1e-7)


def test_september_constants(di_params):
    """Decision 1 uses exactly the constants of the 11 Feb 2026 document."""
    assert di_params["equation2_coefficients"]["extraversion"] == 0.00680238
    assert [di_params["categorical_intercepts"][f"level_{k}"] for k in range(1, 6)] == [
        0.0089007, 0.0055352, 0.0023109, -0.0032216, -0.0145324]


def test_shipped_default_intercept_unchanged(di_params):
    """Matching Stata is done by setting beta0 = 0.1; the app default stays 0.75."""
    assert di_params["intercept"] == APP_DEFAULT_BETA0


def test_quintile_multipliers_default_to_the_uniform_multiplier(di_params):
    """The five per-level σ multipliers default to the uniform one (scale_factor), so
    switching the σ mode to quintiles does not by itself change the noise; and both are
    1.0, one unit of the observed variability, like Decisions 2 and 3."""
    stochastic = di_params["stochastic"]
    assert stochastic["scale_factor"] == 1.0
    assert stochastic["quintile_scale_factors"] == {level: stochastic["scale_factor"] for level in "12345"}


class _SigmaRecorder:
    """An rng whose normal() returns the mean and records the sigma it was asked for."""
    def __init__(self):
        self.sigmas = []

    def normal(self, mean, sigma):
        self.sigmas.append(float(sigma))
        return mean

    def random(self):
        return 0.5


@pytest.mark.parametrize("strategy, level, coefficient", [
    ("overall", 3, 1.0), ("overall", 3, 0.25), ("quintile", 1, 1.0), ("quintile", 5, 0.5)])
def test_sigma_is_drawn_in_the_anchors_units(merged, di_params, strategy, level, coefficient):
    """Decision 1 document, section 8 step 1: anchor and σ in the same units. The file's σ
    is sd(TWT+Sospeso) in credits (9.899547 overall; 5.7 to 16.9 per level) and the anchor
    is a mix of z-scores, so the draw's σ is the file's value × coefficient ÷ sd(TWT+Sospeso):
    1.00 at coefficient 1 overall, 1.70 for level 5. (Before 2026-10-24 the credits value
    was used as it stood: twelve times the anchor's own spread at coefficient 1.)"""
    params = copy.deepcopy(di_params)
    params["income_mode"] = "categorical"
    stochastic = params["stochastic"]
    stochastic["sigma_value"] = 9.899547           # the Research tick box: on
    stochastic["sigma_strategy"] = strategy
    stochastic["scale_factor"] = coefficient
    stochastic["quintile_scale_factors"] = {k: coefficient for k in "12345"}
    agent = merged[merged["Assigned Allowance Level"].astype(int) == level].iloc[0].to_dict()

    rng = _SigmaRecorder()
    out = disclose_income_stochastic(agent, params, rng, {}, pop_context="documentation")

    twt_sd = params["z_scoring"]["TWT_Sospeso"]["sd"]
    base = stochastic["sigma_overall"] if strategy == "overall" else stochastic["sigma_quintile"][str(level)]
    assert rng.sigmas == [pytest.approx(base * coefficient / twt_sd)]
    assert twt_sd == 9.899547
    # the recorder returns the mean, so the draw equals the deterministic anchor
    assert out["disclose_income_anchored_pb"] == pytest.approx(out["disclose_income_anchored_pb_deterministic"])


def test_frozen_csv_equals_professor_dta():
    require_reference_file(PROF_DTA, "PROF_DTA in tests/test_disclose_income_stata.py")
    require_reference_file(FROZEN, "FROZEN in tests/test_disclose_income_stata.py")
    frozen = pd.read_csv(FROZEN, float_precision="round_trip")
    dta = pd.read_stata(PROF_DTA)[FROZEN_COLS].reset_index(drop=True)
    assert list(frozen.columns) == FROZEN_COLS
    for c in FROZEN_COLS:
        np.testing.assert_array_equal(frozen[c].values, dta[c].astype(frozen[c].dtype).values, err_msg=c)
