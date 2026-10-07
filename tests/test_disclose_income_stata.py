"""
Validation tests for Decision 1: Disclose Income against the professor's CORRECTED Stata run.

Reference data: "Decision 1 - Disclosure of Income 110226_Final.docx" and its data file
"Stata_File_Decision 1_Updated.dta" (280 participants). The corrected columns are
disclose_categorical (167 Y), disclose_cont (169 Y) and fs_deterministic_*; the old error
model kept in the *_err columns is NOT the reference.

Owner ruling (R-D1, superseded 2026-10-07): the model uses exactly the owner's September values,
i.e. Andrei's constants from "Decision 1 - Disclosure of Income 110226_Final_Andrei_Fix_Issue.docx"
(Extraversion 0.00680238, level intercepts 0.0089007 ... -0.0145324, composite SDs 0.025040462 /
0.7984211971). Andrei applied the Extraversion weight typo fix 641.15 -> 645.15; the professor's
.dta does not, so the two runs differ slightly. These tests DOCUMENT that known, accepted
difference: with beta0 = 0.1, continuous matches 280/280, categorical matches 279/280 with the
single mismatch participant 848 (a near-threshold case), and the fs scores agree within
FS_TOL_ANDREI. The app default beta0 stays 0.75.

The corrected Stata columns are frozen in data/stata_d1_verification.csv so the test runs
without the professor's file; if the .dta is present locally we also check the frozen CSV
still equals it.
"""
import os
import copy
import numpy as np
import pandas as pd
import yaml
import pytest

from src.decisions.disclose_income_stochastic import (
    disclose_income_stochastic, compute_continuous_de_stats,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURVEY = os.path.join(REPO, "Student Survey Results - Period 1.xlsx")
EXPERIMENT = os.path.join(REPO, "Student Experiment Results - Period 1-2.xlsx")
FROZEN = os.path.join(REPO, "data", "stata_d1_verification.csv")
PROF_DTA = "/Users/suedagul/Downloads/Stata_File_Decision 1_Updated.dta"

STATA_BETA0 = 0.1        # `gen beta0 = 0.1` in the professor's corrected Stata code
APP_DEFAULT_BETA0 = 0.75  # the app's research default (unchanged by ruling R-D1)
# Measured max |fs_model - fs_stata| with the September/Andrei constants: 0.003414 (categorical)
# and 0.003373 (continuous), both at participant 838. The bound sits just above that.
FS_TOL_ANDREI = 3.5e-3
KNOWN_CATEGORICAL_MISMATCHES = [848]  # model Y (fs +0.00046) vs Stata N (fs -0.00022)
SEPTEMBER_SD_CATEGORICAL = 0.025040462
SEPTEMBER_SD_ANCHORED_PB = 0.7984211971

FROZEN_COLS = [
    "participantid", "assignedallowancelevel", "income", "income_high", "income_high_cat",
    "weighted_disclosure_categorical", "weighted_disclosure_cont", "anchored_prosocial_behavior",
    "fs_deterministic_categorical", "fs_deterministic_cont",
    "disclose_categorical", "disclose_cont",
]


def _load_stata():
    """Prefer the frozen CSV extract; fall back to the professor's .dta."""
    if os.path.exists(FROZEN):
        return pd.read_csv(FROZEN, float_precision="round_trip")
    if os.path.exists(PROF_DTA):
        return pd.read_stata(PROF_DTA)[FROZEN_COLS]
    pytest.skip("neither data/stata_d1_verification.csv nor the professor's Decision 1 .dta is present")


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
    with open(os.path.join(REPO, "config", "decisions.yaml")) as f:
        return yaml.safe_load(f)["disclose_income"]


def _run(merged, di_params, income_mode):
    """Run Decision 1 deterministically (no noise) with beta0 = 0.1 on the 280 participants."""
    params = copy.deepcopy(di_params)
    params["intercept"] = STATA_BETA0
    params["income_mode"] = income_mode
    params.setdefault("stochastic", {})["sigma_value"] = 0.0

    # the professor's own income realization; SD with ddof=1 like Stata `egen std()`
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
    """Known, accepted difference (R-D1 superseded): 279/280, the one mismatch is participant 848."""
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


def test_composite_sds_are_september_values(stata, di_params):
    """The fixed composite SDs are the owner's September (Andrei) values, not Stata's `egen std()`
    of the professor's .dta (0.0250386349 / 0.7971466830) -- the documented R-D1 difference."""
    cz = di_params["composite_z_scoring"]
    assert cz["weighted_disclosure_categorical"]["sd"] == SEPTEMBER_SD_CATEGORICAL
    assert cz["anchored_pb"]["sd"] == SEPTEMBER_SD_ANCHORED_PB
    # the difference from the .dta is small (< 0.2%) and is the accepted one
    assert cz["weighted_disclosure_categorical"]["sd"] == pytest.approx(
        stata["weighted_disclosure_categorical"].astype(float).std(ddof=1), rel=2e-3)
    assert cz["anchored_pb"]["sd"] == pytest.approx(
        stata["anchored_prosocial_behavior"].astype(float).std(ddof=1), rel=2e-3)


def test_september_constants(di_params):
    """Owner ruling: Decision 1 uses exactly the September (Andrei) constants."""
    assert di_params["equation2_coefficients"]["extraversion"] == 0.00680238
    assert [di_params["categorical_intercepts"][f"level_{k}"] for k in range(1, 6)] == [
        0.0089007, 0.0055352, 0.0023109, -0.0032216, -0.0145324]


def test_shipped_default_intercept_unchanged(di_params):
    """Ruling R-D1: matching Stata is done by setting beta0 = 0.1; the app default stays 0.75."""
    assert di_params["intercept"] == APP_DEFAULT_BETA0


@pytest.mark.skipif(not (os.path.exists(PROF_DTA) and os.path.exists(FROZEN)),
                    reason="needs both the professor's Decision 1 .dta and the frozen CSV")
def test_frozen_csv_equals_professor_dta():
    frozen = pd.read_csv(FROZEN, float_precision="round_trip")
    dta = pd.read_stata(PROF_DTA)[FROZEN_COLS].reset_index(drop=True)
    assert list(frozen.columns) == FROZEN_COLS
    for c in FROZEN_COLS:
        np.testing.assert_array_equal(frozen[c].values, dta[c].astype(frozen[c].dtype).values, err_msg=c)
