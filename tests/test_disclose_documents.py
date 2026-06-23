"""
Validation tests for Decision 2: Disclose Documents.

The deterministic privacy-calculus model is validated against the professor's Stata
Decision 2 results:
- CATEGORICAL: 110/280 = 39.29% disclosure (every per-allowance-level cell).
- CONTINUOUS:  102/280 = 36.43% disclosure, using the FROZEN income realization
  (data/stata_incomes.csv == the professor's `income` column). Continuous income is a
  fresh random draw at simulation time, so it is only reproducible bit-for-bit against
  the frozen income (see notes.md).

If the professor's `.dta` is present locally, we also assert the model matches his
`disclosedoc_cont` / `disclosedoc_categorical` columns participant-by-participant.
"""
import os
import copy
import numpy as np
import pandas as pd
import yaml
import pytest

from src.decisions.disclose_documents_stochastic import (
    compute_dd_score, compute_continuous_dd_stats, disclose_documents_stochastic,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURVEY = os.path.join(REPO, "Student Survey Results - Period 1.xlsx")
EXPERIMENT = os.path.join(REPO, "Student Experiment Results - Period 1-2.xlsx")
FROZEN_INCOME = os.path.join(REPO, "data", "stata_incomes.csv")
PROF_DTA = "/Users/suedagul/Downloads/Stata_File_Decision 2_260326 - Final.dta"

# Document / professor gold targets
DOC_CATEGORICAL = {12: 96.49, 32: 78.69, 72: 13.46, 128: 0.0, 200: 0.0}
DOC_CONTINUOUS = {12: 94.74, 32: 68.85, 72: 11.54, 128: 0.0, 200: 0.0}


@pytest.fixture(scope="module")
def merged():
    survey = pd.read_excel(SURVEY)
    exp = pd.read_excel(EXPERIMENT)
    m = survey.merge(exp, on="Participant ID", how="inner")
    m = m[m["Participant ID"].notna()].reset_index(drop=True)
    assert len(m) == 280, f"expected 280 merged participants, got {len(m)}"
    return m


@pytest.fixture(scope="module")
def dd_params():
    with open(os.path.join(REPO, "config", "decisions.yaml")) as f:
        return yaml.safe_load(f)["disclose_documents"]


def _rates_by_allowance(disclose, total_allowance):
    rates = {}
    for ta in [12, 32, 72, 128, 200]:
        mask = total_allowance == ta
        rates[ta] = round(100 * disclose[mask].mean(), 2)
    return rates


def test_categorical_reproduces_professor_table(merged, dd_params):
    """Categorical model (no income) reproduces 110/280 = 39.29%, every cell."""
    params = copy.deepcopy(dd_params)
    params["income_mode"] = "Categorical only"
    ta = merged["Total Allowance"].astype(float).values

    disclose = np.array([
        1 if compute_dd_score(row.to_dict(), params, {})["dd_deterministic"] > 0 else 0
        for _, row in merged.iterrows()
    ])

    assert disclose.sum() == 110, f"categorical total {disclose.sum()} != 110"
    assert round(100 * disclose.mean(), 2) == 39.29
    rates = _rates_by_allowance(disclose, ta)
    for level, expected in DOC_CATEGORICAL.items():
        assert rates[level] == expected, f"TA{level}: {rates[level]}% != {expected}%"


def test_continuous_reproduces_professor_table_with_frozen_income(merged, dd_params):
    """Continuous model with the FROZEN income realization reproduces 102/280 = 36.43%."""
    assert os.path.exists(FROZEN_INCOME), "data/stata_incomes.csv (frozen income) is required"
    frozen = pd.read_csv(FROZEN_INCOME).sort_values("original_index").reset_index(drop=True)
    incomes = frozen["income"].values
    assert len(incomes) == len(merged)

    params = copy.deepcopy(dd_params)
    params["income_mode"] = "continuous"

    # income_stats / composite stats computed with ddof=1 to match Stata's egen std()
    sim_config = {"income_stats": {"mean": float(np.mean(incomes)), "sd": float(np.std(incomes, ddof=1))}}
    sim_config["dd_cont_stats"] = compute_continuous_dd_stats(merged, list(incomes), params, sim_config)

    ta = merged["Total Allowance"].astype(float).values
    disclose = np.empty(len(merged), dtype=int)
    for i, (_, row) in enumerate(merged.iterrows()):
        agent = row.to_dict()
        agent["income"] = float(incomes[i])  # inject frozen income (cached)
        disclose[i] = 1 if compute_dd_score(agent, params, sim_config)["dd_deterministic"] > 0 else 0

    assert disclose.sum() == 102, f"continuous total {disclose.sum()} != 102"
    assert round(100 * disclose.mean(), 2) == 36.43
    rates = _rates_by_allowance(disclose, ta)
    for level, expected in DOC_CONTINUOUS.items():
        assert rates[level] == expected, f"TA{level}: {rates[level]}% != {expected}%"


@pytest.mark.skipif(not os.path.exists(PROF_DTA), reason="professor's Decision 2 .dta not present")
def test_matches_professor_dta_columns(merged, dd_params):
    """Model matches the professor's disclosedoc_cont / disclosedoc_categorical 280/280."""
    p = pd.read_stata(PROF_DTA)

    # categorical
    params_cat = copy.deepcopy(dd_params); params_cat["income_mode"] = "Categorical only"
    cat = np.array([
        1 if compute_dd_score(row.to_dict(), params_cat, {})["dd_deterministic"] > 0 else 0
        for _, row in merged.iterrows()
    ])
    assert int((cat == p["disclosedoc_categorical"].astype(int).values).sum()) == 280

    # continuous (professor's own income column)
    params_cont = copy.deepcopy(dd_params); params_cont["income_mode"] = "continuous"
    inc = p["income"].astype(float).values
    sim_config = {"income_stats": {"mean": float(np.mean(inc)), "sd": float(np.std(inc, ddof=1))}}
    sim_config["dd_cont_stats"] = compute_continuous_dd_stats(merged, list(inc), params_cont, sim_config)
    cont = np.empty(len(merged), dtype=int)
    for i, (_, row) in enumerate(merged.iterrows()):
        agent = row.to_dict(); agent["income"] = float(inc[i])
        cont[i] = 1 if compute_dd_score(agent, params_cont, sim_config)["dd_deterministic"] > 0 else 0
    assert int((cont == p["disclosedoc_cont"].astype(int).values).sum()) == 280


def test_eligibility_gate():
    """The simulation entry point returns NA unless disclose_income==Y AND income<threshold."""
    params = {"income_mode": "Categorical only"}
    sim = {"discount_income_threshold": 12500.0}
    rng = np.random.default_rng(0)

    # did not disclose income -> NA
    out = disclose_documents_stochastic(
        {"disclose_income": "N", "income": 5000, "Assigned Allowance Level": 1}, params, rng, sim)
    assert out["disclose_documents"] == "NA"

    # disclosed income but income >= threshold -> NA
    out = disclose_documents_stochastic(
        {"disclose_income": "Y", "income": 50000, "Assigned Allowance Level": 5}, params, rng, sim)
    assert out["disclose_documents"] == "NA"

    # disclosed income AND below threshold -> a real Y/N decision
    out = disclose_documents_stochastic(
        {"disclose_income": "Y", "income": 5000, "Assigned Allowance Level": 1,
         "ExtraversionBig5": 3.5, "NeuroticismBig5": 2.7, "Agreeable": 3.5}, params, rng, sim)
    assert out["disclose_documents"] in ("Y", "N")


def test_default_short_circuit_when_unconfigured():
    """If disclose_documents is unconfigured (in default_decisions_list), eligible agents
    get the simple random default rather than the full model."""
    params = {"income_mode": "Categorical only"}
    sim = {
        "discount_income_threshold": 12500.0,
        "default_decisions_list": ["disclose_documents"],
        "default_decisions": {"disclose_documents": {"type": "random_probability", "probability_y": 1.0, "options": ["Y", "N"]}},
    }
    out = disclose_documents_stochastic(
        {"disclose_income": "Y", "income": 5000, "Assigned Allowance Level": 1}, params,
        np.random.default_rng(0), sim)
    assert out["disclose_documents"] == "Y"  # probability_y=1.0 forces Y
