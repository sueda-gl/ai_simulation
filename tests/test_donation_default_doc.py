"""
Decision 3 (Default Donation Rate) against the methodology document (ruling R-D3).

Reference: "Donation_Rate_Decision_Methodology-2-1 250925.docx". The document is the ground
truth; it has no .dta of its own, so its Stata commands (sections 3 and 5) and its section 6
step 4 formula are re-implemented on the professor's 280-participant raw data
(Stata_File_Decision 1_Updated.dta) by experiments/d3_doc_reference.py and frozen in
data/d3_doc_reference.csv (doc_final = score_k / 100 at sigma = 0).

Owner ruling: the app keeps its defaults (adjustment shift -4.0, sigma coefficient 1.0), but
with the document's parameters (shift 0, sigma 0 for the deterministic run) the engine must
reproduce the document participant by participant, in both income modes.
"""
import contextlib
import copy
import io
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import participants as P
from src.decisions.donation_default import (
    DOC_COEFFICIENTS, DOC_PREDICTED_RANGE, PROGRAM_TO_CATEGORY, STUDY_CATEGORY_KEYS,
    donation_default, study_category_key,
)
from src.engine.core import Engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(REPO, "data", "d3_doc_reference.csv")
DECISIONS = os.path.join(REPO, "config", "decisions.yaml")
PROF_DTA = "/Users/suedagul/Downloads/Stata_File_Decision 1_Updated.dta"
TOL = 1e-6
MODES = ("categorical", "continuous")

# The document's final regression tables (section 3), as printed by Stata.
DOC_REPORTED = {
    "categorical": {  # regress ... i.groupcat i.totalallowance i.studyprogramcategorycat honesty_humility
        "intercept": 1.519818,
        "beta_group": {"MidSub": 0.8773149, "NoSub": -0.913715, "FullSub": 0.0},
        "beta_income_q": {"Q1": 0.0, "Q2": -0.4214298, "Q3": -0.7364032, "Q4": 3.539434, "Q5": 3.784071},
        "beta_study": {"Incoming": -6.882558, "Law5yr": -2.003814, "UG3yr": -2.11522, "Grad2yr": 0.0},
        "beta_hh": 0.6042141,
    },
    "continuous": {  # regress ... i.groupcat totalallowance i.studyprogramcategorycat honesty_humility
        "intercept": -0.139596,
        "beta_group": {"MidSub": 0.7952602, "NoSub": -0.8990889, "FullSub": 0.0},
        "beta_income_linear": 0.0255512,
        "beta_study": {"Incoming": -7.305585, "Law5yr": -2.048199, "UG3yr": -2.016326, "Grad2yr": 0.0},
        "beta_hh": 0.7840063,
    },
}


@pytest.fixture(scope="module")
def reference():
    ref = pd.read_csv(FROZEN, float_precision="round_trip")
    assert len(ref) == 560 and set(ref.income_mode) == set(MODES)
    return ref


@pytest.fixture(scope="module")
def config():
    with open(DECISIONS) as f:
        return yaml.safe_load(f)["donation_default"]


@pytest.fixture(scope="module")
def participants():
    """The 280 participants in engine order, with their Participant IDs."""
    agents = P.original_data()
    pids = P.merged().loc[agents.index, "Participant ID"].astype(int).values
    return agents, pids


def run_engine(profile, agents, mode, shift, sigma_value=0.0, in_copula=False, seed=42):
    engine = Engine(profile)
    cfg = engine.config["donation_default"]
    cfg["regression_coefficients"]["income_mode"] = mode
    cfg["adjustment"]["shift_value"] = shift
    cfg["stochastic"]["sigma_value"] = sigma_value
    cfg["stochastic"]["in_copula"] = in_copula
    with contextlib.redirect_stdout(io.StringIO()):
        out = engine.run_simulation(len(agents), seed, single_decision="donation_default",
                                    agents_df=agents.copy())
    return out["donation_default"].to_numpy(dtype=float), engine


def doc_final(reference, mode, pids):
    return reference[reference.income_mode == mode].set_index("participantid").loc[pids, "doc_final"].to_numpy()


# --------------------------------------------------------------------- reference integrity

def test_frozen_reference_matches_the_doc_reimplementation():
    """If the professor's .dta is present, the frozen CSV is exactly what the script derives."""
    if not os.path.exists(PROF_DTA):
        pytest.skip("professor's Stata_File_Decision 1_Updated.dta not present")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "d3_doc_reference", os.path.join(REPO, "experiments", "d3_doc_reference.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh, coefs = mod.build(PROF_DTA)
    frozen = pd.read_csv(FROZEN, float_precision="round_trip")
    assert (fresh.participantid.values == frozen.participantid.values).all()
    for col in ("doc_pred", "doc_anchor", "doc_final"):
        assert np.abs(fresh[col].values - frozen[col].values).max() < 1e-12
    # the re-implemented regressions reproduce the document's printed coefficients
    cat, cont = coefs["categorical"], coefs["continuous"]
    printed = {
        (cat["_cons"], 1.519818), (cat["honesty_humility"], 0.6042141),
        (cat["totalallowance=200"], 3.784071), (cat["studyprogramcategorycat=UG 3-year Program"], -2.11522),
        (cont["_cons"], -0.1395969), (cont["totalallowance"], 0.0255512), (cont["honesty_humility"], 0.7840063),
    }
    for got, doc in printed:
        assert got == pytest.approx(doc, abs=1e-6)


# ------------------------------------------------------------------------ configuration

@pytest.mark.parametrize("mode", MODES)
def test_config_coefficients_equal_the_doc(config, mode):
    assert config["regression_coefficients"][mode] == DOC_REPORTED[mode]
    for key, value in DOC_COEFFICIENTS[mode].items():  # module fallbacks mirror the doc too
        assert value == DOC_REPORTED[mode][key]
    if mode == "categorical":  # the legacy flat block carries the categorical set
        legacy = config["regression"]
        for key in ("intercept", "beta_group", "beta_study", "beta_hh"):
            assert legacy[key] == DOC_REPORTED[mode][key]


@pytest.mark.parametrize("mode", MODES)
def test_predicted_range_is_the_280_sample_min_max(config, reference, mode):
    pred = reference.loc[reference.income_mode == mode, "doc_pred"]
    lo, hi = config["scaling"]["predicted_range"][mode]
    assert (lo, hi) == DOC_PREDICTED_RANGE[mode]
    assert lo == pytest.approx(pred.min(), abs=1e-9) and hi == pytest.approx(pred.max(), abs=1e-9)
    assert config["scaling"]["observed_range"] == [0.0, 112.0]


def test_shipped_defaults_unchanged(config):
    """Owner ruling R-D3: the app defaults stay; only bugs were fixed."""
    from app.seam import build_plan as bp
    from app.seam.config_repo import get_config_repo
    assert config["adjustment"]["shift_value"] == -4.0
    assert config["stochastic"]["sigma_overall"] == 9.899547
    repo = get_config_repo()
    assert repo.donation_adjustment_shift() == -4.0
    # an untouched session: sigma coefficient 1.0 -> sigma_value = sigma_overall x 1.0
    patch = bp.build_donation_patch({}, repo, "documentation", "categorical")
    assert patch["stochastic"]["sigma_value"] == pytest.approx(9.899547 * 1.0)
    assert "adjustment" not in patch  # the file's -4.0 applies


# ------------------------------------------------------------------ study programme mapping

def test_program_to_category_is_the_data_mapping(config):
    m = P.merged()
    pairs = m.groupby("Study Program")["Study Program Category"].agg(lambda s: set(s))
    assert all(len(v) == 1 for v in pairs)
    data_map = {k: next(iter(v)) for k, v in pairs.items()}
    assert data_map == PROGRAM_TO_CATEGORY == config["study_program"]["program_to_category"]
    assert STUDY_CATEGORY_KEYS == config["study_program"]["category_keys"]


def test_study_category_key_research_and_copula_agents(config):
    # the old substring rule put CLEACC / CLEAM / CLEF (UG 3-year) and RI (Incoming) in Grad2yr
    for program, key in [("CLEACC", "UG3yr"), ("CLEAM", "UG3yr"), ("CLEF", "UG3yr"), ("RI", "Incoming"),
                         ("CLMG", "Law5yr"), ("CLELI", "Grad2yr"), ("BIEM", "UG3yr")]:
        assert study_category_key({"Study Program": program}, config) == key          # copula agent
        cat = PROGRAM_TO_CATEGORY[program]
        assert study_category_key({"Study Program": program, "Study Program Category": cat}, config) == key
    with pytest.raises(ValueError):
        study_category_key({"Study Program": "NOPE"}, config)


def test_copula_model_carries_study_program_only():
    """The fitted copula has `Study Program` but not its category; every decoded value maps."""
    from src.trait_engine import TraitEngine
    with contextlib.redirect_stdout(io.StringIO()):
        te = TraitEngine()
    assert "Study Program" in te.traits and "Study Program Category" not in te.traits
    with contextlib.redirect_stdout(io.StringIO()):
        programs = set(te.sample(2000, 7)["Study Program"])
    assert programs <= set(PROGRAM_TO_CATEGORY)


# ------------------------------------------------------------- code == doc, participant-wise

@pytest.mark.parametrize("profile", ["baseline", "documentation", "copula"])
@pytest.mark.parametrize("mode", MODES)
def test_engine_reproduces_doc_at_shift0_sigma0(reference, participants, profile, mode):
    agents, pids = participants
    got, engine = run_engine(profile, agents, mode, shift=0.0)
    want = doc_final(reference, mode, pids)
    diff = np.abs(got - want)
    assert int((diff < TOL).sum()) == 280, f"max |code - doc| = {diff.max():.3e}"
    assert got.max() == 1.0


@pytest.mark.parametrize("mode", MODES)
def test_copula_agents_without_category_column_match(reference, participants, mode):
    """Copula agents carry only Study Program: the config mapping gives the same numbers."""
    agents, pids = participants
    got, _ = run_engine("copula", agents.drop(columns=["Study Program Category"]), mode, shift=0.0)
    assert np.abs(got - doc_final(reference, mode, pids)).max() < TOL


@pytest.mark.parametrize("mode", MODES)
def test_direct_call_anchor_matches_doc(reference, participants, config, mode):
    """Without a population (direct call) the rate is the anchor / 100 - the doc's s100donationanchor."""
    agents, pids = participants
    params = copy.deepcopy(config)
    params["regression_coefficients"]["income_mode"] = mode
    params["adjustment"]["shift_value"] = 0.0
    params["stochastic"]["sigma_value"] = 0.0
    got = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _, row in agents.iterrows():
            state = row.to_dict()
            state["actual_allowance"] = float({1: 12, 2: 32, 3: 72, 4: 128, 5: 200}[int(state["Assigned Allowance Level"])])
            got.append(donation_default(state, params, np.random.default_rng(0), {},
                                        pop_context="documentation")["donation_default"])
    want = reference[reference.income_mode == mode].set_index("participantid").loc[pids, "doc_anchor"].to_numpy()
    assert np.abs(np.array(got) - want).max() < TOL


# ------------------------------------------------------------------------------ noise path

def test_noise_gate_and_population_rescale(participants):
    agents, _ = participants
    det, _ = run_engine("documentation", agents, "categorical", shift=0.0)
    sigma = 9.899547 * 0.1  # the doc's beta = 0.1

    # Research Specification, tick on (sigma_value > 0): noisy, rescaled by the max of the
    # floored draws -> the Pass-1 replay equals the Pass-2 draws exactly (max is exactly 1).
    noisy, engine = run_engine("documentation", agents, "categorical", shift=0.0, sigma_value=sigma)
    assert noisy.max() == 1.0 and noisy.min() >= 0.0
    assert np.abs(noisy - det).max() > 1e-3
    assert engine.simulation_config["donation_population_max"] > 0

    # Research Baseline never adds noise (R5/R6), whatever sigma_value says
    base, _ = run_engine("baseline", agents, "categorical", shift=0.0, sigma_value=sigma)
    assert np.abs(base - det).max() < 1e-12

    # Copula: noise iff the copula tick is on
    off, _ = run_engine("copula", agents, "categorical", shift=0.0, sigma_value=sigma, in_copula=False)
    assert np.abs(off - det).max() < 1e-12
    on, _ = run_engine("copula", agents, "categorical", shift=0.0, sigma_value=sigma, in_copula=True)
    assert on.max() == 1.0 and np.abs(on - det).max() > 1e-3


def test_app_default_shift_floors_then_rescales(participants):
    """Default shift -4.0: anchors below 4 floor at 0, the rest divide by the population max."""
    agents, _ = participants
    got, engine = run_engine("baseline", agents, "categorical", shift=-4.0)
    assert got.max() == 1.0 and got.min() == 0.0
    assert 0.0 <= got.min() and np.all(got <= 1.0)
