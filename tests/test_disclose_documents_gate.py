"""
Decision 2's platform gate and default short-circuit, run on the shipped configuration.

These two tests lived in tests/test_disclose_documents.py with ``params = {"income_mode":
"Categorical only"}``: they ran only because the module silently fell back to its own
copies of every constant. config/decisions.yaml is now the only source (a missing key
raises - tests/test_config_is_single_source.py), so the same assertions run on the
decision's block of the file.
"""
import copy

import numpy as np

from src.decisions.disclose_documents_stochastic import disclose_documents_stochastic

from conftest import load_decisions_yaml


def _params():
    p = copy.deepcopy(load_decisions_yaml()["disclose_documents"])
    p["income_mode"] = "Categorical only"
    return p


TRAITS = {"ExtraversionBig5": 3.5, "NeuroticismBig5": 2.7, "Agreeable": 3.5}


def test_eligibility_gate():
    """The simulation entry point returns NA unless disclose_income==Y AND income<threshold."""
    params = _params()
    sim = {"discount_income_threshold": 12500.0}
    rng = np.random.default_rng(0)

    # did not disclose income -> NA
    out = disclose_documents_stochastic(
        {"disclose_income": "N", "income": 5000, "Assigned Allowance Level": 1, **TRAITS}, params, rng, sim)
    assert out["disclose_documents"] == "NA"

    # disclosed income but income >= threshold -> NA
    out = disclose_documents_stochastic(
        {"disclose_income": "Y", "income": 50000, "Assigned Allowance Level": 5, **TRAITS}, params, rng, sim)
    assert out["disclose_documents"] == "NA"

    # disclosed income AND below threshold -> a real Y/N decision
    out = disclose_documents_stochastic(
        {"disclose_income": "Y", "income": 5000, "Assigned Allowance Level": 1, **TRAITS}, params, rng, sim)
    assert out["disclose_documents"] in ("Y", "N")


def test_default_short_circuit_when_unconfigured():
    """If disclose_documents is unconfigured (in default_decisions_list), eligible agents
    get the simple random default rather than the full model."""
    params = _params()
    sim = {
        "discount_income_threshold": 12500.0,
        "default_decisions_list": ["disclose_documents"],
        "default_decisions": {"disclose_documents": {"type": "random_probability", "probability_y": 1.0,
                                                     "options": ["Y", "N"]}},
    }
    out = disclose_documents_stochastic(
        {"disclose_income": "Y", "income": 5000, "Assigned Allowance Level": 1, **TRAITS}, params,
        np.random.default_rng(0), sim)
    assert out["disclose_documents"] == "Y"  # probability_y=1.0 forces Y
