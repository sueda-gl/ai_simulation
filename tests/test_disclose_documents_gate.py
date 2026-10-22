"""
Decision 2's platform gate and default short-circuit, run on the shipped configuration.

The first two tests lived in tests/test_disclose_documents.py with ``params = {"income_mode":
"Categorical only"}``: they ran only because the module silently fell back to its own
copies of every constant. config/decisions.yaml is now the only source (a missing key
raises - tests/test_config_is_single_source.py), so the same assertions run on the
decision's block of the file.

The gate without Decision 1's output follows the platform documentation (COOPECON AI Agent
Documentation 2026-10-07, Decision 2, Eligibility Gate): "If Decision 1 is not run prior to
Decision 2, the eligibility gets determined randomly on a 50% basis". The draw is the first
draw of the agent's Decision 2 rng (``is_eligible``); until 2026-10-22 the model ran ungated
for everyone in that case.
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

    # with Decision 1's output the gate draws nothing: the sigma draw is the stream's first value
    params["stochastic"]["sigma_value"] = 1.0
    sigma = params["stochastic"]["sigma_overall"] * params["stochastic"]["scale_factor"]
    out = disclose_documents_stochastic(
        {"disclose_income": "Y", "income": 5000, "Assigned Allowance Level": 1, **TRAITS}, params,
        np.random.default_rng(7), sim, pop_context="documentation")
    assert out["disclose_documents_raw"] == np.random.default_rng(7).normal(out["disclose_documents_score"], sigma)


def test_without_decision_1_eligibility_is_a_bernoulli_draw_drawn_first():
    """No 'disclose_income' in the agent state (Decision 1 has not run): the disclosure
    condition is a Bernoulli(gate.probability_without_decision_1 = 0.5) draw, the FIRST draw
    from the agent's Decision 2 rng, combined with the same income threshold; an eligible
    agent's sigma draw is the stream's next value; an ineligible agent is never-eligible ('NA')."""
    params = _params()
    assert params["gate"]["probability_without_decision_1"] == 0.5
    params["stochastic"]["sigma_value"] = 1.0      # research draws on: the sigma draw follows the gate draw
    sigma = params["stochastic"]["sigma_overall"] * params["stochastic"]["scale_factor"]
    sim = {"discount_income_threshold": 12500.0}
    below = {"income": 5000, "Assigned Allowance Level": 1, **TRAITS}

    outcomes = []
    for seed in range(200):
        out = disclose_documents_stochastic(dict(below), params, np.random.default_rng(seed), sim,
                                            pop_context="documentation")
        replay = np.random.default_rng(seed)
        if replay.random() < 0.5:                  # the gate's draw, first
            expected = replay.normal(out["disclose_documents_score"], sigma)   # the sigma draw, next
            assert out["disclose_documents_raw"] == expected
            assert out["disclose_documents"] == ("Y" if expected > 0 else "N")
        else:
            assert out["disclose_documents"] == "NA"
            assert out["disclose_documents_sigma_used"] == 0.0
        outcomes.append(out["disclose_documents"])
    assert 75 <= outcomes.count("NA") <= 125       # about half (binomial(200, 0.5))

    # income at or above the threshold: never eligible, whatever the draw
    above = {"income": 50000, "Assigned Allowance Level": 5, **TRAITS}
    assert {disclose_documents_stochastic(dict(above), params, np.random.default_rng(s), sim,
                                          pop_context="documentation")["disclose_documents"]
            for s in range(20)} == {"NA"}

    # the probability is the file's value, not a module constant: 1.0 -> everyone below the
    # threshold is eligible, 0.0 -> nobody
    for probability, allowed in ((1.0, {"Y", "N"}), (0.0, {"NA"})):
        params["gate"]["probability_without_decision_1"] = probability
        got = {disclose_documents_stochastic(dict(below), params, np.random.default_rng(s), sim,
                                             pop_context="documentation")["disclose_documents"]
               for s in range(20)}
        assert got and got <= allowed, (probability, got)


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
