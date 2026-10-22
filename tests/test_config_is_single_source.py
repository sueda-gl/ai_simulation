"""
config/decisions.yaml is the ONLY source of every constant of the four modelled decisions.

(a) No module under src/decisions carries a literal copy of a value that is in its
    decision's YAML block under the same key. The scan is deliberately narrow, so that
    an unrelated number can never trip it:

      1. ``<expr>.get('<key>', <number>)`` - a fallback for a YAML key, flagged when
         <key> is a leaf key of the decision's block and <number> equals the file's value;
      2. a dict literal ``{'<key>': <number>, ...}`` anywhere in the module (also
         ``{<int>: <number>}``, matched as ``level_<int>`` / ``'<int>'``) - the former
         CATEGORICAL_INTERCEPTS / *_COEFFS / DEFAULT_Z_SCORING shape - flagged on the
         same key-and-value match;
      3. a module-level ``NAME = <number>`` (the former B_EXTRA / BETA0 shape), flagged
         when the number equals ANY numeric leaf of the block.

    The neutral values 0, 1 and -1 are never flagged (they are identity defaults, not
    constants), so ``income_stats.get('mean', 0)`` on an engine dict passes; a number
    only used in arithmetic (``100.0 / 112.0``) is not scanned.

(b) Deleting a required key from a copy of the decision's block and running the
    decision on one agent raises a KeyError naming the decision and the dotted key.
    The same run on the full block succeeds, so it is the deletion that raises.

(c) The two file readers in src/config.py the tabs use (``donation_adjustment_shift``,
    ``rtd_sigma_coefficient_defaults``) raise, naming the key, when the file lacks it.

The only keys not read strictly are two Decision-4 blocks whose absent keys existing
tests pin as meaningful: ``intercepts`` (absent = 0,
tests/test_rejected_transaction_defaults.py) and ``aggregation`` (a partial block is
merged over the file's, tests/test_rtd_rank_aggregation.py) - (b) does not delete them.
"""
import ast
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.decisions._shared import MissingDecisionKey
from src.decisions.disclose_income_stochastic import disclose_income_stochastic
from src.decisions.disclose_documents_stochastic import disclose_documents_stochastic
from src.decisions.donation_default import donation_default
from src.decisions.rejected_transaction_defaults import (
    MECHANISMS, compute_rtd_population_stats, rejected_transaction_defaults)

from conftest import ROOT, load_decisions_yaml

DECISIONS_DIR = ROOT / "src" / "decisions"
MODULES = {
    "disclose_income": "disclose_income_stochastic.py",
    "disclose_documents": "disclose_documents_stochastic.py",
    "donation_default": "donation_default.py",
    "rejected_transaction_defaults": "rejected_transaction_defaults.py",
}
NEUTRAL = (0.0, 1.0, -1.0)


# ----------------------------------------------------------------------------- (a)

def _numeric_leaves(node, path=()):
    """(dotted path, leaf key, float value) for every int/float leaf (bools excluded)."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += _numeric_leaves(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += _numeric_leaves(v, path + (str(i),))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.append((".".join(path), path[-1], float(node)))
    return out


def _number(node):
    """The float of a numeric literal (``-0.75`` included), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _number(node.operand)
        return None if inner is None else -inner
    return None


def _key_names(node):
    """The YAML key names a dict-literal / .get key may stand for."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return (node.value,)
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return (f"level_{node.value}", str(node.value))
    return ()


def _is_neutral(value):
    return any(math.isclose(value, n, abs_tol=0.0) for n in NEUTRAL)


def _same(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)


def literal_copies(source: str, block: dict):
    """Every (line, description) where ``source`` carries a literal copy of ``block``."""
    leaves = _numeric_leaves(block)
    by_key = {}
    for _, key, value in leaves:
        by_key.setdefault(key, []).append(value)
    all_values = [v for _, _, v in leaves]
    tree = ast.parse(source)
    hits = []

    def check_pair(key_node, value_node, where):
        value = _number(value_node)
        if value is None or _is_neutral(value):
            return
        for key in _key_names(key_node):
            if any(_same(value, v) for v in by_key.get(key, ())):
                hits.append((value_node.lineno, f"{where}: '{key}' = {value!r} is the file's value"))
                return

    for node in ast.walk(tree):
        # 1. <expr>.get('<key>', <number>)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get" and len(node.args) == 2:
            check_pair(node.args[0], node.args[1], ".get() fallback")
        # 2. {'<key>': <number>, ...} anywhere
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is not None:
                    check_pair(k, v, "dict literal")
    # 3. module-level NAME = <number>
    for node in tree.body:
        if isinstance(node, ast.Assign):
            value = _number(node.value)
            if value is not None and not _is_neutral(value) and any(_same(value, v) for v in all_values):
                names = ", ".join(getattr(t, "id", "?") for t in node.targets)
                hits.append((node.lineno, f"module constant {names} = {value!r} is a value of the file"))
    return sorted(hits)


@pytest.mark.parametrize("decision,filename", sorted(MODULES.items()))
def test_no_module_carries_a_literal_copy_of_its_yaml_block(decision, filename):
    source = (DECISIONS_DIR / filename).read_text()
    block = load_decisions_yaml()[decision]
    hits = literal_copies(source, block)
    assert hits == [], f"{filename}: literal copies of config/decisions.yaml values:\n" + \
        "\n".join(f"  line {line}: {what}" for line, what in hits)


def test_the_scan_catches_each_shape_it_claims_to():
    """The scanner is not vacuous: each of the three shapes is reported, neutral
    numbers and unrelated numbers are not."""
    block = load_decisions_yaml()["disclose_income"]
    sample = (
        "SIGMA = 9.899547\n"                                   # 3. module constant
        "INTERCEPTS = {1: 0.0089007, 2: 0.0055352}\n"          # 2. dict literal, int keys
        "def f(p):\n"
        "    s = p.get('sigma_overall', 9.899547)\n"           # 1. .get fallback
        "    a = p.get('mean', 0)\n"                           # neutral: not flagged
        "    b = p.get('income_median', 0.5)\n"                # not a key of the block
        "    c = {'sd': 123.456}\n"                            # key of the block, other value
        "    return 100.0 / 112.0\n"                            # arithmetic: not scanned
    )
    hits = literal_copies(sample, block)
    assert [line for line, _ in hits] == [1, 2, 2, 4], hits


# ----------------------------------------------------------------------------- (b)

def _delete(block: dict, dotted: str) -> dict:
    """A deep copy of ``block`` without the key at ``dotted``."""
    copied = copy.deepcopy(block)
    node = copied
    *parents, last = dotted.split(".")
    for key in parents:
        node = node[key]
    del node[last]
    return copied


def _assert_names_decision_and_key(exc_info, decision, dotted):
    message = str(exc_info.value)
    assert decision in message, message
    assert dotted in message, message


D1_AGENT = {"Agreeable": 3.5, "OpennessBig5": 4.0, "Honesty_Humility": 3.4, "ExtraversionBig5": 3.5,
            "NeuroticismBig5": 2.7, "ReligiousAffiliation": 1, "ReligiousService": 2,
            "TWT+Sospeso [=AW2+AX2]{Periods 1+2}": 10.0, "Assigned Allowance Level": 3, "income": 5000.0}
D1_KEYS = ["equation1_coefficients.religious", "equation2_coefficients.neuroticism", "intercept",
           "z_scoring.Agreeable.sd", "z_scoring.religious_composite",
           "z_scoring.TWT_Sospeso.mean", "z_scoring.TWT_Sospeso.sd",
           "religiosity.service_max", "composite_z_scoring.weighted_prosocial.sd",
           "composite_z_scoring.weighted_disclosure_categorical.mean", "composite_z_scoring.anchored_pb.sd",
           "anchor_weights.observed_prosocial", "anchor_weights.prosocial_weight",
           "categorical_intercepts.level_3",
           "stochastic.sigma_overall", "stochastic.scale_factor",
           "stochastic.sigma_quintile.3", "stochastic.quintile_scale_factors.3"]


def _d1_params(dotted=None):
    p = load_decisions_yaml()["disclose_income"]
    p["income_mode"] = "Categorical only"
    p["stochastic"]["sigma_value"] = 1.0                      # sigma on -> the sigma keys are read
    p["stochastic"]["sigma_strategy"] = "quintile" if dotted and "quintile" in dotted else "overall"
    return _delete(p, dotted) if dotted else p


def test_disclose_income_runs_on_the_full_block():
    out = disclose_income_stochastic(dict(D1_AGENT), _d1_params(), np.random.default_rng(0), {},
                                     pop_context="documentation")
    assert out["disclose_income"] in ("Y", "N")


@pytest.mark.parametrize("dotted", D1_KEYS)
def test_disclose_income_missing_key_raises(dotted):
    with pytest.raises(MissingDecisionKey) as e:
        disclose_income_stochastic(dict(D1_AGENT), _d1_params(dotted), np.random.default_rng(0), {},
                                   pop_context="documentation")
    _assert_names_decision_and_key(e, "disclose_income", dotted)


D2_AGENT = {"ExtraversionBig5": 3.5, "NeuroticismBig5": 2.7, "Agreeable": 3.5,
            "Assigned Allowance Level": 2, "income": 5000.0, "disclose_income": "Y"}
D2_KEYS = ["equation_coefficients.agreeable", "intercept", "z_scoring.NeuroticismBig5.mean",
           "categorical_intercepts.level_2", "composite_z_scoring.weighted_dd_categorical.sd",
           "composite_z_scoring.weighted_dd_cont.sd",
           "stochastic.sigma_overall", "stochastic.scale_factor",
           "stochastic.sigma_quintile.2", "stochastic.quintile_scale_factors.2"]


def _d2_params(dotted=None):
    p = load_decisions_yaml()["disclose_documents"]
    p["income_mode"] = "continuous" if dotted and "weighted_dd_cont" in dotted else "Categorical only"
    p["stochastic"]["sigma_value"] = 1.0
    p["stochastic"]["sigma_strategy"] = "quintile" if dotted and "quintile" in dotted else "overall"
    return _delete(p, dotted) if dotted else p


def test_disclose_documents_runs_on_the_full_block():
    sim = {"discount_income_threshold": 12500.0, "income_stats": {"mean": 5000.0, "sd": 1000.0}}
    out = disclose_documents_stochastic(dict(D2_AGENT), _d2_params(), np.random.default_rng(0), sim,
                                        pop_context="documentation")
    assert out["disclose_documents"] in ("Y", "N")


@pytest.mark.parametrize("dotted", D2_KEYS)
def test_disclose_documents_missing_key_raises(dotted):
    sim = {"discount_income_threshold": 12500.0, "income_stats": {"mean": 5000.0, "sd": 1000.0}}
    with pytest.raises(MissingDecisionKey) as e:
        disclose_documents_stochastic(dict(D2_AGENT), _d2_params(dotted), np.random.default_rng(0), sim,
                                      pop_context="documentation")
    _assert_names_decision_and_key(e, "disclose_documents", dotted)


D3_AGENT = {"Honesty_Humility": 3.5, "Assigned Allowance Level": 3, "Group_experiment": "NoSub",
            "TWT+Sospeso [=AW2+AX2]{Periods 1+2}": 20.0, "Study Program": "CLEAM"}
D3_SIM = {"simulation": {"income_distribution": "lognormal", "lognormal_mu": 10.0,
                         "lognormal_sigma": 0.5, "lognormal_min": 0.0}}
D3_KEYS = ["regression_coefficients.categorical.intercept", "regression_coefficients.categorical.beta_hh",
           "regression_coefficients.categorical.beta_group", "regression_coefficients.categorical.beta_income_q",
           "regression_coefficients.categorical.beta_study",
           "regression_coefficients.continuous.beta_income_linear",
           "scaling.observed_range", "scaling.predicted_range.categorical",
           "study_program.category_keys", "study_program.program_to_category",
           "anchor_weights.observed", "anchor_weights.predicted", "adjustment.shift_value",
           "stochastic.sigma_value", "stochastic.sigma_overall", "stochastic.scale_factor",
           "stochastic.sigma_quintile.3", "stochastic.quintile_scale_factors.3"]


def _d3_params(dotted=None):
    p = load_decisions_yaml()["donation_default"]
    p["regression_coefficients"]["income_mode"] = "continuous" if dotted and ".continuous." in dotted else "categorical"
    p["stochastic"]["in_copula"] = True                      # copula draws -> the sigma keys are read
    p["stochastic"]["sigma_strategy"] = "quintile" if dotted and "quintile" in dotted else "overall"
    return _delete(p, dotted) if dotted else p


def test_donation_default_runs_on_the_full_block():
    out = donation_default(dict(D3_AGENT), _d3_params(), np.random.default_rng(0), copy.deepcopy(D3_SIM),
                           pop_context="copula")
    assert 0.0 <= out["donation_default"] <= 1.0


@pytest.mark.parametrize("dotted", D3_KEYS)
def test_donation_default_missing_key_raises(dotted):
    with pytest.raises(MissingDecisionKey) as e:
        donation_default(dict(D3_AGENT), _d3_params(dotted), np.random.default_rng(0), copy.deepcopy(D3_SIM),
                         pop_context="copula")
    _assert_names_decision_and_key(e, "donation_default", dotted)


D4_AGENTS = pd.DataFrame([
    {"ExtraversionBig5": 3.5, "Agreeable": 3.5, "NeuroticismBig5": 2.7, "ConscientiousnessBig5": 3.6,
     "OpennessBig5": 4.0, "Education": 1, "Assigned Allowance Level": 3, "income": 5000.0, "stdactions": 1.1},
    {"ExtraversionBig5": 2.5, "Agreeable": 3.9, "NeuroticismBig5": 3.1, "ConscientiousnessBig5": 4.1,
     "OpennessBig5": 3.6, "Education": 2, "Assigned Allowance Level": 1, "income": 2000.0, "stdactions": 0.7},
    {"ExtraversionBig5": 4.2, "Agreeable": 3.1, "NeuroticismBig5": 2.2, "ConscientiousnessBig5": 3.2,
     "OpennessBig5": 4.4, "Education": 1, "Assigned Allowance Level": 5, "income": 9000.0, "stdactions": 1.6},
])
D4_KEYS = ["coefficients.ttp.education", "coefficients.loyalty.openness", "coefficients.wtp.income",
           "coefficients.risk_taking.conscientiousness", "coefficients.flexibility.neuroticism",
           "z_scoring.ConscientiousnessBig5.sd", "z_scoring.stdactions.mean",
           "flexibility_anchor.observed_weight", "flexibility_anchor.calculated_weight",
           "categorical_income_effects.wtp.level_3", "categorical_income_effects.risk_taking.intercept",
           "stochastic.mechanisms.wtp.sigma_overall", "stochastic.mechanisms.wtp.scale_factor",
           "stochastic.mechanisms.wtp.sigma_quintile.3", "stochastic.mechanisms.wtp.quintile_scale_factors.3"]


def _d4_run(params, sim):
    rng = np.random.default_rng(int(sim["seeds"][0]) + 4000)
    return rejected_transaction_defaults(D4_AGENTS.iloc[0].to_dict(), params, rng, sim,
                                         pop_context="documentation")


@pytest.fixture(scope="module")
def d4_full():
    """The full block (sigma on) and the Pass-1 population statistics computed from it."""
    p = load_decisions_yaml()["rejected_transaction_defaults"]
    p["stochastic"]["sigma_value"] = 1.0
    incomes = D4_AGENTS["income"].tolist()
    sim = {"income_stats": {"mean": float(np.mean(incomes)), "sd": float(np.std(incomes, ddof=1))},
           "seeds": [11, 22, 33]}
    sim["rtd_population_stats"] = compute_rtd_population_stats(
        D4_AGENTS, incomes, p, sim, pop_context="documentation",
        agent_base_seeds=sim["seeds"], decision_offset=4000)
    return p, sim


def test_rejected_transaction_defaults_runs_on_the_full_block(d4_full):
    p, sim = d4_full
    out = _d4_run(p, sim)
    assert set(MECHANISMS) <= {"ttp", "loyalty", "wtp", "risk_taking", "flexibility"}
    assert isinstance(out["rejected_transaction_defaults"], list)


@pytest.mark.parametrize("dotted", D4_KEYS)
def test_rejected_transaction_defaults_missing_key_raises(d4_full, dotted):
    p, sim = d4_full
    p = _delete(p, dotted)
    if dotted.startswith("categorical_income_effects"):
        p["income_mode"] = "categorical"
    if "quintile" in dotted:                                   # agent 0 is at allowance level 3
        p["stochastic"]["mechanisms"]["wtp"]["sigma_strategy"] = "quintile"
    with pytest.raises(MissingDecisionKey) as e:
        _d4_run(p, sim)
    _assert_names_decision_and_key(e, "rejected_transaction_defaults", dotted)


# ----------------------------------------------------------------------------- (c)

def test_config_readers_raise_when_the_file_lacks_the_key():
    from src.config import DecisionsConfig, rtd_sigma_coefficient_defaults

    rtd = load_decisions_yaml()["rejected_transaction_defaults"]
    assert rtd_sigma_coefficient_defaults(rtd) == (0.5, {lvl: 0.5 for lvl in "12345"})
    for dotted in ("stochastic.mechanisms.wtp.scale_factor",
                   "stochastic.mechanisms.wtp.quintile_scale_factors.3",
                   "stochastic.mechanisms"):
        with pytest.raises(KeyError) as e:
            rtd_sigma_coefficient_defaults(_delete(rtd, dotted))
        assert "rejected_transaction_defaults." + dotted in str(e.value), str(e.value)

    decisions = load_decisions_yaml()
    repo = DecisionsConfig(decisions, {}, decisions_path="<test>")
    assert repo.donation_adjustment_shift() == -4.0
    del decisions["donation_default"]["adjustment"]["shift_value"]
    with pytest.raises(KeyError) as e:
        DecisionsConfig(decisions, {}, decisions_path="<test>").donation_adjustment_shift()
    assert "donation_default.adjustment.shift_value" in str(e.value)
