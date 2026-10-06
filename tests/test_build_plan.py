"""
Plan-builder tests for the UI -> engine seam (app/seam/build_plan.py).

Two kinds of checks:

1. Pure plan tests: dict snapshots go in, and the result keys, sub-run order,
   messages (verbatim), decisions_to_run, default_decisions_list and the
   R14 expectations come out as specified.

2. Parity with the pre-seam mapping code: the six ``_apply_*`` functions of
   the last pre-migration ``app/simulation.py`` (commit 80ee344, extracted with
   ``git show`` and executed against a stub ``st.session_state``) are run on a
   fake orchestrator, and the seam's patches - applied to a fresh copy of
   config/decisions.yaml - must produce the same decision blocks, except for
   exactly the deltas the step's rulings prescribe (R10 flat coefficient set,
   R12 sigma constant, R18 sigma_enabled default, no dead keys).
"""
import copy
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.contract.plan import (
    Replace, RunPlan, SavedExpectation, SubRun, UiMessage, apply_patches, deep_merge_patch,
    hash_result_columns,
)
from src.engine.core import DECISION_ORDER
from app.seam import build_plan as bp
from app.seam.config_repo import DecisionsConfig, get_config_repo
from app.seam.execute import configure_engine, execute, verify_saved_expectations
from app.seam.snapshot import SessionSnapshot, take_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMMIT = "80ee344"          # last pre-migration app/simulation.py
# The owner's September 2026 work in the original repo (Decision 4: Flexibility element,
# Section-6 rank aggregation, "Use This Config"). Decision 4's mapping is checked
# against THIS version of app/simulation.py; every other mapping against LEGACY_COMMIT.
SEPTEMBER_COMMIT = "e973a99"
_SEPTEMBER_FUNCTIONS = ("_apply_rejected_transaction_config", "_apply_saved_rejected_transaction_config")
ALL = list(DECISION_ORDER)

# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def repo():
    return get_config_repo()


@pytest.fixture(scope="module")
def default_values():
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    return DEFAULT_DECISION_VALUES


def _sim_params():
    from app.models import SimulationParameters
    return SimulationParameters()


def _donation_keys(repo):
    """The donation_coeff_*_{cat,cont} keys exactly as app.models.load_coefficient_set writes them."""
    keys = {}
    for suffix, mode in (("cat", "categorical"), ("cont", "continuous")):
        c = repo.donation_coefficient_set(mode)
        keys[f"donation_coeff_intercept_{suffix}"] = c["intercept"]
        keys[f"donation_coeff_hh_{suffix}"] = c["beta_hh"]
        keys[f"donation_coeff_linear_{suffix}"] = c.get("beta_income_linear", 0.0)
        for stem, label in (("midsub", "MidSub"), ("nosub", "NoSub"), ("fullsub", "FullSub")):
            keys[f"donation_coeff_{stem}_{suffix}"] = c["beta_group"][label]
        q = c.get("beta_income_q", {})
        for stem, label in (("q1", "Q1"), ("q2", "Q2"), ("q3", "Q3"), ("q4", "Q4"), ("q5", "Q5")):
            keys[f"donation_coeff_{stem}_{suffix}"] = q.get(label, 0.0)
        for stem, label in (("incoming", "Incoming"), ("law", "Law5yr"), ("ug", "UG3yr"), ("grad", "Grad2yr")):
            keys[f"donation_coeff_{stem}_{suffix}"] = c["beta_study"][label]
    return keys


def _unsuffixed_donation_keys(repo, mode):
    """The unsuffixed donation_coeff_* keys (what the legacy get_current_coefficients read)."""
    suffix = "cont" if mode == "continuous" else "cat"
    return {k[:-len(suffix) - 1]: v for k, v in _donation_keys(repo).items() if k.endswith("_" + suffix)}


def base_state(repo, selected=None, **overrides):
    """A session snapshot dict with the app's defaults for every key the seam reads."""
    state = {
        "sim_params": _sim_params(),
        "decision_params": SimpleNamespace(selected_decisions=list(selected if selected is not None else ALL)),
        "population_mode": "Copula (synthetic)",
        "income_spec_mode": "categorical only",
        "seed_input": 42, "seed": 42, "n_agents": 1000, "base_seed": 7,
        # run_individual_decision sets ([decision], []); run_combined_simulation (custom, the rest)
        "custom_decisions": list(selected if selected is not None else ALL),
        "default_decisions": ([] if selected is not None and len(selected) == 1
                              else [d for d in ALL if d not in (selected if selected is not None else ALL)]),
        # donation tab
        "sigma_in_copula": False, "sigma_in_research": True, "sigma_coefficient": 1.0,
        "sigma_value_ui": 9.8995, "anchor_observed_weight": 0.75,
        "donation_sigma_strategy": "overall",
        "donation_quintile_scale_factors": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
        "donation_adjustment_shift": repo.donation_adjustment_shift(),
        # DI tab
        "di_income_mode": "Categorical only", "di_intercept": 0.75, "di_wopb": 0.25, "di_wpb": 0.5,
        "di_sigma_enabled": True, "di_sigma_in_copula": False, "di_sigma_strategy": "overall",
        "di_scale_factor": 1.0, "di_quintile_scale_factors": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
        # DD tab
        "dd_income_mode": "Categorical only", "dd_intercept": -0.75,
        "dd_sigma_enabled": True, "dd_sigma_in_copula": False, "dd_sigma_strategy": "overall",
        "dd_scale_factor": 1.0, "dd_quintile_scale_factors": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
        # RTD tab
        "rtd_income_mode": "Continuous only", "rtd_sigma_enabled": True, "rtd_sigma_in_copula": False,
        "rtd_sigma_strategy": "overall", "rtd_scale_factor": 1.0,
        "rtd_quintile_scale_factors": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
        "rtd_intercept_ttp": 0.05, "rtd_intercept_loyalty": 0.0, "rtd_intercept_wtp": 0.0,
        "rtd_intercept_risk_taking": 0.0, "rtd_intercept_flexibility": 0.0,
        "rtd_flex_observed_weight": 0.25, "rtd_aggregation_enabled": True,
    }
    state.update(_donation_keys(repo))
    state.update(_unsuffixed_donation_keys(repo, "categorical"))
    state.update(overrides)
    return state


def plan_for(repo, default_values, state):
    return bp.build_run_plan(state, repo, default_decision_values=default_values)


def texts(plan):
    return [(m.kind, m.text) for m in plan.messages]


def defaults_message(plan):
    return next(m for m in plan.messages if m.kind == "success" and m.text.startswith("🎲"))


# ------------------------------------------------------------------ legacy extraction

class _StubSessionState:
    """Attribute + item access over a dict, like st.session_state."""

    def __init__(self, data):
        object.__setattr__(self, "_d", dict(data))

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self._d[name] = value

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value

    def __delitem__(self, key):
        del self._d[key]

    def __contains__(self, key):
        return key in self._d

    def get(self, key, default=None):
        return self._d.get(key, default)


_LEGACY_FUNCTIONS = (
    "_apply_simulation_params", "_apply_decision_settings", "_apply_donation_config",
    "_apply_disclose_income_config", "_apply_saved_disclose_income_config",
    "_apply_disclose_income_from_session_state", "_apply_disclose_documents_config",
    "_apply_rejected_transaction_config", "_apply_saved_rejected_transaction_config",
    "apply_selected_donation_config", "collect_decision_settings",
)

_IMPORT_RE = re.compile(r"^(\s*)from app\.(?:pages\.decision_execution|models) import (\w+)\s*$", re.M)


@pytest.fixture(scope="module")
def legacy(default_values):
    """The pre-migration mapping functions, bound to a stub session state."""
    import ast

    def functions_at(commit, names):
        try:
            source = subprocess.run(
                ["git", "show", f"{commit}:app/simulation.py"],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
            pytest.skip(f"legacy app/simulation.py not available from git: {exc}")
        tree = ast.parse(source)
        lines = source.splitlines()
        found = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                found[node.name] = "\n".join(lines[node.lineno - 1:node.end_lineno])
        assert set(found) == set(names), f"legacy function set changed at {commit}"
        return found

    legacy_chunks = functions_at(LEGACY_COMMIT, [n for n in _LEGACY_FUNCTIONS
                                                 if n not in _SEPTEMBER_FUNCTIONS])
    legacy_chunks.update(functions_at(SEPTEMBER_COMMIT, _SEPTEMBER_FUNCTIONS))
    chunks = [legacy_chunks[name] for name in _LEGACY_FUNCTIONS]
    code = "\n\n".join(chunks)
    # the runtime switch removed in step 1 (owner ruling R23)
    code = "\n".join(l for l in code.splitlines() if "model_enabled" not in l)
    # lazy UI imports -> stubs bound to the stub session state
    code = _IMPORT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)} = _stubs[{m.group(2)!r}]", code)

    stub_state = _StubSessionState({})
    st = SimpleNamespace(session_state=stub_state)

    def get_decision_config(name):
        return (stub_state.get("selected_decision_configs") or {}).get(name)

    def get_current_coefficients():
        s = stub_state
        return {
            "intercept": s.donation_coeff_intercept,
            "beta_group": {"MidSub": s.donation_coeff_midsub, "NoSub": s.donation_coeff_nosub,
                           "FullSub": s.donation_coeff_fullsub},
            "beta_income_q": {"Q1": s.donation_coeff_q1, "Q2": s.donation_coeff_q2, "Q3": s.donation_coeff_q3,
                              "Q4": s.get("donation_coeff_q4", 0.0),
                              "Q5": s.get("donation_coeff_q5", s.get("donation_coeff_q45", 0.0))},
            "beta_income_linear": s.donation_coeff_linear,
            "beta_study": {"Incoming": s.donation_coeff_incoming, "Law5yr": s.donation_coeff_law,
                           "UG3yr": s.donation_coeff_ug, "Grad2yr": s.donation_coeff_grad},
            "beta_hh": s.donation_coeff_hh,
        }

    stubs = {
        "get_decision_config": get_decision_config,
        "get_current_coefficients": get_current_coefficients,
        "load_donation_coefficients_from_yaml": lambda: None,
        "DEFAULT_DECISION_VALUES": default_values,
    }
    namespace = {"st": st, "_stubs": stubs, "np": np, "pd": pd, "print": lambda *a, **k: None}
    exec(compile(code, "<legacy app/simulation.py>", "exec"), namespace)

    def bind(state):
        stub_state._d.clear()
        stub_state._d.update(state)

    return SimpleNamespace(bind=bind, state=stub_state, **{n: namespace[n] for n in _LEGACY_FUNCTIONS})


def legacy_orchestrator_config(legacy, repo, state, pop_mode, inc_mode, single_decision, force_di_default=False):
    """Run the legacy runner's apply sequence on a fake orchestrator; return (config, simulation_config)."""
    st_state = dict(state)
    if force_di_default:
        st_state["_dd_standalone_force_di_default"] = True
    legacy.bind(st_state)
    orch = SimpleNamespace(config=repo.fresh_decisions_dict(), simulation_config=copy.deepcopy(repo.simulation_dict()))
    legacy._apply_donation_config(orch, pop_mode, inc_mode)
    legacy._apply_disclose_income_config(orch, pop_mode, inc_mode)
    legacy._apply_disclose_documents_config(orch, pop_mode, inc_mode)
    legacy._apply_rejected_transaction_config(
        orch, pop_mode, inc_mode if single_decision == ["rejected_transaction_defaults"] else None)
    legacy._apply_simulation_params(orch)
    legacy._apply_decision_settings(orch, legacy.collect_decision_settings())
    return orch.config, orch.simulation_config


def seam_engine_config(repo, sub_run, custom_decisions=None):
    engine = SimpleNamespace(config={}, simulation_config=copy.deepcopy(repo.simulation_dict()))
    configure_engine(engine, sub_run, repo, custom_decisions=custom_decisions)
    return engine.config, engine.simulation_config


def assert_donation_parity(legacy_block, new_block, repo, expected_coefficient):
    """The donation block must match the legacy one except for R10 / R12 / the dead keys."""
    legacy_rc = legacy_block["regression_coefficients"]
    new_rc = new_block["regression_coefficients"]
    mode = new_rc["income_mode"]
    assert mode == legacy_rc["income_mode"]
    assert new_block["regression"]["income_mode"] == legacy_block["regression"]["income_mode"]
    # R10: the nested blocks are gone; the flat set equals the file's block for the mode
    assert "categorical" not in new_rc and "continuous" not in new_rc
    for key, value in legacy_rc[mode].items():
        assert new_rc[key] == value, key
    # R12: sigma_value = sigma_overall x coefficient, not 9.8995 x coefficient
    legacy_stoch = dict(legacy_block["stochastic"])
    new_stoch = dict(new_block["stochastic"])
    if legacy_stoch["sigma_value"] != 0.0:
        assert legacy_stoch.pop("sigma_value") == pytest.approx(9.8995 * expected_coefficient)
        assert new_stoch.pop("sigma_value") == pytest.approx(repo.donation_sigma_overall() * expected_coefficient)
    for dead in ("sigma_coefficient", "sigma_in_copula", "sigma_in_research"):
        legacy_stoch.pop(dead, None)
    assert new_stoch == legacy_stoch
    assert new_block["anchor_weights"] == legacy_block["anchor_weights"]
    assert new_block["adjustment"] == legacy_block["adjustment"]
    assert new_block["truncation"] == legacy_block["truncation"]


def assert_full_parity(legacy, repo, state, sub_run, single_decision, force_di_default=False,
                       expected_coefficient=1.0):
    legacy_cfg, legacy_sim = legacy_orchestrator_config(
        legacy, repo, state, sub_run.population, sub_run.income_mode, single_decision, force_di_default)
    # the legacy runner wrote custom_decisions only when both session keys existed
    custom = (tuple(state["custom_decisions"])
              if "custom_decisions" in state and "default_decisions" in state else None)
    new_cfg, new_sim = seam_engine_config(repo, sub_run, custom_decisions=custom)
    for decision in ("disclose_income", "disclose_documents", "rejected_transaction_defaults"):
        assert new_cfg[decision] == legacy_cfg[decision], decision
    assert_donation_parity(legacy_cfg["donation_default"], new_cfg["donation_default"], repo, expected_coefficient)
    for other in set(legacy_cfg) - {"donation_default", "disclose_income", "disclose_documents",
                                    "rejected_transaction_defaults"}:
        assert new_cfg[other] == legacy_cfg[other], other
    assert new_sim == legacy_sim  # includes custom_decisions (df.attrs parity)


# ============================================================================ contract

def test_deep_merge_and_replace():
    base = {"a": {"x": 1, "y": {"z": 2}}, "b": 1}
    deep_merge_patch(base, {"a": {"y": {"w": 3}}, "b": Replace({"n": 1}), "c": [1, 2]})
    assert base == {"a": {"x": 1, "y": {"z": 2, "w": 3}}, "b": {"n": 1}, "c": [1, 2]}
    cfg = apply_patches({"d1": {"k": 1}}, {"d1": {"k": 2}, "d2": {"k": 3}})
    assert cfg == {"d1": {"k": 2}, "d2": {"k": 3}}


def test_hash_matches_saved_config_writer():
    from app.state.saved_configs import hash_result_columns as writer_hash
    df = pd.DataFrame({"disclose_income": ["Y", "N", "Y"], "disclose_income_raw": [0.1, -0.2, 0.3],
                       "other": [1, 2, 3]})
    cols = ["disclose_income", "disclose_income_raw"]
    assert hash_result_columns(df, cols) == writer_hash(df, cols)
    assert hash_result_columns(df, cols) != hash_result_columns(df.assign(disclose_income=["N", "N", "Y"]), cols)
    assert hash_result_columns(df, []) is None
    assert hash_result_columns(df, ["missing"]) is None


def test_snapshot_is_frozen_and_mapping_like():
    snap = take_snapshot({"a": 1})
    assert snap["a"] == 1 and snap.get("b", 2) == 2 and "a" in snap and len(snap) == 1
    with pytest.raises(TypeError):
        snap.a = 2
    assert take_snapshot(snap) is snap
    assert isinstance(SessionSnapshot({}), SessionSnapshot)


def test_sub_run_validates_population_and_income_mode():
    with pytest.raises(ValueError):
        SubRun("k", "nope", "categorical", 1, 1, None, {}, {}, {}, (), None)
    with pytest.raises(ValueError):
        UiMessage("toast", "x")


# ============================================================================ plans

def test_complete_run_copula_categorical(repo, default_values):
    state = base_state(repo)
    plan = plan_for(repo, default_values, state)
    assert plan.result_keys == ("categorical",)
    (sub,) = plan.sub_runs
    assert (sub.population, sub.income_mode, sub.seed, sub.n_agents) == ("copula", "categorical", 42, 1000)
    assert sub.decisions_to_run is None
    assert sub.default_decisions_list == ()
    assert sub.purchasing_limits is None
    assert plan.saved_expectations == ()
    assert texts(plan) == [
        ("success", "🎲 Using configured defaults: donation_default: 10.0%, disclose_income: 50% Y / 50% N, "
                    "disclose_documents: 50% Y / 50% N, rejected_transaction_defaults: forgo_transaction only, "
                    "vendor_choice_weights: 4 params selected (price, quality, proximity, sustainability), "
                    "purchasing_quantity: RANDOM_WITHIN_LIMIT, purchasing_frequency: CALCULATED, "
                    "vendor_selection: deterministic, purchase_vs_bid: 50% Purchase Now / 50% bid, "
                    "bid_value: RANDOM_WITHIN_RANGE, rejected_transaction_option: forgo_transaction, "
                    "rejected_bid_value: NA, final_donation_rate: 10.0%"),
        ("info", "🎲 Using synthetic agents from copula"),
    ]
    meta = plan.metadata
    assert (meta.effective_population_mode, meta.effective_income_mode) == ("Copula (synthetic)", "categorical only")
    assert meta.result_keys == ("categorical",) and meta.is_comparison is False
    assert meta.custom_decisions == tuple(ALL) and meta.default_decisions == ()
    assert (meta.seed, meta.n_agents) == (42, 1000)


def test_compare_all_compare_both_keys_and_order(repo, default_values):
    state = base_state(repo, population_mode="Compare all", income_spec_mode="Compare both")
    plan = plan_for(repo, default_values, state)
    assert plan.result_keys == ("copula_categorical", "copula_continuous",
                                "research_spec_categorical", "research_spec_continuous",
                                "research_baseline_categorical", "research_baseline_continuous")
    assert [s.population for s in plan.sub_runs] == ["copula", "copula", "documentation", "documentation",
                                                     "baseline", "baseline"]
    assert texts(plan)[-1] == ("info", "🔄 Running Compare All mode - each population uses its natural agent source")
    assert plan.metadata.is_comparison is True


def test_compare_all_single_income_and_research_messages(repo, default_values):
    plan = plan_for(repo, default_values, base_state(repo, population_mode="Compare all",
                                                     income_spec_mode="continuous only"))
    assert plan.result_keys == ("copula_continuous", "research_spec_continuous", "research_baseline_continuous")
    plan = plan_for(repo, default_values, base_state(repo, population_mode="Research Specification",
                                                     income_spec_mode="Compare both"))
    assert plan.result_keys == ("categorical", "continuous")
    assert texts(plan)[-1] == ("info", "📊 Using original 280 participants")
    assert all(s.population == "documentation" for s in plan.sub_runs)
    plan = plan_for(repo, default_values, base_state(repo, population_mode="Research Baseline"))
    assert plan.sub_runs[0].population == "baseline"


def test_disclose_income_only_branch(repo, default_values):
    state = base_state(repo, selected=["disclose_income"], di_income_mode="Compare both")
    plan = plan_for(repo, default_values, state)
    assert plan.result_keys == ("categorical", "continuous")
    assert plan.sub_runs[0].decisions_to_run == ("disclose_income",)
    assert texts(plan)[1] == ("caption", "🎯 Using Disclose Income specific mode: Compare both")
    assert plan.metadata.effective_income_mode == "Compare both"
    # each sub-run's DI patch carries ITS income mode (Compare both), RTD keeps the tab mode
    assert plan.sub_runs[0].decision_config_patches["disclose_income"]["income_mode"] == "categorical"
    assert plan.sub_runs[1].decision_config_patches["disclose_income"]["income_mode"] == "continuous"
    assert plan.sub_runs[1].decision_config_patches["rejected_transaction_defaults"]["income_mode"] == "continuous"
    assert plan.saved_expectations == ()


def test_disclose_documents_only_forces_di_default(repo, default_values):
    state = base_state(repo, selected=["disclose_documents"], dd_income_mode="Continuous only")
    plan = plan_for(repo, default_values, state)
    assert plan.result_keys == ("continuous",)
    (sub,) = plan.sub_runs
    assert sub.decisions_to_run == ("disclose_income", "disclose_documents")
    assert sub.default_decisions_list == ("disclose_income",)
    assert plan.metadata.default_decisions == ()  # the session's own list is NOT changed
    assert texts(plan)[1:] == [
        ("caption", "🎯 Using Disclose Documents specific mode: Continuous only"),
        ("caption", "🔐 Eligibility: assigning Disclose Income via its default probability (50% Y) "
                    "to identify the qualified subgroup (income < threshold AND disclosed income)."),
        ("info", "🎲 Using synthetic agents from copula"),
    ]


def test_disclose_documents_only_with_saved_di_uses_it(repo, default_values):
    di_cfg = {"result_key": "categorical", "params": {"income_mode": "Categorical only", "intercept": 0.9,
                                                      "anchor_weights": {"observed_prosocial": 0.3, "prosocial_weight": 0.6}},
              "income_mode": "Categorical only", "population_mode": "Copula (synthetic)",
              "source": "individual_disclose_income_run", "original_seed": 5, "original_n_agents": 50}
    state = base_state(repo, selected=["disclose_documents"], selected_decision_configs={"disclose_income": di_cfg})
    plan = plan_for(repo, default_values, state)
    (sub,) = plan.sub_runs
    assert sub.default_decisions_list == ()
    assert (sub.seed, sub.n_agents) == (5, 50)
    assert texts(plan)[:3] == [
        ("info", "📋 Using 1 saved decision configuration(s)"),
        ("caption", "  📋 Disclose Income: Categorical only"),
        ("caption", "🔑 Using saved seed: 5, agents: 50"),
    ]
    assert ("caption", "🔐 Eligibility: using the selected Disclose Income configuration to "
                       "identify the qualified subgroup (income < threshold AND disclosed income).") in texts(plan)
    di_patch = sub.decision_config_patches["disclose_income"]
    assert di_patch["intercept"] == 0.9
    assert di_patch["anchor_weights"] == {"observed_prosocial": 0.3, "prosocial_weight": 0.6}


def test_rejected_transaction_only_branch(repo, default_values):
    state = base_state(repo, selected=["rejected_transaction_defaults"], rtd_income_mode="Compare both")
    plan = plan_for(repo, default_values, state)
    assert plan.result_keys == ("categorical", "continuous")
    assert texts(plan)[1] == ("caption", "🎯 Using Rejected Transaction Defaults specific mode: Compare both")
    assert plan.sub_runs[0].decision_config_patches["rejected_transaction_defaults"]["income_mode"] == "categorical"
    assert plan.sub_runs[1].decision_config_patches["rejected_transaction_defaults"]["income_mode"] == "continuous"


def test_donation_only_branch(repo, default_values):
    state = base_state(repo, selected=["donation_default"], income_spec_mode="Compare both")
    plan = plan_for(repo, default_values, state)
    assert plan.result_keys == ("categorical", "continuous")
    assert texts(plan)[1] == ("caption", "🎯 Using Donation Default income mode: Compare both")
    cat = plan.sub_runs[0].decision_config_patches["donation_default"]["regression_coefficients"]
    cont = plan.sub_runs[1].decision_config_patches["donation_default"]["regression_coefficients"]
    assert isinstance(cat, Replace) and isinstance(cont, Replace)
    assert cat.value["intercept"] == repo.donation_coefficient_set("categorical")["intercept"]
    assert cont.value["intercept"] == repo.donation_coefficient_set("continuous")["intercept"]
    assert cont.value["beta_income_linear"] == repo.donation_coefficient_set("continuous")["beta_income_linear"]


def test_combined_with_saved_configs_pins_seed_population_income_and_expectations(repo, default_values):
    di_cfg = {"result_key": "research_spec_continuous", "params": {"income_mode": "Continuous only"},
              "income_mode": "Continuous only", "population_mode": "Research Specification",
              "source": "individual_disclose_income_run", "original_seed": 11, "original_n_agents": 60,
              "result_columns": ["disclose_income", "disclose_income_raw"], "result_sha256": "abc"}
    don_cfg = {"result_key": "continuous", "donation_income_mode": "continuous only",
               "income_spec_mode": "continuous only", "population_mode": "Research Specification",
               "coefficients": {}, "stochastic_params": {"stochastic": {"sigma_coefficient": 0.5},
                                                          "anchor_weights": {"observed": 0.6, "predicted": 0.4}},
               "source": "individual_donation_default_run", "original_seed": 11, "original_n_agents": 60,
               "result_columns": ["donation_default"], "result_sha256": "def"}
    state = base_state(repo, population_mode="Compare all", income_spec_mode="Compare both",
                       selected_decision_configs={"disclose_income": di_cfg, "donation_default": don_cfg})
    plan = plan_for(repo, default_values, state)
    assert plan.result_keys == ("continuous",)
    (sub,) = plan.sub_runs
    assert (sub.population, sub.seed, sub.n_agents) == ("documentation", 11, 60)
    assert texts(plan) == [
        ("info", "📋 Using 2 saved decision configuration(s)"),
        ("caption", "  📋 Disclose Income: Continuous only"),
        ("caption", "  🎯 Donation Default: continuous only"),
        ("caption", "🔑 Using saved seed: 11, agents: 60"),
        defaults_message(plan).kind and ("success", defaults_message(plan).text),
        ("caption", "📋 Using saved Disclose Income mode: Continuous only"),
        ("caption", "🔄 Running with saved population mode: Research Specification"),
        ("info", "📊 Using original 280 participants"),
    ]
    assert plan.saved_expectations == (
        SavedExpectation("disclose_income", "continuous", ("disclose_income", "disclose_income_raw"), "abc"),
        SavedExpectation("donation_default", "continuous", ("donation_default",), "def"),
    )
    donation = sub.decision_config_patches["donation_default"]
    assert donation["anchor_weights"] == {"observed": 0.6, "predicted": 0.4}
    assert donation["stochastic"]["sigma_value"] == pytest.approx(repo.donation_sigma_overall() * 0.5)
    assert donation["stochastic"]["scale_factor"] == 0.5
    assert donation["regression_coefficients"].value["income_mode"] == "continuous"
    assert plan.metadata.effective_population_mode == "Research Specification"
    assert plan.metadata.effective_income_mode == "Continuous only"


def test_saved_disclose_documents_config_pins_population_and_drives_dd(repo, default_values):
    dd_cfg = {"result_key": "baseline_categorical", "params": {"income_mode": "Categorical only", "intercept": -0.2},
              "income_mode": "Categorical only", "population_mode": None,
              "source": "individual_disclose_documents_run", "original_seed": 3, "original_n_agents": 40,
              "result_columns": ["disclose_documents"], "result_sha256": "x"}
    state = base_state(repo, population_mode="Compare all", income_spec_mode="continuous only",
                       selected_decision_configs={"disclose_documents": dd_cfg}, dd_income_mode="Continuous only")
    plan = plan_for(repo, default_values, state)
    (sub,) = plan.sub_runs
    assert sub.population == "baseline"      # R14: pinned from the result key
    assert plan.result_keys == ("continuous",)   # DD does not pin the effective income mode
    assert ("caption", "🔄 Running with saved population mode: Research Baseline") in texts(plan)
    assert ("caption", "  ✓ Disclose Documents") in texts(plan)
    dd_patch = sub.decision_config_patches["disclose_documents"]
    # R17: saved precedence - saved mode + intercept, live stochastic toggles, unconditional strategy copy
    assert dd_patch["income_mode"] == "categorical"
    assert dd_patch["intercept"] == -0.2
    assert dd_patch["stochastic"] == {"sigma_value": 0.0, "in_copula": False, "sigma_strategy": "overall",
                                      "scale_factor": 1.0, "quintile_scale_factors": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0}}
    # no income-mode match ('categorical' saved, 'continuous' run) -> first sub-run
    assert plan.saved_expectations == (SavedExpectation("disclose_documents", "continuous", ("disclose_documents",), "x"),)


def test_saved_di_and_dd_patches_share_precedence(repo, default_values):
    common = {"params": {"income_mode": "Compare both", "intercept": 0.1}, "income_mode": "Continuous only",
              "population_mode": "Copula (synthetic)", "original_seed": 1, "original_n_agents": 10}
    state = base_state(repo, selected_decision_configs={
        "disclose_income": dict(common, source="individual_disclose_income_run"),
        "disclose_documents": dict(common, source="individual_disclose_documents_run")},
        di_sigma_in_copula=True, dd_sigma_in_copula=True)
    di = bp.build_disclose_income_patch(state, repo, "copula", "categorical")
    dd = bp.build_disclose_documents_patch(state, repo, "copula", "categorical")
    assert di["income_mode"] == dd["income_mode"] == "continuous"   # saved beats the per-run mode
    assert di["intercept"] == dd["intercept"] == 0.1
    assert di["stochastic"]["in_copula"] is True and dd["stochastic"]["in_copula"] is True
    assert di["stochastic"]["sigma_value"] == repo.disclose_income_sigma_overall()
    assert dd["stochastic"]["sigma_value"] == repo.disclose_documents_sigma_overall()
    assert "anchor_weights" not in dd


def test_auto_implied_configs_are_ignored(repo, default_values):
    implied = {"result_key": "implied_copula_synthetic_categorical_only", "source": "auto_implied_single_config",
               "original_seed": 99, "original_n_agents": 5, "population_mode": "Copula (synthetic)",
               "donation_income_mode": "categorical only", "coefficients": {}, "stochastic_params": {}}
    state = base_state(repo, selected_decision_configs={"donation_default": implied})
    plan = plan_for(repo, default_values, state)
    assert (plan.sub_runs[0].seed, plan.sub_runs[0].n_agents) == (42, 1000)
    assert all(not m.text.startswith("📋 Using") for m in plan.messages)
    assert bp.resolve_seed_and_n(state) == (42, 1000, "session_state")


def test_seed_resolution_rules(repo):
    state = base_state(repo)
    assert bp.resolve_seed_and_n(state) == (42, 1000, "session_state")
    state["sim_params"].simulation_mode = "Monte-Carlo Study"
    assert bp.resolve_seed_and_n(state) == (7, 1000, "session_state")
    state = base_state(repo, selected_decision_configs={"disclose_income": {
        "source": "individual_disclose_income_run", "original_seed": 8, "original_n_agents": 80}})
    assert bp.resolve_seed_and_n(state) == (8, 80, "configs")
    plan = bp.build_run_plan(state, get_config_repo(), default_decision_values={"donation_default": 0.1},
                             seed_resolution=(1, 2, "session_state"))
    assert (plan.sub_runs[0].seed, plan.sub_runs[0].n_agents) == (1, 2)
    assert not any(m.text.startswith("🔑") for m in plan.messages)


def test_sigma_enabled_defaults_true_when_absent(repo):
    state = base_state(repo)
    for key in ("di_sigma_enabled", "dd_sigma_enabled", "rtd_sigma_enabled"):
        state.pop(key)
    explicit = base_state(repo)
    for pop in ("documentation", "copula", "baseline"):
        assert bp.build_disclose_income_patch(state, repo, pop, "categorical") == \
            bp.build_disclose_income_patch(explicit, repo, pop, "categorical")
        assert bp.build_disclose_documents_patch(state, repo, pop, "categorical") == \
            bp.build_disclose_documents_patch(explicit, repo, pop, "categorical")
        assert bp.build_rejected_transaction_patch(state, repo, pop, None) == \
            bp.build_rejected_transaction_patch(explicit, repo, pop, None)
    assert bp.build_disclose_income_patch(state, repo, "documentation", "categorical")["stochastic"]["sigma_value"] > 0


def test_donation_patch_reads_session_keys_not_file(repo):
    state = base_state(repo, donation_coeff_intercept_cat=2.5, donation_adjustment_shift=-1.5, sigma_coefficient=0.8,
                       sigma_in_copula=True)
    patch = bp.build_donation_patch(state, repo, "copula", "categorical")
    assert patch["regression_coefficients"].value["intercept"] == 2.5
    assert patch["adjustment"] == {"shift_value": -1.5}
    assert patch["stochastic"]["sigma_value"] == pytest.approx(repo.donation_sigma_overall() * 0.8)
    assert patch["stochastic"]["scale_factor"] == 0.8 and patch["stochastic"]["in_copula"] is True
    # research tick off -> no draw in documentation mode
    patch = bp.build_donation_patch(dict(state, sigma_in_research=False), repo, "documentation", "categorical")
    assert patch["stochastic"]["sigma_value"] == 0.0 and "in_copula" not in patch["stochastic"]
    # no session coefficient keys at all -> the file's set for the mode
    bare = {k: v for k, v in state.items() if not k.startswith("donation_coeff_")}
    patch = bp.build_donation_patch(bare, repo, "copula", "continuous")
    assert patch["regression_coefficients"].value["intercept"] == repo.donation_coefficient_set("continuous")["intercept"]


def test_collect_decision_settings_precedence(repo, default_values):
    state = base_state(repo, disclose_income_probability_y=0.9,
                       _persistent_defaults={"disclose_documents_default_probability_y": 0.2,
                                             "final_donation_rate_default_value": 0.3},
                       vendor_choice_weights_selection=["price", "quality"],
                       rejected_transaction_option_selection="place_bid")
    settings = bp.collect_decision_settings(state, default_values)
    assert list(settings) == list(default_values)
    assert settings["disclose_income"]["probability_y"] == 0.9
    assert settings["disclose_documents"]["probability_y"] == 0.2
    assert settings["vendor_choice_weights"]["weights"] == {"price": 0.5, "quality": 0.5, "proximity": 0.0, "sustainability": 0.0}
    assert settings["rejected_transaction_option"] == {"selected_option": "place_bid", "type": "radio_selection"}
    assert settings["final_donation_rate"] == {"value": 0.3, "type": "numeric"}
    assert settings["purchasing_quantity"] == {"value": "RANDOM_WITHIN_LIMIT", "type": "placeholder"}


def test_verify_saved_expectations():
    df = pd.DataFrame({"donation_default": [0.1, 0.2]})
    good = SavedExpectation("donation_default", "categorical", ("donation_default",), hash_result_columns(df, ["donation_default"]))
    bad = SavedExpectation("donation_default", "categorical", ("donation_default",), "nope")
    missing_key = SavedExpectation("disclose_income", "continuous", ("disclose_income",), "nope")
    missing_col = SavedExpectation("disclose_income", "categorical", ("disclose_income",), "nope")
    assert verify_saved_expectations((good, bad, missing_key, missing_col), {"categorical": df}) == (bad,)


# ==================================================================== legacy parity

@pytest.mark.parametrize("pop_mode", ["copula", "documentation", "baseline"])
@pytest.mark.parametrize("inc_mode", ["categorical", "continuous"])
def test_complete_run_patches_match_legacy(legacy, repo, default_values, pop_mode, inc_mode):
    population_mode = {"copula": "Copula (synthetic)", "documentation": "Research Specification",
                       "baseline": "Research Baseline"}[pop_mode]
    state = base_state(repo, population_mode=population_mode, income_spec_mode=f"{inc_mode} only",
                       sigma_in_copula=True, di_sigma_in_copula=True, dd_sigma_in_copula=True,
                       rtd_sigma_in_copula=True, sigma_coefficient=1.3, sigma_value_ui=9.8995 * 1.3,
                       **_unsuffixed_donation_keys(repo, inc_mode))
    plan = plan_for(repo, default_values, state)
    (sub,) = plan.sub_runs
    assert (sub.population, sub.income_mode) == (pop_mode, inc_mode)
    assert_full_parity(legacy, repo, state, sub, None, expected_coefficient=1.3)


def test_individual_runs_match_legacy(legacy, repo, default_values):
    # DD-only run: the force-DI-default flag became SubRun.default_decisions_list
    state = base_state(repo, selected=["disclose_documents"], population_mode="Research Specification")
    plan = plan_for(repo, default_values, state)
    (sub,) = plan.sub_runs
    assert_full_parity(legacy, repo, state, sub, ["disclose_documents"], force_di_default=True)
    # RTD-only Compare both: the explicit per-run mode reaches Decision 4
    state = base_state(repo, selected=["rejected_transaction_defaults"], rtd_income_mode="Compare both",
                       rtd_intercept_ttp=-0.05, rtd_scale_factor=0.7, rtd_sigma_strategy="quintile")
    plan = plan_for(repo, default_values, state)
    for sub in plan.sub_runs:
        assert_full_parity(legacy, repo, state, sub, ["rejected_transaction_defaults"])
    # DI-only Compare both with quintile sigma and custom anchors
    state = base_state(repo, selected=["disclose_income"], di_income_mode="Compare both", di_wopb=0.4, di_wpb=0.7,
                       di_sigma_strategy="quintile", di_scale_factor=0.5,
                       di_quintile_scale_factors={"1": 0.5, "2": 1.0, "3": 1.5, "4": 1.0, "5": 2.0},
                       population_mode="Research Specification")
    plan = plan_for(repo, default_values, state)
    for sub in plan.sub_runs:
        assert_full_parity(legacy, repo, state, sub, ["disclose_income"])


def test_saved_di_config_patch_matches_legacy(legacy, repo, default_values):
    di_cfg = {"result_key": "continuous", "params": {"income_mode": "Compare both", "intercept": 0.9,
                                                     "anchor_weights": {"observed_prosocial": 0.3, "prosocial_weight": 0.6},
                                                     "stochastic": {"sigma_enabled": False}},
              "income_mode": "Continuous only", "population_mode": "Research Specification",
              "source": "individual_disclose_income_run", "original_seed": 5, "original_n_agents": 50}
    state = base_state(repo, selected_decision_configs={"disclose_income": di_cfg}, di_scale_factor=0.3)
    plan = plan_for(repo, default_values, state)
    (sub,) = plan.sub_runs
    assert sub.population == "documentation" and sub.income_mode == "continuous"
    legacy_cfg, _ = legacy_orchestrator_config(legacy, repo, state, "documentation", "continuous", None)
    new_cfg, _ = seam_engine_config(repo, sub)
    assert new_cfg["disclose_income"] == legacy_cfg["disclose_income"]


def test_saved_rtd_config_patch_matches_september(legacy, repo, default_values):
    """A saved Decision 4 configuration ("Use This Config", September 2026) pins the
    population, and its MODEL settings override the tab in a complete run, while the
    stochastic settings follow the current tab - the e973a99 mapping, via the seam."""
    rtd_cfg = {"result_key": "research_spec_categorical",
               "params": {"income_mode": "Compare both",
                          "intercepts": {"ttp": 0.1, "loyalty": -0.2, "wtp": 0.0,
                                         "risk_taking": 0.3, "flexibility": 0.15},
                          "flexibility_anchor": {"observed_weight": 0.4, "calculated_weight": 0.6},
                          "aggregation": {"enabled": True},
                          "stochastic": {"sigma_enabled": False}},
               "income_mode": "Categorical only", "population_mode": "Research Specification",
               "source": "individual_rejected_transaction_defaults_run",
               "original_seed": 5, "original_n_agents": 50}
    state = base_state(repo, selected_decision_configs={"rejected_transaction_defaults": rtd_cfg},
                       rtd_income_mode="Continuous only", rtd_intercept_ttp=-0.3,
                       rtd_flex_observed_weight=0.9, rtd_scale_factor=0.6)
    plan = plan_for(repo, default_values, state)
    (sub,) = plan.sub_runs
    assert sub.population == "documentation"           # pinned by the saved config
    assert (sub.seed, sub.n_agents) == (5, 50)
    assert "🔄 Running with saved population mode: Research Specification" in [t for _, t in texts(plan)]
    assert_full_parity(legacy, repo, state, sub, None)
    rtd = seam_engine_config(repo, sub)[0]["rejected_transaction_defaults"]
    assert rtd["income_mode"] == "categorical"
    assert rtd["intercepts"]["ttp"] == 0.1 and rtd["intercepts"]["flexibility"] == 0.15
    assert rtd["flexibility_anchor"] == {"observed_weight": 0.4, "calculated_weight": 0.6}
    # stochastic: current tab, not the saved snapshot
    assert rtd["stochastic"]["sigma_value"] > 0
    assert rtd["stochastic"]["mechanisms"]["flexibility"]["scale_factor"] == 0.6
    # an individual Decision 4 run keeps reflecting the tab
    state["decision_params"] = SimpleNamespace(selected_decisions=["rejected_transaction_defaults"])
    patch = bp.build_rejected_transaction_patch(state, repo, "documentation", "continuous")
    assert patch["intercepts"]["ttp"] == -0.3
    assert patch["flexibility_anchor"]["observed_weight"] == 0.9


def test_default_settings_message_matches_legacy_builder(legacy, repo, default_values):
    state = base_state(repo, _persistent_defaults={"rejected_transaction_defaults_priority_template": ["place_bid", "forgo_transaction"]})
    legacy.bind(state)
    assert bp.collect_decision_settings(state, default_values) == legacy.collect_decision_settings()
    assert "rejected_transaction_defaults: 2 priorities" in defaults_message(plan_for(repo, default_values, state)).text


# =============================================================== engine end-to-end

def test_execute_reproduces_a_pinned_individual_run(repo, default_values):
    """R14 in practice: an individual DI run pinned by 'Use This Config' is reproduced bit-for-bit
    by the complete run (same seed / agents / decision RNG), so no error is raised."""
    n = 12
    state = base_state(repo, selected=["disclose_income"], population_mode="Research Baseline", n_agents=n)
    single = execute(plan_for(repo, default_values, state), repo)
    assert list(single) == ["categorical"]
    df = single["categorical"]
    cols = [c for c in df.columns if c.startswith("disclose_income")]
    saved = {"result_key": "categorical", "params": {"income_mode": "Categorical only", "intercept": 0.75,
                                                     "anchor_weights": {"observed_prosocial": 0.25, "prosocial_weight": 0.5}},
             "income_mode": "Categorical only", "population_mode": "Research Baseline",
             "source": "individual_disclose_income_run", "original_seed": 42, "original_n_agents": n,
             "result_columns": cols, "result_sha256": hash_result_columns(df, cols)}
    complete = plan_for(repo, default_values, base_state(repo, selected_decision_configs={"disclose_income": saved}))
    assert complete.sub_runs[0].decisions_to_run is None and complete.sub_runs[0].n_agents == n
    results = execute(complete, repo)
    assert verify_saved_expectations(complete.saved_expectations, results) == ()
    assert "purchase_requests" in results["categorical"].columns
    # a different intercept in the pinned record is NOT reproduced -> reported
    tampered = dict(saved, params=dict(saved["params"], intercept=0.2))
    plan = plan_for(repo, default_values, base_state(repo, selected_decision_configs={"disclose_income": tampered}))
    failed = verify_saved_expectations(plan.saved_expectations, execute(plan, repo))
    assert [f.decision for f in failed] == ["disclose_income"]
