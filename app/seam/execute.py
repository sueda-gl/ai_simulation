# app/seam/execute.py
"""
Run a RunPlan against the engine.  No Streamlit anywhere in here.

Per sub-run, in this order:
  1. sample the agents - copula: TraitEngine().sample(n, seed) (that TraitEngine is
     handed to the Engine, which validates the traits with it: one copula load per sub-run);
     research: load_original_participants(n, seed, random_sample) - Specification
     random (seeded by the run seed), Baseline the 280 in file order, cycling
  2. build the decisions dict: a deep copy of the loaded file + the plan's patches
  3. build the simulation dict: a deep copy of the loaded file with
  4. simulation_config['simulation'] <- simulation_params; random_decisions /
     default_decisions <- the SAME decision_settings object; purchasing_limits
     (only when the plan carries one); custom_decisions (the plan's
     metadata.custom_decisions - the former st.session_state.custom_decisions;
     no engine consumer, but it travels into df.attrs['simulation_config']);
     default_decisions_list
  5. Engine(PROFILES[population], decisions=..., simulation=...) and
     engine.run_simulation(n_agents, seed, decisions_to_run, agents_df=agents_df)
then assign_global_transaction_ids on every result.
"""

import copy
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from src.contract.plan import RunPlan, SavedExpectation, SubRun, apply_patches, hash_result_columns
from src.engine.core import Engine
from src.engine.postprocess import assign_global_transaction_ids
from src.engine.profile import PROFILES
from src.engine.sampling import load_original_participants
from src.trait_engine import TraitEngine
from app.seam.config_repo import DecisionsConfig, get_config_repo


def sample_agents(population: str, n_agents: int, seed: int,
                  trait_engine: Optional[TraitEngine] = None) -> pd.DataFrame:
    """The agent source of each population mode (identical to the former runners).
    ``trait_engine``: the loaded copula to sample with (loaded here when None)."""
    if population == "copula":
        return (trait_engine if trait_engine is not None else TraitEngine()).sample(n_agents, seed)
    if population in ("documentation", "baseline"):
        # Research Specification samples randomly from default_rng(seed); Research
        # Baseline cycles through the participants in file order
        return load_original_participants(n_agents, seed,
                                          random_sample=(population == "documentation"))
    raise ValueError(f"Unknown population {population!r}")


def apply_simulation_settings(simulation_config: dict, sub_run: SubRun,
                              custom_decisions: Optional[Sequence[str]] = None) -> dict:
    """
    Write one sub-run's Page-1 parameters and decision settings into a
    simulation.yaml dict IN PLACE (step 4 of the module docstring).

    ``custom_decisions`` (the run's metadata.custom_decisions) is written to
    simulation_config['custom_decisions'] so it travels into
    df.attrs['simulation_config']; None leaves the key alone.
    """
    if "simulation" not in simulation_config or not isinstance(simulation_config.get("simulation"), dict):
        simulation_config["simulation"] = {}
    simulation_config["simulation"].update(sub_run.simulation_params)

    if sub_run.decision_settings:
        # Both keys, the same object: the engine reads default_decisions; random_decisions
        # is kept because df.attrs['simulation_config'] carries it
        simulation_config["random_decisions"] = sub_run.decision_settings
        simulation_config["default_decisions"] = sub_run.decision_settings

    if sub_run.purchasing_limits is not None:
        simulation_config["purchasing_limits"] = sub_run.purchasing_limits

    # Pass information about custom vs default decisions (custom_decisions is written
    # before default_decisions_list so the key order in df.attrs['simulation_config'] is stable)
    if custom_decisions is not None:
        simulation_config["custom_decisions"] = list(custom_decisions)

    simulation_config["default_decisions_list"] = list(sub_run.default_decisions_list)
    return simulation_config


def engine_configuration(sub_run: SubRun, config_repo: DecisionsConfig,
                         custom_decisions: Optional[Sequence[str]] = None) -> Tuple[dict, dict]:
    """(decisions, simulation): the two dicts one sub-run's Engine is built with."""
    decisions = apply_patches(config_repo.fresh_decisions_dict(), sub_run.decision_config_patches)
    simulation = apply_simulation_settings(copy.deepcopy(config_repo.simulation_dict()),
                                           sub_run, custom_decisions)
    return decisions, simulation


def configure_engine(engine: Engine, sub_run: SubRun, config_repo: DecisionsConfig,
                     custom_decisions: Optional[Sequence[str]] = None) -> Engine:
    """
    Apply one sub-run's patches and simulation settings to an already built
    Engine (or any object with ``config`` / ``simulation_config``), in place.
    ``run_sub_run`` hands the same two dicts to the Engine at construction
    instead (:func:`engine_configuration`).
    """
    engine.config = apply_patches(config_repo.fresh_decisions_dict(), sub_run.decision_config_patches)
    apply_simulation_settings(engine.simulation_config, sub_run, custom_decisions)
    return engine


def build_engine(sub_run: SubRun, config_repo: DecisionsConfig,
                 custom_decisions: Optional[Sequence[str]] = None,
                 trait_engine: Optional[TraitEngine] = None) -> Engine:
    """A fresh Engine for one sub-run, built with its patched configuration (and
    the copula the agents were sampled with, so it is loaded once per sub-run)."""
    decisions, simulation = engine_configuration(sub_run, config_repo, custom_decisions)
    return Engine(PROFILES[sub_run.population], decisions=decisions, simulation=simulation,
                  trait_engine=trait_engine)


def run_sub_run(sub_run: SubRun, config_repo: Optional[DecisionsConfig] = None,
                custom_decisions: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Sample agents, configure a fresh Engine and run one sub-run."""
    repo = config_repo if config_repo is not None else get_config_repo()
    trait_engine = TraitEngine() if sub_run.population == "copula" else None
    agents_df = sample_agents(sub_run.population, sub_run.n_agents, sub_run.seed, trait_engine)
    engine = build_engine(sub_run, repo, custom_decisions=custom_decisions, trait_engine=trait_engine)
    decisions = list(sub_run.decisions_to_run) if sub_run.decisions_to_run is not None else None
    return engine.run_simulation(sub_run.n_agents, sub_run.seed, decisions, agents_df=agents_df)


def execute(plan: RunPlan, config_repo: Optional[DecisionsConfig] = None) -> Dict[str, pd.DataFrame]:
    """Run every sub-run in plan order; returns {result_key: DataFrame}."""
    repo = config_repo if config_repo is not None else get_config_repo()
    results: Dict[str, pd.DataFrame] = {}
    for sub_run in plan.sub_runs:
        results[sub_run.result_key] = run_sub_run(sub_run, repo,
                                                  custom_decisions=plan.metadata.custom_decisions)

    # Assign global transaction IDs to ensure consistency across exports
    for key in results:
        results[key] = assign_global_transaction_ids(results[key])
    return results


def verify_saved_expectations(expectations: Sequence[SavedExpectation],
                              results: Dict[str, pd.DataFrame]) -> Tuple[SavedExpectation, ...]:
    """
    Recompute each pinned decision's output-column hash on the run's result
    and return the expectations that were NOT reproduced (a pinned decision
    must reproduce the run it was pinned from).  An expectation whose
    result key or columns are absent from the results cannot be checked and is
    not reported.
    """
    failed = []
    for expectation in expectations:
        result_df = results.get(expectation.result_key)
        if result_df is None:
            continue
        recomputed = hash_result_columns(result_df, expectation.columns)
        if recomputed is None:
            continue
        if recomputed != expectation.sha256:
            failed.append(expectation)
    return tuple(failed)


__all__ = ["apply_simulation_settings", "build_engine", "configure_engine", "engine_configuration",
           "execute", "run_sub_run", "sample_agents", "verify_saved_expectations"]
