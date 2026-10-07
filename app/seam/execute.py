# app/seam/execute.py
"""
Run a RunPlan against the engine.  No Streamlit anywhere in here.

Per sub-run (exactly what the former mode runners did, in the same order):
  1. sample the agents - copula: TraitEngine().sample(n, seed);
     research: load_original_participants(n, seed, random_sample) - Specification
     random (seeded), Baseline the 280 in file order, cycling (R-CYC, clarified)
  2. build Engine(PROFILES[population]) (a fresh yaml load, as each Orchestrator() was)
  3. replace its decisions dict with a deep copy of the loaded file + the plan's patches
  4. simulation_config['simulation'] <- simulation_params; random_decisions /
     default_decisions <- the SAME decision_settings object; purchasing_limits
     (only when the plan carries one); custom_decisions (the plan's
     metadata.custom_decisions - the former st.session_state.custom_decisions;
     no engine consumer, but it travels into df.attrs['simulation_config']);
     default_decisions_list
  5. engine.run_simulation(n_agents, seed, decisions_to_run, agents_df=agents_df)
then assign_global_transaction_ids on every result.
"""

from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from src.contract.plan import RunPlan, SavedExpectation, SubRun, apply_patches, hash_result_columns
from src.engine.core import Engine
from src.engine.postprocess import assign_global_transaction_ids
from src.engine.profile import PROFILES
from src.engine.sampling import load_original_participants
from src.trait_engine import TraitEngine
from app.seam.config_repo import DecisionsConfig, get_config_repo


def sample_agents(population: str, n_agents: int, seed: int) -> pd.DataFrame:
    """The agent source of each population mode (identical to the former runners)."""
    if population == "copula":
        return TraitEngine().sample(n_agents, seed)
    if population in ("documentation", "baseline"):
        # Ruling R-CYC (owner clarification 2026-10-07): Research Specification samples
        # randomly from default_rng(seed); Research Baseline cycles in file order
        return load_original_participants(n_agents, seed,
                                          random_sample=(population == "documentation"))
    raise ValueError(f"Unknown population {population!r}")


def configure_engine(engine: Engine, sub_run: SubRun, config_repo: DecisionsConfig,
                     custom_decisions: Optional[Sequence[str]] = None) -> Engine:
    """
    Apply one sub-run's patches and simulation settings to a fresh Engine.

    ``custom_decisions`` (the run's metadata.custom_decisions) is written to
    simulation_config['custom_decisions'] exactly as the legacy
    _apply_decision_settings wrote st.session_state.custom_decisions; None
    leaves the key alone.
    """
    engine.config = apply_patches(config_repo.fresh_decisions_dict(), sub_run.decision_config_patches)

    simulation_config = engine.simulation_config
    if "simulation" not in simulation_config or not isinstance(simulation_config.get("simulation"), dict):
        simulation_config["simulation"] = {}
    simulation_config["simulation"].update(sub_run.simulation_params)

    if sub_run.decision_settings:
        simulation_config["random_decisions"] = sub_run.decision_settings
        simulation_config["default_decisions"] = sub_run.decision_settings

    if sub_run.purchasing_limits is not None:
        simulation_config["purchasing_limits"] = sub_run.purchasing_limits

    # Pass information about custom vs default decisions (df.attrs parity with
    # the legacy runner: custom_decisions is written before default_decisions_list)
    if custom_decisions is not None:
        simulation_config["custom_decisions"] = list(custom_decisions)

    simulation_config["default_decisions_list"] = list(sub_run.default_decisions_list)
    return engine


def run_sub_run(sub_run: SubRun, config_repo: Optional[DecisionsConfig] = None,
                custom_decisions: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Sample agents, configure a fresh Engine and run one sub-run."""
    repo = config_repo if config_repo is not None else get_config_repo()
    agents_df = sample_agents(sub_run.population, sub_run.n_agents, sub_run.seed)
    engine = configure_engine(Engine(PROFILES[sub_run.population]), sub_run, repo,
                              custom_decisions=custom_decisions)
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
    R14: recompute each pinned decision's output-column hash on the run's result
    and return the expectations that were NOT reproduced.  An expectation whose
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


__all__ = ["configure_engine", "execute", "run_sub_run", "sample_agents", "verify_saved_expectations"]
