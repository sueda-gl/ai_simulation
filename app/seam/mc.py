# app/seam/mc.py
"""
Monte-Carlo runs on the SAME plan as a single run.  No Streamlit.

A Monte-Carlo study is ``runs`` repetitions of one engine run, each with seed
``base_seed + i``.  Before 2026-10-07 the subprocess that performs them
(``scripts/run_mc_study.py`` -> ``scripts/run_simulation.py``) received only
the population mode, the global income mode and the donation anchor weight on
its command line, so every Page-2 tab setting (Decision 1/2/4 intercepts,
income modes, sigma tick boxes, the donation coefficient set, the default
decisions, the Page-1 parameters, ...) was silently replaced by the file's
values (rulings-and-quirks Q-29).

Now the app builds the ordinary ``RunPlan`` from the session (the same
``build_run_plan`` a single run uses), picks the one sub-run the study
repeats, and hands it to the subprocess as a plan file.  The subprocess
replays that sub-run through ``app.seam.execute.run_sub_run`` with only the
seed (and the agent count) replaced - so ``runs = 1, base_seed = S`` gives
exactly the per-agent results of a single run with seed ``S``.

The file is a pickle: the patches carry ``Replace`` markers and Python tuples
that JSON would not round-trip exactly.  It is written by the app into
``outputs/`` and read back only by the project's own scripts.
"""

import dataclasses
import pickle
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import pandas as pd

from src.contract.plan import RunPlan, SubRun
from src.engine.postprocess import assign_global_transaction_ids
from app.seam.config_repo import DecisionsConfig
from app.seam.execute import run_sub_run

PLAN_FILE_VERSION = 1


def select_mc_sub_run(plan: RunPlan, population: Optional[str] = None,
                      income_mode: Optional[str] = None) -> SubRun:
    """
    The sub-run a Monte-Carlo study repeats.

    A plan with several sub-runs (Compare all / Compare both, or a decision
    tab's own "Compare both") is reduced to one, as the Monte-Carlo screen has
    always done ("Comparison Mode Limitation"): the first sub-run of the
    requested population whose income mode matches the requested one, else the
    first sub-run of that population, else the plan's first sub-run.  A
    single-decision run whose tab chose its own income mode therefore keeps
    that mode (the single-run rule), whatever the global setting says.
    """
    if not plan.sub_runs:
        raise ValueError("the plan has no sub-runs")
    same_pop = [s for s in plan.sub_runs if population is None or s.population == population]
    for sub_run in same_pop:
        if income_mode is None or sub_run.income_mode == income_mode:
            return sub_run
    return same_pop[0] if same_pop else plan.sub_runs[0]


def write_mc_plan_file(path: Union[str, Path], sub_run: SubRun,
                       custom_decisions: Sequence[str] = ()) -> Path:
    """Store the sub-run (and the run's custom_decisions) for the subprocess."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": PLAN_FILE_VERSION, "sub_run": sub_run,
               "custom_decisions": tuple(custom_decisions)}
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_mc_plan_file(path: Union[str, Path]) -> Tuple[SubRun, Tuple[str, ...]]:
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or payload.get("version") != PLAN_FILE_VERSION:
        raise ValueError(f"{path} is not a Monte-Carlo plan file (version {PLAN_FILE_VERSION})")
    sub_run = payload["sub_run"]
    if not isinstance(sub_run, SubRun):
        raise ValueError(f"{path} does not contain a SubRun")
    return sub_run, tuple(payload.get("custom_decisions") or ())


def run_mc_repetition(sub_run: SubRun, seed: int, n_agents: Optional[int] = None,
                      custom_decisions: Optional[Sequence[str]] = None,
                      config_repo: Optional[DecisionsConfig] = None) -> pd.DataFrame:
    """One Monte-Carlo repetition: the plan's sub-run with the seed (and agent
    count) replaced, executed exactly like ``execute`` executes a sub-run."""
    changes = {"seed": int(seed)}
    if n_agents is not None:
        changes["n_agents"] = int(n_agents)
    repetition = dataclasses.replace(sub_run, **changes)
    df = run_sub_run(repetition, config_repo, custom_decisions=custom_decisions)
    return assign_global_transaction_ids(df)


__all__ = ["load_mc_plan_file", "run_mc_repetition", "select_mc_sub_run", "write_mc_plan_file"]
