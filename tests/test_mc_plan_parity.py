"""
Monte-Carlo uses exactly the run plan of a single run (app/seam/mc.py).

Before 2026-10-07 the Monte-Carlo subprocess received only population / income /
anchor weight on its command line, so every Page-2 tab setting (Decision 4's
intercepts, anchor mix, sigma tick box, each decision's own income mode, ...) was
replaced by the file's values.  Now the app repeats the single-run plan's sub-run
in process (``run_mc_repetition``; the CLI replays the same sub-run from a plan
file), and ``runs = 1, base_seed = S`` must reproduce a single run with seed ``S``
agent by agent - checked here with NON-default Decision 4 settings (beta1 = 0.3,
flexibility observed weight 0.4, sigma on, Decision 4 on its own continuous income
mode while the global mode is categorical).
"""
import contextlib
import io
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from app.seam import build_plan as bp
from app.seam.execute import execute
from app.seam.mc import load_mc_plan_file, run_mc_repetition, select_mc_sub_run, write_mc_plan_file
from test_build_plan import base_state

from conftest import ROOT as PROJECT_ROOT   # repo / default_values: conftest fixtures

pytestmark = pytest.mark.slow

SEED = 314
N_AGENTS = 24

D4_SETTINGS = dict(
    rtd_intercept_loyalty=0.3,          # beta1
    rtd_flex_observed_weight=0.4,       # W_OFlex
    rtd_sigma_enabled=True,             # sigma on (Research Specification)
    rtd_scale_factor=1.3,
    rtd_income_mode="Continuous only",  # Decision 4's own mode; the global one is categorical
)


def _state(repo, **overrides):
    return base_state(repo, population_mode="Research Specification",
                      income_spec_mode="categorical only",
                      n_agents=N_AGENTS, seed_input=SEED, seed=SEED, **overrides)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _assert_same_rows(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    for col in a.columns:
        # pickle bytes: exact equality of every cell, nested dicts / lists included
        assert pickle.dumps(a[col].tolist()) == pickle.dumps(b[col].tolist()), col


@pytest.fixture(scope="module")
def single_run(repo, default_values):
    plan = bp.build_run_plan(_state(repo, **D4_SETTINGS), repo, default_decision_values=default_values)
    assert len(plan.sub_runs) == 1 and plan.sub_runs[0].seed == SEED
    results = _quiet(execute, plan, repo)
    return plan, results[plan.sub_runs[0].result_key]


def test_plan_carries_the_decision_4_tab_settings(single_run):
    plan, df = single_run
    patch = plan.sub_runs[0].decision_config_patches["rejected_transaction_defaults"]
    assert patch["income_mode"] == "continuous"            # its own mode, not the global one
    assert patch["intercepts"]["loyalty"] == pytest.approx(0.3)
    assert patch["flexibility_anchor"]["observed_weight"] == pytest.approx(0.4)
    assert (df["rtd_sigma_used_ttp"] > 0).any()             # sigma reached the model


def test_mc_repetition_in_process_is_the_single_run(single_run, tmp_path):
    plan, df = single_run
    sub = select_mc_sub_run(plan, "documentation", "categorical")
    path = write_mc_plan_file(tmp_path / "plan.pkl", sub, plan.metadata.custom_decisions)
    loaded, custom = load_mc_plan_file(path)
    assert loaded == sub and custom == plan.metadata.custom_decisions
    rep = _quiet(run_mc_repetition, loaded, SEED, N_AGENTS, custom_decisions=custom)
    _assert_same_rows(df, rep)


def test_mc_settings_are_not_the_defaults(repo, default_values, single_run):
    """Negative control: the same seed with the tab's DEFAULT Decision 4 settings gives
    different Decision 4 results - so the parity above is not vacuous."""
    _, df = single_run
    plan = bp.build_run_plan(_state(repo), repo, default_decision_values=default_values)
    rep = _quiet(run_mc_repetition, select_mc_sub_run(plan), SEED, N_AGENTS,
                 custom_decisions=plan.metadata.custom_decisions)
    assert not df["rtd_flex_score"].equals(rep["rtd_flex_score"])


def _parquet_roundtrip(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """The exact transformation scripts/run_simulation.py applies before writing."""
    from src.contract.columns import rename_consensus_columns
    out = rename_consensus_columns(df.copy())
    out["purchase_requests"] = out["purchase_requests"].apply(
        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x))
    out.to_parquet(path, index=False)
    return pd.read_parquet(path)


def test_mc_study_runs_1_equals_single_run_end_to_end(single_run, tmp_path):
    """scripts/run_mc_study.py --runs 1 --base-seed S --plan-file F: the per-agent
    output of its one repetition equals the single run with seed S."""
    plan, df = single_run
    plan_file = write_mc_plan_file(tmp_path / "plan.pkl",
                                   select_mc_sub_run(plan, "documentation", "categorical"),
                                   plan.metadata.custom_decisions)
    out_dir = tmp_path / "out"
    cmd = [sys.executable, "scripts/run_mc_study.py", "--agents", str(N_AGENTS), "--runs", "1",
           "--base-seed", str(SEED), "--plan-file", str(plan_file), "--output-dir", str(out_dir),
           "--keep-individual"]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    files = sorted(out_dir.glob(f"simulation_seed{SEED}_agents{N_AGENTS}_*.parquet"))
    assert len(files) == 1, proc.stdout[-2000:]
    mc = pd.read_parquet(files[0])
    expected = _parquet_roundtrip(df, tmp_path / "single.parquet")
    pd.testing.assert_frame_equal(mc, expected)


# ---------------------------------------------------------------------------
# The app's Monte-Carlo button path repeats the single-run plan's sub-run in process
# ---------------------------------------------------------------------------
MC_BASE_SEED = 7
MC_RUNS = 2


def _apptest_mc_script():
    import streamlit as st
    from app.models import initialize_session_state
    initialize_session_state()
    st.session_state.n_agents = 12
    st.session_state.n_runs = 2          # MC_RUNS (the script body cannot see module globals)
    st.session_state.base_seed = 7       # MC_BASE_SEED
    st.session_state.population_mode = 'Research Specification'
    st.session_state.rtd_intercept_loyalty = 0.3
    st.session_state.rtd_income_mode = 'Continuous only'
    st.session_state.decision_params.selected_decisions = ['rejected_transaction_defaults']
    st.session_state.custom_decisions = ['rejected_transaction_defaults']
    st.session_state.default_decisions = []
    from app.simulation import run_monte_carlo_study
    st.session_state._mc_probe = run_monte_carlo_study()


def test_app_monte_carlo_repeats_the_single_run_plan_in_process(monkeypatch):
    """The app's study calls ``run_mc_repetition`` once per repetition with the
    single-run plan's sub-run (the Decision 4 tab's own settings included), seeds
    base_seed + i, the sub-run's own agent count and the plan's custom list; it
    writes no plan file, and its frames come from app/reports/mc.py."""
    from streamlit.testing.v1 import AppTest
    import app.simulation as sim

    calls = []

    def _fake_repetition(sub_run, seed, n_agents=None, custom_decisions=None, config_repo=None):
        calls.append((sub_run, seed, n_agents, tuple(custom_decisions or ())))
        return pd.DataFrame({"agent_id": [1, 2], "donation_default": [0.1, 0.3]})

    monkeypatch.setattr(sim, "run_mc_repetition", _fake_repetition)
    outputs = PROJECT_ROOT / "outputs"
    files_before = set(outputs.glob("*")) if outputs.exists() else set()
    at = AppTest.from_function(_apptest_mc_script)
    at.run(timeout=300)
    assert not at.exception
    files_after = set(outputs.glob("*")) if outputs.exists() else set()
    assert files_after == files_before                       # nothing written to outputs/

    # one repetition per run, seeds base_seed + i, the sub-run's own agent count
    assert [seed for _, seed, _, _ in calls] == [MC_BASE_SEED + i for i in range(MC_RUNS)]
    assert all(n_agents is None for _, _, n_agents, _ in calls)
    sub_run = calls[0][0]
    assert all(s == sub_run for s, _, _, _ in calls)
    # the Decision-4-only run keeps Decision 4's own income mode, as a single run does
    assert sub_run.income_mode == "continuous" and sub_run.population == "documentation"
    assert sub_run.n_agents == 12
    assert sub_run.decisions_to_run == ("rejected_transaction_defaults",)
    patch = sub_run.decision_config_patches["rejected_transaction_defaults"]
    assert patch["intercepts"]["loyalty"] == pytest.approx(0.3)
    assert patch["income_mode"] == "continuous"
    assert all(custom == ("rejected_transaction_defaults",) for _, _, _, custom in calls)

    # the study's frames: the per-run statistics and the summary of app/reports/mc.py
    summary, detailed, log = at.session_state["_mc_probe"]
    assert detailed["run"].tolist() == list(range(MC_RUNS))
    assert detailed["seed"].tolist() == [MC_BASE_SEED + i for i in range(MC_RUNS)]
    assert detailed["n_agents"].tolist() == [2, 2]
    assert detailed["donation_default_mean"].tolist() == pytest.approx([0.2, 0.2])
    assert summary["decision"].tolist() == ["donation_default"]
    assert summary["runs"].tolist() == [MC_RUNS] and summary["agents_per_run"].tolist() == [12]
    assert f"Run   1/{MC_RUNS}: seed={MC_BASE_SEED} ... donation_mean=0.2000" in log
    assert "✅ Monte-Carlo study completed!" in [str(s.value) for s in at.success]


def test_mc_repetitions_resample_research_specification(single_run):
    """Research Specification samples the participants randomly from the run seed.
    Repetition i of a study (seed base_seed + i)
    therefore draws its own random subset - the one a single run with that seed draws -
    while repetition base_seed reproduces the single run's agents (test above)."""
    from app.seam.execute import sample_agents
    plan, df = single_run
    sub = select_mc_sub_run(plan, "documentation", "categorical")
    rep = _quiet(run_mc_repetition, sub, SEED + 1, N_AGENTS,
                 custom_decisions=plan.metadata.custom_decisions)
    own = sample_agents("documentation", N_AGENTS, SEED + 1)["ExtraversionBig5"].tolist()
    assert rep["ExtraversionBig5"].tolist() == own
    assert own != sample_agents("documentation", N_AGENTS, SEED)["ExtraversionBig5"].tolist()
    assert df["ExtraversionBig5"].tolist() == \
        sample_agents("documentation", N_AGENTS, SEED)["ExtraversionBig5"].tolist()
