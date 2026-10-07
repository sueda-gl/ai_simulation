"""
Monte-Carlo uses exactly the run plan of a single run (app/seam/mc.py, Q-29 fixed).

Before 2026-10-07 the Monte-Carlo subprocess received only population / income /
anchor weight on its command line, so every Page-2 tab setting (Decision 4's
intercepts, anchor mix, sigma tick box, each decision's own income mode, ...) was
replaced by the file's values.  Now the app hands the subprocess the single-run
plan's sub-run as a plan file, and ``runs = 1, base_seed = S`` must reproduce a
single run with seed ``S`` agent by agent - checked here with NON-default Decision 4
settings (beta1 = 0.3, flexibility observed weight 0.4, sigma on, Decision 4 on its
own continuous income mode while the global mode is categorical).
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
from app.seam.config_repo import get_config_repo
from app.seam.execute import execute
from app.seam.mc import load_mc_plan_file, run_mc_repetition, select_mc_sub_run, write_mc_plan_file
from test_build_plan import base_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 314
N_AGENTS = 24

D4_SETTINGS = dict(
    rtd_intercept_loyalty=0.3,          # beta1
    rtd_flex_observed_weight=0.4,       # W_OFlex
    rtd_sigma_enabled=True,             # sigma on (Research Specification)
    rtd_scale_factor=1.3,
    rtd_income_mode="Continuous only",  # Decision 4's own mode; the global one is categorical
)


@pytest.fixture(scope="module")
def repo():
    return get_config_repo()


@pytest.fixture(scope="module")
def default_values():
    from app.pages.decision_execution import DEFAULT_DECISION_VALUES
    return DEFAULT_DECISION_VALUES


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
    from src.decisions.rejected_transaction_defaults import rename_consensus_columns
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
# The app's Monte-Carlo button path hands the subprocess the plan file
# ---------------------------------------------------------------------------
def _apptest_mc_script():
    import streamlit as st
    from app.models import initialize_session_state
    initialize_session_state()
    st.session_state.n_agents = 12
    st.session_state.n_runs = 1
    st.session_state.population_mode = 'Research Specification'
    st.session_state.rtd_intercept_loyalty = 0.3
    st.session_state.rtd_income_mode = 'Continuous only'
    st.session_state.decision_params.selected_decisions = ['rejected_transaction_defaults']
    st.session_state.custom_decisions = ['rejected_transaction_defaults']
    st.session_state.default_decisions = []
    from app.simulation import run_monte_carlo_study
    run_monte_carlo_study()


def test_app_monte_carlo_passes_the_single_run_plan(monkeypatch):
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    import app.simulation as sim

    captured = {}

    class _FakeProcess:
        returncode = 1
        stdout = io.StringIO("")

        def poll(self):
            return 1

        def communicate(self):
            return "", ""

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _FakeProcess()

    monkeypatch.setattr(sim, "subprocess", SimpleNamespace(Popen=_fake_popen, PIPE=subprocess.PIPE))
    at = AppTest.from_function(_apptest_mc_script)
    at.run(timeout=300)
    assert not at.exception
    cmd = captured["cmd"]
    plan_file = Path(cmd[cmd.index("--plan-file") + 1])
    try:
        sub_run, custom = load_mc_plan_file(plan_file)
    finally:
        plan_file.unlink(missing_ok=True)
    # the Decision-4-only run keeps Decision 4's own income mode, as a single run does
    assert cmd[cmd.index("--income-mode") + 1] == "continuous"
    assert cmd[cmd.index("--population-mode") + 1] == "documentation"
    assert sub_run.income_mode == "continuous" and sub_run.population == "documentation"
    assert sub_run.decisions_to_run == ("rejected_transaction_defaults",)
    patch = sub_run.decision_config_patches["rejected_transaction_defaults"]
    assert patch["intercepts"]["loyalty"] == pytest.approx(0.3)
    assert patch["income_mode"] == "continuous"
    assert custom == ("rejected_transaction_defaults",)


def test_mc_repetitions_resample_research_specification(single_run):
    """Owner clarification of R-CYC (2026-10-07): Research Specification samples the
    participants randomly from the run seed. Repetition i of a study (seed base_seed + i)
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
