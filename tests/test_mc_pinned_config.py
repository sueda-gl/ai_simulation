"""
Monte Carlo honours a selected ("Use This Config") saved configuration exactly as a
single / complete run does (owner ruling, 2026-10-07).

A complete run with a pinned configuration runs with the pinned seed, agent count
and population mode (``build_plan.resolve_seed_and_n``, R14), whatever Page 1 says.
Monte Carlo used to replay the plan's decision settings but take the agent count
from Page 1 (``--agents``) - so a configuration pinned from a 280-agent Research
Baseline run was repeated with Page 1's 1000 agents.  Now the study runs the plan's
sub-run with its own agent count and population; only the seed varies
(``base_seed + i``), and the screen says so:

    🔑 Using saved config: agents 280, population Research Baseline; seeds vary per run

Acceptance: pin a Decision 4 configuration from a 280-agent Research Baseline run,
set Page 1 to 1000 agents / Copula, run the complete simulation, then the
Monte-Carlo study with base seed = the pinned seed - its first repetition equals the
complete run agent by agent.  Without a pinned configuration Monte Carlo uses Page 1.
"""
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.seam.mc import load_mc_plan_file, pinned_config_caption, population_mode_name
from src.contract.plan import SubRun

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_SEED = 2718
PINNED_N = 280
CAPTION = ("🔑 Using saved config: agents 280, population Research Baseline; "
           "seeds vary per run")


def _pinned_mc_script():
    """The Decision 4 tab (+ its results once run).  Page-1 values are seeded once;
    probes switch Page 1 and launch the complete run / the Monte-Carlo study through
    the real ``run_combined_simulation`` path."""
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()
    if not st.session_state.get('_page1_seeded'):
        st.session_state._page1_seeded = True
        st.session_state.population_mode = 'Research Baseline'
        st.session_state.n_agents = 280
        st.session_state.seed_input = 2718
        st.session_state.seed = 2718

    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()

    if st.session_state.get('simulation_results') and not st.session_state.get('_probe'):
        from app.pages.results.main_results import render_single_run_results
        render_single_run_results()

    probe = st.session_state.pop('_probe', None)
    if probe is not None:
        # Page 1 now says 1000 agents, Copula - the pinned configuration must win
        st.session_state.population_mode = 'Copula (synthetic)'
        st.session_state.n_agents = 1000
        st.session_state.base_seed = 2718
        st.session_state.base_seed_input = 2718
        st.session_state.n_runs = 1
        st.session_state.sim_params.simulation_mode = (
            "Monte-Carlo Study" if probe == 'mc' else "Single Run")
        from app.pages.decision_execution import run_combined_simulation
        run_combined_simulation(['rejected_transaction_defaults'])


class _FakeProcess:
    """Popen stand-in: records nothing, finishes at once with an error code, so the
    app stops after building the command line (no results, no rerun)."""
    returncode = 1
    stdout = io.StringIO("")

    def poll(self):
        return 1

    def communicate(self):
        return "", ""


@pytest.fixture
def captured_popen(monkeypatch):
    import app.simulation as sim
    captured = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _FakeProcess()

    monkeypatch.setattr(sim, "subprocess", SimpleNamespace(Popen=_fake_popen, PIPE=subprocess.PIPE))
    return captured


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _captions(at):
    return [str(c.value) for c in at.caption]


def _parquet_roundtrip(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """The transformation scripts/run_simulation.py applies before writing parquet."""
    import json
    from src.decisions.rejected_transaction_defaults import rename_consensus_columns
    out = rename_consensus_columns(df.copy())
    out["purchase_requests"] = out["purchase_requests"].apply(
        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x))
    out.to_parquet(path, index=False)
    return pd.read_parquet(path)


def test_pinned_config_mc_first_repetition_is_the_pinned_complete_run(captured_popen, tmp_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_pinned_mc_script)
    at.run(timeout=600)
    assert not at.exception
    # a non-default Decision 4 setting, so the saved decision settings matter too
    at.number_input(key='rtd_tab_intercept_loyalty').set_value(0.3).run(timeout=600)
    assert at.session_state['rtd_intercept_loyalty'] == pytest.approx(0.3)

    # individual Decision 4 run at 280 agents, Research Baseline, seed 2718 -> pin it
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    result_key = next(iter(at.session_state['simulation_results']))
    at.button(key=f'rtd_inline_select_{result_key}').click().run(timeout=600)
    assert not at.exception
    cfg = at.session_state['selected_decision_configs']['rejected_transaction_defaults']
    assert (cfg['original_seed'], cfg['original_n_agents'], cfg['population_mode']) == (
        PINNED_SEED, PINNED_N, 'Research Baseline')

    # the tab moves on afterwards; the pinned configuration keeps 0.3
    at.number_input(key='rtd_tab_intercept_loyalty').set_value(0.0).run(timeout=600)

    # complete run with Page 1 at 1000 agents / Copula -> pinned seed, 280, Baseline
    at.session_state['_probe'] = 'single'
    at.run(timeout=900)
    assert not at.exception
    meta = at.session_state['_run_metadata']
    assert (meta['seed'], meta['n_agents'], meta['effective_population_mode']) == (
        PINNED_SEED, PINNED_N, 'Research Baseline')
    complete_key = meta['result_keys'][0]
    complete = at.session_state['simulation_results'][complete_key]
    assert len(complete) == PINNED_N

    # the Monte-Carlo study from the same screen
    at.session_state['_probe'] = 'mc'
    at.run(timeout=900)
    assert not at.exception
    cmd = captured_popen["cmd"]
    plan_file = Path(_arg(cmd, "--plan-file"))
    try:
        sub_run, custom = load_mc_plan_file(plan_file)
        assert _arg(cmd, "--agents") == str(PINNED_N)
        assert _arg(cmd, "--population-mode") == "baseline"
        assert _arg(cmd, "--base-seed") == str(PINNED_SEED)
        assert sub_run.n_agents == PINNED_N and sub_run.population == "baseline"
        assert sub_run.decision_config_patches["rejected_transaction_defaults"][
            "intercepts"]["loyalty"] == pytest.approx(0.3)
        assert CAPTION in _captions(at)

        # its first repetition (runs = 1, the app's own command line) is the complete run
        out_dir = tmp_path / "out"
        real_cmd = cmd + ["--output-dir", str(out_dir), "--keep-individual"]
        proc = subprocess.run(real_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    finally:
        plan_file.unlink(missing_ok=True)
    files = sorted(out_dir.glob(f"simulation_seed{PINNED_SEED}_agents{PINNED_N}_*.parquet"))
    assert len(files) == 1, proc.stdout[-2000:]
    mc = pd.read_parquet(files[0])
    expected = _parquet_roundtrip(complete, tmp_path / "complete.parquet")
    pd.testing.assert_frame_equal(mc, expected)


def _unpinned_mc_script():
    import streamlit as st
    from app.models import initialize_session_state
    from app.models import ALL_DECISIONS

    initialize_session_state()
    st.session_state.population_mode = 'Copula (synthetic)'
    st.session_state.n_agents = 1000
    st.session_state.n_runs = 1
    st.session_state.selected_decision_configs = {}
    st.session_state.sim_params.simulation_mode = "Monte-Carlo Study"
    st.session_state.decision_params.selected_decisions = list(ALL_DECISIONS)
    st.session_state.custom_decisions = list(ALL_DECISIONS)
    st.session_state.default_decisions = []
    from app.simulation import run_monte_carlo_study
    run_monte_carlo_study()


def test_without_a_pinned_config_mc_uses_page_1(captured_popen):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_unpinned_mc_script)
    at.run(timeout=300)
    assert not at.exception
    cmd = captured_popen["cmd"]
    plan_file = Path(_arg(cmd, "--plan-file"))
    try:
        sub_run, _ = load_mc_plan_file(plan_file)
    finally:
        plan_file.unlink(missing_ok=True)
    assert _arg(cmd, "--agents") == "1000"
    assert _arg(cmd, "--population-mode") == "copula"
    assert sub_run.n_agents == 1000 and sub_run.population == "copula"
    assert not any(c.startswith("🔑 Using saved config") for c in _captions(at))


def test_pinned_config_caption_wording():
    sub_run = SubRun(result_key="categorical", population="documentation", income_mode="categorical",
                     seed=1, n_agents=1500, decisions_to_run=None, decision_config_patches={},
                     simulation_params={}, decision_settings={}, default_decisions_list=(),
                     purchasing_limits={})
    assert pinned_config_caption(sub_run) == (
        "🔑 Using saved config: agents 1,500, population Research Specification; seeds vary per run")
    assert population_mode_name("baseline") == "Research Baseline"
    assert population_mode_name("copula") == "Copula (synthetic)"
