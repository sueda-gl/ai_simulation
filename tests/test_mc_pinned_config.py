"""
Monte Carlo honours a selected ("Use This Config") saved configuration exactly as a
single / complete run does.

A complete run with a pinned configuration runs with the pinned seed, agent count
and population mode (``build_plan.resolve_seed_and_n``), whatever Page 1 says.
Monte Carlo used to replay the plan's decision settings but take the agent count
from Page 1 - so a configuration pinned from a 280-agent Research Baseline run was
repeated with Page 1's 1000 agents.  Now the study repeats the plan's sub-run in
process with its own agent count and population; only the seed varies
(``base_seed + i``), and the screen says so:

    🔑 Using saved config: agents 280, population Research Baseline; seeds vary per run

Acceptance: pin a Decision 4 configuration from a 280-agent Research Baseline run,
set Page 1 to 1000 agents / Copula, run the complete simulation, then the
Monte-Carlo study with base seed = the pinned seed - its first repetition equals the
complete run agent by agent.  Without a pinned configuration Monte Carlo uses Page 1.
"""
import pandas as pd
import pytest

from app.seam.mc import pinned_config_caption, population_mode_name
from src.contract.plan import SubRun

from test_mc_plan_parity import _assert_same_rows
from ui_strings import (MC_PINNED_CAPTION_1500_SPEC, MC_PINNED_CAPTION_280_BASELINE,
                        SAVED_CONFIG_CAPTION_PREFIX)

pytestmark = pytest.mark.slow

PINNED_SEED = 2718
PINNED_N = 280
CAPTION = MC_PINNED_CAPTION_280_BASELINE


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


@pytest.fixture
def captured_repetitions(monkeypatch):
    """Records every repetition the app's study runs - (sub_run, seed, n_agents,
    custom_decisions) - and the frame it returned.  With ``real`` set the repetition
    runs for real (``app.seam.mc.run_mc_repetition``); otherwise a one-agent stand-in
    frame is returned, so the app stops after choosing what to run."""
    import app.simulation as sim
    from app.seam.mc import run_mc_repetition

    captured = {"calls": [], "frames": [], "real": False}

    def _repetition(sub_run, seed, n_agents=None, custom_decisions=None, config_repo=None):
        captured["calls"].append((sub_run, seed, n_agents, tuple(custom_decisions or ())))
        if captured["real"]:
            df = run_mc_repetition(sub_run, seed, n_agents, custom_decisions=custom_decisions,
                                   config_repo=config_repo)
        else:
            df = pd.DataFrame({"agent_id": [1], "donation_default": [0.2]})
        captured["frames"].append(df)
        return df

    monkeypatch.setattr(sim, "run_mc_repetition", _repetition)
    return captured


def _captions(at):
    return [str(c.value) for c in at.caption]


def test_pinned_config_mc_first_repetition_is_the_pinned_complete_run(captured_repetitions):
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

    # the Monte-Carlo study from the same screen, its one repetition run for real
    captured_repetitions["real"] = True
    at.session_state['_probe'] = 'mc'
    at.run(timeout=900)
    assert not at.exception
    (sub_run, seed, n_agents, custom), = captured_repetitions["calls"]
    assert seed == PINNED_SEED and n_agents is None            # the sub-run's own count
    assert sub_run.n_agents == PINNED_N and sub_run.population == "baseline"
    assert sub_run.decision_config_patches["rejected_transaction_defaults"][
        "intercepts"]["loyalty"] == pytest.approx(0.3)
    assert custom == ("rejected_transaction_defaults",)
    assert CAPTION in _captions(at)

    # its first repetition (runs = 1) is the complete run, agent by agent
    repetition, = captured_repetitions["frames"]
    _assert_same_rows(complete, repetition)

    # and the study's frames say so: one run of the pinned agent count with the pinned seed
    mc = at.session_state['mc_results']
    assert mc['summary']['runs'].tolist() == [1]
    assert mc['summary']['agents_per_run'].tolist() == [PINNED_N]
    assert mc['detailed']['seed'].tolist() == [PINNED_SEED]
    assert mc['detailed']['n_agents'].tolist() == [PINNED_N]


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


def test_without_a_pinned_config_mc_uses_page_1(captured_repetitions):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_unpinned_mc_script)
    at.run(timeout=300)
    assert not at.exception
    (sub_run, seed, n_agents, _), = captured_repetitions["calls"]
    assert seed == at.session_state['base_seed'] and n_agents is None
    assert sub_run.n_agents == 1000 and sub_run.population == "copula"
    assert not any(c.startswith(SAVED_CONFIG_CAPTION_PREFIX) for c in _captions(at))


def test_pinned_config_caption_wording():
    sub_run = SubRun(result_key="categorical", population="documentation", income_mode="categorical",
                     seed=1, n_agents=1500, decisions_to_run=None, decision_config_patches={},
                     simulation_params={}, decision_settings={}, default_decisions_list=(),
                     purchasing_limits={})
    assert pinned_config_caption(sub_run) == (
        MC_PINNED_CAPTION_1500_SPEC)
    assert population_mode_name("baseline") == "Research Baseline"
    assert population_mode_name("copula") == "Copula (synthetic)"
