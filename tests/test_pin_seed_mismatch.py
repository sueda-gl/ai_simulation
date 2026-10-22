"""The results page says when a configuration could NOT be pinned.

``save_selected_configuration`` refuses a donation configuration whose run used a
different seed or agent count than an already pinned decision
(``validate_seed_consistency``). The screen used to discard that refusal and show
"✅ Configuration saved" anyway; now the function returns the failure and the page
shows one st.error with the reason, and nothing is pinned.

The mismatch is constructed the way the check reads it: after an individual
donation run (seed 42, 40 agents) a Disclose Income record pinned from a run with
another seed is put in ``selected_decision_configs``; the save then conflicts.
"""
from datetime import datetime

import pytest

from conftest import start_app
from app.pages.results.config_selection import seed_mismatch_message

pytestmark = pytest.mark.slow

N_AGENTS = 40
OTHER_SEED = 99


def _conflicting_disclose_income_config():
    return {
        'result_key': 'categorical',
        'params': {'intercept': 0.75, 'income_mode': 'Categorical only'},
        'income_mode': 'Categorical only',
        'population_mode': 'Copula (synthetic)',
        'metrics': {'y_rate': 0.5, 'n_rate': 0.5, 'y_count': 20, 'n_count': 20},
        'selected_timestamp': datetime(2025, 1, 15, 12, 0, 0),
        'total_agents': N_AGENTS,
        'source': 'individual_disclose_income_run',
        'original_seed': OTHER_SEED,
        'original_n_agents': N_AGENTS,
        'result_columns': ['disclose_income'],
        'result_sha256': 'deadbeef',
    }


def test_seed_mismatch_shows_error_and_pins_nothing():
    at = start_app()
    at.run()
    at.number_input(key="n_agents_input").set_value(N_AGENTS).run()
    next_buttons = [b for b in at.button if "Next: Decision Parameters" in b.label]
    next_buttons[0].click().run()
    at.checkbox(key="page2_select_all_checkbox").uncheck().run()
    at.multiselect(key="page2_manual_multiselect").set_value(["donation_default"]).run()
    at.button(key="run_donation_default_only_btn").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    meta = at.session_state["_run_metadata"]
    assert (meta["seed"], meta["n_agents"]) == (42, N_AGENTS)

    # another decision pinned from a run with a different seed
    at.session_state["selected_decision_configs"] = {
        "disclose_income": _conflicting_disclose_income_config()}
    at.run()
    assert not at.exception

    at.button(key="save_single_config").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]

    expected = seed_mismatch_message({
        'new_seed': 42, 'new_n_agents': N_AGENTS,
        'existing_seed': OTHER_SEED, 'existing_n_agents': N_AGENTS,
        'conflicting_decision': 'disclose_income'})
    assert expected == ("Configuration not saved: this run used seed 42 with 40 agents, but the "
                        "pinned Disclose Income configuration used seed 99 with 40 agents.")
    assert expected in [str(e.value) for e in at.error]
    assert not any("Configuration saved" in str(s.value) for s in at.success)
    assert "donation_default" not in at.session_state["selected_decision_configs"]


def test_matching_seed_still_saves_with_the_same_success_message():
    at = start_app()
    at.run()
    at.number_input(key="n_agents_input").set_value(N_AGENTS).run()
    next_buttons = [b for b in at.button if "Next: Decision Parameters" in b.label]
    next_buttons[0].click().run()
    at.checkbox(key="page2_select_all_checkbox").uncheck().run()
    at.multiselect(key="page2_manual_multiselect").set_value(["donation_default"]).run()
    at.button(key="run_donation_default_only_btn").click().run()
    assert not at.exception

    at.button(key="save_single_config").click().run()
    assert not at.exception
    pinned = at.session_state["selected_decision_configs"]["donation_default"]
    assert (pinned["original_seed"], pinned["original_n_agents"]) == (42, N_AGENTS)
    assert not any("Configuration not saved" in str(e.value) for e in at.error)
