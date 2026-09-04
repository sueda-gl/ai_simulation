"""
Decision 4 configuration selection - the same "Use This Config" flow as the other
modelled decisions (disclose_income is the reference):

- An individual Decision 4 run shows a "🎯 Use This Config" button UNDER its results
  (after the workbook download, like the other decisions);
  clicking it stores the run's tab settings (income mode, per-element intercepts,
  stochastic anchors, rank-aggregation settings, stochastic UI) plus metrics in the
  unified selected_decision_configs store and the button turns into "✅ Selected".
- Page 2 "Saved Decision Configurations" shows the saved Decision 4 configuration with
  a Clear button; the results page shows the selected configuration (Clear) and the
  "Run Complete Simulation" section, like disclose_income.
- Combined / complete simulations apply the saved MODEL settings (income mode,
  intercepts, anchors, aggregation) instead of the tab state; individual Decision 4
  runs keep reflecting the tab.
- The complete-simulation gate blocks a Decision 4 "Compare both" (or Compare-all
  population) selection until one configuration is selected, like disclose_income.
"""
import pytest


def _rtd_config_app_script():
    """Decision 4 tab + results (once run) + the Page-2 saved-config display, and the
    combined-run application probes stashed into session state for assertions."""
    import streamlit as st
    from app.models import initialize_session_state

    initialize_session_state()
    st.session_state.population_mode = 'Research Baseline'
    st.session_state.n_agents = 60

    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()

    if st.session_state.get('simulation_results'):
        from app.pages.results.main_results import render_single_run_results
        render_single_run_results()

    from app.pages.page2_decisions import render_selected_rejected_transaction_config_display
    st.markdown("PAGE2_SAVED_CONFIGS_MARKER")
    render_selected_rejected_transaction_config_display()

    # Probe: how a combined run (inc_mode=None) vs an individual run (explicit inc_mode)
    # would configure the orchestrator with the current saved config / tab state.
    if st.session_state.get('_probe_apply'):
        from src.orchestrator_baseline import OrchestratorBaseline
        from app.simulation import _apply_rejected_transaction_config
        probe = {}
        for label, inc_mode in (('combined', None), ('individual', 'continuous')):
            orch = OrchestratorBaseline()
            _apply_rejected_transaction_config(orch, "baseline", inc_mode)
            cfg = orch.config['rejected_transaction_defaults']
            probe[label] = {
                'income_mode': cfg.get('income_mode'),
                'intercepts': dict(cfg.get('intercepts', {})),
                'aggregation': dict(cfg.get('aggregation', {})),
                'anchor_loyalty': cfg['stochastic']['mechanisms']['loyalty'].get('anchor'),
            }
        st.session_state['_probe_result'] = probe

    if st.session_state.get('_probe_gate'):
        # a Decision 4 selection on Page 2 (the gate reads decision_params.selected_decisions)
        st.session_state.decision_params.selected_decisions = ['rejected_transaction_defaults']
        from app.pages.decision_execution import can_run_complete_simulation
        can_run, reason, count, block_type, issues = can_run_complete_simulation()
        st.session_state['_gate_result'] = {
            'can_run': can_run, 'reason': reason, 'count': count,
            'block_type': block_type, 'issues': [i['decision'] for i in issues]}


def _all_markdown(at):
    return "\n".join(str(m.value) for m in at.markdown)


def _success_texts(at):
    return [str(e.value) for e in at.success]


def _saved_configs(at):
    try:
        return dict(at.session_state['selected_decision_configs'])
    except KeyError:
        return {}


def test_apptest_select_display_apply_and_clear():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_config_app_script)
    at.run(timeout=600)
    assert not at.exception
    assert 'rejected_transaction_defaults' not in _saved_configs(at)

    # individual Decision 4 run -> "Use This Config" offered under the results
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run(timeout=600)
    assert not at.exception
    assert at.session_state['custom_decisions'] == ['rejected_transaction_defaults']
    results = at.session_state['simulation_results']
    result_key = next(iter(results.keys()))
    assert result_key == 'continuous'            # tab default income mode
    select_btn = at.button(key=f'rtd_inline_select_{result_key}')
    assert select_btn is not None
    assert "4. Rejected Transaction Defaults" not in _all_markdown(at).split("PAGE2_SAVED_CONFIGS_MARKER")[-1]
    # placement: the button sits UNDER the decision's results (after the whole-decision
    # workbook download and its preview), like the other decisions' buttons
    captions = [str(c.value) for c in at.caption]
    idx_quick_summary = next(i for i, c in enumerate(captions) if c.startswith("📊 Quick Summary"))
    idx_last_preview = max(i for i, c in enumerate(captions) if c.startswith("Rows:"))
    assert idx_quick_summary > idx_last_preview
    # no "Run Complete Simulation" offer before a configuration is selected
    assert "🚀 Run Complete Simulation" not in _all_markdown(at).split("PAGE2_SAVED_CONFIGS_MARKER")[0]

    # select it
    select_btn.click().run(timeout=600)
    assert not at.exception
    configs = at.session_state['selected_decision_configs']
    assert 'rejected_transaction_defaults' in configs
    cfg = configs['rejected_transaction_defaults']
    assert cfg['result_key'] == result_key
    assert cfg['income_mode'] == 'Continuous only'
    assert cfg['population_mode'] == 'Research Baseline'
    assert cfg['source'] == 'individual_rejected_transaction_defaults_run'
    assert cfg['total_agents'] == 60
    assert set(cfg['params']['intercepts']) == {'ttp', 'loyalty', 'wtp', 'risk_taking', 'flexibility'}
    assert cfg['params']['intercepts']['ttp'] == pytest.approx(0.05)     # research default
    assert cfg['params']['aggregation'] == {'enabled': True}
    assert set(cfg['params']['anchors']) == {'loyalty', 'risk_taking', 'flexibility'}
    assert cfg['params']['stochastic']['sigma_strategy'] == 'overall'
    assert 'mean_choice_length' in cfg['metrics'] and 'first_option_shares' in cfg['metrics']
    assert abs(cfg['metrics']['mean_choice_length'] - results[result_key]['rtd_choice_length'].mean()) < 1e-9
    # button replaced by the selected marker; Page 2 display present
    assert f'rtd_inline_select_{result_key}' not in [b.key for b in at.button]   # replaced by the marker
    assert "✅ Selected" in _success_texts(at)
    page2 = _all_markdown(at).split("PAGE2_SAVED_CONFIGS_MARKER")[-1]
    assert "4. Rejected Transaction Defaults" in page2
    assert any("Rejected Transaction Defaults Configuration" in t and "Research Baseline + Continuous only" in t
               for t in _success_texts(at))
    # like disclose_income: the results page now shows the selected configuration
    # (with Clear) and offers "Run Complete Simulation" right there
    results_md = _all_markdown(at).split("PAGE2_SAVED_CONFIGS_MARKER")[0]
    assert "🚀 Run Complete Simulation" in results_md
    assert at.button(key='run_complete_from_results') is not None
    assert at.button(key='clear_rtd_selection') is not None
    assert any("Selected Rejected Transaction Defaults Configuration" in t for t in _success_texts(at))

    # combined-run application: change the tab afterwards (through its widgets) ->
    # combined runs still use the saved model settings, individual runs follow the tab
    at.number_input(key='rtd_tab_intercept_ttp').set_value(0.9)
    at.checkbox(key='rtd_tab_aggregation_enabled').uncheck()
    at.radio(key='rtd_tab_income_mode').set_value('Categorical only')
    at.session_state['_probe_apply'] = True
    at.run(timeout=600)
    assert at.session_state['rtd_intercept_ttp'] == pytest.approx(0.9)
    assert at.session_state['rtd_aggregation_enabled'] is False
    assert not at.exception
    probe = at.session_state['_probe_result']
    assert probe['combined']['income_mode'] == 'continuous'
    assert probe['combined']['intercepts']['ttp'] == pytest.approx(0.05)
    assert probe['combined']['aggregation']['enabled'] is True
    assert probe['individual']['income_mode'] == 'continuous'      # explicit inc_mode of the sub-run
    assert probe['individual']['intercepts']['ttp'] == pytest.approx(0.9)
    assert probe['individual']['aggregation']['enabled'] is False

    # clear (the button sits below the display it removes; a follow-up run shows the
    # post-clear page like the st.rerun() does in the app)
    at.button(key='clear_rejected_transaction_config').click().run(timeout=600)
    assert not at.exception
    at.run(timeout=600)
    assert 'rejected_transaction_defaults' not in at.session_state['selected_decision_configs']
    assert "4. Rejected Transaction Defaults" not in _all_markdown(at).split("PAGE2_SAVED_CONFIGS_MARKER")[-1]
    # the results-page complete-simulation offer disappears with the selection
    button_keys = [b.key for b in at.button]
    assert 'run_complete_from_results' not in button_keys
    assert 'clear_rtd_selection' not in button_keys


def test_apptest_complete_simulation_gate_requires_a_selected_config():
    """Decision 4 selected with 'Compare both' -> 2 configurations -> the complete
    simulation is blocked until one is selected; a saved config lifts the block and
    is named in the success reason."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_rtd_config_app_script)
    at.run(timeout=600)
    at.radio(key='rtd_tab_income_mode').set_value('Compare both')
    at.session_state['_probe_gate'] = True
    at.run(timeout=600)
    assert at.session_state['rtd_income_mode'] == 'Compare both'
    assert not at.exception
    gate = at.session_state['_gate_result']
    assert gate['can_run'] is False
    assert gate['block_type'] == 'rejected_transaction_defaults'
    assert gate['issues'] == ['rejected_transaction_defaults']
    assert gate['count'] == 2

    # a selected configuration lifts the block
    at.session_state['selected_decision_configs'] = {'rejected_transaction_defaults': {
        'result_key': 'continuous', 'params': {'income_mode': 'Compare both'},
        'income_mode': 'Continuous only', 'population_mode': 'Research Baseline',
        'metrics': {}, 'total_agents': 60, 'source': 'individual_rejected_transaction_defaults_run',
        'original_seed': 42, 'original_n_agents': 60}}
    at.run(timeout=600)
    assert not at.exception
    gate = at.session_state['_gate_result']
    assert gate['can_run'] is True
    assert "Rejected Transaction Defaults: Research Baseline + Continuous only" in gate['reason']

    # single income mode never blocks
    at.session_state['selected_decision_configs'] = {}
    at.radio(key='rtd_tab_income_mode').set_value('Continuous only')
    at.run(timeout=600)
    assert at.session_state['_gate_result']['can_run'] is True
