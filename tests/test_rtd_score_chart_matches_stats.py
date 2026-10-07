"""
Lavie 2026-10: "For the Options List Length results, the graph for Tendency to Plan
score for Research Spec is identical to that for Research Baseline even though the
statistics are different. This problem occurs for all the other decision elements as
well."

Root cause: every score chart plotted the DETERMINISTIC score (weighted_ttp, the
standardized loyalty / WTP / RT / anchored-flexibility scores), which is the same for
Research Specification and Research Baseline (same 280 participants), while the
allocation charts, their tables and the overview statistics count the outcome of the
Normal(anchor, sigma) DRAW (document: draw_k ~ Normal(anchor, sigma), re-rescaled over
the population and re-binned).

Now, whenever an element's scores were drawn, its chart plots the drawn scores, so
that the plotted values, re-cut into equal-width bins over their own min..max exactly
as the engine does, reproduce the very lengths / segments the statistics count -
agent by agent. Research Baseline (no draw) is unchanged: the deterministic score,
the Stata figure.
"""
import numpy as np
import pandas as pd
import pytest

import app.pages.results.visualizations.transaction_viz as viz
from app.reports.rtd import RTD_SCORE_CHARTS, rtd_score_chart_data
from src.decisions.rejected_transaction_defaults import _RESCALE

SEGMENT_COLS = {'loyalty': 'rtd_loyalty_segment', 'wtp': 'rtd_wtp_segment',
                'risk_taking': 'rtd_rt_segment', 'flexibility': 'rtd_flex_segment'}


def _app_script():
    import streamlit as st
    from app.models import initialize_session_state
    initialize_session_state()
    st.session_state.population_mode = st.session_state.get('_test_population',
                                                            'Research Specification')
    st.session_state.n_agents = 280
    from app.pages.decision_tabs.rejected_transaction import (
        render_rejected_transaction_defaults_tab)
    render_rejected_transaction_defaults_tab()


def _run(population):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_app_script, default_timeout=600)
    at.session_state['_test_population'] = population
    at.run()
    assert not at.exception
    at.button(key='run_rejected_transaction_defaults_only_btn').click().run()
    assert not at.exception
    results = at.session_state['simulation_results']
    return {k: v for k, v in results.items()}


@pytest.fixture(scope="module")
def spec_frame():
    return next(iter(_run('Research Specification').values()))


@pytest.fixture(scope="module")
def baseline_frame():
    return next(iter(_run('Research Baseline').values()))


def _rebin(values, mech):
    """The engine's re-binning of drawn values: min-max over the population, then
    floor(low + (span - 0.0001) * u), clipped to the element's range."""
    x = np.asarray(values, dtype=float)
    low, span = _RESCALE[mech]
    u = (x - x.min()) / (x.max() - x.min())
    out = np.floor(low + (span - 0.0001) * u).astype(int)
    return np.clip(out, int(low), int(low) + int(span) - 1)


def _plotted(df, mech, monkeypatch):
    """The series the results page hands to the histogram for this element."""
    captured = {}
    monkeypatch.setattr(viz, '_rtd_density_hist',
                        lambda series, title, x_title, key: captured.update(
                            series=pd.Series(series), title=title, x_title=x_title))

    def both(a, b):
        a()
    if mech == 'ttp':
        viz._render_rtd_ttp_section(df, 'rtd', '', both, False)
    else:
        viz._render_rtd_ranking_section(df, 'rtd', '', both, mech, False)
    return captured


@pytest.mark.parametrize("mech", list(RTD_SCORE_CHARTS))
def test_spec_chart_plots_the_values_the_statistics_count(spec_frame, monkeypatch, mech):
    df = spec_frame
    plotted = _plotted(df, mech, monkeypatch)
    series = plotted['series'].to_numpy(dtype=float)
    assert "stochastic draw" in plotted['title']
    if mech == 'ttp':
        counted = df['rtd_choice_length'].astype(int).to_numpy()
    else:
        counted = df[SEGMENT_COLS[mech]].astype(int).to_numpy()
    # graph and statistics describe the same values, agent by agent
    assert (_rebin(series, mech) == counted).all()
    # ... which the deterministic score (the Baseline graph) does not
    det_col = RTD_SCORE_CHARTS[mech][0]
    assert not np.allclose(series, df[det_col].to_numpy(dtype=float))


@pytest.mark.parametrize("mech", list(RTD_SCORE_CHARTS))
def test_baseline_chart_is_unchanged(baseline_frame, monkeypatch, mech):
    """Research Baseline: the deterministic score (the Stata figure), as before."""
    df = baseline_frame
    plotted = _plotted(df, mech, monkeypatch)
    det_col, title = RTD_SCORE_CHARTS[mech][:2]
    assert plotted['title'] == title and plotted['x_title'] == title
    assert plotted['series'].to_numpy(dtype=float) == pytest.approx(
        df[det_col].to_numpy(dtype=float))
    if mech == 'ttp':
        assert (_rebin(plotted['series'], mech)
                == df['rtd_choice_length'].astype(int).to_numpy()).all()


def test_spec_and_baseline_charts_differ_while_baseline_matches_stata(spec_frame,
                                                                     baseline_frame):
    spec, _, _, drawn = rtd_score_chart_data(spec_frame, 'ttp')
    base, _, _, base_drawn = rtd_score_chart_data(baseline_frame, 'ttp')
    assert drawn and not base_drawn
    assert np.histogram(spec, bins=16)[0].tolist() != np.histogram(base, bins=16)[0].tolist()
    # Baseline still reproduces the .dta's length distribution
    assert baseline_frame['rtd_choice_length'].value_counts().to_dict() == \
        {0: 20, 1: 92, 2: 95, 3: 57, 4: 14, 5: 2}


def test_undrawn_agents_fall_back_to_their_anchor():
    """Quintiles mode with one budget level's coefficient at 0: those agents carry no
    draw; the chart plots their anchor (the value the draw range used for them)."""
    df = pd.DataFrame({
        'rtd_weighted_ttp': [0.0, 0.1, 0.2],
        'rtd_weighted_ttp06': [0.0, 3.0, 5.9999],
        'rtd_ttp_draw': [0.4, np.nan, 6.2],
    })
    series, title, x_title, drawn = rtd_score_chart_data(df, 'ttp')
    assert drawn
    assert series.tolist() == [0.4, 3.0, 6.2]


def test_no_draw_column_or_all_missing_means_deterministic():
    df = pd.DataFrame({'rtd_loyalty_z': [0.1, 0.2], 'rtd_loyalty_draw': [np.nan, np.nan]})
    series, title, _, drawn = rtd_score_chart_data(df, 'loyalty')
    assert not drawn and title == "Loyalty score"
    assert series.tolist() == [0.1, 0.2]
