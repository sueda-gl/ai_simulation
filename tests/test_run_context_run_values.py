"""The results page reads the RUN's Page 1 parameters, not the live widgets.

``RunMetadata`` carries the SimulationParameters snapshot the run used, the Decision 4
element its run button named (``rtd_run_element``) and the donation anchor weight it
ran with; ``RunContext`` exposes them and every results reader asks RunContext. Two
contexts are built in one session whose LIVE values differ from the run's: with the
metadata, the readers show the run's values; with metadata that predates these
fields (an old session), they fall back to the live values, the only ones it has.
"""
import dataclasses

import pandas as pd
import pytest

from app.models import SimulationParameters
from app.seam.build_plan import donation_anchor_weights, simulation_params_snapshot
from app.simulation import _run_metadata_dict
from src.contract.plan import RunMetadata

pytestmark = pytest.mark.slow

RUN = dict(platform_markup=0.1, price_range=0.25, market_price=100.0, periods=1,
           duration_hours=1.0, vendor_price_min=50.0, vendor_price_max=150.0)
LIVE = dict(platform_markup=0.2, price_range=0.4, market_price=120.0, periods=3,
            duration_hours=2.0, vendor_price_min=60.0, vendor_price_max=140.0)


def _metadata(with_run_values=True):
    meta = RunMetadata(
        effective_population_mode="Copula (synthetic)", effective_income_mode="Continuous only",
        result_keys=("continuous",), custom_decisions=("rejected_transaction_defaults",),
        default_decisions=(), seed=42, n_agents=40, is_comparison=False,
        started_at="2025-01-15T00:00:00",
        simulation_params=simulation_params_snapshot(SimulationParameters(**RUN)),
        rtd_run_element="ttp", anchor_observed_weight=0.75)
    record = _run_metadata_dict(meta)
    if not with_run_values:
        for key in ("simulation_params", "rtd_run_element", "anchor_observed_weight"):
            del record[key]
    return record


def _script():
    import streamlit as st
    import pandas as pd
    from app.models import initialize_session_state
    from app.pages.results.overview import rtd_page_element, show_overview
    from app.pages.results.run_context import RunContext
    from app.pages.results.visualizations.bidding_viz import render_bid_value

    initialize_session_state()
    # the live widgets have moved on since the run
    for name, value in st.session_state["_live"].items():
        setattr(st.session_state.sim_params, name, value)
    st.session_state.anchor_observed_weight = 0.3
    st.session_state.rtd_run_element = "loyalty"
    st.session_state.n_agents = 60

    df = pd.DataFrame({"agent_id": [1, 2], "rtd_choice_length": [2, 3],
                       "donation_default": [0.1, 0.2]})
    st.session_state.simulation_results = {"continuous": df}
    st.session_state._run_metadata = st.session_state["_metadata"]

    ctx = RunContext.from_session()
    st.session_state["_probe"] = {
        "platform_markup": ctx.platform_markup, "price_range": ctx.price_range,
        "market_price": ctx.market_price, "periods": ctx.periods,
        "duration_hours": ctx.duration_hours, "vendor_price_min": ctx.vendor_price_min,
        "vendor_price_max": ctx.vendor_price_max, "n_agents": ctx.n_agents,
        "rtd_run_element": ctx.rtd_run_element, "anchor": ctx.anchor_observed_weight,
        "page_element": rtd_page_element(),
    }
    show_overview(df)                                   # the "Anchor mix" caption
    render_bid_value(df, "bid_value", "Bid Value", df["agent_id"])   # the markup line


def _run(metadata):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_script)
    at.session_state["_live"] = LIVE
    at.session_state["_metadata"] = metadata
    at.run(timeout=300)
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _texts(at):
    return [str(e.value) for e in list(at.caption) + list(at.markdown)]


def test_metadata_snapshot_is_a_plain_copy_of_the_page1_parameters():
    params = SimulationParameters(**RUN)
    snapshot = simulation_params_snapshot(params)
    assert snapshot == dataclasses.asdict(params)
    params.platform_markup = 0.99                          # a later edit of the object
    assert snapshot["platform_markup"] == 0.1              # does not reach the snapshot
    assert simulation_params_snapshot(None) == {}


def test_anchor_weight_is_the_slider_unless_a_pinned_donation_record_overrides_it():
    assert donation_anchor_weights({"anchor_observed_weight": 0.6}, None) == {"observed": 0.6, "predicted": 0.4}
    pinned = {"stochastic_params": {"anchor_weights": {"observed": 0.9, "predicted": 0.1}}}
    assert donation_anchor_weights({"anchor_observed_weight": 0.6}, pinned) == {"observed": 0.9, "predicted": 0.1}


def test_readers_use_the_run_metadata_not_the_live_widgets():
    at = _run(_metadata(with_run_values=True))
    probe = at.session_state["_probe"]
    assert {k: probe[k] for k in RUN} == RUN
    assert probe["n_agents"] == 40
    assert probe["rtd_run_element"] == "ttp" and probe["page_element"] == "ttp"
    assert probe["anchor"] == 0.75
    texts = _texts(at)
    assert "📊 Anchor mix: 0.75 observed | 0.25 predicted" in texts
    assert any("**Platform Markup (m)**: 10.0%" in t for t in texts)
    assert not any("20.0%" in t for t in texts)


def test_metadata_without_the_run_values_falls_back_to_the_live_widgets():
    at = _run(_metadata(with_run_values=False))
    probe = at.session_state["_probe"]
    assert {k: probe[k] for k in LIVE} == LIVE
    assert probe["n_agents"] == 40                        # n_agents was always in the metadata
    assert probe["rtd_run_element"] == "loyalty" and probe["page_element"] == "loyalty"
    assert probe["anchor"] == 0.3
    texts = _texts(at)
    assert "📊 Anchor mix: 0.30 observed | 0.70 predicted" in texts
    assert any("**Platform Markup (m)**: 20.0%" in t for t in texts)
