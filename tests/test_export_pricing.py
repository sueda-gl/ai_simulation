"""The two export rules the platform documentation fixes (docs/reference/COOPECON AI
Agent Documentation 2026-10-07.docx, Transaction-Level export and Decision 8):

* Customer Price: Purchase Now = (1 + price_range) x (1 + platform_markup) x the
  VENDOR's price; Bid = the bid value; Fixed / Discount = N/A.  Money columns are
  stored rounded to two decimals (`float(f"{x:.2f}")`), the price computed from the
  exact vendor price first.
* The integrated vendor score is one formula everywhere, so the Agent-Level sheet's
  avg_vendor_score equals the mean of the Transaction-Level sheet's Vendor Integrated
  Score over the same agent's requests (including the single-vendor case, where the
  Page-1 price bounds coincide).
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.reports import build_agent_level_dataframe, build_transaction_level_dataframe

PLATFORM_MARKUP = 0.1
PRICE_RANGE = 0.25
MARKET_PRICE = 100.0


def _run(num_vendors, price_min, price_max, n_agents=30, seed=7):
    from app.seam.execute import sample_agents
    from src.engine.core import Engine
    from src.engine.profile import PROFILES

    engine = Engine(PROFILES["copula"])
    engine.simulation_config["simulation"].update({
        "num_vendors": num_vendors,
        "vendor_price_min": price_min,
        "vendor_price_max": price_max,
        "market_price": MARKET_PRICE,
        "platform_markup": PLATFORM_MARKUP,
        "price_range": PRICE_RANGE,
    })
    df = engine.run_simulation(n_agents, seed, agents_df=sample_agents("copula", n_agents, seed))
    return df, df.attrs["vendors"], price_min, price_max


def _sheets(run):
    df, vendors, price_min, price_max = run
    transactions = build_transaction_level_dataframe(
        df, vendors_data=vendors, base_time=datetime(2025, 1, 15), periods=1,
        duration_hours=1.0, market_price=MARKET_PRICE, platform_markup=PLATFORM_MARKUP,
        price_range=PRICE_RANGE, vendor_price_min=price_min, vendor_price_max=price_max,
    )
    transactions.attrs["vendors"] = vendors      # the exact vendor prices, by vendor_id
    agents = build_agent_level_dataframe(
        df, vendors_data=vendors, vendor_price_min=price_min, vendor_price_max=price_max)
    return transactions, agents


@pytest.fixture(scope="module")
def multi_vendor():
    return _sheets(_run(num_vendors=3, price_min=50.0, price_max=150.0))


@pytest.fixture(scope="module")
def single_vendor():
    # Page 1 forces both bounds to the market price when there is one vendor
    return _sheets(_run(num_vendors=1, price_min=MARKET_PRICE, price_max=MARKET_PRICE))


def test_multi_vendor_run_has_distinct_vendor_prices(multi_vendor):
    transactions, _ = multi_vendor
    assert transactions["Vendor Price"].nunique() > 1


def test_purchase_now_price_is_derived_from_the_vendor_price(multi_vendor):
    transactions, _ = multi_vendor
    pn = transactions[transactions["Purchase Request Type"] == "Purchase Now"]
    assert len(pn) > 0
    exact = {f"Vendor {int(v['vendor_id'])}": float(v["price"]) for v in transactions.attrs["vendors"]}
    # computed from the EXACT vendor price, then stored rounded to two decimals
    expected = [float(f"{(1 + PRICE_RANGE) * (1 + PLATFORM_MARKUP) * exact[vid]:.2f}")
                for vid in pn["Vendor ID"]]
    assert pn["Customer Price"].astype(float).tolist() == expected
    # the stored Vendor Price is the exact price rounded the same way
    assert pn["Vendor Price"].astype(float).tolist() == [float(f"{exact[vid]:.2f}") for vid in pn["Vendor ID"]]
    # and NOT the market-price convention for vendors priced away from the market
    off_market = pn[np.abs(pn["Vendor Price"].astype(float) - MARKET_PRICE) > 1.0]
    assert len(off_market) > 0
    market_pn = (1 + PRICE_RANGE) * (1 + PLATFORM_MARKUP) * MARKET_PRICE
    assert not np.allclose(off_market["Customer Price"].astype(float), market_pn)


def test_bid_price_is_the_bid_value(multi_vendor):
    transactions, _ = multi_vendor
    bids = transactions[transactions["Purchase Request Type"] == "Bid"]
    assert len(bids) > 0
    np.testing.assert_allclose(bids["Customer Price"].astype(float),
                               bids["Bid Value"].astype(float), rtol=1e-12)


def test_fixed_and_discount_rows_carry_no_price(multi_vendor):
    transactions, _ = multi_vendor
    other = transactions[~transactions["Purchase Request Type"].isin(["Purchase Now", "Bid"])]
    assert len(other) > 0
    assert (other["Customer Price"] == "N/A").all()


@pytest.mark.parametrize("fixture_name", ["multi_vendor", "single_vendor"])
def test_agent_level_score_is_the_mean_of_the_transaction_level_score(fixture_name, request):
    transactions, agents = request.getfixturevalue(fixture_name)
    per_agent = (transactions.dropna(subset=["Vendor Integrated Score"])
                 .groupby("Agent ID")["Vendor Integrated Score"].mean())
    merged = agents.set_index("Agent ID")["avg_vendor_score"].to_frame().join(per_agent, how="inner")
    assert len(merged) > 0
    np.testing.assert_allclose(merged["avg_vendor_score"].astype(float),
                               merged["Vendor Integrated Score"].astype(float), rtol=1e-12)


def test_single_vendor_price_component_is_full_weight(single_vendor):
    # the documentation gives no equal-bounds rule; the shared function scores the only
    # vendor as the cheapest available (norm_price = 1.0), and every sheet must agree
    transactions, _ = single_vendor
    scored = transactions.dropna(subset=["Standardized Vendor Price Score"])
    assert len(scored) > 0
    assert (scored["Standardized Vendor Price Score"].astype(float) == 1.0).all()
