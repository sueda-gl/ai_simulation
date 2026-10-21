"""
The interface of the ``app/reports`` modules that had no test of their own: what each
builder returns (a frame, records, bytes), the sheet names of its workbook and the
columns it promises. The values are pinned by the golden tests
(tests/test_golden_runs.py); these tests pin the shape.

The builders that take a results frame get one 30-agent Research Baseline engine run
(every decision, seed 42), with the transaction ids assigned as app.seam.execute does.
``mc``, ``preview``, ``timestamps`` and ``xlsx`` take no results frame; they get small
inputs of the documented shape.
"""
import io
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

BASE_TIME = datetime(2025, 1, 15)
PRICES = dict(market_price=100.0, platform_markup=0.1, price_range=0.25)
TIMES = dict(base_time=BASE_TIME, duration_hours=1.0, periods=1)


@pytest.fixture(scope="module")
def run():
    from app.seam.execute import sample_agents
    from src.engine.core import Engine
    from src.engine.postprocess import assign_global_transaction_ids
    from src.engine.profile import PROFILES
    engine = Engine(PROFILES["baseline"])
    df = engine.run_simulation(30, 42, None, agents_df=sample_agents("baseline", 30, 42))
    return assign_global_transaction_ids(df)


def _sheets(workbook):
    """Sheet names of xlsx bytes (asserting they are xlsx bytes: a zip archive)."""
    assert isinstance(workbook, bytes) and workbook[:2] == b"PK"
    return pd.ExcelFile(io.BytesIO(workbook)).sheet_names


def test_bidding_report_interface(run):
    from app.reports import bidding
    records = bidding.build_bid_value_export(run, run.attrs["vendors"], BASE_TIME, 1.0, 1)
    assert isinstance(records, list) and records
    frame = pd.DataFrame(records)
    assert list(frame.columns) == [
        "Transaction ID", "Agent ID", "Honesty_Humility", "Assigned Allowance Level",
        "Study Program", "Group_experiment", "TWT+Sospeso [=AW2+AX2]{Periods 1+2}", "income",
        "Vendor ID", "Vendor Price", "Bid Value", "Purchase Timestamp", "Period"]
    assert _sheets(bidding.bid_values_xlsx(frame)) == ["Total", "Period 1"]
    bids = bidding.collect_bid_values(run)
    assert len(bids) == len(records)
    stats = bidding.bid_stats_frame(bids)
    assert isinstance(stats, pd.DataFrame) and list(stats.columns) == ["Metric", "Value"]
    assert len(bidding.bidding_range(0.1, 0.25, 100.0)) == 3


def test_disclosure_report_interface(run):
    from app.reports import disclosure as d
    di = d.prepare_disclose_income_excel_data(run)
    dd = d.prepare_disclose_documents_excel_data(run)
    both = d.prepare_disclosure_excel_data(run)
    for frame, width in ((di, 20), (dd, 15), (both, 12)):
        assert isinstance(frame, pd.DataFrame) and frame.shape == (len(run), width)
    assert {"Agent ID", "PB_i", "Disclosure Income", "Disclose Income (Y=1)"} <= set(di.columns)
    assert {"Agent ID", "Disclose Documents (Y=1)", "customer_type"} <= set(dd.columns)
    assert {"disclose_income", "disclose_documents", "Regular", "Fixed", "Discount"} <= set(both.columns)
    assert _sheets(d.build_disclose_income_xlsx(di)) == ["Agent Disclose Income Data"]
    assert _sheets(d.build_disclose_documents_xlsx(dd)) == ["Agent Disclose Documents Data"]
    assert _sheets(d.build_disclosure_xlsx(both)) == ["Agent Disclosure Data"]
    assert _sheets(d.build_disclose_income_results_xlsx(di)) == ["Disclose Income Results"]
    assert _sheets(d.build_disclose_documents_results_xlsx(dd)) == ["Disclose Documents"]
    both_modes = {"categorical": run, "continuous": run}
    assert list(d.disclose_income_compare_both_sheets(both_modes)) == ["Categorical", "Continuous"]
    assert list(d.disclose_documents_export_sheets(both_modes, False)) == ["Categorical", "Continuous"]
    compare_all = {key: run for key, _ in d.DISCLOSURE_COMPARE_ALL_SHEETS}
    names = [name for _, name in d.DISCLOSURE_COMPARE_ALL_SHEETS]
    assert list(d.disclose_income_compare_all_sheets(compare_all)) == names
    assert list(d.disclose_documents_export_sheets(compare_all, True)) == names
    assert _sheets(d.build_disclosure_sheets_xlsx(d.disclose_income_compare_all_sheets(compare_all))) == names


def test_donation_report_interface(run):
    from app.reports import donation as dn
    export = dn.build_donation_default_export(run, dn.donation_default_export_columns(run, "donation_default"))
    assert list(export.columns) == [
        "Agent ID", "Assigned Allowance Level", "Group_experiment", "Honesty_Humility",
        "Study Program", "TWT+Sospeso [=AW2+AX2]{Periods 1+2}", "donation_default"]
    assert len(export) == len(run)
    assert _sheets(dn.build_donation_default_xlsx(export)) == ["Donation Results"]
    records = dn.build_donation_transaction_export(run, vendors=run.attrs["vendors"], **TIMES, **PRICES)
    transactions = dn.donation_transactions_frame(records)
    assert isinstance(records, list) and len(transactions) == len(records) > 0
    assert {"Transaction ID", "Agent ID", "Period", "Customer Price", "Default Donation Rate",
            "Final Donation Rate"} <= set(transactions.columns)
    assert _sheets(dn.build_donation_transactions_xlsx(transactions)) == ["Total", "Period 1"]
    configs = dn.donation_all_configs_frame({"categorical": run, "continuous": run})
    assert {"Agent ID", "donation_default_Categorical", "donation_default_Continuous"} <= set(configs.columns)
    single = dn.donation_single_export_frame(run)
    assert len(single) == len(run) and "Agent ID" in single.columns


def test_mc_report_interface():
    from app.reports import mc
    summary = pd.DataFrame({"decision": ["donation_default", "disclose_income"], "mean": [0.1, 0.4],
                            "std": [0.01, 0.05], "p2.5": [0.08, 0.3], "p97.5": [0.12, 0.5], "runs": [3, 3]})
    detailed = pd.DataFrame({"run": [1, 2, 3], "donation_default_mean": [0.1, 0.11, 0.09]})
    assert list(mc.overview_metrics(summary)) == [
        "Mean Donation Rate", "Standard Deviation", "95% CI Lower", "95% CI Upper"]
    assert mc.overview_metrics(summary.iloc[1:]) is None            # donation not in the study
    shown = mc.format_summary_for_display(summary)
    assert list(shown.columns) == list(summary.columns) and shown["mean"].tolist() == ["10.00%", "40.00%"]
    assert summary["mean"].tolist() == [0.1, 0.4]                     # formatted on a copy
    assert mc.has_convergence_series(detailed)
    assert mc.add_running_mean(detailed) is detailed and "running_mean" in detailed.columns
    assert set(mc.convergence_interval(detailed)) == {"final_mean", "final_std", "ci_upper", "ci_lower"}
    assert mc.convergence_interval(detailed.head(1)) is None
    assert mc.summary_csv(summary).splitlines()[0] == "decision,mean,std,p2.5,p97.5,runs"
    assert mc.detailed_csv(detailed).splitlines()[0] == "run,donation_default_mean,running_mean"


DISTRIBUTION = dict(
    income_distribution="lognormal", lognormal_mu=10.0, lognormal_sigma=0.5, lognormal_min=0.0,
    lognormal_max=None, gg_k=1.5, gg_c=2.0, gg_lambda=20000.0, gg_min=0.0, gg_max=None,
    dagum_a=2.0, dagum_p=1.5, dagum_b=25000.0, dagum_min=0.0, dagum_max=None,
    income_min=0.0, income_max=100000.0)


def test_preview_report_interface():
    from app.reports import preview
    assert set(preview.DISTRIBUTION_PARAM_NAMES) == set(DISTRIBUTION)
    assert preview.distribution_kwargs(SimpleNamespace(**DISTRIBUTION, periods=3)) == DISTRIBUTION
    for distribution in ("lognormal", "generalised_gamma", "dagum", "uniform"):
        params = dict(DISTRIBUTION, income_distribution=distribution)
        samples = preview.sample_income_distribution(n_samples=50, seed=3, **params)
        assert isinstance(samples, np.ndarray) and samples.shape == (50,), distribution
        np.testing.assert_array_equal(samples, preview.sample_income_distribution(n_samples=50, seed=3, **params))
    rate = preview.discount_qualification_rate(discount_income_threshold=12500.0, n_samples=200, **DISTRIBUTION)
    assert 0.0 <= rate <= 1.0


def test_purchasing_report_interface(run):
    from app.reports import purchasing as pu
    transactions = pu.build_transaction_export(run, BASE_TIME, 1.0, 1)
    assert isinstance(transactions, pd.DataFrame) and len(transactions) > 0
    assert list(transactions.columns) == [
        "transaction_id", "Agent ID", "vendorID", "platformProductID", "purchase type",
        "purchase_bid_value", "Purchase Timestamp", "Period"]
    assert _sheets(pu.purchasing_transactions_xlsx(transactions)) == ["Transactions"]
    records = pu.build_agent_level_purchases(run, 1, 1.0)
    assert isinstance(records, list) and records
    agents = pd.DataFrame(records)
    assert {"Agent ID", "Count of Purchase Requests", "Count of Completed Transactions",
            "% Completed Transactions", "Period"} <= set(agents.columns)
    assert _sheets(pu.agent_level_purchases_xlsx(agents, 1)) == ["Total", "P1"]


def test_timestamps_report_interface(run):
    from app.reports.timestamps import TimestampConverter, calculate_period
    converter = TimestampConverter(BASE_TIME, 2.0, 3)
    converted = converter.convert(4.5)
    assert set(converted) == {"datetime", "formatted", "date", "time", "period", "timestamp_hours"}
    assert converted["datetime"] == converter.to_datetime(4.5) == datetime(2025, 1, 15, 4, 30)
    assert converted["formatted"] == converter.to_formatted(4.5) == "15/01/2025 04:30"
    assert converted["period"] == converter.to_period(4.5) == calculate_period(4.5, 2.0) == 3
    assert converter.get_term_duration() == 6.0
    hours = [r["timestamp_hours"] for requests in run["purchase_requests"] for r in requests]
    assert hours and all(converter.convert(h)["datetime"] >= BASE_TIME for h in hours)


def test_transactions_report_interface(run):
    from app.reports import transactions as t
    from app.reports.timestamps import TimestampConverter
    records = t.build_purchase_vs_bid_export(run, ts_converter=TimestampConverter(BASE_TIME, 1.0, 1),
                                             vendors=run.attrs["vendors"], **PRICES)
    frame = pd.DataFrame(records)
    assert {"Purchase Request ID", "Agent ID", "Purchase Request Type", "Vendor", "Vendor Price",
            "Purchase Timestamp", "Period", "Customer Price"} <= set(frame.columns)
    assert _sheets(t.purchase_vs_bid_xlsx_bytes(frame)) == ["Total", "Period 1"]
    counts, total = t.purchase_vs_bid_request_counts(run)
    assert set(counts) <= {"PN", "BID"} and sum(counts.values()) == total == len(records)
    priorities = t.prepare_priority_lists_export(run, run["rejected_transaction_defaults"])
    assert len(priorities) == len(run)
    assert list(priorities.columns)[-5:] == [f"Priority {i}" for i in range(1, 6)]
    assert _sheets(t.priority_lists_xlsx_bytes(priorities)) == ["Priority Lists"]


def test_vendor_report_interface(run):
    from app.reports import vendor as v
    vendors = run.attrs["vendors"]
    requests = pd.DataFrame(v.build_purchase_request_export(
        run, vendors, BASE_TIME, 1.0, 1, platform_markup=0.1, price_range=0.25,
        price_min_config=50.0, price_max_config=150.0))
    assert {"Purchase Request ID", "Agent ID", "Selected Vendor", "Vendor Price",
            "Vendor Integrated Score", "Customer Paid Price", "Period"} <= set(requests.columns)
    assert _sheets(v.purchase_requests_xlsx(requests)) == ["Total", "Period 1"]
    weights = pd.DataFrame(v.build_vendor_choice_weights_export(run, run["vendor_choice_weights"]))
    assert len(weights) == len(run)
    assert {"Agent ID", "Price", "Quality", "Proximity", "Sustainability"} <= set(weights.columns)
    assert _sheets(v.vendor_choice_weights_xlsx(weights)) == ["Vendor Choice Weights"]
    proximity = pd.DataFrame(v.build_proximity_matrix(run))
    assert list(proximity.columns) == ["Agent ID"] + [f"Vendor {x['vendor_id']} Proximity" for x in vendors]
    assert _sheets(v.proximity_matrix_xlsx(proximity)) == ["Agent-Vendor Proximity"]


def test_xlsx_report_interface():
    from app.reports import xlsx
    frame = pd.DataFrame({"Agent ID": [1, 2, 3], "Period": [2, 1, 2], "Vendor Price": [10.0, 11.0, 12.0]})
    sheets = xlsx.period_sheets(frame)
    assert list(sheets) == ["Total", "Period 1", "Period 2"]
    assert sheets["Total"] is frame and sheets["Period 2"]["Agent ID"].tolist() == [1, 3]
    assert list(xlsx.period_sheets(frame[["Agent ID"]])) == ["Total"]
    with pytest.raises(KeyError):
        xlsx.period_sheets(frame[["Agent ID"]], missing_ok=False)
    assert _sheets(xlsx.to_xlsx_bytes(sheets)) == ["Total", "Period 1", "Period 2"]
    calls = []
    xlsx.to_xlsx_bytes({"Only": frame}, formatter=lambda writer, name, df: calls.append((name, len(df))))
    assert calls == [("Only", 3)]
    for formatter in (xlsx.apply_export_price_formatting, xlsx.apply_disclosure_price_formatting,
                      xlsx.apply_transaction_price_formatting, xlsx.apply_bid_price_formatting,
                      xlsx.apply_vendor_price_formatting, xlsx.apply_donation_price_formatting):
        assert _sheets(xlsx.to_xlsx_bytes({"Sheet": frame}, formatter)) == ["Sheet"], formatter.__name__
