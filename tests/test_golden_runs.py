"""Golden runs: the parity net for refactoring.

A fixed set of runs - the six complete runs (three populations x two income modes),
each modelled decision alone in Copula and in Research Baseline, and Research
Specification with exactly 280 agents - goes through the same path the app uses
(session snapshot -> run plan -> execute). Every result frame and vendor list is
compared, cell by cell, with the copy stored under tests/golden/<scenario>/; for the
six 40-agent complete runs so is the output of the thirteen export builders in
``export_frames`` (frames and records; the downloaded files' content hashes are
recorded by tests/test_golden_screens.py). A refactor that moves a single number,
column or row fails here.

To regenerate the golden files after an INTENDED change:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/test_golden_runs.py -q

then review `git diff --stat tests/golden` and commit the new files with the change
that caused them.
"""
import gzip
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from helpers.seam import base_state, plan_for  # the app's session-snapshot shape

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"
BASE_TIME = datetime(2025, 1, 15)
N_AGENTS = 40
SEED = 42

POPULATIONS = {
    "copula": "Copula (synthetic)",
    "documentation": "Research Specification",
    "baseline": "Research Baseline",
}
INCOME_MODES = {"categorical": "categorical only", "continuous": "continuous only"}
MODELLED = ["disclose_income", "disclose_documents", "donation_default",
            "rejected_transaction_defaults"]


# ----------------------------------------------------------------- scenarios

def scenarios(repo):
    """Name -> session snapshot. Every scenario is one click in the app."""
    out = {}
    for pop_key, pop in POPULATIONS.items():
        for inc_key, inc in INCOME_MODES.items():
            out[f"complete_{pop_key}_{inc_key}"] = base_state(
                repo, population_mode=pop, income_spec_mode=inc,
                n_agents=N_AGENTS, seed=SEED, seed_input=SEED)
    for decision in MODELLED:
        for pop_key in ("copula", "baseline"):
            out[f"single_{decision}_{pop_key}"] = base_state(
                repo, selected=[decision], population_mode=POPULATIONS[pop_key],
                n_agents=N_AGENTS, seed=SEED, seed_input=SEED)
    # the 280 participants exactly once: pins the research sampling rule
    out["complete_documentation_categorical_280"] = base_state(
        repo, population_mode=POPULATIONS["documentation"], n_agents=280,
        seed=SEED, seed_input=SEED)
    return out


# ------------------------------------------------------------------- exports

def _frames(name, obj):
    """Normalise a builder's return value to {artifact_name: DataFrame}."""
    if isinstance(obj, pd.DataFrame):
        return {name: obj}
    if isinstance(obj, list):
        return {name: pd.DataFrame(obj)}
    if isinstance(obj, dict) and all(isinstance(v, pd.DataFrame) for v in obj.values()):
        return {f"{name}.{sheet}": frame for sheet, frame in obj.items()}
    if isinstance(obj, tuple):
        out = {}
        for i, part in enumerate(obj):
            out.update(_frames(f"{name}.{i}", part))
        return out
    raise TypeError(f"{name}: unexpected export type {type(obj).__name__}")


def export_frames(df, sim_params):
    """The thirteen frame / record export builders below, built from one result frame."""
    from app.reports import (
        build_agent_level_dataframe, build_agent_level_purchases, build_bid_value_export,
        build_donation_default_export, build_donation_transaction_export,
        build_purchase_request_export, build_purchase_vs_bid_export, build_transaction_export,
        build_transaction_level_dataframe, donation_default_export_columns,
        prepare_disclose_documents_excel_data, prepare_disclose_income_excel_data,
        prepare_disclosure_excel_data, prepare_rtd_model_export,
    )
    from app.reports.timestamps import TimestampConverter

    vendors = df.attrs.get("vendors")
    sp = sim_params
    prices = dict(market_price=sp.market_price, platform_markup=sp.platform_markup,
                  price_range=sp.price_range)
    times = dict(base_time=BASE_TIME, duration_hours=sp.duration_hours, periods=sp.periods)
    bounds = dict(vendor_price_min=sp.vendor_price_min, vendor_price_max=sp.vendor_price_max)
    converter = TimestampConverter(BASE_TIME, sp.duration_hours, sp.periods)

    builders = {
        "agent_level": lambda: build_agent_level_dataframe(df, vendors_data=vendors, **bounds),
        "transaction_level": lambda: build_transaction_level_dataframe(
            df, vendors_data=vendors, **times, **prices, **bounds),
        "donation_default": lambda: build_donation_default_export(
            df, donation_default_export_columns(df, "donation_default")),
        "donation_transactions": lambda: build_donation_transaction_export(
            df, vendors=vendors, **times, **prices),
        "purchase_vs_bid": lambda: build_purchase_vs_bid_export(
            df, ts_converter=converter, vendors=vendors, **prices),
        "purchase_requests": lambda: build_purchase_request_export(
            df, vendors, BASE_TIME, sp.duration_hours, sp.periods,
            platform_markup=sp.platform_markup, price_range=sp.price_range,
            price_min_config=sp.vendor_price_min, price_max_config=sp.vendor_price_max),
        "bid_values": lambda: build_bid_value_export(
            df, vendors, BASE_TIME, sp.duration_hours, sp.periods),
        "purchasing_transactions": lambda: build_transaction_export(
            df, BASE_TIME, sp.duration_hours, sp.periods),
        "agent_level_purchases": lambda: build_agent_level_purchases(
            df, sp.periods, sp.duration_hours),
        "disclose_income_sheet": lambda: prepare_disclose_income_excel_data(df),
        "disclose_documents_sheet": lambda: prepare_disclose_documents_excel_data(df),
        "disclosure_sheet": lambda: prepare_disclosure_excel_data(df),
        "rtd_model": lambda: prepare_rtd_model_export(df),
    }
    out = {}
    for name, build in builders.items():
        out.update(_frames(name, build()))
    return out


# -------------------------------------------------------------- serialisation

def frame_text(df):
    """A canonical, human-diffable JSON form of a frame (values to 15 digits)."""
    return df.to_json(orient="split", double_precision=15, default_handler=str,
                      date_format="iso")


def column_hashes(df):
    """Exact per-column digests (bytes of the values, not their printed form)."""
    out = {}
    for col in df.columns:
        series = df[col]
        try:
            hashed = pd.util.hash_pandas_object(series, index=False)
        except TypeError:
            hashed = pd.util.hash_pandas_object(series.astype(str), index=False)
        out[str(col)] = hashlib.sha256(hashed.to_numpy(dtype="uint64").tobytes()).hexdigest()
    return out


def _describe_difference(expected_text, actual_text):
    """Name the columns and the first rows that differ, for the assertion message."""
    try:
        exp, act = json.loads(expected_text), json.loads(actual_text)
    except json.JSONDecodeError:
        return "golden file is not valid JSON"
    lines = []
    exp_cols, act_cols = list(exp["columns"]), list(act["columns"])
    if exp_cols != act_cols:
        lines.append(f"columns: removed {sorted(set(exp_cols) - set(act_cols))}, "
                     f"added {sorted(set(act_cols) - set(exp_cols))}, "
                     f"order changed: {set(exp_cols) == set(act_cols)}")
    if len(exp["data"]) != len(act["data"]):
        lines.append(f"rows: {len(exp['data'])} expected, {len(act['data'])} actual")
    for col in [c for c in exp_cols if c in act_cols][:200]:
        i, j = exp_cols.index(col), act_cols.index(col)
        for row, (e, a) in enumerate(zip(exp["data"], act["data"])):
            if e[i] != a[j]:
                lines.append(f"column {col!r}: first difference at row {row}: {e[i]!r} -> {a[j]!r}")
                break
        if len(lines) > 12:
            lines.append("...")
            break
    return "\n".join(lines) or "texts differ but no cell difference found (formatting?)"


def check_or_write(path: Path, text: str, hashes: dict):
    """Compare `text`/`hashes` with the golden files at `path`, or write them.

    The text is stored gzip-compressed (`<name>.json.gz`); the per-column hashes
    stay plain so `git diff` shows which columns a change touched."""
    gz_path = path.with_name(path.name + ".gz")
    hash_path = path.with_suffix(".hashes.json")
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        gz_path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        hash_path.write_text(json.dumps(hashes, indent=1, sort_keys=True))
        return
    assert gz_path.exists(), f"no golden file {gz_path.relative_to(GOLDEN_DIR.parent)}; run with GOLDEN_UPDATE=1"
    expected_hashes = json.loads(hash_path.read_text())
    changed = sorted(c for c in set(expected_hashes) | set(hashes)
                     if expected_hashes.get(c) != hashes.get(c))
    expected_text = gzip.decompress(gz_path.read_bytes()).decode("utf-8")
    if changed or expected_text != text:
        raise AssertionError(
            f"{path.relative_to(GOLDEN_DIR.parent)} differs from the golden copy.\n"
            f"columns with changed values: {changed[:20]}\n"
            + _describe_difference(expected_text, text))


# --------------------------------------------------------------------- tests

@pytest.fixture(scope="module")
def repo():
    from app.seam.config_repo import get_config_repo
    return get_config_repo()


@pytest.fixture(scope="module")
def default_values():
    from src.contract.defaults import DEFAULT_DECISION_VALUES
    return DEFAULT_DECISION_VALUES


@pytest.fixture(scope="module")
def runs(repo, default_values):
    """{scenario: (sim_params, {result_key: DataFrame})}, executed once per session."""
    from app.seam.execute import execute
    out = {}
    for name, state in scenarios(repo).items():
        plan = plan_for(repo, default_values, state)
        out[name] = (state["sim_params"], execute(plan, config_repo=repo))
    return out


def _scenario_names():
    from app.seam.config_repo import get_config_repo
    return list(scenarios(get_config_repo()))


@pytest.mark.parametrize("scenario", _scenario_names())
def test_result_frames_match_golden(runs, scenario):
    _, results = runs[scenario]
    assert results, scenario
    for result_key, df in results.items():
        check_or_write(GOLDEN_DIR / scenario / f"{result_key}.frame.json",
                       frame_text(df), column_hashes(df))
        vendors = json.dumps(df.attrs.get("vendors"), default=str, sort_keys=True, indent=1)
        check_or_write(GOLDEN_DIR / scenario / f"{result_key}.vendors.json",
                       vendors, {"vendors": hashlib.sha256(vendors.encode()).hexdigest()})


@pytest.mark.parametrize("scenario", [s for s in _scenario_names()
                                      if s.startswith("complete_") and not s.endswith("_280")])
def test_exports_match_golden(runs, scenario):
    sim_params, results = runs[scenario]
    for result_key, df in results.items():
        for name, frame in export_frames(df, sim_params).items():
            check_or_write(GOLDEN_DIR / scenario / f"{result_key}.export.{name}.json",
                           frame_text(frame), column_hashes(frame))


def test_golden_directory_is_complete(runs):
    """Every golden file on disk belongs to a scenario that still exists (no stale files)."""
    if UPDATE or not GOLDEN_DIR.exists():
        return
    on_disk = {p.parent.name for p in GOLDEN_DIR.rglob("*.frame.json.gz")}
    assert on_disk <= set(runs), f"stale golden scenarios: {sorted(on_disk - set(runs))}"
