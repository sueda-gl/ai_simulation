"""
Lavie #14 (professor 2026-10): user-facing outputs say "integrated", never "consensus".

The engine keeps its internal rtd_consensus_* model columns; every export and on-screen
table renames them to rtd_integrated_*:

* the complete-run Agent-Level export (app/reports/agent_level.py);
* the Decision 4 workbooks (app/reports/rtd.py: integrated_ranking, kemeny_status, ...);
* whole-frame outputs - the donation single-config export, the raw-data view, the
  Individual Agent Details panel - through rename_consensus_columns / integrated_column_name;
* the CLI / Monte-Carlo result files written by scripts/run_simulation.py.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.decisions.rejected_transaction_defaults import (integrated_column_name,
                                                         rename_consensus_columns)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rtd_frame():
    from src.engine.core import Engine
    from src.engine.profile import PROFILES
    from app.seam.execute import sample_agents
    engine = Engine(PROFILES["baseline"])
    df = engine.run_simulation(30, 42, ["rejected_transaction_defaults"],
                               agents_df=sample_agents("baseline", 30, 42))
    assert any(c.startswith("rtd_consensus_") for c in df.columns)   # internal names kept
    return df


def _no_consensus(columns):
    bad = [c for c in columns if "consensus" in str(c).lower()]
    assert not bad, bad


def test_column_rename_helpers(rtd_frame):
    assert integrated_column_name("rtd_consensus_ranking") == "rtd_integrated_ranking"
    assert integrated_column_name("rtd_consensus_settled_by") == "rtd_integrated_settled_by"
    assert integrated_column_name("rtd_default_list") == "rtd_default_list"
    assert integrated_column_name(3) == 3
    renamed = rename_consensus_columns(rtd_frame)
    _no_consensus(renamed.columns)
    assert len(renamed.columns) == len(rtd_frame.columns)
    pd.testing.assert_series_equal(renamed["rtd_integrated_kemeny_status"],
                                   rtd_frame["rtd_consensus_kemeny_status"], check_names=False)
    # the engine frame itself is untouched
    assert "rtd_consensus_ranking" in rtd_frame.columns


def test_agent_level_export_headers(rtd_frame):
    from app.reports import build_agent_level_dataframe
    agent_df = build_agent_level_dataframe(rtd_frame)
    _no_consensus(agent_df.columns)
    for col in ("rtd_integrated_ranking", "rtd_integrated_kemeny_status",
                "rtd_integrated_n_kemeny_optimal", "rtd_integrated_is_kemeny_optimal",
                "rtd_integrated_settled_by", "rtd_integrated_truncated_by"):
        assert col in agent_df.columns, col


def test_decision4_workbooks(rtd_frame):
    from app.reports.rtd import prepare_rtd_model_export
    sheets = prepare_rtd_model_export(rtd_frame)
    assert sheets
    for name, sheet in sheets.items():
        assert "consensus" not in name.lower()
        _no_consensus(sheet.columns)


def test_cli_result_file(tmp_path, rtd_frame):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import run_simulation
    finally:
        sys.path.pop(0)
    for fmt in ("csv", "parquet"):
        out_dir = tmp_path / fmt
        args = argparse.Namespace(output_dir=str(out_dir), seed=42, agents=30,
                                  decision=["rejected_transaction_defaults"], format=fmt)
        run_simulation._save_results(rtd_frame, args)
        (path,) = list(out_dir.glob(f"*.{fmt}"))
        written = pd.read_csv(path) if fmt == "csv" else pd.read_parquet(path)
        _no_consensus(written.columns)
        assert "rtd_integrated_kemeny_status" in written.columns
