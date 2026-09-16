"""
Stata parity of the Decision 4 score histograms.

The professor's design document draws every Decision 4 score with Stata's DEFAULT
histogram rule:

    k = round( min( sqrt(N), 10 * log10(N) ) )

equal-width bins spanning min..max (N = 280 -> 16.733 -> k = 17). Stata simply does not
DRAW bins that no observation falls into, which is why the document's figures show fewer
bars than k: 16 bars for weighted_ttp, 14 for z_WTP_calculated, and so on. This module
pins both halves of that: the rule itself, and the resulting non-empty bin counts on the
professor's own Stata data.

The bin computation under test is the pure helper
`transaction_viz._rtd_stata_bins(series) -> (edges, counts)`, which `_rtd_density_hist`
uses to build the plotly xbins, so these assertions describe the rendered chart without
needing Streamlit.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.pages.results.visualizations.transaction_viz import (
    _rtd_stata_bin_count,
    _rtd_stata_bins,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# The professor's Stata file (not in the repo - skipped when it is not on this machine).
STATA_DTA = Path("/Users/suedagul/Downloads/Stata_File_Decision4_050926.dta")
# Frozen extract of the same run that ships with the repo.
STATA_CSV = REPO_ROOT / "data" / "stata_d4_verification.csv"

# Column -> number of BARS the design document's figure shows for that score, i.e. the
# number of non-empty bins among the k = 17 equal-width bins over 280 participants.
# `z_stdactions` is the extract's name for the .dta's `z_stdactionsP`; both are listed
# because the extract is regenerated independently of this test.
DOCUMENT_BAR_COUNTS = {
    'weighted_ttp': 16,
    'z_WTP_calculated': 14,
    'z_RT_calculated_hs': 16,
    'z_Flexibility_calculated_ivw': 17,
    'z_stdactions': 12,
    'z_stdactionsP': 12,
    'weighted_WTP_categorical': 14,
    'weighted_RT_categorical': 16,
    'z_RT_categorical': 16,
    'z_WTP_categorical': 14,
}


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, expected_k", [
    (280, 17),    # min(16.733, 24.472) = 16.733 -> 17: the document's Decision 4 figures
    (1000, 30),   # min(31.623, 30.000) = 30
    (100, 10),    # min(10.000, 20.000) = 10
    (10, 3),      # min(3.162, 10.000) = 3.162 -> 3
    (2, 1),       # min(1.414, 3.010) = 1.414 -> 1
])
def test_stata_bin_count_rule(n, expected_k):
    assert _rtd_stata_bin_count(n) == expected_k


def test_stata_bin_count_matches_the_closed_form_everywhere():
    """No off-by-one against round(min(sqrt(N), 10*log10(N))) over a wide range. Stata
    rounds half AWAY FROM ZERO, so the implementation is floor(x + 0.5), not Python's
    round(), which rounds half to even."""
    for n in range(1, 5001):
        raw = min(math.sqrt(n), 10.0 * math.log10(n))
        assert _rtd_stata_bin_count(n) == max(1, math.floor(raw + 0.5)), n


def test_bin_count_never_below_one():
    for n in (0, 1):
        assert _rtd_stata_bin_count(n) >= 1


# ---------------------------------------------------------------------------
# Shape of the bins the chart draws
# ---------------------------------------------------------------------------
def test_bins_span_min_to_max_in_k_equal_width_intervals():
    rng = np.random.default_rng(11)
    s = pd.Series(rng.normal(size=280))
    edges, counts = _rtd_stata_bins(s)

    assert len(counts) == 17
    assert len(edges) == 18
    assert edges[0] == pytest.approx(float(s.min()))
    assert edges[-1] == pytest.approx(float(s.max()))
    widths = np.diff(edges)
    assert widths == pytest.approx(np.full(17, widths[0]))
    # maximum included in the LAST bin, nothing lost -> proportions sum to 1
    assert counts.sum() == len(s)
    assert (counts / counts.sum()).sum() == pytest.approx(1.0)


def test_missing_values_are_dropped_before_the_rule_is_applied():
    """k is computed from the number of NON-MISSING observations."""
    s = pd.Series(list(np.linspace(0.0, 1.0, 280)) + [np.nan] * 40)
    edges, counts = _rtd_stata_bins(s)
    assert _rtd_stata_bin_count(280) == 17
    assert len(counts) == 17
    assert counts.sum() == 280


def test_maximum_falls_in_the_last_bin():
    s = pd.Series([0.0] * 100 + [1.0])
    _edges, counts = _rtd_stata_bins(s)
    assert counts[-1] == 1
    assert counts.sum() == len(s)


def test_constant_and_empty_series_degrade_to_one_bin():
    _edges, counts = _rtd_stata_bins(pd.Series([2.5] * 50))
    assert list(counts) == [50]
    _edges, counts = _rtd_stata_bins(pd.Series([], dtype=float))
    assert list(counts) == [0]


def test_empty_bins_have_zero_height_and_are_not_dropped():
    """A gap in the data leaves zero-count bins in place (Stata omits drawing them; the
    app draws them at zero height) - the count vector still has length k."""
    s = pd.Series(list(np.linspace(0.0, 1.0, 140)) + list(np.linspace(9.0, 10.0, 140)))
    _edges, counts = _rtd_stata_bins(s)
    assert len(counts) == 17
    assert (counts == 0).any()
    assert counts.sum() == 280


# ---------------------------------------------------------------------------
# Parity with the professor's Stata figures
# ---------------------------------------------------------------------------
def _stata_frames():
    """(label, DataFrame) for every Stata source available on this machine."""
    frames = []
    if STATA_CSV.exists():
        frames.append(("stata_d4_verification.csv", pd.read_csv(STATA_CSV)))
    if STATA_DTA.exists():
        frames.append((STATA_DTA.name, pd.read_stata(STATA_DTA)))
    return frames


SOURCES = _stata_frames()


@pytest.mark.skipif(not SOURCES, reason="no Stata source available on this machine")
@pytest.mark.parametrize("label, df", SOURCES, ids=[s[0] for s in SOURCES])
def test_stata_source_has_280_participants_and_17_bins(label, df):
    assert len(df) == 280
    assert _rtd_stata_bin_count(280) == 17


@pytest.mark.skipif(not SOURCES, reason="no Stata source available on this machine")
@pytest.mark.parametrize("label, df", SOURCES, ids=[s[0] for s in SOURCES])
def test_non_empty_bin_counts_match_the_document_figures(label, df):
    """Read defensively: the frozen extract is regenerated independently of this test,
    so absent columns are skipped rather than failed. At least one score must be
    checkable, otherwise the test is meaningless and fails."""
    checked = {}
    for col, expected_bars in DOCUMENT_BAR_COUNTS.items():
        if col not in df.columns:
            continue
        s = pd.Series(df[col]).dropna().astype(float)
        if s.empty:
            continue
        n = len(s)
        assert _rtd_stata_bin_count(n) == 17, f"{label}:{col} N={n}"
        edges, counts = _rtd_stata_bins(s)
        assert len(counts) == 17, f"{label}:{col}"
        assert counts.sum() == n, f"{label}:{col} lost observations"
        checked[col] = int((counts > 0).sum())

    assert checked, f"{label}: none of the document's score columns were present"
    expected = {col: DOCUMENT_BAR_COUNTS[col] for col in checked}
    assert checked == expected


@pytest.mark.skipif(not SOURCES, reason="no Stata source available on this machine")
@pytest.mark.parametrize("label, df", SOURCES, ids=[s[0] for s in SOURCES])
def test_bar_heights_sum_to_one(label, df):
    """histnorm='probability': every bar is a proportion and the heights sum to 1."""
    for col in DOCUMENT_BAR_COUNTS:
        if col not in df.columns:
            continue
        s = pd.Series(df[col]).dropna().astype(float)
        if s.empty:
            continue
        _edges, counts = _rtd_stata_bins(s)
        heights = counts / counts.sum()
        assert heights.sum() == pytest.approx(1.0), f"{label}:{col}"


@pytest.mark.skipif(len(SOURCES) < 2, reason="needs both the .dta and the frozen extract")
def test_extract_and_dta_agree_on_the_shared_scores():
    """Where the frozen extract and the professor's .dta carry the same score, the bar
    counts agree (a regenerated extract that drifts from the .dta shows up here)."""
    by_label = dict(SOURCES)
    csv_df = by_label["stata_d4_verification.csv"]
    dta_df = by_label[STATA_DTA.name]
    shared = [c for c in DOCUMENT_BAR_COUNTS
              if c in csv_df.columns and c in dta_df.columns]
    assert shared
    for col in shared:
        _e1, c1 = _rtd_stata_bins(csv_df[col])
        _e2, c2 = _rtd_stata_bins(dta_df[col])
        assert int((c1 > 0).sum()) == int((c2 > 0).sum()), col
