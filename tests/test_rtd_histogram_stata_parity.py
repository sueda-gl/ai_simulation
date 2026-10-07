"""
Stata parity of the Decision 4 score histograms.

Stata's `histogram` default is bins = min(sqrt(N), 10*ln(N)/ln(10)) TRUNCATED to an
integer (N = 280: 16.73 -> 16 bins). The design document's Stata figure for
weighted_loyalty is bar-for-bar np.histogram(.dta weighted_loyalty, bins=16); an
earlier version of the app rounded (-> 17 bins) and so drew a different figure.

The frozen reference is data/stata_d4_verification.csv, the 280 participants' values
exported from Stata_File_Decision4_050926.dta.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.reports.rtd import (
    RTD_MAX_BINS,
    rtd_bin_count,
    rtd_stata_bin_count,
    rtd_stata_bins,
)

D4_CSV = Path(__file__).resolve().parents[1] / "data" / "stata_d4_verification.csv"
D4_SCORES = ["weighted_ttp", "weighted_loyalty", "z_WTP_calculated",
             "z_RT_calculated_hs", "z_anchored_flexibility"]

# Bar heights (Stata density = count / (N * width)) of the document's weighted_loyalty
# figure, N = 280, 16 bins.
DOC_LOYALTY_DENSITIES = [0.016, 0.008, 0.065, 0.105, 0.234, 0.258, 0.396, 0.396,
                         0.347, 0.234, 0.105, 0.073, 0.008, 0.008, 0.0, 0.008]


@pytest.fixture(scope="module")
def d4():
    return pd.read_csv(D4_CSV)


def _stata_counts(x, k):
    """Stata's own assignment: start = min, width = (max - min)/k,
    bin = floor((x - start)/width), the maximum folded into the last bin."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    b = np.floor((x - lo) / ((hi - lo) / k)).astype(int)
    return np.bincount(np.minimum(b, k - 1), minlength=k)


# ---------------------------------------------------------------------------
# Bin count rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, expected", [
    (1, 1),        # sqrt 1, 10*log10 0 -> floor at one bin
    (10, 3),       # sqrt 3.16
    (50, 7),       # sqrt 7.07 vs 10*log10 16.99
    (74, 8),       # `sysuse auto` / `histogram mpg`: Stata reports bin=8 (sqrt 8.60)
    (100, 10),     # sqrt exactly 10
    (280, 16),     # sqrt 16.73 -> 16, NOT 17 (the old round() bug)
    (500, 22),     # sqrt 22.36
    (1000, 29),    # sqrt 31.62 vs 10*ln(1000)/ln(10) = 29.999999999999996 -> 29
    (10000, 40),   # sqrt 100 vs 10*ln/ln 40.0
])
def test_stata_bin_count_truncates(n, expected):
    assert rtd_stata_bin_count(n) == expected
    raw = min(math.sqrt(n), 10 * math.log(n) / math.log(10))
    assert rtd_stata_bin_count(n) == max(1, int(raw))


def test_bin_count_is_capped_at_the_280_participant_value():
    assert RTD_MAX_BINS == 16 == rtd_stata_bin_count(280)
    for n, expected in [(50, 7), (100, 10), (279, 16), (280, 16), (289, 16),
                        (500, 16), (1000, 16), (10000, 16)]:
        assert rtd_bin_count(n) == expected
    assert rtd_stata_bin_count(0) == 1 and rtd_bin_count(0) == 1


# ---------------------------------------------------------------------------
# Bin edges and counts against the frozen .dta export
# ---------------------------------------------------------------------------
def test_weighted_loyalty_bars_equal_the_documents_16_bin_figure(d4):
    s = d4["weighted_loyalty"].dropna().astype(float)
    assert len(s) == 280
    edges, counts = rtd_stata_bins(s)
    assert len(counts) == 16
    width = edges[1] - edges[0]
    densities = counts / (len(s) * width)
    assert np.round(densities, 3).tolist() == DOC_LOYALTY_DENSITIES
    # the 17-bin (rounded) version is NOT the document's figure
    h17 = np.histogram(s, bins=17, density=True)[0]
    assert np.round(h17, 3).tolist() != DOC_LOYALTY_DENSITIES


@pytest.mark.parametrize("col", D4_SCORES)
def test_bins_match_numpy_and_stata_assignment(d4, col):
    s = d4[col].dropna().astype(float).to_numpy()
    edges, counts = rtd_stata_bins(s)
    k = len(counts)
    assert k == 16
    # Stata's start / width
    assert edges[0] == s.min()
    assert np.diff(edges) == pytest.approx(np.full(k, (s.max() - s.min()) / k))
    assert edges[-1] >= s.max()
    # np.histogram(bins=k) and Stata's floor((x - min)/width) assignment, bin for bin
    assert counts.tolist() == np.histogram(s, bins=k)[0].tolist()
    assert counts.tolist() == _stata_counts(s, k).tolist()
    # the maximum lands in the last bin; nothing is lost
    assert counts[-1] >= int((s == s.max()).sum())
    assert counts.sum() == len(s)


def test_left_closed_bins_and_max_in_last_bin():
    # 0..4 in 4 bins of width 1: [0,1) [1,2) [2,3) [3,4]
    s = pd.Series([0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0] + [0.5] * 9)
    assert rtd_bin_count(len(s)) == 4
    edges, counts = rtd_stata_bins(s)
    assert edges.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert counts.tolist() == [10, 2, 1, 3]


def test_app_chart_draws_the_document_figure(d4, monkeypatch):
    """The Decision 4 score chart draws exactly these 16 bars (as proportions), the
    maximum included."""
    import app.pages.results.visualizations.transaction_viz as viz
    captured = {}
    monkeypatch.setattr(viz.st, 'plotly_chart',
                        lambda fig, **kw: captured.__setitem__('fig', fig))
    s = d4["weighted_loyalty"].astype(float)
    viz._rtd_density_hist(s, "t", "x", "k")
    ys = np.asarray(captured['fig'].data[0].y, dtype=float)
    edges, counts = rtd_stata_bins(s)
    width = edges[1] - edges[0]
    assert len(ys) == 16
    assert ys.sum() == pytest.approx(1.0)
    assert np.round(ys / width, 3).tolist() == DOC_LOYALTY_DENSITIES
