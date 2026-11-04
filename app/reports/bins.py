# app/reports/bins.py
"""The bins of the decision-score histograms (Decisions 1 to 3).

Every distribution chart of a decision score draws STATA_BINS = 16 equal-width bins
spanning min..max, the count Stata's default rule gives for the 280 participants (see
app.reports.rtd.rtd_stata_bin_count), so the app's charts can be compared bar for bar
with the Stata figures in the decision documents (research author's request,
4 November 2026). Decision 4 keeps its own rule (app.reports.rtd.rtd_stata_bins), which
yields the same 16 bins at every N >= 256.

Bin semantics are Stata's and np.histogram's: every bin is closed on the left and open
on the right except the LAST, which also includes the maximum, so the counts always sum
to N. A bin no agent falls into has a count of 0 (Stata draws no bar there).
"""
import numpy as np
import pandas as pd

STATA_BINS = 16


def fixed_bins(series, k=STATA_BINS):
    """(edges, counts, size) of the k equal-width bins of ``series`` from its minimum to
    its maximum; ``size`` is the bin width. A constant or empty series gets one
    unit-wide bin holding everything."""
    s = pd.Series(series).dropna().astype(float)
    n = len(s)
    vmin, vmax = (float(s.min()), float(s.max())) if n else (0.0, 0.0)
    if not n or vmax <= vmin:
        return np.array([vmin, vmin + 1.0]), np.array([n]), 1.0
    size = (vmax - vmin) / k
    edges = vmin + size * np.arange(k + 1, dtype=float)
    edges[-1] = max(edges[-1], vmax)
    counts = np.histogram(s.to_numpy(), bins=edges)[0]
    return edges, counts, size


__all__ = ["STATA_BINS", "fixed_bins"]
