"""The decision-score histograms of Decisions 1 to 3 use 16 equal-width bins (research
author's request, 4 November 2026: "change all relevant distribution graphs to 16 bins
so it would be easier to compare them to Stata output"). 16 is what Stata's default
rule gives for the 280 participants (test_rtd_histogram_stata_parity.py pins the rule
for Decision 4).

Pinned here: the binning helper (16 bins from min to max, Stata / np.histogram closure,
counts summing to N, the degenerate cases), and that no Decision 1-3 chart module
binning is left to plotly (no nbins / nbinsx in those modules).
"""
import ast
import os

import numpy as np
import pandas as pd

from app.reports.bins import STATA_BINS, fixed_bins
from app.reports.rtd import rtd_stata_bin_count

from conftest import REPO

CHART_MODULES = [
    os.path.join(REPO, "app", "pages", "results", "overview.py"),
    os.path.join(REPO, "app", "pages", "results", "visualizations", "disclosure_viz.py"),
    os.path.join(REPO, "app", "pages", "results", "visualizations", "donation_viz.py"),
]


def test_sixteen_is_statas_count_for_the_280_participants():
    assert STATA_BINS == 16 == rtd_stata_bin_count(280)


def test_fixed_bins_are_sixteen_equal_width_bins_from_min_to_max():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=1000))
    edges, counts, size = fixed_bins(x)
    assert len(edges) == STATA_BINS + 1 and len(counts) == STATA_BINS
    assert edges[0] == x.min() and edges[-1] >= x.max()
    assert np.allclose(np.diff(edges), size)
    assert counts.sum() == len(x), "the last bin must include the maximum (Stata closure)"
    assert counts[-1] >= 1


def test_fixed_bins_match_np_histogram_with_sixteen_bins():
    x = np.linspace(-2.0, 3.0, 281)
    edges, counts, _ = fixed_bins(x)
    ref_counts, ref_edges = np.histogram(x, bins=16)
    assert np.array_equal(counts, ref_counts)
    assert np.allclose(edges, ref_edges)


def test_fixed_bins_degenerate_series():
    edges, counts, size = fixed_bins(pd.Series([0.25, 0.25, 0.25]))
    assert len(counts) == 1 and counts[0] == 3 and size == 1.0
    edges, counts, size = fixed_bins(pd.Series([], dtype=float))
    assert len(counts) == 1 and counts[0] == 0
    _, counts, _ = fixed_bins(pd.Series([1.0, np.nan, 2.0]))
    assert counts.sum() == 2


def test_no_chart_module_leaves_binning_to_plotly():
    """No px.histogram(nbins=...) / go.Histogram(nbinsx=...) in the Decision 1-3 chart
    modules: every score histogram goes through score_histogram_bar (fixed_bins)."""
    for path in CHART_MODULES:
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    assert kw.arg not in ("nbins", "nbinsx"), f"{os.path.relpath(path, REPO)}: {kw.arg}= at line {node.lineno}"
