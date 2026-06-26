# src/build_dd_sigma.py
"""
Derive the Decision 2 (Disclose Documents) stochastic sigma from the raw experiment data.

The stochastic component of Disclose Documents draws ``Normal(dd_deterministic, sigma)``, where
``sigma`` is the standard deviation of ``constranssospeso2periods01`` — the per-participant
two-period consumption of community donations (sospeso) and private transfers, normalised to
[0, 1]. Those sigma constants are stored (frozen) in ``config/decisions.yaml`` under
``disclose_documents.stochastic`` (``sigma_overall`` + ``sigma_quintile``).

This module RECONSTRUCTS that variable bit-for-bit from the two raw per-period experiment files,
reproducing the professor's ``consumedtransferssospeso2periods`` column in
``Stata_File_Decision 2_260326 - Final.dta`` (280/280 participants), and derives the sigma
constants from it. Run it to verify ``decisions.yaml`` is faithful, or import ``derive_dd_sigma``
to recompute from raw data.

Source columns (codebook ``decisions.pdf``):
    "Claim (consume) a Sospeso someone else paid for" -> ``Number of Sospeso Concumed``
    "Consume a received transfer"                     -> ``Number of Transfers Consumed``

Reconstruction recipe:
    consumed[i] = (Sospeso Consumed + Transfers Consumed) from Period 1   [master row order]
                + (Sospeso Consumed + Transfers Consumed) from Period 2   [aligned by ROW POSITION]

*** PARTICIPANT-ID MERGE CAVEAT (important) ***
    Period 2 is merged by ROW POSITION (the first ``len(period2)`` master rows; the 11 participants
    inactive in Period 2 contribute 0), NOT by Participant ID. The Period 2 export's
    ``Participant ID`` column is offset relative to its data rows, so joining the two periods by
    Participant ID produces 89/280 per-participant mismatches (the total, 351, is unchanged — the
    same 269 values are simply attributed to adjacent participants). The professor's Stata merge
    matched by position, so we replicate the positional merge to reproduce his validated
    ``disclosedoc_*`` output. See ``notes.md`` for the full forensic trail.

Note: the repo's combined ``Student Experiment Results - Period 1-2.xlsx`` only carries Period-1
consumption (its consume columns sum to 268, identical to the standalone Period 1 file). The true
two-period total (351) therefore REQUIRES ``Student Experiment Results - Period 2.xlsx`` as well.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# The combined file supplies period-1 consumption, the Assigned Allowance Level, and the master
# participant row order. The Period 2 file supplies period-2 consumption.
PERIOD12_PATH = DATA_DIR / "Student Experiment Results - Period 1-2.xlsx"
PERIOD2_PATH = DATA_DIR / "Student Experiment Results - Period 2.xlsx"

SOSPESO_COL = "Number of Sospeso Concumed"   # original (mis)spelling in the experiment export
TRANSFER_COL = "Number of Transfers Consumed"
LEVEL_COL = "Assigned Allowance Level"

# Known-good targets from the professor's Stata_File_Decision 2_260326 - Final.dta, for self-check.
GOLD = {"sum": 351, "max": 13, "nonzero": 125, "sigma_overall": 0.1606568355}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df = df[df["Participant ID"].notna()].copy()
    df["Participant ID"] = df["Participant ID"].astype(int)
    return df.reset_index(drop=True)


def _consumed(df: pd.DataFrame) -> np.ndarray:
    """Per-row count of sospeso + transfers consumed."""
    s = pd.to_numeric(df.get(SOSPESO_COL), errors="coerce").fillna(0)
    t = pd.to_numeric(df.get(TRANSFER_COL), errors="coerce").fillna(0)
    return (s + t).astype(int).to_numpy()


def reconstruct_consumed_2periods(period1_df: pd.DataFrame, period2_df: pd.DataFrame) -> np.ndarray:
    """Reconstruct the professor's ``consumedtransferssospeso2periods`` (one int per participant).

    Period 2 is aligned by ROW POSITION, not by Participant ID — see the module docstring's
    PARTICIPANT-ID MERGE CAVEAT. ``period1_df`` defines the master order.
    """
    c1 = _consumed(period1_df)
    c2_vals = _consumed(period2_df)
    if len(c2_vals) > len(c1):
        raise ValueError(
            f"Period 2 has more rows ({len(c2_vals)}) than Period 1 ({len(c1)}); "
            "positional merge expects Period 2 to be a subset."
        )
    c2 = np.zeros(len(c1), dtype=int)
    c2[: len(c2_vals)] = c2_vals
    return c1 + c2


def derive_dd_sigma(
    period12_path: Path = PERIOD12_PATH,
    period2_path: Path = PERIOD2_PATH,
) -> dict:
    """Derive the Disclose-Documents stochastic sigma constants from the raw experiment files.

    Returns ``{consumed, sigma_overall, sigma_quintile{1..5}}`` where ``sigma`` values are the
    standard deviation (ddof=1) of ``consumed / max(consumed)`` — overall and within each
    allowance level. Mirrors the Stata ``egen ... = sd(constranssospeso2periods01)`` derivation.
    """
    master = _load(period12_path)
    period2 = _load(period2_path)
    consumed = reconstruct_consumed_2periods(master, period2)

    consumed01 = consumed / consumed.max()
    sigma_overall = float(np.std(consumed01, ddof=1))

    levels = pd.to_numeric(master[LEVEL_COL], errors="coerce").astype(int).to_numpy()
    sigma_quintile = {L: float(np.std(consumed01[levels == L], ddof=1)) for L in (1, 2, 3, 4, 5)}

    return {"consumed": consumed, "sigma_overall": sigma_overall, "sigma_quintile": sigma_quintile}


if __name__ == "__main__":
    out = derive_dd_sigma()
    c = out["consumed"]
    n_sum, n_max, n_nz = int(c.sum()), int(c.max()), int((c > 0).sum())
    print(
        f"consumedtransferssospeso2periods: sum={n_sum} max={n_max} nonzero={n_nz}  "
        f"(gold {GOLD['sum']}/{GOLD['max']}/{GOLD['nonzero']})"
    )
    assert (n_sum, n_max, n_nz) == (GOLD["sum"], GOLD["max"], GOLD["nonzero"]), (
        "Reconstruction does not match the professor's gold variable — check the period files / merge."
    )

    print(f"\nsigma_overall = {out['sigma_overall']:.10f}  (decisions.yaml: {GOLD['sigma_overall']})")
    for L, s in out["sigma_quintile"].items():
        print(f"  sigma_quintile[{L}] = {s:.10f}")
    assert abs(out["sigma_overall"] - GOLD["sigma_overall"]) < 1e-6, "sigma_overall drift vs decisions.yaml!"

    print("\n[OK] Reconstruction reproduces the gold variable and sigma (decisions.yaml is faithful).")
