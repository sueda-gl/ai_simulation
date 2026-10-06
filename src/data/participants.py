# src/data/participants.py
"""
Lazy, cached access to the 280-participant master dataset.

This module is the single place that reads the two Excel workbooks and merges
them.  It reproduces exactly what ``src/validate_traits.py`` and
``src/build_master_traits.py`` do today, with two differences that are required
by the migration and that cannot change any number:

* nothing happens at import time — the Excel files are read on the first call
  to :func:`merged` (or :func:`original_data`) and cached afterwards;
* the "missing columns" check raises instead of printing and calling
  ``sys.exit(1)``.

Two per-participant columns from the professor's Stata files are merged in by
``Participant ID`` (ported 2026-10-06 from the original repo's Sep-2026 change to
``src/validate_traits.py``, which this module replaces):

* ``stdactions`` (``data/stata_stdactions.csv``) - the SD in the number of actions
  per cycle over the eight experiment cycle-weeks. The experiment workbook only
  carries per-participant totals, so the value comes from
  ``Stata_File_Decision4_050926.dta``. It is a trait (Decision 4 Flexibility,
  ``config/trait_requirements.yaml``) and therefore also a copula column.
* ``income`` (``data/stata_incomes.csv``) - the professor's ONE fixed continuous
  income per participant, identical across his Decision 1 / 2 / 4 Stata files.
  :func:`original_data` carries it next to the traits, so the Research modes find
  it already cached in ``get_agent_income()`` and reproduce his files instead of
  drawing a fresh income. Copula agents have no Participant ID and keep the
  Page-1 generated income.

Neither merge changes the row order or the 280-row selection (``dropna`` stays
restricted to the trait columns).

Row order, column selection, merge semantics and ``dropna`` are identical to the
module-level ``src.validate_traits.merged`` / ``OrchestratorBaseline.original_data``
they replace, so Research-mode agent order (and therefore every RNG stream that
depends on it) is unchanged.
"""

from pathlib import Path
from typing import List

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]

# Same paths as src/validate_traits.py:6-7 and src/build_master_traits.py:5
SURVEY_PATH = _ROOT / "data" / "Student Survey Results - Period 1.xlsx"
EXPERIMENT_PATH = _ROOT / "data" / "Student Experiment Results - Period 1-2.xlsx"
REQ_PATH = _ROOT / "config" / "trait_requirements.yaml"
# Per-participant SD in actions per cycle (Decision 4 Flexibility; professor's .dta).
STDACTIONS_PATH = _ROOT / "data" / "stata_stdactions.csv"
# The professor's fixed per-participant continuous income (Decision 1 / 2 / 4 .dta).
INCOMES_PATH = _ROOT / "data" / "stata_incomes.csv"

# Caches (populated on first use)
_TRAITS = None          # tuple of str
_MERGED = None          # pd.DataFrame (the shared frame, as validate_traits.merged is today)
_ORIGINAL_DATA = None    # pd.DataFrame (master copy; callers get a fresh copy)


def get_master_trait_list() -> List[str]:
    """
    Sorted union of every trait listed in config/trait_requirements.yaml.

    Same logic as src/build_master_traits.get_master_trait_list (:7-17); the
    parsed result is cached, and a fresh list is returned each call so callers
    cannot corrupt the cache.
    """
    global _TRAITS
    if _TRAITS is None:
        req = yaml.safe_load(REQ_PATH.read_text())
        traits = set()
        for decision, cols in req.items():
            for col in cols:
                if not isinstance(col, str):
                    raise ValueError(f"{decision} has non-string entry: {col}")
                if col.strip().lower() == "placeholder_trait":
                    continue                 # ignore placeholders
                traits.add(col)
        _TRAITS = tuple(sorted(traits))
    return list(_TRAITS)


def merged() -> pd.DataFrame:
    """
    The survey x experiment inner merge — the same frame that
    ``src.validate_traits.merged`` built at import time (280 x 147), plus the columns merged from ``data/stata_stdactions.csv`` (``original_index``, ``assignedallowancelevel``, ``stdactions``) and ``income`` (280 x 151).

    Cached: the same DataFrame object is returned on every call, exactly as the
    module-level ``merged`` was shared by every importer.
    """
    global _MERGED
    if _MERGED is None:
        traits = get_master_trait_list()

        survey = pd.read_excel(SURVEY_PATH, sheet_name=0)
        experiment = pd.read_excel(EXPERIMENT_PATH, sheet_name=0)
        frame = survey.merge(experiment, on="Participant ID", how="inner",
                             suffixes=("_survey", "_experiment"))
        # Left merges keyed by Participant ID: row order and count are unchanged.
        if STDACTIONS_PATH.exists():
            stdactions = pd.read_csv(STDACTIONS_PATH)
            frame = frame.merge(stdactions, on="Participant ID", how="left")
        if INCOMES_PATH.exists():
            # float_precision="round_trip": pandas' default CSV float parser is not
            # correctly rounded and loses the last ULP on two of the 280 values.
            incomes = pd.read_csv(INCOMES_PATH, float_precision="round_trip")
            frame = frame.merge(incomes[["Participant ID", "income"]],
                                on="Participant ID", how="left")
            n_missing_income = int(frame["income"].isna().sum())
            if n_missing_income:
                print(f"⚠️  {n_missing_income}/{len(frame)} participants have no income in "
                      f"{INCOMES_PATH.name}; their income will be generated from the Page-1 "
                      "distribution instead.")

        missing = [c for c in traits if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in merged participant data: {missing}")

        _MERGED = frame
    return _MERGED


def original_data() -> pd.DataFrame:
    """
    The master traits for the 280 participants, plus the professor's fixed ``income``
    column when it is available, ``dropna``-ed on the TRAIT columns only — identical to
    the Sep-2026 original's ``merged[traits + ['income']].copy().dropna(subset=traits)``
    in OrchestratorBaseline / OrchestratorDocMode, so the 280-participant selection
    and order are unchanged and a missing income stays NaN (then generated).

    A fresh copy is returned on every call, mirroring the per-orchestrator copy
    of today, so a caller mutating its frame cannot affect anyone else.
    """
    global _ORIGINAL_DATA
    if _ORIGINAL_DATA is None:
        traits = get_master_trait_list()
        frame = merged()
        income_cols = ["income"] if "income" in frame.columns else []
        _ORIGINAL_DATA = frame[traits + income_cols].copy().dropna(subset=traits)
    return _ORIGINAL_DATA.copy()
