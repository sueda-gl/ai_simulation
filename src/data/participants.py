# src/data/participants.py
"""
Lazy, cached access to the 280-participant master dataset.

This module is the single place that reads the two Excel workbooks and merges
them.  It reproduces exactly what the former ``src/validate_traits.py`` and
``src/build_master_traits.py`` did, with two differences that cannot change any
number:

* nothing happens at import time — the Excel files are read on the first call
  to :func:`merged` (or :func:`original_data`) and cached afterwards;
* the "missing columns" check raises instead of printing and calling
  ``sys.exit(1)``.

Two per-participant columns from the reference Stata files are merged in by
``Participant ID``:

* ``stdactions`` (``data/stata_stdactions.csv``) - the SD in the number of actions
  per cycle over the eight experiment cycle-weeks. The experiment workbook only
  carries per-participant totals, so the value comes from
  ``Stata_File_Decision4_050926.dta``. It is a trait (Decision 4 Flexibility,
  ``config/trait_requirements.yaml``) and therefore also a copula column.
* ``income`` is NOT merged. Every population mode draws each agent's continuous
  income from the Page 1 distribution, within the band of the agent's allowance level
  (``src.engine.income.get_agent_income``, seeded per agent), as the deployed app
  always did. ``data/stata_incomes.csv`` is the one fixed income per participant that
  the reference Stata files embed; the parity tests inject it themselves. From 6 to
  24 October 2026 the loader merged it into the research populations, which made
  Page 1's income settings inert for the Research modes; reverted.

The merge changes neither the row order or the 280-row selection (``dropna`` stays
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
# Per-participant SD in actions per cycle (Decision 4 Flexibility; Stata_File_Decision4_050926.dta).
STDACTIONS_PATH = _ROOT / "data" / "stata_stdactions.csv"
# The fixed per-participant continuous income the reference Decision 1 / 2 / 4 .dta files
# embed. Read by the parity tests only; never merged into the population.
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

        missing = [c for c in traits if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in merged participant data: {missing}")

        _MERGED = frame
    return _MERGED


def original_data() -> pd.DataFrame:
    """
    The master traits for the 280 participants, ``dropna``-ed on the trait columns
    (the same 280-row selection and order as the original orchestrators'
    ``merged[traits].copy().dropna()``), with no income column: the engine generates
    each participant's income from the Page 1 distribution.

    A fresh copy is returned on every call, so a caller mutating its frame cannot
    affect anyone else.
    """
    global _ORIGINAL_DATA
    if _ORIGINAL_DATA is None:
        traits = get_master_trait_list()
        frame = merged()
        _ORIGINAL_DATA = frame[traits].copy().dropna(subset=traits)
    return _ORIGINAL_DATA.copy()
