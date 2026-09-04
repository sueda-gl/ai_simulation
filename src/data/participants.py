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
    ``src.validate_traits.merged`` builds at import time today (280 x 147).

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

        missing = [c for c in traits if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in merged participant data: {missing}")

        _MERGED = frame
    return _MERGED


def original_data() -> pd.DataFrame:
    """
    The 13 master traits for the 280 participants, ``dropna``-ed — identical to
    ``merged[get_master_trait_list()].copy().dropna()`` as computed today by
    OrchestratorBaseline (:48), OrchestratorDocMode (:38) and orchestrator_depvar (:32).

    A fresh copy is returned on every call, mirroring the per-orchestrator copy
    of today, so a caller mutating its frame cannot affect anyone else.
    """
    global _ORIGINAL_DATA
    if _ORIGINAL_DATA is None:
        _ORIGINAL_DATA = merged()[get_master_trait_list()].copy().dropna()
    return _ORIGINAL_DATA.copy()
