# src/config.py
"""
The one loader of ``config/decisions.yaml`` and ``config/simulation.yaml``.

* :data:`DEFAULT_DECISIONS_PATH` / :data:`DEFAULT_SIMULATION_PATH` - the two files.
* :func:`read_yaml_file` - ``yaml.safe_load`` of a file, cached per
  (path, mtime, size), always handing out a deep copy.
* :class:`DecisionsConfig` - the two files loaded once, read-only, with the
  derived constants the seam needs (sigma constants, donation coefficient sets,
  Decision 4's sigma coefficient defaults).

The engine (``src/engine/core.py``) and the app (``app/seam/config_repo.py``,
which re-exports everything here) both load through this module, so one file is
the only source of every constant. Nothing here imports Streamlit or ``app``.
"""

import copy
import functools
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISIONS_PATH = PROJECT_ROOT / "config" / "decisions.yaml"
DEFAULT_SIMULATION_PATH = PROJECT_ROOT / "config" / "simulation.yaml"


@functools.lru_cache(maxsize=16)
def _parse_yaml_file(path: str, mtime_ns: int, size: int):
    """The parsed file, cached per (path, modification time, size)."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_yaml_file(path) -> dict:
    """``yaml.safe_load(open(path))`` without re-parsing an unchanged file.

    The Page-2 tabs read config/decisions.yaml (a ~60 ms parse) on every script
    run - fourteen times per rerun with Decisions 1-4 selected, ~80 % of a rerun
    (rapid +/- clicks on a Decision 4 Override value froze the page). The
    parse is cached per (path, mtime, size), so an edited file is
    re-read, and every call returns a DEEP COPY, so callers see exactly what a
    fresh load returned and may mutate it freely.
    """
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return copy.deepcopy(_parse_yaml_file(str(resolved), stat.st_mtime_ns, stat.st_size))


class DecisionsConfig:
    """The two configuration files, loaded once, never written."""

    def __init__(self, decisions: dict, simulation: dict,
                 decisions_path: Optional[Path] = None,
                 simulation_path: Optional[Path] = None):
        self._decisions = decisions
        self._simulation = simulation
        self.decisions_path = decisions_path
        self.simulation_path = simulation_path

    @classmethod
    def load(cls, decisions_path: Optional[Path] = None,
             simulation_path: Optional[Path] = None) -> "DecisionsConfig":
        decisions_path = Path(decisions_path or DEFAULT_DECISIONS_PATH)
        simulation_path = Path(simulation_path or DEFAULT_SIMULATION_PATH)
        decisions = read_yaml_file(decisions_path) or {}
        simulation = read_yaml_file(simulation_path) or {}
        return cls(decisions, simulation, decisions_path, simulation_path)

    # ------------------------------------------------------------- raw dicts

    def fresh_decisions_dict(self) -> dict:
        """A deep copy of decisions.yaml for one engine run to patch and mutate."""
        return copy.deepcopy(self._decisions)

    def simulation_dict(self) -> dict:
        """The cached simulation.yaml dict (read-only by convention)."""
        return self._simulation

    def decision(self, name: str) -> dict:
        """One decision's block (read-only by convention); ``{}`` when absent."""
        block = self._decisions.get(name)
        return block if isinstance(block, dict) else {}

    # -------------------------------------------------------- sigma constants

    def _sigma_overall(self, decision: str) -> float:
        stochastic = self.decision(decision).get("stochastic") or {}
        if "sigma_overall" not in stochastic:
            raise KeyError(f"{decision}.stochastic.sigma_overall missing from {self.decisions_path}")
        return float(stochastic["sigma_overall"])

    def donation_sigma_overall(self) -> float:
        """donation_default.stochastic.sigma_overall (9.899547) - the ONE donation sigma constant (never a literal in app/)."""
        return self._sigma_overall("donation_default")

    def disclose_income_sigma_overall(self) -> float:
        return self._sigma_overall("disclose_income")

    def disclose_documents_sigma_overall(self) -> float:
        return self._sigma_overall("disclose_documents")

    # ------------------------------------------------------ donation defaults

    def donation_coefficient_set(self, income_mode: str) -> dict:
        """
        Deep copy of the file's coefficient set for ``income_mode``
        ('categorical' / 'continuous'): the nested ``regression_coefficients``
        block when the file has one, else the older single ``regression`` block.
        """
        donation = self.decision("donation_default")
        regression_coeffs = donation.get("regression_coefficients") or {}
        if "categorical" in regression_coeffs and "continuous" in regression_coeffs:
            block = regression_coeffs["continuous" if income_mode == "continuous" else "categorical"]
        else:
            block = regression_coeffs if regression_coeffs else donation.get("regression", {})
        block = {k: v for k, v in (block or {}).items() if k != "income_mode"}
        return copy.deepcopy(block)

    def donation_adjustment_shift(self) -> float:
        adjustment = self.decision("donation_default").get("adjustment") or {}
        return float(adjustment.get("shift_value", 0.0))

    # ---------------------------------------------------- Decision 4 defaults

    def rtd_sigma_coefficient_defaults(self) -> Tuple[float, Dict[str, float]]:
        """Decision 4's research-default σ coefficients (see the module function)."""
        return rtd_sigma_coefficient_defaults(self.decision("rejected_transaction_defaults"))


RTD_LEVELS = ("1", "2", "3", "4", "5")


def rtd_sigma_coefficient_defaults(rtd_block: Mapping) -> Tuple[float, Dict[str, float]]:
    """
    Decision 4's research-default σ coefficients: ``(overall, {level: coefficient})``,
    read from ``rejected_transaction_defaults.stochastic.mechanisms.*.scale_factor`` /
    ``quintile_scale_factors`` (0.5 for every element and budget level, per the
    platform documentation, docs/reference/COOPECON AI Agent Documentation 2026-10-07.docx; previously 1.0).

    The file carries the coefficients per element, but the tab applies ONE
    decision-wide set to all five elements, so the five must agree - a file in
    which they differ is rejected instead of silently picking one. An element
    without the keys falls back as the engine does (``scale_factor`` 1.0,
    quintile factors = ``scale_factor``).
    """
    mechanisms = ((rtd_block or {}).get("stochastic") or {}).get("mechanisms") or {}
    sets = {}
    for name, mech in mechanisms.items():
        mech = mech or {}
        scale = float(mech.get("scale_factor", 1.0))
        quintiles = mech.get("quintile_scale_factors") or {}
        sets[name] = (scale, {lvl: float(quintiles.get(lvl, quintiles.get(int(lvl), scale)))
                              for lvl in RTD_LEVELS})
    if not sets:
        return 1.0, {lvl: 1.0 for lvl in RTD_LEVELS}
    distinct = {repr(v) for v in sets.values()}
    if len(distinct) > 1:
        raise ValueError("rejected_transaction_defaults: the five elements' scale_factor / "
                         f"quintile_scale_factors must agree (decision-wide σ): {sets}")
    scale, quintiles = next(iter(sets.values()))
    return scale, dict(quintiles)


__all__ = [
    "DEFAULT_DECISIONS_PATH",
    "DEFAULT_SIMULATION_PATH",
    "PROJECT_ROOT",
    "RTD_LEVELS",
    "DecisionsConfig",
    "read_yaml_file",
    "rtd_sigma_coefficient_defaults",
]
