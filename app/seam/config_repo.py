# app/seam/config_repo.py
"""
Read-only access to ``config/decisions.yaml`` and ``config/simulation.yaml``.

The app never writes these files again (ruling R11): the tabs keep their
overrides in session keys and the plan builder reads the file only for the
values the session does not carry (the donation coefficient sets when a session
was never initialised, the sigma constants, the adjustment shift default).

The loaded dicts are cached per instance and must be treated as immutable;
:meth:`DecisionsConfig.fresh_decisions_dict` hands out a deep copy for the
engine to patch.
"""

import copy
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECISIONS_PATH = PROJECT_ROOT / "config" / "decisions.yaml"
DEFAULT_SIMULATION_PATH = PROJECT_ROOT / "config" / "simulation.yaml"


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
        with open(decisions_path, "r") as f:
            decisions = yaml.safe_load(f) or {}
        with open(simulation_path, "r") as f:
            simulation = yaml.safe_load(f) or {}
        return cls(decisions, simulation, decisions_path, simulation_path)

    # ------------------------------------------------------------- raw dicts

    def decisions_dict(self) -> dict:
        """The cached decisions.yaml dict (read-only by convention)."""
        return self._decisions

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
        """donation_default.stochastic.sigma_overall (9.899547) - the ONE donation sigma (R12)."""
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
        block when the file has one, else the legacy ``regression`` block.
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


_DEFAULT_REPO: Optional[DecisionsConfig] = None


def get_config_repo() -> DecisionsConfig:
    """The process-wide repo for the project's own config files (loaded on first use)."""
    global _DEFAULT_REPO
    if _DEFAULT_REPO is None:
        _DEFAULT_REPO = DecisionsConfig.load()
    return _DEFAULT_REPO


def reset_config_repo() -> None:
    """Drop the cached repo (tests that swap the config files)."""
    global _DEFAULT_REPO
    _DEFAULT_REPO = None


__all__ = [
    "DEFAULT_DECISIONS_PATH",
    "DEFAULT_SIMULATION_PATH",
    "DecisionsConfig",
    "get_config_repo",
    "reset_config_repo",
]
