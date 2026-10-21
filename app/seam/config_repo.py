# app/seam/config_repo.py
"""
Read-only access to ``config/decisions.yaml`` and ``config/simulation.yaml``.

The loader, the two path constants and :class:`DecisionsConfig` live in
:mod:`src.config` (the engine loads through the same module); this module
re-exports them and holds the process-wide repo the app uses.

The app never writes these files: the tabs keep their overrides in session
keys and the plan builder reads the file only for the values the session does
not carry (the donation coefficient sets when a session was never initialised,
the sigma constants, the adjustment shift default). The loaded dicts are
cached per instance and must be treated as immutable;
:meth:`DecisionsConfig.fresh_decisions_dict` hands out a deep copy for the
engine to patch.
"""

from typing import Optional

from src.config import (  # noqa: F401 - re-exported
    DEFAULT_DECISIONS_PATH,
    DEFAULT_SIMULATION_PATH,
    PROJECT_ROOT,
    RTD_LEVELS,
    DecisionsConfig,
    read_yaml_file,
    rtd_sigma_coefficient_defaults,
)

_DEFAULT_REPO: Optional[DecisionsConfig] = None


def get_config_repo() -> DecisionsConfig:
    """The process-wide repo for the project's own config files (loaded on first use)."""
    global _DEFAULT_REPO
    if _DEFAULT_REPO is None:
        _DEFAULT_REPO = DecisionsConfig.load()
    return _DEFAULT_REPO


__all__ = [
    "DEFAULT_DECISIONS_PATH",
    "DEFAULT_SIMULATION_PATH",
    "DecisionsConfig",
    "get_config_repo",
    "read_yaml_file",
    "rtd_sigma_coefficient_defaults",
]
