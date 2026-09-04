# src/orchestrator.py
"""Copula (synthetic population) mode - the unified Engine bound to PROFILES['copula']."""

from src.engine.core import Engine, CONFIG_PATH, SIMULATION_CONFIG_PATH  # noqa: F401
from src.engine.profile import PROFILES


class Orchestrator(Engine):
    """
    Coordinates trait sampling and decision execution for the copula mode.

    Agents come from ``TraitEngine.sample(n_agents, seed)`` unless a pre-sampled
    ``agents_df`` is supplied. Everything else is the shared engine loop.
    """

    def __init__(self):
        super().__init__(PROFILES['copula'])


__all__ = ["Orchestrator"]
