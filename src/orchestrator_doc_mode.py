# src/orchestrator_doc_mode.py
"""Research Specification (documentation) mode - the unified Engine bound to PROFILES['documentation']."""

from src.engine.core import Engine, CONFIG_PATH, SIMULATION_CONFIG_PATH  # noqa: F401
from src.engine.profile import PROFILES


class OrchestratorDocMode(Engine):
    """
    Research Specification mode: the 280 original participants with the modelled
    decisions' stochastic component gated by ``pop_context='documentation'``.

    When no ``agents_df`` is supplied, the agents are the 280 participants in file
    order, cycling (agent k = participant k mod 280; ruling R-CYC).
    """

    def __init__(self):
        super().__init__(PROFILES['documentation'])


__all__ = ["OrchestratorDocMode"]
