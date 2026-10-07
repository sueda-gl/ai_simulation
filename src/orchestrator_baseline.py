# src/orchestrator_baseline.py
"""Research Baseline mode - the unified Engine bound to PROFILES['baseline']."""

from src.engine.core import Engine, CONFIG_PATH, SIMULATION_CONFIG_PATH  # noqa: F401
from src.engine.profile import PROFILES


class OrchestratorBaseline(Engine):
    """
    Research Baseline mode: the 280 original participants with NO stochastic
    component (``pop_context='baseline'``; donation_default sigma forced to 0).

    When no ``agents_df`` is supplied, the agents are the 280 participants in file
    order, cycling (agent k = participant k mod 280; ruling R-CYC).
    """

    def __init__(self):
        super().__init__(PROFILES['baseline'])


__all__ = ["OrchestratorBaseline"]
