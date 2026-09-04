# src/decisions/donation_default_stochastic.py
# Decision 3 has ONE module for every population mode (src/decisions/donation_default.py).
# This name is kept only because src/orchestrator_depvar.py still imports it.
from src.decisions.donation_default import donation_default as donation_default_stochastic  # noqa: F401
