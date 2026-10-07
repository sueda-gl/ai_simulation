# src/engine/profile.py
"""
Per-mode profiles for the unified engine.

After the owner rulings (one 13-decision order, one row-building mechanism, one
donation module, errors always raise) the three population modes differ only in:

* ``pop_context`` - the string handed to the four modelled decisions
  (donation_default, disclose_income, disclose_documents,
  rejected_transaction_defaults); it is what gates the stochastic component.
* where agents come from (``agent_source``, ``random_sample``): the copula draws
  ``TraitEngine.sample(n, seed)``; the research modes take the 280 original
  participants - Research Specification RANDOMLY (seeded: a subset without
  replacement for N < 280, all 280 for N = 280, a bootstrap for N > 280), Research
  Baseline in file order, cycling (agent k = participant k mod 280); ruling R-CYC as
  clarified by the owner 2026-10-07, see src/engine/sampling.py.
* ``force_donation_sigma_zero`` - Research Baseline never adds noise: the
  donation_default params are shallow-copied with ``stochastic.sigma_value = 0.0``.
* the console log prefix.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModeProfile:
    name: str
    pop_context: str
    agent_source: str               # 'copula' | 'research'
    random_sample: bool             # research only: Specification True, Baseline False
    force_donation_sigma_zero: bool
    log_prefix: str

    @property
    def is_research(self) -> bool:
        return self.agent_source == "research"


PROFILES: Dict[str, ModeProfile] = {
    "copula": ModeProfile(
        name="copula",
        pop_context="copula",
        agent_source="copula",
        random_sample=False,
        force_donation_sigma_zero=False,
        log_prefix="[Copula]",
    ),
    "documentation": ModeProfile(
        name="documentation",
        pop_context="documentation",
        agent_source="research",
        random_sample=True,
        force_donation_sigma_zero=False,
        log_prefix="[DocMode]",
    ),
    "baseline": ModeProfile(
        name="baseline",
        pop_context="baseline",
        agent_source="research",
        random_sample=False,
        force_donation_sigma_zero=True,
        log_prefix="[Baseline]",
    ),
}

__all__ = ["ModeProfile", "PROFILES"]
