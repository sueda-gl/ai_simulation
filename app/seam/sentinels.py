# app/seam/sentinels.py
"""
The sigma constants the seam hands to the engine - read from
``config/decisions.yaml``, never spelled as literals in ``app/`` (ruling R12).

* donation_default: ``sigma_value = donation_sigma_overall() * sigma_coefficient``
  is the actual sigma the engine draws with (overall / continuous strategy).
* disclose_income / disclose_documents: ``sigma_value`` is only an ENABLE
  sentinel (the engine gates on ``sigma_value > 0`` in Research Specification
  mode and takes the magnitude from its own ``sigma_overall * scale_factor``);
  the seam writes the file's ``sigma_overall`` there, exactly the value the
  legacy code wrote as a literal.
* rejected_transaction_defaults: ``RTD_SIGMA_ENABLE`` (1.0) is a pure enable
  sentinel; each mechanism's sigma lives in the config.
"""

from typing import Optional

from app.seam.config_repo import DecisionsConfig, get_config_repo

RTD_SIGMA_ENABLE: float = 1.0


def _repo(config_repo: Optional[DecisionsConfig]) -> DecisionsConfig:
    return config_repo if config_repo is not None else get_config_repo()


def donation_sigma_overall(config_repo: Optional[DecisionsConfig] = None) -> float:
    """donation_default.stochastic.sigma_overall from the config (9.899547)."""
    return _repo(config_repo).donation_sigma_overall()


def disclose_income_sigma_sentinel(config_repo: Optional[DecisionsConfig] = None) -> float:
    """The >0 value written to disclose_income.stochastic.sigma_value when draws are on."""
    return _repo(config_repo).disclose_income_sigma_overall()


def disclose_documents_sigma_sentinel(config_repo: Optional[DecisionsConfig] = None) -> float:
    """The >0 value written to disclose_documents.stochastic.sigma_value when draws are on."""
    return _repo(config_repo).disclose_documents_sigma_overall()


__all__ = [
    "RTD_SIGMA_ENABLE",
    "donation_sigma_overall",
    "disclose_income_sigma_sentinel",
    "disclose_documents_sigma_sentinel",
]
