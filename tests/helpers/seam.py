"""The session snapshot the run-plan seam starts from, and its plan.

``base_state`` is a session snapshot dict with the app's defaults for every key the seam
(``app/seam/build_plan.py``) reads; ``plan_for`` builds its run plan; ``ALL`` is the
decision order. tests/test_golden_runs.py starts every scenario from ``base_state``.
"""
from types import SimpleNamespace

from app.seam import build_plan as bp
from src.engine.core import DECISION_ORDER

ALL = list(DECISION_ORDER)


def _sim_params():
    from app.models import SimulationParameters
    return SimulationParameters()


def _donation_keys(repo):
    """The donation_coeff_*_{cat,cont} keys exactly as app.models.load_coefficient_set writes them."""
    keys = {}
    for suffix, mode in (("cat", "categorical"), ("cont", "continuous")):
        c = repo.donation_coefficient_set(mode)
        keys[f"donation_coeff_intercept_{suffix}"] = c["intercept"]
        keys[f"donation_coeff_hh_{suffix}"] = c["beta_hh"]
        keys[f"donation_coeff_linear_{suffix}"] = c.get("beta_income_linear", 0.0)
        for stem, label in (("midsub", "MidSub"), ("nosub", "NoSub"), ("fullsub", "FullSub")):
            keys[f"donation_coeff_{stem}_{suffix}"] = c["beta_group"][label]
        q = c.get("beta_income_q", {})
        for stem, label in (("q1", "Q1"), ("q2", "Q2"), ("q3", "Q3"), ("q4", "Q4"), ("q5", "Q5")):
            keys[f"donation_coeff_{stem}_{suffix}"] = q.get(label, 0.0)
        for stem, label in (("incoming", "Incoming"), ("law", "Law5yr"), ("ug", "UG3yr"), ("grad", "Grad2yr")):
            keys[f"donation_coeff_{stem}_{suffix}"] = c["beta_study"][label]
    return keys


def _unsuffixed_donation_keys(repo, mode):
    """The unsuffixed donation_coeff_* keys (what the original get_current_coefficients read)."""
    suffix = "cont" if mode == "continuous" else "cat"
    return {k[:-len(suffix) - 1]: v for k, v in _donation_keys(repo).items() if k.endswith("_" + suffix)}


def base_state(repo, selected=None, **overrides):
    """A session snapshot dict with the app's defaults for every key the seam reads."""
    state = {
        "sim_params": _sim_params(),
        "decision_params": SimpleNamespace(selected_decisions=list(selected if selected is not None else ALL)),
        "population_mode": "Copula (synthetic)",
        "income_spec_mode": "categorical only",
        "seed_input": 42, "seed": 42, "n_agents": 1000, "base_seed": 7,
        # run_individual_decision sets ([decision], []); run_combined_simulation (custom, the rest)
        "custom_decisions": list(selected if selected is not None else ALL),
        "default_decisions": ([] if selected is not None and len(selected) == 1
                              else [d for d in ALL if d not in (selected if selected is not None else ALL)]),
        # donation tab
        "sigma_in_copula": False, "sigma_in_research": True, "sigma_coefficient": 1.0,
        "sigma_value_ui": 9.8995, "anchor_observed_weight": 0.75,
        "donation_sigma_strategy": "overall",
        "donation_quintile_scale_factors": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
        "donation_adjustment_shift": repo.donation_adjustment_shift(),
        # DI tab
        "di_income_mode": "Categorical only", "di_intercept": 0.75, "di_wopb": 0.25, "di_wpb": 0.5,
        "di_sigma_enabled": True, "di_sigma_in_copula": False, "di_sigma_strategy": "overall",
        "di_scale_factor": 1.0, "di_quintile_scale_factors": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
        # DD tab
        "dd_income_mode": "Categorical only", "dd_intercept": -0.75,
        "dd_sigma_enabled": True, "dd_sigma_in_copula": False, "dd_sigma_strategy": "overall",
        "dd_scale_factor": 1.0, "dd_quintile_scale_factors": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
        # RTD tab
        "rtd_income_mode": "Continuous only", "rtd_sigma_enabled": True, "rtd_sigma_in_copula": False,
        "rtd_sigma_strategy": "overall", "rtd_scale_factor": 1.0,
        "rtd_quintile_scale_factors": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
        "rtd_intercept_ttp": 0.0, "rtd_intercept_loyalty": 0.0, "rtd_intercept_wtp": 0.0,
        "rtd_intercept_risk_taking": 0.0, "rtd_intercept_flexibility": 0.0,
        "rtd_flex_observed_weight": 0.25, "rtd_aggregation_enabled": True,
    }
    state.update(_donation_keys(repo))
    state.update(_unsuffixed_donation_keys(repo, "categorical"))
    state.update(overrides)
    return state


def plan_for(repo, default_values, state):
    return bp.build_run_plan(state, repo, default_decision_values=default_values)
