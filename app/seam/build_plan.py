# app/seam/build_plan.py
"""
Pure plan builder: SessionSnapshot + DecisionsConfig -> RunPlan.

This is the former ``app/simulation.py`` resolution logic (seed / agent count,
the DI-only / DD-only / RTD-only / donation-only / combined branches, the
Compare-all and Compare-both result keys and sub-run order, every st.info /
caption / success text) plus the six ``_apply_*`` mappings rewritten as pure
patch functions.  It reads ONLY the snapshot and the config repo, writes
nothing, and never imports Streamlit.

Rulings applied on top of the verbatim port (each marked inline):
  R10  the Donation tab's flat coefficient set (donation_coeff_*_{cat,cont}
       session keys, loaded from the file) is what the engine runs - the
       nested categorical/continuous blocks are replaced, never merged onto.
  R11  intercepts / adjustment shift come from session keys, never the file.
  R12  donation sigma_value = config sigma_overall x the coefficient slider.
  R13  auto-implied saved configs do not exist (any lingering one is ignored).
  R14  an explicit saved config pins seed, n, population and that decision's
       parameters; its recorded output-column hash becomes a SavedExpectation
       that the complete run re-checks.
  R15  no session writes; settings are read once from the snapshot.
  R17  a saved Disclose Documents config is applied exactly like a saved
       Disclose Income config (same precedence).
  R18  di_/dd_/rtd_sigma_enabled default to True when the key is absent.
  R28  the DI-only / DD-only / RTD-only branches no longer rewrite
       income_spec_mode; the effective modes travel in RunMetadata.
"""

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.contract.defaults import DEFAULT_DECISION_VALUES
from src.contract.plan import (
    Replace,
    RunMetadata,
    RunPlan,
    SavedExpectation,
    SubRun,
    UiMessage,
)
from src.engine.core import DECISION_ORDER
from app.seam.config_repo import DecisionsConfig
from app.seam.sentinels import (
    RTD_SIGMA_ENABLE,
    disclose_documents_sigma_sentinel,
    disclose_income_sigma_sentinel,
    donation_sigma_overall,
)
from app.seam.snapshot import SessionSnapshot, take_snapshot

ALL_DECISIONS: Tuple[str, ...] = tuple(DECISION_ORDER)

AUTO_IMPLIED_SOURCE = "auto_implied_single_config"

# (result-key prefix, population) in Compare-all execution order
COMPARE_ALL_POPULATIONS: Tuple[Tuple[str, str], ...] = (
    ("copula", "copula"),
    ("research_spec", "documentation"),
    ("research_baseline", "baseline"),
)

POP_TYPE_BY_MODE: Dict[str, str] = {
    "Copula (synthetic)": "copula",
    "Research Specification": "documentation",
    "Research Baseline": "baseline",
}

RTD_MECHANISMS: Tuple[str, ...] = ("ttp", "loyalty", "wtp", "risk_taking", "flexibility")

# session-key stem -> coefficient location in the flat donation set
_DONATION_SCALAR_KEYS = {
    "intercept": "intercept",
    "hh": "beta_hh",
    "linear": "beta_income_linear",
}
_DONATION_GROUP_KEYS = {"midsub": "MidSub", "nosub": "NoSub", "fullsub": "FullSub"}
_DONATION_QUINTILE_KEYS = {"q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4", "q5": "Q5"}
_DONATION_STUDY_KEYS = {"incoming": "Incoming", "law": "Law5yr", "ug": "UG3yr", "grad": "Grad2yr"}

# SimulationParameters attributes copied 1:1 into simulation_config['simulation']
_SIMULATION_PARAM_ATTRS: Tuple[str, ...] = (
    # Income distribution parameters - CRITICAL for disclose_documents eligibility
    "income_distribution", "discount_income_threshold",
    # Lognormal parameters
    "lognormal_mu", "lognormal_sigma", "lognormal_min", "lognormal_max",
    # Generalised Gamma parameters
    "gg_k", "gg_c", "gg_lambda", "gg_min", "gg_max",
    # Dagum parameters
    "dagum_a", "dagum_p", "dagum_b", "dagum_min", "dagum_max",
    # Market parameters - used by bid_value and other decisions
    "market_price", "platform_markup", "price_range", "bidding_percentage", "num_vendors",
    # Vendor configuration parameters
    "vendor_config_mode", "vendor_price_source", "vendor_price_min", "vendor_price_max",
    "vendor_products_min", "vendor_products_max", "vendor_products_avg",
    # Vendor carryover parameters
    "vendor_carryover_probability", "override_carryover", "global_carryover",
)


# ----------------------------------------------------------------- helpers

def normalize_income_mode(mode: Any) -> str:
    """'continuous' if the string mentions it, else 'categorical'."""
    return "continuous" if "continuous" in str(mode).lower() else "categorical"


def is_compare_income_mode(mode: Any) -> bool:
    lowered = str(mode).lower()
    return "compare" in lowered or "both" in lowered


def get_pop_type(population_mode: str) -> str:
    """Map UI population mode name to internal type."""
    return POP_TYPE_BY_MODE.get(population_mode, "copula")


def get_population_mode_from_result_key(result_key: Optional[str]) -> Optional[str]:
    """
    Extract population mode from a result_key.

    Result keys from Compare All mode contain the population mode information:
    - "copula_categorical", "copula_continuous" -> "Copula (synthetic)"
    - "research_spec_categorical", "research_spec_continuous" -> "Research Specification"
    - "research_baseline_categorical", "research_baseline_continuous" -> "Research Baseline"

    Result keys from single mode runs ("categorical", "continuous") don't contain
    population mode info, so we return None to indicate the current mode should be kept.

    IMPORTANT: Return values MUST match the exact strings used by Page 1's radio button
    and all downstream comparisons (DI tab, DD tab, page2_decisions, etc.).
    """
    if not result_key:
        return None

    result_key_lower = result_key.lower()

    if "copula" in result_key_lower:
        return "Copula (synthetic)"
    elif "research_spec" in result_key_lower or "documentation" in result_key_lower:
        return "Research Specification"
    elif "baseline" in result_key_lower or "research_baseline" in result_key_lower:
        return "Research Baseline"

    # Single mode result keys like "categorical" or "continuous" don't indicate population mode
    return None


def explicit_saved_configs(snapshot: Mapping) -> Dict[str, dict]:
    """
    The saved-config store minus any auto-implied entry (R13: those no longer
    exist; one left over in an old session is ignored, never applied).
    """
    configs = snapshot.get("selected_decision_configs") or {}
    return {
        name: config for name, config in configs.items()
        if isinstance(config, dict) and config.get("source") != AUTO_IMPLIED_SOURCE
    }


def resolve_seed_and_n(snapshot: Mapping) -> Tuple[Any, Any, str]:
    """
    (seed, n_agents, source) - the same rule as
    ``app.state.saved_configs.get_simulation_seed_from_configs``: the first
    explicit saved config pins both (source 'configs'); otherwise Single Run
    uses seed_input -> seed -> 42 and Monte-Carlo base_seed_input -> base_seed
    -> 42, with n_agents from the session (default 1000) (source 'session_state').
    """
    configs = explicit_saved_configs(snapshot)
    if configs:
        first_config = next(iter(configs.values()))
        return first_config["original_seed"], first_config["original_n_agents"], "configs"

    sim_params = snapshot.get("sim_params")
    if sim_params is not None and getattr(sim_params, "simulation_mode", None) == "Single Run":
        seed = snapshot.get("seed_input", snapshot.get("seed", 42))
    else:
        seed = snapshot.get("base_seed_input", snapshot.get("base_seed", 42))

    return seed, snapshot.get("n_agents", 1000), "session_state"


def _selected_decisions(snapshot: Mapping) -> List[str]:
    decision_params = snapshot.get("decision_params")
    if decision_params is None:
        raise KeyError("snapshot has no 'decision_params' (decision_params.selected_decisions is required)")
    if isinstance(decision_params, Mapping):
        selected = decision_params.get("selected_decisions")
    else:
        selected = getattr(decision_params, "selected_decisions", None)
    if selected is None:
        raise KeyError("snapshot decision_params has no 'selected_decisions'")
    return list(selected)


# ------------------------------------------------------- decision settings

def collect_decision_settings(snapshot: Mapping, default_decision_values: Mapping) -> dict:
    """Collect current default decision settings from the snapshot (probabilities, selections, etc.)

    IMPORTANT: This function checks both session state AND _persistent_defaults (shadow state).
    The _persistent_defaults dictionary is used by default_config.py widgets to preserve values
    across page navigation. We must check it here to ensure configured values are used even when
    Page 2 hasn't rendered (e.g., when running simulation from Results page).
    """
    decision_settings = {}

    # Get the persistent defaults dictionary (shadow state from default_config.py)
    # This is critical for preserving user-configured values when Page 2 hasn't rendered
    persistent_defaults = snapshot.get("_persistent_defaults", {})

    # Check each decision for configured settings
    for decision_name, default_value in default_decision_values.items():
        if isinstance(default_value, dict):
            decision_type = default_value.get("type")

            # Handle random probability decisions (disclose_income, disclose_documents, purchase_vs_bid)
            if decision_type == "random_probability":
                # Priority order for probability values:
                # 1. Post-simulation adjustment from Results page ({decision_name}_probability_y)
                # 2. Pre-configured default from Overview tab - persistent storage (_persistent_defaults)
                # 3. Pre-configured default from Overview tab - session state ({decision_name}_default_probability_y)
                # 4. Hard-coded default from DEFAULT_DECISION_VALUES

                post_sim_key = f"{decision_name}_probability_y"
                pre_config_key = f"{decision_name}_default_probability_y"
                hardcoded_default = default_value.get("probability_y", 0.5)

                # Check in priority order
                if post_sim_key in snapshot:
                    current_prob = snapshot[post_sim_key]
                elif pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    # Session state keys might be reset to defaults by initialization scripts during reruns
                    current_prob = persistent_defaults[pre_config_key]
                elif pre_config_key in snapshot:
                    current_prob = snapshot[pre_config_key]
                else:
                    current_prob = hardcoded_default

                decision_settings[decision_name] = {
                    "probability_y": current_prob,
                    "options": default_value.get("options", ["Y", "N"]),
                    "type": "random_probability"
                }

            # Handle checkbox selection decisions (vendor_choice_weights)
            elif decision_type == "checkbox_selection":
                # Priority order:
                # 1. Post-simulation adjustment from Results page (vendor_choice_weights_selection)
                # 2. Pre-configured default from Overview tab - persistent storage (_persistent_defaults)
                # 3. Pre-configured default from Overview tab - session state (vendor_choice_weights_default_params)
                # 4. Hard-coded default from DEFAULT_DECISION_VALUES

                post_sim_key = f"{decision_name}_selection"  # e.g., "vendor_choice_weights_selection"
                pre_config_key = f"{decision_name}_default_params"  # e.g., "vendor_choice_weights_default_params"
                hardcoded_default = default_value.get("default_selection", [])

                # Check in priority order
                if post_sim_key in snapshot:
                    selected_params = snapshot[post_sim_key]
                elif pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    selected_params = persistent_defaults[pre_config_key]
                elif pre_config_key in snapshot:
                    selected_params = snapshot[pre_config_key]
                else:
                    selected_params = hardcoded_default

                # Calculate equal weights for selected parameters
                if len(selected_params) > 0:
                    weight_per_param = 1.0 / len(selected_params)
                    weights = {}

                    # Set weights for all parameters
                    for param_key in default_value.get("parameters", {}).keys():
                        if param_key in selected_params:
                            weights[param_key] = weight_per_param
                        else:
                            weights[param_key] = 0.0
                else:
                    # Fallback to equal weights if nothing selected
                    params = list(default_value.get("parameters", {}).keys())
                    weight_per_param = 1.0 / len(params) if params else 0.25
                    weights = {param: weight_per_param for param in params}

                decision_settings[decision_name] = {
                    "selected_params": selected_params,
                    "weights": weights,
                    "type": "checkbox_selection"
                }

            # Handle prioritized selection decisions (rejected_transaction_defaults with priority lists)
            elif decision_type == "prioritized_selection":
                # Priority order:
                # 1. Pre-configured priority template from Overview tab - persistent storage (_persistent_defaults)
                # 2. Pre-configured priority template from Overview tab - session state ({decision_name}_priority_template)
                # 3. Hard-coded default from DEFAULT_DECISION_VALUES

                pre_config_key = f"{decision_name}_priority_template"
                hardcoded_default = default_value.get("priority_template", ["forgo_transaction"])

                # Check in priority order
                if pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    priority_template = persistent_defaults[pre_config_key]
                elif pre_config_key in snapshot:
                    priority_template = snapshot[pre_config_key]
                else:
                    priority_template = hardcoded_default

                decision_settings[decision_name] = {
                    "priority_template": priority_template,
                    "type": "prioritized_selection"
                }

            # Handle radio selection decisions (rejected_transaction_option)
            elif decision_type == "radio_selection":
                # Priority order:
                # 1. Post-simulation adjustment from Results page (specific to each decision)
                # 2. Pre-configured default from Overview tab - persistent storage (_persistent_defaults)
                # 3. Pre-configured default from Overview tab - session state ({decision_name}_default_selection)
                # 4. Hard-coded default from DEFAULT_DECISION_VALUES

                # Map decision names to their post-simulation keys
                post_sim_keys = {
                    "rejected_transaction_defaults": "rejected_transaction_defaults_option",
                    "rejected_transaction_option": "rejected_transaction_option_selection"
                }

                post_sim_key = post_sim_keys.get(decision_name)
                pre_config_key = f"{decision_name}_default_selection"
                hardcoded_default = default_value.get("default_option", "")

                # Check in priority order
                if post_sim_key and post_sim_key in snapshot:
                    selected_option = snapshot[post_sim_key]
                elif pre_config_key in persistent_defaults:
                    # CRITICAL: Check persistent storage BEFORE session state
                    selected_option = persistent_defaults[pre_config_key]
                elif pre_config_key in snapshot:
                    selected_option = snapshot[pre_config_key]
                else:
                    selected_option = hardcoded_default

                decision_settings[decision_name] = {
                    "selected_option": selected_option,
                    "type": "radio_selection"
                }

        else:
            # Handle simple values (numeric or string placeholders)
            pre_config_key = f"{decision_name}_default_value"

            # Check in priority order: persistent_defaults -> session state -> hardcoded default
            if pre_config_key in persistent_defaults:
                configured_value = persistent_defaults[pre_config_key]
            elif pre_config_key in snapshot:
                configured_value = snapshot[pre_config_key]
            else:
                configured_value = default_value

            # Determine if it's numeric or a placeholder string
            if isinstance(configured_value, (int, float)):
                decision_settings[decision_name] = {
                    "value": configured_value,
                    "type": "numeric"
                }
            else:
                # It's a placeholder string like "RANDOM_WITHIN_LIMIT", "NA", etc.
                decision_settings[decision_name] = {
                    "value": configured_value,
                    "type": "placeholder"
                }

    return decision_settings


def describe_decision_settings(decision_settings: Mapping) -> List[str]:
    """The per-decision fragments of the '🎲 Using configured defaults: ...' message."""
    setting_info = []
    for decision, settings in decision_settings.items():
        decision_type = settings.get("type")
        if decision_type == "random_probability":
            prob_y = settings["probability_y"]
            options = settings["options"]
            setting_info.append(f"{decision}: {prob_y:.0%} {options[0]} / {1-prob_y:.0%} {options[1]}")
        elif decision_type == "prioritized_selection":
            priority_template = settings.get("priority_template", [])
            if len(priority_template) == 1:
                setting_info.append(f"{decision}: {priority_template[0]} only")
            else:
                setting_info.append(f"{decision}: {len(priority_template)} priorities")
        elif decision_type == "checkbox_selection":
            selected = settings.get("selected_params", [])
            setting_info.append(f"{decision}: {len(selected)} params selected ({', '.join(selected)})")
        elif decision_type == "radio_selection":
            selected = settings.get("selected_option", "unknown")
            setting_info.append(f"{decision}: {selected}")
        elif decision_type == "numeric":
            value = settings.get("value", 0)
            try:
                float_value = float(value)
                if 0 <= float_value <= 1:
                    setting_info.append(f"{decision}: {float_value:.1%}")
                else:
                    setting_info.append(f"{decision}: {float_value}")
            except (ValueError, TypeError):
                setting_info.append(f"{decision}: {value}")
        elif decision_type == "placeholder":
            value = settings.get("value", "default")
            setting_info.append(f"{decision}: {value}")
    return setting_info


# ------------------------------------------------------ Page 1 parameters

def build_simulation_params(snapshot: Mapping) -> dict:
    """
    The ['simulation'] mapping: Page 1 parameters copied 1:1 from
    ``sim_params`` (the former ``_apply_simulation_params``), so the user's
    values take precedence over config/simulation.yaml.
    """
    sim_params = snapshot["sim_params"]
    sim_config: Dict[str, Any] = {}

    for attr in _SIMULATION_PARAM_ATTRS:
        sim_config[attr] = getattr(sim_params, attr)

    # Vendor configuration data (if uploaded via CSV)
    if getattr(sim_params, "vendor_config_data", None) is not None:
        sim_config["vendor_config_data"] = sim_params.vendor_config_data

    # Legacy vendor parameters (for backward compatibility)
    sim_config["products_per_vendor"] = sim_params.products_per_vendor
    sim_config["carryover"] = sim_params.carryover
    if getattr(sim_params, "vendor_prices", None) is not None:
        sim_config["vendor_prices"] = sim_params.vendor_prices

    # Time parameters
    sim_config["periods"] = sim_params.periods
    sim_config["duration_hours"] = sim_params.duration_hours

    # Income categories
    sim_config["num_discount_categories"] = sim_params.num_discount_categories
    sim_config["num_fixed_categories"] = sim_params.num_fixed_categories

    # Consumption parameters
    sim_config["max_purchases_per_term"] = sim_params.max_purchases_per_term

    return sim_config


def build_purchasing_limits(snapshot: Mapping) -> Optional[dict]:
    """sim_params.purchasing_limits when enabled on Page 1, else None (yaml value stays)."""
    sim_params = snapshot["sim_params"]
    if sim_params.apply_purchasing_limits:
        return sim_params.purchasing_limits
    return None


# --------------------------------------------------------- donation_default

def session_donation_coefficient_set(snapshot: Mapping, income_mode: str) -> Optional[dict]:
    """
    R10: the flat coefficient set the Donation tab shows for ``income_mode``,
    read from the donation_coeff_*_{cat,cont} session keys (populated from the
    file by app.models.load_coefficient_set).  None when any key is missing.
    """
    suffix = "cont" if income_mode == "continuous" else "cat"

    def key(stem: str) -> str:
        return f"donation_coeff_{stem}_{suffix}"

    needed = list(_DONATION_SCALAR_KEYS) + list(_DONATION_GROUP_KEYS) \
        + list(_DONATION_QUINTILE_KEYS) + list(_DONATION_STUDY_KEYS)
    if any(key(stem) not in snapshot for stem in needed):
        return None

    return {
        "intercept": snapshot[key("intercept")],
        "beta_group": {label: snapshot[key(stem)] for stem, label in _DONATION_GROUP_KEYS.items()},
        "beta_income_q": {label: snapshot[key(stem)] for stem, label in _DONATION_QUINTILE_KEYS.items()},
        "beta_income_linear": snapshot[key("linear")],
        "beta_study": {label: snapshot[key(stem)] for stem, label in _DONATION_STUDY_KEYS.items()},
        "beta_hh": snapshot[key("hh")],
    }


def build_donation_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                         pop_mode: str, inc_mode: str) -> dict:
    """
    Decision 3 patch (the former ``_apply_donation_config`` +
    ``apply_selected_donation_config``, without their session writes).

    A saved donation config pins the income mode, the anchor weights and the
    stochastic toggles/coefficient it recorded (R14); the coefficient set is the
    tab's set for that mode (R10); sigma_value is the config's sigma_overall x
    the coefficient (R12); adjustment.shift_value comes from the
    donation_adjustment_shift session key (R11).
    """
    saved = explicit_saved_configs(snapshot).get("donation_default")

    # Income mode. Priority: saved config > passed parameter
    if saved:
        actual_inc_mode = normalize_income_mode(
            saved.get("donation_income_mode", saved.get("income_spec_mode", inc_mode)))
    else:
        actual_inc_mode = normalize_income_mode(inc_mode)

    # R10: the flat set for the run's income mode, WITHOUT the nested blocks
    coefficients = session_donation_coefficient_set(snapshot, actual_inc_mode)
    if coefficients is None:
        coefficients = config_repo.donation_coefficient_set(actual_inc_mode)
    coefficients["income_mode"] = actual_inc_mode

    # Stochastic toggles: live tab values, pinned by the saved record when present (R14)
    saved_stochastic = ((saved or {}).get("stochastic_params") or {}).get("stochastic") or {}
    sigma_in_copula = saved_stochastic.get("sigma_in_copula", snapshot.get("sigma_in_copula", False))
    sigma_in_research = saved_stochastic.get("sigma_in_research", snapshot.get("sigma_in_research", True))
    sigma_coefficient = saved_stochastic.get("sigma_coefficient", snapshot.get("sigma_coefficient", 1.0))

    stochastic: Dict[str, Any] = {}
    if pop_mode == "copula":
        stochastic["in_copula"] = sigma_in_copula

    if pop_mode == "documentation" and not sigma_in_research:
        # Research mode with sigma disabled - set to 0
        stochastic["sigma_value"] = 0.0
    else:
        # R12: one sigma constant - the config's sigma_overall times the coefficient slider
        stochastic["sigma_value"] = donation_sigma_overall(config_repo) * float(sigma_coefficient)

    if "donation_sigma_strategy" in snapshot:
        stochastic["sigma_strategy"] = snapshot["donation_sigma_strategy"]
    if saved_stochastic.get("sigma_coefficient") is not None or "sigma_coefficient" in snapshot:
        stochastic["scale_factor"] = sigma_coefficient
    if "donation_quintile_scale_factors" in snapshot:
        stochastic["quintile_scale_factors"] = copy.deepcopy(snapshot["donation_quintile_scale_factors"])

    # Anchor weights: the tab slider, overridden by the saved record
    anchor_observed = snapshot.get("anchor_observed_weight", 0.75)
    anchor_weights: Dict[str, Any] = {"observed": anchor_observed, "predicted": 1 - anchor_observed}
    if saved:
        anchor_weights.update(((saved.get("stochastic_params") or {}).get("anchor_weights") or {}))

    patch: Dict[str, Any] = {
        "regression": {"income_mode": actual_inc_mode},
        "regression_coefficients": Replace(coefficients),
        "stochastic": stochastic,
        "anchor_weights": anchor_weights,
    }

    # R11: the adjustment shift lives in session, the file is never rewritten
    shift = snapshot.get("donation_adjustment_shift")
    if shift is not None:
        patch["adjustment"] = {"shift_value": float(shift)}

    return patch


# ---------------------------------------------------------- disclose_income

def _copy_prefixed_sigma_settings(snapshot: Mapping, prefix: str, stochastic: Dict[str, Any]) -> None:
    """Copy {prefix}_sigma_strategy / _scale_factor / _quintile_scale_factors when present."""
    if f"{prefix}_sigma_strategy" in snapshot:
        stochastic["sigma_strategy"] = snapshot[f"{prefix}_sigma_strategy"]
    if f"{prefix}_scale_factor" in snapshot:
        stochastic["scale_factor"] = snapshot[f"{prefix}_scale_factor"]
    if f"{prefix}_quintile_scale_factors" in snapshot:
        stochastic["quintile_scale_factors"] = copy.deepcopy(snapshot[f"{prefix}_quintile_scale_factors"])


def _three_way_stochastic(pop_mode: str, sigma_enabled: bool, sigma_in_copula: bool,
                          sentinel: float) -> Dict[str, Any]:
    """baseline off / copula via flag / documentation via checkbox (the shared rule)."""
    if pop_mode == "baseline":
        return {"sigma_value": 0.0, "in_copula": False}
    if pop_mode == "copula":
        return {"in_copula": sigma_in_copula, "sigma_value": sentinel if sigma_in_copula else 0.0}
    if sigma_enabled:
        return {"sigma_value": sentinel, "in_copula": False}
    return {"sigma_value": 0.0, "in_copula": False}


def build_disclose_income_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                                pop_mode: str, inc_mode: Optional[str] = None) -> dict:
    """
    Decision 1 patch.  Priority: saved config (R14) > explicit inc_mode
    (Compare both) > session state.
    """
    saved = explicit_saved_configs(snapshot).get("disclose_income")
    if saved is not None:
        return _saved_disclose_income_patch(snapshot, config_repo, pop_mode, saved)
    return _session_disclose_income_patch(snapshot, config_repo, pop_mode, inc_mode)


def _saved_disclose_income_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                                 pop_mode: str, saved: Mapping) -> dict:
    params = saved.get("params", {}) or {}

    # Priority: top-level income_mode (from result_key) > params.income_mode (at save time)
    income_mode = saved.get("income_mode", params.get("income_mode", "Categorical only"))
    patch: Dict[str, Any] = {"income_mode": normalize_income_mode(income_mode)}

    if "intercept" in params:
        patch["intercept"] = params["intercept"]

    anchor_weights: Dict[str, Any] = {}
    saved_anchors = params.get("anchor_weights", {}) or {}
    if "observed_prosocial" in saved_anchors:
        anchor_weights["observed_prosocial"] = saved_anchors["observed_prosocial"]
    if "prosocial_weight" in saved_anchors:
        anchor_weights["prosocial_weight"] = saved_anchors["prosocial_weight"]
    patch["anchor_weights"] = anchor_weights

    # Stochastic on/off from the CURRENT tab toggles (R18: absent key = the tab's default, on)
    stochastic = _three_way_stochastic(
        pop_mode,
        sigma_enabled=snapshot.get("di_sigma_enabled", True),
        sigma_in_copula=snapshot.get("di_sigma_in_copula", False),
        sentinel=disclose_income_sigma_sentinel(config_repo),
    )
    # Always the CURRENT strategy / scale / quintiles (saved precedence)
    _copy_prefixed_sigma_settings(snapshot, "di", stochastic)
    patch["stochastic"] = stochastic
    return patch


def _session_disclose_income_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                                   pop_mode: str, inc_mode: Optional[str]) -> dict:
    patch: Dict[str, Any] = {}

    # Income mode. Priority: explicit inc_mode parameter > session state
    if inc_mode is not None:
        patch["income_mode"] = normalize_income_mode(inc_mode)
    elif "di_income_mode" in snapshot:
        session_mode = snapshot["di_income_mode"]
        if is_compare_income_mode(session_mode):
            # "Compare both" without an explicit mode - categorical as safeguard
            patch["income_mode"] = "categorical"
        else:
            patch["income_mode"] = session_mode

    if "di_intercept" in snapshot:
        patch["intercept"] = snapshot["di_intercept"]

    anchor_weights: Dict[str, Any] = {}
    if "di_wopb" in snapshot:
        anchor_weights["observed_prosocial"] = snapshot["di_wopb"]
    if "di_wpb" in snapshot:
        anchor_weights["prosocial_weight"] = snapshot["di_wpb"]
    patch["anchor_weights"] = anchor_weights

    sigma_enabled = snapshot.get("di_sigma_enabled", True)  # R18
    sigma_in_copula = snapshot.get("di_sigma_in_copula", False)
    stochastic = _three_way_stochastic(
        pop_mode, sigma_enabled, sigma_in_copula, disclose_income_sigma_sentinel(config_repo))
    # Strategy / scale / quintiles only when the draws are on (session precedence)
    if (pop_mode == "copula" and sigma_in_copula) or (pop_mode == "documentation" and sigma_enabled):
        _copy_prefixed_sigma_settings(snapshot, "di", stochastic)
    patch["stochastic"] = stochastic
    return patch


# ------------------------------------------------------- disclose_documents

def build_disclose_documents_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                                   pop_mode: str, inc_mode: Optional[str] = None) -> dict:
    """
    Decision 2 patch.  R17: a saved Disclose Documents config is applied with
    the saved-Disclose-Income precedence; otherwise explicit inc_mode (Compare
    both) > session state, as before.
    """
    saved = explicit_saved_configs(snapshot).get("disclose_documents")
    if saved is not None:
        return _saved_disclose_documents_patch(snapshot, config_repo, pop_mode, saved)
    return _session_disclose_documents_patch(snapshot, config_repo, pop_mode, inc_mode)


def _saved_disclose_documents_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                                    pop_mode: str, saved: Mapping) -> dict:
    params = saved.get("params", {}) or {}
    income_mode = saved.get("income_mode", params.get("income_mode", "Categorical only"))
    patch: Dict[str, Any] = {"income_mode": normalize_income_mode(income_mode)}
    if "intercept" in params:
        patch["intercept"] = params["intercept"]

    stochastic = _three_way_stochastic(
        pop_mode,
        sigma_enabled=snapshot.get("dd_sigma_enabled", True),   # R18
        sigma_in_copula=snapshot.get("dd_sigma_in_copula", False),
        sentinel=disclose_documents_sigma_sentinel(config_repo),
    )
    _copy_prefixed_sigma_settings(snapshot, "dd", stochastic)
    patch["stochastic"] = stochastic
    return patch


def _session_disclose_documents_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                                      pop_mode: str, inc_mode: Optional[str]) -> dict:
    patch: Dict[str, Any] = {}

    # Income mode: explicit inc_mode (Compare both) > session state > config default
    if inc_mode is not None:
        patch["income_mode"] = normalize_income_mode(inc_mode)
    elif "dd_income_mode" in snapshot:
        session_mode = snapshot["dd_income_mode"]
        if is_compare_income_mode(session_mode):
            patch["income_mode"] = "categorical"  # safeguard; the runner passes an explicit mode
        else:
            patch["income_mode"] = session_mode

    # Optional intercept override (R11: from the tab's session key)
    if "dd_intercept" in snapshot:
        patch["intercept"] = snapshot["dd_intercept"]

    stochastic = _three_way_stochastic(
        pop_mode,
        sigma_enabled=snapshot.get("dd_sigma_enabled", True),   # R18
        sigma_in_copula=snapshot.get("dd_sigma_in_copula", False),
        sentinel=disclose_documents_sigma_sentinel(config_repo),
    )
    _copy_prefixed_sigma_settings(snapshot, "dd", stochastic)
    patch["stochastic"] = stochastic
    return patch


# ------------------------------------------- rejected_transaction_defaults

def build_rejected_transaction_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                                     pop_mode: str, inc_mode: Optional[str] = None) -> dict:
    """
    Decision 4 patch (the former ``_apply_rejected_transaction_config``).

    The five mechanisms' coefficients, priority sequences and sigma constants
    are fixed in config/decisions.yaml.  MODEL settings, by precedence:

    * a saved configuration ("Use This Config" on an individual Decision-4
      run, R14) - applied when the run is NOT an individual Decision-4 run
      (inc_mode is None, i.e. combined / complete simulations): its income
      mode (top-level, from the selected result cell, over params.income_mode),
      per-element intercepts, Flexibility anchor mix and rank-aggregation flag
      override the tab, exactly as a saved Disclose Income config does;
    * otherwise the tab: explicit inc_mode (passed ONLY for an individual
      Decision-4 run, incl. the two Compare-both sub-runs) > rtd_income_mode;
      a 'Compare both' tab setting without an explicit mode falls back to
      'continuous'.  Per-element intercepts from rtd_intercept_{mech}, the
      Flexibility anchor mix from rtd_flex_observed_weight
      (W_CFlex = 1 - W_OFlex), the Section-6 aggregation flag from
      rtd_aggregation_enabled.

    The stochastic toggles and settings always follow the CURRENT tab: the
    three-way enable rule (rtd_sigma_enabled defaults True, R18); sigma
    strategy / scale / quintiles are decision-wide and replicated to each
    mechanism.  (The September 2026 tab removed the per-element stochastic
    anchor control, so no ``anchor`` is sent; the model keeps its default.)
    """
    patch: Dict[str, Any] = {}

    saved = explicit_saved_configs(snapshot).get("rejected_transaction_defaults") if inc_mode is None else None
    if saved is not None:
        _saved_rejected_transaction_model_settings(patch, saved)
    else:
        if inc_mode is not None:
            patch["income_mode"] = normalize_income_mode(inc_mode)
        elif "rtd_income_mode" in snapshot:
            session_mode = str(snapshot["rtd_income_mode"])
            if is_compare_income_mode(session_mode):
                patch["income_mode"] = "continuous"  # safeguard; runner passes explicit mode
            else:
                patch["income_mode"] = normalize_income_mode(session_mode)

        # Per-element intercepts (doc notation beta0..beta4; research defaults from
        # config/decisions.yaml - all five 0; TTP beta0 on the raw composite)
        intercepts: Dict[str, float] = {}
        for mech in RTD_MECHANISMS:
            key = f"rtd_intercept_{mech}"
            if key in snapshot:
                intercepts[mech] = float(snapshot[key])
        patch["intercepts"] = intercepts

        # Flexibility Anchor Mix (sub-tab 5): W_OFlex slider, W_CFlex = 1 - W_OFlex
        if "rtd_flex_observed_weight" in snapshot:
            w_obs = float(snapshot["rtd_flex_observed_weight"])
            patch["flexibility_anchor"] = {"observed_weight": w_obs,
                                           "calculated_weight": 1.0 - w_obs}

        # Section-6 rank aggregation: enable flag (the last-resort tie-break is
        # always the document's random rule, from the config file).
        aggregation: Dict[str, Any] = {}
        if "rtd_aggregation_enabled" in snapshot:
            aggregation["enabled"] = bool(snapshot["rtd_aggregation_enabled"])
        patch["aggregation"] = aggregation

    sigma_enabled = snapshot.get("rtd_sigma_enabled", True)
    sigma_in_copula = snapshot.get("rtd_sigma_in_copula", False)
    stochastic: Dict[str, Any] = _three_way_stochastic(pop_mode, sigma_enabled, sigma_in_copula, RTD_SIGMA_ENABLE)

    mechanisms: Dict[str, Dict[str, Any]] = {}
    for mech in RTD_MECHANISMS:
        mech_cfg: Dict[str, Any] = {}
        if "rtd_sigma_strategy" in snapshot:
            mech_cfg["sigma_strategy"] = snapshot["rtd_sigma_strategy"]
        if "rtd_scale_factor" in snapshot:
            mech_cfg["scale_factor"] = snapshot["rtd_scale_factor"]
        if "rtd_quintile_scale_factors" in snapshot:
            mech_cfg["quintile_scale_factors"] = copy.deepcopy(snapshot["rtd_quintile_scale_factors"])
        mechanisms[mech] = mech_cfg
    stochastic["mechanisms"] = mechanisms
    patch["stochastic"] = stochastic
    return patch


def _saved_rejected_transaction_model_settings(patch: Dict[str, Any], saved: Mapping) -> None:
    """
    A saved Decision 4 configuration's MODEL settings, written into ``patch``:
    income mode (top-level, from the selected result cell, over
    params.income_mode), per-element intercepts, the Flexibility anchor mix and
    the rank-aggregation enable flag.  The stochastic on/off toggles, sigma
    strategy and coefficients follow the CURRENT tab, as for disclose_income.
    """
    params = saved.get("params", {}) or {}
    income_mode = saved.get("income_mode", params.get("income_mode", "Continuous only"))
    patch["income_mode"] = "categorical" if "categorical" in str(income_mode).lower() else "continuous"

    patch["intercepts"] = {mech: float(value)
                           for mech, value in (params.get("intercepts") or {}).items()}

    aggregation: Dict[str, Any] = {}
    saved_agg = params.get("aggregation") or {}
    if "enabled" in saved_agg:
        aggregation["enabled"] = bool(saved_agg["enabled"])
    patch["aggregation"] = aggregation

    saved_anchor = params.get("flexibility_anchor") or {}
    if "observed_weight" in saved_anchor:
        w_obs = float(saved_anchor["observed_weight"])
        patch["flexibility_anchor"] = {
            "observed_weight": w_obs,
            "calculated_weight": float(saved_anchor.get("calculated_weight", 1.0 - w_obs)),
        }


def build_decision_patches(snapshot: Mapping, config_repo: DecisionsConfig, pop_mode: str,
                           inc_mode: str, decisions_to_run: Optional[Sequence[str]]) -> Dict[str, dict]:
    """
    The four decision patches for one sub-run, in the order the runners applied
    them (donation -> DI -> DD -> RTD).  Decision 4 receives the explicit
    per-run inc_mode only for an individual Decision-4 run.
    """
    rtd_inc_mode = inc_mode if list(decisions_to_run or []) == ["rejected_transaction_defaults"] else None
    return {
        "donation_default": build_donation_patch(snapshot, config_repo, pop_mode, inc_mode),
        "disclose_income": build_disclose_income_patch(snapshot, config_repo, pop_mode, inc_mode),
        "disclose_documents": build_disclose_documents_patch(snapshot, config_repo, pop_mode, inc_mode),
        "rejected_transaction_defaults": build_rejected_transaction_patch(snapshot, config_repo, pop_mode, rtd_inc_mode),
    }


# ------------------------------------------------------ per-decision income

# The modelled decisions that carry their own income mode, and how each engine
# module reads it (Decision 4 defaults to continuous and tests for 'categorical';
# the others default to categorical and test for 'continuous').
INCOME_MODE_DECISIONS: Tuple[str, ...] = (
    "disclose_income", "disclose_documents", "donation_default", "rejected_transaction_defaults")


def decision_income_mode(decision: str, patch: Mapping, config_repo: DecisionsConfig) -> str:
    """The income mode ``decision`` runs with under ``patch`` (patch over the file),
    resolved exactly as the engine module resolves it."""
    if decision == "donation_default":
        mode = ((patch or {}).get("regression") or {}).get("income_mode")
        if mode is None:
            block = config_repo.decision(decision)
            mode = ((block.get("regression_coefficients") or {}).get("income_mode")
                    or (block.get("regression") or {}).get("income_mode") or "categorical")
    else:
        mode = (patch or {}).get("income_mode")
        if mode is None:
            mode = config_repo.decision(decision).get(
                "income_mode", "continuous" if decision == "rejected_transaction_defaults" else "categorical")
    if decision == "rejected_transaction_defaults":
        return "categorical" if "categorical" in str(mode).lower() else "continuous"
    return normalize_income_mode(mode)


def sub_run_decision_income_modes(sub_run: SubRun, config_repo: DecisionsConfig) -> Dict[str, str]:
    """{decision: mode} for the income-dependent decisions this sub-run executes."""
    executed = set(sub_run.decisions_to_run) if sub_run.decisions_to_run is not None else set(ALL_DECISIONS)
    return {d: decision_income_mode(d, sub_run.decision_config_patches.get(d) or {}, config_repo)
            for d in INCOME_MODE_DECISIONS if d in executed}


# ------------------------------------------------------------- expectations

def _saved_config_income_mode(decision_name: str, config: Mapping) -> Optional[str]:
    """The normalized income mode a saved config was taken in (None for compare / unknown)."""
    if decision_name == "donation_default":
        mode = config.get("donation_income_mode", config.get("income_spec_mode"))
    else:
        mode = config.get("income_mode", (config.get("params") or {}).get("income_mode"))
    if mode is None or is_compare_income_mode(mode):
        result_key = str(config.get("result_key") or "").lower()
        if "continuous" in result_key:
            return "continuous"
        if "categorical" in result_key:
            return "categorical"
        return None
    return normalize_income_mode(mode)


def build_saved_expectations(configs: Mapping[str, Mapping], sub_runs: Sequence[SubRun]) -> Tuple[SavedExpectation, ...]:
    """
    R14: one expectation per explicit saved config that recorded its output
    columns and hash, bound to the sub-run whose income mode matches the saved
    run (the first sub-run when none does).
    """
    if not sub_runs:
        return ()
    expectations = []
    for decision_name, config in configs.items():
        columns = config.get("result_columns")
        sha256 = config.get("result_sha256")
        if not columns or not sha256:
            continue
        mode = _saved_config_income_mode(decision_name, config)
        result_key = sub_runs[0].result_key
        for sub_run in sub_runs:
            if mode is not None and sub_run.income_mode == mode:
                result_key = sub_run.result_key
                break
        expectations.append(SavedExpectation(decision_name, result_key, tuple(columns), str(sha256)))
    return tuple(expectations)


# -------------------------------------------------------------- the plan

def build_run_plan(snapshot: Any, config_repo: DecisionsConfig, *,
                   default_decision_values: Optional[Mapping] = None,
                   seed_resolution: Optional[Tuple[Any, Any, str]] = None) -> RunPlan:
    """
    Turn one click into a RunPlan.

    snapshot               : SessionSnapshot (or any mapping - a test's dict)
    config_repo            : the read-only config files
    default_decision_values: the DEFAULT_DECISION_VALUES registry; when omitted
                             the contract's own src.contract.defaults registry
    seed_resolution        : (seed, n_agents, source) as returned by
                             app.state.saved_configs.get_simulation_seed_from_configs;
                             when omitted the same rule is applied to the snapshot
    """
    snap: SessionSnapshot = take_snapshot(snapshot)
    if default_decision_values is None:
        default_decision_values = DEFAULT_DECISION_VALUES

    messages: List[UiMessage] = []

    def info(text: str) -> None:
        messages.append(UiMessage("info", text))

    def caption(text: str) -> None:
        messages.append(UiMessage("caption", text))

    def success(text: str) -> None:
        messages.append(UiMessage("success", text))

    # Seed / n_agents from the explicit saved configs (if any) or the session
    if seed_resolution is not None:
        seed, n_agents, source = seed_resolution
    else:
        seed, n_agents, source = resolve_seed_and_n(snap)

    saved_configs = explicit_saved_configs(snap)
    if saved_configs:
        info(f"📋 Using {len(saved_configs)} saved decision configuration(s)")

        # Show details for each saved config
        for decision_name, config in saved_configs.items():
            decision_title = decision_name.replace("_", " ").title()
            if decision_name == "donation_default":
                income_mode = config.get("donation_income_mode",
                                         config.get("income_spec_mode", "categorical only"))
                caption(f"  🎯 {decision_title}: {income_mode}")
            elif decision_name == "disclose_income":
                income_mode = config.get("income_mode",
                                         config.get("params", {}).get("income_mode", "Categorical only"))
                caption(f"  📋 {decision_title}: {income_mode}")
            else:
                caption(f"  ✓ {decision_title}")

        if source == "configs":
            caption(f"🔑 Using saved seed: {seed}, agents: {n_agents:,}")

    # Determine which decisions to run
    selected_decisions = _selected_decisions(snap)
    single_decision = None if len(selected_decisions) == len(ALL_DECISIONS) else selected_decisions

    # execution_single_decision is the list actually passed to the engine. It usually
    # equals single_decision, but for a STANDALONE disclose_documents run we additionally
    # execute disclose_income FIRST so the platform eligibility gate can be applied.
    execution_single_decision = single_decision

    # Collect current decision settings
    decision_settings = collect_decision_settings(snap, default_decision_values)

    if decision_settings:
        setting_info = describe_decision_settings(decision_settings)
        if setting_info:
            success(f"🎲 Using configured defaults: {', '.join(setting_info)}")

    # Default: use the user's current population mode (never mutated)
    effective_pop_mode = snap.get("population_mode", "Copula (synthetic)")
    force_di_default = False

    di_saved = saved_configs.get("disclose_income")
    donation_saved = saved_configs.get("donation_default")
    dd_saved = saved_configs.get("disclose_documents")

    if single_decision == ["disclose_income"]:
        # Running ONLY disclose_income - use its specific mode (R28: no income_spec_mode write)
        effective_income_mode = snap.get("di_income_mode", "Categorical only")
        caption(f"🎯 Using Disclose Income specific mode: {effective_income_mode}")
    elif single_decision == ["disclose_documents"]:
        # Running ONLY disclose_documents - use its specific mode
        effective_income_mode = snap.get("dd_income_mode", "Categorical only")
        caption(f"🎯 Using Disclose Documents specific mode: {effective_income_mode}")

        # ELIGIBILITY (qualified subgroup) for the STANDALONE DD run: Decision 1 runs first.
        #   (i)  no saved disclose_income config -> DI via its DEFAULT probability (the
        #        default-decisions short-circuit, forced through default_decisions_list)
        #   (ii) a saved disclose_income config -> the full DI model with that config
        execution_single_decision = ["disclose_income", "disclose_documents"]
        if di_saved is None:
            force_di_default = True
            caption(
                "🔐 Eligibility: assigning Disclose Income via its default probability "
                f"({decision_settings.get('disclose_income', {}).get('probability_y', 0.5):.0%} Y) "
                "to identify the qualified subgroup (income < threshold AND disclosed income)."
            )
        else:
            caption(
                "🔐 Eligibility: using the selected Disclose Income configuration to "
                "identify the qualified subgroup (income < threshold AND disclosed income)."
            )
    elif single_decision == ["rejected_transaction_defaults"]:
        # Running ONLY rejected_transaction_defaults - use its specific mode
        effective_income_mode = snap.get("rtd_income_mode", "Continuous only")
        caption(f"🎯 Using Rejected Transaction Defaults specific mode: {effective_income_mode}")
    elif single_decision == ["donation_default"]:
        # Running ONLY donation_default - check for saved config first
        if donation_saved:
            effective_income_mode = donation_saved.get(
                "donation_income_mode", donation_saved.get("income_spec_mode", "categorical only"))
        else:
            effective_income_mode = snap.get("income_spec_mode", "categorical only")
        caption(f"🎯 Using Donation Default income mode: {effective_income_mode}")
    else:
        # Combined simulation or other decisions: a saved config's income mode wins
        effective_income_mode = None

        if di_saved:
            effective_income_mode = di_saved.get("income_mode",
                                                 di_saved.get("params", {}).get("income_mode"))
            if effective_income_mode:
                caption(f"📋 Using saved Disclose Income mode: {effective_income_mode}")

        if effective_income_mode is None and donation_saved:
            effective_income_mode = donation_saved.get("donation_income_mode",
                                                       donation_saved.get("income_spec_mode"))
            if effective_income_mode:
                caption(f"🎯 Using saved Donation Default mode: {effective_income_mode}")

        if effective_income_mode is None:
            effective_income_mode = snap.get("income_spec_mode", "categorical only")

        # Population mode for THIS run: a saved config pins it (R14) - the user's
        # population_mode is read, never changed.
        effective_pop_mode = None
        rtd_saved = saved_configs.get("rejected_transaction_defaults")
        for saved_cfg in (di_saved, donation_saved, dd_saved, rtd_saved):
            if saved_cfg:
                pop_mode = saved_cfg.get("population_mode")
                if not pop_mode:
                    pop_mode = get_population_mode_from_result_key(saved_cfg.get("result_key"))
                if pop_mode:
                    effective_pop_mode = pop_mode
                    caption(f"🔄 Running with saved population mode: {pop_mode}")
                    break

        if effective_pop_mode is None:
            effective_pop_mode = snap.get("population_mode", "Copula (synthetic)")

    # Default-decision list handed to the engine (the force-DI-default rule lives here)
    default_decisions_list = list(snap.get("default_decisions", []) or [])
    if force_di_default and "disclose_income" not in default_decisions_list:
        default_decisions_list.append("disclose_income")

    simulation_params = build_simulation_params(snap)
    purchasing_limits = build_purchasing_limits(snap)
    decisions_to_run = tuple(execution_single_decision) if execution_single_decision is not None else None

    def make_sub_run(result_key: str, pop_type: str, inc_mode: str) -> SubRun:
        return SubRun(
            result_key=result_key,
            population=pop_type,
            income_mode=inc_mode,
            seed=seed,
            n_agents=n_agents,
            decisions_to_run=decisions_to_run,
            decision_config_patches=build_decision_patches(
                snap, config_repo, pop_type, inc_mode, execution_single_decision),
            simulation_params=simulation_params,
            decision_settings=decision_settings,
            default_decisions_list=tuple(default_decisions_list),
            purchasing_limits=purchasing_limits,
        )

    def income_modes_for(mode: Any) -> Tuple[str, ...]:
        if mode == "Compare both":
            return ("categorical", "continuous")
        if "continuous" in str(mode).lower():
            return ("continuous",)
        return ("categorical",)  # categorical only

    sub_runs: List[SubRun] = []
    if effective_pop_mode == "Compare all":
        # Compare all three population modes - each uses its natural agent source
        info("🔄 Running Compare All mode - each population uses its natural agent source")
        for result_name, pop_type in COMPARE_ALL_POPULATIONS:
            for inc_mode in income_modes_for(effective_income_mode):
                sub_runs.append(make_sub_run(f"{result_name}_{inc_mode}", pop_type, inc_mode))
    else:
        # Single population mode
        pop_type = get_pop_type(effective_pop_mode)
        if pop_type == "copula":
            info("🎲 Using synthetic agents from copula")
        elif pop_type in ("documentation", "baseline"):
            info("📊 Using original 280 participants")
        for inc_mode in income_modes_for(effective_income_mode):
            sub_runs.append(make_sub_run(inc_mode, pop_type, inc_mode))

    # R14: only a complete run re-checks the pinned decisions' output hashes
    saved_expectations: Tuple[SavedExpectation, ...] = ()
    if execution_single_decision is None:
        saved_expectations = build_saved_expectations(saved_configs, sub_runs)

    # What each income-dependent decision actually ran with (the result keys follow the
    # run's income mode; Decision 4 in a combined run follows its own tab instead).
    decision_income_modes = {sub_run.result_key: sub_run_decision_income_modes(sub_run, config_repo)
                             for sub_run in sub_runs}
    rtd_ran_combined = (
        execution_single_decision != ["rejected_transaction_defaults"]
        and any("rejected_transaction_defaults" in modes for modes in decision_income_modes.values()))
    rtd_compare_both_fallback = bool(
        rtd_ran_combined
        and "rejected_transaction_defaults" not in saved_configs
        and is_compare_income_mode(snap.get("rtd_income_mode", "Continuous only")))

    metadata = RunMetadata(
        effective_population_mode=effective_pop_mode,
        effective_income_mode=effective_income_mode,
        result_keys=tuple(sub_run.result_key for sub_run in sub_runs),
        custom_decisions=tuple(snap.get("custom_decisions", []) or []),
        default_decisions=tuple(snap.get("default_decisions", []) or []),
        seed=seed,
        n_agents=n_agents,
        is_comparison=len(sub_runs) > 1,
        decision_income_modes=decision_income_modes,
        rtd_compare_both_fallback=rtd_compare_both_fallback,
    )

    return RunPlan(
        sub_runs=tuple(sub_runs),
        messages=tuple(messages),
        metadata=metadata,
        saved_expectations=saved_expectations,
    )


__all__ = [
    "ALL_DECISIONS",
    "AUTO_IMPLIED_SOURCE",
    "DEFAULT_DECISION_VALUES",
    "COMPARE_ALL_POPULATIONS",
    "build_decision_patches",
    "build_disclose_documents_patch",
    "build_disclose_income_patch",
    "build_donation_patch",
    "build_purchasing_limits",
    "build_rejected_transaction_patch",
    "build_run_plan",
    "build_saved_expectations",
    "build_simulation_params",
    "collect_decision_settings",
    "decision_income_mode",
    "describe_decision_settings",
    "explicit_saved_configs",
    "get_pop_type",
    "get_population_mode_from_result_key",
    "is_compare_income_mode",
    "normalize_income_mode",
    "resolve_seed_and_n",
    "session_donation_coefficient_set",
    "sub_run_decision_income_modes",
]
