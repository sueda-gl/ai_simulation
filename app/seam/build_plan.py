# app/seam/build_plan.py
"""
Pure plan builder: SessionSnapshot + DecisionsConfig -> RunPlan.

It resolves the seed and agent count, the DI-only / DD-only / RTD-only /
donation-only / combined branches, the Compare-all and Compare-both result
keys and sub-run order, every st.info / caption / success text, and the
per-decision patch functions that map the tabs' settings onto the engine's
configuration.  It reads ONLY the snapshot and the config repo, writes
nothing, and never imports Streamlit.

Facts first, then messages: ``resolve_run_facts`` reads the snapshot into a
``RunFacts`` (seed source, pinned configs, effective modes, decisions run with
defaults, research sampling parameters); ``enumerate_sub_runs`` and
``build_metadata`` build the plan from it; ``messages_for(facts)`` is the one
place the screen's texts are composed.

Rules the builder follows (each noted inline where it applies):
  * The Donation tab's flat coefficient set (donation_coeff_*_{cat,cont}
    session keys, loaded from the file) is what the engine runs - the nested
    categorical/continuous blocks are replaced, never merged onto.
  * Intercepts and the adjustment shift come from session keys, never from
    the file: the app never rewrites a configuration file.
  * Donation sigma_value = the file's one sigma_overall constant x the
    coefficient slider; no sigma constant is spelled as a literal in app/.
  * Auto-implied saved configs do not exist; only an explicit "Use This
    Config" pins anything (a lingering implied record is ignored).
  * An explicit saved config pins seed, n, population and that decision's
    parameters, because a pinned decision must reproduce the run it was
    pinned from; its recorded output-column hash becomes a SavedExpectation
    that the complete run re-checks.
  * No session writes; settings are read once from the snapshot, so nothing
    the run writes can feed back into it.
  * A saved Disclose Documents config is applied exactly like a saved
    Disclose Income config (same precedence).
  * di_/dd_/rtd_sigma_enabled default to True when the key is absent: an
    absent key means the tab's own default, never "off", so navigation
    history cannot change the numbers.
  * The DI-only / DD-only / RTD-only branches never rewrite
    income_spec_mode; the modes a run used travel in RunMetadata, so a run
    cannot change the mode shown on Page 2.
"""

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.contract.defaults import DEFAULT_DECISION_VALUES
from src.contract.plan import (
    Replace,
    RunMetadata,
    RunPlan,
    RunRequest,
    SavedExpectation,
    SubRun,
    UiMessage,
)
from src.engine.core import DECISION_ORDER
from src.engine.sampling import research_population_text
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
    # Run mode, income summary, price grid and purchasing-limit settings: every
    # Page-1 field reaches the mapping.  Of these the engine reads income_min and
    # income_max (src/engine/income.py, the uniform fallback; purchasing_quantity's
    # income_max); the others have no reader in src/ (purchasing_quantity reads the
    # purchasing limits from simulation_config['purchasing_limits'], the top-level
    # key apply_simulation_settings writes from SubRun.purchasing_limits).
    "simulation_execution_mode", "simulation_mode",
    "income_min", "income_max", "income_avg", "income_avg_type",
    "price_grid",
    "apply_purchasing_limits", "purchasing_limits", "purchasing_limits_source",
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
    The saved-config store minus any auto-implied entry (those no longer
    exist - only an explicit "Use This Config" creates a saved config; one
    left over in an old session is ignored, never applied).
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


def simulation_params_snapshot(sim_params: Any) -> Dict[str, Any]:
    """``RunMetadata.simulation_params``: every Page 1 parameter the run used, as a
    plain dict copied from the ``sim_params`` object (a SimulationParameters
    dataclass, or any object with attributes).  The results page and the exports
    read the run's values from here, never from the live Page 1 object."""
    if sim_params is None:
        return {}
    if dataclasses.is_dataclass(sim_params) and not isinstance(sim_params, type):
        return dataclasses.asdict(sim_params)          # deep-copies nested values
    return copy.deepcopy({k: v for k, v in vars(sim_params).items() if not k.startswith("_")})


def build_purchasing_limits(snapshot: Mapping) -> Optional[dict]:
    """sim_params.purchasing_limits when enabled on Page 1, else None (yaml value stays)."""
    sim_params = snapshot["sim_params"]
    if sim_params.apply_purchasing_limits:
        return sim_params.purchasing_limits
    return None


# --------------------------------------------------------- donation_default

def session_donation_coefficient_set(snapshot: Mapping, income_mode: str) -> Optional[dict]:
    """
    The flat coefficient set the Donation tab shows for ``income_mode`` - the
    set the engine runs - read from the donation_coeff_*_{cat,cont} session
    keys (populated from the file by app.models.load_coefficient_set).  None
    when any key is missing.
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


def donation_anchor_weights(snapshot: Mapping, saved: Optional[Mapping],
                            config_repo: DecisionsConfig) -> Dict[str, Any]:
    """The donation anchor weights a run uses: the tab slider
    (``anchor_observed_weight``; absent from the session -> the file's
    ``donation_default.anchor_weights.observed``), overridden by a pinned donation
    record's. Shared by the Decision 3 patch and the run metadata (the results
    page shows the run's mix, not the live slider)."""
    anchor_observed = snapshot.get(
        "anchor_observed_weight",
        config_repo.decision("donation_default")["anchor_weights"]["observed"])
    anchor_weights: Dict[str, Any] = {"observed": anchor_observed, "predicted": 1 - anchor_observed}
    if saved:
        anchor_weights.update(((saved.get("stochastic_params") or {}).get("anchor_weights") or {}))
    return anchor_weights


def build_donation_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                         pop_mode: str, inc_mode: str) -> dict:
    """
    Decision 3 patch.

    A saved donation config pins the income mode, the anchor weights and the
    stochastic toggles/coefficient it recorded (a pinned decision must
    reproduce the run it was pinned from); the coefficient set is the tab's
    set for that mode; sigma_value is the file's one sigma_overall constant x
    the coefficient; adjustment.shift_value comes from the
    donation_adjustment_shift session key, never from the file.
    """
    saved = explicit_saved_configs(snapshot).get("donation_default")

    # Income mode. Priority: saved config > passed parameter
    if saved:
        actual_inc_mode = normalize_income_mode(
            saved.get("donation_income_mode", saved.get("income_spec_mode", inc_mode)))
    else:
        actual_inc_mode = normalize_income_mode(inc_mode)

    # The flat set for the run's income mode, WITHOUT the nested blocks (replaced, not merged)
    coefficients = session_donation_coefficient_set(snapshot, actual_inc_mode)
    if coefficients is None:
        coefficients = config_repo.donation_coefficient_set(actual_inc_mode)
    coefficients["income_mode"] = actual_inc_mode

    # Stochastic toggles: live tab values, pinned by the saved record when present
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
        # One sigma constant - the config's sigma_overall times the coefficient slider
        stochastic["sigma_value"] = donation_sigma_overall(config_repo) * float(sigma_coefficient)

    if "donation_sigma_strategy" in snapshot:
        stochastic["sigma_strategy"] = snapshot["donation_sigma_strategy"]
    if saved_stochastic.get("sigma_coefficient") is not None or "sigma_coefficient" in snapshot:
        stochastic["scale_factor"] = sigma_coefficient
    if "donation_quintile_scale_factors" in snapshot:
        stochastic["quintile_scale_factors"] = copy.deepcopy(snapshot["donation_quintile_scale_factors"])

    # Anchor weights: the tab slider, overridden by the saved record
    anchor_weights = donation_anchor_weights(snapshot, saved, config_repo)

    patch: Dict[str, Any] = {
        "regression": {"income_mode": actual_inc_mode},
        "regression_coefficients": Replace(coefficients),
        "stochastic": stochastic,
        "anchor_weights": anchor_weights,
    }

    # The adjustment shift lives in session; the file is never rewritten
    shift = snapshot.get("donation_adjustment_shift")
    if shift is not None:
        patch["adjustment"] = {"shift_value": float(shift)}

    return patch


# ------------------------------------------- the modelled decisions' patches
#
# Decisions 1, 2 and 4 map the tab onto the engine the same way: an income mode, a
# few scalar tab values (each also kept by a saved "Use This Config" record), and
# the stochastic block. The table below says, per decision, which session key
# feeds which path of the patch, where a saved record keeps the same value, and
# the transform applied on the way; ``_apply_scalar_fields`` is the one applier.
# What does not fit the table is written out as an explicit special case:
# donation's ``Replace(...)`` coefficient block (above), a saved Decision 4
# record's whole ``intercepts`` block, the Flexibility anchor mix (W_CFlex is
# derived) and Decision 4's per-mechanism sigma replication.

@dataclass(frozen=True)
class ScalarField:
    """One tab value in a decision's patch: ``session_key`` (the tab), ``saved_path``
    (inside a saved record's ``params``; None when the saved record is read another
    way), ``config_path`` (inside the decision block) and the transform applied."""
    session_key: str
    saved_path: Optional[Tuple[str, ...]]
    config_path: Tuple[str, ...]
    transform: Optional[Callable[[Any], Any]] = None


SCALAR_FIELDS: Dict[str, Tuple[ScalarField, ...]] = {
    "disclose_income": (
        ScalarField("di_intercept", ("intercept",), ("intercept",)),
        ScalarField("di_wopb", ("anchor_weights", "observed_prosocial"), ("anchor_weights", "observed_prosocial")),
        ScalarField("di_wpb", ("anchor_weights", "prosocial_weight"), ("anchor_weights", "prosocial_weight")),
    ),
    "disclose_documents": (
        ScalarField("dd_intercept", ("intercept",), ("intercept",)),
    ),
    "rejected_transaction_defaults": (
        # Per-element intercepts (doc notation beta0..beta4; research defaults from
        # config/decisions.yaml - all five 0; TTP beta0 on the raw composite). A saved
        # record's intercepts block is taken whole (see _saved_rtd_intercepts).
        *(ScalarField(f"rtd_intercept_{mech}", None, ("intercepts", mech), float) for mech in RTD_MECHANISMS),
        # Section-6 rank aggregation: enable flag (the last-resort tie-break is
        # always the document's random rule, from the config file).
        ScalarField("rtd_aggregation_enabled", ("aggregation", "enabled"), ("aggregation", "enabled"), bool),
    ),
}

# Patch sub-dicts that are always present (possibly empty), per decision
_PATCH_CONTAINERS: Dict[str, Tuple[str, ...]] = {
    "disclose_income": ("anchor_weights",),
    "disclose_documents": (),
    "rejected_transaction_defaults": ("intercepts", "aggregation"),
}


@dataclass(frozen=True)
class IncomeModeRule:
    """How a decision's patch income mode is read: the tab key, the saved record's
    default spelling, the mode a 'Compare both' tab value falls back to when no
    explicit per-run mode is passed, whether the tab value is normalised (Decision 4)
    or passed through (Decisions 1 and 2), and the saved value's normaliser."""
    session_key: str
    saved_default: str
    compare_fallback: str
    normalise_session_value: bool
    saved_normalise: Callable[[Any], str]


def _rtd_normalize_income_mode(mode: Any) -> str:
    """Decision 4's reading: 'categorical' if the string mentions it, else 'continuous'."""
    return "categorical" if "categorical" in str(mode).lower() else "continuous"


INCOME_MODE_RULES: Dict[str, IncomeModeRule] = {
    "disclose_income": IncomeModeRule("di_income_mode", "Categorical only", "categorical", False, normalize_income_mode),
    "disclose_documents": IncomeModeRule("dd_income_mode", "Categorical only", "categorical", False, normalize_income_mode),
    "rejected_transaction_defaults": IncomeModeRule("rtd_income_mode", "Continuous only", "continuous", True,
                                                    _rtd_normalize_income_mode),
}

# Session-key prefix of each decision's stochastic toggles / sigma settings
_SIGMA_PREFIX = {"disclose_income": "di", "disclose_documents": "dd", "rejected_transaction_defaults": "rtd"}


def _lookup_path(mapping: Mapping, path: Tuple[str, ...]) -> Tuple[bool, Any]:
    """(present, value) of ``path`` inside nested mappings; a missing or None
    intermediate block counts as empty (``block.get(key, {}) or {}``)."""
    current: Any = mapping
    for key in path[:-1]:
        current = current.get(key, {}) or {}
        if not isinstance(current, Mapping):
            return False, None
    return (path[-1] in current), current.get(path[-1])


def _set_path(patch: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    target = patch
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def _apply_scalar_fields(patch: Dict[str, Any], fields: Sequence[ScalarField],
                         source: Mapping, from_saved: bool) -> None:
    """The one applier: copy each present field from the tab (``source`` is the
    snapshot) or from a saved record's ``params`` into the patch."""
    for field_ in fields:
        if from_saved:
            if field_.saved_path is None:
                continue
            present, value = _lookup_path(source, field_.saved_path)
        else:
            present, value = field_.session_key in source, source.get(field_.session_key)
        if not present:
            continue
        _set_path(patch, field_.config_path, field_.transform(value) if field_.transform else value)


def _patch_income_mode(decision: str, snapshot: Mapping, inc_mode: Optional[str],
                       saved: Optional[Mapping]) -> Optional[str]:
    """
    Priority: saved record (top-level income_mode, from the selected result cell,
    over params.income_mode) > explicit inc_mode (Compare both) > the tab key; a
    'Compare both' tab value without an explicit mode falls back to the rule's
    mode. None when none of them says anything (the file's value stays).
    """
    rule = INCOME_MODE_RULES[decision]
    if saved is not None:
        params = saved.get("params", {}) or {}
        return rule.saved_normalise(saved.get("income_mode", params.get("income_mode", rule.saved_default)))
    if inc_mode is not None:
        return normalize_income_mode(inc_mode)
    if rule.session_key in snapshot:
        session_mode = snapshot[rule.session_key]
        if is_compare_income_mode(session_mode):
            return rule.compare_fallback  # safeguard; the runner passes an explicit mode
        return normalize_income_mode(session_mode) if rule.normalise_session_value else session_mode
    return None


def _model_settings_patch(decision: str, snapshot: Mapping, inc_mode: Optional[str],
                          saved: Optional[Mapping]) -> Dict[str, Any]:
    """Income mode + the table's scalar fields, from the saved record or the tab."""
    patch: Dict[str, Any] = {}
    income_mode = _patch_income_mode(decision, snapshot, inc_mode, saved)
    if income_mode is not None:
        patch["income_mode"] = income_mode
    for container in _PATCH_CONTAINERS[decision]:
        patch[container] = {}
    if saved is not None:
        _apply_scalar_fields(patch, SCALAR_FIELDS[decision], saved.get("params", {}) or {}, from_saved=True)
    else:
        _apply_scalar_fields(patch, SCALAR_FIELDS[decision], snapshot, from_saved=False)
    return patch


def _sigma_toggles(snapshot: Mapping, prefix: str) -> Tuple[bool, bool]:
    """(sigma_enabled, sigma_in_copula): absent key = the tab's default (on / off)."""
    return snapshot.get(f"{prefix}_sigma_enabled", True), snapshot.get(f"{prefix}_sigma_in_copula", False)


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


def _disclosure_stochastic(decision: str, snapshot: Mapping, pop_mode: str, sentinel: float,
                           only_when_draws_on: bool) -> Dict[str, Any]:
    """Decision 1 / 2 stochastic block: the three-way rule plus the CURRENT sigma
    strategy / scale / quintiles - always, or (Decision 1 from the tab) only when
    the draws are on."""
    prefix = _SIGMA_PREFIX[decision]
    sigma_enabled, sigma_in_copula = _sigma_toggles(snapshot, prefix)
    stochastic = _three_way_stochastic(pop_mode, sigma_enabled, sigma_in_copula, sentinel)
    draws_on = (pop_mode == "copula" and sigma_in_copula) or (pop_mode == "documentation" and sigma_enabled)
    if draws_on or not only_when_draws_on:
        _copy_prefixed_sigma_settings(snapshot, prefix, stochastic)
    return stochastic


# ---------------------------------------------------------- disclose_income

def build_disclose_income_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                                pop_mode: str, inc_mode: Optional[str] = None) -> dict:
    """
    Decision 1 patch.  Priority: saved config (it pins what the pinned run
    used) > explicit inc_mode (Compare both) > session state.  The stochastic
    on/off comes from the CURRENT tab toggles (absent key = the tab's default,
    on - never "off"); with a saved config the current strategy / scale /
    quintiles always travel, from the tab only when the draws are on.
    """
    saved = explicit_saved_configs(snapshot).get("disclose_income")
    patch = _model_settings_patch("disclose_income", snapshot, inc_mode, saved)
    patch["stochastic"] = _disclosure_stochastic(
        "disclose_income", snapshot, pop_mode, disclose_income_sigma_sentinel(config_repo),
        only_when_draws_on=saved is None)
    return patch


# ------------------------------------------------------- disclose_documents

def build_disclose_documents_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                                   pop_mode: str, inc_mode: Optional[str] = None) -> dict:
    """
    Decision 2 patch.  A saved Disclose Documents config is applied exactly
    like a saved Disclose Income config (same precedence); otherwise explicit
    inc_mode (Compare both) > session state.  The intercept override comes from
    the tab's session key (never from the file).
    """
    saved = explicit_saved_configs(snapshot).get("disclose_documents")
    patch = _model_settings_patch("disclose_documents", snapshot, inc_mode, saved)
    patch["stochastic"] = _disclosure_stochastic(
        "disclose_documents", snapshot, pop_mode, disclose_documents_sigma_sentinel(config_repo),
        only_when_draws_on=False)
    return patch


# ------------------------------------------- rejected_transaction_defaults

def _saved_rtd_intercepts(saved: Mapping) -> Dict[str, float]:
    """A saved Decision 4 record's intercepts block, taken whole (every element it carries)."""
    params = saved.get("params", {}) or {}
    return {mech: float(value) for mech, value in (params.get("intercepts") or {}).items()}


def _flexibility_anchor(snapshot: Mapping, saved: Optional[Mapping]) -> Optional[Dict[str, float]]:
    """Flexibility Anchor Mix (sub-tab 5): W_OFlex from the saved record or the tab
    slider, W_CFlex = 1 - W_OFlex (a saved record may carry its own)."""
    if saved is not None:
        saved_anchor = (saved.get("params", {}) or {}).get("flexibility_anchor") or {}
        if "observed_weight" in saved_anchor:
            w_obs = float(saved_anchor["observed_weight"])
            return {"observed_weight": w_obs,
                    "calculated_weight": float(saved_anchor.get("calculated_weight", 1.0 - w_obs))}
        return None
    if "rtd_flex_observed_weight" in snapshot:
        w_obs = float(snapshot["rtd_flex_observed_weight"])
        return {"observed_weight": w_obs, "calculated_weight": 1.0 - w_obs}
    return None


def build_rejected_transaction_patch(snapshot: Mapping, config_repo: DecisionsConfig,
                                     pop_mode: str, inc_mode: Optional[str] = None) -> dict:
    """
    Decision 4 patch (the former ``_apply_rejected_transaction_config``).

    The five mechanisms' coefficients, priority sequences and sigma constants
    are fixed in config/decisions.yaml.  MODEL settings, by precedence:

    * a saved configuration ("Use This Config" on an individual Decision-4
      run) - applied when the run is NOT an individual Decision-4 run
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
    three-way enable rule (rtd_sigma_enabled defaults True when absent - the
    tab's own default, never "off"); sigma
    strategy / scale / quintiles are decision-wide and replicated to each
    mechanism.  (The September 2026 tab removed the per-element stochastic
    anchor control, so no ``anchor`` is sent; the model keeps its default.)
    """
    saved = explicit_saved_configs(snapshot).get("rejected_transaction_defaults") if inc_mode is None else None
    patch = _model_settings_patch("rejected_transaction_defaults", snapshot, inc_mode, saved)
    if saved is not None:
        patch["intercepts"] = _saved_rtd_intercepts(saved)
    flexibility_anchor = _flexibility_anchor(snapshot, saved)
    if flexibility_anchor is not None:
        patch["flexibility_anchor"] = flexibility_anchor

    # Stochastic: the three-way rule, then the decision-wide sigma settings replicated
    # to every mechanism (explicit special case)
    sigma_enabled, sigma_in_copula = _sigma_toggles(snapshot, "rtd")
    stochastic: Dict[str, Any] = _three_way_stochastic(pop_mode, sigma_enabled, sigma_in_copula, RTD_SIGMA_ENABLE)
    mechanisms: Dict[str, Dict[str, Any]] = {}
    for mech in RTD_MECHANISMS:
        mech_cfg: Dict[str, Any] = {}
        _copy_prefixed_sigma_settings(snapshot, "rtd", mech_cfg)
        mechanisms[mech] = mech_cfg
    stochastic["mechanisms"] = mechanisms
    patch["stochastic"] = stochastic
    return patch


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
    One expectation per explicit saved config that recorded its output
    columns and hash (a pinned decision must reproduce the run it was pinned
    from), bound to the sub-run whose income mode matches the saved
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


# ----------------------------------------------------------------- the facts

@dataclass(frozen=True)
class RunFacts:
    """
    What one click resolved to, before any text is written: the seed and its
    source, the pinned configurations in play, the decisions and the effective
    population / income modes with where each came from, the default-decision
    settings, and the research sampling parameters (``n_agents`` with the
    population).  ``messages_for`` turns these into the screen's texts.
    """
    seed: Any
    n_agents: Any
    seed_source: str                                 # 'configs' | 'session_state'
    saved_configs: Dict[str, dict]                   # the explicit "Use This Config" records, store order
    decision_settings: dict                          # collect_decision_settings() output
    selected_decisions: Tuple[str, ...]
    single_decision: Optional[Tuple[str, ...]]       # None = all 13
    decisions_to_run: Optional[Tuple[str, ...]]      # the list handed to the engine (DD-only adds DI first)
    force_di_default: bool                           # standalone DD run without a saved DI config
    default_decisions_list: Tuple[str, ...]
    branch: str                                      # the single modelled decision run alone, or 'combined'
    effective_income_mode: Any
    income_mode_source: str                          # 'tab' | 'saved_disclose_income' | 'saved_donation_default' | 'session'
    effective_population_mode: str
    pinned_population_mode: Optional[str]            # a saved config's population mode, when one pinned it
    population: Optional[str]                        # internal population type; None for Compare all


def _effective_decisions(snapshot: Mapping,
                         request: Optional[RunRequest] = None) -> Tuple[Tuple[str, ...], Optional[Tuple[str, ...]]]:
    """(selected, single_decision): single_decision is None when every decision is selected.

    The decisions come from the RunRequest when the click handed one over,
    otherwise from the snapshot's decision_params.selected_decisions."""
    selected = tuple(request.decisions_to_run) if request is not None else tuple(_selected_decisions(snapshot))
    return selected, (None if len(selected) == len(ALL_DECISIONS) else selected)


def _resolve_modes(snapshot: Mapping, saved_configs: Mapping[str, dict],
                   single_decision: Optional[Tuple[str, ...]]) -> Dict[str, Any]:
    """
    The effective income and population modes of the run and where they came
    from (the DI-only / DD-only / RTD-only / donation-only / combined branches).
    The single-decision branches never rewrite income_spec_mode; the modes a
    run used travel in RunMetadata, so a run cannot change the mode shown on
    Page 2.  Returns the RunFacts fields it decides.
    """
    di_saved = saved_configs.get("disclose_income")
    donation_saved = saved_configs.get("donation_default")
    dd_saved = saved_configs.get("disclose_documents")

    modes: Dict[str, Any] = {
        "decisions_to_run": single_decision,
        "force_di_default": False,
        "income_mode_source": "tab",
        # Default: the user's current population mode (never mutated)
        "effective_population_mode": snapshot.get("population_mode", "Copula (synthetic)"),
        "pinned_population_mode": None,
    }

    if single_decision == ("disclose_income",):
        modes["branch"] = "disclose_income"
        modes["effective_income_mode"] = snapshot.get("di_income_mode", "Categorical only")
    elif single_decision == ("disclose_documents",):
        modes["branch"] = "disclose_documents"
        modes["effective_income_mode"] = snapshot.get("dd_income_mode", "Categorical only")
        # ELIGIBILITY (qualified subgroup) for the STANDALONE DD run: Decision 1 runs first.
        #   (i)  no saved disclose_income config -> DI via its DEFAULT probability (the
        #        default-decisions short-circuit, forced through default_decisions_list)
        #   (ii) a saved disclose_income config -> the full DI model with that config
        modes["decisions_to_run"] = ("disclose_income", "disclose_documents")
        modes["force_di_default"] = di_saved is None
    elif single_decision == ("rejected_transaction_defaults",):
        modes["branch"] = "rejected_transaction_defaults"
        modes["effective_income_mode"] = snapshot.get("rtd_income_mode", "Continuous only")
    elif single_decision == ("donation_default",):
        modes["branch"] = "donation_default"
        if donation_saved:
            modes["effective_income_mode"] = donation_saved.get(
                "donation_income_mode", donation_saved.get("income_spec_mode", "categorical only"))
        else:
            modes["effective_income_mode"] = snapshot.get("income_spec_mode", "categorical only")
    else:
        # Combined simulation or other decisions: a saved config's income mode wins
        modes["branch"] = "combined"
        effective_income_mode = None
        if di_saved:
            effective_income_mode = di_saved.get("income_mode",
                                                 di_saved.get("params", {}).get("income_mode"))
            modes["income_mode_source"] = "saved_disclose_income"
        if effective_income_mode is None and donation_saved:
            effective_income_mode = donation_saved.get("donation_income_mode",
                                                       donation_saved.get("income_spec_mode"))
            modes["income_mode_source"] = "saved_donation_default"
        if effective_income_mode is None:
            effective_income_mode = snapshot.get("income_spec_mode", "categorical only")
            modes["income_mode_source"] = "session"
        modes["effective_income_mode"] = effective_income_mode

        # Population mode for THIS run: a saved config pins it (to reproduce the pinned run) - the user's
        # population_mode is read, never changed.
        rtd_saved = saved_configs.get("rejected_transaction_defaults")
        for saved_cfg in (di_saved, donation_saved, dd_saved, rtd_saved):
            if saved_cfg:
                pop_mode = saved_cfg.get("population_mode")
                if not pop_mode:
                    pop_mode = get_population_mode_from_result_key(saved_cfg.get("result_key"))
                if pop_mode:
                    modes["pinned_population_mode"] = pop_mode
                    modes["effective_population_mode"] = pop_mode
                    break

    effective_pop_mode = modes["effective_population_mode"]
    modes["population"] = None if effective_pop_mode == "Compare all" else get_pop_type(effective_pop_mode)
    return modes


def resolve_run_facts(snapshot: Mapping, default_decision_values: Mapping,
                      seed_resolution: Optional[Tuple[Any, Any, str]] = None,
                      request: Optional[RunRequest] = None) -> RunFacts:
    """Everything the plan needs, read once from the snapshot (no text yet).

    ``request`` (the click's RunRequest) supplies the decision lists when
    given; otherwise they are read from the snapshot's session keys."""
    # Seed / n_agents from the explicit saved configs (if any) or the session
    if seed_resolution is not None:
        seed, n_agents, source = seed_resolution
    else:
        seed, n_agents, source = resolve_seed_and_n(snapshot)

    saved_configs = explicit_saved_configs(snapshot)
    selected_decisions, single_decision = _effective_decisions(snapshot, request)
    decision_settings = collect_decision_settings(snapshot, default_decision_values)
    modes = _resolve_modes(snapshot, saved_configs, single_decision)

    # Default-decision list handed to the engine (the force-DI-default rule lives here)
    if request is not None:
        default_decisions_list = list(request.default_decisions)
    else:
        default_decisions_list = list(snapshot.get("default_decisions", []) or [])
    if modes["force_di_default"] and "disclose_income" not in default_decisions_list:
        default_decisions_list.append("disclose_income")

    return RunFacts(
        seed=seed,
        n_agents=n_agents,
        seed_source=source,
        saved_configs=saved_configs,
        decision_settings=decision_settings,
        selected_decisions=selected_decisions,
        single_decision=single_decision,
        default_decisions_list=tuple(default_decisions_list),
        **modes,
    )


# -------------------------------------------------------------- the messages

def _saved_config_caption(decision_name: str, config: Mapping) -> str:
    decision_title = decision_name.replace("_", " ").title()
    if decision_name == "donation_default":
        income_mode = config.get("donation_income_mode",
                                 config.get("income_spec_mode", "categorical only"))
        return f"  🎯 {decision_title}: {income_mode}"
    if decision_name == "disclose_income":
        income_mode = config.get("income_mode",
                                 config.get("params", {}).get("income_mode", "Categorical only"))
        return f"  📋 {decision_title}: {income_mode}"
    return f"  ✓ {decision_title}"


def messages_for(facts: RunFacts) -> Tuple[UiMessage, ...]:
    """The st.info / caption / success texts the screen shows before the run, in order."""
    messages: List[UiMessage] = []

    def info(text: str) -> None:
        messages.append(UiMessage("info", text))

    def caption(text: str) -> None:
        messages.append(UiMessage("caption", text))

    def success(text: str) -> None:
        messages.append(UiMessage("success", text))

    if facts.saved_configs:
        info(f"📋 Using {len(facts.saved_configs)} saved decision configuration(s)")
        for decision_name, config in facts.saved_configs.items():
            caption(_saved_config_caption(decision_name, config))
        if facts.seed_source == "configs":
            caption(f"🔑 Using saved seed: {facts.seed}, agents: {facts.n_agents:,}")

    if facts.decision_settings:
        setting_info = describe_decision_settings(facts.decision_settings)
        if setting_info:
            success(f"🎲 Using configured defaults: {', '.join(setting_info)}")

    mode = facts.effective_income_mode
    if facts.branch == "disclose_income":
        caption(f"🎯 Using Disclose Income specific mode: {mode}")
    elif facts.branch == "disclose_documents":
        caption(f"🎯 Using Disclose Documents specific mode: {mode}")
        if facts.force_di_default:
            probability = facts.decision_settings.get("disclose_income", {}).get("probability_y", 0.5)
            caption(
                "🔐 Eligibility: assigning Disclose Income via its default probability "
                f"({probability:.0%} Y) "
                "to identify the qualified subgroup (income < threshold AND disclosed income)."
            )
        else:
            caption(
                "🔐 Eligibility: using the selected Disclose Income configuration to "
                "identify the qualified subgroup (income < threshold AND disclosed income)."
            )
    elif facts.branch == "rejected_transaction_defaults":
        caption(f"🎯 Using Rejected Transaction Defaults specific mode: {mode}")
    elif facts.branch == "donation_default":
        caption(f"🎯 Using Donation Default income mode: {mode}")
    else:
        if facts.income_mode_source == "saved_disclose_income" and mode:
            caption(f"📋 Using saved Disclose Income mode: {mode}")
        elif facts.income_mode_source == "saved_donation_default" and mode:
            caption(f"🎯 Using saved Donation Default mode: {mode}")
        if facts.pinned_population_mode:
            caption(f"🔄 Running with saved population mode: {facts.pinned_population_mode}")

    if facts.population is None:
        # Compare all three population modes - each uses its natural agent source
        info("🔄 Running Compare All mode - each population uses its natural agent source")
    elif facts.population == "copula":
        info("🎲 Using synthetic agents from copula")
    elif facts.population in ("documentation", "baseline"):
        # Research Specification samples the participants randomly (seeded by the run
        # seed); Research Baseline takes them in file order, cycling
        info(research_population_text(facts.n_agents, random_sample=(facts.population == "documentation")))

    return tuple(messages)


# -------------------------------------------------------------- the sub-runs

def income_modes_for(mode: Any) -> Tuple[str, ...]:
    """The per-run income modes of an effective income mode ('Compare both' -> both)."""
    if mode == "Compare both":
        return ("categorical", "continuous")
    if "continuous" in str(mode).lower():
        return ("continuous",)
    return ("categorical",)  # categorical only


def enumerate_sub_runs(snapshot: Mapping, config_repo: DecisionsConfig, facts: RunFacts) -> Tuple[SubRun, ...]:
    """One SubRun per result key, in execution order (Compare all: the three
    populations in COMPARE_ALL_POPULATIONS order, each with its income modes)."""
    simulation_params = build_simulation_params(snapshot)
    purchasing_limits = build_purchasing_limits(snapshot)
    decisions_to_run = facts.decisions_to_run

    def make_sub_run(result_key: str, pop_type: str, inc_mode: str) -> SubRun:
        return SubRun(
            result_key=result_key,
            population=pop_type,
            income_mode=inc_mode,
            seed=facts.seed,
            n_agents=facts.n_agents,
            decisions_to_run=decisions_to_run,
            decision_config_patches=build_decision_patches(
                snapshot, config_repo, pop_type, inc_mode, decisions_to_run),
            simulation_params=simulation_params,
            decision_settings=facts.decision_settings,
            default_decisions_list=facts.default_decisions_list,
            purchasing_limits=purchasing_limits,
        )

    sub_runs: List[SubRun] = []
    if facts.population is None:
        for result_name, pop_type in COMPARE_ALL_POPULATIONS:
            for inc_mode in income_modes_for(facts.effective_income_mode):
                sub_runs.append(make_sub_run(f"{result_name}_{inc_mode}", pop_type, inc_mode))
    else:
        for inc_mode in income_modes_for(facts.effective_income_mode):
            sub_runs.append(make_sub_run(inc_mode, facts.population, inc_mode))
    return tuple(sub_runs)


# -------------------------------------------------------------- the metadata

def build_metadata(snapshot: Mapping, config_repo: DecisionsConfig, facts: RunFacts,
                   sub_runs: Sequence[SubRun], request: Optional[RunRequest] = None) -> RunMetadata:
    """What was actually run, for the results page.  The custom / default
    decision lists come from the click's RunRequest when given, otherwise from
    the snapshot's custom_decisions / default_decisions keys."""
    # What each income-dependent decision actually ran with (the result keys follow the
    # run's income mode; Decision 4 in a combined run follows its own tab instead).
    decision_income_modes = {sub_run.result_key: sub_run_decision_income_modes(sub_run, config_repo)
                             for sub_run in sub_runs}
    rtd_ran_combined = (
        facts.decisions_to_run != ("rejected_transaction_defaults",)
        and any("rejected_transaction_defaults" in modes for modes in decision_income_modes.values()))
    rtd_compare_both_fallback = bool(
        rtd_ran_combined
        and "rejected_transaction_defaults" not in facts.saved_configs
        and is_compare_income_mode(snapshot.get("rtd_income_mode", "Continuous only")))

    return RunMetadata(
        effective_population_mode=facts.effective_population_mode,
        effective_income_mode=facts.effective_income_mode,
        result_keys=tuple(sub_run.result_key for sub_run in sub_runs),
        custom_decisions=(tuple(request.custom_decisions) if request is not None
                          else tuple(snapshot.get("custom_decisions", []) or [])),
        default_decisions=(tuple(request.default_decisions) if request is not None
                           else tuple(snapshot.get("default_decisions", []) or [])),
        seed=facts.seed,
        n_agents=facts.n_agents,
        is_comparison=len(sub_runs) > 1,
        decision_income_modes=decision_income_modes,
        rtd_compare_both_fallback=rtd_compare_both_fallback,
        # what the results page and the exports read instead of the live widgets
        simulation_params=simulation_params_snapshot(snapshot.get("sim_params")),
        rtd_run_element=snapshot.get("rtd_run_element"),
        anchor_observed_weight=donation_anchor_weights(
            snapshot, facts.saved_configs.get("donation_default"), config_repo)["observed"],
        # a copy: the snapshot is shallow, so its lists are the live session's
        default_decision_settings=copy.deepcopy(facts.decision_settings),
    )


# -------------------------------------------------------------- the plan

def build_run_plan(snapshot: Any, config_repo: DecisionsConfig, *,
                   default_decision_values: Optional[Mapping] = None,
                   seed_resolution: Optional[Tuple[Any, Any, str]] = None,
                   request: Optional[RunRequest] = None) -> RunPlan:
    """
    Turn one click into a RunPlan: facts first, then the sub-runs, the
    metadata and the messages.

    snapshot               : SessionSnapshot (or any mapping - a test's dict)
    config_repo            : the read-only config files
    default_decision_values: the DEFAULT_DECISION_VALUES registry; when omitted
                             the contract's own src.contract.defaults registry
    seed_resolution        : (seed, n_agents, source) as returned by
                             app.state.saved_configs.get_simulation_seed_from_configs;
                             when omitted the same rule is applied to the snapshot
    request                : the click's RunRequest (decisions to run, custom and
                             default lists); when omitted those three come from the
                             snapshot's decision_params.selected_decisions,
                             custom_decisions and default_decisions
    """
    snap: SessionSnapshot = take_snapshot(snapshot)
    if default_decision_values is None:
        default_decision_values = DEFAULT_DECISION_VALUES

    facts = resolve_run_facts(snap, default_decision_values, seed_resolution, request)
    sub_runs = enumerate_sub_runs(snap, config_repo, facts)

    # Only a complete run re-checks the pinned decisions' output hashes (a pinned
    # decision must reproduce the run it was pinned from)
    saved_expectations: Tuple[SavedExpectation, ...] = ()
    if facts.decisions_to_run is None:
        saved_expectations = build_saved_expectations(facts.saved_configs, sub_runs)

    return RunPlan(
        sub_runs=sub_runs,
        messages=messages_for(facts),
        metadata=build_metadata(snap, config_repo, facts, sub_runs, request),
        saved_expectations=saved_expectations,
    )


__all__ = [
    "ALL_DECISIONS",
    "AUTO_IMPLIED_SOURCE",
    "DEFAULT_DECISION_VALUES",
    "COMPARE_ALL_POPULATIONS",
    "build_decision_patches",
    "INCOME_MODE_RULES",
    "IncomeModeRule",
    "RunFacts",
    "SCALAR_FIELDS",
    "ScalarField",
    "build_disclose_documents_patch",
    "build_disclose_income_patch",
    "build_donation_patch",
    "build_metadata",
    "build_purchasing_limits",
    "build_rejected_transaction_patch",
    "build_run_plan",
    "build_saved_expectations",
    "donation_anchor_weights",
    "simulation_params_snapshot",
    "build_simulation_params",
    "collect_decision_settings",
    "decision_income_mode",
    "describe_decision_settings",
    "enumerate_sub_runs",
    "explicit_saved_configs",
    "get_pop_type",
    "get_population_mode_from_result_key",
    "income_modes_for",
    "is_compare_income_mode",
    "messages_for",
    "normalize_income_mode",
    "resolve_run_facts",
    "resolve_seed_and_n",
    "session_donation_coefficient_set",
    "sub_run_decision_income_modes",
]
