# app/state/registry.py
"""Registry of every ``st.session_state`` key the app uses.

One :class:`KeySpec` per key, or per key *family* when the code builds the name
with an f-string (``f"{decision_name}_default_value"``).  The registry is plain
data - it imports nothing from the app and never touches Streamlit - so a test,
a doc build or a reviewer can read it without starting a session.

``tests/test_session_key_registry.py`` walks ``app/`` with ``ast`` and holds the
code against this file in both directions: a key the code touches must have an
entry, and an entry no code touches must go.
``docs/migration/session-state.md`` explains the four persistence layers the
``category`` field names and lists the keys that are written but never read.

Conventions for ``default``:

* a Python value - the literal the initialiser stores;
* :data:`WIDGET` - Streamlit owns the value; nothing seeds it up front;
* ``"<expr>"`` (angle brackets) - the value is copied from ``expr`` when the key
  is first created, e.g. ``"<sim_params.periods>"``.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

#: Streamlit owns this value; no initialiser seeds it.
WIDGET = "widget"

#: The persistence layer / role a key belongs to.
CATEGORIES = (
    "navigation",        # which screen renders
    "page1-widget",      # a Page-1 Streamlit widget's own key
    "page1-mirror",      # the canonical Page-1 value a widget mirrors
    "page2-widget",      # a Page-2 / decision-tab widget's own key
    "page2-mirror",      # the canonical Page-2 value the seam reads
    "tab-persistence",   # the per-tab dict that outlives an un-rendered widget
    "saved-config",      # "Use This Config" pins
    "run-flag",          # what the click that started a run recorded
    "results",           # run output and the widgets that display it
    "defaults",          # default-decision parameters and their shadow store
)

#: Variables that name a key only at run time - a helper's parameter, or a loop
#: over ``st.session_state.keys()``.  The scanner cannot resolve these; they are
#: reviewed by hand and listed here so a *new* one fails the test.
#:
#: ``key``        - the parameter of ``default_config.save_to_persistent_storage``
#:                  / ``restore_from_persistent_storage``, and the loop variable
#:                  of the ``dd_`` / ``di_`` / ``rtd_`` prefix resets and of the
#:                  "Clear Results" wipe.
#: ``k``          - the per-category purchasing-limit key on Page 1
#:                  (``purchasing_limit_{i}``, bound in a lambda default).
#: ``chart_key``  - the plotly key built by the shared chart helpers
#:                  (``donation_hist_*`` / ``*_pie_*`` / ``*_raw_hist_*`` and the
#:                  Decision-4 density charts).
DYNAMIC_KEY_VARIABLES = ("key", "k", "chart_key")

#: ``[k for k in st.session_state.keys() if k.startswith(<prefix>)]`` - the
#: whole-decision resets.  Every key beginning with one of these is deleted.
PREFIX_DELETE_PREFIXES = ("dd_", "di_", "rtd_")


@dataclass(frozen=True)
class KeySpec:
    """One key, or one f-string family of keys."""

    name: str
    default: object
    #: where the key is born, as ``module.function`` prose
    initialised_by: str
    category: str
    #: the value reaches ``src/`` numerics through the seam snapshot
    engine_input: bool = False
    #: some ``st.<widget>(..., key=...)`` binds it
    is_widget: bool = False
    #: nothing in ``app/`` reads it back through ``st.session_state``.  The seam
    #: may still read it from the snapshot - see ``engine_input``.
    write_only: bool = False
    notes: str = ""

    @property
    def is_family(self) -> bool:
        return "{" in self.name


_PLACEHOLDER = re.compile(r"\{[^{}]*\}")
_REGEX_CACHE = {}


def pattern_regex(pattern: str) -> "re.Pattern":
    """Compile a family pattern; every ``{placeholder}`` matches one segment.

    ``f"{decision_name}_default_value"`` and ``f"{d}_default_value"`` are the
    same family, so the placeholder's *name* is ignored on both sides.
    """
    cached = _REGEX_CACHE.get(pattern)
    if cached is None:
        parts = _PLACEHOLDER.split(pattern)
        cached = re.compile(".+".join(re.escape(part) for part in parts))
        _REGEX_CACHE[pattern] = cached
    return cached


def normalise(key: str) -> str:
    """A family pattern with its placeholder names stripped."""
    return _PLACEHOLDER.sub("{}", key)


def specificity(pattern: str) -> Tuple[int, int, int]:
    """How tightly a family pattern is pinned down; bigger wins a tie.

    Most literal characters first (``x_default_params`` beats ``x_{p}``), then
    fewest placeholders (``{d}_default_params`` beats ``{d}_default_param_{p}``,
    which would otherwise also match it), then the longer pattern.
    """
    literal = len(_PLACEHOLDER.sub("", pattern))
    return (literal, -len(_PLACEHOLDER.findall(pattern)), len(pattern))


def _row(name, default, initialised_by, category, **kwargs) -> KeySpec:
    return KeySpec(name=name, default=default, initialised_by=initialised_by,
                   category=category, **kwargs)


def _rows(rows, initialised_by, category, **common) -> Tuple[KeySpec, ...]:
    """A compact table of ``(name, default[, notes])`` sharing one origin."""
    out = []
    for row in rows:
        name, default = row[0], row[1]
        notes = row[2] if len(row) > 2 else ""
        out.append(KeySpec(name=name, default=default,
                           initialised_by=initialised_by, category=category,
                           notes=notes, **common))
    return tuple(out)


P1_INIT = "page1_common_params.initialize_widget_keys"
P1_INLINE = "page1_common_params.render_page1 (inline, just above the widget)"
P2_INIT = "page2_decisions.initialize_page2_widget_keys"
MODELS_INIT = "models.initialize_session_state"
MODELS_DEFAULTS = "models.initialize_session_state (defaults dict)"
MODELS_DECISIONS = "models.initialize_default_decision_parameters"
DON_INIT = "decision_tabs.donation_default.initialize_donation_widget_keys"
DON_YAML = "models.load_donation_coefficients_from_yaml"
DI_INIT = "decision_tabs.disclose_income.initialize_disclose_income_session_state"
DD_INIT = "decision_tabs.disclose_documents.initialize_disclose_documents_session_state"
RTD_INIT = "decision_tabs.rejected_transaction.initialize_rtd_session_state"


# --------------------------------------------------------------------------- #
# navigation
# --------------------------------------------------------------------------- #

_NAVIGATION = (
    _row("page", "page1", MODELS_INIT, "navigation",
         notes="'page1' | 'page2' | 'results'; app_enhanced_new.py dispatches on it."),
)


# --------------------------------------------------------------------------- #
# Page 1 - widget keys and the canonical values they mirror
# --------------------------------------------------------------------------- #

_PAGE1_WIDGETS = _rows((
    ("n_agents_input", "<n_agents>"),
    ("seed_input", "<seed>", "the Single-Run seed the seam pins a saved config against"),
    ("n_runs_input", "<n_runs>"),
    ("base_seed_input", "<base_seed>", "the Monte-Carlo seed"),
    ("periods_input", "<sim_params.periods>"),
    ("duration_hours_input", "<int(sim_params.duration_hours)>"),
    ("num_vendors_input", "<sim_params.num_vendors>"),
    ("single_vendor_price_input", "<sim_params.market_price>"),
    ("single_vendor_products_input", "<sim_params.vendor_products_avg>"),
    ("vendor_price_min_input", "<sim_params.vendor_price_min>",
     "R8 says these bounds are the vendor-scoring reference; the results page and "
     "the exports use them, src/decisions/purchasing_quantity.py does not (see the "
     "Known gap in docs/migration/README.md)"),
    ("vendor_price_max_input", "<sim_params.vendor_price_max>"),
    ("market_price_input", "<sim_params.market_price>"),
    ("vendor_products_min_input", "<sim_params.vendor_products_min>"),
    ("vendor_products_max_input", "<sim_params.vendor_products_max>"),
    ("vendor_products_avg_input", "<sim_params.vendor_products_avg>"),
    ("vendor_carryover_probability_slider", "<sim_params.vendor_carryover_probability>"),
    ("platform_markup_slider", "<sim_params.platform_markup>"),
    ("price_range_slider", "<sim_params.price_range>"),
    ("bidding_percentage_slider", "<sim_params.bidding_percentage>"),
    ("price_grid_input", "<sim_params.price_grid>"),
    ("lognormal_mu_input", "<sim_params.lognormal_mu, else 10.0>"),
    ("lognormal_sigma_input", "<sim_params.lognormal_sigma, else 0.5>"),
    ("lognormal_min_input", "<sim_params.lognormal_min, else 0.0>"),
    ("gg_k_input", "<sim_params.gg_k, else 1.5>"),
    ("gg_c_input", "<sim_params.gg_c, else 2.0>"),
    ("gg_lambda_input", "<sim_params.gg_lambda, else 20000.0>"),
    ("gg_min_input", "<sim_params.gg_min, else 0.0>"),
    ("dagum_a_input", "<sim_params.dagum_a, else 2.0>"),
    ("dagum_p_input", "<sim_params.dagum_p, else 1.5>"),
    ("dagum_b_input", "<sim_params.dagum_b, else 25000.0>"),
    ("dagum_min_input", "<sim_params.dagum_min, else 0.0>"),
    ("num_discount_categories_input", "<sim_params.num_discount_categories>"),
    ("num_fixed_categories_input", "<sim_params.num_fixed_categories>"),
    ("artificial_limit_input", "<sim_params.max_purchases_per_term>"),
    ("discount_threshold_input", "<sim_params.discount_income_threshold>"),
), P1_INIT, "page1-widget", is_widget=True)

_PAGE1_INLINE_WIDGETS = _rows((
    ("page1_simulation_execution_mode", "<'Snapshot' if sim_params.simulation_execution_mode == 'snapshot' else 'Live Simulation'>"),
    ("page1_simulation_mode", "<sim_params.simulation_mode>",
     "Single Run vs Monte Carlo; decides which seed key the seam reads"),
    ("page1_vendor_setup_mode", "<'Generate Randomly' if sim_params.vendor_config_mode == 'random' else 'Upload Vendor Config File'>"),
    ("page1_carryover_mode", "<derived from sim_params.override_carryover / global_carryover>"),
    ("page1_income_distribution", "<sim_params.income_distribution, else 'lognormal'>"),
    ("page1_population_mode", "<population_mode>"),
    ("page1_apply_limits", "<'Yes' if sim_params.apply_purchasing_limits else 'No'>"),
    ("single_vendor_carryover", "<sim_params.global_carryover>"),
    ("show_individual_agents_checkbox", "<show_individual_agents>",
     "on_change mirrors it into show_individual_agents"),
    ("uniform_purchasing_limit_input", "<uniform_purchasing_limit>"),
    ("purchasing_limit_{i}", "<sim_params.purchasing_limits[cat]>",
     "one per fixed income category; on_change writes purchasing_limits_temp"),
    ("lognormal_max_text_input", WIDGET, "free-text upper bound of the income axis"),
    ("gg_max_text_input", WIDGET),
    ("dagum_max_text_input", WIDGET),
    ("reset_page1_defaults", WIDGET,
     "button; raises StreamlitAPIException on click (documented quirk, left as is)"),
), P1_INLINE, "page1-widget", is_widget=True)

_PAGE1_MIRRORS = (
    _row("sim_params", "<SimulationParameters()>", MODELS_INIT, "page1-mirror",
         engine_input=True,
         notes="the canonical Page-1 object; every *_input / *_slider key mirrors "
               "one attribute, and the seam copies it into simulation_config."),
    _row("n_agents", 1000, MODELS_DEFAULTS, "page1-mirror", engine_input=True),
    _row("seed", 42, MODELS_DEFAULTS, "page1-mirror", engine_input=True,
         notes="fallback when seed_input is absent"),
    _row("n_runs", 10, MODELS_DEFAULTS, "page1-mirror",
         notes="Monte-Carlo only; the MC subprocess takes it as --runs"),
    _row("base_seed", 42, MODELS_DEFAULTS, "page1-mirror", engine_input=True,
         notes="fallback when base_seed_input is absent"),
    _row("population_mode", "Copula (synthetic)", MODELS_DEFAULTS, "page1-mirror",
         engine_input=True,
         notes="'Copula (synthetic)' | 'Research documentation' | 'Research baseline' "
               "| 'Compare all'; a saved config's population_mode overrides it (R14)."),
    _row("show_individual_agents", False, MODELS_DEFAULTS, "page1-mirror"),
    _row("nfic_manually_set", False, P1_INLINE, "page1-mirror",
         notes="True once the user typed a fixed-category count, so the auto-derived "
               "value stops overwriting it."),
    _row("uniform_purchasing_limit", 10, P1_INLINE, "page1-mirror"),
    _row("purchasing_limits_temp", "<dict of the per-category limits>", P1_INLINE,
         "page1-mirror",
         notes="scratch dict the per-category on_change callbacks write into."),
)


# --------------------------------------------------------------------------- #
# Page 2 - the canonical values the seam reads
# --------------------------------------------------------------------------- #

_PAGE2_MIRRORS = (
    _row("income_spec_mode", "categorical only", MODELS_DEFAULTS, "page2-mirror",
         engine_input=True,
         notes="'categorical only' | 'continuous only' | 'Compare both'; mirrors "
               "page2_tab_income_spec_mode.  R28: a DI/DD/RTD-only run no longer "
               "rewrites it - the effective mode travels in RunMetadata."),
    _row("sigma_in_copula", False, MODELS_DEFAULTS, "page2-mirror", engine_input=True,
         notes="donation sigma inside the copula; mirrors tab_sigma_in_copula."),
    _row("sigma_in_research", True, MODELS_DEFAULTS, "page2-mirror", engine_input=True,
         notes="donation sigma in the documentation population; mirrors "
               "tab_sigma_in_research."),
    _row("sigma_coefficient", 1.0, MODELS_DEFAULTS, "page2-mirror", engine_input=True,
         notes="R12: sigma_value = the one sigma constant x this coefficient."),
    _row("sigma_value_ui", 9.8995, P2_INIT, "page2-mirror",
         notes="display-only product of the sigma constant and the coefficient; "
               "R12 means the seam recomputes it rather than reading it."),
    _row("anchor_observed_weight", 0.75, MODELS_DEFAULTS, "page2-mirror",
         engine_input=True,
         notes="donation anchor split; mirrors tab_anchor_weight."),
    _row("donation_sigma_strategy", "overall", DON_INIT, "page2-mirror",
         engine_input=True, notes="'overall' | 'quintile'"),
    _row("donation_quintile_scale_factors", {"1": 1.0, "2": 1.0, "3": 1.0,
                                             "4": 1.0, "5": 1.0},
         DON_INIT, "page2-mirror", engine_input=True),
    _row("donation_adjustment_shift", 0.0, DON_YAML, "page2-mirror",
         engine_input=True, write_only=True,
         notes="R11: the shift comes from this key, never from the file.  Nothing "
               "in app/ reads it back - only the seam does, from the snapshot."),
    _row("donation_coeff_{name}", "<config/decisions.yaml regression_coefficients>",
         DON_YAML, "page2-mirror",
         notes="the flat display set the Donation tab edits (intercept, hh, linear, "
               "midsub, nosub, fullsub, q1..q5, the legacy q45, incoming, law, ug, "
               "grad).  R10: the engine runs the _cat / _cont set below, so this "
               "one is UI state."),
    _row("donation_coeff_intercept", "<config/decisions.yaml intercept>", DON_YAML,
         "page2-mirror",
         notes="the presence sentinel: models.initialize_session_state reloads the "
               "whole coefficient set from the file only when this key is absent."),
    _row("donation_coeff_{name}_{mode_suffix}",
         "<config/decisions.yaml regression_coefficients.{categorical,continuous}>",
         "models.load_coefficient_set", "page2-mirror", engine_input=True,
         notes="R10: suffix 'cat' / 'cont'; this is the set the seam hands the "
               "engine (build_plan._donation_coefficients)."),
    _row("intercept_override_values", "<{} until an intercept is typed>",
         "decision_tabs.donation_default.render_intercept_override_section",
         "page2-mirror",
         notes="R11: the typed categorical / continuous intercepts, kept in session "
               "instead of written back to the file."),
    _row("adjustment_override_values", "<{} until a shift is typed>",
         "decision_tabs.donation_default.render_adjustment_override_section",
         "page2-mirror"),
    _row("di_intercept_override_values", "<{} until an intercept is typed>",
         "decision_tabs.disclose_income.render_intercept_override", "page2-mirror"),
    _row("dd_intercept_override_values", "<{} until an intercept is typed>",
         "decision_tabs.disclose_documents.render_intercept_override", "page2-mirror"),
    _row("decision_params", "<DecisionParameters()>", MODELS_INIT, "page2-mirror",
         engine_input=True,
         notes="only .selected_decisions is used; a run swaps it and restores it "
               "through _pending_decisions_restore."),
    _row("page2_manual_selections", "<the current decision multiselect>",
         "page2_decisions.render_page2", "page2-mirror"),
    _row("page2_select_all_state", False, "page2_decisions.render_page2",
         "page2-mirror"),
)

_DI_MIRRORS = _rows((
    ("di_intercept", "<config/decisions.yaml intercept, else 0.75>"),
    ("di_wopb", "<anchor_weights.observed_prosocial, else 0.25>"),
    ("di_wpb", "<anchor_weights.prosocial_weight, else 0.50>"),
    ("di_income_mode", "<config income_mode, else 'categorical'>"),
    ("di_sigma_enabled", True, "R18: the tab default is ON, so an absent key must read True too"),
    ("di_sigma_in_copula", False),
    ("di_sigma_strategy", "<stochastic.sigma_strategy, else 'overall'>"),
    ("di_scale_factor", "<stochastic.scale_factor, else 1.0>"),
    ("di_quintile_scale_factors", "<stochastic.quintile_scale_factors, else all scale_factor>"),
), DI_INIT, "page2-mirror", engine_input=True)

_DD_MIRRORS = _rows((
    ("dd_intercept", "<config/decisions.yaml intercept, else the research default>"),
    ("dd_income_mode", "<config income_mode, else 'Categorical only'>"),
    ("dd_sigma_enabled", True, "R18"),
    ("dd_sigma_in_copula", False),
    ("dd_sigma_strategy", "<stochastic.sigma_strategy, else 'overall'>"),
    ("dd_scale_factor", "<stochastic.scale_factor, else 1.0>"),
    ("dd_quintile_scale_factors", "<stochastic.quintile_scale_factors, else all scale_factor>"),
), DD_INIT, "page2-mirror", engine_input=True)

_RTD_MIRRORS = _rows((
    ("rtd_income_mode", "Continuous only"),
    ("rtd_sigma_strategy", "overall", "decision-wide: one strategy for all four mechanisms"),
    ("rtd_scale_factor", 1.0, "decision-wide"),
    ("rtd_quintile_scale_factors", {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0}),
    ("rtd_intercept_{mech}", "<config intercepts[mech], else 0.0>",
     "mech in ttp / loyalty / wtp / risk_taking"),
    ("rtd_anchor_{mech}", "<mechanism anchor, else 'continuous'>"),
), RTD_INIT, "page2-mirror", engine_input=True)

_RTD_SIGMA_TOGGLES = _rows((
    ("rtd_sigma_enabled", True, "R18; written by the tab, read back only through the seam snapshot"),
    ("rtd_sigma_in_copula", False, "written by the tab, read back only through the seam snapshot"),
), RTD_INIT, "page2-mirror", engine_input=True, write_only=True)

_RTD_DISPLAY = (
    _row("rtd_run_element", None, RTD_INIT, "page2-mirror",
         notes="which Decision-4 element the per-element Run button selected; "
               "display and export only - the model always computes all four."),
)


# --------------------------------------------------------------------------- #
# Page 2 - widget keys
# --------------------------------------------------------------------------- #

#: ``clear_input_field_cache()`` deletes sixteen ``donation_coeff_*_input`` keys
#: whenever the income mode changes, but no widget binds them any more - the
#: coefficient number inputs were removed - so the list only ever finds them
#: absent.  Left exactly as it is; a removal candidate for the owner, see
#: docs/migration/session-state.md.
_LEGACY_COEFF_INPUTS = (
    _row("donation_coeff_{name}_input", "<never written>",
         "nothing - a stale cache-clearing list", "page2-widget",
         notes="deleted by decision_tabs.donation_default.clear_input_field_cache; "
               "no Streamlit widget binds these keys."),
)

_DONATION_WIDGETS = _rows((
    ("tab_sigma_in_copula", "<sigma_in_copula, else False>"),
    ("tab_sigma_in_research", "<sigma_in_research, else True>"),
    ("tab_sigma_coefficient_{mode_suffix}", "<sigma_coefficient, else 1.0>",
     "mode_suffix is always 'stochastic'; R12 multiplies the one sigma constant by this slider"),
    ("tab_anchor_weight", "<anchor_observed_weight, else 0.75>"),
    ("donation_tab_sigma_strategy_{mode_suffix}", "<donation_sigma_strategy>"),
    ("donation_tab_sigma_q{level}_{mode_suffix}", "<donation_quintile_scale_factors[level]>"),
    ("override_categorical_intercept", "<the current categorical intercept>"),
    ("override_continuous_intercept", "<the current continuous intercept>"),
    ("override_adjustment_shift", "<donation_adjustment_shift>"),
    ("reset_intercept_btn", WIDGET),
    ("reset_adjustment_btn", WIDGET),
    ("reset_config_btn", WIDGET),
), DON_INIT + " / render_donation_default_tab", "page2-widget", is_widget=True)

_DI_WIDGETS = _rows((
    ("di_tab_income_mode", "<di_income_mode>"),
    ("di_tab_sigma_enabled", "<di_sigma_enabled>"),
    ("di_tab_sigma_in_copula", "<di_sigma_in_copula, else False>"),
    ("di_tab_sigma_coefficient_{mode_suffix}", "<di_scale_factor>"),
    ("di_tab_sigma_strategy_{mode_suffix}", "<di_sigma_strategy>"),
    ("di_tab_sigma_q{level}_{mode_suffix}", "<di_quintile_scale_factors[level]>"),
    ("di_wopb_widget", "<di_wopb>"),
    ("di_wpb_widget", "<di_wpb>"),
    ("di_override_intercept", "<di_intercept>"),
    ("di_reset_btn", WIDGET),
    ("di_reload_btn", WIDGET),
), DI_INIT + " / render_disclose_income_tab", "page2-widget", is_widget=True)

_DD_WIDGETS = _rows((
    ("dd_tab_income_mode", "<dd_income_mode>"),
    ("dd_tab_sigma_enabled", "<dd_sigma_enabled>"),
    ("dd_tab_sigma_in_copula", "<dd_sigma_in_copula, else False>"),
    ("dd_tab_sigma_coefficient_{mode_suffix}", "<dd_scale_factor>"),
    ("dd_tab_sigma_strategy_{mode_suffix}", "<dd_sigma_strategy>"),
    ("dd_tab_sigma_q{level}_{mode_suffix}", "<dd_quintile_scale_factors[level]>"),
    ("dd_override_intercept", "<dd_intercept>"),
    ("dd_reset_btn", WIDGET),
    ("dd_reload_btn", WIDGET),
), DD_INIT + " / render_disclose_documents_tab", "page2-widget", is_widget=True)

_RTD_WIDGETS = _rows((
    ("rtd_tab_income_mode", "<rtd_income_mode>"),
    ("rtd_tab_sigma_enabled", "<rtd_sigma_enabled>"),
    ("rtd_tab_sigma_in_copula", "<rtd_sigma_in_copula>"),
    ("rtd_tab_sigma_coefficient", "<rtd_scale_factor>"),
    ("rtd_tab_sigma_strategy", "<rtd_sigma_strategy>"),
    ("rtd_tab_sigma_q{level}", "<rtd_quintile_scale_factors[level]>"),
    ("rtd_tab_intercept_{mech}", "<rtd_intercept_{mech}>"),
    ("rtd_tab_anchor_{mech}", "<rtd_anchor_{mech}>"),
    ("rtd_reset_btn", WIDGET),
    ("rtd_reset_{mech}_btn", WIDGET, "per-element reset"),
    ("rtd_run_{mech}_btn", WIDGET, "per-element run"),
), RTD_INIT + " / render_rejected_transaction_tab", "page2-widget", is_widget=True)

_PAGE2_WIDGETS = _rows((
    ("page2_tab_income_spec_mode", "<income_spec_mode mapped onto the radio options>"),
    ("page2_manual_multiselect", "<page2_manual_selections>"),
    ("page2_select_all_checkbox", "<page2_select_all_state>"),
    ("clear_donation_config", WIDGET),
    ("clear_disclose_income_config", WIDGET),
    ("run_complete_simulation", WIDGET),
    ("run_complete_simulation_disabled", WIDGET, "the disabled twin of the button above"),
), P2_INIT + " / page2_decisions.render_page2", "page2-widget", is_widget=True)

_RUN_BUTTONS = _rows((
    ("run_{decision_name}_only_btn", WIDGET,
     "the per-decision Run button; the Decision-4 tab reads its state back"),
    ("run_complete_from_{decision_name}_btn", WIDGET),
    ("run_complete_from_{decision_name}_btn_disabled", WIDGET),
), "decision_execution.render_decision_run_controls", "page2-widget", is_widget=True)


# --------------------------------------------------------------------------- #
# tab persistence dicts
# --------------------------------------------------------------------------- #

_TAB_PERSISTENCE = (
    _row("donation_tab_persistence", {}, DON_INIT, "tab-persistence",
         notes="written only from an on_change callback (save_to_donation_storage); "
               "read back before the widget renders (restore_widget_from_storage)."),
    _row("disclose_income_tab_persistence", {}, DI_INIT, "tab-persistence",
         notes="wiped by the di_ prefix reset."),
    _row("disclose_documents_tab_persistence", {}, DD_INIT, "tab-persistence",
         notes="wiped by the dd_ prefix reset."),
    _row("rejected_transaction_tab_persistence", {}, RTD_INIT, "tab-persistence",
         notes="wiped by the rtd_ prefix reset; the per-element reset pops just "
               "that element's entries."),
)


# --------------------------------------------------------------------------- #
# saved configurations
# --------------------------------------------------------------------------- #

_SAVED_CONFIG = (
    _row("selected_decision_configs", {}, "state.saved_configs.get_selected_decision_configs",
         "saved-config", engine_input=True,
         notes="R13: only explicit 'Use This Config' selections.  R14: each entry "
               "pins seed, agent count, population mode, that decision's parameters "
               "and the sha256 of the columns it produced.  R17: a saved Disclose "
               "Documents config is applied exactly like a Disclose Income one."),
    _row("custom_coefficients", "<{} until an individual donation run>",
         "decision_execution.run_individual_decision", "saved-config",
         notes="the donation coefficient set the last individual run used."),
)


# --------------------------------------------------------------------------- #
# run flags
# --------------------------------------------------------------------------- #

_RUN_FLAGS = (
    _row("custom_decisions", "<the decisions this click ran>",
         "decision_execution.run_individual_decision / run_combined_simulation",
         "run-flag", engine_input=True,
         notes="every results-page branch detects an individual run with "
               "custom_decisions == ['x'] and default_decisions == []."),
    _row("default_decisions", "<the decisions left on their default branch>",
         "decision_execution.run_individual_decision / run_combined_simulation",
         "run-flag", engine_input=True),
    _row("_pending_decisions_restore", "<{'selected_decisions': [...]} >",
         "decision_execution.run_individual_decision", "run-flag",
         notes="run_full_simulation reruns before the caller can restore "
               "decision_params, so the original selection is parked here."),
    _row("_reset_intercept_flag", True,
         "decision_tabs.donation_default.reset_intercepts_to_defaults", "run-flag",
         notes="one-rerun flag so the number_input picks the reset value up."),
    _row("_reset_adjustment_flag", True,
         "decision_tabs.donation_default.reset_adjustment_to_defaults", "run-flag"),
    _row("_reset_config_to_defaults_flag", True,
         "decision_tabs.donation_default.render_actions_and_management_section",
         "run-flag"),
    _row("simulation_running", False, MODELS_DEFAULTS, "run-flag", write_only=True,
         notes="seeded and never read - removal candidate."),
    _row("save_results", True, MODELS_DEFAULTS, "run-flag", write_only=True,
         notes="the disabled save-to-disk feature; seeded and never read - "
               "removal candidate."),
)


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

_RESULTS = (
    _row("simulation_results", None, MODELS_INIT, "results",
         notes="{result_key: DataFrame} after a run; None after a Monte-Carlo run."),
    _row("mc_results", None, MODELS_INIT, "results",
         notes="{'summary', 'detailed', 'log'} from the Monte-Carlo subprocess."),
    _row("_run_metadata", "<RunMetadata as a dict>", "simulation.run_full_simulation",
         "results",
         notes="R28: the results page reads the run's real population / income mode "
               "from here, so the chart titles cannot drift from what ran."),
    _row("vendors", "<df.attrs['vendors'] of the first result>",
         "simulation.run_full_simulation", "results"),
    _row("individual_results", {}, MODELS_DEFAULTS, "results", write_only=True,
         notes="seeded and never read - removal candidate."),
)

_RESULT_WIDGETS = _rows((
    ("inline_select_{result_key}", WIDGET, "donation 'Use This Config' radio"),
    ("di_inline_select_{result_key}", WIDGET),
    ("dd_inline_select_{result_key}", WIDGET),
    ("clear_conflict_{decision_name}_{conflicting}", WIDGET,
     "shown when a saved config's seed clashes with another one"),
    ("clear_di_selection", WIDGET),
    ("clear_dd_selection", WIDGET),
    ("clear_selection_top", WIDGET),
    ("save_single_config", WIDGET),
    ("run_complete_from_results", WIDGET),
    ("run_complete_from_results_disabled", WIDGET),
    ("donation_hist_{title_suffix}", WIDGET, "chart keys built by the shared helpers"),
    ("disclose_income_pie_{title_suffix}", WIDGET),
    ("disclose_documents_pie_{title_suffix}", WIDGET),
    ("di_raw_hist_{title_suffix}", WIDGET),
    ("dd_raw_hist_{title_suffix}", WIDGET),
    ("rejected_transaction_option_chart", WIDGET),
    ("{decision_name}_option_{idx}_chart", WIDGET),
    ("rtd_dl_{mech}{chart_suffix}", WIDGET, "per-element Decision-4 download"),
    ("rtd_model_download{chart_suffix}", WIDGET),
    ("rtd_export_section_download", WIDGET),
    ("vendor_choice_weights_chart", WIDGET),
    ("show_all_proximity_matrix", WIDGET),
), "the results renderers (R30: they never write settings)", "results",
    is_widget=True)


# --------------------------------------------------------------------------- #
# default-decision parameters and their shadow store
# --------------------------------------------------------------------------- #

_DEFAULTS = (
    _row("_persistent_defaults", {}, "decision_tabs.default_config.save_to_persistent_storage",
         "defaults", engine_input=True,
         notes="the shadow store: it outlives a widget key that Page 2 never "
               "rendered, and the seam consults it before the session key."),
    _row("_default_params_initialized", True, MODELS_DECISIONS, "defaults",
         write_only=True, notes="debug marker; nothing reads it - removal candidate."),
    _row("{decision_name}_default_probability_y", "<DEFAULT_DECISION_VALUES probability_y, else 0.5>",
         MODELS_DECISIONS, "defaults", engine_input=True, is_widget=True,
         notes="random_probability decisions."),
    _row("{decision_name}_default_params", "<DEFAULT_DECISION_VALUES default_selection>",
         MODELS_DECISIONS, "defaults", engine_input=True,
         notes="checkbox_selection decisions (vendor_choice_weights)."),
    _row("{decision_name}_default_param_{param_key}", "<param in default_selection>",
         MODELS_DECISIONS, "defaults", is_widget=True,
         notes="one checkbox per weight parameter."),
    _row("{decision_name}_default_selection", "<DEFAULT_DECISION_VALUES default_option, else ''>",
         MODELS_DECISIONS, "defaults", engine_input=True, is_widget=True,
         notes="radio_selection decisions."),
    _row("{decision_name}_priority_template", "<DEFAULT_DECISION_VALUES priority_template>",
         MODELS_DECISIONS, "defaults", engine_input=True,
         notes="prioritized_selection decisions (Decision 4's ranking)."),
    _row("{decision_name}_default_value", "<DEFAULT_DECISION_VALUES numeric/str placeholder>",
         MODELS_DECISIONS, "defaults", engine_input=True, is_widget=True),
    _row("final_donation_rate_default_value", 0.10, MODELS_DECISIONS, "defaults",
         engine_input=True, is_widget=True,
         notes="the {decision_name}_default_value of decision 13, plus one extra "
               "rule: saving a donation_default config syncs it (and its "
               "_persistent_defaults entry) to that run's mean donation."),
    _row("{decision_name}_probability_y", "<never written>",
         "nothing - read-only fallback", "defaults", engine_input=True,
         notes="the post-simulation override the Results page was meant to write; "
               "the seam still looks for it first."),
    _row("{decision_name}_selection", "<never written>",
         "nothing - read-only fallback", "defaults", engine_input=True,
         notes="same for checkbox_selection decisions."),
    _row("vendor_choice_weights_selection", "<never written>",
         "nothing - read-only fallback", "defaults", engine_input=True,
         notes="the concrete instance of {decision_name}_selection the comment in "
               "the seam names."),
    _row("rejected_transaction_defaults_option", "<never written>",
         "nothing - read-only fallback", "defaults", engine_input=True,
         notes="the legacy post-simulation key for decision 4's radio; the seam "
               "and get_actual_default_value both still look for it first."),
    _row("rejected_transaction_option_selection", "<never written>",
         "nothing - read-only fallback", "defaults", engine_input=True,
         notes="the same for decision 11."),
    _row("{decision_name}_config", "<never written>",
         "nothing - read-only fallback", "defaults",
         notes="third-priority lookup for a numeric default."),
    _row("reset_all_defaults", WIDGET,
         "decision_tabs.default_config.render_default_decisions_config", "defaults",
         is_widget=True),
    _row("{decision_name}_reset", WIDGET,
         "decision_tabs.default_config.render_decision_default_config", "defaults",
         is_widget=True),
    _row("{decision_name}_add_btn", WIDGET,
         "decision_tabs.default_config.render_prioritized_default_config", "defaults",
         is_widget=True),
    _row("{decision_name}_add_selector", WIDGET,
         "decision_tabs.default_config.render_prioritized_default_config", "defaults",
         is_widget=True),
    _row("{decision_name}_remove_{i}", WIDGET,
         "decision_tabs.default_config.render_prioritized_default_config", "defaults",
         is_widget=True),
)


REGISTRY: Tuple[KeySpec, ...] = (
    _NAVIGATION
    + _PAGE1_WIDGETS + _PAGE1_INLINE_WIDGETS + _PAGE1_MIRRORS
    + _PAGE2_MIRRORS + _DI_MIRRORS + _DD_MIRRORS + _RTD_MIRRORS + _RTD_SIGMA_TOGGLES + _RTD_DISPLAY
    + _LEGACY_COEFF_INPUTS + _DONATION_WIDGETS + _DI_WIDGETS + _DD_WIDGETS + _RTD_WIDGETS
    + _PAGE2_WIDGETS + _RUN_BUTTONS
    + _TAB_PERSISTENCE + _SAVED_CONFIG + _RUN_FLAGS
    + _RESULTS + _RESULT_WIDGETS + _DEFAULTS
)

_BY_NAME = {spec.name: spec for spec in REGISTRY}
_BY_NAME.update({normalise(spec.name): spec for spec in REGISTRY if spec.is_family})
_FAMILIES = tuple(spec for spec in REGISTRY if spec.is_family)


def matching_families(key: str) -> Tuple[KeySpec, ...]:
    """Every family pattern whose regex covers ``key``, most specific first."""
    hits = [spec for spec in _FAMILIES if pattern_regex(spec.name).fullmatch(key)]
    hits.sort(key=lambda spec: specificity(spec.name), reverse=True)
    return tuple(hits)


def lookup(key: str) -> Optional[KeySpec]:
    """The entry for ``key``: exact name, then family pattern (most specific)."""
    spec = _BY_NAME.get(key)
    if spec is not None:
        return spec
    spec = _BY_NAME.get(normalise(key))
    if spec is not None:
        return spec
    hits = matching_families(key)
    return hits[0] if hits else None


__all__ = [
    "CATEGORIES", "DYNAMIC_KEY_VARIABLES", "PREFIX_DELETE_PREFIXES", "WIDGET",
    "KeySpec", "REGISTRY", "lookup", "matching_families", "normalise",
    "pattern_regex", "specificity",
]
