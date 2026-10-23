# app/state/registry.py
"""Registry of every ``st.session_state`` key the app uses.

One :class:`KeySpec` per key, or per key *family* when the code builds the name
with an f-string (``f"{decision_name}_default_value"``).  The registry is plain
data - it imports nothing from the app and never touches Streamlit - so a test,
a doc build or a reviewer can read it without starting a session.

``tests/test_session_key_registry.py`` walks ``app/`` with ``ast`` and holds the
code against this file in both directions: a key the code touches must have an
entry, and an entry no code touches must go.
``docs/session-state.md`` explains the three persistence layers the
``category`` field names and lists the keys that are written but never read.

Conventions for ``default``:

* a Python value - the literal the initialiser stores;
* :data:`WIDGET` - Streamlit owns the value; nothing seeds it up front;
* ``"<expr>"`` (angle brackets) - the value is copied from ``expr`` when the key
  is first created, e.g. ``"<sim_params.periods>"``.
"""

import re
from collections import namedtuple
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
    "saved-config",      # "Use This Config" pins
    "run-flag",          # what the click that started a run recorded
    "results",           # run output and the widgets that display it
    "defaults",          # default-decision parameters and their shadow store
)

#: Variables that name a key only at run time - a helper's parameter, or a loop
#: over ``st.session_state.keys()``.  The scanner cannot resolve these; they are
#: reviewed by hand and listed here so a *new* one fails the test.
#:
#: ``key``        - the parameter of ``state.widgets.sync_widget_key`` (the key of
#:                  every ``stateful()`` widget); of ``default_config.save_to_persistent_storage``
#:                  / ``restore_from_persistent_storage``; of the
#:                  Decision-4 download helper ``transaction_viz._rtd_excel_download``
#:                  (``rtd_dl_{mech}{chart_suffix}``, ``rtd_dl_aggregation{chart_suffix}``,
#:                  ``rtd_model_download[_combined]{chart_suffix}``); and the loop variable
#:                  of the Disclose Income / Documents initialisers' defaults loops, of
#:                  the ``dd_`` / ``di_`` / ``rtd_`` prefix resets and of the
#:                  "Clear Results" wipe.
#: ``k``          - the per-category purchasing-limit key on Page 1
#:                  (``purchasing_limit_{i}``, bound in a lambda default).
#: ``chart_key``  - the plotly key built by the shared chart helpers
#:                  (``donation_hist_*`` / ``*_pie_*`` / ``*_raw_hist_*`` and the
#:                  Decision-4 density charts).
#: ``widget_key`` / ``canonical_key`` - the decision tabs' seeding helpers
#:                  (``decision_tabs._tab_state``: ``restore`` /
#:                  ``seed_if_absent`` / ``seed_if_changed``): the widget key they
#:                  are handed, and the canonical key ``restore`` looks it up in
#:                  TAB_WIDGET_CANONICAL below.  Every caller still binds the
#:                  widget key literally (``key=``) or through
#:                  KEY_PARAMETER_TABLES, and every canonical key of the table is
#:                  written literally next to its widget (the test holds the
#:                  table to both).
DYNAMIC_KEY_VARIABLES = ("key", "k", "chart_key", "widget_key", "canonical_key")

#: ``[k for k in st.session_state.keys() if k.startswith(<prefix>)]`` - the
#: whole-decision resets, and the per-category purchasing limits Page 1's reset
#: drops (a reset restores exactly what a fresh session shows).  Every key
#: beginning with one of these is deleted.
PREFIX_DELETE_PREFIXES = ("dd_", "di_", "rtd_", "purchasing_limit_")


# --------------------------------------------------------------------------- #
# Keys built from per-tab parameters
# --------------------------------------------------------------------------- #
# Two modules draw the same widgets for several decision tabs and name their keys
# with f-strings over a per-tab row of parameters:
#
#   f"{keys.wp}sigma_coefficient{keys.suffix}"   ->  tab_sigma_coefficient_stochastic,
#                                                    di_tab_sigma_coefficient_stochastic,
#                                                    dd_tab_sigma_coefficient_stochastic,
#                                                    rtd_tab_sigma_coefficient
#
# The module reads its rows from HERE (``decision_tabs._controls`` builds its specs
# from CONTROL_KEYS, ``decision_tabs._disclosure`` from DISCLOSURE_KEYS), so the
# keys the app uses are these values, and ``tests/test_session_key_registry.py``
# expands every such f-string of the module over every row - a placeholder
# ``{<name>.<field>}`` whose last segment is a field of the row - and holds each
# expansion against the rows below (one row per concrete key, as for every other
# key).  A placeholder ``{level}`` that is not a field stays a family placeholder.
# A placeholder that is neither is a dynamic key and fails the test unless it is in
# DYNAMIC_KEY_VARIABLES.

#: The four tabs' shared controls (σ strategy radio, coefficient and per-quintile
#: sliders, the two stochastic tick boxes, the income-mode radio).
#:   dp       - prefix of the canonical keys and of the strategy / quintile widget keys
#:   wp       - prefix of the coefficient slider's and the tick boxes' widget keys
#:   sp       - prefix of the copula / research canonical keys ("" on the Donation tab)
#:   suffix   - the widget-key suffix of the σ controls ("" on Decision 4)
#:   research - the research-box stem: Donation "sigma_in_research", the others "sigma_enabled"
#:   coefficient   - the canonical key of the σ coefficient slider (Donation
#:                   "donation_sigma_coefficient_setting": the engine input
#:                   sigma_coefficient is derived from it, see _PAGE2_MIRRORS)
#:   income_widget - the income-mode radio's widget key
#:   income        - the canonical income-mode key
ControlKeys = namedtuple("ControlKeys",
                         "dp wp sp suffix research coefficient income_widget income")
CONTROL_KEYS = {
    "donation_default": ControlKeys(
        dp="donation_", wp="tab_", sp="", suffix="_stochastic", research="sigma_in_research",
        coefficient="donation_sigma_coefficient_setting", income_widget="page2_tab_income_spec_mode",
        income="income_spec_mode"),
    "disclose_income": ControlKeys(
        dp="di_", wp="di_tab_", sp="di_", suffix="_stochastic", research="sigma_enabled",
        coefficient="di_scale_factor", income_widget="di_tab_income_mode",
        income="di_income_mode"),
    "disclose_documents": ControlKeys(
        dp="dd_", wp="dd_tab_", sp="dd_", suffix="_stochastic", research="sigma_enabled",
        coefficient="dd_scale_factor", income_widget="dd_tab_income_mode",
        income="dd_income_mode"),
    "rejected_transaction_defaults": ControlKeys(
        dp="rtd_", wp="rtd_tab_", sp="rtd_", suffix="", research="sigma_enabled",
        coefficient="rtd_scale_factor", income_widget="rtd_tab_income_mode",
        income="rtd_income_mode"),
}

#: The Disclose Income / Disclose Documents twin machinery (intercept override,
#: resets, actions): ``dp`` prefixes every key of the tab, ``wp`` + ``suffix`` the
#: σ widget keys the reset writes.
DisclosureKeys = namedtuple("DisclosureKeys", "dp wp suffix")
DISCLOSURE_KEYS = {
    "disclose_income": DisclosureKeys(dp="di_", wp="di_tab_", suffix="_stochastic"),
    "disclose_documents": DisclosureKeys(dp="dd_", wp="dd_tab_", suffix="_stochastic"),
}

#: module (path from the repository root) -> the rows its f-string keys expand over
KEY_PARAMETER_TABLES = {
    "app/pages/decision_tabs/_controls.py": CONTROL_KEYS,
    "app/pages/decision_tabs/_disclosure.py": DISCLOSURE_KEYS,
}


# --------------------------------------------------------------------------- #
# Where a decision-tab widget's value lives when its widget is not drawn
# --------------------------------------------------------------------------- #
#: widget key -> (canonical key, entry): the canonical home of every decision-tab
#: widget that ``decision_tabs._tab_state.restore`` seeds.  Streamlit drops a
#: widget key once its widget is not drawn (another page, the decision
#: unselected, the σ controls hidden); the canonical key is written right after
#: the widget on every render and is never dropped, so ``restore`` seeds an absent
#: widget key from it (the live widget key always wins).  ``entry`` is None for a
#: flat key, or the entry of a dict-valued canonical key (the per-quintile
#: sliders map to entries '1'..'5' of ``<prefix>quintile_scale_factors``).  Until
#: 2026-10-22 a per-tab ``*_tab_persistence`` dict, filled from ``on_change``,
#: played this part; it only ever held what the canonical key held (the Donation
#: σ coefficient slider, whose engine-facing key is overwritten in quintile mode,
#: got a canonical key of its own, donation_sigma_coefficient_setting).
#:
#: ``tests/test_session_key_registry.py`` holds this table to CONTROL_KEYS (the
#: shared controls name their keys from those rows) and to the registry rows.
#: A widget drawn through ``restore`` that is not listed here raises at its first
#: draw.
_QUINTILE_LEVELS = ("1", "2", "3", "4", "5")
_RTD_MECHANISMS = ("ttp", "loyalty", "wtp", "risk_taking", "flexibility")
TAB_WIDGET_CANONICAL = {
    # Donation Default (decision 3)
    "tab_sigma_in_copula": ("sigma_in_copula", None),
    "tab_sigma_in_research": ("sigma_in_research", None),
    "donation_tab_sigma_strategy_stochastic": ("donation_sigma_strategy", None),
    "tab_sigma_coefficient_stochastic": ("donation_sigma_coefficient_setting", None),
    **{f"donation_tab_sigma_q{level}_stochastic": ("donation_quintile_scale_factors", level)
       for level in _QUINTILE_LEVELS},
    "tab_anchor_weight": ("anchor_observed_weight", None),
    "page2_tab_income_spec_mode": ("income_spec_mode", None),
    # Disclose Income (decision 1)
    "di_tab_sigma_in_copula": ("di_sigma_in_copula", None),
    "di_tab_sigma_enabled": ("di_sigma_enabled", None),
    "di_tab_sigma_strategy_stochastic": ("di_sigma_strategy", None),
    "di_tab_sigma_coefficient_stochastic": ("di_scale_factor", None),
    **{f"di_tab_sigma_q{level}_stochastic": ("di_quintile_scale_factors", level)
       for level in _QUINTILE_LEVELS},
    "di_wopb_widget": ("di_wopb", None),
    "di_wpb_widget": ("di_wpb", None),
    "di_tab_income_mode": ("di_income_mode", None),
    # Disclose Documents (decision 2)
    "dd_tab_sigma_in_copula": ("dd_sigma_in_copula", None),
    "dd_tab_sigma_enabled": ("dd_sigma_enabled", None),
    "dd_tab_sigma_strategy_stochastic": ("dd_sigma_strategy", None),
    "dd_tab_sigma_coefficient_stochastic": ("dd_scale_factor", None),
    **{f"dd_tab_sigma_q{level}_stochastic": ("dd_quintile_scale_factors", level)
       for level in _QUINTILE_LEVELS},
    "dd_tab_income_mode": ("dd_income_mode", None),
    # Rejected Transaction Defaults (decision 4)
    "rtd_tab_sigma_in_copula": ("rtd_sigma_in_copula", None),
    "rtd_tab_sigma_enabled": ("rtd_sigma_enabled", None),
    "rtd_tab_sigma_strategy": ("rtd_sigma_strategy", None),
    "rtd_tab_sigma_coefficient": ("rtd_scale_factor", None),
    **{f"rtd_tab_sigma_q{level}": ("rtd_quintile_scale_factors", level)
       for level in _QUINTILE_LEVELS},
    **{f"rtd_tab_intercept_{mech}": (f"rtd_intercept_{mech}", None)
       for mech in _RTD_MECHANISMS},
    "rtd_tab_flex_observed_weight": ("rtd_flex_observed_weight", None),
    "rtd_tab_income_mode": ("rtd_income_mode", None),
}


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


P1_INIT = "page1_common_params.initialize_widget_keys (from page1_widget_values)"
P1_INLINE = "page1_common_params.render_page1 (inline, just above the widget)"
P2_INIT = "page2_decisions.initialize_page2_widget_keys"   # mirrors canonical keys only (2026-10-22)
MODELS_INIT = "models.initialize_session_state"
MODELS_DEFAULTS = "models.initialize_session_state (models.SESSION_DEFAULTS)"
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
    _row("_widget_render_log", {}, "state.widgets.begin_script_run", "navigation",
         notes="widget key -> argument signature of every stateful() widget drawn in "
               "the current run (what the browser will hold)."),
    _row("_widget_render_log_previous", {}, "state.widgets.begin_script_run",
         "navigation",
         notes="the previous run's render log; stateful() re-pushes a key's value "
               "to the browser when its widget is not in it (2026-10-07)."),
    _row("_widget_render_log_complete", False, "state.widgets.begin_script_run",
         "navigation",
         notes="True once the current run reached end_script_run(); after a run that "
               "a newer interaction stopped early, the next run MERGES its render log "
               "into the previous one instead of replacing it."),
)


# --------------------------------------------------------------------------- #
# Page 1 - widget keys and the canonical values they mirror
# --------------------------------------------------------------------------- #

_PAGE1_WIDGETS = _rows((
    ("n_agents_input", "<n_agents>"),
    ("n_runs_input", "<n_runs>"),
    ("periods_input", "<sim_params.periods>"),
    ("duration_hours_input", "<int(sim_params.duration_hours)>"),
    ("num_vendors_input", "<sim_params.num_vendors>"),
    ("single_vendor_price_input", "<sim_params.market_price>"),
    ("single_vendor_products_input", "<sim_params.vendor_products_avg>"),
    ("vendor_price_min_input", "<sim_params.vendor_price_min>",
     "these bounds are the vendor-scoring reference everywhere -- the results "
     "page, the exports and the engine "
     "(src/decisions/purchasing_quantity.py reads them via get_simulation_param) -- "
     "so the three cannot disagree; market_price is the reference price"),
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
    ("page1_simulation_execution_mode", "<'Snapshot' if sim_params.simulation_execution_mode == 'snapshot' else 'Live Simulation'>"),
    ("page1_simulation_mode", "<sim_params.simulation_mode>",
     "Single Run vs Monte Carlo; decides which seed key the seam reads"),
    ("page1_vendor_setup_mode", "<'Generate Randomly' if sim_params.vendor_config_mode == 'random' else 'Upload Vendor Config File'>"),
    ("page1_carryover_mode", "<derived from sim_params.override_carryover / global_carryover>"),
    ("page1_income_distribution", "<sim_params.income_distribution, else 'lognormal'>"),
    ("page1_population_mode", "<population_mode>"),
    ("page1_apply_limits", "<'Yes' if sim_params.apply_purchasing_limits else 'No'>"),
    ("single_vendor_carryover", "<sim_params.global_carryover>"),
), P1_INIT, "page1-widget", is_widget=True)

_PAGE1_SEED_WIDGETS = (
    _row("seed_input", "<seed>", P1_INIT, "page1-widget", is_widget=True,
         engine_input=True,
         notes="the Single-Run seed; the seam reads it before seed "
               "(build_plan.resolve_seed_and_n) unless a saved config pins the seed"),
    _row("base_seed_input", "<base_seed>", P1_INIT, "page1-widget", is_widget=True,
         engine_input=True,
         notes="the Monte-Carlo seed; the seam reads it before base_seed"),
)

_PAGE1_INLINE_WIDGETS = _rows((
    ("show_individual_agents_checkbox", "<show_individual_agents>",
     "on_change mirrors it into show_individual_agents"),
    ("uniform_purchasing_limit_input", "<uniform_purchasing_limit>"),
    ("purchasing_limit_{i}", "<sim_params.purchasing_limits[cat]>",
     "one per fixed income category; on_change writes purchasing_limits_temp"),
    ("lognormal_max_text_input", WIDGET, "free-text upper bound of the income axis"),
    ("gg_max_text_input", WIDGET),
    ("dagum_max_text_input", WIDGET),
    ("reset_page1_defaults", WIDGET,
     "button; on_click puts sim_params, the Page-1 session values and every Page-1 "
     "widget key back to what a fresh session shows"),
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
         notes="Monte-Carlo only; the number of in-process repetitions"),
    _row("base_seed", 42, MODELS_DEFAULTS, "page1-mirror", engine_input=True,
         notes="fallback when base_seed_input is absent"),
    _row("population_mode", "Copula (synthetic)", MODELS_DEFAULTS, "page1-mirror",
         engine_input=True,
         notes="'Copula (synthetic)' | 'Research documentation' | 'Research baseline' "
               "| 'Compare all'; a saved config's population_mode overrides it (to "
               "reproduce the pinned run)."),
    _row("show_individual_agents", False, MODELS_DEFAULTS, "page1-mirror"),
    _row("nfic_manually_set", False, P1_INIT, "page1-mirror",
         notes="True once the user typed a fixed-category count, so the auto-derived "
               "value stops overwriting it."),
    _row("uniform_purchasing_limit", 10, P1_INIT, "page1-mirror"),
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
               "page2_tab_income_spec_mode.  A DI/DD/RTD-only run never rewrites it "
               "- the effective mode travels in RunMetadata, so a run cannot change "
               "the mode shown on Page 2."),
    _row("sigma_in_copula", False, MODELS_DEFAULTS, "page2-mirror", engine_input=True,
         notes="donation sigma inside the copula; mirrors tab_sigma_in_copula."),
    _row("sigma_in_research", True, MODELS_DEFAULTS, "page2-mirror", engine_input=True,
         notes="donation sigma in the documentation population; mirrors "
               "tab_sigma_in_research."),
    _row("sigma_coefficient", 1.0, MODELS_DEFAULTS, "page2-mirror", engine_input=True,
         notes="sigma_value = the one sigma constant x this coefficient.  The "
               "coefficient IN EFFECT: the slider (donation_sigma_coefficient_setting) "
               "while the σ mode is 'overall', 1.0 while it is 'quintile' "
               "(render_donation_default_tab writes it after the σ controls; the "
               "whole-tab reset writes the file's, before the Overview) - so "
               "it cannot seed the slider."),
    _row("donation_sigma_coefficient_setting", "<absent until the σ coefficient slider is drawn>",
         "decision_tabs._controls.render_coefficient_slider", "page2-mirror",
         notes="the Donation σ coefficient slider's own canonical home, written "
               "right after the slider on every render (the Disclose Income / "
               "Documents / Decision 4 twins are di_/dd_/rtd_scale_factor); "
               "TAB_WIDGET_CANONICAL seeds the slider from it.  Not an engine "
               "input: the seam reads sigma_coefficient.  The whole-tab reset "
               "writes it from config/decisions.yaml "
               "(donation_default.stochastic.scale_factor) and deletes the "
               "slider's widget key, so the slider comes back at the file's value "
               "whether it was drawn or hidden at the reset (owner-approved, "
               "2026-10-22; before, the reset deleted this key and left the widget "
               "key alone, so a drawn slider kept its value).  The slider's value "
               "survives culling through this key: what the slider shows is what "
               "the run uses."),
    _row("sigma_value_ui", 9.8995, P2_INIT, "page2-mirror",
         notes="display-only product of the sigma constant and the coefficient; "
               "the seam recomputes it from the one constant rather than reading it."),
    _row("anchor_observed_weight", "<config/decisions.yaml donation_default.anchor_weights.observed>",
         MODELS_DEFAULTS, "page2-mirror",
         engine_input=True,
         notes="donation anchor split; mirrors tab_anchor_weight. Read from the file "
               "(0.75 there); no copy of the value in the app since 2026-10-22."),
    _row("donation_sigma_strategy", "overall", DON_INIT, "page2-mirror",
         engine_input=True, notes="'overall' | 'quintile'"),
    _row("donation_quintile_scale_factors", {"1": 1.0, "2": 1.0, "3": 1.0,
                                             "4": 1.0, "5": 1.0},
         DON_INIT, "page2-mirror", engine_input=True),
    _row("donation_adjustment_shift", "<config/decisions.yaml adjustment.shift_value (-4.0), else 0.0>",
         DON_YAML, "page2-mirror",
         engine_input=True,
         notes="the shift comes from this key, never from the file.  The seam reads "
               "it from the snapshot; in app/ only "
               "donation_default.reload_coefficients_for_income_mode reads it, to "
               "carry it across a coefficient reload."),
    _row("donation_coeff_{name}", "<config/decisions.yaml regression_coefficients>",
         DON_YAML, "page2-mirror",
         notes="the flat set of the active income mode (intercept, hh, linear, "
               "midsub, nosub, fullsub, q1..q5, incoming, law, ug, grad); "
               "state.saved_configs.get_current_coefficients stores it with a pinned "
               "donation configuration.  The older q45 is never written; "
               "state.saved_configs still names it as a fallback for q5.  The engine "
               "runs the _cat / _cont set below, so this one is UI state."),
    _row("donation_coeff_intercept", "<config/decisions.yaml intercept>", DON_YAML,
         "page2-mirror",
         notes="the presence sentinel: models.initialize_session_state reloads the "
               "whole coefficient set from the file only when this key is absent."),
    _row("donation_coeff_{name}_{mode_suffix}",
         "<config/decisions.yaml regression_coefficients.{categorical,continuous}>",
         "models.load_coefficient_set", "page2-mirror", engine_input=True,
         notes="suffix 'cat' / 'cont'; this is the set the seam hands the "
               "engine (build_plan.session_donation_coefficient_set) and the set the "
               "Donation tab's formula tables show."),
    _row("intercept_override_values", "<{} until an intercept is typed>",
         "decision_tabs.donation_default.render_intercept_override_section",
         "page2-mirror",
         notes="the typed categorical / continuous intercepts, kept in session "
               "instead of written back to the file."),
    _row("adjustment_override_values", "<{} until a shift is typed>",
         "decision_tabs.donation_default.render_adjustment_override_section",
         "page2-mirror"),
    _row("di_intercept_override_values", "<{} until an intercept is typed>",
         "decision_tabs._disclosure.render_intercept_override_section (Disclose Income)", "page2-mirror"),
    _row("dd_intercept_override_values", "<{} until an intercept is typed>",
         "decision_tabs._disclosure.render_intercept_override_section (Disclose Documents)", "page2-mirror"),
    _row("decision_params", "<DecisionParameters()>", MODELS_INIT, "page2-mirror",
         engine_input=True,
         notes="only .selected_decisions is used.  The run buttons hand the seam a "
               "RunRequest instead of swapping it; the seam reads it only when no "
               "request is given (tests building a plan from a bare session)."),
    _row("page2_manual_selections", "<the current decision multiselect>",
         "page2_decisions.render_page2", "page2-mirror"),
    _row("page2_select_all_state", False, "page2_decisions.render_page2",
         "page2-mirror"),
)

_DI_MIRRORS = _rows((
    ("di_intercept", "<config/decisions.yaml intercept>", "read strictly; a key the file lacks raises"),
    ("di_wopb", "<anchor_weights.observed_prosocial>", "read strictly"),
    ("di_wpb", "<anchor_weights.prosocial_weight>", "read strictly"),
    ("di_income_mode", "<config income_mode, else 'categorical'>"),
    ("di_sigma_enabled", True, "the tab default is ON, so an absent key must read True too"),
    ("di_sigma_in_copula", False),
    ("di_sigma_strategy", "<stochastic.sigma_strategy, else 'overall'>"),
    ("di_scale_factor", "<stochastic.scale_factor>",
     "read strictly (the file says 0.1; the former 'else 1.0' fallback contradicted it); "
     "the saved-config store falls back to the same file value"),
    ("di_quintile_scale_factors", "<stochastic.quintile_scale_factors, else all scale_factor>"),
), DI_INIT, "page2-mirror", engine_input=True)

_DD_MIRRORS = _rows((
    ("dd_intercept", "<config/decisions.yaml intercept>", "read strictly; a key the file lacks raises"),
    ("dd_income_mode", "<config income_mode, else 'Categorical only'>"),
    ("dd_sigma_enabled", True, "the tab default is ON, so an absent key must read True too"),
    ("dd_sigma_in_copula", False),
    ("dd_sigma_strategy", "<stochastic.sigma_strategy, else 'overall'>"),
    ("dd_scale_factor", "<stochastic.scale_factor, else 1.0>"),
    ("dd_quintile_scale_factors", "<stochastic.quintile_scale_factors, else all scale_factor>"),
), DD_INIT, "page2-mirror", engine_input=True)

_RTD_MIRRORS = _rows((
    ("rtd_income_mode", "Continuous only"),
    ("rtd_sigma_strategy", "overall", "decision-wide: one strategy for all five mechanisms"),
    ("rtd_scale_factor", "<config stochastic.mechanisms.*.scale_factor (0.5), else 1.0>",
     "decision-wide; 0.5 per the platform documentation (previously 1.0)"),
    ("rtd_quintile_scale_factors", "<config stochastic.mechanisms.*.quintile_scale_factors (all 0.5)>"),
    ("rtd_intercept_{mech}", "<config intercepts[mech], else 0.0>",
     "mech in ttp / loyalty / wtp / risk_taking / flexibility"),
    ("rtd_flex_observed_weight", "<config flexibility_anchor.observed_weight>",
     "Flexibility Anchor Mix W_OFlex (September 2026); W_CFlex = 1 - W_OFlex"),
    ("rtd_aggregation_enabled", True,
     "Section-6 rank aggregation; always on - no widget renders it (September 2026)"),
    ("rtd_sigma_enabled", True, "the tab default is ON, so an absent key must read True too; "
     "also read by the saved-config store (get_current_rejected_transaction_params)"),
    ("rtd_sigma_in_copula", False, "also read by the saved-config store"),
), RTD_INIT, "page2-mirror", engine_input=True)

_RTD_DISPLAY = (
    _row("rtd_run_element", None, RTD_INIT, "page2-mirror",
         notes="which Decision-4 Run button started the run: an element, 'aggregation' "
               "(Run Integrated Default List Only) or None (whole decision); display and "
               "export only - the model always computes every element."),
)


# --------------------------------------------------------------------------- #
# Page 2 - widget keys
# --------------------------------------------------------------------------- #

_DONATION_WIDGETS = _rows((
    ("tab_sigma_in_copula", "<sigma_in_copula, else False>"),
    ("tab_sigma_in_research", "<sigma_in_research, else True>"),
    ("tab_sigma_coefficient_stochastic", "<donation_sigma_coefficient_setting, else 1.0>",
     "seeded by _controls.render_sigma_controls right before the slider (not by the "
     "initialiser: the key must not be written in a run that does not draw the slider); "
     "the seam multiplies the one sigma constant by this slider"),
    ("tab_anchor_weight", "<anchor_observed_weight, else the file's donation_default.anchor_weights.observed>"),
    ("donation_tab_sigma_strategy_stochastic", "<donation_sigma_strategy>"),
    ("donation_tab_sigma_q{level}_stochastic", "<donation_quintile_scale_factors[level]>"),
    ("override_categorical_intercept", "<the current categorical intercept>"),
    ("override_continuous_intercept", "<the current continuous intercept>"),
    ("override_adjustment_shift", "<donation_adjustment_shift>"),
    ("page2_tab_income_spec_mode", "<income_spec_mode, else 'categorical only'>",
     "the Donation tab's income-mode radio; seeded by render_donation_default_tab right before it "
     "(Page 2 pre-seeded it from income_spec_mode until 2026-10-22)"),
    ("reset_intercept_btn", WIDGET),
    ("reset_adjustment_btn", WIDGET),
    ("reset_config_btn", WIDGET),
), DON_INIT + " / render_donation_default_tab", "page2-widget", is_widget=True)

_DI_WIDGETS = _rows((
    ("di_tab_income_mode", "<di_income_mode>"),
    ("di_tab_sigma_enabled", "<di_sigma_enabled>"),
    ("di_tab_sigma_in_copula", "<di_sigma_in_copula, else False>"),
    ("di_tab_sigma_coefficient_stochastic", "<di_scale_factor>"),
    ("di_tab_sigma_strategy_stochastic", "<di_sigma_strategy>"),
    ("di_tab_sigma_q{level}_stochastic", "<di_quintile_scale_factors[level]>"),
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
    ("dd_tab_sigma_coefficient_stochastic", "<dd_scale_factor>"),
    ("dd_tab_sigma_strategy_stochastic", "<dd_sigma_strategy>"),
    ("dd_tab_sigma_q{level}_stochastic", "<dd_quintile_scale_factors[level]>"),
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
    ("rtd_tab_flex_observed_weight", "<rtd_flex_observed_weight>", "Flexibility Anchor Mix slider"),
    ("rtd_reset_btn", WIDGET),
    ("rtd_reset_{mech}_btn", WIDGET, "per-element reset"),
    ("rtd_run_{mech}_btn", WIDGET, "per-element run"),
    ("rtd_run_aggregation_btn", WIDGET, "Run Integrated Default List Only"),
    ("rtd_tab_unselect_btn", WIDGET, "unselects a saved Decision-4 configuration"),
    ("rtd_subtabs", WIDGET, "st.container key of the sub-decision tabs (CSS hook for the bold labels)"),
), RTD_INIT + " / render_rejected_transaction_defaults_tab", "page2-widget", is_widget=True)

#: The whole-decision "Reset Config to Defaults" of the Disclose Income / Documents
#: tabs is applied at the TOP of the next script run, before any of the tab's widgets
#: is instantiated (September 2026) and, since 2026-10-23, before the Page-2 Overview
#: that reads the tab's keys; the button only raises this flag.
_RESET_PENDING_FLAGS = (
    _row("_di_reset_to_defaults_pending", True,
         "decision_tabs._disclosure.reset_to_defaults (Disclose Income)", "page2-mirror",
         notes="consumed (deleted) by apply_pending_reset, called by "
               "page2_decisions.apply_pending_tab_resets before the Overview while "
               "Disclose Income is selected (the tab's own call at its top then "
               "finds no flag)"),
    _row("_dd_reset_to_defaults_pending", True,
         "decision_tabs._disclosure.reset_to_defaults (Disclose Documents)", "page2-mirror",
         notes="consumed (deleted) by apply_pending_reset, called by "
               "page2_decisions.apply_pending_tab_resets before the Overview while "
               "Disclose Documents is selected (the tab's own call at its top then "
               "finds no flag)"),
)

_PAGE2_WIDGETS = _rows((
    ("page2_manual_multiselect", "<page2_manual_selections>"),
    ("page2_select_all_checkbox", "<page2_select_all_state>"),
    ("clear_donation_config", WIDGET),
    ("clear_disclose_income_config", WIDGET),
    ("clear_disclose_documents_config", WIDGET),
    ("clear_rejected_transaction_config", WIDGET),
    ("run_complete_simulation", WIDGET),
    ("run_complete_simulation_disabled", WIDGET, "the disabled twin of the button above"),
), "page2_decisions.render_page2", "page2-widget", is_widget=True)

_RUN_BUTTONS = _rows((
    ("run_{decision_name}_only_btn", WIDGET,
     "the per-decision Run button; the Decision-4 tab reads its state back"),
    ("run_complete_from_{decision_name}_btn", WIDGET),
    ("run_complete_from_{decision_name}_btn_disabled", WIDGET),
), "decision_execution.render_simulation_buttons", "page2-widget", is_widget=True)


# --------------------------------------------------------------------------- #
# saved configurations
# --------------------------------------------------------------------------- #

_SAVED_CONFIG = (
    _row("selected_decision_configs", {}, "state.saved_configs.get_selected_decision_configs",
         "saved-config", engine_input=True,
         notes="only explicit 'Use This Config' selections.  Each entry "
               "pins seed, agent count, population mode, that decision's parameters "
               "and the sha256 of the columns it produced, so a pinned decision "
               "reproduces the run it was pinned from.  A saved Disclose "
               "Documents config is applied exactly like a Disclose Income one."),
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
    _row("_reset_intercept_flag", True,
         "decision_tabs.donation_default.render_actions_and_management_section", "run-flag",
         notes="one-rerun flag so the number_input picks the reset value up."),
    _row("_reset_adjustment_flag", True,
         "decision_tabs.donation_default.render_actions_and_management_section", "run-flag"),
    _row("_reset_config_to_defaults_flag", True,
         "decision_tabs.donation_default.render_actions_and_management_section",
         "run-flag",
         notes="consumed by donation_default.apply_pending_donation_reset in the "
               "next run, before the Page-2 Overview and the tab's widgets "
               "(page2_decisions.apply_pending_tab_resets, while Donation Default is "
               "selected; the tab's own call at its top then finds no flag): writes "
               "the intercept, adjustment, income-mode, anchor and σ tick-box widget "
               "keys, the canonical σ mode, coefficient and quintile dict and the "
               "coefficient in effect (sigma_coefficient, sigma_value_ui), and "
               "deletes the σ mode / coefficient / quintile widget keys (seeded "
               "again from the canonical keys on their next draw)."),
)


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

_RESULTS = (
    _row("simulation_results", None, MODELS_INIT, "results",
         notes="{result_key: DataFrame} after a run; None after a Monte-Carlo run."),
    _row("mc_results", None, MODELS_INIT, "results",
         notes="{'summary', 'detailed', 'log'} from the in-process Monte-Carlo "
               "study (simulation.run_monte_carlo_study)."),
    _row("_run_metadata", "<RunMetadata as a dict>", "simulation.run_full_simulation",
         "results",
         notes="the results page reads the run's real population / income mode "
               "from here, so the chart titles cannot drift from what ran; its "
               "started_at is the base time of every export's timestamps."),
    _row("vendors", "<df.attrs['vendors'] of the first result>",
         "simulation.run_full_simulation", "results"),
)

_RESULT_WIDGETS = _rows((
    ("inline_select_{result_key}", WIDGET, "donation 'Use This Config' button"),
    ("di_inline_select_{result_key}", WIDGET),
    ("dd_inline_select_{result_key}", WIDGET),
    ("rtd_inline_select_{result_key}", WIDGET, "Decision-4 'Use This Config', under its results"),
    ("rtd_inline_unselect_{result_key}", WIDGET, "the selected cell's unselect button"),
    ("clear_conflict_{decision_name}_{conflicting}", WIDGET,
     "shown when a saved config's seed clashes with another one"),
    ("clear_di_selection", WIDGET),
    ("clear_dd_selection", WIDGET),
    ("clear_rtd_selection", WIDGET),
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
    ("rtd_export_section_download", WIDGET),
    ("vendor_choice_weights_chart", WIDGET),
    ("show_all_proximity_matrix", WIDGET),
), "the results renderers (they never write settings)", "results",
    is_widget=True)


# --------------------------------------------------------------------------- #
# default-decision parameters and their shadow store
# --------------------------------------------------------------------------- #

_DEFAULTS = (
    _row("_persistent_defaults", {}, "decision_tabs.default_config.save_to_persistent_storage",
         "defaults", engine_input=True,
         notes="the shadow store: it outlives a widget key that Page 2 never "
               "rendered, and the seam consults it before the session key."),
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
         notes="named in the radio_selection branch of "
               "build_plan.collect_decision_settings; decision 4 is a "
               "prioritized_selection decision, so that lookup is never reached."),
    _row("rejected_transaction_option_selection", "<never written>",
         "nothing - read-only fallback", "defaults", engine_input=True,
         notes="the same for decision 11."),
    _row("reset_all_defaults", WIDGET,
         "decision_tabs.default_config.render_default_decisions_config", "defaults",
         is_widget=True),
    _row("{decision_name}_reset", WIDGET,
         "decision_tabs.default_config.render_prioritized_default_config", "defaults",
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
    + _PAGE1_WIDGETS + _PAGE1_SEED_WIDGETS + _PAGE1_INLINE_WIDGETS + _PAGE1_MIRRORS
    + _PAGE2_MIRRORS + _DI_MIRRORS + _DD_MIRRORS + _RTD_MIRRORS + _RTD_DISPLAY
    + _DONATION_WIDGETS + _DI_WIDGETS + _DD_WIDGETS + _RTD_WIDGETS
    + _RESET_PENDING_FLAGS
    + _PAGE2_WIDGETS + _RUN_BUTTONS
    + _SAVED_CONFIG + _RUN_FLAGS
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
    "CONTROL_KEYS", "ControlKeys", "DISCLOSURE_KEYS", "DisclosureKeys", "KEY_PARAMETER_TABLES",
    "TAB_WIDGET_CANONICAL",
    "KeySpec", "REGISTRY", "lookup", "matching_families", "normalise",
    "pattern_regex", "specificity",
]
