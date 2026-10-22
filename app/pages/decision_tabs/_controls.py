# app/pages/decision_tabs/_controls.py
"""The controls the four modelled decision tabs share, drawn once.

The Donation Default, Disclose Income, Disclose Documents and Decision 4 tabs
each show the same stochastic controls: the two tick boxes (σ in Copula runs, σ
in Research Specification runs), the σ-mode radio (uniform / per quintile), the
σ coefficient slider with its "Effective σ" line, the five per-quintile sliders
with their metrics, and the income-mode radio. Until 2026-10-22 each tab carried
its own copy, with small drifts between the copies. This module draws them once;
every drift is a parameter of :class:`TabControls`, so each tab still renders
what it rendered, except the σ figures, which the owner has since set to two
decimals on every tab (the remaining drifts are listed on the fields; harmonising
them is an owner decision, not a refactoring).

Keys
----
A control's widget key and canonical key are built from the tab's row of
``app.state.registry.CONTROL_KEYS`` (``keys``): for example
``f"{keys.wp}sigma_coefficient{keys.suffix}"`` is ``tab_sigma_coefficient_stochastic``
on the Donation tab and ``rtd_tab_sigma_coefficient`` on Decision 4. The registry
holds one row per concrete key, and ``tests/test_session_key_registry.py`` expands
every such f-string over every row of the table, so a key cannot leave the
registry's sight by being built here.

Seeding follows ``_tab_state`` (docs/session-state.md §2): the live widget key
wins, the canonical key (``registry.TAB_WIDGET_CANONICAL``) seeds an absent one,
and the canonical key is written after the widget on every render. No widget
here has an ``on_change`` callback except the two radios, which mirror their
canonical key in the callback so the Overview (drawn before the tabs) sees the
new mode in the same run.

Decision 4 shares the tick boxes, the income radio, the σ-mode radio and the
coefficient slider; its per-element "Effective σ" lines, its per-quintile sliders
and its per-element σ table are decision-wide and stay in ``rejected_transaction``.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

from app.state.registry import ControlKeys
from app.state.widgets import stateful
from app.pages.decision_tabs._tab_state import restore, seed_if_absent, seed_if_changed

LEVELS = ('1', '2', '3', '4', '5')

#: Per-quintile base σ shown by the Donation Default and Disclose Income tabs
#: (display only: the model takes its σ from config/decisions.yaml).
EMPIRICAL_BASE_SIGMAS = {
    '1': 5.705052,   # Level 1 (€12)
    '2': 3.069326,   # Level 2 (€32)
    '3': 3.532226,   # Level 3 (€72)
    '4': 12.219622,  # Level 4 (€128)
    '5': 16.854622,  # Level 5 (€200)
}

LEVEL_LABELS = {
    '1': 'Level 1 (€12)',
    '2': 'Level 2 (€32)',
    '3': 'Level 3 (€72)',
    '4': 'Level 4 (€128)',
    '5': 'Level 5 (€200)',
}

MODE_OPTIONS = ["Categorical only", "Continuous only", "Compare both"]


@dataclass(frozen=True)
class TabControls:
    """One tab's controls: its keys, its σ constants, and every textual drift."""

    keys: ControlKeys

    # -- income-mode radio ---------------------------------------------------
    income_label: str
    income_help: str
    income_options: list
    income_default: str
    #: how the widget key is seeded from the canonical key before the first draw:
    #: 'match' (Disclose Income: a case-insensitive match of the canonical value,
    #: else the default), 'exact' (Disclose Documents, Decision 4: the canonical
    #: value when it is an option, else the default) or None (Donation: not seeded;
    #: restore() alone, and a value outside the options is left as it is)
    income_seed: Optional[str]
    #: called after the canonical key is written
    after_income_change: Optional[Callable[[], None]] = None

    # -- tick boxes ----------------------------------------------------------
    #: 'anchor' ("Add Normal(anchor, σ) draw ...") or 'score' (Disclose Documents)
    draw_noun: str = 'anchor'
    #: Decision 4 only: after restore(), seed_if_changed() the normalised value
    #: (bool for the boxes, 'overall' / 'quintile' for the radio, the clamped
    #: coefficient) - a no-op while the stored value is already normalised
    coerce: bool = False

    # -- σ-mode radio --------------------------------------------------------
    #: the "**σ mode**" line above the radio (Disclose Income / Documents, Decision 4;
    #: not on the Donation tab)
    strategy_header: bool = True
    #: "Apply σ uniformly or per budget level?" ("... per income level?" on the Donation tab)
    strategy_label: str = "Apply σ uniformly or per budget level?"

    # -- coefficient slider and quintile sliders -----------------------------
    # Every σ on screen (base, effective, per-quintile base in the slider labels
    # and the metrics) is printed with two decimals (owner decision, 2026-10-22).
    base_sigma: Optional[float] = None
    #: the parenthesised overall-σ summary of the quintile-mode warning / note,
    #: a str.format template over base, coeff (the overall coefficient) and eff
    overall_summary: str = "{coeff:.2f} × {base:.2f} = {eff:.2f}"
    #: the slider's help; None = "Coefficient to multiply the base σ. Final σ = <base> × coefficient"
    coefficient_help: Optional[str] = None
    #: the coefficient slider's default when neither its key nor its canonical
    #: key holds one: a number (Donation 1.0), or None = the canonical coefficient
    #: key (Disclose Income / Documents, which raises when the key is absent)
    coefficient_fallback: Optional[float] = None
    #: a quintile's coefficient when the quintile dict lacks it: a number
    #: (Donation 1.0) or None = the canonical coefficient (else 1.0)
    quintile_level_fallback: Optional[float] = None
    base_sigmas: Optional[dict] = None
    level_labels: dict = None


# --------------------------------------------------------------------------- #
# income mode
# --------------------------------------------------------------------------- #

def render_income_mode(spec):
    """The income-mode radio; writes the canonical key and returns the mode."""
    keys = spec.keys
    mode_options = spec.income_options

    if spec.income_seed is not None:
        # Seed the widget key from the canonical key before its first draw
        current_mode = st.session_state.get(keys.income, spec.income_default)
        if spec.income_seed == 'match':
            seed = next((m for m in mode_options if m.lower() == str(current_mode).lower()),
                        spec.income_default)
        else:
            seed = current_mode if current_mode in mode_options else spec.income_default
        seed_if_absent(keys.income_widget, seed)

    def on_income_mode_change():
        st.session_state[keys.income] = st.session_state[keys.income_widget]
        if spec.after_income_change is not None:
            spec.after_income_change()

    income_val = restore(keys.income_widget, spec.income_default)

    # Ensure value is valid (handling case sensitivity)
    if spec.income_seed is not None and income_val not in mode_options:
        income_val = next((opt for opt in mode_options if str(income_val).lower() == opt.lower()),
                          spec.income_default)
        st.session_state[keys.income_widget] = income_val

    income_mode = stateful(st.radio,
        spec.income_label,
        mode_options,
        help=spec.income_help,
        key=keys.income_widget,
        on_change=on_income_mode_change
    )
    st.session_state[keys.income] = income_mode
    return income_mode


# --------------------------------------------------------------------------- #
# the two stochastic tick boxes
# --------------------------------------------------------------------------- #

def render_stochastic_tick_boxes(spec):
    """"Copula Mode" and "Research Specification Mode" tick boxes; writes both
    canonical keys and returns (sigma_in_copula, sigma_enabled)."""
    keys = spec.keys
    noun = spec.draw_noun

    # --- Copula sigma checkbox (always visible) ---
    st.markdown("**Copula Mode:**")
    copula_widget_key = f'{keys.wp}sigma_in_copula'
    copula_canonical_key = f'{keys.sp}sigma_in_copula'
    copula_val = restore(copula_widget_key, False)
    if spec.coerce:
        seed_if_changed(copula_widget_key, bool(copula_val))
    sigma_in_copula = stateful(st.checkbox,
        f"Add Normal({noun}, σ) draw to Copula runs",
        help="When enabled, Copula mode will also use the stochastic component",
        key=copula_widget_key,
    )
    st.session_state[copula_canonical_key] = sigma_in_copula

    # --- Research Specification sigma checkbox (always visible) ---
    st.markdown("**Research Specification Mode:**")
    research_widget_key = f'{keys.wp}{keys.research}'
    research_canonical_key = f'{keys.sp}{keys.research}'
    research_val = restore(research_widget_key, True)
    if spec.coerce:
        seed_if_changed(research_widget_key, bool(research_val))
    sigma_enabled = stateful(st.checkbox,
        f"Use Normal({noun}, σ) draw in Research Specification mode",
        help=f"When enabled, adds stochastic variation via Normal({noun}, σ) draws.",
        key=research_widget_key,
    )
    st.session_state[research_canonical_key] = sigma_enabled
    return sigma_in_copula, sigma_enabled


# --------------------------------------------------------------------------- #
# σ mode, coefficient, quintiles
# --------------------------------------------------------------------------- #

def render_strategy_radio(spec):
    """The σ-mode radio (uniform / per quintile) and the rule below it; writes
    the canonical strategy key and returns 'overall' or 'quintile'."""
    keys = spec.keys
    if spec.strategy_header:
        st.markdown("**σ mode**")

    strategy_widget_key = f'{keys.dp}tab_sigma_strategy{keys.suffix}'
    strategy_canonical_key = f'{keys.dp}sigma_strategy'
    current_strategy = st.session_state.get(strategy_canonical_key, 'overall')

    strategy_val = restore(strategy_widget_key, current_strategy)
    if spec.coerce:
        strategy_val = 'quintile' if 'quintile' in str(strategy_val).lower() else 'overall'
        seed_if_changed(strategy_widget_key, strategy_val)

    def on_strategy_change():
        """Handle sigma strategy changes."""
        st.session_state[strategy_canonical_key] = st.session_state[strategy_widget_key]

    sigma_strategy = stateful(st.radio,
        spec.strategy_label,
        options=['overall', 'quintile'],
        format_func=lambda x: 'Uniformly (single σ for all)' if x == 'overall' else 'Quintiles (σ per budget level)',
        key=strategy_widget_key,
        on_change=on_strategy_change,
        horizontal=True
    )
    st.session_state[strategy_canonical_key] = sigma_strategy

    st.markdown("---")
    return sigma_strategy


def render_coefficient_slider(spec, scale_fallback):
    """The overall σ coefficient slider, seeded with ``scale_fallback`` when
    neither its key nor its canonical key holds one; writes the canonical
    coefficient key and returns the coefficient."""
    keys = spec.keys
    coeff_widget_key = f'{keys.wp}sigma_coefficient{keys.suffix}'

    coeff_val = restore(coeff_widget_key, scale_fallback)
    if spec.coerce:
        coeff_val = max(0.0, min(float(coeff_val), 2.0))
        seed_if_changed(coeff_widget_key, coeff_val)

    if spec.coefficient_help is not None:
        help_text = spec.coefficient_help
    else:
        help_text = f"Coefficient to multiply the base σ. Final σ = {spec.base_sigma:.2f} × coefficient"

    sigma_coefficient = stateful(st.slider,
        "σ Coefficient (multiplier)",
        min_value=0.0,
        max_value=2.0,
        step=0.01,
        help=help_text,
        key=coeff_widget_key,
    )
    st.session_state[keys.coefficient] = sigma_coefficient
    return sigma_coefficient


def render_sigma_controls(spec):
    """The σ-mode radio, then either the overall coefficient slider with its
    "Base σ" and "Effective σ" lines, or the five per-quintile sliders with their
    "Effective σ" metrics (after the continuous-mode warning / note).

    Returns (sigma_strategy, sigma_coefficient): the coefficient is None in
    quintile mode."""
    keys = spec.keys
    sigma_strategy = render_strategy_radio(spec)

    if sigma_strategy == 'overall':
        # OVERALL MODE: Single slider
        st.markdown(f"Base σ = {spec.base_sigma:.2f} (empirical from 280 participants)")

        # The slider starts from the canonical coefficient (seeded by the tab's
        # initialiser from the configuration) or the tab's own fallback. 0.0 is a
        # legal, persisted value: tick box on + coefficient 0 -> sigma 0, no noise.
        if spec.coefficient_fallback is not None:
            scale_fallback = spec.coefficient_fallback
        else:
            scale_fallback = st.session_state[keys.coefficient]
        sigma_coefficient = render_coefficient_slider(spec, scale_fallback)

        effective_sigma = spec.base_sigma * sigma_coefficient
        st.markdown(f"Effective σ = {spec.base_sigma:.2f} × {sigma_coefficient:.2f} = {effective_sigma:.2f}")
        return sigma_strategy, sigma_coefficient

    # QUINTILE MODE: 5 sliders (one per income level)
    # Warn user that quintile sigma only applies to categorical mode
    current_income_mode = st.session_state.get(keys.income, spec.income_default)
    overall_coeff = st.session_state.get(keys.coefficient, 1.0)
    summary = spec.overall_summary.format(base=spec.base_sigma, coeff=overall_coeff,
                                          eff=spec.base_sigma * overall_coeff)
    if 'continuous' in str(current_income_mode).lower() and 'compare' not in str(current_income_mode).lower():
        st.warning(
            "**Continuous mode uses overall σ.** "
            "Per-quintile σ values are based on categorical budget levels and are "
            "not applicable to the continuous income specification. "
            f"The simulation will use the overall σ ({summary}) for continuous runs."
        )
    elif 'compare' in str(current_income_mode).lower():
        st.info(
            "**Note:** Per-quintile σ values will only apply to the **categorical** run. "
            f"The continuous run will use the overall σ ({summary})."
        )
    st.markdown("**Per-Quintile σ Coefficients**")
    st.markdown("Each level has its own base σ from empirical data:")

    quintile_coefficients = {}
    quintiles_key = f'{keys.dp}quintile_scale_factors'
    if spec.quintile_level_fallback is not None:
        default_scale = spec.quintile_level_fallback
    else:
        default_scale = st.session_state.get(keys.coefficient, 1.0)
    current_quintile_scales = st.session_state.get(quintiles_key, {
        '1': default_scale, '2': default_scale, '3': default_scale,
        '4': default_scale, '5': default_scale
    })

    for level in LEVELS:
        level_scale = current_quintile_scales.get(level, default_scale)
        level_scale = max(0.0, min(float(level_scale), 2.0))

        widget_key = f'{keys.dp}tab_sigma_q{level}{keys.suffix}'

        restore(widget_key, level_scale)

        base_sigma = spec.base_sigmas[level]

        col_slider, col_result = st.columns([3, 1])
        with col_slider:
            q_coeff = stateful(st.slider,
                f"{spec.level_labels[level]} (base σ={base_sigma:.2f})",
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                key=widget_key,
            )
        with col_result:
            effective = base_sigma * q_coeff
            st.metric("Effective σ", f"{effective:.2f}")

        quintile_coefficients[level] = q_coeff

    # Save quintile scale factors to session state
    st.session_state[quintiles_key] = quintile_coefficients
    return sigma_strategy, None
