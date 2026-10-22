# app/pages/decision_tabs/_disclosure.py
"""What the Disclose Income and Disclose Documents tabs do alike, once: the
intercept override (its three-column section and the auto-save of a typed
value), the whole-tab reset (scheduled by the button, applied at the top of the
next run), the intercept reset, and the "Actions & Management" buttons.

Each tab describes itself in a :class:`DisclosureTab`: its keys (a row of
``app.state.registry.DISCLOSURE_KEYS``, so ``f"{keys.dp}override_intercept"`` is
``di_override_intercept`` or ``dd_override_intercept`` and the registry test can
expand it), its configuration loader, its research intercept and the few strings
that name the decision. The two things the tabs reset differently - the Disclose
Income anchor-mix sliders, the Disclose Documents σ widget keys - are the tab's
own ``apply_extra_widget_keys`` hook.

Reset order (docs/session-state.md §7): the button only raises
``_{dp}reset_to_defaults_pending``; ``apply_pending_reset`` runs at the TOP of the
next script run, before any of the tab's widgets exists, deletes every ``{dp}``
key, and writes the widget keys the same run draws from the configuration
file. The per-quintile σ sliders are deliberately not written:
their keys drop with the bulk delete and the controls seed them again from the
canonical quintile dict when the sliders are next drawn.
"""
from dataclasses import dataclass
from typing import Callable

import streamlit as st

from app.state.registry import DisclosureKeys
from app.state.widgets import stateful
from app.pages.decision_tabs._tab_state import seed_if_absent


@dataclass(frozen=True)
class DisclosureTab:
    keys: DisclosureKeys
    #: "disclose income" / "disclose documents", for the reset button's help
    decision_label: str
    #: the tab's block of config/decisions.yaml, read fresh (the file is read-only)
    load_config: Callable[[], dict]
    #: the research intercept, used when the configuration has none
    default_intercept: float
    #: how the research default is printed on the intercept-reset button and toast
    #: ("0.75"; "−0.75" with a typographic minus on Disclose Documents)
    research_default_label: str
    intercept_min: float
    intercept_max: float
    intercept_help: str
    #: writes the tab's OWN widget / canonical keys on a whole-tab reset
    apply_extra_widget_keys: Callable[[dict], None]


def current_yaml_intercept(spec):
    """Get current intercept value from YAML config."""
    return spec.load_config().get('intercept', spec.default_intercept)


# --------------------------------------------------------------------------- #
# intercept override
# --------------------------------------------------------------------------- #

def render_intercept_override_section(spec):
    """Render the intercept override section with ability to modify the intercept value."""
    keys = spec.keys
    override_values_key = f'{keys.dp}intercept_override_values'
    override_key = f'{keys.dp}override_intercept'

    # Initialize override values if not present
    if override_values_key not in st.session_state:
        st.session_state[override_values_key] = {}

    # Show current configuration values for reference
    try:
        # Research default — Impact Preview always compares against this. The
        # configuration file is read-only, so it is the file's intercept.
        research_default = float(spec.load_config().get('intercept', spec.default_intercept))

        # The configuration file's intercept seeds the override input until a
        # value is typed.
        current_config_value = current_yaml_intercept(spec)

        # Use 3 columns: Current Value, Override Value, Impact Preview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**Research Default: {research_default:.4f}**")
            st.markdown("Intercept (β₀)")
            st.markdown("Fixed reference value from original research")

        with col2:
            st.markdown("**Override Value**")

            # Seed the widget key ONLY when it is absent (after navigation); otherwise the
            # live widget value wins. No rerun-dependent `value=` is passed below, so the
            # widget id stays stable - re-seeding on every rerun plus a changing default
            # made the +/- buttons jump back one step under fast clicks.
            seed_if_absent(override_key, float(
                st.session_state[override_values_key].get('intercept', current_config_value)))

            new_intercept = stateful(st.number_input,
                "Baseline disclosure tendency",
                min_value=spec.intercept_min,
                max_value=spec.intercept_max,
                step=0.01,
                format="%.4f",
                help=spec.intercept_help,
                key=override_key,
                on_change=lambda: auto_save_intercept(spec, st.session_state[override_key])
            )
            st.session_state[override_values_key]['intercept'] = new_intercept

        with col3:
            st.markdown("**Impact Preview**")
            change = new_intercept - research_default
            if abs(change) > 0.00001:
                impact = "Higher baseline" if change > 0 else "Lower baseline"
                st.metric("Change", f"{change:+.4f}", delta=impact)
            else:
                st.metric("Change", "No change")

    except Exception as e:
        st.error(f"Error loading configuration values: {e}")


def auto_save_intercept(spec, new_value):
    """Store the typed intercept override in session state (the configuration
    file is read-only). The seam reads the canonical intercept from the session
    snapshot, so the next run uses it.
    """
    keys = spec.keys
    override_values_key = f'{keys.dp}intercept_override_values'
    if override_values_key not in st.session_state:
        st.session_state[override_values_key] = {}

    st.session_state[override_values_key]['intercept'] = new_value
    st.session_state[f'{keys.dp}intercept'] = new_value


# --------------------------------------------------------------------------- #
# whole-tab reset
# --------------------------------------------------------------------------- #

def apply_config_to_widget_keys(spec, config):
    """Explicitly set the widget keys this tab resets, from a config dict.

    Only keys whose widget the SAME run draws are written (the reset is applied
    at the top of the run, before the tab's widgets): a widget key written in a
    run that does not draw its widget stays in Streamlit's old session state under
    the user key and resurfaces, stale, once the widget has been drawn and culled
    again - in front of the value the user set. So the per-quintile sigma keys
    (``{dp}tab_sigma_q{1..5}_stochastic``), whose sliders are not drawn while the
    strategy is back at 'overall', are NOT written: the bulk delete of every
    ``{dp}`` key dropped them, and the σ controls seed them from the canonical
    quintile dict (reset below) when the sliders are drawn. The tab's own keys
    (``apply_extra_widget_keys``) come first.
    """
    keys = spec.keys
    stochastic = config.get('stochastic', {})

    spec.apply_extra_widget_keys(config)
    st.session_state[f'{keys.dp}override_intercept'] = config.get('intercept', spec.default_intercept)
    # The Research Specification tick box defaults to on, so a reset leaves it on.
    st.session_state[f'{keys.wp}sigma_enabled'] = True
    st.session_state[f'{keys.wp}income_mode'] = config.get('income_mode', 'Categorical only')
    st.session_state[f'{keys.wp}sigma_in_copula'] = False
    st.session_state[f'{keys.dp}sigma_in_copula'] = False

    default_scale = stochastic.get('scale_factor', 1.0)
    quintile_scales = stochastic.get('quintile_scale_factors', {})

    # Also set the READ-keys the simulation consumes; the configuration file is
    # read-only, so they can no longer be re-read from a rewritten file.
    st.session_state[f'{keys.dp}intercept'] = config.get('intercept', spec.default_intercept)
    st.session_state[f'{keys.dp}income_mode'] = config.get('income_mode', 'Categorical only')
    st.session_state[f'{keys.dp}sigma_enabled'] = True
    st.session_state[f'{keys.dp}sigma_strategy'] = stochastic.get('sigma_strategy', 'overall')
    st.session_state[f'{keys.dp}scale_factor'] = default_scale
    st.session_state[f'{keys.dp}quintile_scale_factors'] = {
        level: quintile_scales.get(level, default_scale) for level in ['1', '2', '3', '4', '5']
    }


def reset_to_defaults(spec):
    """Schedule the session-state reset.

    The actual session-state reset runs at the TOP of the next script run (see
    apply_pending_reset). Clearing / re-seeding the tab's widget keys here would
    happen AFTER those widgets were already instantiated in this run - a
    Streamlit anti-pattern that leaves widgets whose element id no longer has a
    session-state entry (values jumped back after a reset).
    """
    st.session_state[f'_{spec.keys.dp}reset_to_defaults_pending'] = True
    return True


def apply_pending_reset(spec):
    """Apply a scheduled reset before any of the tab's widgets renders: clear
    every ``{dp}`` key, then seed the defaults."""
    keys = spec.keys
    pending_key = f'_{keys.dp}reset_to_defaults_pending'
    if not st.session_state.get(pending_key, False):
        return
    del st.session_state[pending_key]
    reset_config = spec.load_config()

    # 1. Clear all the tab's keys (widget keys and canonical keys alike)
    for key in [k for k in st.session_state.keys() if k.startswith(keys.dp)]:
        del st.session_state[key]

    # 2. SET every widget key to the default value.  This prevents
    #    Streamlit's internal widget cache from retaining stale slider
    #    values on the next rerun.
    apply_config_to_widget_keys(spec, reset_config)


# --------------------------------------------------------------------------- #
# intercept reset and the buttons
# --------------------------------------------------------------------------- #

def reset_intercept_to_default(spec):
    """Restore the configuration file's intercept into session state."""
    keys = spec.keys
    try:
        default_intercept = current_yaml_intercept(spec)
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        return False

    # Update session state for intercept only
    st.session_state[f'{keys.dp}intercept'] = default_intercept
    override_values_key = f'{keys.dp}intercept_override_values'
    if override_values_key in st.session_state:
        st.session_state[override_values_key]['intercept'] = default_intercept
    # The override input is seeded only while its key is absent (stable widget id
    # under fast clicks), so drop the key: the rerun that follows re-seeds it from
    # the value just stored instead of keeping the old number on screen.
    override_key = f'{keys.dp}override_intercept'
    if override_key in st.session_state:
        del st.session_state[override_key]

    return True


def render_actions_and_management_section(spec):
    """Render the combined actions and management section."""
    keys = spec.keys

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔄 Reset All**")
        if st.button("Reset Config to Defaults", type="secondary", use_container_width=True,
                     help=f"Reset all {spec.decision_label} values to research defaults",
                     key=f'{keys.dp}reset_btn'):
            success = reset_to_defaults(spec)
            if success:
                st.toast("Configuration reset to defaults", icon="🔄")
                st.rerun()
            else:
                st.toast("Failed to reset configuration", icon="⚠️")

    with col2:
        st.markdown("**🔄 Reset Intercept**")
        if st.button("Reset Intercept to Default Value", type="secondary", use_container_width=True,
                     help=f"Reset intercept value to research default ({spec.research_default_label})",
                     key=f'{keys.dp}reload_btn'):
            # Only reset the intercept value - NOT all configuration values
            success = reset_intercept_to_default(spec)
            if success:
                st.toast(f"✅ Intercept reset to research default ({spec.research_default_label})", icon="🔄")
                st.rerun()
            else:
                st.toast("❌ Failed to reset intercept", icon="⚠️")
