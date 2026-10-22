"""Seeding the decision tabs' widget keys, in one place.

A tab widget's value has two homes (docs/session-state.md §2): the widget key
Streamlit owns, and the canonical key the seam reads, written right after the
widget on every render. Streamlit drops the widget key once the widget is not
drawn (another page, the decision unselected, the σ controls hidden); the
canonical key is never dropped and holds the last rendered value, so an absent
widget key is seeded from it. ``app.state.registry.TAB_WIDGET_CANONICAL`` says
which canonical key (or which entry of a dict-valued one) each widget key reads.
Until 2026-10-22 every tab had a third home, a ``*_tab_persistence`` dict written
only from ``on_change``; it only ever held what the canonical key held, so it went.

One order, for every tab: the LIVE widget key wins when it exists; the canonical
value, then the default, only seed the key when it is absent (after navigation, a
mode switch or a reset). Writing the stored value over the key on every rerun -
the order the Donation, Disclose Income and Disclose Documents tabs kept until
2026-10-22 - made Streamlit re-send the value to the browser with every rerun's
response, which overwrote the +/- clicks of a number input that arrived while a
rerun was in flight (2-4 of six fast clicks landed on the Donation intercept
inputs; the Decision 4 inputs, restored widget-first, kept all six).

The order is safe only while no widget key is written in a run that does not draw
its widget. Streamlit keeps such a write in its old session state under the user
key, never refreshes it once the widget exists (later writes go under the widget
id) and lets it resurface, stale, once the widget is culled - in front of the
canonical value. Page 2 therefore no longer pre-seeds the tabs' keys before a tab
is drawn, and a reset writes only keys whose widget the same run draws (the
Disclose Documents per-quintile sliders drop their keys; the Donation intercept
inputs, whose reset writes both keys, are seeded again from the typed override
whenever they return - ``donation_default._seed_intercept_input``).

Every key here is named at run time through a parameter; the session-key registry
lists ``widget_key`` and ``canonical_key`` in ``DYNAMIC_KEY_VARIABLES`` for that
reason. The callers still bind each widget key literally (``key=``).
"""
import streamlit as st

from app.state.registry import TAB_WIDGET_CANONICAL


def restore(widget_key, default):
    """Seed ``widget_key`` before its widget renders and return its current value:
    the live widget key when it exists; else the canonical value
    ``TAB_WIDGET_CANONICAL`` maps it to, when that key (and, for a dict-valued
    key, its entry) exists; else ``default``. The key is written only when it is
    absent. A widget key the table does not list raises at its first draw."""
    if widget_key in st.session_state:
        return st.session_state[widget_key]
    canonical_key, entry = TAB_WIDGET_CANONICAL[widget_key]
    value = default
    if canonical_key in st.session_state:
        if entry is None:
            value = st.session_state[canonical_key]
        else:
            holder = st.session_state[canonical_key]
            if entry in holder:
                value = holder[entry]
    st.session_state[widget_key] = value
    return value


def seed_if_absent(widget_key, value):
    """Write ``value`` into ``widget_key`` only when the key does not exist yet."""
    if widget_key not in st.session_state:
        st.session_state[widget_key] = value


def seed_if_changed(widget_key, value):
    """Overwrite a seeded widget key ONLY when the normalised value differs (an
    unconditional assignment marks the key as changed and re-pushes it each rerun)."""
    if st.session_state.get(widget_key) != value:
        st.session_state[widget_key] = value
