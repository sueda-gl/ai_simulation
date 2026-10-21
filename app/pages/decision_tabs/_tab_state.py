# app/pages/decision_tabs/_tab_state.py
"""The decision tabs' widget-key persistence, in one place.

A tab widget's value lives in three keys (docs/session-state.md §2): the widget
key Streamlit owns, the tab's persistence dict (written only from an ``on_change``
callback, so it outlives a widget Streamlit culled on another page) and the
canonical key the seam reads. These helpers are the traffic between the first two.

Two restore orders exist, and each tab keeps the one it had:

* ``widget_first=True`` (Decision 4): the LIVE widget key wins when it exists; the
  persistence dict, then the default, only seed the key when it is absent.
  Re-pushing the stored value on every rerun forced the backend to send the value
  to the browser each rerun, which - together with a rerun-dependent ``value=``
  default - made the +/- buttons of the intercept inputs jump back one step under
  fast clicks.
* ``widget_first=False`` (Donation, Disclose Income, Disclose Documents): the
  persistence dict wins and is written over the widget key on every rerun; else the
  live widget key; else the default. These tabs' widget keys are also seeded by
  ``page2_decisions.initialize_page2_widget_keys`` in runs where no widget binds
  them, and such a key resurfaces with that stale value once Streamlit culls the
  widget - the widget-first order would then show the stale value after leaving
  and re-entering the tab (verified 2026-10-21), so the dict must keep winning.

Every key here is named at run time through a parameter; the session-key registry
lists ``widget_key`` and ``storage_dict_name`` in ``DYNAMIC_KEY_VARIABLES`` for
that reason. The callers still bind each widget key literally (``key=``).
"""
import streamlit as st


def restore(widget_key, storage_dict_name, storage_key, default, *, widget_first):
    """Seed ``widget_key`` before its widget renders and return its current value.

    ``widget_first=True``: the live widget key; else the entry ``storage_key`` of
    the persistence dict ``st.session_state[storage_dict_name]``; else ``default``.
    ``widget_first=False``: the dict entry (written over the widget key); else the
    live widget key; else ``default``."""
    storage = st.session_state.get(storage_dict_name)
    if widget_first and widget_key in st.session_state:
        return st.session_state[widget_key]
    if storage and storage_key in storage:
        value = storage[storage_key]
        st.session_state[widget_key] = value
        return value
    if widget_key in st.session_state:
        return st.session_state[widget_key]
    st.session_state[widget_key] = default
    return default


def save(widget_key, storage_dict_name, storage_key):
    """Copy the widget's value into the persistence dict (an ``on_change`` callback)."""
    if storage_dict_name not in st.session_state:
        st.session_state[storage_dict_name] = {}
    if widget_key in st.session_state:
        st.session_state[storage_dict_name][storage_key] = st.session_state[widget_key]


def seed_if_absent(widget_key, value):
    """Write ``value`` into ``widget_key`` only when the key does not exist yet."""
    if widget_key not in st.session_state:
        st.session_state[widget_key] = value


def seed_if_changed(widget_key, value):
    """Overwrite a seeded widget key ONLY when the normalised value differs (an
    unconditional assignment marks the key as changed and re-pushes it each rerun)."""
    if st.session_state.get(widget_key) != value:
        st.session_state[widget_key] = value
