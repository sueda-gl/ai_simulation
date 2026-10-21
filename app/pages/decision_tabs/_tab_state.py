"""The decision tabs' widget-key persistence, in one place.

A tab widget's value lives in three keys (docs/session-state.md §2): the widget
key Streamlit owns, the tab's persistence dict (written only from an ``on_change``
callback, so it outlives a widget Streamlit culled on another page) and the
canonical key the seam reads. These helpers are the traffic between the first two.

One restore order, for every tab: the LIVE widget key wins when it exists; the
persistence dict, then the default, only seed the key when it is absent (after
navigation, a mode switch or a reset). Writing the stored value over the key on
every rerun - the order the Donation, Disclose Income and Disclose Documents tabs
kept until 2026-10-22 - made Streamlit re-send the value to the browser with every
rerun's response, which overwrote the +/- clicks of a number input that arrived
while a rerun was in flight (2-4 of six fast clicks landed on the Donation
intercept inputs; the Decision 4 inputs, restored widget-first, kept all six).

The order is safe only while no widget key is written in a run that does not draw
its widget. Streamlit keeps such a write in its old session state under the user
key, never refreshes it once the widget exists (later writes go under the widget
id) and lets it resurface, stale, once the widget is culled - in front of the
stored value. Page 2 therefore no longer pre-seeds the tabs' keys before a tab is
drawn, and a reset writes only keys whose widget the same run draws (the Disclose
Documents per-quintile sliders drop their keys; the Donation intercept inputs,
whose reset writes both keys, are seeded again from the typed override whenever
they return - ``donation_default._seed_intercept_input``).

Every key here is named at run time through a parameter; the session-key registry
lists ``widget_key`` and ``storage_dict_name`` in ``DYNAMIC_KEY_VARIABLES`` for
that reason. The callers still bind each widget key literally (``key=``).
"""
import streamlit as st


def restore(widget_key, storage_dict_name, storage_key, default):
    """Seed ``widget_key`` before its widget renders and return its current value:
    the live widget key when it exists; else the entry ``storage_key`` of the
    persistence dict ``st.session_state[storage_dict_name]``; else ``default``.
    The key is written only when it is absent."""
    if widget_key in st.session_state:
        return st.session_state[widget_key]
    storage = st.session_state.get(storage_dict_name)
    if storage and storage_key in storage:
        value = storage[storage_key]
    else:
        value = default
    st.session_state[widget_key] = value
    return value


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
