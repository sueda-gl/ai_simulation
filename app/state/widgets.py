# app/state/widgets.py
"""Keyed widgets whose value lives in ``st.session_state`` - drawn so the BROWSER
shows that value too.

Why this module exists (2026-10-07, regression from e51c3fc)
-----------------------------------------------------------
The pages keep each widget's value in its session-state key and pass no
``value=`` / ``index=`` (Q-41: passing both a default and a key written through
the Session State API makes Streamlit print "The widget with key ... was created
with a default value but also had its value set via the Session State API").

Streamlit, however, only sends a key's value to the browser in the run in which
the key was WRITTEN through ``st.session_state`` (``set_value`` on the element).
In every other run the element carries just its ``default=``, and without
``value=`` that default is the widget's built-in one: ``min_value`` for a slider
or number input, ``False`` for a checkbox, option 0 for a radio / selectbox. A
browser that already shows the widget keeps its own value, but a browser that
draws it for the FIRST time - a fresh session, a tab whose decision was just
selected, a page navigated back to, or a widget whose label / range / help (and
therefore its element id) changed - shows that built-in default, and sends it
back on the next rerun, so the run computes with it. The script itself (and
AppTest) still sees the seeded value, so nothing on the server looks wrong.

Concretely: ``initialize_page2_widget_keys`` seeds ``tab_anchor_weight = 0.75``
on the first Page-2 run, when the Donation tab is not drawn yet; one rerun later
"Select All Decisions" draws the slider, the key already exists so nobody writes
it again, the browser shows 0 and the next run takes ``anchor_observed_weight =
0.0``. Before e51c3fc every one of these widgets passed ``value=<the key's
value>``, which made ``default`` right (and the warning appear).

The fix
-------
``stateful(widget_fn, *args, key=..., initial=..., **kwargs)`` draws the widget
without ``value=`` / ``index=`` (so no warning) and, immediately before, writes
the key through the Session State API whenever the browser may not hold the
widget already:

* the key is absent (first render, or Streamlit dropped it because the widget
  was not drawn in some run) -> it is seeded with ``initial``;
* the widget was not drawn in the PREVIOUS script run, or was drawn with
  different arguments (a different element id) -> the current value is written
  back unchanged, which makes Streamlit push it to the browser.

A widget drawn in the previous run with the same arguments is left alone: the
browser holds exactly that value, and re-pushing it on every rerun is what made
the +/- buttons of number inputs jump back a step under fast clicks (professor,
2026-09-17).

``begin_script_run()`` must be called once at the top of every script run (the
entry point does it); it rotates the per-run render log the rule above reads.
"""
import streamlit as st

#: widget key -> argument signature, for the widgets drawn in the CURRENT run
RENDER_LOG_KEY = "_widget_render_log"
#: the same for the PREVIOUS run (what the browser currently shows)
PREVIOUS_RENDER_LOG_KEY = "_widget_render_log_previous"

_MISSING = object()

# Arguments that do not contribute to a widget's element id (callbacks), or that
# are folded into the signature in another form (format_func -> formatted options).
_NOT_IDENTITY = {"on_change", "on_click", "args", "kwargs", "format_func"}


def begin_script_run():
    """Rotate the render log. Call once per script run, before any widget."""
    st.session_state[PREVIOUS_RENDER_LOG_KEY] = st.session_state.get(RENDER_LOG_KEY) or {}
    st.session_state[RENDER_LOG_KEY] = {}


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_freeze(v) for v in value]
        return tuple(sorted(items, key=repr)) if isinstance(value, (set, frozenset)) else tuple(items)
    if callable(value):
        return None
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def widget_signature(widget_fn, args, kwargs):
    """Everything that can change the element id Streamlit gives the widget."""
    name = getattr(widget_fn, "__name__", repr(widget_fn))
    sig_kwargs = {k: v for k, v in kwargs.items() if k not in _NOT_IDENTITY}
    format_func = kwargs.get("format_func")
    options = kwargs.get("options", args[1] if len(args) > 1 else None)
    formatted = None
    if format_func is not None and options is not None:
        try:
            formatted = tuple(str(format_func(o)) for o in options)
        except Exception:
            formatted = None
    return (name, _freeze(args), _freeze(sig_kwargs), formatted)


def sync_widget_key(key, initial=_MISSING, signature=None):
    """Make the browser show st.session_state[key] for a widget about to be drawn.

    Seeds an absent key with ``initial``; otherwise re-writes the current value
    (unchanged) when the widget was not drawn in the previous run with the same
    signature. Returns the value the widget will show (or None when the key stays
    absent because no initial value was given)."""
    if key not in st.session_state:
        if initial is not _MISSING:
            st.session_state[key] = initial
    else:
        previous = st.session_state.get(PREVIOUS_RENDER_LOG_KEY) or {}
        if key not in previous or previous[key] != signature:
            st.session_state[key] = st.session_state[key]
    log = st.session_state.get(RENDER_LOG_KEY)
    if log is None:
        log = {}
        st.session_state[RENDER_LOG_KEY] = log
    log[key] = signature
    return st.session_state[key] if key in st.session_state else None


def stateful(widget_fn, *args, key, initial=_MISSING, **kwargs):
    """Draw ``widget_fn(*args, key=key, **kwargs)`` - a keyed widget with NO
    ``value=`` / ``index=`` / ``default=`` - so that both the script and the
    browser use ``st.session_state[key]`` (seeded with ``initial`` when absent)."""
    for forbidden in ("value", "index", "default"):
        if forbidden in kwargs:
            raise TypeError(f"stateful(): pass the value through session state / initial=, "
                            f"not {forbidden}= (key {key!r})")
    sync_widget_key(key, initial, widget_signature(widget_fn, args, kwargs))
    return widget_fn(*args, key=key, **kwargs)
