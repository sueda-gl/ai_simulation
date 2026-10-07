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
``end_script_run()`` must be called as the run's LAST statement.

Interrupted runs (Lavie 2026-10: rapid +/- clicks on a Decision 4 Override value
made the value jump)
---------------------------------------------------------------------------------
A click while a run is still going makes Streamlit stop that run early and start
a new one. The browser then keeps every element of the last COMPLETED run (shown
faded) plus the ones the stopped run had already re-sent - it clears elements only
when a run completes. The render log of a stopped run is therefore partial: taken
alone as "the previous run", it would make the next run re-push every widget drawn
after the stopping point (the Decision 4 tab is near the end of Page 2), and that
push carries the value of the click the run started from, overwriting the clicks
the user made since - the number jumped back (or forward, after later +/-). So
``begin_script_run`` only REPLACES the previous log when the last run completed
(``end_script_run`` was reached); after a stopped run it MERGES the stopped run's
log into the previous one, which is exactly what the browser still holds.
"""
import streamlit as st

#: widget key -> argument signature, for the widgets drawn in the CURRENT run
RENDER_LOG_KEY = "_widget_render_log"
#: the same for the PREVIOUS run (what the browser currently shows)
PREVIOUS_RENDER_LOG_KEY = "_widget_render_log_previous"
#: True once the run that wrote RENDER_LOG_KEY reached end_script_run()
RUN_COMPLETE_KEY = "_widget_render_log_complete"

_MISSING = object()

# Arguments that do not contribute to a widget's element id (callbacks), or that
# are folded into the signature in another form (format_func -> formatted options).
_NOT_IDENTITY = {"on_change", "on_click", "args", "kwargs", "format_func"}


def begin_script_run():
    """Rotate the render log. Call once per script run, before any widget.

    After a completed run the previous log is that run's log; after a run that was
    stopped early (a newer interaction arrived) it is the previous log updated with
    whatever the stopped run drew - see "Interrupted runs" above."""
    log = st.session_state.get(RENDER_LOG_KEY) or {}
    if st.session_state.get(RUN_COMPLETE_KEY, True):
        previous = dict(log)
    else:
        previous = {**(st.session_state.get(PREVIOUS_RENDER_LOG_KEY) or {}), **log}
    st.session_state[PREVIOUS_RENDER_LOG_KEY] = previous
    st.session_state[RENDER_LOG_KEY] = {}
    st.session_state[RUN_COMPLETE_KEY] = False


def end_script_run():
    """Mark the current run as completed. Call as the LAST statement of the run."""
    st.session_state[RUN_COMPLETE_KEY] = True


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
    log = st.session_state.get(RENDER_LOG_KEY)
    if log is None:
        log = {}
        st.session_state[RENDER_LOG_KEY] = log
    if key not in st.session_state:
        if initial is not _MISSING:
            st.session_state[key] = initial
    else:
        previous = st.session_state.get(PREVIOUS_RENDER_LOG_KEY) or {}
        # The CURRENT log can already hold the key only in a fragment rerun
        # (st.fragment: begin_script_run does not run, the log is the last full
        # run's): the browser then holds the widget drawn by that run.
        held = ((key in previous and previous[key] == signature)
                or (key in log and log[key] == signature))
        if not held:
            st.session_state[key] = st.session_state[key]
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
