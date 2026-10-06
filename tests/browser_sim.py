"""
A browser-faithful driver for Streamlit's AppTest.

AppTest is not a browser. It reports a widget's value straight from the
server-side session state and sends that same value back on the next run, so a
widget that a real browser draws with the WRONG value still looks right to it.

That is exactly the situation of a keyed widget that passes no ``value=`` /
``index=`` and whose key was written through ``st.session_state`` in an EARLIER
script run than the one that draws it:

* the script reads the key and gets the seeded value (AppTest is happy);
* the element sent to the browser carries ``set_value = False`` (the key is not a
  "new" Session-State value in this run) and ``default =`` the widget's built-in
  default - ``min_value`` for a slider / number input, ``False`` for a checkbox,
  option 0 for a radio / selectbox;
* the browser has never drawn a widget with that element id (a fresh session, a
  tab whose decision was just selected, a page navigated back to, a widget whose
  label / range / help changed), so it shows ``default`` - and on the next rerun
  it SENDS ``default`` back, and the run computes with it.

``BrowserSim`` models the frontend's widget-state manager:

* after each run, for every widget element in the tree, the browser value is
  ``proto.value`` when ``proto.set_value`` is true, else the value the browser
  already holds for that element id (if it drew it in the previous run), else
  ``proto.default``;
* widgets not drawn in a run are forgotten;
* the next run sends the browser values - except for the widgets the test itself
  set (``el.set_value(...)`` / ``button.click()``), which are a user interaction.

``mismatches()`` lists every widget whose browser value differs from the value the
server-side script holds - the invariant that broke at e51c3fc.
"""
from __future__ import annotations

import math

from streamlit.proto.WidgetStates_pb2 import WidgetState, WidgetStates
from streamlit.testing.v1.element_tree import (
    Checkbox, InitialValue, NumberInput, Radio, Selectbox, Slider,
)

_SIMULATED = (Slider, Checkbox, NumberInput, Radio, Selectbox)


def _user_set(el) -> bool:
    v = getattr(el, "_value", None)
    return v is not None and not isinstance(v, InitialValue)


def _browser_raw(el, previous):
    """The value the browser holds for `el` after this run, in proto form."""
    p = el.proto
    if isinstance(el, Slider):
        if p.set_value:
            return list(p.value)
        return previous if previous is not None else list(p.default)
    if isinstance(el, Checkbox):
        if p.set_value:
            return bool(p.value)
        return previous if previous is not None else bool(p.default)
    if isinstance(el, NumberInput):
        if p.set_value:
            return p.value if p.HasField("value") else None
        if previous is not None:
            return previous
        return p.default if p.HasField("default") else None
    if isinstance(el, (Radio, Selectbox)):
        # an option index
        if p.set_value:
            if isinstance(el, Selectbox):
                return list(p.options).index(p.raw_value) if p.HasField("raw_value") else None
            return p.value if p.HasField("value") else None
        if previous is not None:
            return previous
        return p.default if p.HasField("default") else None
    raise TypeError(type(el))


def _server_raw(el):
    """The value the server-side script holds for `el`, in the same proto form."""
    if isinstance(el, Slider):
        v = el.value
        return list(v) if isinstance(v, (list, tuple)) else [v]
    if isinstance(el, (Checkbox, NumberInput)):
        return el.value
    if isinstance(el, (Radio, Selectbox)):
        return el.index
    raise TypeError(type(el))


def _widget_state(el, raw) -> WidgetState:
    ws = WidgetState()
    ws.id = el.id
    if raw is None:
        return ws
    if isinstance(el, Slider):
        ws.double_array_value.data[:] = raw
    elif isinstance(el, Checkbox):
        ws.bool_value = bool(raw)
    elif isinstance(el, NumberInput):
        if el.proto.data_type == el.proto.INT:
            ws.int_value = int(raw)
        else:
            ws.double_value = float(raw)
    elif isinstance(el, Radio):
        ws.int_value = int(raw)
    elif isinstance(el, Selectbox):
        ws.string_value = el.options[int(raw)]
    return ws


def _same(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b) and a is not None and b is not None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


class BrowserSim:
    def __init__(self, at):
        self.at = at
        self.browser = {}          # element id -> raw value the browser holds

    def _widgets(self):
        return [n for n in self.at._tree if isinstance(n, _SIMULATED) and n.id]

    def run(self, timeout=300):
        states = self.at._tree.get_widget_states()
        by_id = {el.id: el for el in self._widgets()}
        out = WidgetStates()
        for ws in states.widgets:
            el = by_id.get(ws.id)
            if el is not None and not _user_set(el) and ws.id in self.browser:
                out.widgets.append(_widget_state(el, self.browser[ws.id]))
            else:
                out.widgets.append(ws)
        user_values = {el.id: _server_raw(el) for el in by_id.values() if _user_set(el)}
        self.at._run(out, timeout=timeout)
        new = {}
        for el in self._widgets():
            previous = user_values.get(el.id, self.browser.get(el.id))
            new[el.id] = _browser_raw(el, previous)
        self.browser = new
        return self.at

    # --- reading -----------------------------------------------------------
    def element(self, key):
        found = [el for el in self._widgets() if getattr(el, "key", None) == key]
        assert found, f"no widget with key {key!r} on screen"
        return found[0]

    def shown(self, key):
        """What the browser displays for the widget bound to `key`."""
        el = self.element(key)
        raw = self.browser.get(el.id)
        if isinstance(el, Slider):
            return raw[0] if raw and len(raw) == 1 else raw
        if isinstance(el, (Radio, Selectbox)):
            return None if raw is None else el.options[int(raw)]
        return raw

    def mismatches(self):
        """Widgets whose browser value differs from the script's value."""
        bad = []
        for el in self._widgets():
            raw = self.browser.get(el.id)
            server = _server_raw(el)
            if not _same(raw, server):
                bad.append((getattr(el, "key", None) or el.label, raw, server))
        return bad

    # --- acting --------------------------------------------------------------
    def set(self, key, value):
        self.element(key).set_value(value)
        return self.run()

    def click(self, label_part):
        buttons = [b for b in self.at.button if label_part in b.label and not b.disabled]
        assert buttons, f"no enabled button containing {label_part!r}"
        buttons[0].click()
        return self.run()

    def click_key(self, key):
        button = self.at.button(key=key)
        assert not button.disabled, f"button {key!r} is disabled"
        button.click()
        return self.run()
