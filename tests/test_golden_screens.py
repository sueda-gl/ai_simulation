"""Golden screens: the parity net for the user interface.

Each journey drives the real app (``app_enhanced_new.py``, every page, every tab)
through Streamlit's headless AppTest and records, after every click, every element
on screen (type, key, label, value), every data table (shape, columns, content
hash) and every download button (label, file name, content hash with the Excel
metadata and the wall-clock timestamps normalised). The record is compared with
the copy under tests/golden_screens/. A change to what the user sees, or to what
they download, fails here with the first differing element named.

Regenerate after an INTENDED screen change:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/test_golden_screens.py -q

The journeys take about two minutes; they are marked ``slow``.
"""
import gzip
import hashlib
import io
import json
import os
import re
from datetime import datetime

import pandas as pd
import pytest

pytestmark = pytest.mark.slow

from conftest import ROOT

APP = ROOT / "app_enhanced_new.py"
GOLDEN_DIR = ROOT / "tests" / "golden_screens"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"
FROZEN_STARTED_AT = datetime(2025, 1, 15)   # the run's start, every export's base time
N_AGENTS = 40
TIMEOUT = 900

MODELLED = ["disclose_income", "disclose_documents", "donation_default",
            "rejected_transaction_defaults"]
CONTAINERS = {"root", "main", "column", "flex_container", "block", "form", "tab", "sidebar"}

# wall-clock fragments that legitimately differ between runs
_STAMP_RES = [
    (re.compile(r"\d{8}_\d{6}(_\d+)?"), "<STAMP>"),           # file names
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?"), "<DATETIME>"),
    (re.compile(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}(:\d{2})?"), "<DATETIME>"),
]


def normalise(text):
    for pattern, replacement in _STAMP_RES:
        text = pattern.sub(replacement, text)
    return text


# --------------------------------------------------------------- recording

class Recorder:
    """Patches the two things that make a screen non-deterministic: the export base
    time (the moment the run started) and the download buttons' bytes."""

    def __init__(self):
        self.downloads = []

    def __enter__(self):
        import random
        import streamlit as st
        from app import simulation
        self._started_at = simulation.run_started_at
        self._download = st.download_button
        self._uniform = random.uniform
        simulation.run_started_at = lambda: FROZEN_STARTED_AT
        # one caption draws unseeded example bids on every render; pin them
        random.uniform = lambda a, b: (a + b) / 2
        recorder = self

        def download_button(label, data=None, file_name=None, *args, **kwargs):
            recorder.downloads.append(describe_download(label, data, file_name))
            return recorder._download(label, data, file_name, *args, **kwargs)

        st.download_button = download_button
        return self

    def __exit__(self, *exc):
        import random
        import streamlit as st
        from app import simulation
        simulation.run_started_at = self._started_at
        st.download_button = self._download
        random.uniform = self._uniform

    def take(self):
        out, self.downloads = self.downloads, []
        return out


def _frame_digest(df):
    text = df.to_json(orient="split", double_precision=15, default_handler=str, date_format="iso")
    return hashlib.sha256(normalise(text).encode()).hexdigest()


def describe_download(label, data, file_name):
    if hasattr(data, "getvalue"):
        data = data.getvalue()
    if isinstance(data, str):
        data = data.encode("utf-8")
    name = normalise(str(file_name))
    # no byte count: xlsx zip metadata varies by a few bytes between identical workbooks
    record = {"label": normalise(str(label)), "file_name": name}
    if data and name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        record["sheets"] = {s: {"shape": list(f.shape), "digest": _frame_digest(f)} for s, f in sheets.items()}
    elif data:
        record["digest"] = hashlib.sha256(normalise(data.decode("utf-8", "replace")).encode()).hexdigest()
    return record


def describe_element(node):
    etype = getattr(node, "type", type(node).__name__)
    if etype in CONTAINERS:
        return None
    record = {"type": etype}
    for attr in ("key", "label"):
        value = getattr(node, attr, None)
        if value not in (None, ""):
            record[attr] = normalise(str(value))
    if etype == "plotly_chart":
        return record                      # figure JSON carries random ids
    try:
        value = node.value
    except Exception:
        return record
    if isinstance(value, pd.DataFrame):
        record["shape"] = list(value.shape)
        record["columns"] = [str(c) for c in value.columns]
        record["digest"] = _frame_digest(value)
    elif isinstance(value, (list, tuple)):
        record["value"] = [normalise(str(v)) for v in value]
    elif value is not None:
        record["value"] = normalise(str(value))
    return record


def _page(at):
    try:
        return at.session_state["page"]
    except KeyError:
        return None


def snapshot(at, recorder, step):
    assert not at.exception, f"{step}: {[str(e.value) for e in at.exception]}"
    elements = [e for e in (describe_element(n) for n in at._tree) if e]
    return {"step": step, "page": _page(at), "elements": elements,
            "downloads": recorder.take()}


# ---------------------------------------------------------------- driving

def start():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    return at


def click_label(at, label_part):
    buttons = [b for b in at.button if label_part in b.label and not b.disabled]
    assert buttons, f"no enabled button containing {label_part!r}; have {[b.label for b in at.button]}"
    buttons[0].click().run()
    return at


def click_key(at, key):
    at.button(key=key).click().run()
    return at


def set_widget(at, kind, key, value):
    getattr(at, kind)(key=key).set_value(value).run()
    return at


def go_to_page2(at):
    set_widget(at, "number_input", "n_agents_input", N_AGENTS)
    return click_label(at, "Next: Decision Parameters")


def select_all(at):
    at.checkbox(key="page2_select_all_checkbox").check().run()
    return at


def select_only(at, decision):
    at.checkbox(key="page2_select_all_checkbox").uncheck().run()
    at.multiselect(key="page2_manual_multiselect").set_value([decision]).run()
    return at


def population(at, label):
    """Set the Page-1 population mode by its widget (found by label), while on Page 1."""
    widgets = [w for w in at.selectbox if "Population" in (w.label or "")] + \
              [w for w in at.radio if "Population" in (w.label or "")]
    assert widgets, "no population-mode widget on screen"
    widgets[0].set_value(label).run()
    return at


# --------------------------------------------------------------- journeys

def journey_page1(rec):
    at = start()
    steps = [snapshot(at, rec, "page1_fresh")]
    set_widget(at, "number_input", "n_agents_input", 123)
    steps.append(snapshot(at, rec, "page1_edited"))
    click_key(at, "reset_page1_defaults")
    steps.append(snapshot(at, rec, "page1_after_reset"))
    return steps


def journey_page2(rec):
    at = start()
    go_to_page2(at)
    steps = [snapshot(at, rec, "page2_nothing_selected")]
    select_all(at)
    steps.append(snapshot(at, rec, "page2_all_tabs"))
    for decision in MODELLED:
        select_only(at, decision)
        steps.append(snapshot(at, rec, f"page2_only_{decision}"))
    return steps


def _complete(rec, pop_label=None, continuous=False):
    at = start()
    if pop_label:
        population(at, pop_label)
    go_to_page2(at)
    select_all(at)
    if continuous:
        for key in ("income_spec_mode", "di_income_mode", "dd_income_mode", "rtd_income_mode",
                    "di_tab_income_mode", "dd_tab_income_mode", "rtd_tab_income_mode"):
            at.session_state[key] = "Continuous only" if key != "income_spec_mode" else "continuous only"
        at.run()
    click_key(at, "run_complete_simulation")
    steps = [snapshot(at, rec, "results")]
    click_label(at, "Back")
    steps.append(snapshot(at, rec, "back_to_page2_with_results"))
    return steps


def journey_complete_copula(rec):
    return _complete(rec)


def journey_complete_research_spec(rec):
    return _complete(rec, "Research Specification")


def journey_complete_baseline(rec):
    return _complete(rec, "Research Baseline")


def journey_compare_all(rec):
    """Compare all: a complete run is refused (every modelled decision would have
    three configurations); the page says so, and a single decision runs as a grid."""
    at = start()
    population(at, "Compare all")
    go_to_page2(at)
    select_all(at)
    steps = [snapshot(at, rec, "page2_complete_run_refused")]
    select_only(at, "donation_default")
    click_key(at, "run_donation_default_only_btn")
    steps.append(snapshot(at, rec, "results_donation_default_compare_all"))
    return steps


def journey_complete_continuous(rec):
    return _complete(rec, continuous=True)


def _single(rec, decision):
    at = start()
    go_to_page2(at)
    select_only(at, decision)
    click_key(at, f"run_{decision}_only_btn")
    return [snapshot(at, rec, f"results_{decision}_only")]


JOURNEYS = {
    "page1": journey_page1,
    "page2": journey_page2,
    "complete_copula": journey_complete_copula,
    "complete_research_spec": journey_complete_research_spec,
    "complete_baseline": journey_complete_baseline,
    "compare_all": journey_compare_all,
    "complete_continuous": journey_complete_continuous,
    **{f"single_{d}": (lambda rec, d=d: _single(rec, d)) for d in MODELLED},
}


# ------------------------------------------------------------- comparison

def _first_difference(expected, actual):
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            return i, e, a
    if len(expected) != len(actual):
        i = min(len(expected), len(actual))
        return i, (expected[i] if i < len(expected) else None), (actual[i] if i < len(actual) else None)
    return None


def compare(journey, expected, actual):
    problems = []
    for e_step, a_step in zip(expected, actual):
        if e_step["step"] != a_step["step"]:
            problems.append(f"step order: {e_step['step']} vs {a_step['step']}")
            break
        if e_step["page"] != a_step["page"]:
            problems.append(f"{e_step['step']}: page {e_step['page']} -> {a_step['page']}")
        diff = _first_difference(e_step["elements"], a_step["elements"])
        if diff:
            i, e, a = diff
            problems.append(f"{e_step['step']}: elements {len(e_step['elements'])} -> "
                            f"{len(a_step['elements'])}; first difference at #{i}:\n"
                            f"    expected {json.dumps(e)[:300]}\n    actual   {json.dumps(a)[:300]}")
        diff = _first_difference(e_step["downloads"], a_step["downloads"])
        if diff:
            i, e, a = diff
            problems.append(f"{e_step['step']}: downloads {len(e_step['downloads'])} -> "
                            f"{len(a_step['downloads'])}; first difference at #{i}:\n"
                            f"    expected {json.dumps(e)[:300]}\n    actual   {json.dumps(a)[:300]}")
    if len(expected) != len(actual):
        problems.append(f"steps: {len(expected)} expected, {len(actual)} actual")
    assert not problems, f"journey {journey!r} differs from its golden copy:\n" + "\n".join(problems)


@pytest.mark.parametrize("journey", list(JOURNEYS))
def test_screens_match_golden(journey):
    with Recorder() as rec:
        actual = JOURNEYS[journey](rec)
    path = GOLDEN_DIR / f"{journey}.json.gz"
    text = json.dumps(actual, indent=1, sort_keys=True)
    if UPDATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        return
    assert path.exists(), f"no golden screens for {journey!r}; run with GOLDEN_UPDATE=1"
    expected = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    compare(journey, expected, actual)
