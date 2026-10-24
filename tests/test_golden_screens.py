"""Golden screens: the parity net for the user interface.

Each journey drives the real app (``app_enhanced_new.py``, every page, every tab)
through Streamlit's headless AppTest and records, after every click, every element
on screen (type, key, label, value), every data table (shape, columns, content
hash) and every download button (label, file name, content hash with the Excel
metadata and the wall-clock timestamps normalised). The record is compared with
the copy under tests/golden_screens/. A change to what the user sees, or to what
they download, fails here with the first differing element named.

The journeys after the first eleven (tab controls, overrides across navigation,
"Compare both" and Decision 4 element runs, "Use This Config", Monte Carlo, several
vendors, the income distributions, a Page-1 edit after a run, Live mode, a pinned
Disclose Documents configuration) drive the
app through tests/browser_sim.py and take a snapshot after every interaction; each
snapshot also lists the widgets a real browser would show with a different value
than the script holds (``browser_mismatches``).

Regenerate after an INTENDED screen change:

    GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/test_golden_screens.py -q

The journeys take under a minute; they are marked ``slow``.
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

from browser_sim import BrowserSim
from conftest import ROOT
from ui_strings import (
    ANALYSIS_MONTE_CARLO, BACK_TO_PAGE1, BACK_TO_PAGE2, COMPARE_BOTH, CONTINUOUS_ONLY, CATEGORICAL_ONLY,
    EXECUTION_LIVE, NEXT_TO_PAGE2, USE_THIS_CONFIG, VIEW_RESULTS,
)

APP = ROOT / "app_enhanced_new.py"
GOLDEN_DIR = ROOT / "tests" / "golden_screens"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"
FROZEN_STARTED_AT = datetime(2025, 1, 15)   # the run's start, every export's base time
FROZEN_SELECTED_AT = datetime(2025, 1, 15, 12, 0, 0)   # the moment "Use This Config" pins a result
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

class _FrozenSelectionClock(datetime):
    """``datetime`` for app/state/saved_configs.py, whose only clock read is the
    moment a configuration is pinned ("Selected at HH:MM:SS" on the results page)."""

    @classmethod
    def now(cls, tz=None):
        return FROZEN_SELECTED_AT


class Recorder:
    """Patches the things that make a screen non-deterministic: the export base time
    (the moment the run started), the moment a configuration is pinned, the unseeded
    example bids, and the download buttons' bytes."""

    def __init__(self):
        self.downloads = []

    def __enter__(self):
        import random
        import streamlit as st
        from app import simulation
        from app.state import saved_configs
        self._started_at = simulation.run_started_at
        self._download = st.download_button
        self._uniform = random.uniform
        self._selection_clock = saved_configs.datetime
        simulation.run_started_at = lambda: FROZEN_STARTED_AT
        saved_configs.datetime = _FrozenSelectionClock
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
        from app.state import saved_configs
        simulation.run_started_at = self._started_at
        st.download_button = self._download
        random.uniform = self._uniform
        saved_configs.datetime = self._selection_clock

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


# ------------------------------------------- browser-faithful journeys (wider net)
# The journeys below drive the app through tests/browser_sim.py: every rerun sends
# the values a real browser would send back (not AppTest's server-side ones), so a
# widget the browser would draw with a different value than the script holds shows
# up in the run's results. Each snapshot also records those widgets
# (``browser_mismatches``: key, browser value, script value), and a snapshot is taken
# after EVERY interaction.

# Per modelled decision tab: the σ-strategy radio, the quintile sliders, the income
# radio, the two σ tick boxes (Copula, Research Specification), the tab's reset button.
TAB_WIDGETS = {
    "disclose_income": dict(
        strategy="di_tab_sigma_strategy_stochastic", quintile="di_tab_sigma_q{}_stochastic",
        income="di_tab_income_mode", copula_box="di_tab_sigma_in_copula",
        research_box="di_tab_sigma_enabled", reset="di_reset_btn"),
    "disclose_documents": dict(
        strategy="dd_tab_sigma_strategy_stochastic", quintile="dd_tab_sigma_q{}_stochastic",
        income="dd_tab_income_mode", copula_box="dd_tab_sigma_in_copula",
        research_box="dd_tab_sigma_enabled", reset="dd_reset_btn"),
    "donation_default": dict(
        strategy="donation_tab_sigma_strategy_stochastic", quintile="donation_tab_sigma_q{}_stochastic",
        income="page2_tab_income_spec_mode", copula_box="tab_sigma_in_copula",
        research_box="tab_sigma_in_research", reset="reset_config_btn"),
    "rejected_transaction_defaults": dict(
        strategy="rtd_tab_sigma_strategy", quintile="rtd_tab_sigma_q{}",
        income="rtd_tab_income_mode", copula_box="rtd_tab_sigma_in_copula",
        research_box="rtd_tab_sigma_enabled", reset="rtd_reset_btn"),
}
SIGMA_QUINTILE = "quintile"          # the σ-strategy radio's option value (shown as "Quintiles (...)")


class Session:
    """One browser session of the real app, recorded after every interaction."""

    def __init__(self, rec):
        from streamlit.testing.v1 import AppTest
        self.rec = rec
        self.sim = BrowserSim(AppTest.from_file(str(APP), default_timeout=TIMEOUT))
        self.steps = []
        self.sim.run(timeout=TIMEOUT)

    @property
    def at(self):
        return self.sim.at

    def record(self, step):
        snap = snapshot(self.at, self.rec, step)
        snap["browser_mismatches"] = [[normalise(str(v)) for v in m] for m in self.sim.mismatches()]
        self.steps.append(snap)
        return self

    def set(self, step, key, value):
        self.sim.set(key, value)
        return self.record(step)

    def click(self, step, label_part):
        self.sim.click(label_part)
        return self.record(step)

    def click_key(self, step, key):
        self.sim.click_key(key)
        return self.record(step)

    def choose_decisions(self, step, decisions):
        """Page 2's "Select Decisions to Run" multiselect (Select All unticked)."""
        self.at.multiselect(key="page2_manual_multiselect").set_value(list(decisions))
        self.sim.run(timeout=TIMEOUT)
        return self.record(step)

    # --- composite moves (each of their clicks is recorded) ---
    def to_page2(self, n_agents=N_AGENTS):
        self.set("page1_agents", "n_agents_input", n_agents)
        return self.click("page2", NEXT_TO_PAGE2)

    def back_to_page2(self, step="back_to_page2"):
        return self.click(step, BACK_TO_PAGE2)


def journey_tab_controls(rec, decision):
    """A modelled decision tab's σ and income controls, edited and reset, then the
    decision run alone: σ to quintiles, two quintile sliders moved, income to
    "Compare both", both σ boxes unticked and ticked again (Copula first), the tab's
    reset, then "🔬 Run <decision> Only" (the results page shows what the reset left)."""
    w = TAB_WIDGETS[decision]
    s = Session(rec).to_page2()
    s.choose_decisions(f"page2_only_{decision}", [decision])
    s.set("sigma_quintiles", w["strategy"], SIGMA_QUINTILE)
    s.set("quintile_2_moved", w["quintile"].format(2), 1.5)
    s.set("quintile_4_moved", w["quintile"].format(4), 0.5)
    s.set("income_compare_both", w["income"], COMPARE_BOTH)
    s.set("research_sigma_unticked", w["research_box"], False)     # Copula is unticked by default
    s.set("copula_sigma_ticked", w["copula_box"], True)
    s.set("research_sigma_ticked", w["research_box"], True)
    s.click_key("tab_reset", w["reset"])
    s.click_key(f"results_{decision}_only", f"run_{decision}_only_btn")
    return s.steps


def journey_donation_overrides_survive_navigation(rec):
    """Donation Default's typed overrides: the Categorical intercept, then (income mode
    switched to "Compare both", which shows both fields) the Continuous intercept and
    the Distribution Shift Value; the tab is left (Page 1 and back; decision unselected
    and selected again) and the values are still shown; then the two reset buttons."""
    s = Session(rec).to_page2()
    s.choose_decisions("page2_only_donation_default", ["donation_default"])
    s.set("categorical_intercept_typed", "override_categorical_intercept", 1.234567)
    s.set("income_compare_both", "page2_tab_income_spec_mode", COMPARE_BOTH)
    s.set("continuous_intercept_typed", "override_continuous_intercept", -0.5)
    s.set("shift_typed", "override_adjustment_shift", 2.5)
    s.click("left_for_page1", BACK_TO_PAGE1)
    s.click("back_on_page2", NEXT_TO_PAGE2)
    s.choose_decisions("decision_unselected", [])
    s.choose_decisions("decision_selected_again", ["donation_default"])
    s.click_key("intercepts_reset", "reset_intercept_btn")
    s.click_key("adjustment_reset", "reset_adjustment_btn")
    return s.steps


def journey_compare_both_and_d4_elements(rec):
    """Disclose Income, Disclose Documents and Decision 4 each run alone in "Compare
    both"; then, still in "Compare both", Decision 4's Loyalty Ranking element run; then
    Decision 4 back on "Categorical only" (its default) and the Integrated Default List
    (rank aggregation) run."""
    s = Session(rec).to_page2()
    for decision, income_key in (("disclose_income", "di_tab_income_mode"),
                                 ("disclose_documents", "dd_tab_income_mode"),
                                 ("rejected_transaction_defaults", "rtd_tab_income_mode")):
        s.choose_decisions(f"page2_only_{decision}", [decision])
        s.set(f"{decision}_compare_both", income_key, COMPARE_BOTH)
        s.click_key(f"results_{decision}_compare_both", f"run_{decision}_only_btn")
        s.back_to_page2(f"back_from_{decision}")
    s.click_key("results_d4_loyalty_compare_both", "rtd_run_loyalty_btn")
    s.back_to_page2("back_from_loyalty")
    s.set("d4_categorical_only", "rtd_tab_income_mode", CATEGORICAL_ONLY)
    s.click_key("results_d4_integrated_default_list", "rtd_run_aggregation_btn")
    return s.steps


def journey_use_this_config_then_complete(rec):
    """"Use This Config" on a single-decision results page (Decision 4, run alone),
    the pinned-configuration card on the Page 2 Overview, then a complete run that
    uses the pinned configuration."""
    s = Session(rec).to_page2()
    s.choose_decisions("page2_only_d4", ["rejected_transaction_defaults"])
    s.click_key("results_d4_only", "run_rejected_transaction_defaults_only_btn")
    s.click("config_pinned", USE_THIS_CONFIG)
    s.back_to_page2("page2_overview_with_pinned_card")
    s.click_key("results_complete_with_pinned_config", "run_complete_simulation")
    return s.steps


def journey_pin_disclose_documents(rec):
    """"Use This Config" on a Disclose Documents result (the decision run alone, in
    "Compare both": the results page offers the button where it shows more than one
    configuration), then the pinned-configuration card on the Page 2 Overview."""
    s = Session(rec).to_page2()
    s.choose_decisions("page2_only_disclose_documents", ["disclose_documents"])
    s.set("disclose_documents_compare_both", "dd_tab_income_mode", COMPARE_BOTH)
    s.click_key("results_disclose_documents_compare_both", "run_disclose_documents_only_btn")
    s.click("config_pinned", USE_THIS_CONFIG)
    s.back_to_page2("page2_overview_with_pinned_card")
    return s.steps


def journey_monte_carlo_complete(rec):
    """A Monte-Carlo study (2 runs - the smallest the widget allows - of 40 agents),
    every decision selected, through "Run Complete Simulation"."""
    s = Session(rec)
    s.set("analysis_monte_carlo", "page1_simulation_mode", ANALYSIS_MONTE_CARLO)
    s.set("mc_runs_2", "n_runs_input", 2)
    s.to_page2()
    s.set("all_selected", "page2_select_all_checkbox", True)
    s.click_key("results_monte_carlo", "run_complete_simulation")
    return s.steps


def journey_multi_vendor_complete(rec):
    """Page 1 with three vendors (the multi-vendor controls appear), then a complete
    run with every decision selected."""
    s = Session(rec)
    s.set("three_vendors", "num_vendors_input", 3)
    s.to_page2()
    s.set("all_selected", "page2_select_all_checkbox", True)
    s.click_key("results_complete_three_vendors", "run_complete_simulation")
    return s.steps


def journey_page1_income_distributions(rec):
    """Page 1 with each income distribution type the selectbox offers, in turn, and
    back to the first (screens only; the selectbox offers lognormal, generalised gamma
    and Dagum - no uniform and no file upload)."""
    s = Session(rec).record("page1_fresh")
    options = s.at.selectbox(key="page1_income_distribution").options   # the first is drawn
    for option in options[1:] + options[:1]:
        s.set(f"income_distribution_{option}", "page1_income_distribution", option)
    return s.steps


def journey_page1_edit_after_run(rec):
    """A Page-1 edit AFTER a run: a complete run, back to Page 1, Platform Markup and
    Number of Agents changed, then the results page again WITHOUT re-running (via
    Page 2's "View Results"). Pins whatever the page and its downloads show then."""
    s = Session(rec).to_page2()
    s.set("all_selected", "page2_select_all_checkbox", True)
    s.click_key("results_complete", "run_complete_simulation")
    s.click("back_on_page1", BACK_TO_PAGE1)
    s.set("markup_changed", "platform_markup_slider", 0.2)
    s.set("agents_changed", "n_agents_input", 60)
    s.click("page2_after_edit", NEXT_TO_PAGE2)
    s.click("results_after_edit_not_rerun", VIEW_RESULTS)
    return s.steps


def journey_live_execution_mode(rec):
    """Execution Mode "Live Simulation" (Page 1 radio), then a small complete run."""
    s = Session(rec)
    s.set("execution_live", "page1_simulation_execution_mode", EXECUTION_LIVE)
    s.to_page2()
    s.set("all_selected", "page2_select_all_checkbox", True)
    s.click_key("results_live", "run_complete_simulation")
    return s.steps


JOURNEYS = {
    "page1": journey_page1,
    "page2": journey_page2,
    "complete_copula": journey_complete_copula,
    "complete_research_spec": journey_complete_research_spec,
    "complete_baseline": journey_complete_baseline,
    "compare_all": journey_compare_all,
    "complete_continuous": journey_complete_continuous,
    **{f"single_{d}": (lambda rec, d=d: _single(rec, d)) for d in MODELLED},
    # browser-faithful journeys (each step also records browser_mismatches)
    **{f"tab_controls_{d}": (lambda rec, d=d: journey_tab_controls(rec, d)) for d in MODELLED},
    "donation_overrides_survive_navigation": journey_donation_overrides_survive_navigation,
    "compare_both_and_d4_elements": journey_compare_both_and_d4_elements,
    "use_this_config_then_complete": journey_use_this_config_then_complete,
    "pin_disclose_documents": journey_pin_disclose_documents,
    "monte_carlo_complete": journey_monte_carlo_complete,
    "multi_vendor_complete": journey_multi_vendor_complete,
    "page1_income_distributions": journey_page1_income_distributions,
    "page1_edit_after_run": journey_page1_edit_after_run,
    "live_execution_mode": journey_live_execution_mode,
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
        if e_step.get("browser_mismatches") != a_step.get("browser_mismatches"):
            problems.append(f"{e_step['step']}: widgets the browser shows differently "
                            f"(key, browser, script): {e_step.get('browser_mismatches')} -> "
                            f"{a_step.get('browser_mismatches')}")
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
