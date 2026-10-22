"""The complete-run gate, pinned: every outcome of ``can_run_complete_simulation``
and every warning text of the three screens that show it.

``decision_execution.can_run_complete_simulation`` decides whether "Run Complete
Simulation" may run and, when it may not, which decisions block it and why. This
test calls it directly (no Streamlit runtime: ``st.session_state`` is replaced by
a plain mapping) over an enumerated grid of its inputs:

* the population mode ("Compare all" means three populations, anything else one);
* per modelled decision (Disclose Income, Disclose Documents, Rejected Transaction
  Defaults, Donation Default): not selected / selected / selected with a pinned
  configuration, each with its income mode in "Compare both" or not;
* a configuration pinned for a decision that is NOT selected (the gate selects it);
* the shape of a pinned record (where its income mode and population mode sit),
  which only changes the success message;
* the Donation tab's three-level income-mode lookup and the "compare" / "both"
  detection.

The seed and the agent count are NOT gate inputs: a pin with another seed or agent
count is refused when it is made (``saved_configs.validate_seed_consistency``),
so the gate never sees one.

Every outcome - the ``(can_run, reason, config_count, block_type, blocking_issues)``
tuple - and every rendered text (``blocking_issue_text`` and
``configuration_required_text`` for each screen and block type, plus what
``render_blocking_issues`` draws) is held against ``tests/golden/complete_run_gate.json.gz``,
recorded before the gate was turned into one loop over a table. Run with
``GOLDEN_UPDATE=1`` to re-record (only when the owner changes the gate).
"""
import gzip
import itertools
import json
import os
import types

import pytest
import streamlit as st

from conftest import ROOT

import app.pages.decision_execution as execution

GOLDEN = ROOT / "tests" / "golden" / "complete_run_gate.json.gz"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"

COPULA = "Copula (synthetic)"
COMPARE_ALL = "Compare all"

#: decision -> (session key of its income mode, its non-compare mode, its compare mode)
DECISIONS = {
    "disclose_income": ("di_income_mode", "Categorical only", "Compare both"),
    "disclose_documents": ("dd_income_mode", "Categorical only", "Compare both"),
    "rejected_transaction_defaults": ("rtd_income_mode", "Continuous only", "Compare both"),
    "donation_default": ("income_spec_mode", "categorical only", "Compare both"),
}
SHORT = {"disclose_income": "di", "disclose_documents": "dd",
         "rejected_transaction_defaults": "rtd", "donation_default": "don"}

#: the per-decision states of the grid
STATES = ("off", "on", "on+pin", "on+compare", "on+pin+compare")

#: a pinned record per decision, in the shape "Use This Config" stores
PINNED = {
    "disclose_income": {"params": {"income_mode": "Categorical only"},
                        "population_mode": COPULA},
    "disclose_documents": {"params": {"income_mode": "Continuous only"},
                           "population_mode": COPULA},
    "rejected_transaction_defaults": {"params": {"income_mode": "Continuous only"},
                                      "income_mode": "Continuous only",
                                      "population_mode": COPULA},
    "donation_default": {"population_mode": COPULA,
                         "donation_income_mode": "categorical only",
                         "income_spec_mode": "categorical only"},
}


class _State(dict):
    """``st.session_state`` as the gate uses it: a mapping with attribute access
    (``hasattr`` / ``.x``) and ``.get``."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


def _session(population, selected, configs, modes, **extra):
    state = _State()
    if population is not None:
        state["population_mode"] = population
    if selected is not None:
        state["decision_params"] = types.SimpleNamespace(selected_decisions=selected)
    state["selected_decision_configs"] = dict(configs)
    for key, value in modes.items():
        state[key] = value
    state.update(extra)
    return state


def _gate(monkeypatch, state):
    monkeypatch.setattr(st, "session_state", state)
    can_run, reason, count, block_type, issues = execution.can_run_complete_simulation()
    return [can_run, reason, count, block_type,
            [[i["decision"], i["block_type"], i["config_count"], i["reason"]] for i in issues]]


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #

def _grid_rows():
    """(row key, session) for every population x per-decision state."""
    for population in (COPULA, COMPARE_ALL):
        for states in itertools.product(STATES, repeat=len(DECISIONS)):
            selected, configs, modes = [], {}, {}
            parts = [f"pop={population}"]
            for (decision, (mode_key, plain, compare)), state in zip(DECISIONS.items(), states):
                parts.append(f"{SHORT[decision]}={state}")
                modes[mode_key] = compare if "compare" in state else plain
                if state != "off":
                    selected.append(decision)
                if "pin" in state:
                    configs[decision] = PINNED[decision]
            session = _session(population, selected, configs, modes)
            # the Donation tab's own lookup: the persistence dict first, then its
            # widget key, then the global income_spec_mode (all three agree here)
            session["donation_tab_persistence"] = {"income_spec_mode": modes["income_spec_mode"]}
            session["page2_tab_income_spec_mode"] = modes["income_spec_mode"]
            yield "|".join(parts), session


def _extra_rows():
    """The cases the grid does not reach."""
    plain = {mode_key: plain for mode_key, plain, _ in DECISIONS.values()}

    # a configuration pinned for a decision that is NOT selected: the gate selects it
    for decision, (mode_key, plain_mode, compare) in DECISIONS.items():
        for population in (COPULA, COMPARE_ALL):
            for mode in (plain_mode, compare):
                modes = dict(plain, **{mode_key: mode})
                session = _session(population, [], {decision: PINNED[decision]}, modes)
                session["donation_tab_persistence"] = {"income_spec_mode": modes["income_spec_mode"]}
                yield f"pin-only|pop={population}|{SHORT[decision]}={mode}", session

    # the shape of a pinned record: where the gate finds the saved mode it reports
    shapes = {
        "disclose_income": [
            {"params": {"income_mode": "P"}, "income_mode": "T"}, {"income_mode": "T"}, {},
            {"params": {}}, {"params": {"income_mode": ""}}],
        "disclose_documents": [
            {"params": {"income_mode": "P"}, "income_mode": "T"}, {"income_mode": "T"}, {},
            {"params": {}}, {"params": {"income_mode": ""}}],
        "rejected_transaction_defaults": [
            {"params": {"income_mode": "P"}, "income_mode": "T", "population_mode": "RB"},
            {"params": {"income_mode": "P"}}, {"income_mode": "T"}, {}, {"population_mode": "RB"},
            {"params": {"income_mode": "P"}, "income_mode": ""}],
        "donation_default": [
            {"population_mode": "RB", "donation_income_mode": "D", "income_spec_mode": "S"},
            {"donation_income_mode": "D"}, {"income_spec_mode": "S"}, {}, {"population_mode": "RB"},
            {"donation_income_mode": ""}],
    }
    for decision, records in shapes.items():
        for n, record in enumerate(records):
            for population in (COPULA, None):
                session = _session(population, [decision], {decision: record}, plain)
                yield f"shape|{SHORT[decision]}#{n}|pop={population}", session

    # every pinned decision at once, in the gate's own order of the success message
    session = _session(COPULA, list(DECISIONS), PINNED, plain)
    yield "all-pinned", session
    session = _session(COPULA, ["donation_default", "disclose_income"],
                       {d: PINNED[d] for d in ("donation_default", "disclose_income")}, plain)
    yield "pinned-di-and-donation", session

    # no selection object, no selected_decisions attribute, selection None
    yield "no-decision_params", _session(COPULA, None, {}, plain)
    session = _session(COPULA, None, {}, plain)
    session["decision_params"] = types.SimpleNamespace()
    yield "decision_params-without-selected_decisions", session
    yield "selected_decisions-None", _session(COPULA, None, {}, plain,
                                             decision_params=types.SimpleNamespace(selected_decisions=None))
    # an unmodelled decision selected / pinned: nothing to check
    yield "only-bid_value-selected", _session(COPULA, ["bid_value"], {}, plain)
    yield "only-bid_value-pinned", _session(COPULA, [], {"bid_value": {"params": {}}}, plain)
    # no population mode at all (the gate's own default)
    yield "no-population_mode", _session(None, list(DECISIONS), {}, plain)
    # absent income modes (the gate's own defaults)
    yield "no-income-modes", _session(COPULA, list(DECISIONS), {}, {})

    # "compare" / "both" detection, per decision
    for decision, (mode_key, plain_mode, _) in DECISIONS.items():
        for mode in ("compare", "both", "COMPARE BOTH", "Both specifications", "x", "", None, 3):
            modes = dict(plain, **{mode_key: mode})
            session = _session(COPULA, [decision], {}, modes)
            session["donation_tab_persistence"] = {"income_spec_mode": modes["income_spec_mode"]}
            yield f"detect|{SHORT[decision]}={mode!r}", session

    # the Donation tab's three-level income-mode lookup
    lookups = {
        "persistence-wins": dict(donation_tab_persistence={"income_spec_mode": "Compare both"},
                                 page2_tab_income_spec_mode="categorical only",
                                 income_spec_mode="categorical only"),
        "persistence-without-entry": dict(donation_tab_persistence={},
                                          page2_tab_income_spec_mode="Compare both",
                                          income_spec_mode="categorical only"),
        "widget-wins": dict(page2_tab_income_spec_mode="Compare both",
                            income_spec_mode="categorical only"),
        "global-only": dict(income_spec_mode="Compare both"),
        "global-plain": dict(income_spec_mode="categorical only"),
        "nothing": dict(),
    }
    for name, keys in lookups.items():
        session = _session(COPULA, ["donation_default"], {}, {}, **keys)
        yield f"donation-lookup|{name}", session
        session = _session(COPULA, ["donation_default"], {"donation_default": PINNED["donation_default"]},
                           {}, **keys)
        yield f"donation-lookup-pinned|{name}", session


def _all_rows():
    rows = {}
    for key, session in itertools.chain(_grid_rows(), _extra_rows()):
        assert key not in rows, key
        rows[key] = session
    return rows


# --------------------------------------------------------------------------- #
# the texts
# --------------------------------------------------------------------------- #

SCREENS = ("overview", "tab", "results")
BLOCK_TYPES = ("disclose_income", "disclose_documents", "rejected_transaction_defaults",
               "donation_config", "something_else")
REASON = "R1\n\nR2"


def _texts():
    out = {}
    for screen in SCREENS:
        for block_type in BLOCK_TYPES:
            out[f"issue|{screen}|{block_type}"] = execution.blocking_issue_text(
                screen, 2, block_type, REASON)
            out[f"required|{screen}|{block_type}"] = execution.configuration_required_text(
                screen, block_type, REASON)
    return out


def _dispatch(monkeypatch):
    """What render_blocking_issues draws, as (widget, text) pairs."""
    drawn = []
    monkeypatch.setattr(st, "error", lambda text: drawn.append(["error", text]))
    monkeypatch.setattr(st, "warning", lambda text: drawn.append(["warning", text]))
    issues = [{"decision": "disclose_income", "block_type": "disclose_income",
               "config_count": 2, "reason": "A"},
              {"decision": "donation_default", "block_type": "donation_config",
               "config_count": 6, "reason": "B"}]
    out = {}
    for screen in SCREENS:
        drawn.clear()
        execution.render_blocking_issues(screen, "A\n\nB", "disclose_income", issues)
        out[f"two-issues|{screen}"] = list(drawn)
        drawn.clear()
        execution.render_blocking_issues(screen, "A", "disclose_income", issues[:1])
        out[f"one-issue|{screen}"] = list(drawn)
        drawn.clear()
        execution.render_blocking_issues(screen, "A", "disclose_income", [])
        out[f"no-issue-list|{screen}"] = list(drawn)
    return out


# --------------------------------------------------------------------------- #
# golden
# --------------------------------------------------------------------------- #

def _actual(monkeypatch):
    return {
        "outcomes": {key: _gate(monkeypatch, session) for key, session in _all_rows().items()},
        "texts": _texts(),
        "dispatch": _dispatch(monkeypatch),
    }


def _load_golden():
    assert GOLDEN.exists(), f"no golden for the gate at {GOLDEN}; run with GOLDEN_UPDATE=1"
    return json.loads(gzip.decompress(GOLDEN.read_bytes()).decode("utf-8"))


def test_gate_outcomes_and_texts_match_golden(monkeypatch):
    actual = _actual(monkeypatch)
    if UPDATE:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(actual, indent=1, sort_keys=True, ensure_ascii=False)
        GOLDEN.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        return
    expected = _load_golden()
    for section in ("outcomes", "texts", "dispatch"):
        exp, act = expected[section], actual[section]
        missing = sorted(set(exp) - set(act))
        new = sorted(set(act) - set(exp))
        assert not missing and not new, (
            f"{section}: cases changed; missing {missing[:5]}, new {new[:5]}")
        differing = [k for k in exp if exp[k] != act[k]]
        assert not differing, (
            f"{section}: {len(differing)} case(s) differ from the golden, first:\n"
            + "\n".join(f"  {k}\n    expected {json.dumps(exp[k], ensure_ascii=False)[:600]}\n"
                        f"    actual   {json.dumps(act[k], ensure_ascii=False)[:600]}"
                        for k in differing[:3]))


def test_grid_is_complete():
    """The grid covers every population x per-decision state once, and the
    golden holds every case (so a case cannot vanish quietly)."""
    rows = _all_rows()
    assert sum(1 for k in rows if not k.startswith(("pin-only", "shape", "all-", "pinned-",
                                                   "no-", "only-", "decision_params", "selected_",
                                                   "detect", "donation-lookup"))) \
        == 2 * len(STATES) ** len(DECISIONS)
    if not UPDATE:
        assert set(_load_golden()["outcomes"]) == set(rows)


@pytest.mark.parametrize("decision", list(DECISIONS))
def test_a_compare_mode_without_a_pin_blocks_and_a_pin_unblocks(monkeypatch, decision):
    """The shape of the gate, spelled out for one decision at a time."""
    mode_key, plain, compare = DECISIONS[decision]
    modes = {k: p for k, p, _ in DECISIONS.values()}
    modes[mode_key] = compare
    session = _session(COMPARE_ALL, [decision], {}, modes)
    session["donation_tab_persistence"] = {"income_spec_mode": modes["income_spec_mode"]}
    can_run, reason, count, block_type, issues = _gate(monkeypatch, session)
    title = execution.GATE_BLOCK_TYPE_TITLES[
        "donation_config" if decision == "donation_default" else decision]
    assert (can_run, count) == (False, 6)
    assert reason == (f"{title} has 6 configurations (population: 3, income: 2) - "
                      f"please run {decision} only and select one")
    assert issues == [[decision, block_type, 6, reason]]

    session["selected_decision_configs"] = {decision: PINNED[decision]}
    can_run, reason, count, block_type, issues = _gate(monkeypatch, session)
    assert (can_run, count, block_type, issues) == (True, 6, None, [])
    assert reason.startswith("Using saved configuration(s): ")


def test_blocking_issues_come_in_the_gates_order(monkeypatch):
    """Disclose Income, Disclose Documents, Rejected Transaction Defaults, Donation
    Default - and the first one's block type is the headline."""
    modes = {k: c for k, _, c in DECISIONS.values()}
    session = _session(COPULA, list(DECISIONS), {}, modes)
    session["donation_tab_persistence"] = {"income_spec_mode": "Compare both"}
    can_run, reason, count, block_type, issues = _gate(monkeypatch, session)
    assert not can_run and count == 2 and block_type == "disclose_income"
    assert [i[0] for i in issues] == list(DECISIONS)
    assert reason == "\n\n".join(i[3] for i in issues)
