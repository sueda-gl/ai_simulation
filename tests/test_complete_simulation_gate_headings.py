"""
The "Run Complete Simulation" gate names the decision that blocks it.

``can_run_complete_simulation()`` reports one blocking issue per decision whose
"Compare both" (or Compare-all population) setting would produce several
configurations.  With a single issue, every gate - each decision tab's
"🚀 Simulation Options" (``render_simulation_buttons``), the Page-2 Overview's
"Complete Simulation" section and the results page - shows that decision's own
"Configuration Required" heading.  "⚠️ Multiple Donation Configurations Detected"
belongs to the donation block only: it used to be the ``else`` of the
single-issue branch, i.e. the heading of any block type a gate did not list
(Page 2 did not list Disclose Documents, so a Disclose Documents block was
announced as a donation one).

Reproduced in the app through the browser model: Page 2, Select All, the Decision
4 tab's income specification set to "Compare both" (the only blocking decision),
then every tab's Simulation Options read.
"""
import pytest
from streamlit.testing.v1 import AppTest

from browser_sim import BrowserSim
from test_widget_browser_state import APP_FILE

DONATION_HEADING = "⚠️ **Multiple Donation Configurations Detected**"


def _gate_headings(at):
    """{tab label: [first line of each gate warning]} for every top-level Page-2 tab."""
    headings = {}
    for tab in at.tabs:
        if not (tab.label.startswith("🎯") or tab.label.startswith("📊")):
            continue   # the Decision 4 tab's own sub-tabs
        headings[tab.label] = [str(w.value).strip().splitlines()[0] for w in tab.warning
                               if "Configuration" in str(w.value).strip().splitlines()[0]]
    return headings


def _page2_all_selected():
    sim = BrowserSim(AppTest.from_file(APP_FILE, default_timeout=300))
    sim.run()
    sim.click("Decision Parameters")
    sim.set("page2_select_all_checkbox", True)
    assert not sim.at.exception
    return sim


@pytest.mark.parametrize("tab_key, session_key, heading", [
    ("rtd_tab_income_mode", "rtd_income_mode",
     "⚠️ **Rejected Transaction Defaults Configuration Required**"),
    ("dd_tab_income_mode", "dd_income_mode",
     "⚠️ **Disclose Documents Configuration Required**"),
])
def test_single_block_shows_its_own_heading_in_every_gate(tab_key, session_key, heading):
    sim = _page2_all_selected()
    sim.set(tab_key, "Compare both")
    at = sim.at
    assert not at.exception
    assert at.session_state[session_key] == "Compare both"

    headings = _gate_headings(at)

    decision_tabs = [label for label in headings if label.startswith("🎯")]
    assert len(decision_tabs) == 13
    for label, found in headings.items():
        assert found == [heading], (label, found)
        assert DONATION_HEADING not in found, label
    # every decision tab's Complete Simulation button is disabled
    disabled = [b for b in at.button if b.key and b.key.endswith("_btn_disabled")]
    assert len(disabled) == 13 and all(b.disabled for b in disabled)


def test_donation_block_keeps_the_donation_heading():
    """The donation block itself still reads 'Multiple Donation Configurations Detected'."""
    sim = _page2_all_selected()
    sim.set("page2_tab_income_spec_mode", "Compare both")
    at = sim.at
    assert not at.exception
    headings = _gate_headings(at)
    assert headings and all(found == [DONATION_HEADING] for found in headings.values()), headings
