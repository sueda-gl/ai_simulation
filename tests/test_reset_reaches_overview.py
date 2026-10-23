"""A tab's "Reset Config to Defaults" reaches the Page-2 Overview in the run it causes.

The Overview tab is drawn BEFORE the decision tabs, and its complete-run gate reads
their income modes. The whole-tab reset of Disclose Income, Disclose Documents and
Donation Default only raises a flag and reruns; until 2026-10-23 the reset itself
ran inside the tab, after the Overview had drawn, so after resetting a tab out of
"Compare both" the Overview still showed "<decision> has 2 configurations ..." and
a disabled "Run Complete Simulation" while the tab below already showed the reset
mode - until some other interaction reran the page (switching tabs does not).
Page 2 now applies a pending reset before the Overview
(``page2_decisions.apply_pending_tab_resets``).

Each case drives the real app through ``tests/browser_sim.py`` (every decision
selected, as a researcher running a scenario has them): the tab's income radio to
"Compare both" (the Overview's gate blocks), then the tab's reset; in the run the
reset causes, the Overview's button must be enabled and its warning gone. Decision
4's reset deletes its keys in the click's own run and never had the lag; it is not
a case here.
"""
import pytest

from browser_sim import BrowserSim
from conftest import APP_FILE
from ui_strings import COMPARE_BOTH, NEXT_TO_PAGE2

pytestmark = pytest.mark.slow

TIMEOUT = 900

#: decision -> (its income radio's key, the mode its reset restores, its reset button's
#: key, the title the gate's warning names)
CASES = {
    "disclose_income": ("di_tab_income_mode", "Categorical only", "di_reset_btn",
                        "Disclose Income"),
    "disclose_documents": ("dd_tab_income_mode", "Categorical only", "dd_reset_btn",
                           "Disclose Documents"),
    "donation_default": ("page2_tab_income_spec_mode", "categorical only", "reset_config_btn",
                         "Donation Default"),
}


def _overview(at):
    """The Overview tab (Page 2's first tab): its Run Complete Simulation button as
    (key, disabled), and the text of its warnings."""
    overview = at.tabs[0]
    assert overview.label == "📊 Overview", overview.label
    buttons = [(b.key, b.disabled) for b in overview.button
               if (b.key or "").startswith("run_complete_simulation")]
    return buttons, [str(w.value) for w in overview.warning]


@pytest.mark.parametrize("decision", list(CASES))
def test_a_reset_out_of_compare_both_unblocks_the_overview_in_the_same_run(decision):
    income_key, reset_mode, reset_key, title = CASES[decision]
    from streamlit.testing.v1 import AppTest
    sim = BrowserSim(AppTest.from_file(APP_FILE, default_timeout=TIMEOUT))
    sim.run(timeout=TIMEOUT)
    sim.set("n_agents_input", 40)
    sim.click(NEXT_TO_PAGE2)
    sim.set("page2_select_all_checkbox", True)
    assert _overview(sim.at) == ([("run_complete_simulation", False)], [])

    sim.set(income_key, COMPARE_BOTH)
    buttons, warnings = _overview(sim.at)
    assert buttons == [("run_complete_simulation_disabled", True)]
    assert any(f"{title} has 2 configurations" in w for w in warnings), warnings

    sim.click_key(reset_key)
    # the tab shows the reset mode ...
    assert sim.at.session_state[income_key] == reset_mode
    assert sim.shown(income_key) == reset_mode
    # ... and the Overview, drawn before the tab, agrees in the same run
    buttons, warnings = _overview(sim.at)
    assert buttons == [("run_complete_simulation", False)], (buttons, warnings)
    assert warnings == []
    assert not sim.mismatches()
