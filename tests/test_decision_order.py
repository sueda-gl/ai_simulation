"""
The decision order is defined once, in the engine (src/engine/core.py
DECISION_ORDER): app.models.ALL_DECISIONS (the screens) and
app.seam.build_plan.ALL_DECISIONS (the seam) are copies of it, and the
contract's default registries cover exactly those decisions.

Every decision in the order must load: a decision whose module cannot be imported,
or lacks its function, stops the Engine at construction, naming the decision, and a
decision without a loaded module stops the run (the engine used to print a warning
and run on without that decision's columns).
"""
import pandas as pd
import pytest

from app.models import ALL_DECISIONS as SCREEN_ORDER
from app.seam.build_plan import ALL_DECISIONS as SEAM_ORDER
from src.contract.defaults import DEFAULT_DECISION_DESCRIPTIONS, DEFAULT_DECISION_VALUES
from src.engine.core import DECISION_ORDER


def test_the_three_decision_lists_are_the_engine_order():
    assert list(SCREEN_ORDER) == list(DECISION_ORDER)
    assert list(SEAM_ORDER) == list(DECISION_ORDER)
    assert len(DECISION_ORDER) == 13
    assert len(set(DECISION_ORDER)) == 13


def test_default_registries_cover_exactly_the_decision_order():
    assert set(DEFAULT_DECISION_VALUES) == set(DECISION_ORDER)
    assert set(DEFAULT_DECISION_DESCRIPTIONS) == set(DECISION_ORDER)


def test_every_decision_in_the_order_loads():
    from src.engine.core import Engine
    assert Engine("baseline").get_available_decisions() == list(DECISION_ORDER)


def test_a_decision_whose_module_cannot_be_imported_stops_the_engine(monkeypatch):
    from src.engine import core
    monkeypatch.setattr(core, "DECISION_ORDER", list(DECISION_ORDER) + ["no_such_decision"])
    with pytest.raises(ImportError, match="decision 'no_such_decision': could not load "
                                          r"src\.decisions\.no_such_decision"):
        core.Engine("baseline")


def test_a_decision_module_without_its_function_stops_the_engine(monkeypatch):
    import src.decisions.final_donation_rate as module
    from src.engine.core import Engine
    monkeypatch.delattr(module, "final_donation_rate")
    with pytest.raises(ImportError, match="decision 'final_donation_rate'"):
        Engine("baseline")


def test_a_decision_without_a_loaded_module_stops_the_run():
    from src.engine.core import Engine
    engine = Engine("baseline")
    del engine.decision_modules["final_donation_rate"]
    agents = pd.DataFrame([{"Assigned Allowance Level": 3, "income": 5000.0}])
    with pytest.raises(KeyError, match="final_donation_rate"):
        engine.run_simulation(n_agents=1, seed=1, single_decision="final_donation_rate",
                              agents_df=agents)
