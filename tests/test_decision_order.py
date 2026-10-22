"""
The decision order is defined once, in the engine (src/engine/core.py
DECISION_ORDER): app.models.ALL_DECISIONS (the screens) and
app.seam.build_plan.ALL_DECISIONS (the seam) are copies of it, and the
contract's default registries cover exactly those decisions.
"""
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
