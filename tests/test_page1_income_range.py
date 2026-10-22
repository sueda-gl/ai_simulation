"""Page 1: a "b - Maximum Value ($)" at or below "a - Minimum Value ($)" is reported,
not a crash.

Reproduced in a browser (2026-10-22): lognormal minimum 20000 with maximum 10000
raised StreamlitValueAboveMaxError in ``_render_discount_threshold`` - the threshold
was clamped up to the minimum and the number_input drawn with max_value below its
min_value. The engine itself only clamps incomes to the maximum (src/engine/income.py),
so the run would not have failed; only the screen did. Now the page shows one
st.error and draws the threshold widget as if no maximum were set, for every
distribution the selectbox offers.
"""
import pytest

from conftest import start_app
from app.pages.page1_common_params import INCOME_MAX_NOT_ABOVE_MIN

pytestmark = pytest.mark.slow

DISTRIBUTIONS = [
    # distribution, minimum widget, maximum widget, sim_params attribute of the maximum
    ("lognormal", "lognormal_min_input", "lognormal_max_text_input", "lognormal_max"),
    ("generalised_gamma", "gg_min_input", "gg_max_text_input", "gg_max"),
    ("dagum", "dagum_min_input", "dagum_max_text_input", "dagum_max"),
]


def _errors(at):
    return [str(e.value) for e in at.error]


@pytest.mark.parametrize("distribution, min_key, max_key, max_attr", DISTRIBUTIONS)
@pytest.mark.parametrize("maximum", ["10000", "20000"])   # below the minimum, and equal to it
def test_max_not_above_min_shows_error_and_keeps_the_page_alive(distribution, min_key, max_key,
                                                                maximum, max_attr):
    at = start_app()
    at.run()
    at.selectbox(key="page1_income_distribution").set_value(distribution).run()
    at.number_input(key=min_key).set_value(20000.0).run()
    at.text_input(key=max_key).set_value(maximum).run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert getattr(at.session_state["sim_params"], max_attr) == float(maximum), \
        "the typed maximum still travels to the run (the engine clamps to it)"
    assert INCOME_MAX_NOT_ABOVE_MIN in _errors(at)

    # the threshold widget is on screen, bounded as if no maximum were set
    threshold = at.number_input(key="discount_threshold_input")
    assert threshold.value >= 20000.0

    # a maximum above the minimum clears the error
    at.text_input(key=max_key).set_value("30000").run()
    assert not at.exception
    assert INCOME_MAX_NOT_ABOVE_MIN not in _errors(at)
    assert at.number_input(key="discount_threshold_input").value <= 30000.0


def test_default_page_shows_no_range_error():
    at = start_app()
    at.run()
    assert not at.exception
    assert INCOME_MAX_NOT_ABOVE_MIN not in _errors(at)
