"""
Three small units that had no test of their own:

* ``src.utils.stochastic.should_use_stochastic`` - the one rule every decision uses to
  decide whether its noise is drawn;
* ``src.engine.postprocess.assign_global_transaction_ids`` - the transaction ids every
  export shares;
* ``app.seam.build_plan.build_simulation_params`` - Page 1's parameters on their way
  into the engine's ``simulation`` mapping.
"""
import dataclasses

import pandas as pd
import pytest

from app.seam.build_plan import build_simulation_params
from src.engine.postprocess import assign_global_transaction_ids
from src.utils.stochastic import should_use_stochastic


@pytest.mark.parametrize("pop_context", ["copula", "documentation", "baseline"])
@pytest.mark.parametrize("in_copula", [False, True])
@pytest.mark.parametrize("sigma_value", [0, 0.5])
def test_should_use_stochastic_three_way_rule(pop_context, in_copula, sigma_value):
    """Copula follows its tick box; Research Specification draws when sigma > 0 (its
    tick box); Research Baseline never draws."""
    draws = {("copula", True, 0), ("copula", True, 0.5),
             ("documentation", False, 0.5), ("documentation", True, 0.5)}
    params = {"in_copula": in_copula, "sigma_value": sigma_value}
    assert should_use_stochastic(params, pop_context) is ((pop_context, in_copula, sigma_value) in draws)


def test_assign_global_transaction_ids_are_global_ordered_and_stable():
    df = pd.DataFrame({"agent_id": [1, 2, 3], "purchase_requests": [
        [{"timestamp_hours": 0.5}, {"timestamp_hours": 2.0}],
        [{"timestamp_hours": 1.0}, {"timestamp_hours": 0.5}],   # ties with agent 1's first
        [],
    ]})

    def ids():
        return [[r["transaction_id"] for r in requests] for requests in df["purchase_requests"]]

    assert assign_global_transaction_ids(df) is df             # written in place
    # one sequence across all agents, by timestamp; a tie keeps agent / request order
    assert ids() == [[1, 4], [3, 2], []]
    assign_global_transaction_ids(df)
    assert ids() == [[1, 4], [3, 2], []]                       # re-running changes nothing
    no_requests = pd.DataFrame({"agent_id": [1]})
    assert assign_global_transaction_ids(no_requests) is no_requests


# SimulationParameters fields that do NOT go into the simulation mapping (pinned, so a
# new field has to be either forwarded or added here on purpose).
NOT_FORWARDED = {
    "simulation_execution_mode", "simulation_mode", "income_min", "income_max", "income_avg",
    "income_avg_type", "price_grid", "apply_purchasing_limits", "purchasing_limits",
    "purchasing_limits_source",
}


def test_build_simulation_params_forwards_the_page1_fields():
    from app.models import SimulationParameters
    fields = [f.name for f in dataclasses.fields(SimulationParameters)]
    values = {name: object() for name in fields}                # distinct, never None
    simulation = build_simulation_params({"sim_params": SimulationParameters(**values)})
    assert set(simulation) == set(fields) - NOT_FORWARDED       # nothing extra, nothing lost
    for name in simulation:
        assert simulation[name] is values[name], name           # the field's own value
