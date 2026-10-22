# src/decisions/__init__.py
"""
One module per decision; one function per module, with the same shape.

The engine (src/engine/core.py) imports ``src.decisions.<name>`` and calls the function
of the same name (Decisions 1 and 2 live in ``<name>_stochastic``) once per agent and
decision as::

    decision(agent_state, params, decision_rng, simulation_config=..., pop_context=...)

``agent_state`` is the agent's row so far (traits, income, earlier decisions' outputs);
``params`` is the decision's block of ``config/decisions.yaml`` after the run's patches;
``decision_rng`` is a ``numpy.random.Generator`` seeded for this agent and this decision
alone; ``simulation_config`` holds the Page-1 parameters under ``'simulation'`` and the
population statistics computed in Pass 1. ``pop_context`` (the population mode's name)
is passed to every decision; a decision that does not use it takes it in its
``**kwargs``, which every decision accepts.

The function returns a dict of new columns, ``{"<name>": value, ...}``, which the
engine merges into the agent state (the mutated state is the output row).
"""

from typing import Any, Dict, Protocol

import numpy as np


class Decision(Protocol):
    """The decision-function contract described in the module docstring."""

    def __call__(self, agent_state: Dict[str, Any], params: Dict[str, Any],
                 rng: np.random.Generator, simulation_config: Dict[str, Any] = None,
                 **kwargs: Any) -> Dict[str, Any]: ...
