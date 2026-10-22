"""A row of a Decision 4 reference extract as a decision function's agent state.

The extracts (``data/stata_d4_verification.csv``,
``data/stata_d4_flexibility_verification.csv``) name their columns as the reference
Stata file does (``extraversionbig5``, ``assignedallowancelevel``, ...); the decision
functions take the agent-state keys (``ExtraversionBig5``, ``Assigned Allowance Level``, ...).
"""


def agent_state(row, with_stdactions=True):
    """The agent-state dict of one reference participant. ``stdactions`` is the
    Flexibility observed anchor; ``with_stdactions=False`` leaves it out
    (tests/test_rtd_flexibility.py checks the fallback to the calculated score)."""
    s = {
        "ExtraversionBig5": row["extraversionbig5"], "Agreeable": row["agreeable"],
        "NeuroticismBig5": row["neuroticismbig5"], "ConscientiousnessBig5": row["conscientiousnessbig5"],
        "OpennessBig5": row["opennessbig5"], "Education": row["education"],
        "Assigned Allowance Level": row["assignedallowancelevel"], "income": row["income"],
    }
    if with_stdactions:
        s["stdactions"] = row["stdactions"]
    return s
