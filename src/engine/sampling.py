# src/engine/sampling.py
"""
Participant selection for the Research modes (Research Specification and Research
Baseline).

Owner ruling R-CYC (2026-10-07): the research population modes never bootstrap,
resample or permute. The agents are the 280 original participants in their real
(file) order, cycling:

    agent k (0-based) = participant k mod 280

* N = 280  -> exactly the 280, in file order;
* N = 1000 -> three full cycles of the 280 and then the first 160;
* N < 280  -> the first N participants.

The same rule serves both research modes and both entry points (the app's
:func:`load_original_participants` and the engine's CLI fallback
:func:`sample_participants_internal`), and it draws no random numbers: the research
population no longer depends on the seed. Each repeated agent is still its own agent
for the RNG (its base seed comes from ``rng_pass1`` by agent index), so Research
Specification noise differs between the copies of one participant.

Before R-CYC the app bootstrapped (with replacement) for Research Specification when
N > 280 and drew a random subset when N < 280, and the CLI permuted Research
Specification even at N = 280 and bootstrapped Research Baseline above 280 (rulings
Q-30 / Q-65 in docs/migration/rulings-and-quirks.md).
"""

import numpy as np
import pandas as pd

from src.data.participants import original_data


def cyclic_indices(n_agents: int, n_original: int) -> np.ndarray:
    """Row positions [0, 1, ..., n_original-1, 0, 1, ...] of length n_agents (k mod n_original)."""
    if n_agents < 0:
        raise ValueError(f"n_agents must be >= 0, got {n_agents}")
    if n_original <= 0:
        raise ValueError("no participants to select from")
    return np.arange(n_agents) % n_original


def cyclic_participants(original_df: pd.DataFrame, n_agents: int) -> pd.DataFrame:
    """Agent k = row k mod len(original_df), file order, index reset to range(n_agents)."""
    indices = cyclic_indices(n_agents, len(original_df))
    df = original_df.iloc[indices].copy()
    df.index = range(len(df))
    return df


def load_original_participants(n_agents: int, seed: int = None, random_sample: bool = None) -> pd.DataFrame:
    """
    The Research-mode agents: the 280 original participants in file order, cycling
    (agent k = participant k mod 280), for BOTH research modes (ruling R-CYC).

    ``seed`` and ``random_sample`` are accepted for call compatibility and ignored:
    the selection draws no random numbers and is the same in both research modes.
    """
    return cyclic_participants(original_data(), n_agents)


def sample_participants_internal(original_df: pd.DataFrame, n_agents: int,
                                 rng: np.random.Generator = None,
                                 random_sample: bool = None) -> pd.DataFrame:
    """
    Engine-internal participant selection (used when the caller supplies no agents_df,
    i.e. CLI runs): the same cyclic rule as :func:`load_original_participants`
    (ruling R-CYC). ``rng`` (the setup RNG) is no longer consumed; ``random_sample``
    is ignored.
    """
    return cyclic_participants(original_df, n_agents)


# The size of the research population (len(original_data())), for on-screen text only;
# the selection itself always uses the loaded frame's length.
N_ORIGINAL_PARTICIPANTS = 280


def research_population_text(n_agents: int, n_original: int = N_ORIGINAL_PARTICIPANTS) -> str:
    """How a research run of n_agents uses the participants (the run-time info line)."""
    if n_agents == n_original:
        return f"📊 Using original {n_original} participants"
    if n_agents < n_original:
        return (f"📊 Using the first {n_agents} of the original {n_original} participants, "
                "in their original order")
    return (f"📊 Using original {n_original} participants: the {n_agents} agents cycle through "
            f"them in their original order (agent k = participant k mod {n_original})")
