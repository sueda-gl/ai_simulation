# src/engine/sampling.py
"""
Participant selection for the Research modes (Research Specification and Research
Baseline).

Owner ruling R-CYC (2026-10-07), as clarified by the owner the same day:

* RESEARCH BASELINE takes the 280 original participants in their real (file) order,
  cycling - no random numbers:

      agent k (0-based) = participant k mod 280

  N = 280 -> exactly the 280 in file order; N = 1000 -> three full cycles and then
  the first 160; N < 280 -> the first N.

* RESEARCH SPECIFICATION samples RANDOMLY, seeded by the run seed (reproducible;
  ``default_rng(seed)``, a stream of its own - the user manual: "If Number of Agents
  <= 280 the simulation randomly selects a unique subset of the real participants;
  if > 280 it uses bootstrap sampling with replacement"):

      N < 280 -> a random subset of N distinct participants (without replacement);
      N = 280 -> all 280 (the unique subset of size 280), kept in file order;
      N > 280 -> a bootstrap sample of N draws WITH replacement from the 280.

  N = 280 keeps the file order (as before e0d3d97): the set is the same either way,
  and file order keeps the per-agent RNG streams on the same participants as Research
  Baseline, so a Research Specification run with every sigma off equals the Baseline
  run row for row and the Stata row-by-row comparisons at N = 280 hold.

The same rule serves both entry points: the app's :func:`load_original_participants`
(``app.seam.execute.sample_agents``) and the engine's CLI fallback
:func:`sample_participants_internal` (``Engine._resolve_agents``), so a CLI / Monte-Carlo
repetition with seed S selects exactly the agents of an app run with seed S. Each agent
keeps its own RNG streams (base seed from ``rng_pass1`` by agent index), so repeated
participants (Specification bootstrap, Baseline cycles) get different noise.

History: before e0d3d97 the app did the Specification rule above, while the CLI
fallback drew from the setup RNG (after the vendor draws) and permuted Specification at
N = 280, and bootstrapped Baseline above 280; e0d3d97 (R-CYC) made BOTH modes cyclic;
the owner's clarification restored random sampling for Research Specification only
(rulings Q-30 / Q-65 in docs/migration/rulings-and-quirks.md).
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


def random_indices(n_agents: int, n_original: int, seed: int) -> np.ndarray:
    """Research Specification row positions, drawn from ``default_rng(seed)``:
    a random subset without replacement for N < n_original, all rows in file order for
    N = n_original, a bootstrap (with replacement) for N > n_original."""
    if n_agents < 0:
        raise ValueError(f"n_agents must be >= 0, got {n_agents}")
    if n_original <= 0:
        raise ValueError("no participants to select from")
    if seed is None:
        raise ValueError("Research Specification sampling needs the run seed")
    if n_agents == n_original:
        return np.arange(n_original)
    rng = np.random.default_rng(int(seed))
    return rng.choice(n_original, size=n_agents, replace=n_agents > n_original)


def research_indices(n_agents: int, n_original: int, random_sample: bool,
                     seed: int = None) -> np.ndarray:
    """Row positions of the research agents: random (Specification) or cyclic (Baseline)."""
    if random_sample:
        return random_indices(n_agents, n_original, seed)
    return cyclic_indices(n_agents, n_original)


def _select(original_df: pd.DataFrame, indices) -> pd.DataFrame:
    df = original_df.iloc[np.asarray(indices, dtype=int)].copy()
    df.index = range(len(df))
    return df


def cyclic_participants(original_df: pd.DataFrame, n_agents: int) -> pd.DataFrame:
    """Agent k = row k mod len(original_df), file order, index reset to range(n_agents)."""
    return _select(original_df, cyclic_indices(n_agents, len(original_df)))


def research_participants(original_df: pd.DataFrame, n_agents: int, random_sample: bool,
                          seed: int = None) -> pd.DataFrame:
    """The research agents drawn from ``original_df``, index reset to range(n_agents)."""
    return _select(original_df,
                   research_indices(n_agents, len(original_df), random_sample, seed))


def load_original_participants(n_agents: int, seed: int = None,
                               random_sample: bool = False) -> pd.DataFrame:
    """
    The Research-mode agents from the 280 original participants:
    ``random_sample=True`` (Research Specification) - random, seeded by ``seed``;
    ``random_sample=False`` (Research Baseline) - file order, cycling.
    """
    return research_participants(original_data(), n_agents, random_sample, seed)


def sample_participants_internal(original_df: pd.DataFrame, n_agents: int,
                                 seed: int = None, random_sample: bool = False) -> pd.DataFrame:
    """
    Engine-internal participant selection (used when the caller supplies no agents_df,
    i.e. CLI runs): the same rule, and the same ``default_rng(seed)`` stream, as
    :func:`load_original_participants`. The engine's setup RNG is not consumed.
    """
    return research_participants(original_df, n_agents, random_sample, seed)


# The size of the research population (len(original_data())), for on-screen text only;
# the selection itself always uses the loaded frame's length.
N_ORIGINAL_PARTICIPANTS = 280


def research_population_text(n_agents: int, random_sample: bool = False,
                             n_original: int = N_ORIGINAL_PARTICIPANTS) -> str:
    """How a research run of n_agents uses the participants (the run-time info line)."""
    if n_agents == n_original:
        return f"📊 Using original {n_original} participants"
    if random_sample:
        if n_agents < n_original:
            return (f"📊 Using a random subset of {n_agents} of the original {n_original} "
                    "participants (without replacement, seeded by the run seed)")
        return (f"📊 Using original {n_original} participants: the {n_agents} agents are a "
                "bootstrap sample with replacement (seeded by the run seed)")
    if n_agents < n_original:
        return (f"📊 Using the first {n_agents} of the original {n_original} participants, "
                "in their original order")
    return (f"📊 Using original {n_original} participants: the {n_agents} agents cycle through "
            f"them in their original order (agent k = participant k mod {n_original})")
