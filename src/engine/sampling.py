# src/engine/sampling.py
"""
Participant sampling for the Research modes.

Two mechanisms, both moved unchanged so that agent order — and therefore every
per-agent RNG stream derived from it — is bit-identical to today:

* :func:`load_original_participants` — the app-level selection, moved verbatim
  from app/simulation.py:74-129 (``_load_original_participants``); the
  throw-away ``OrchestratorBaseline()`` that existed only to reach
  ``original_data`` is replaced by :func:`src.data.participants.original_data`.
* :func:`sample_participants_internal` — the orchestrator-internal fallback used
  when no ``agents_df`` is supplied (CLI runs), reproducing
  src/orchestrator_doc_mode.py:106-115 (``random_sample=True``) and
  src/orchestrator_baseline.py:131-139 (``random_sample=False``) exactly,
  including how many draws each branch takes from ``rng``.
"""

import numpy as np
import pandas as pd

from src.data.participants import original_data


def load_original_participants(n_agents: int, seed: int, random_sample: bool = True) -> pd.DataFrame:
    """
    Load original 280 participants with configurable sampling.
    
    Args:
        n_agents: Number of agents to load
        seed: Random seed for reproducibility
        random_sample: 
            True (Research Spec) → Random sampling for n<280, sequential for n==280
            False (Research Baseline) → Sequential selection for n≤280
    
    Behavior:
        n_agents == n_original: Both modes → Sequential [0, 1, ..., n-1]
            (All participants included; shuffling would cause different income
            generation per agent, breaking mode equivalence in continuous mode)
        n_agents > 280:  Both modes → Bootstrap (random WITH replacement)
        n_agents < 280:
            random_sample=True  → Random WITHOUT replacement (Research Spec)
            random_sample=False → Sequential [0, 1, 2, ..., n-1] (Research Baseline)
    """
    original = original_data()
    n_original = len(original)
    rng = np.random.default_rng(seed)
    
    if n_agents <= n_original:
        if n_agents == n_original:
            # All participants included — use sequential order for BOTH modes.
            # Random permutation serves no purpose when every agent is included
            # and would cause different position-dependent income RNG seeds,
            # breaking Research Spec / Baseline equivalence in continuous mode.
            indices = list(range(n_original))
        elif random_sample:
            # Research Spec with subset: Random sample WITHOUT replacement
            indices = rng.choice(n_original, size=n_agents, replace=False)
        else:
            # Research Baseline with subset: Sequential/in-order selection
            indices = list(range(n_agents))
        df = original.iloc[indices].copy()
        df.index = range(len(df))
        return df
    else:
        # n_agents > 280
        if random_sample:
            # Research Spec: Bootstrap sample WITH replacement
            indices = rng.choice(n_original, size=n_agents, replace=True)
        else:
            # Research Baseline: "Always in order" -> Deterministic wrapping
            # [0, 1, ..., 279, 0, 1, ..., 279, 0, 1, ...]
            full_repeats = n_agents // n_original
            remainder = n_agents % n_original
            indices = list(range(n_original)) * full_repeats + list(range(remainder))

        df = original.iloc[indices].copy()
        df.index = range(len(df))
        return df


def sample_participants_internal(original_df: pd.DataFrame, n_agents: int,
                                 rng: np.random.Generator,
                                 random_sample: bool) -> pd.DataFrame:
    """
    Orchestrator-internal participant sampling (used when the caller supplies no
    agents_df, i.e. CLI runs).

    random_sample=True  → Research Specification rules (orchestrator_doc_mode.py:106-115):
        one rng.choice draw either way, replace=True iff n_agents > len(original_df),
        then reset_index(drop=True).
    random_sample=False → Research Baseline rules (orchestrator_baseline.py:131-139):
        the first n_agents rows when n_agents <= len(original_df) (NO rng draw),
        otherwise a bootstrap rng.choice(replace=True) with the index reset to range.

    ``rng`` is the setup RNG (``rng_setup``) and is consumed exactly as it is today.
    """
    if random_sample:
        # src/orchestrator_doc_mode.py:106-115
        n_original = len(original_df)

        if n_agents > n_original:
            # Bootstrap with replacement on participants then repeat draws
            indices = rng.choice(n_original, size=n_agents, replace=True)
            agents_df = original_df.iloc[indices].reset_index(drop=True)
        else:
            indices = rng.choice(n_original, size=n_agents, replace=False)
            agents_df = original_df.iloc[indices].reset_index(drop=True)
        return agents_df

    # src/orchestrator_baseline.py:131-139
    if n_agents <= len(original_df):
        # Use first n_agents participants
        agents_df = original_df.iloc[:n_agents].copy()
    else:
        # Bootstrap sample to reach n_agents using setup RNG
        indices = rng.choice(len(original_df), size=n_agents, replace=True)
        agents_df = original_df.iloc[indices].copy()
        agents_df.index = range(len(agents_df))  # Reset index
    return agents_df
