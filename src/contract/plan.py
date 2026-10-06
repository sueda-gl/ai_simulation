# src/contract/plan.py
"""
Frozen description of ONE click on a run button.

The Streamlit adapter (``app/simulation.py``) takes a snapshot of the session
state, the pure builder (``app/seam/build_plan.py``) turns it into a
:class:`RunPlan`, and ``app/seam/execute.py`` runs the plan against the engine.
Nothing in this module imports Streamlit or the app package.

A :class:`RunPlan` carries

* ``sub_runs``           - one :class:`SubRun` per result key, in execution order;
* ``messages``           - the st.info/caption/success texts the adapter shows,
                           in the order the legacy ``run_full_simulation`` showed them;
* ``metadata``           - what was actually run (for the results page);
* ``saved_expectations`` - the sha256 of each pinned decision's output columns
                           (ruling R14): the complete run recomputes and compares.

Decision-config patches
-----------------------
``SubRun.decision_config_patches`` maps a decision name to a patch that is
deep-merged onto that decision's block of the freshly loaded
``config/decisions.yaml`` dict (:func:`apply_patches`).  Plain nested dicts
merge key by key; a value wrapped in :class:`Replace` replaces the whole
subtree instead (ruling R10 needs this to drop the nested
categorical/continuous coefficient blocks).
"""

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd

POPULATIONS: Tuple[str, ...] = ("copula", "documentation", "baseline")
INCOME_MODES: Tuple[str, ...] = ("categorical", "continuous")
MESSAGE_KINDS: Tuple[str, ...] = ("info", "caption", "success", "warning", "error")


# ------------------------------------------------------------------ patches

class Replace:
    """Marker: replace the target subtree with ``value`` instead of merging into it."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Replace) and other.value == self.value

    def __hash__(self) -> int:  # pragma: no cover - only for completeness
        return hash(("Replace", repr(self.value)))

    def __repr__(self) -> str:
        return f"Replace({self.value!r})"


def deep_merge_patch(target: dict, patch: Mapping) -> dict:
    """
    Deep-merge ``patch`` into ``target`` IN PLACE and return ``target``.

    * a ``Replace`` value overwrites the key with a deep copy of its payload;
    * a plain dict recurses (creating the target dict when missing or when the
      existing value is not a dict);
    * anything else is assigned as-is (scalars, lists, tuples).
    """
    for key, value in patch.items():
        if isinstance(value, Replace):
            target[key] = copy.deepcopy(value.value)
        elif isinstance(value, Mapping):
            existing = target.get(key)
            if not isinstance(existing, dict):
                existing = {}
                target[key] = existing
            deep_merge_patch(existing, value)
        else:
            target[key] = value
    return target


def apply_patches(config: dict, patches: Mapping[str, Mapping]) -> dict:
    """
    Apply ``{decision_name: patch}`` onto a decisions.yaml dict IN PLACE.

    The caller passes a fresh copy of the loaded file; a missing decision block
    is created so a patch can never be silently dropped.
    """
    for decision_name, patch in patches.items():
        block = config.get(decision_name)
        if not isinstance(block, dict):
            block = {}
            config[decision_name] = block
        deep_merge_patch(block, patch)
    return config


# ------------------------------------------------------------- expectations

def hash_result_columns(result_df: Optional[pd.DataFrame], columns: Iterable[str]) -> Optional[str]:
    """
    Deterministic sha256 over the given columns of a result DataFrame (ruling R14).

    Identical to ``app.state.saved_configs.hash_result_columns`` (the writer of
    ``result_sha256``): the digest covers the column names (in the given order)
    and ``pd.util.hash_pandas_object`` of those columns, falling back to their
    string form when a cell is unhashable (lists / dicts).  Returns ``None``
    when there is nothing to hash so a caller can tell "no expectation
    recorded" apart from "hashes differ".
    """
    columns = list(columns)
    if result_df is None or not columns:
        return None

    present = [c for c in columns if c in result_df.columns]
    if not present:
        return None

    subset = result_df[present]
    try:
        row_hashes = pd.util.hash_pandas_object(subset, index=False)
    except TypeError:
        row_hashes = pd.util.hash_pandas_object(subset.astype(str), index=False)

    digest = hashlib.sha256()
    digest.update("\x1f".join(present).encode("utf-8"))
    digest.update(row_hashes.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------- dataclasses

@dataclass(frozen=True)
class UiMessage:
    """One st.<kind>(text) call the adapter makes before the run starts."""
    kind: str
    text: str

    def __post_init__(self) -> None:
        if self.kind not in MESSAGE_KINDS:
            raise ValueError(f"UiMessage.kind must be one of {MESSAGE_KINDS}, got {self.kind!r}")


@dataclass(frozen=True)
class SubRun:
    """
    One engine run, producing ``results[result_key]``.

    decision_config_patches : {decision: patch} deep-merged onto decisions.yaml
    simulation_params       : the ['simulation'] mapping (Page 1 parameters)
    decision_settings       : collect_decision_settings() output; becomes BOTH
                              simulation_config['random_decisions'] and
                              ['default_decisions'] (the same object, as today)
    default_decisions_list  : simulation_config['default_decisions_list']
                              (the force-DI-default rule for a standalone
                              Disclose Documents run lives here, not in a flag)
    purchasing_limits       : simulation_config['purchasing_limits'] or None
                              (None = leave the yaml value)
    decisions_to_run        : the single_decision list, or None for all 13
    """
    result_key: str
    population: str
    income_mode: str
    seed: int
    n_agents: int
    decisions_to_run: Optional[Tuple[str, ...]]
    decision_config_patches: Dict[str, dict]
    simulation_params: dict
    decision_settings: dict
    default_decisions_list: Tuple[str, ...]
    purchasing_limits: Optional[dict]

    def __post_init__(self) -> None:
        if self.population not in POPULATIONS:
            raise ValueError(f"SubRun.population must be one of {POPULATIONS}, got {self.population!r}")
        if self.income_mode not in INCOME_MODES:
            raise ValueError(f"SubRun.income_mode must be one of {INCOME_MODES}, got {self.income_mode!r}")


@dataclass(frozen=True)
class RunMetadata:
    """What was actually run - stored as st.session_state._run_metadata (ruling R28)."""
    effective_population_mode: str
    effective_income_mode: str
    result_keys: Tuple[str, ...]
    custom_decisions: Tuple[str, ...]
    default_decisions: Tuple[str, ...]
    seed: int
    n_agents: int
    is_comparison: bool
    # The income mode each modelled decision ACTUALLY ran with, per result key:
    # {result_key: {decision: 'categorical' | 'continuous'}}. In a complete run the
    # result keys follow the global income mode, but Decision 4 runs with its own
    # tab mode, so the page must not infer a decision's mode from the key.
    decision_income_modes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Complete run while the Decision 4 tab said "Compare both": Decision 4 ran
    # continuous only (a complete run cannot split it); the page says so.
    rtd_compare_both_fallback: bool = False


@dataclass(frozen=True)
class SavedExpectation:
    """A pinned decision's recorded output-column hash, checked after a complete run (R14)."""
    decision: str
    result_key: str
    columns: Tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class RunPlan:
    sub_runs: Tuple[SubRun, ...]
    messages: Tuple[UiMessage, ...]
    metadata: RunMetadata
    saved_expectations: Tuple[SavedExpectation, ...]

    @property
    def result_keys(self) -> Tuple[str, ...]:
        return tuple(sub_run.result_key for sub_run in self.sub_runs)


__all__ = [
    "INCOME_MODES",
    "MESSAGE_KINDS",
    "POPULATIONS",
    "Replace",
    "RunMetadata",
    "RunPlan",
    "SavedExpectation",
    "SubRun",
    "UiMessage",
    "apply_patches",
    "deep_merge_patch",
    "hash_result_columns",
]
