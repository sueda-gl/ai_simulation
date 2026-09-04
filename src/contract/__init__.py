# src/contract/__init__.py
"""
The run contract between the Streamlit app and the engine.

``src.contract.plan`` defines the frozen dataclasses a UI (or a test) hands to
``app.seam.execute.execute`` and the helpers that both sides must agree on:
the deep-merge semantics of decision-config patches and the sha256 used to
prove a saved configuration was reproduced.  ``src.contract.defaults`` holds
the 13 decisions' default values and default descriptions, the pure data the
seam and the screens both read.  Nothing here imports Streamlit or the app
package.
"""

from src.contract.defaults import (
    DEFAULT_DECISION_DESCRIPTIONS,
    DEFAULT_DECISION_VALUES,
)
from src.contract.plan import (
    INCOME_MODES,
    MESSAGE_KINDS,
    POPULATIONS,
    Replace,
    RunMetadata,
    RunPlan,
    SavedExpectation,
    SubRun,
    UiMessage,
    apply_patches,
    deep_merge_patch,
    hash_result_columns,
)

__all__ = [
    "DEFAULT_DECISION_DESCRIPTIONS",
    "DEFAULT_DECISION_VALUES",
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
