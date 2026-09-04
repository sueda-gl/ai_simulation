# src/contract/__init__.py
"""
The run contract between the Streamlit app and the engine.

``src.contract.plan`` defines the frozen dataclasses a UI (or a test) hands to
``app.seam.execute.execute`` and the helpers that both sides must agree on:
the deep-merge semantics of decision-config patches and the sha256 used to
prove a saved configuration was reproduced.  Nothing here imports Streamlit
or the app package.
"""

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
