# app/seam/snapshot.py
"""
A frozen copy of the session state, taken ONCE per click (ruling R15).

The plan builder reads only from a :class:`SessionSnapshot`; it never sees
``st.session_state`` itself, so a run cannot observe a value that another part
of the run wrote, and a test can hand the builder a plain dict.
"""

from collections.abc import Mapping
from typing import Any, Iterator


class SessionSnapshot(Mapping):
    """Read-only mapping view over a shallow copy of the session state."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping):
        object.__setattr__(self, "_data", dict(data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("SessionSnapshot is read-only")

    def __repr__(self) -> str:
        return f"SessionSnapshot({len(self._data)} keys)"


def take_snapshot(state: Any) -> SessionSnapshot:
    """
    Copy ``state`` (``st.session_state`` or any mapping) into a SessionSnapshot.

    Streamlit's proxy exposes ``to_dict()`` (every user key plus keyed widget
    values); any other mapping is copied key by key.  Values are NOT deep
    copied - the builder treats them as read-only and copies what it stores.
    """
    if isinstance(state, SessionSnapshot):
        return state
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        return SessionSnapshot(to_dict())
    return SessionSnapshot(state)


__all__ = ["SessionSnapshot", "take_snapshot"]
